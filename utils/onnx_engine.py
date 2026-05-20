"""
ONNX 推理加速引擎
提供 PyTorch -> ONNX 模型导出 + ONNX Runtime 滑动窗口推理
目标: 大图推理速度提升 3-5x (相比原生 PyTorch)

依赖: onnxruntime (必需), onnx + torch (仅导出时需要)
"""

import os
import time
import warnings
import tempfile
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================
# 常量配置
# ============================================

# ONNX Runtime 可用的执行提供者 (按优先级)
ONNX_PROVIDERS = [
    "CUDAExecutionProvider",      # NVIDIA GPU
    "TensorrtExecutionProvider",  # TensorRT 加速
    "ROCMExecutionProvider",      # AMD GPU
    "CPUExecutionProvider",       # CPU 回退
]

# Sentinel-2 常见波段数
S2_BAND_COUNTS = {4, 6, 10, 13}

# 默认滑动窗口参数
DEFAULT_WINDOW_SIZE = 512
DEFAULT_OVERLAP = 256
DEFAULT_BATCH_SIZE = 4


# ============================================
# 工具函数: 检查 ONNX Runtime 可用性
# ============================================

def check_onnx_available() -> Dict[str, Any]:
    """
    检查 ONNX Runtime 及其可用 Provider

    返回:
        dict: {
            "onnx_available": bool,
            "providers": list,
            "version": str,
            "cuda_available": bool,
            "recommended_provider": str,
        }
    """
    result = {
        "onnx_available": False,
        "providers": [],
        "version": "N/A",
        "cuda_available": False,
        "recommended_provider": "CPUExecutionProvider",
    }

    try:
        import onnxruntime as ort
        result["onnx_available"] = True
        result["version"] = ort.__version__
        result["providers"] = ort.get_available_providers()

        # 检查 CUDA
        if "CUDAExecutionProvider" in result["providers"]:
            result["cuda_available"] = True
            result["recommended_provider"] = "CUDAExecutionProvider"
        elif "TensorrtExecutionProvider" in result["providers"]:
            result["recommended_provider"] = "TensorrtExecutionProvider"
        elif "ROCMExecutionProvider" in result["providers"]:
            result["recommended_provider"] = "ROCMExecutionProvider"

    except ImportError:
        pass

    return result


def get_ort_session(
    model_path: str,
    provider: Optional[str] = None,
    session_options: Optional[Dict] = None,
):
    """
    创建 ONNX Runtime 推理会话

    参数:
        model_path: ONNX 模型路径 (.onnx)
        provider: 执行提供者 (None=自动选择最佳)
        session_options: 会话选项 dict

    返回:
        onnxruntime.InferenceSession 或 None
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX Runtime 未安装。请运行: pip install onnxruntime "
            "(CPU) 或 pip install onnxruntime-gpu (GPU)"
        )

    # 选择 providers
    if provider:
        providers = [provider]
    else:
        providers = ort.get_available_providers()

    # 构建会话选项
    opts = ort.SessionOptions()
    if session_options:
        for k, v in session_options.items():
            if hasattr(opts, k):
                setattr(opts, k, v)

    # 优化选项 (减少内存使用)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        return session
    except Exception as e:
        # 如果指定 provider 失败, 回退到 CPU
        if "CPUExecutionProvider" not in providers:
            warnings.warn(f"{provider} 不可用, 回退到 CPU: {e}")
            return get_ort_session(model_path, provider="CPUExecutionProvider")
        raise


# ============================================
# 模型导出: PyTorch -> ONNX
# ============================================

def export_to_onnx(
    model: Any,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 6, 512, 512),
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 13,
    simplify: bool = True,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    将 PyTorch 语义分割模型导出为 ONNX 格式

    参数:
        model: PyTorch nn.Module
        output_path: 输出 ONNX 文件路径
        input_shape: (batch, channels, height, width)
        input_names: 输入节点名称
        output_names: 输出节点名称
        dynamic_axes: 动态轴配置 (支持可变 batch/空间尺寸)
        opset_version: ONNX opset 版本
        simplify: 是否使用 onnx-simplifier 简化模型
        verify: 是否验证导出模型的输出一致性

    返回:
        dict: {
            "success": bool,
            "output_path": str,
            "model_size_mb": float,
            "error": str or None,
            "verified": bool or None,
        }
    """
    import torch

    result = {
        "success": False,
        "output_path": output_path,
        "model_size_mb": 0.0,
        "error": None,
        "verified": None,
    }

    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 设置默认名称
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # 设置默认动态轴 (batch + 空间可变)
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

        # 创建虚拟输入
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)

        # 导出 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

        # 简化模型 (可选)
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                onnx_model = onnx.load(output_path)
                simplified_model, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, output_path)
            except ImportError:
                warnings.warn("onnx-simplifier 未安装, 跳过模型简化 (pip install onnx-simplifier)")

        # 验证 (可选)
        if verify:
            result["verified"] = _verify_onnx_export(model, output_path, dummy_input, input_names, output_names)

        # 文件大小
        result["model_size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def _verify_onnx_export(model, onnx_path, dummy_input, input_names, output_names) -> bool:
    """验证 ONNX 导出结果与 PyTorch 输出的一致性"""
    import torch
    import onnx
    import onnxruntime as ort

    try:
        # PyTorch 输出
        model.eval()
        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        # ONNX 输出
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {input_names[0]: dummy_input.cpu().numpy()}
        ort_out = ort_session.run(output_names, ort_inputs)[0]

        # 比较
        max_diff = np.max(np.abs(torch_out - ort_out))
        return max_diff < 1e-4
    except Exception:
        return False


# ============================================
# ONNX 滑动窗口推理 (核心)
# ============================================

def run_onnx_inference(
    onnx_model_path: str,
    input_array: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_classes: int = 2,
    activation: str = "softmax",
    progress_callback: Optional[Callable] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 ONNX Runtime 对大幅影像进行滑动窗口语义分割推理

    支持任意尺寸输入, 自动分块 + overlap 拼接输出

    参数:
        onnx_model_path: ONNX 模型文件路径
        input_array: 输入数组, shape (C, H, W) 或 (H, W, C)
        window_size: 滑动窗口大小 (像素)
        overlap: 窗口重叠大小 (像素)
        batch_size: 批处理大小 (目前单张处理, 预留扩展)
        num_classes: 分类数
        activation: 输出激活函数 "softmax" | "sigmoid" | "none"
        progress_callback: 进度回调 callback(completed, total)
        provider: ONNX 执行提供者

    返回:
        dict: {
            "output": np.ndarray (H, W) 分类结果,
            "prob_map": np.ndarray (H, W, C) 概率图,
            "inference_time_s": float,
            "peak_memory_mb": float,
        }
    """
    start_time = time.time()

    # 确保输入为 (C, H, W)
    if input_array.ndim == 3:
        if input_array.shape[0] > input_array.shape[-1]:
            pass  # already (C, H, W)
        elif input_array.shape[-1] < min(input_array.shape[:2]):
            pass  # already (C, H, W)
        else:
            input_array = np.transpose(input_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    n_channels, h, w = input_array.shape

    # 创建 ONNX 会话
    session = get_ort_session(onnx_model_path, provider=provider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 计算步长
    stride = window_size - overlap

    # 输出数组
    output = np.zeros((h, w), dtype=np.int32)
    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)  # 重叠区域取平均

    # 计算窗口位置
    h_starts = list(range(0, max(1, h - overlap), stride))
    w_starts = list(range(0, max(1, w - overlap), stride))

    # 确保覆盖边缘
    if h_starts[-1] + window_size < h:
        h_starts.append(max(0, h - window_size))
    if w_starts[-1] + window_size < w:
        w_starts.append(max(0, w - window_size))

    total_windows = len(h_starts) * len(w_starts)
    completed = 0

    # 滑动窗口循环
    for i, hs in enumerate(h_starts):
        he = min(hs + window_size, h)
        actual_h = he - hs

        for j, ws in enumerate(w_starts):
            we = min(ws + window_size, w)
            actual_w = we - ws

            # 裁剪窗口
            patch = input_array[:, hs:he, ws:we]  # (C, patch_h, patch_w)

            # 填充到 window_size (如果边缘不足)
            if actual_h < window_size or actual_w < window_size:
                pad_h = window_size - actual_h
                pad_w = window_size - actual_w
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            # 添加 batch 维度 -> (1, C, H, W)
            ort_input = patch[np.newaxis, ...].astype(np.float32)

            # ONNX 推理
            ort_outputs = session.run([output_name], {input_name: ort_input})
            pred = ort_outputs[0][0]  # (C, patch_h, patch_w)

            # 裁剪回实际尺寸
            pred = pred[:, :actual_h, :actual_w]

            # 激活函数
            if activation == "softmax":
                pred_probs = _softmax(pred, axis=0)
            elif activation == "sigmoid":
                pred_probs = 1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))
                if num_classes == 2:
                    pred_probs = np.stack([1 - pred_probs[0], pred_probs[0]], axis=0)
            else:
                pred_probs = pred

            # 调整维度 -> (H, W, C)
            pred_probs = np.transpose(pred_probs, (1, 2, 0))  # (H, W, C)

            # 写入概率图 (重叠区域加权平均)
            prob_map[hs:he, ws:we] += pred_probs
            count_map[hs:he, ws:we] += 1.0

            completed += 1
            if progress_callback:
                progress_callback(completed, total_windows)

    # 平均重叠区域
    mask = count_map > 0
    for c in range(num_classes):
        prob_map[:, :, c][mask] /= count_map[mask]

    # 取 argmax 得到分类结果
    output = np.argmax(prob_map, axis=-1)

    inference_time = time.time() - start_time

    return {
        "output": output,
        "prob_map": prob_map,
        "inference_time_s": round(inference_time, 2),
        "num_windows": total_windows,
    }


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """稳定的 softmax 计算"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ============================================
# 简化的 ONNX 水体分割 (替代 geoai.segment_water)
# ============================================

def segment_water_onnx(
    input_path: str,
    onnx_model_path: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    device: str = "cpu",
    output_raster: Optional[str] = None,
    output_vector: Optional[str] = None,
    band_indices: Optional[List[int]] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    使用 ONNX 模型进行水体语义分割 (替代 geoai.segment_water 的独立方案)

    不依赖 geoai-py, 仅需 onnxruntime + numpy + rasterio

    参数:
        input_path: 输入多波段 GeoTIFF
        onnx_model_path: ONNX 水体分割模型路径
        window_size: 滑动窗口大小
        overlap: 窗口重叠大小
        device: "cpu" 或 "cuda"
        output_raster: 输出掩膜 GeoTIFF 路径
        output_vector: 输出 GeoJSON 路径
        band_indices: 要使用的波段索引 (1-based), None=使用全部
        progress_callback: 进度回调

    返回:
        dict: 同 segment_water_ai 返回值格式
    """
    import rasterio

    result = {
        "raster_path": None,
        "vector_path": None,
        "mask_array": None,
        "stats": None,
        "success": False,
        "error": None,
    }

    # 检查 ONNX 模型
    if not os.path.exists(onnx_model_path):
        result["error"] = f"ONNX 模型文件不存在: {onnx_model_path}"
        return result

    # 检查 ONNX Runtime
    ort_info = check_onnx_available()
    if not ort_info["onnx_available"]:
        result["error"] = "ONNX Runtime 未安装。请运行: pip install onnxruntime"
        return result

    # 输出路径
    if output_raster is None:
        tmp_dir = tempfile.gettempdir()
        output_raster = os.path.join(
            tmp_dir, f"onnx_water_mask_{os.path.basename(input_path).replace('.tif', '')}.tif"
        )
    os.makedirs(os.path.dirname(output_raster) or ".", exist_ok=True)

    try:
        # 读取输入影像
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            pixel_size_x = abs(src.transform.a)
            pixel_size_y = abs(src.transform.e)
            pixel_size_m = (pixel_size_x + pixel_size_y) / 2

            if band_indices is None:
                band_indices = list(range(1, src.count + 1))

            # 读取波段
            bands = []
            for b in band_indices:
                if 1 <= b <= src.count:
                    bands.append(src.read(b))
                else:
                    result["error"] = f"波段索引 {b} 超出范围 (1-{src.count})"
                    return result

            input_array = np.stack(bands, axis=0).astype(np.float32)

        # 数据标准化 (简单 min-max)
        for c in range(input_array.shape[0]):
            c_min, c_max = np.percentile(input_array[c], [2, 98])
            if c_max > c_min:
                input_array[c] = np.clip((input_array[c] - c_min) / (c_max - c_min), 0, 1)

        # 选择 ONNX provider
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"

        # ONNX 推理
        inference_result = run_onnx_inference(
            onnx_model_path=onnx_model_path,
            input_array=input_array,
            window_size=window_size,
            overlap=overlap,
            num_classes=2,
            activation="sigmoid",
            progress_callback=progress_callback,
            provider=provider,
        )

        mask_array = inference_result["output"].astype(np.uint8)

        # 保存掩膜 GeoTIFF
        profile.update({
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
        })
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(mask_array, 1)

        result["raster_path"] = output_raster
        result["mask_array"] = mask_array

        # 面积统计
        water_pixels = int(np.sum(mask_array == 1))
        total_pixels = int(mask_array.size)
        water_area_km2 = water_pixels * (pixel_size_m ** 2) / 1e6

        result["stats"] = {
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "water_ratio": water_pixels / total_pixels if total_pixels > 0 else 0,
            "water_area_km2": round(water_area_km2, 4),
            "pixel_size_m": round(pixel_size_m, 2),
            "method": "AI (ONNX Runtime)",
            "inference_time_s": inference_result["inference_time_s"],
            "num_windows": inference_result["num_windows"],
        }

        # 矢量输出 (可选)
        if output_vector:
            try:
                _raster_to_vector(mask_array, profile, output_vector)
                result["vector_path"] = output_vector
            except Exception as e:
                result["vector_path"] = None
                warnings.warn(f"矢量转换失败: {e}")

        result["success"] = True

    except Exception as e:
        result["error"] = f"ONNX 推理失败: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def _raster_to_vector(mask_array: np.ndarray, profile: dict, output_path: str):
    """将掩膜栅格转换为 GeoJSON 多边形 (使用 rasterio.features)"""
    import rasterio.features
    import json

    # 提取水体多边形 (mask=1)
    shapes = rasterio.features.shapes(
        mask_array.astype(np.uint8),
        mask=mask_array == 1,
        transform=profile["transform"],
    )

    features = []
    for geom, value in shapes:
        if value == 1:
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {"class": "water"},
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": profile.get("crs", {}).get("init", "EPSG:4326")}},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)


# ============================================
# 性能基准对比
# ============================================

def benchmark_onnx_vs_pytorch(
    onnx_model_path: str,
    pytorch_model: Any = None,
    input_shape: Tuple[int, int, int, int] = (1, 6, 512, 512),
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    对比 ONNX Runtime 和 PyTorch 的推理速度

    参数:
        onnx_model_path: ONNX 模型路径
        pytorch_model: PyTorch 模型实例 (None=仅测ONNX)
        input_shape: 输入形状 (B, C, H, W)
        num_runs: 正式测试次数
        warmup_runs: 预热次数
        device: "cpu" 或 "cuda"

    返回:
        dict: {onnx_time_avg, torch_time_avg, speedup_ratio, ...}
    """
    import torch

    result = {
        "input_shape": input_shape,
        "device": device,
        "num_runs": num_runs,
        "onnx_time_avg_ms": None,
        "torch_time_avg_ms": None,
        "speedup_ratio": None,
    }

    dummy_input = torch.randn(*input_shape)

    # ONNX 基准
    try:
        session = get_ort_session(onnx_model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        ort_input = {input_name: dummy_input.numpy().astype(np.float32)}

        # 预热
        for _ in range(warmup_runs):
            session.run([output_name], ort_input)

        # 正式测试
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run([output_name], ort_input)
            times.append((time.perf_counter() - t0) * 1000)

        result["onnx_time_avg_ms"] = round(np.mean(times), 2)
        result["onnx_time_std_ms"] = round(np.std(times), 2)

    except Exception as e:
        result["onnx_error"] = str(e)

    # PyTorch 基准
    if pytorch_model is not None:
        try:
            pytorch_model.eval()
            torch_device = torch.device(device)
            pytorch_model.to(torch_device)
            dummy_input_t = dummy_input.to(torch_device)

            # 预热
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = pytorch_model(dummy_input_t)

            # 正式测试
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = pytorch_model(dummy_input_t)
                    times.append((time.perf_counter() - t0) * 1000)

            result["torch_time_avg_ms"] = round(np.mean(times), 2)
            result["torch_time_std_ms"] = round(np.std(times), 2)

        except Exception as e:
            result["torch_error"] = str(e)

    # 加速比
    if result["onnx_time_avg_ms"] and result["torch_time_avg_ms"]:
        result["speedup_ratio"] = round(
            result["torch_time_avg_ms"] / result["onnx_time_avg_ms"], 2
        )

    return result


# ============================================
# 模型信息展示
# ============================================

def inspect_onnx_model(model_path: str) -> Dict[str, Any]:
    """
    检查 ONNX 模型结构和元信息

    参数:
        model_path: ONNX 模型路径

    返回:
        dict: 包含输入/输出/算子统计等
    """
    try:
        import onnx
    except ImportError:
        return {"error": "onnx 库未安装 (pip install onnx)"}

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    graph = model.graph

    # 输入信息
    inputs = []
    for inp in graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else dim.dim_param)
        inputs.append({
            "name": inp.name,
            "shape": shape,
            "dtype": inp.type.tensor_type.elem_type,
        })

    # 输出信息
    outputs = []
    for out in graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else dim.dim_param)
        outputs.append({
            "name": out.name,
            "shape": shape,
            "dtype": out.type.tensor_type.elem_type,
        })

    # 算子统计
    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    return {
        "ir_version": model.ir_version,
        "opset_import": [(oi.domain, oi.version) for oi in model.opset_import],
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "inputs": inputs,
        "outputs": outputs,
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "top_ops": sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10],
    }
