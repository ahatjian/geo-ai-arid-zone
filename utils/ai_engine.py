"""
AI 推理引擎封装
基于 geoai-py 的预训练模型，提供 CPU-safe 语义分割推理接口
依赖: geoai-py >= 0.37, torch, rasterio
"""

import os
import tempfile
import warnings
import numpy as np
from typing import Optional, List, Dict, Union, Tuple

warnings.filterwarnings("ignore")

# 从统一配置导入缓存 TTL
from config import CACHE_CONFIG

# ---- 条件缓存装饰器 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    """条件缓存装饰器"""
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=CACHE_CONFIG.get("show_spinner", True))
    else:
        def noop_decorator(func):
            return func
        return noop_decorator


# ============================================
# 常量配置
# ============================================

# OmniWaterMask 预训练模型的 Sentinel-2 默认 band_order
# [R, G, B, NIR] — 1-based 波段索引
SENTINEL2_BAND_ORDER = [3, 2, 1, 4]  # B4(Red), B3(Green), B2(Blue), B8(NIR)

# 默认 AI 分割参数
DEFAULT_SEGMENT_PARAMS = {
    "patch_size": 1000,
    "overlap_size": 300,
    "batch_size": 4,
    "device": "cpu",
    "dtype": "float32",
    "min_size": 10,
    "smooth": True,
    "smooth_iterations": 3,
    "use_osm_water": False,  # 关闭 OSM 过滤，使用纯 ML 结果
    "use_osm_building": False,
    "use_osm_roads": False,
    "overwrite": True,
    "verbose": False,
}


# ============================================
# 核心: AI 水体分割
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_short"])
def segment_water_ai(
    input_path: str,
    band_order: Optional[List[int]] = None,
    output_raster: Optional[str] = None,
    output_vector: Optional[str] = None,
    device: str = "cpu",
    patch_size: int = 1000,
    overlap_size: int = 300,
    batch_size: int = 4,
    use_osm_filter: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, any]:
    """
    使用 geoai.segment_water() (OmniWaterMask 预训练模型) 进行水体语义分割

    OmniWaterMask 是无需额外训练的开箱即用水体分割模型，基于 4 波段
    (R, G, B, NIR) 输入。

    参数:
        input_path: 输入多波段 GeoTIFF 路径 (至少包含 R, G, B, NIR)
        band_order: 波段顺序 [R_idx, G_idx, B_idx, NIR_idx]，1-based。
                    默认 Sentinel-2: [3, 2, 1, 4] 即 B4, B3, B2, B8
        output_raster: 输出水体掩膜 GeoTIFF 路径 (可选)
        output_vector: 输出水体多边形 GeoJSON 路径 (可选)
        device: 推理设备 "cpu" 或 "cuda"
        patch_size: 滑动窗口大小 (像素)
        overlap_size: 窗口重叠大小 (像素)
        batch_size: 批处理大小
        use_osm_filter: 是否使用 OSM 数据过滤误检 (默认 False=纯 ML)
        cache_dir: 模型缓存目录 (可选)
        **kwargs: 传递给 segment_water 的其他参数

    返回:
        dict: {
            "raster_path": 水体掩膜 GeoTIFF 路径 (str),
            "vector_path": 水体多边形 GeoJSON 路径 (str or None),
            "mask_array": 水体掩膜 numpy 数组 (0=非水体, 1=水体),
            "stats": 面积统计 dict,
            "success": bool,
            "error": str or None,
        }
    """
    import geoai
    import rasterio

    result = {
        "raster_path": None,
        "vector_path": None,
        "mask_array": None,
        "stats": None,
        "success": False,
        "error": None,
    }

    # 设置 band_order
    if band_order is None:
        band_order = SENTINEL2_BAND_ORDER

    # 输出路径处理
    if output_raster is None:
        tmp_dir = tempfile.gettempdir()
        output_raster = os.path.join(
            tmp_dir, f"ai_water_mask_{os.path.basename(input_path).replace('.tif', '')}.tif"
        )

    if output_vector is None:
        tmp_dir = tempfile.gettempdir()
        output_vector = os.path.join(
            tmp_dir, f"ai_water_vector_{os.path.basename(input_path).replace('.tif', '')}.geojson"
        )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_raster) if os.path.dirname(output_raster) else tempfile.gettempdir(),
                exist_ok=True)
    os.makedirs(os.path.dirname(output_vector) if os.path.dirname(output_vector) else tempfile.gettempdir(),
                exist_ok=True)

    try:
        # ============================================
        # 调用 geoai.segment_water
        # ============================================
        segment_result = geoai.segment_water(
            input_path=input_path,
            band_order=band_order,
            output_raster=output_raster,
            output_vector=output_vector,
            batch_size=batch_size,
            device=device,
            dtype=DEFAULT_SEGMENT_PARAMS["dtype"],
            patch_size=patch_size,
            overlap_size=overlap_size,
            use_osm_water=use_osm_filter or DEFAULT_SEGMENT_PARAMS["use_osm_water"],
            use_osm_building=use_osm_filter or DEFAULT_SEGMENT_PARAMS["use_osm_building"],
            use_osm_roads=use_osm_filter or DEFAULT_SEGMENT_PARAMS["use_osm_roads"],
            cache_dir=cache_dir,
            min_size=DEFAULT_SEGMENT_PARAMS["min_size"],
            smooth=DEFAULT_SEGMENT_PARAMS["smooth"],
            smooth_iterations=DEFAULT_SEGMENT_PARAMS["smooth_iterations"],
            overwrite=DEFAULT_SEGMENT_PARAMS["overwrite"],
            verbose=DEFAULT_SEGMENT_PARAMS["verbose"],
            **kwargs,
        )

        # ============================================
        # 读取结果掩膜
        # ============================================
        if os.path.exists(output_raster):
            with rasterio.open(output_raster) as src:
                mask_arr = src.read(1)

            result["raster_path"] = output_raster
            result["mask_array"] = mask_arr
        else:
            result["error"] = f"输出掩膜文件未生成: {output_raster}"
            return result

        # ============================================
        # 检查矢量输出
        # ============================================
        if os.path.exists(output_vector):
            result["vector_path"] = output_vector

        # ============================================
        # 计算面积统计
        # ============================================
        # 从原始影像获取 geotransform 以计算像元大小
        with rasterio.open(input_path) as src:
            pixel_size_x = abs(src.transform.a)
            pixel_size_y = abs(src.transform.e)
            pixel_size_m = (pixel_size_x + pixel_size_y) / 2  # 取平均

        water_pixels = int(np.sum(mask_arr == 1))
        total_pixels = int(mask_arr.size)
        water_area_km2 = water_pixels * (pixel_size_m ** 2) / 1e6

        result["stats"] = {
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "water_ratio": water_pixels / total_pixels if total_pixels > 0 else 0,
            "water_area_km2": round(water_area_km2, 4),
            "pixel_size_m": round(pixel_size_m, 2),
            "method": "AI (OmniWaterMask UNet)",
        }

        result["success"] = True

    except ImportError as e:
        result["error"] = f"缺少依赖: {e}. 请确保 geoai-py 已安装 (conda activate geo-ai)"
    except Exception as e:
        result["error"] = f"AI 分割失败: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


# ============================================
# 从用户波段映射创建 band_order
# ============================================

def build_band_order(
    band_red: int,
    band_green: int,
    band_blue: int,
    band_nir: int,
) -> List[int]:
    """
    根据用户指定的波段索引构建 segment_water 所需的 band_order

    segment_water 的 band_order 格式: [R_idx, G_idx, B_idx, NIR_idx] (1-based)

    参数:
        band_red: 红波段索引 (1-based)
        band_green: 绿波段索引 (1-based)
        band_blue: 蓝波段索引 (1-based)
        band_nir: 近红外波段索引 (1-based)

    返回:
        [red_idx, green_idx, blue_idx, nir_idx]
    """
    return [band_red, band_green, band_blue, band_nir]


# ============================================
# 批量 AI 水体分割 (Phase 2)
# ============================================

def segment_water_ai_batch(
    input_paths: List[str],
    output_dir: str,
    band_order: Optional[List[int]] = None,
    device: str = "cpu",
    patch_size: int = 1000,
    overlap_size: int = 300,
    batch_size: int = 4,
    **kwargs,
) -> List[Dict[str, any]]:
    """
    批量 AI 水体分割 (多景影像)

    参数:
        input_paths: 输入影像路径列表
        output_dir: 输出目录
        band_order: 波段顺序
        device: 推理设备
        patch_size: 窗口大小
        overlap_size: 重叠大小
        batch_size: 批处理大小
        **kwargs: 传递给 segment_water_ai 的其他参数

    返回:
        list[dict]: 每景影像的分割结果
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, input_path in enumerate(input_paths):
        basename = os.path.splitext(os.path.basename(input_path))[0]
        raster_out = os.path.join(output_dir, f"{basename}_water_mask.tif")
        vector_out = os.path.join(output_dir, f"{basename}_water_vector.geojson")

        result = segment_water_ai(
            input_path=input_path,
            band_order=band_order,
            output_raster=raster_out,
            output_vector=vector_out,
            device=device,
            patch_size=patch_size,
            overlap_size=overlap_size,
            batch_size=batch_size,
            **kwargs,
        )
        results.append(result)

    return results


# ============================================
# 模型信息
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_long"])
def get_model_info() -> Dict[str, str]:
    """
    获取当前使用的 AI 模型信息

    返回:
        dict: 包含模型名称、架构、输入波段等信息的字典
    """
    return {
        "model_name": "OmniWaterMask",
        "architecture": "UNet",
        "encoder": "ResNet-based (预训练)",
        "num_bands": 4,
        "input_bands": "Red, Green, Blue, NIR (按此顺序)",
        "output_classes": "2 (背景, 水体)",
        "training_data": "全球多源水体标注数据集",
        "framework": "geoai-py >= 0.37 + PyTorch",
        "citation": "geoai-py OmniWaterMask — 开箱即用，无需额外训练",
    }
