# TODO: 拆分超大文件 (933行) — 按功能模式拆为独立组件
"""
水体监测页面 — MNDWI / AWEIsh 指数阈值法 + AI 语义分割 (OmniWaterMask)
支持多波段 GeoTIFF 输入
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS, INDEX_THRESHOLDS
from utils.error_handler import StreamlitErrorBoundary, safe_execute

st.set_page_config(page_title="水体监测", page_icon="💧", layout="wide")

# ============================================
# 侧边栏 - 方法选择 + 参数配置
# ============================================
with st.sidebar:
    st.title("💧 水体监测设置")

    # --- 分析方法选择 ---
    st.subheader("🔬 分析方法")
    analysis_method = st.radio(
        "选择水体提取方法",
        [
            "📊 指数阈值法 (MNDWI/AWEIsh)",
            "🤖 AI 智能分割 (OmniWaterMask)",
            "⚡ ONNX 推理加速",
        ],
        help="指数阈值法: 传统遥感指数 + 阈值分割\n"
             "AI 分割: geoai-py 预训练 UNet 深度学习模型\n"
             "ONNX 加速: 独立 ONNX Runtime 推理，无需 geoai-py"
    )

    st.divider()

    # --- 方法标志 ---
    is_ai_mode = "🤖" in analysis_method
    is_onnx_mode = "⚡" in analysis_method

    # --- 数据源 (共享) ---
    st.subheader("数据源")
    data_mode = st.radio(
        "选择数据来源",
        ["📡 从数据浏览页获取", "📁 上传本地多波段 GeoTIFF"],
    )

    st.divider()

    if is_ai_mode:
        # ============================================
        # AI 模式参数
        # ============================================
        st.subheader("🤖 AI 模型配置")

        st.info(
            "**OmniWaterMask** — 预训练 UNet 水体分割模型\n\n"
            "基于全球多源水体标注数据训练，无需额外训练即可使用。\n"
            "输入: RGB + NIR 4波段"
        )

        # 推理设备
        ai_device = st.selectbox(
            "推理设备",
            ["cpu", "cuda"],
            index=0,
            help="CPU 推理较慢但稳定；有 NVIDIA GPU 可选 CUDA 加速"
        )

        # 高级参数折叠
        with st.expander("⚙️ 高级参数"):
            ai_patch_size = st.slider(
                "滑动窗口大小", 256, 2000, 1000, 100,
                help="越大越快但对显存要求更高"
            )
            ai_overlap = st.slider(
                "窗口重叠", 0, 500, 300, 50,
                help="重叠越大边缘越平滑，但推理更慢"
            )
            ai_batch_size = st.slider(
                "批处理大小", 1, 16, 4,
                help="CPU 推荐 1-4, GPU 可设更大"
            )
            ai_min_size = st.number_input(
                "最小水体面积 (像素)", 1, 500, 10,
                help="过滤小于此面积的小斑块"
            )
            ai_use_osm = st.checkbox(
                "使用 OSM 辅助过滤",
                value=False,
                help="利用 OpenStreetMap 水系数据过滤误检（可能引入漏检）"
            )

        st.divider()

        # 像元尺寸 (AI 自动从影像读取，这里仅作参考)
        st.subheader("像元参数")
        pixel_size = st.number_input(
            "像元尺寸 (米)",
            value=10.0, min_value=1.0, max_value=100.0,
            help="仅用于面积估算；AI 模式会自动从 GeoTIFF 读取实际像元大小"
        )

    elif is_onnx_mode:
        # ============================================
        # ONNX 推理加速参数
        # ============================================
        st.subheader("⚡ ONNX 推理配置")

        # 检查 ONNX Runtime 可用性
        try:
            from utils.onnx_engine import check_onnx_available
            onnx_info = check_onnx_available()
            if onnx_info["onnx_available"]:
                st.success(f"✅ ONNX Runtime {onnx_info['version']}")
                if onnx_info["cuda_available"]:
                    st.success("🚀 CUDA 加速可用")
                provider_list = ", ".join(onnx_info["providers"])
                st.caption(f"可用 Provider: {provider_list}")
            else:
                st.error("❌ ONNX Runtime 未安装")
                st.code("pip install onnxruntime  # CPU\npip install onnxruntime-gpu  # GPU")
        except Exception as e:
            st.warning(f"ONNX 检测失败: {e}")

        st.info(
            "**ONNX Runtime** — 独立推理引擎\n\n"
            "无需 geoai-py 和 PyTorch，仅需 onnxruntime + numpy + rasterio。\n"
            "推理速度比原生 PyTorch 快 2-5x。"
        )

        # ONNX 模型选择
        onnx_model_source = st.radio(
            "ONNX 模型来源",
            ["📦 上传 .onnx 模型文件", "📂 使用预设路径"],
            help="上传本地的 ONNX 水体分割模型，或指定服务器上的路径"
        )

        if "上传" in onnx_model_source:
            onnx_model_file = st.file_uploader(
                "上传水体分割 ONNX 模型",
                type=["onnx"],
                key="onnx_model_upload",
                help="由 geoai-py 训练导出的水体分割 ONNX 模型 (2类: 背景/水体)"
            )
            onnx_model_path = None  # 运行时处理
        else:
            from config import ONNX_MODELS
            default_onnx = ONNX_MODELS.get("water_seg", "models/water_seg_unet_resnet34.onnx")
            onnx_model_path = st.text_input(
                "ONNX 模型路径",
                value=default_onnx,
                help="服务器上的 ONNX 模型文件绝对路径"
            )
            onnx_model_file = None
            if os.path.exists(onnx_model_path):
                st.success(f"✅ 模型已就绪: {os.path.basename(onnx_model_path)}")
            else:
                st.warning(f"⚠️ 模型文件未找到: {onnx_model_path}")

        # 推理提供者选择
        try:
            from utils.onnx_engine import check_onnx_available
            available_providers = check_onnx_available()["providers"]
            if not available_providers:
                available_providers = ["CPUExecutionProvider"]
        except Exception:
            available_providers = ["CPUExecutionProvider"]

        onnx_provider = st.selectbox(
            "推理提供者",
            available_providers,
            index=0,
            help="CPUExecutionProvider: CPU推理\nCUDAExecutionProvider: GPU加速"
        )

        # 高级参数
        with st.expander("⚙️ ONNX 高级参数"):
            from config import ONNX_CONFIG
            onnx_window_size = st.slider(
                "滑动窗口大小", 128, 2000, ONNX_CONFIG.get("window_size", 512), 64,
                help="越大推理越快但内存消耗更高"
            )
            onnx_overlap = st.slider(
                "窗口重叠", 0, 500, ONNX_CONFIG.get("overlap", 256), 32,
                help="重叠越大边缘越平滑"
            )
            onnx_batch_size = st.slider(
                "批次大小", 1, 8, ONNX_CONFIG.get("batch_size", 4),
                help="CPU 推荐 1-2"
            )

        st.divider()

        # 像元尺寸
        st.subheader("像元参数")
        pixel_size = st.number_input(
            "像元尺寸 (米)",
            value=10.0, min_value=1.0, max_value=100.0,
            help="仅用于面积估算；ONNX 模式会自动从 GeoTIFF 读取实际像元大小"
        )

    else:
        # ============================================
        # 指数模式参数 (原有)
        # ============================================
        st.subheader("水体指数")
        index_type = st.radio(
            "选择指数",
            ["MNDWI (推荐)", "AWEIsh"],
            help="MNDWI 适合开放水体，AWEIsh 可区分阴影"
        )

        st.divider()

        st.subheader("提取阈值")
        threshold = st.slider(
            "水体阈值",
            -1.0, 1.0,
            0.0, 0.05,
            help="指数值 > 阈值 视为水体。默认 0.0，可调高以过滤误检"
        )

        st.divider()

        st.subheader("像元参数")
        pixel_size = st.number_input(
            "像元尺寸 (米)",
            value=10.0, min_value=1.0, max_value=100.0,
            help="Sentinel-2=10m, Landsat=30m"
        )

# ============================================
# 主页面
# ============================================
if is_ai_mode:
    st.title("💧 水体动态监测 — AI 智能分割")
    st.markdown("基于 OmniWaterMask 预训练 UNet 模型的深度学习水体分割，无需手动调整阈值")
elif is_onnx_mode:
    st.title("💧 水体动态监测 — ONNX 推理加速")
    st.markdown("基于 ONNX Runtime 的独立推理引擎，速度比 PyTorch 快 2-5x，无需 geoai-py")
else:
    st.title("💧 水体动态监测")
    st.markdown("MNDWI / AWEIsh 自动计算 + 水体面积统计")

# ============================================
# 数据加载 (共享)
# ============================================
geotiff_path = None

if "📡" in data_mode:
    st.info("👆 请先在「数据浏览」页面搜索影像并下载全波段 GeoTIFF，然后在此上传进行分析")
    st.markdown("""
    **推荐流程**:
    1. 进入「🗺️ 数据浏览」页面
    2. 搜索数据 → 选择卫星 → 搜索影像
    3. 选择一景影像 → 「全波段合成下载」
    4. 回到本页，上传下载的 GeoTIFF
    """)

    uploaded = st.file_uploader("或直接在此上传多波段 GeoTIFF", type=["tif", "tiff"], key="water_local")
    if uploaded:
        tmp_dir = tempfile.gettempdir()
        geotiff_path = os.path.join(tmp_dir, f"water_{uploaded.name}")
        with open(geotiff_path, "wb") as f:
            f.write(uploaded.getvalue())
        st.success(f"✅ 已加载: {uploaded.name}")
else:
    required_bands = "R, G, B, NIR (4波段)" if is_ai_mode else "Green, NIR, SWIR1 (至少3波段)"
    uploaded = st.file_uploader(
        f"上传多波段 GeoTIFF ({required_bands})",
        type=["tif", "tiff"]
    )
    if uploaded:
        tmp_dir = tempfile.gettempdir()
        geotiff_path = os.path.join(tmp_dir, f"water_{uploaded.name}")
        with open(geotiff_path, "wb") as f:
            f.write(uploaded.getvalue())
        st.success(f"✅ 已加载: {uploaded.name}")

# ============================================
# 影像信息 + 波段映射
# ============================================
if geotiff_path:
    import rasterio

    with rasterio.open(geotiff_path) as src:
        n_bands = src.count
        img_shape = (src.height, src.width)
        # 读取实际像元大小
        actual_pixel_x = abs(src.transform.a)
        actual_pixel_y = abs(src.transform.e)
        actual_pixel_m = (actual_pixel_x + actual_pixel_y) / 2

    st.divider()
    st.subheader("📊 影像信息")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("波段数", n_bands)
    with col2:
        st.metric("尺寸", f"{img_shape[1]} × {img_shape[0]}")
    with col3:
        st.metric("像元尺寸", f"{actual_pixel_m:.1f}m")
    with col4:
        area_km2 = img_shape[0] * img_shape[1] * actual_pixel_m * actual_pixel_m / 1e6
        st.metric("覆盖面积", f"{area_km2:.1f} km²")

    # --- 波段映射 ---
    st.subheader("🔧 波段映射设置")

    if is_ai_mode or is_onnx_mode:
        st.markdown("AI/ONNX 模式仅需 **R, G, B, NIR** 四个波段 (按此顺序)")
        col1, col2 = st.columns(2)
        with col1:
            band_red = st.number_input("Red 红", 1, max(1, n_bands), 3, key="ai_band_red")
            band_green = st.number_input("Green 绿", 1, max(1, n_bands), 2, key="ai_band_green")
        with col2:
            band_blue = st.number_input("Blue 蓝", 1, max(1, n_bands), 1, key="ai_band_blue")
            band_nir = st.number_input("NIR 近红外", 1, max(1, n_bands), 4, key="ai_band_nir")
    else:
        st.markdown("请指定 6 个波段在 GeoTIFF 中的位置 (1-based 索引)")
        col1, col2, col3 = st.columns(3)
        with col1:
            band_blue = st.number_input("Blue 蓝", 1, max(1, n_bands), 1)
            band_green = st.number_input("Green 绿", 1, max(1, n_bands), 2)
        with col2:
            band_red = st.number_input("Red 红", 1, max(1, n_bands), 3)
            band_nir = st.number_input("NIR 近红外", 1, max(1, n_bands), 4)
        with col3:
            band_swir1 = st.number_input("SWIR1", 1, max(1, n_bands), 5)
            band_swir2 = st.number_input("SWIR2", 1, max(1, n_bands), 6)

    # ============================================
    # ============================================
    # 指数阈值法 执行逻辑
    # ============================================
    # ============================================
    if not is_ai_mode and not is_onnx_mode:
        if st.button("🔬 计算水体指数", type="primary"):
            with st.spinner("正在计算水体指数..."):
                with StreamlitErrorBoundary("水体指数计算", st=st, show_traceback=True):
                    from utils.indices import (
                        calc_mndwi, calc_aweish,
                        extract_water_mndwi, extract_water_aweish,
                        calc_water_area, save_index_geotiff, save_mask_geotiff,
                    )
                    from utils.visualization import render_mndwi, render_aweish, render_water_mask

                    # 读取波段
                    with rasterio.open(geotiff_path) as src:
                        blue_arr = src.read(band_blue).astype(np.float32)
                        green_arr = src.read(band_green).astype(np.float32)
                        red_arr = src.read(band_red).astype(np.float32)
                        nir_arr = src.read(band_nir).astype(np.float32)
                        swir1_arr = src.read(band_swir1).astype(np.float32)
                        swir2_arr = src.read(band_swir2).astype(np.float32)

                    # 计算指数
                    if "MNDWI" in index_type:
                        index_arr = calc_mndwi(green_arr, swir1_arr)
                        water_mask = extract_water_mndwi(index_arr, threshold=threshold)
                    else:
                        index_arr = calc_aweish(green_arr, nir_arr, swir1_arr, swir2_arr)
                        water_mask = extract_water_aweish(index_arr, threshold=threshold)

                    stats = calc_water_area(water_mask, pixel_size_m=pixel_size)

                    # 保存到 session
                    st.session_state["water_index"] = index_arr
                    st.session_state["water_mask"] = water_mask
                    st.session_state["water_stats"] = stats
                    st.session_state["water_geotiff"] = geotiff_path
                    st.session_state["water_index_type"] = index_type

                    # ---- 结果展示 ----
                    st.divider()
                    st.subheader("📊 水体面积统计")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("水体像元数", f"{stats['water_pixels']:,}")
                    with cols[1]:
                        st.metric("总像元数", f"{stats['total_pixels']:,}")
                    with cols[2]:
                        st.metric("水体占比", f"{stats['water_ratio']*100:.2f}%")
                    with cols[3]:
                        st.metric("水体面积", f"{stats['water_area_km2']:.2f} km²")

                    # AI 智能解读 (统一组件)
                    from utils.ai_insight import render_ai_insight_block
                    render_ai_insight_block(
                        analysis_type="水体监测分析",
                        metrics={
                            "水体占比": stats["water_ratio"],
                            "水体面积(km²)": stats["water_area_km2"],
                        },
                        key_suffix="water_ai",
                        show_button=True,
                    )

                    st.divider()
                    st.subheader("🗺️ 水体指数 / 掩膜可视化")
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        if "MNDWI" in index_type:
                            img = render_mndwi(index_arr, title=f"MNDWI (阈值={threshold})")
                        else:
                            img = render_aweish(index_arr, title=f"AWEIsh (阈值={threshold})")
                        st.image(img)
                    with viz_col2:
                        mask_img = render_water_mask(water_mask, title=f"水体提取结果 (>{threshold})")
                        st.image(mask_img)

                    st.divider()
                    st.subheader("📈 指数分布")
                    from utils.visualization import plot_histogram
                    hist_fig = plot_histogram(
                        index_arr, bins=80,
                        x_label=index_type.split(" ")[0],
                        title=f"{index_type.split(' ')[0]} 分布直方图",
                    )
                    st.plotly_chart(hist_fig)

                    # ---- 导出 ----
                    st.divider()
                    st.subheader("💾 结果导出")
                    exp_col1, exp_col2, exp_col3 = st.columns(3)
                    tmp_dir = tempfile.gettempdir()
                    index_name = "MNDWI" if "MNDWI" in index_type else "AWEIsh"

                    with exp_col1:
                        index_tif = os.path.join(tmp_dir, f"{index_name}.tif")
                        save_index_geotiff(index_arr.astype(np.float32), geotiff_path, index_tif)
                        with open(index_tif, "rb") as f:
                            st.download_button(
                                f"⬇️ {index_name} GeoTIFF", f,
                                file_name=f"{index_name}.tif",
                                mime="image/tiff", width="stretch",
                            )
                    with exp_col2:
                        mask_tif = os.path.join(tmp_dir, f"{index_name}_water_mask.tif")
                        save_mask_geotiff(water_mask, geotiff_path, mask_tif)
                        with open(mask_tif, "rb") as f:
                            st.download_button(
                                "⬇️ 水体掩膜 GeoTIFF", f,
                                file_name=f"{index_name}_water_mask.tif",
                                mime="image/tiff", width="stretch",
                            )
                    with exp_col3:
                        csv_buf = BytesIO()
                        pd.DataFrame([stats]).to_csv(csv_buf, index=False, encoding="utf-8-sig")
                        csv_buf.seek(0)
                        st.download_button(
                            "⬇️ 统计结果 CSV", csv_buf,
                                file_name=f"{index_name}_stats.csv",
                                mime="text/csv", width="stretch",
                            )

    # ============================================
    # ============================================
    # AI 智能分割 执行逻辑
    # ============================================
    # ============================================
    elif is_ai_mode:
        if st.button("🤖 运行 AI 水体分割", type="primary"):
            with st.spinner("🤖 AI 模型正在推理中... 这可能需要几分钟，请耐心等待"):
                with StreamlitErrorBoundary("AI 水体分割", st=st, show_traceback=True):
                    from utils.ai_engine import segment_water_ai, build_band_order, get_model_info
                    from utils.visualization import render_water_mask

                    # 构建 band_order
                    band_order = build_band_order(band_red, band_green, band_blue, band_nir)
                    st.info(f"波段顺序: [R={band_red}, G={band_green}, B={band_blue}, NIR={band_nir}]")

                    # 运行 AI 分割
                    result = segment_water_ai(
                        input_path=geotiff_path,
                        band_order=band_order,
                        device=ai_device,
                        patch_size=ai_patch_size,
                        overlap_size=ai_overlap,
                        batch_size=ai_batch_size,
                        use_osm_filter=ai_use_osm,
                    )

                    if not result["success"]:
                        st.error(f"❌ AI 分割失败: {result['error']}")
                        if "traceback" in result:
                            with st.expander("🔍 错误详情"):
                                st.code(result["traceback"])
                    else:
                        water_mask = result["mask_array"]
                        stats = result["stats"]

                        # 降级提示 (模型不可用时自动降级为 Otsu)
                        if result.get("error") and "降级" in str(result["error"]):
                            st.warning(f"⚠️ {result['error']}")

                        # ---- 面积统计 ----
                        st.divider()
                        st.subheader("📊 AI 水体面积统计")
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("水体像元数", f"{stats['water_pixels']:,}")
                        with cols[1]:
                            st.metric("总像元数", f"{stats['total_pixels']:,}")
                        with cols[2]:
                            st.metric("水体占比", f"{stats['water_ratio']*100:.2f}%")
                        with cols[3]:
                            st.metric("水体面积", f"{stats['water_area_km2']:.2f} km²")
                        st.caption(f"方法: {stats['method']} | 像元大小: {stats['pixel_size_m']}m")

                        # ---- 可视化 ----
                        st.divider()
                        st.subheader("🗺️ AI 水体分割结果")

                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            mask_img = render_water_mask(
                                water_mask,
                                title="AI 水体分割掩膜 (OmniWaterMask)"
                            )
                            st.image(mask_img)

                        with viz_col2:
                            # 显示 RGB 预览 + 叠加
                            from utils.visualization import render_index
                            with rasterio.open(geotiff_path) as src:
                                rgb_data = np.stack([
                                    src.read(band_red).astype(np.float32),
                                    src.read(band_green).astype(np.float32),
                                    src.read(band_blue).astype(np.float32),
                                ], axis=-1)
                                # 归一化到 0-1
                                for i in range(3):
                                    p2, p98 = np.percentile(rgb_data[..., i], (2, 98))
                                    rgb_data[..., i] = np.clip((rgb_data[..., i] - p2) / (p98 - p2), 0, 1)

                            from PIL import Image
                            rgb_img = Image.fromarray((rgb_data * 255).astype(np.uint8))
                            overlay_img = render_water_mask(
                                water_mask, rgb_image=rgb_img,
                                title="AI 水体叠加 RGB"
                            )
                            st.image(overlay_img)

                        # ---- 直方图 ----
                        st.divider()
                        st.subheader("📈 水体掩膜分布")
                        from utils.visualization import plot_histogram
                        hist_fig = plot_histogram(
                            water_mask.astype(np.float32),
                            bins=3,
                            x_label="类别 (0=非水体, 1=水体)",
                            title="AI 水体分割类别分布",
                            color="#0066FF",
                        )
                        st.plotly_chart(hist_fig)

                        # ---- 模型信息 ----
                        with st.expander("🔬 模型详情"):
                            model_info = get_model_info()
                            for k, v in model_info.items():
                                st.text(f"{k}: {v}")

                        # ---- 导出 ----
                        st.divider()
                        st.subheader("💾 结果导出")

                        exp_col1, exp_col2, exp_col3 = st.columns(3)

                        with exp_col1:
                            # 导出 AI 掩膜 GeoTIFF
                            if result["raster_path"] and os.path.exists(result["raster_path"]):
                                with open(result["raster_path"], "rb") as f:
                                    st.download_button(
                                        "⬇️ AI 水体掩膜 GeoTIFF", f,
                                        file_name="ai_water_mask.tif",
                                        mime="image/tiff", width="stretch",
                                    )
                            else:
                                st.warning("掩膜文件不可用")

                        with exp_col2:
                            # 导出矢量 GeoJSON
                            if result["vector_path"] and os.path.exists(result["vector_path"]):
                                with open(result["vector_path"], "rb") as f:
                                    st.download_button(
                                        "⬇️ 水体多边形 GeoJSON", f,
                                        file_name="ai_water_polygons.geojson",
                                        mime="application/geo+json", width="stretch",
                                    )
                            else:
                                st.info("矢量文件未生成 (AI 模式下可选)")

                        with exp_col3:
                            # 导出统计 CSV
                            csv_buf = BytesIO()
                            pd.DataFrame([stats]).to_csv(csv_buf, index=False, encoding="utf-8-sig")
                            csv_buf.seek(0)
                            st.download_button(
                                "⬇️ 统计结果 CSV", csv_buf,
                                file_name="ai_water_stats.csv",
                                mime="text/csv", width="stretch",
                            )

    # ============================================
    # ============================================
    # ONNX 推理加速 执行逻辑
    # ============================================
    # ============================================
    elif is_onnx_mode:
        onnx_btn_disabled = False
        onnx_btn_help = None

        # 检查 ONNX 模型是否可用
        actual_onnx_path = None
        if onnx_model_file is not None:
            # 用户上传的模型
            tmp_dir = tempfile.gettempdir()
            actual_onnx_path = os.path.join(tmp_dir, f"uploaded_water_{onnx_model_file.name}")
            with open(actual_onnx_path, "wb") as f:
                f.write(onnx_model_file.getvalue())
        else:
            actual_onnx_path = onnx_model_path

        if actual_onnx_path is None or not os.path.exists(actual_onnx_path):
            onnx_btn_disabled = True
            onnx_btn_help = "请先上传 ONNX 模型文件或指定有效路径"

        # 检查 ONNX Runtime
        try:
            from utils.onnx_engine import check_onnx_available
            onnx_status = check_onnx_available()
            if not onnx_status["onnx_available"]:
                onnx_btn_disabled = True
                onnx_btn_help = "ONNX Runtime 未安装: pip install onnxruntime"
        except Exception:
            pass

        if st.button(
            "⚡ 运行 ONNX 水体分割",
            type="primary",
            width="stretch",
            disabled=onnx_btn_disabled,
            help=onnx_btn_help,
        ):
            with st.spinner("⚡ ONNX Runtime 正在推理中..."):
                with StreamlitErrorBoundary("ONNX 水体分割", st=st, show_traceback=True):
                    from utils.onnx_engine import segment_water_onnx, inspect_onnx_model

                    # 波段索引 (1-based → band_indices)
                    band_indices = [band_red, band_green, band_blue, band_nir]

                    # 输出路径
                    tmp_dir = tempfile.gettempdir()
                    raster_out = os.path.join(
                        tmp_dir,
                        f"onnx_water_mask_{os.path.basename(geotiff_path).replace('.tif', '')}.tif"
                    )
                    vector_out = os.path.join(
                        tmp_dir,
                        f"onnx_water_vector_{os.path.basename(geotiff_path).replace('.tif', '')}.geojson"
                    )

                    # 进度回调
                    onnx_progress = st.progress(0, "ONNX 推理中...")
                    def progress_callback(completed, total):
                        onnx_progress.progress(
                            min(completed / total, 1.0),
                            f"ONNX 推理: {completed}/{total} 窗口"
                        )

                    # 执行 ONNX 推理
                    result = segment_water_onnx(
                        input_path=geotiff_path,
                        onnx_model_path=actual_onnx_path,
                        window_size=onnx_window_size,
                        overlap=onnx_overlap,
                        device="cuda" if "CUDA" in onnx_provider else "cpu",
                        output_raster=raster_out,
                        output_vector=vector_out,
                        band_indices=band_indices,
                        progress_callback=progress_callback,
                    )
                    onnx_progress.empty()

                    if not result["success"]:
                        st.error(f"❌ ONNX 推理失败: {result['error']}")
                        if "traceback" in result:
                            with st.expander("🔍 错误详情"):
                                st.code(result["traceback"])
                    else:
                        water_mask = result["mask_array"]
                        stats = result["stats"]

                        # ---- 面积统计 ----
                        st.divider()
                        st.subheader("📊 ONNX 水体面积统计")
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("水体像元数", f"{stats['water_pixels']:,}")
                        with cols[1]:
                            st.metric("总像元数", f"{stats['total_pixels']:,}")
                        with cols[2]:
                            st.metric("水体占比", f"{stats['water_ratio']*100:.2f}%")
                        with cols[3]:
                            st.metric("水体面积", f"{stats['water_area_km2']:.2f} km²")

                        # ONNX 性能指标
                        perf_cols = st.columns(3)
                        with perf_cols[0]:
                            st.metric("推理耗时", f"{stats.get('inference_time_s', 0):.1f}s")
                        with perf_cols[1]:
                            st.metric("窗口数", stats.get('num_windows', 0))
                        with perf_cols[2]:
                            st.metric("方法", stats.get('method', 'ONNX'))
                        st.caption(f"Provider: {onnx_provider} | 像元: {stats['pixel_size_m']}m")

                        # ---- 可视化 ----
                        st.divider()
                        st.subheader("🗺️ ONNX 水体分割结果")

                        from utils.visualization import render_water_mask
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            mask_img = render_water_mask(
                                water_mask,
                                title="ONNX 水体分割掩膜"
                            )
                            st.image(mask_img)

                        with viz_col2:
                            # RGB 叠加
                            with rasterio.open(geotiff_path) as src:
                                rgb_data = np.stack([
                                    src.read(band_red).astype(np.float32),
                                    src.read(band_green).astype(np.float32),
                                    src.read(band_blue).astype(np.float32),
                                ], axis=-1)
                                for i in range(3):
                                    p2, p98 = np.percentile(rgb_data[..., i], (2, 98))
                                    rgb_data[..., i] = np.clip(
                                        (rgb_data[..., i] - p2) / (p98 - p2), 0, 1
                                    )
                            from PIL import Image
                            rgb_img = Image.fromarray((rgb_data * 255).astype(np.uint8))
                            overlay_img = render_water_mask(
                                water_mask, rgb_image=rgb_img,
                                title="ONNX 水体叠加 RGB"
                            )
                            st.image(overlay_img)

                        # ---- ONNX 模型信息 ----
                        with st.expander("🔬 ONNX 模型详情"):
                            try:
                                model_info = inspect_onnx_model(actual_onnx_path)
                                if "error" in model_info:
                                    st.warning(model_info["error"])
                                else:
                                    st.json({
                                        "IR 版本": model_info.get("ir_version"),
                                        "Opset": model_info.get("opset_import"),
                                        "模型大小": f"{model_info.get('model_size_mb', 0)} MB",
                                        "节点数": model_info.get("num_nodes"),
                                        "输入": model_info.get("inputs"),
                                        "输出": model_info.get("outputs"),
                                    })
                            except Exception as e:
                                st.warning(f"模型检查失败: {e}")

                        # ---- 导出 ----
                        st.divider()
                        st.subheader("💾 结果导出")
                        exp_col1, exp_col2, exp_col3 = st.columns(3)

                        with exp_col1:
                            if result["raster_path"] and os.path.exists(result["raster_path"]):
                                with open(result["raster_path"], "rb") as f:
                                    st.download_button(
                                        "⬇️ ONNX 水体掩膜 GeoTIFF", f,
                                        file_name="onnx_water_mask.tif",
                                        mime="image/tiff", width="stretch",
                                    )
                            else:
                                st.warning("掩膜文件不可用")

                        with exp_col2:
                            if result["vector_path"] and os.path.exists(result["vector_path"]):
                                with open(result["vector_path"], "rb") as f:
                                    st.download_button(
                                        "⬇️ 水体多边形 GeoJSON", f,
                                        file_name="onnx_water_polygons.geojson",
                                        mime="application/geo+json", width="stretch",
                                    )
                            else:
                                st.info("矢量未生成")

                        with exp_col3:
                            csv_buf = BytesIO()
                            pd.DataFrame([stats]).to_csv(csv_buf, index=False, encoding="utf-8-sig")
                            csv_buf.seek(0)
                            st.download_button(
                                "⬇️ 统计结果 CSV", csv_buf,
                                file_name="onnx_water_stats.csv",
                                mime="text/csv", width="stretch",
                            )

else:
    # ============================================
    # 无数据时 — 显示使用说明
    # ============================================
    st.divider()
    st.subheader("📖 使用方法")

    if is_ai_mode:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🤖 AI 智能分割 (OmniWaterMask)
            
            **原理**: 基于预训练 UNet 深度学习模型，自动识别水体
            
            **优势**:
            - 🎯 **无需调阈值**: AI 自动判断水体/非水体
            - 🌊 **精度更高**: 可区分水体与阴影/暗表面
            - 🔬 **全球泛化**: 基于多源水体标注数据训练
            - 📐 **输出多样**: 掩膜 GeoTIFF + 矢量多边形
            
            **所需波段**: R, G, B, NIR (4波段)
            """)

        with col2:
            st.markdown("""
            ### 📊 指数阈值法 (传统)
            
            **原理**: 遥感指数计算 + 手动阈值分割
            
            **特点**:
            - ⚡ **计算快速**: numpy 直接运算，秒级出结果
            - 🔧 **可控性强**: 可手动调整阈值
            - 📚 **文献支持**: 经典方法，解释性强
            - ⚠️ **阈值敏感**: 不同区域可能需要不同阈值
            
            **所需波段**: Green, NIR, SWIR1 (MNDWI) / +SWIR2 (AWEIsh)
            """)

        st.info("""
        **操作步骤**:
        1. 在「数据浏览」页面下载全波段 GeoTIFF (至少含 R,G,B,NIR)
        2. 在本页面上传 GeoTIFF 文件
        3. 在左侧选择「🤖 AI 智能分割」
        4. 指定 R, G, B, NIR 波段映射
        5. 点击「运行 AI 水体分割」→ 等待推理 → 查看结果 → 导出
        """)

    elif is_onnx_mode:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### ⚡ ONNX 推理加速
            
            **原理**: 将 PyTorch 模型导出为 ONNX 格式，使用 ONNX Runtime 独立推理发动机
            
            **优势**:
            - 🚀 **速度更快**: 推理速度比原生 PyTorch 快 2-5x
            - 📦 **轻量依赖**: 仅需 onnxruntime + numpy + rasterio
            - 🔌 **独立运行**: 无需 geoai-py 和完整 PyTorch
            - 🎯 **精度不变**: ONNX 导出保持模型精度
            
            **所需波段**: R, G, B, NIR (4波段)
            """)

        with col2:
            st.markdown("""
            ### 🤖 AI 智能分割 (对比)
            
            **原理**: geoai-py + PyTorch 原生推理
            
            **特点**:
            - 🔧 **功能完整**: 内置 OSM 过滤、后处理
            - 📚 **集成度高**: 与 geoai-py 生态无缝
            - ⚠️ **依赖较重**: 需要完整 PyTorch + geoai-py
            - 🐢 **速度较慢**: 原生 PyTorch 推理较 ONNX 慢
            
            **所需波段**: R, G, B, NIR (4波段)
            """)

        st.info("""
        **操作步骤**:
        1. 准备好 ONNX 水体分割模型 (可用 geoai-py 训练后导出)
        2. 在「数据浏览」页面下载全波段 GeoTIFF
        3. 在本页面上传 GeoTIFF 文件
        4. 在左侧选择「⚡ ONNX 推理加速」
        5. 上传 ONNX 模型或指定模型路径
        6. 点击「运行 ONNX 水体分割」→ 查看推理速度 → 导出
        """)

        st.warning("""
        ⚠️ **前置准备**:
        - 安装 ONNX Runtime: `pip install onnxruntime` (CPU) 或 `pip install onnxruntime-gpu` (GPU)
        - 获取水体分割 ONNX 模型: 可用 `utils/onnx_engine.export_to_onnx()` 从 PyTorch 导出
        - 模型默认路径: `models/water_seg_unet_resnet34.onnx`
        """)

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### MNDWI 修正归一化水体指数
            
            $$
            MNDWI = \\frac{Green - SWIR1}{Green + SWIR1}
            $$
            
            - **MNDWI > 0** → 水体
            - **MNDWI < 0** → 非水体 (裸地、植被)
            - 比 NDWI 更能区分建成区
            
            **所需波段**: Sentinel-2 B3(绿) + B11(SWIR1)
            """)
        with col2:
            st.markdown("""
            ### AWEIsh 自动水体提取指数
            
            $$
            AWEI_{sh} = 4 \\times (Green - SWIR2) - 0.25 \\times NIR + 2.75 \\times SWIR1
            $$
            
            - **AWEIsh > 0** → 水体
            - 专为阴影和暗表面设计
            - 能有效区分水体与地形阴影
            
            **所需波段**: Green, NIR, SWIR1, SWIR2
            """)

        st.info("""
        **操作步骤**:
        1. 在「数据浏览」页面下载全波段 GeoTIFF
        2. 在本页面上传 GeoTIFF 文件
        3. 指定波段映射 (Blue, Green, Red, NIR, SWIR1, SWIR2)
        4. 选择水体指数 → 点击计算 → 查看结果 → 导出
        """)
