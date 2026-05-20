"""
变化检测页面 — 双时相遥感影像对比分析
v2.0: STAC自动下载 + 7级变化分类 + RGB叠加 + 土地覆盖交叉表
支持差值法/比值法，可视化变化区域，面积统计与导出
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLORMAPS, INDEX_THRESHOLDS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary

st.set_page_config(page_title="变化检测", page_icon="🔄", layout="wide")

# ============================================
# Session State 初始化
# ============================================
SESSION_KEYS = [
    "cd_t1_results", "cd_t2_results",
    "cd_t1_selected_item", "cd_t2_selected_item",
    "cd_t1_path", "cd_t2_path",
    "cd_t1_date", "cd_t2_date",
    "cd_analysis_done", "cd_change_class",
    "cd_index_t1", "cd_index_t2",
    "cd_rgb_t1", "cd_rgb_t2",
    "cd_stats", "cd_level_counts",
    "cd_landcover_cross",
]
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("🔄 双时相变化检测")
st.markdown("自动搜索两期卫星影像或手动上传，计算指数差异并识别多级变化区域")

# ============================================
# 侧边栏 - 模式 + 参数
# ============================================
with st.sidebar:
    st.header("⚙️ 配置")

    # --- 模式选择 ---
    st.subheader("📡 数据源模式")
    data_mode = st.radio(
        "选择数据获取方式",
        ["🛰️ STAC 自动下载", "📁 手动上传"],
        help="STAC自动下载：从Planetary Computer搜索并下载影像\n手动上传：自行上传GeoTIFF文件",
    )

    # --- 指数选择 ---
    st.subheader("分析指数")
    index_choice = st.selectbox(
        "选择指数",
        ["NDVI (植被变化)", "MNDWI (水体变化)", "EVI (增强植被)"],
        help="不同指数适用于不同类型的变化检测",
    )

    INDEX_CONFIG = {
        "NDVI (植被变化)": {"key": "ndvi", "label": "NDVI"},
        "MNDWI (水体变化)": {"key": "mndwi", "label": "MNDWI"},
        "EVI (增强植被)": {"key": "evi", "label": "EVI"},
    }
    idx_cfg = INDEX_CONFIG[index_choice]

    # --- 检测方法 ---
    st.subheader("检测方法")
    method = st.radio(
        "变化检测方法",
        ["差值法 (Difference)", "比值法 (Ratio)"],
        help="差值法：Index_T2 - Index_T1\n比值法：Index_T2 / Index_T1",
    )

    # --- 阈值 ---
    st.subheader("变化阈值")
    threshold = st.slider(
        "基础阈值 (区分变化/不变)",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="|差值| > 阈值 则判定为变化区域",
    )

    # --- 像素面积 ---
    st.subheader("像素参数")
    pixel_size = st.number_input(
        "单像素边长 (m)",
        min_value=1.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        help="Sentinel-2: 10m, Landsat: 30m",
    )

    # --- 文献参考 ---
    st.divider()
    with st.expander("📖 多级变化阈值参考"):
        st.markdown("""
        **7级变化分类 (差值法)**:
        | 级别 | 条件 | 含义 |
        |------|------|------|
        | -3 | Δ < -0.30 | 严重减少 |
        | -2 | -0.30≤Δ<-0.15 | 中度减少 |
        | -1 | -0.15≤Δ<-阈值 | 轻度减少 |
        | 0 | |Δ| ≤ 阈值 | 稳定 |
        | +1 | 阈值<Δ≤0.15 | 轻度增加 |
        | +2 | 0.15<Δ≤0.30 | 中度增加 |
        | +3 | Δ > 0.30 | 严重增加 |
        """)

# ============================================
# 7级变化分类工具函数
# ============================================
def classify_multilevel(diff: np.ndarray, valid_mask: np.ndarray, thresh: float) -> np.ndarray:
    """
    将差值数组分为7级

    返回: int8 数组, -3 ~ +3
    """
    change_class = np.zeros(diff.shape, dtype=np.int8)

    # 严重减少: Δ < -0.30
    change_class[(diff < -0.30) & valid_mask] = -3
    # 中度减少: -0.30 ≤ Δ < -0.15
    change_class[(diff >= -0.30) & (diff < -0.15) & valid_mask] = -2
    # 轻度减少: -0.15 ≤ Δ < -threshold
    change_class[(diff >= -0.15) & (diff < -thresh) & valid_mask] = -1
    # 稳定: |Δ| ≤ threshold
    change_class[(np.abs(diff) <= thresh) & valid_mask] = 0
    # 轻度增加: threshold < Δ ≤ 0.15
    change_class[(diff > thresh) & (diff <= 0.15) & valid_mask] = 1
    # 中度增加: 0.15 < Δ ≤ 0.30
    change_class[(diff > 0.15) & (diff <= 0.30) & valid_mask] = 2
    # 严重增加: Δ > 0.30
    change_class[(diff > 0.30) & valid_mask] = 3

    return change_class


def count_levels(change_class: np.ndarray, valid_mask: np.ndarray) -> dict:
    """统计各变化级别的像素数"""
    counts = {}
    for level in range(-3, 4):
        counts[level] = int(np.sum((change_class == level) & valid_mask))
    return counts


# ============================================
# STAC 模式 — 研究区 + 双时相搜索
# ============================================
if data_mode == "🛰️ STAC 自动下载":
    st.sidebar.divider()
    st.sidebar.subheader("📍 研究区")

    area_names = list(STUDY_AREAS.keys())
    selected_area = st.sidebar.selectbox("选择研究区", area_names)
    bbox = STUDY_AREAS[selected_area]["bbox"]
    st.sidebar.caption(f"BBOX: {bbox}")

    st.sidebar.subheader("🛰️ 卫星源")
    satellite = st.sidebar.selectbox("卫星", list(COLLECTIONS.keys()))

    # --- T1 时相 ---
    st.sidebar.divider()
    st.sidebar.subheader("📅 时相 1 (T1 — 早期)")

    col_t1a, col_t1b = st.sidebar.columns(2)
    with col_t1a:
        t1_start = st.date_input("开始", value=datetime(2020, 6, 1), key="t1_start")
    with col_t1b:
        t1_end = st.date_input("结束", value=datetime(2020, 9, 30), key="t1_end")

    t1_cloud = st.sidebar.slider("最大云量 (%)", 0, 100, 20, key="t1_cloud")
    t1_max = st.sidebar.slider("最大影像数", 1, 20, 5, key="t1_max")

    if st.sidebar.button("🔍 搜索 T1 影像", use_container_width=True):
        with st.spinner("正在搜索 T1 影像..."):
            with StreamlitErrorBoundary("T1 影像搜索", st=st, show_traceback=False):
                from utils.pc_data import search_images
                results = search_images(
                    bbox=bbox,
                    start_date=t1_start.strftime("%Y-%m-%d"),
                    end_date=t1_end.strftime("%Y-%m-%d"),
                    collection=satellite,
                    cloud_cover_max=t1_cloud,
                    max_items=t1_max,
                )
                st.session_state["cd_t1_results"] = results
                st.session_state["cd_t1_selected_item"] = None
                if results:
                    st.sidebar.success(f"✅ 找到 {len(results)} 景 T1 影像")
                else:
                    st.sidebar.warning("⚠️ 未找到 T1 影像")

    # --- T2 时相 ---
    st.sidebar.divider()
    st.sidebar.subheader("📅 时相 2 (T2 — 后期)")

    col_t2a, col_t2b = st.sidebar.columns(2)
    with col_t2a:
        t2_start = st.date_input("开始", value=datetime(2024, 6, 1), key="t2_start")
    with col_t2b:
        t2_end = st.date_input("结束", value=datetime(2024, 9, 30), key="t2_end")

    t2_cloud = st.sidebar.slider("最大云量 (%)", 0, 100, 20, key="t2_cloud")
    t2_max = st.sidebar.slider("最大影像数", 1, 20, 5, key="t2_max")

    if st.sidebar.button("🔍 搜索 T2 影像", use_container_width=True):
        with st.spinner("正在搜索 T2 影像..."):
            with StreamlitErrorBoundary("T2 影像搜索", st=st, show_traceback=False):
                from utils.pc_data import search_images
                results = search_images(
                    bbox=bbox,
                    start_date=t2_start.strftime("%Y-%m-%d"),
                    end_date=t2_end.strftime("%Y-%m-%d"),
                    collection=satellite,
                    cloud_cover_max=t2_cloud,
                    max_items=t2_max,
                )
                st.session_state["cd_t2_results"] = results
                st.session_state["cd_t2_selected_item"] = None
                if results:
                    st.sidebar.success(f"✅ 找到 {len(results)} 景 T2 影像")
                else:
                    st.sidebar.warning("⚠️ 未找到 T2 影像")

    # ============================================
    # 主面板 — STAC 搜索结果显示
    # ============================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 T1 搜索结果")
        t1_results = st.session_state.get("cd_t1_results")
        if t1_results:
            # 表格
            t1_df = pd.DataFrame([
                {
                    "日期": r["datetime"],
                    "云量(%)": r["cloud_cover"],
                    "影像ID": r["id"][:30],
                }
                for r in t1_results
            ])
            st.dataframe(t1_df, use_container_width=True, hide_index=True)

            # 单选
            t1_options = [f"{r['datetime']} | 云量 {r['cloud_cover']}% | {r['id'][:20]}" for r in t1_results]
            t1_choice = st.radio("选择 T1 影像", t1_options, key="t1_radio")
            if t1_choice:
                idx = t1_options.index(t1_choice)
                st.session_state["cd_t1_selected_item"] = t1_results[idx]

                # 预览
                with st.spinner("加载 T1 预览..."):
                    try:
                        from utils.pc_data import get_rgb_preview
                        preview = get_rgb_preview(
                            t1_results[idx]["item"],
                            collection=satellite,
                            width=400,
                        )
                        if preview:
                            st.image(preview, caption=f"T1: {t1_results[idx]['datetime']}", use_container_width=True)
                    except Exception:
                        pass
        else:
            st.info("👈 请在侧边栏搜索 T1 影像")

    with col2:
        st.subheader("📅 T2 搜索结果")
        t2_results = st.session_state.get("cd_t2_results")
        if t2_results:
            t2_df = pd.DataFrame([
                {
                    "日期": r["datetime"],
                    "云量(%)": r["cloud_cover"],
                    "影像ID": r["id"][:30],
                }
                for r in t2_results
            ])
            st.dataframe(t2_df, use_container_width=True, hide_index=True)

            t2_options = [f"{r['datetime']} | 云量 {r['cloud_cover']}% | {r['id'][:20]}" for r in t2_results]
            t2_choice = st.radio("选择 T2 影像", t2_options, key="t2_radio")
            if t2_choice:
                idx = t2_options.index(t2_choice)
                st.session_state["cd_t2_selected_item"] = t2_results[idx]

                with st.spinner("加载 T2 预览..."):
                    try:
                        from utils.pc_data import get_rgb_preview
                        preview = get_rgb_preview(
                            t2_results[idx]["item"],
                            collection=satellite,
                            width=400,
                        )
                        if preview:
                            st.image(preview, caption=f"T2: {t2_results[idx]['datetime']}", use_container_width=True)
                    except Exception:
                        pass
        else:
            st.info("👈 请在侧边栏搜索 T2 影像")

    # ============================================
    # STAC 模式 — 分析按钮
    # ============================================
    t1_sel = st.session_state.get("cd_t1_selected_item")
    t2_sel = st.session_state.get("cd_t2_selected_item")

    st.divider()

    can_analyze_stac = t1_sel is not None and t2_sel is not None

    if can_analyze_stac:
        st.success(f"✅ 已选择 T1 ({t1_sel['datetime']}) 和 T2 ({t2_sel['datetime']})，点击下方按钮开始分析")
    else:
        st.info("👆 请分别在左右两侧选择 T1 和 T2 影像")

    # --- 土地覆盖交叉表选项 ---
    st.sidebar.divider()
    enable_landcover = st.sidebar.checkbox(
        "🔀 土地覆盖交叉分析",
        value=False,
        help="分析变化区域在不同土地覆盖类型上的分布 (需额外下载ESA数据)",
    )

    if st.button("🔍 执行变化检测分析", type="primary", use_container_width=True, disabled=not can_analyze_stac):
        with StreamlitErrorBoundary("STAC变化检测分析", st=st, show_traceback=True):
            with st.spinner("正在下载影像并计算变化..."):
                from utils.pc_data import download_multiband, get_rgb_preview
                from utils.indices import load_bands_from_geotiff, calc_ndvi, calc_mndwi, calc_evi

                # --- 下载 T1 和 T2 ---
                t1_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                t1_path = t1_tmp.name
                t1_tmp.close()

                t2_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                t2_path = t2_tmp.name
                t2_tmp.close()

                st.info("⏳ 下载 T1 影像 (6波段, 可能需要1-2分钟)...")
                download_multiband(t1_sel["item"], t1_path, collection=satellite)
                st.info("⏳ 下载 T2 影像 (6波段, 可能需要1-2分钟)...")
                download_multiband(t2_sel["item"], t2_path, collection=satellite)

                st.session_state["cd_t1_path"] = t1_path
                st.session_state["cd_t2_path"] = t2_path
                st.session_state["cd_t1_date"] = t1_sel["datetime"]
                st.session_state["cd_t2_date"] = t2_sel["datetime"]

                # RGB 预览
                try:
                    t1_rgb = get_rgb_preview(t1_sel["item"], collection=satellite, width=512)
                    t2_rgb = get_rgb_preview(t2_sel["item"], collection=satellite, width=512)
                    st.session_state["cd_rgb_t1"] = t1_rgb
                    st.session_state["cd_rgb_t2"] = t2_rgb
                except Exception:
                    pass

                # --- 执行变化检测 ---
                _run_change_analysis(t1_path, t2_path, idx_cfg, method, threshold, pixel_size)

                # --- 土地覆盖交叉表 ---
                if enable_landcover:
                    with st.spinner("正在分析土地覆盖变化..."):
                        try:
                            _run_landcover_cross(st.session_state["cd_change_class"], bbox)
                        except Exception as e:
                            st.warning(f"土地覆盖分析失败: {e}")

                st.session_state["cd_analysis_done"] = True
                st.rerun()

# ============================================
# 手动上传模式
# ============================================
else:
    col1, col2 = st.columns(2)
    t1_path = None
    t2_path = None

    with col1:
        st.subheader("📅 时相 1 (T1 — 早期)")
        t1_file = st.file_uploader("上传 GeoTIFF", type=["tif", "tiff"], key="t1_manual")
        if t1_file:
            with StreamlitErrorBoundary("T1 文件上传", st=st, show_traceback=False):
                t1_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                t1_tmp.write(t1_file.read())
                t1_path = t1_tmp.name
                t1_tmp.close()
                st.session_state["cd_t1_path"] = t1_path
                st.success(f"✅ {t1_file.name}")
                st.caption(f"大小: {t1_file.size / 1024 / 1024:.1f} MB")

    with col2:
        st.subheader("📅 时相 2 (T2 — 后期)")
        t2_file = st.file_uploader("上传 GeoTIFF", type=["tif", "tiff"], key="t2_manual")
        if t2_file:
            with StreamlitErrorBoundary("T2 文件上传", st=st, show_traceback=False):
                t2_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                t2_tmp.write(t2_file.read())
                t2_path = t2_tmp.name
                t2_tmp.close()
                st.session_state["cd_t2_path"] = t2_path
                st.success(f"✅ {t2_file.name}")
                st.caption(f"大小: {t2_file.size / 1024 / 1024:.1f} MB")

    # --- 波段映射 ---
    both_loaded = t1_path is not None and t2_path is not None
    if both_loaded:
        with st.expander("🔧 波段映射配置 (如自动识别错误请手动调整)", expanded=False):
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            with col_b1:
                band_blue = st.number_input("蓝波段索引", 0, 20, 0)
            with col_b2:
                band_green = st.number_input("绿波段索引", 0, 20, 1)
            with col_b3:
                band_red = st.number_input("红波段索引", 0, 20, 2)
            with col_b4:
                band_nir = st.number_input("近红外索引", 0, 20, 3)
            with col_b5:
                band_swir1 = st.number_input("短波红外1索引", 0, 20, 4)

        st.divider()

        if st.button("🔍 执行变化检测分析", type="primary", use_container_width=True):
            with StreamlitErrorBoundary("手动上传变化检测分析", st=st, show_traceback=True):
                with st.spinner("正在计算两期指数并检测变化..."):
                    from utils.indices import load_bands_from_geotiff, calc_ndvi, calc_mndwi, calc_evi
                    import rasterio

                    # 确定需要加载的波段索引
                    needed_indices = []
                    if idx_cfg["key"] in ("ndvi", "evi"):
                        needed_indices.extend([band_red, band_nir])
                    if idx_cfg["key"] in ("evi",):
                        needed_indices.append(band_blue)
                    if idx_cfg["key"] in ("ndvi",):
                        needed_indices.append(band_blue)
                    if idx_cfg["key"] == "mndwi":
                        needed_indices.extend([band_green, band_swir1])
                    needed_indices = list(set(needed_indices))

                    bands_t1 = load_bands_from_geotiff(t1_path, band_indices=needed_indices)
                    bands_t2 = load_bands_from_geotiff(t2_path, band_indices=needed_indices)

                    def get_band(bands_dict, target_idx):
                        for key, arr in bands_dict.items():
                            if key == target_idx:
                                return arr
                        return None

                    # 计算指数
                    if idx_cfg["key"] == "ndvi":
                        red1 = get_band(bands_t1, band_red)
                        nir1 = get_band(bands_t1, band_nir)
                        red2 = get_band(bands_t2, band_red)
                        nir2 = get_band(bands_t2, band_nir)
                        if any(x is None for x in [red1, nir1, red2, nir2]):
                            st.error("❌ 无法提取所需波段 (Red + NIR)，请检查波段映射")
                            st.stop()
                        index_t1 = calc_ndvi(red1, nir1)
                        index_t2 = calc_ndvi(red2, nir2)

                    elif idx_cfg["key"] == "mndwi":
                        green1 = get_band(bands_t1, band_green)
                        swir1_1 = get_band(bands_t1, band_swir1)
                        green2 = get_band(bands_t2, band_green)
                        swir1_2 = get_band(bands_t2, band_swir1)
                        if any(x is None for x in [green1, swir1_1, green2, swir1_2]):
                            st.error("❌ 无法提取所需波段 (Green + SWIR1)，请检查波段映射")
                            st.stop()
                        index_t1 = calc_mndwi(green1, swir1_1)
                        index_t2 = calc_mndwi(green2, swir1_2)

                    elif idx_cfg["key"] == "evi":
                        blue1 = get_band(bands_t1, band_blue)
                        red1 = get_band(bands_t1, band_red)
                        nir1 = get_band(bands_t1, band_nir)
                        blue2 = get_band(bands_t2, band_blue)
                        red2 = get_band(bands_t2, band_red)
                        nir2 = get_band(bands_t2, band_nir)
                        if any(x is None for x in [blue1, red1, nir1, blue2, red2, nir2]):
                            st.error("❌ 无法提取所需波段 (Blue + Red + NIR)，请检查波段映射")
                            st.stop()
                        index_t1 = calc_evi(blue1, red1, nir1)
                        index_t2 = calc_evi(blue2, red2, nir2)

                    # 空间对齐: 裁剪到公共区域
                    h_min = min(index_t1.shape[0], index_t2.shape[0])
                    w_min = min(index_t1.shape[1], index_t2.shape[1])
                    index_t1 = index_t1[:h_min, :w_min]
                    index_t2 = index_t2[:h_min, :w_min]

                    valid_mask = np.isfinite(index_t1) & np.isfinite(index_t2)
                    total_pixels = max(np.sum(valid_mask), 1)

                    # 变化计算
                    if method == "差值法 (Difference)":
                        change_raw = index_t2 - index_t1
                    else:
                        safe_t1 = np.where(index_t1 == 0, 1e-6, index_t1)
                        change_raw = index_t2 / safe_t1

                    # 7级分类
                    change_class = classify_multilevel(change_raw, valid_mask, threshold)

                    # 统计
                    level_counts = count_levels(change_class, valid_mask)
                    pixel_area_km2 = (pixel_size * pixel_size) / 1_000_000

                    idx_t1_mean = np.nanmean(index_t1[valid_mask])
                    idx_t2_mean = np.nanmean(index_t2[valid_mask])

                    # 存储
                    st.session_state["cd_change_class"] = change_class
                    st.session_state["cd_index_t1"] = index_t1
                    st.session_state["cd_index_t2"] = index_t2
                    st.session_state["cd_level_counts"] = level_counts
                    st.session_state["cd_stats"] = {
                        "total_pixels": total_pixels,
                        "pixel_area_km2": pixel_area_km2,
                        "idx_t1_mean": idx_t1_mean,
                        "idx_t2_mean": idx_t2_mean,
                        "method": method,
                        "threshold": threshold,
                        "pixel_size": pixel_size,
                    }
                    st.session_state["cd_analysis_done"] = True
                    st.rerun()


# ============================================
# 共享分析函数 (STAC模式调用)
# ============================================
def _run_change_analysis(t1_path, t2_path, idx_cfg, method, threshold, pixel_size):
    """执行变化检测核心分析，结果存入 session_state"""
    from utils.indices import load_bands_from_geotiff, calc_ndvi, calc_mndwi, calc_evi
    import rasterio
    from rasterio.warp import reproject, Resampling

    # 加载所有6波段 (STAC下载的是标准6波段 multiband)
    bands_t1 = load_bands_from_geotiff(t1_path)
    bands_t2 = load_bands_from_geotiff(t2_path)

    b_t1 = bands_t1["bands"]
    b_t2 = bands_t2["bands"]

    # multiband下载: band 1=blue, 2=green, 3=red, 4=nir, 5=swir1, 6=swir2
    # load_bands_from_geotiff 默认 1-based, 传入None则用标准范围
    # 直接按键获取
    blue1, green1, red1, nir1, swir1_1 = b_t1[1], b_t1[2], b_t1[3], b_t1[4], b_t1[5]
    blue2, green2, red2, nir2, swir1_2 = b_t2[1], b_t2[2], b_t2[3], b_t2[4], b_t2[5]

    # 计算指数
    if idx_cfg["key"] == "ndvi":
        index_t1 = calc_ndvi(red1, nir1)
        index_t2 = calc_ndvi(red2, nir2)
    elif idx_cfg["key"] == "mndwi":
        index_t1 = calc_mndwi(green1, swir1_1)
        index_t2 = calc_mndwi(green2, swir1_2)
    elif idx_cfg["key"] == "evi":
        index_t1 = calc_evi(blue1, red1, nir1)
        index_t2 = calc_evi(blue2, red2, nir2)

    # 空间对齐
    h_min = min(index_t1.shape[0], index_t2.shape[0])
    w_min = min(index_t1.shape[1], index_t2.shape[1])
    index_t1 = index_t1[:h_min, :w_min]
    index_t2 = index_t2[:h_min, :w_min]

    valid_mask = np.isfinite(index_t1) & np.isfinite(index_t2)
    total_pixels = max(np.sum(valid_mask), 1)

    # 变化计算
    if method == "差值法 (Difference)":
        change_raw = index_t2 - index_t1
    else:
        safe_t1 = np.where(index_t1 == 0, 1e-6, index_t1)
        change_raw = index_t2 / safe_t1

    # 7级分类
    change_class = classify_multilevel(change_raw, valid_mask, threshold)

    level_counts = count_levels(change_class, valid_mask)
    pixel_area_km2 = (pixel_size * pixel_size) / 1_000_000

    idx_t1_mean = np.nanmean(index_t1[valid_mask])
    idx_t2_mean = np.nanmean(index_t2[valid_mask])

    st.session_state["cd_change_class"] = change_class
    st.session_state["cd_index_t1"] = index_t1
    st.session_state["cd_index_t2"] = index_t2
    st.session_state["cd_level_counts"] = level_counts
    st.session_state["cd_stats"] = {
        "total_pixels": total_pixels,
        "pixel_area_km2": pixel_area_km2,
        "idx_t1_mean": idx_t1_mean,
        "idx_t2_mean": idx_t2_mean,
        "method": method,
        "threshold": threshold,
        "pixel_size": pixel_size,
    }


def _run_landcover_cross(change_class, bbox):
    """土地覆盖交叉分析"""
    try:
        from utils.landcover import download_esa_landcover

        esa_path = download_esa_landcover(bbox)
        if esa_path is None:
            st.session_state["cd_landcover_cross"] = None
            return

        import rasterio
        with rasterio.open(esa_path) as src:
            esa_data = src.read(1)

        # 对齐 ESA 和 change_class
        h_min = min(change_class.shape[0], esa_data.shape[0])
        w_min = min(change_class.shape[1], esa_data.shape[1])
        cc = change_class[:h_min, :w_min]
        lc = esa_data[:h_min, :w_min]

        valid = (lc > 0) & (lc <= 11)  # ESA 11类

        # ESA 类别简并
        ESA_NAMES = {
            10: "森林", 20: "灌木", 30: "草地",
            40: "农田", 50: "建设用地",
            60: "裸地/稀疏植被", 70: "冰雪",
            80: "水体", 90: "湿地", 100: "苔原",
        }

        # 简并到主要类别
        def simplify_lc(val):
            if val in (10, 20): return "森林/灌木"
            if val == 30: return "草地"
            if val == 40: return "农田"
            if val == 50: return "建设用地"
            if val in (60, 70, 90, 100): return "裸地/稀疏植被"
            if val == 80: return "水体"
            return "其他"

        simple_lc = np.array([simplify_lc(v) for v in lc.flatten()]).reshape(lc.shape)

        # 交叉表: 变化级别 × 土地覆盖
        levels = list(range(-3, 4))
        from utils.visualization import MULTILEVEL_LABELS
        lc_types = ["森林/灌木", "草地", "农田", "建设用地", "裸地/稀疏植被", "水体", "其他"]

        cross = {}
        for lv in levels:
            cross[lv] = {}
            for lt in lc_types:
                cross[lv][lt] = int(np.sum((cc == lv) & (simple_lc == lt)))

        st.session_state["cd_landcover_cross"] = {
            "data": cross,
            "levels": levels,
            "lc_types": lc_types,
        }
    except Exception as e:
        print(f"Landcover cross failed: {e}")
        st.session_state["cd_landcover_cross"] = None


# ============================================
# 结果展示 (共享)
# ============================================
if st.session_state.get("cd_analysis_done"):
    change_class = st.session_state["cd_change_class"]
    index_t1 = st.session_state["cd_index_t1"]
    index_t2 = st.session_state["cd_index_t2"]
    level_counts = st.session_state["cd_level_counts"]
    stats = st.session_state["cd_stats"]
    t1_date = st.session_state.get("cd_t1_date", "T1")
    t2_date = st.session_state.get("cd_t2_date", "T2")

    total_pixels = stats["total_pixels"]
    pixel_area_km2 = stats["pixel_area_km2"]

    from utils.visualization import (
        render_index, render_multilevel_change, render_change_overlay,
        plot_multilevel_change_stacked_bar, MULTILEVEL_LABELS, MULTILEVEL_COLORS,
    )
    from utils.indices import save_mask_geotiff
    import plotly.graph_objects as go

    st.divider()

    # --- KPI 卡片 (按7级汇总) ---
    st.subheader("📊 变化概览")

    # 增加总量 (1+2+3), 减少总量 (-1-2-3)
    inc_total = sum(level_counts.get(l, 0) for l in [1, 2, 3])
    dec_total = sum(level_counts.get(l, 0) for l in [-1, -2, -3])
    stable = level_counts.get(0, 0)

    area_inc = inc_total * pixel_area_km2
    area_dec = dec_total * pixel_area_km2
    area_stable = stable * pixel_area_km2
    area_total = total_pixels * pixel_area_km2

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(f"增加总面积", f"{area_inc:.2f} km²", f"{inc_total / max(total_pixels, 1) * 100:.1f}%")
    with kpi2:
        st.metric(f"减少总面积", f"{area_dec:.2f} km²", f"{dec_total / max(total_pixels, 1) * 100:.1f}%")
    with kpi3:
        st.metric("稳定面积", f"{area_stable:.2f} km²", f"{stable / max(total_pixels, 1) * 100:.1f}%")
    with kpi4:
        net = area_inc - area_dec
        st.metric("净变化", f"{net:+.2f} km²")
    with kpi5:
        st.metric("总分析面积", f"{area_total:.2f} km²")

    st.divider()

    # --- 双时相指数对比 ---
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.caption(f"{t1_date} — 均值: {stats['idx_t1_mean']:.4f}")
        fig_t1 = render_index(index_t1, title=f"T1 {idx_cfg['label']}", cmap="RdYlGn", vmin=-1, vmax=1)
        st.image(fig_t1, use_container_width=True)
    with col_v2:
        st.caption(f"{t2_date} — 均值: {stats['idx_t2_mean']:.4f}")
        fig_t2 = render_index(index_t2, title=f"T2 {idx_cfg['label']}", cmap="RdYlGn", vmin=-1, vmax=1)
        st.image(fig_t2, use_container_width=True)

    st.divider()

    # --- 多级变化检测图 ---
    st.subheader("🗺️ 多级变化检测图")
    st.caption("7级颜色编码: 深红=严重减少 → 灰=稳定 → 深绿=严重增加")

    multilevel_fig = render_multilevel_change(
        change_class,
        title=f"{idx_cfg['label']} 多级变化 ({t1_date} → {t2_date})",
        figsize=(12, 9),
    )
    st.image(multilevel_fig, use_container_width=True)

    st.divider()

    # --- RGB + 变化叠加图 (仅STAC模式有RGB预览) ---
    t1_rgb = st.session_state.get("cd_rgb_t1")
    if t1_rgb is not None or data_mode == "📁 手动上传":
        st.subheader("🗺️ RGB + 变化叠加图")
        st.caption("半透明叠加: 绿色系=增加, 红色系=减少, 透明=稳定")

        if t1_rgb is not None:
            # 对齐 RGB 和 change_class 尺寸
            rgb_arr = np.array(t1_rgb.resize(
                (change_class.shape[1], change_class.shape[0])
            ))
            from PIL import Image as PILImage
            rgb_img = PILImage.fromarray(rgb_arr)

            overlay_fig = render_change_overlay(
                rgb_img, change_class,
                title=f"{idx_cfg['label']} 变化叠加 ({t1_date} → {t2_date})",
                alpha=0.5,
                figsize=(12, 9),
            )
            st.image(overlay_fig, use_container_width=True)
        else:
            # 手动模式: 用 T1 指数图做底图
            st.info("手动上传模式: RGB 叠加需要 RGB 影像底图，当前使用指数灰度图代替")
            norm_idx = (index_t1 - np.nanmin(index_t1)) / (np.nanmax(index_t1) - np.nanmin(index_t1) + 1e-8)
            rgb_arr = np.stack([norm_idx, norm_idx, norm_idx], axis=-1)
            rgb_arr = (rgb_arr * 255).astype(np.uint8)
            from PIL import Image as PILImage
            rgb_img = PILImage.fromarray(rgb_arr)

            overlay_fig = render_change_overlay(
                rgb_img, change_class,
                title=f"{idx_cfg['label']} 变化叠加 ({t1_date} → {t2_date})",
                alpha=0.5,
                figsize=(12, 9),
            )
            st.image(overlay_fig, use_container_width=True)

    st.divider()

    # --- 统计图表 ---
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.subheader("📊 7级变化分布")
        bar_fig = plot_multilevel_change_stacked_bar(
            level_counts, total_pixels,
            title=f"{idx_cfg['label']} 多级变化面积占比",
            height=400,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_ch2:
        st.subheader("🥧 变化方向占比")
        labels_simple = ["增加", "减少", "稳定"]
        vals_simple = [inc_total, dec_total, stable]
        colors_simple = ["#2ecc71", "#e74c3c", "#95a5a6"]

        pie_fig = go.Figure(data=[go.Pie(
            labels=labels_simple,
            values=vals_simple,
            marker_colors=colors_simple,
            hole=0.4,
            textinfo="label+percent",
        )])
        pie_fig.update_layout(title="变化方向占比", height=450, template="plotly_white")
        st.plotly_chart(pie_fig, use_container_width=True)

    st.divider()

    # --- 双时相指数直方图 ---
    st.subheader("📈 双时相指数分布对比")
    valid_mask = np.isfinite(index_t1) & np.isfinite(index_t2)
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=index_t1[valid_mask].flatten(), name=f"T1 {idx_cfg['label']}",
        opacity=0.6, marker_color="#3498db", nbinsx=60,
    ))
    hist_fig.add_trace(go.Histogram(
        x=index_t2[valid_mask].flatten(), name=f"T2 {idx_cfg['label']}",
        opacity=0.6, marker_color="#e74c3c", nbinsx=60,
    ))
    hist_fig.update_layout(
        title="双时相指数分布对比",
        barmode="overlay", height=400,
        xaxis_title=f"{idx_cfg['label']} 值", yaxis_title="像素数",
        template="plotly_white",
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.divider()

    # --- 详细统计表 ---
    st.subheader("📋 7级变化详细统计")

    levels = list(range(-3, 4))
    stats_rows = []
    for lv in levels:
        cnt = level_counts.get(lv, 0)
        area = cnt * pixel_area_km2
        pct = cnt / max(total_pixels, 1) * 100
        stats_rows.append({
            "级别": lv,
            "标签": MULTILEVEL_LABELS[lv],
            "像素数": f"{cnt:,}",
            "面积 (km²)": f"{area:.2f}",
            "占比 (%)": f"{pct:.2f}",
        })

    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # --- 土地覆盖交叉表 ---
    lc_cross = st.session_state.get("cd_landcover_cross")
    if lc_cross is not None:
        st.divider()
        st.subheader("🔀 土地覆盖变化交叉分析")

        data = lc_cross["data"]
        lc_types = lc_cross["lc_types"]

        # 构建矩阵: 行=变化级别, 列=土地覆盖
        matrix = np.zeros((7, len(lc_types)))
        for i, lv in enumerate(range(-3, 4)):
            for j, lt in enumerate(lc_types):
                matrix[i, j] = data[lv][lt]

        # Plotly 热力图
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=lc_types,
            y=[MULTILEVEL_LABELS[lv] for lv in range(-3, 4)],
            colorscale="YlOrRd",
            text=matrix.astype(int),
            texttemplate="%{text:,}",
            textfont={"size": 10},
            colorbar_title="像素数",
        ))
        heatmap_fig.update_layout(
            title="变化级别 × 土地覆盖类型交叉矩阵 (像素数)",
            height=400,
            xaxis_title="土地覆盖类型",
            yaxis_title="变化级别",
            template="plotly_white",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # 导出CSV
        cross_csv = BytesIO()
        cross_df = pd.DataFrame(matrix, columns=lc_types, index=[MULTILEVEL_LABELS[lv] for lv in range(-3, 4)])
        cross_df.to_csv(cross_csv, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载土地覆盖交叉表 (CSV)",
            data=cross_csv.getvalue(),
            file_name=f"landcover_cross_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    st.divider()

    # --- 导出 ---
    st.subheader("📥 导出结果")

    # 保存变化分类图
    t1_path = st.session_state.get("cd_t1_path")
    if t1_path and os.path.exists(t1_path):
        exp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        save_mask_geotiff(change_class.astype(np.int16), reference_path=t1_path, output_path=exp_tmp.name)
        exp_tmp.close()

        col_export1, col_export2, col_export3 = st.columns(3)

        with col_export1:
            with open(exp_tmp.name, "rb") as f:
                st.download_button(
                    label="📥 变化分类图 (GeoTIFF)",
                    data=f.read(),
                    file_name=f"change_{idx_cfg['key']}_{datetime.now().strftime('%Y%m%d')}.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )

        with col_export2:
            stats_csv = BytesIO()
            stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 统计报表 (CSV)",
                data=stats_csv.getvalue(),
                file_name=f"change_stats_{idx_cfg['key']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_export3:
            # 多级变化图 PNG
            png_buf = BytesIO()
            multilevel_fig.save(png_buf, format="png")
            st.download_button(
                label="📥 变化检测图 (PNG)",
                data=png_buf.getvalue(),
                file_name=f"change_map_{idx_cfg['key']}_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png",
                use_container_width=True,
            )

        try:
            os.unlink(exp_tmp.name)
        except Exception:
            pass

    # --- 汇总信息 ---
    st.divider()
    st.subheader("📋 分析参数汇总")
    summary_df = pd.DataFrame({
        "参数": [
            "分析指数", "检测方法", "基础阈值",
            "T1 日期", "T2 日期",
            "T1 均值", "T2 均值", "均值变化",
            "单像素面积", "总分析面积",
            "数据模式",
        ],
        "值": [
            index_choice,
            method,
            str(threshold),
            str(t1_date),
            str(t2_date),
            f"{stats['idx_t1_mean']:.4f}",
            f"{stats['idx_t2_mean']:.4f}",
            f"{stats['idx_t2_mean'] - stats['idx_t1_mean']:+.4f}",
            f"{pixel_size} m",
            f"{area_total:.2f} km²",
            data_mode,
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================
# 底部说明
# ============================================
st.divider()
st.caption("""
**提示**:
- **STAC模式**: 自动从 Planetary Computer 搜索并下载 Sentinel-2/Landsat 影像，无需手动处理数据
- **手动模式**: 上传两期 GeoTIFF 影像文件，适用于自定义数据
- 两期影像应为同一区域、同一卫星源（确保空间分辨率一致）
- 7级变化分类基于文献阈值: 0.15为轻度/中度分界, 0.30为中度/重度分界
- 土地覆盖交叉分析使用 ESA WorldCover 10m 数据，可在侧边栏启用
""")
