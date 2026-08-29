"""
自定义光谱指数计算器页面 — Band Math
======================================
支持 12 种预设指数 + 自定义波段运算表达式
波段: B, G, R, NIR, SWIR1, SWIR2
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.pc_data import search_images, get_rgb_preview_cached, download_multiband
from utils.spectral import (
    BAND_NAMES, PRESET_INDICES, PRESET_INDEX_NAMES,
    get_band_arrays, evaluate_band_math,
    get_preset_index, compute_index_stats,
)

st.set_page_config(page_title="光谱指数计算器", page_icon="🧮", layout="wide")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🧮 指数计算器设置")

    st.subheader("研究区")
    default_area = st.session_state.get("selected_area", "塔里木盆地")
    if default_area not in STUDY_AREAS:
        default_area = "塔里木盆地"
    area_name = st.selectbox(
        "选择研究区",
        list(STUDY_AREAS.keys()),
        index=list(STUDY_AREAS.keys()).index(default_area),
    )
    area_info = STUDY_AREAS[area_name]
    st.session_state["selected_area"] = area_name
    st.session_state["selected_bbox"] = area_info["bbox"]
    st.caption(f"📌 {area_info['description']}")

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据",
        list(COLLECTIONS.keys()),
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )
    st.caption(COLLECTIONS[satellite]["description"])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=90))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 15)
    max_items = st.slider("最大影像数", 1, 15, 5)

    search_clicked = st.button(
        "🔍 搜索影像", type="primary", use_container_width=True
    )

# ============================================================
# 主页面
# ============================================================
st.title("🧮 自定义光谱指数计算器")
st.markdown(
    f"**研究区: {area_name}** | "
    f"数据源: {satellite} | "
    f"日期: {start_date} → {end_date}"
)
st.markdown(
    "选择预设指数或输入自定义波段运算表达式，实时计算并可视化。"
    "波段名: `B` `G` `R` `NIR` `SWIR1` `SWIR2`，支持 `sqrt/exp/abs/clip/where` 等 numpy 函数。"
)

if search_clicked:
    bbox = area_info["bbox"]

    # ---- Step 1: 搜索影像 ----
    with st.spinner("🔍 正在搜索影像..."):
        try:
            results = search_images(
                bbox=bbox,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                collection=satellite,
                cloud_cover_max=cloud_cover,
                max_items=max_items,
            )
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

    if not results:
        st.warning("⚠️ 未找到符合条件的影像。降低云量阈值试试？")
        st.stop()

    st.success(f"✅ 找到 **{len(results)}** 景影像")

    # ---- Step 2: 影像选择 ----
    st.subheader("📋 选择分析影像")
    cols = st.columns(3)
    selected_items = []
    for i, r in enumerate(results):
        with cols[i % 3]:
            try:
                preview = get_rgb_preview_cached(r["id"], collection=satellite, width=256)
            except Exception:
                preview = None
            if preview:
                st.image(preview)
            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_idx_{r['id']}",
                value=i == 0,
            )
            if selected:
                selected_items.append(r)

    if not selected_items:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    # ---- Step 3: 下载波段 ----
    item = selected_items[0]
    st.info(f"📊 正在分析: **{item['datetime']}**")

    with st.spinner("⬇️ 下载 6 波段数据..."):
        with StreamlitErrorBoundary("波段下载", st=st, show_traceback=False):
            tmp_path = os.path.join(tempfile.gettempdir(), f"idx_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)
            import rasterio
            with rasterio.open(tif_path) as src:
                bands_data = src.read().astype(np.float32)
            if os.path.exists(tif_path):
                os.remove(tif_path)

    bands_dict = get_band_arrays(bands_data)

    # ---- Step 4: 选择指数 ----
    st.divider()
    st.subheader("⚙️ 指数设置")

    mode = st.radio(
        "计算模式",
        ["预设指数", "自定义公式"],
        horizontal=True,
    )

    if mode == "预设指数":
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            preset_name = st.selectbox("选择指数", PRESET_INDEX_NAMES)
        preset = get_preset_index(preset_name)
        with col_info:
            st.info(f"**{preset['name']}**: {preset['desc']}\n\n公式: `{preset['formula']}`")
        formula = preset["formula"]
        cmap = preset["cmap"]
        vmin, vmax = preset["vmin"], preset["vmax"]
        display_name = preset["name"]
    else:
        st.markdown("**波段名**: `B`(蓝) `G`(绿) `R`(红) `NIR`(近红外) `SWIR1`(短波红外1) `SWIR2`(短波红外2)")
        formula = st.text_input(
            "输入波段运算表达式",
            value="(NIR - R) / (NIR + R)",
            help="支持 + - * / ** 和 sqrt/exp/abs/clip/where/log 等 numpy 函数",
        )
        cmap = st.selectbox("色带", ["RdYlGn", "viridis", "RdYlBu", "YlOrRd", "Blues", "magma", "coolwarm"])
        vmin = st.number_input("色带最小值 (vmin)", value=-1.0, step=0.1)
        vmax = st.number_input("色带最大值 (vmax)", value=1.0, step=0.1)
        display_name = "自定义指数"

    if st.button("🧮 计算指数", type="primary"):
        with st.spinner("计算中..."):
            with StreamlitErrorBoundary("指数计算", st=st, show_traceback=True):
                index_array = evaluate_band_math(formula, bands_dict)
                stats = compute_index_stats(index_array)

        # ---- 结果展示 ----
        st.divider()
        st.subheader(f"📊 结果 — {display_name}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("均值", stats["mean"])
        with col2:
            st.metric("标准差", stats["std"])
        with col3:
            st.metric("最小值", stats["min"])
        with col4:
            st.metric("最大值", stats["max"])

        st.caption(f"公式: `{formula}` | 有效像元占比: {stats['valid_ratio']*100:.1f}%")

        col_map, col_hist = st.columns([3, 2])
        with col_map:
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{display_name} — {area_name} ({item['datetime']})", fontsize=13)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

        with col_hist:
            fig, ax = plt.subplots(figsize=(6, 6))
            valid = index_array[np.isfinite(index_array)]
            if valid.size > 0:
                ax.hist(valid, bins=50, color="#1f77b4", alpha=0.8)
                ax.set_xlabel("指数值")
                ax.set_ylabel("像元数")
                ax.set_title(f"{display_name} 直方图", fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("⚠️ 计算结果全为无效值，请检查公式。")

        # 导出提示
        st.divider()
        st.info(
            "💡 提示: 如需导出指数结果为 GeoTIFF，可使用「报告导出」模块，"
            "或在下个版本中使用数据导出功能。"
        )
