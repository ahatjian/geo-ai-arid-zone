"""
数据浏览页面 — STAC 搜索 + 影像预览 + 数据下载
Planetary Computer 作为唯一数据源
"""

import streamlit as st
import os
import sys
import tempfile
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary, safe_execute
from utils.aoi import render_aoi_selector

st.set_page_config(page_title="数据浏览", page_icon="🗺️", layout="wide")

# ============================================
# 侧边栏 - 搜索设置
# ============================================
with st.sidebar:
    st.title("🔍 影像搜索")

    # 研究区 / AOI 选择 (预设 + 自定义 bbox + GeoJSON 上传)
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="柴达木盆地",
        help_text="选择预设研究区或自定义 AOI (手动 bbox / GeoJSON)",
        key_prefix="data",
    )
    if bbox is None:
        st.stop()
    bbox_custom = bbox

    st.divider()

    # 卫星选择
    satellite = st.selectbox(
        "卫星数据源",
        list(COLLECTIONS.keys()),
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )
    sat_info = COLLECTIONS[satellite]
    st.caption(sat_info["description"])

    # 日期范围
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("开始日期", date.today() - timedelta(days=365))
    with col2:
        end = st.date_input("结束日期", date.today())

    # 搜索参数
    cloud_cover = st.slider("最大云量 (%)", 0, 100, 20)
    max_items = st.slider("最大结果数", 1, 20, 5)

    # 搜索按钮
    search_clicked = st.button("🔍 搜索影像", type="primary")

    st.divider()

    # 本地文件上传
    st.subheader("📁 本地文件")
    uploaded_file = st.file_uploader("上传 GeoTIFF", type=["tif", "tiff"])
    if uploaded_file:
        tmp_dir = tempfile.gettempdir()
        local_tif = os.path.join(tmp_dir, os.path.basename(uploaded_file.name))
        with open(local_tif, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state["local_tif"] = local_tif
        st.success(f"✅ 已加载: {uploaded_file.name}")

# ============================================
# 主页面
# ============================================
st.title("🗺️ 卫星影像数据浏览")
st.markdown(f"**研究区: {area_name}** | 数据源: Microsoft Planetary Computer")

# 研究区概览地图 (leafmap 失败自动降级为静态图)
from utils.map_utils import render_area_map

col_map, col_info = st.columns([2, 1])

with col_map:
    with StreamlitErrorBoundary("研究区地图", st=st, show_traceback=False):
        mode, static_img = render_area_map(
            area_name, bbox_custom, area_info["center"], STUDY_AREAS
        )
        if mode == "static" and static_img:
            st.image(static_img, caption=f"📍 {area_name} 位置示意", width="stretch")
        elif mode == "error":
            st.warning("地图组件加载失败")

with col_info:
    st.markdown(f"""
    **边界范围**
    - 西: {bbox_custom[0]:.1f}°
    - 南: {bbox_custom[1]:.1f}°
    - 东: {bbox_custom[2]:.1f}°
    - 北: {bbox_custom[3]:.1f}°
    """)

    # 研究区描述卡片
    keywords_html = " ".join(
        f"<span style='display:inline-block;background:#e8f4fd;color:#1f77b4;"
        f"border-radius:10px;padding:2px 8px;margin:2px;font-size:11px;'>{kw}</span>"
        for kw in area_info.get("keywords", [])
    )
    st.markdown(
        f"<div style='background:#f8fbff;border:1px solid #d4e6f1;border-radius:8px;"
        f"padding:10px 12px;margin:6px 0;'>"
        f"<b>📖 {area_name}</b><br>"
        f"<span style='font-size:12px;color:#555;'>{area_info.get('description', '')}</span><br>"
        f"<div style='margin-top:6px;'>{keywords_html}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    **卫星**: {satellite} ({sat_info['resolution']}m)

    **可用波段**: {', '.join(sat_info['bands'].keys())}
    """)

# ============================================
# 搜索执行与结果
# ============================================
if search_clicked:
    with st.spinner(f"🛰️ 正在从 Planetary Computer 搜索 {satellite} 影像..."):
        with StreamlitErrorBoundary("STAC 影像搜索", st=st):
            from utils.pc_data import search_images

            results = search_images(
                bbox=bbox_custom,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                collection=satellite,
                cloud_cover_max=cloud_cover,
                max_items=max_items,
            )

            if len(results) == 0:
                st.warning("⚠️ 未找到符合条件的影像。建议放宽云量限制或扩大时间范围。")
            else:
                st.success(f"✅ 找到 {len(results)} 景影像！")
                st.session_state["search_results"] = results
                st.session_state["search_satellite"] = satellite

# ============================================
# 搜索结果表格
# ============================================
if "search_results" in st.session_state and st.session_state["search_results"]:
    results = st.session_state["search_results"]
    satellite = st.session_state.get("search_satellite", "Sentinel-2 L2A")

    st.divider()
    st.subheader("📋 搜索结果")

    # 表格
    result_data = []
    for i, r in enumerate(results):
        result_data.append(
            {
                "序号": i + 1,
                "日期": r["datetime"],
                "云量(%)": r["cloud_cover"] if isinstance(r["cloud_cover"], (int, float)) else "N/A",
                "ID": r["id"][:40] + "...",
            }
        )
    st.dataframe(result_data)

    # ============================================
    # 影像预览
    # ============================================
    st.subheader("👁️ 影像预览")

    # ---- 多景影像网格预览 (每景完整显示) ----
    st.markdown("**📸 多景影像快速浏览**")
    st.caption("前几景影像的完整 RGB 预览（点击下方选择框可切换单景详细分析）")
    from utils.pc_data import get_rgb_preview_cached

    n_grid = min(len(results), 6)
    grid_cols = st.columns(min(3, n_grid))
    grid_imgs = {}
    for i in range(n_grid):
        with grid_cols[i % 3]:
            r = results[i]
            try:
                img = get_rgb_preview_cached(r["id"], collection=satellite_used, width=400)
                if img:
                    grid_imgs[i] = img
                    st.image(img, caption=f"{i+1}. {r['datetime'][:10]} ☁️{r['cloud_cover']}%",
                             width="stretch")
                else:
                    st.caption(f"{i+1}. {r['datetime'][:10]} — 预览不可用")
            except Exception:
                st.caption(f"{i+1}. {r['datetime'][:10]} — 加载失败")

    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_idx = st.selectbox(
            "选择影像",
            range(len(results)),
            format_func=lambda i: f"{i+1}. {results[i]['datetime']}",
        )

        compare_all = st.toggle("三模式对比显示", value=True,
                                help="同时显示 RGB / NDVI / MNDWI 三张图")

    with col2:
        if selected_idx is not None:
            item = results[selected_idx]["item"]
            satellite_used = st.session_state.get("search_satellite", "Sentinel-2 L2A")

            with st.spinner("⏳ 加载预览中..."):
                with StreamlitErrorBoundary("影像预览", st=st):
                    from utils.pc_data import (
                        get_rgb_preview_cached,
                        get_ndvi_preview_cached,
                        get_mndwi_preview_cached,
                        get_thumbnail,
                    )

                    date_label = results[selected_idx]["datetime"]

                    # ============================================
                    # 三模式对比显示 (默认): RGB + NDVI + MNDWI 并列
                    # ============================================
                    if compare_all:
                        rgb_img = get_rgb_preview_cached(item.id, collection=satellite_used, width=800)
                        ndvi_img = get_ndvi_preview_cached(item.id, collection=satellite_used, width=800)
                        mndwi_img = get_mndwi_preview_cached(item.id, collection=satellite_used, width=800)

                        # 第一行: RGB 大图
                        st.markdown(f"**📷 RGB 真彩色** — {date_label}")
                        if rgb_img:
                            st.image(rgb_img, width="stretch")
                        elif get_thumbnail(item):
                            st.image(get_thumbnail(item), caption="缩略图 (回退)")

                        # 第二行: NDVI + MNDWI 双图并列
                        st.markdown(f"**🧮 指数对比** — {date_label}")
                        comp_cols = st.columns(2)
                        with comp_cols[0]:
                            if ndvi_img:
                                st.image(ndvi_img, caption="NDVI 植被指数",
                                         width="stretch")
                                st.caption("🟢 绿色=植被茂密 | 🔴 红色=裸地/水体")
                            else:
                                st.error("NDVI 预览加载失败")
                        with comp_cols[1]:
                            if mndwi_img:
                                st.image(mndwi_img, caption="MNDWI 水体指数",
                                         width="stretch")
                                st.caption("🔵 蓝色=水体 | ⚪ 白色=非水体")
                            else:
                                st.error("MNDWI 预览加载失败")

                    else:
                        # ============================================
                        # 单选模式 (兼容原逻辑)
                        # ============================================
                        preview_type = st.radio(
                            "预览模式",
                            ["RGB 真彩色", "NDVI 植被指数", "MNDWI 水体指数"],
                            key="preview_mode_single",
                        )
                        if preview_type == "RGB 真彩色":
                            img = get_rgb_preview_cached(item.id, collection=satellite_used, width=800)
                            if img:
                                st.image(img, caption=f"{date_label} RGB 真彩色 ({satellite_used})")
                            else:
                                thumb = get_thumbnail(item)
                                if thumb:
                                    st.image(thumb, caption="缩略图 (回退)")

                        elif preview_type == "NDVI 植被指数":
                            img = get_ndvi_preview_cached(item.id, collection=satellite_used, width=800)
                            if img:
                                st.image(img, caption=f"{date_label} NDVI")
                                st.caption("🟢 绿色=植被茂密 | 🟡 黄色=稀疏 | 🔴 红色=裸地/水体")
                            else:
                                st.error("NDVI 预览加载失败")

                        elif preview_type == "MNDWI 水体指数":
                            img = get_mndwi_preview_cached(item.id, collection=satellite_used, width=800)
                            if img:
                                st.image(img, caption=f"{date_label} MNDWI")
                                st.caption("🔵 蓝色=水体 | ⚪ 白色=非水体")
                            else:
                                st.error("MNDWI 预览加载失败")

    # ============================================
    # 数据下载
    # ============================================
    st.divider()
    st.subheader("💾 数据下载")

    download_tab1, download_tab2 = st.tabs(["单波段下载", "全波段合成下载"])

    with download_tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            band_names = list(COLLECTIONS[satellite_used]["bands"].keys())
            band_display = {
                "blue": "蓝 (Blue)",
                "green": "绿 (Green)",
                "red": "红 (Red)",
                "nir": "近红外 (NIR)",
                "swir1": "短波红外1 (SWIR1)",
                "swir2": "短波红外2 (SWIR2)",
            }
            selected_band = st.selectbox(
                "选择波段",
                band_names,
                format_func=lambda b: band_display.get(b, b),
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("📥 下载波段", type="primary"):
                if selected_idx is not None:
                    with st.spinner("下载中..."):
                        with StreamlitErrorBoundary("单波段下载", st=st):
                            from utils.pc_data import download_band

                            tmp_dir = tempfile.gettempdir()
                            fname = f"{area_name}_{results[selected_idx]['datetime']}_{selected_band}.tif"
                            output_path = os.path.join(tmp_dir, fname)

                            result_path = download_band(
                                results[selected_idx]["item"],
                                selected_band,
                                output_path,
                                collection=satellite_used,
                            )

                            if result_path and os.path.exists(output_path):
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ 保存 GeoTIFF",
                                        f,
                                        file_name=fname,
                                        mime="image/tiff",
                                    )
                                st.success(f"✅ 波段已准备: {fname}")
                            else:
                                st.error("下载失败")

    with download_tab2:
        st.markdown("下载全部 6 个波段并合成为一个多波段 GeoTIFF 文件")
        if st.button("📦 下载全波段 (6波段)", type="primary"):
            if selected_idx is not None:
                with st.spinner("下载并合成中 (可能需要 1-2 分钟)..."):
                    with StreamlitErrorBoundary("全波段下载", st=st):
                        from utils.pc_data import download_multiband

                        tmp_dir = tempfile.gettempdir()
                        fname = f"{area_name}_{results[selected_idx]['datetime']}_6band.tif"
                        output_path = os.path.join(tmp_dir, fname)

                        result_path = download_multiband(
                            results[selected_idx]["item"],
                            output_path,
                            collection=satellite_used,
                            band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
                        )

                        if result_path and os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "⬇️ 保存多波段 GeoTIFF",
                                    f,
                                    file_name=fname,
                                    mime="image/tiff",
                                )
                            st.success(f"✅ 全波段下载完成: {fname}")
                        else:
                            st.error("多波段下载失败")

else:
    # 无搜索结果时显示提示
    st.info("👆 在左侧设置搜索条件后点击「🔍 搜索影像」开始")
    with st.expander("📖 使用说明"):
        st.markdown("""
        **第一步: 设置搜索条件**
        - 选择研究区（6个西北干旱区核心区域可选）
        - 选择卫星数据源（Sentinel-2 分辨率最高 10m）
        - 设置时间范围（夏季云量少，推荐 6-9 月）
        - 调整云量上限（建议 20% 以下）

        **第二步: 搜索并预览**
        - 点击「🔍 搜索影像」
        - 从结果列表中选择一景影像
        - 切换 RGB / NDVI / MNDWI 预览模式

        **第三步: 下载数据**
        - **单波段**: 下载单个波段为 GeoTIFF
        - **全波段**: 下载 6 波段合成文件，可用于水体监测和植被分析页面

        **注意**: 下载需要网络访问 Planetary Computer，首次加载较慢属正常现象
        """)
