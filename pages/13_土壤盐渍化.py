"""
土壤盐渍化监测页面 — 盐分指数 / 盐渍化分级
============================================
基于 Sentinel-2/Landsat 多光谱数据
指标: SI, SI1, SI2, NDSI(盐分), BI
分级: 5 级 (非/轻度/中度/重度/极重度盐渍化)
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.pc_data import (
    search_images, get_rgb_preview_cached,
    download_multiband,
)
from utils.salinity import (
    SALINITY_LEVELS, DEFAULT_NDSI_THRESHOLDS,
    assess_salinity, compute_salinity_stats,
    calc_si, calc_ndsi_salinity, calc_bi,
)
from utils.visualization import render_classification

st.set_page_config(page_title="土壤盐渍化监测", page_icon="🧂", layout="wide")

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
.sal-stat-box {
    background: #fdf6ec;
    border: 1px solid #e8d9c0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.sal-stat-box .label { font-size: 11px; color: #a9742f; margin-bottom: 4px; }
.sal-stat-box .value { font-size: 20px; font-weight: 700; color: #5a3e1b; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🧂 盐渍化监测设置")

    st.subheader("研究区")
    default_area = st.session_state.get("selected_area", "塔里木盆地")
    if default_area not in STUDY_AREAS:
        default_area = "塔里木盆地"
    area_name = st.selectbox(
        "选择研究区",
        list(STUDY_AREAS.keys()),
        index=list(STUDY_AREAS.keys()).index(default_area),
        help="推荐塔里木盆地/河西走廊 (绿洲盐渍化高发区)",
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
        start_date = st.date_input("开始日期", date.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 10,
        help="盐渍化分析需要低云量清晰影像")
    max_items = st.slider("最大影像数", 1, 15, 5)

    st.divider()

    st.subheader("分析选项")
    use_si = st.checkbox("盐分指数 SI/SI1/SI2", value=True)
    use_ndsi = st.checkbox("归一化盐分指数 NDSI", value=True)
    use_bi = st.checkbox("亮度指数 BI", value=True)
    use_classify = st.checkbox("盐渍化分级 (5级)", value=True)

    st.divider()

    # 高级设置
    with st.expander("⚙️ 高级设置"):
        classify_method = st.radio(
            "分级方法",
            ["composite", "ndsi"],
            format_func=lambda x: "综合分级(掩膜植被/水体)" if x == "composite" else "仅按 NDSI 阈值",
            index=0,
            help="composite: 先掩膜植被/水体再分级; ndsi: 直接按 NDSI 阈值分级",
        )
        t1 = st.number_input("轻度阈值 (NDSI)", value=float(DEFAULT_NDSI_THRESHOLDS[0]), step=0.01)
        t2 = st.number_input("中度阈值 (NDSI)", value=float(DEFAULT_NDSI_THRESHOLDS[1]), step=0.01)
        t3 = st.number_input("重度阈值 (NDSI)", value=float(DEFAULT_NDSI_THRESHOLDS[2]), step=0.01)
        t4 = st.number_input("极重度阈值 (NDSI)", value=float(DEFAULT_NDSI_THRESHOLDS[3]), step=0.01)
        veg_threshold = st.slider("植被掩膜阈值 (NDVI)", 0.2, 0.6, 0.4, 0.05,
            help="NDVI 高于此值判为植被, 归入非盐渍化")
        pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)

    thresholds = [t1, t2, t3, t4]

    search_clicked = st.button(
        "🔍 搜索影像 & 分析", type="primary", use_container_width=True
    )

# ============================================================
# 主页面
# ============================================================
st.title("🧂 土壤盐渍化遥感监测")
st.markdown(
    f"**研究区: {area_name}** | "
    f"数据源: {satellite} | "
    f"日期: {start_date} → {end_date}"
)
st.markdown(
    "西北干旱区绿洲盐渍化是耕地退化与生态安全的核心问题。"
    "本模块基于盐分指数 (SI/NDSI/BI) 实现盐渍化程度 5 级评估。"
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
            else:
                st.markdown(
                    '<div style="width:100%;height:120px;background:#f5ede0;'
                    'border:1px dashed #d9c9b0;border-radius:6px;display:flex;'
                    'align-items:center;justify-content:center;color:#999">'
                    '无预览</div>',
                    unsafe_allow_html=True,
                )

            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_sal_{r['id']}",
                value=i < 3,
            )
            if selected:
                selected_items.append(r)

    n_selected = len(selected_items)
    if n_selected == 0:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    st.info(f"📊 已选择 **{n_selected}** 景影像")

    # ---- Step 3: 下载 & 分析 ----
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_bands = []
    image_labels = []

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / n_selected
        progress_bar.progress(pct)
        status_text.text(f"⬇️ 分析中... [{idx+1}/{n_selected}] {item['datetime']}")

        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f"sal_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)

            if tif_path and os.path.exists(tif_path):
                import rasterio
                with rasterio.open(tif_path) as src:
                    bands_data = src.read().astype(np.float32)

                all_bands.append(bands_data)
                image_labels.append(item["datetime"])

                try:
                    os.remove(tif_path)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"⚠️ {item['datetime']} 处理失败: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 分析完成")

    if not all_bands:
        st.error("❌ 没有成功处理的影像")
        st.stop()

    st.success(f"✅ 成功分析 **{len(all_bands)}** 景影像")

    # ====================================================
    # 结果展示 (取最新一景作为主分析影像)
    # ====================================================
    bands_main = all_bands[0]
    main_date = image_labels[0]

    with st.spinner("⏳ 盐渍化评估中..."):
        with StreamlitErrorBoundary("盐渍化评估", st=st, show_traceback=False):
            result = assess_salinity(
                bands_main,
                satellite=satellite,
                pixel_size_m=pixel_size,
                method=classify_method,
                thresholds=thresholds,
                veg_threshold=veg_threshold,
            )

    # ---- 汇总卡片 ----
    st.subheader("📊 盐渍化评估汇总")
    s = result.summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("盐渍化总面积占比 🧂", f"{s['total_salinization_ratio']*100:.1f}%")
    with col2:
        st.metric("重度及以上占比 ⚠️", f"{s['severe_salinization_ratio']*100:.1f}%")
    with col3:
        st.metric("主导等级", s["dominant_level"])
    with col4:
        st.metric("NDSI 盐分均值", f"{s['ndsi_salt_mean']:.4f}")

    st.divider()

    # ---- 标签页 ----
    tab_names = []
    if use_classify:
        tab_names.append("盐渍化分级")
    if use_ndsi:
        tab_names.append("NDSI 盐分指数")
    if use_si:
        tab_names.append("盐分指数 SI")
    if use_bi:
        tab_names.append("亮度指数 BI")
    tab_names.append("分级统计")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # Tab: 盐渍化分级
    if use_classify:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("5 级盐渍化分级 — 绿(非) → 紫(极重度)")
            class_names = [lv["name"] for lv in SALINITY_LEVELS]
            class_colors = [lv["color"] for lv in SALINITY_LEVELS]
            fig_img = render_classification(
                result.category,
                class_names=class_names,
                class_colors=class_colors,
                title=f"土壤盐渍化分级 — {area_name} ({main_date})",
            )
            st.image(fig_img, use_container_width=True)

    # Tab: NDSI 盐分指数
    if use_ndsi:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("归一化盐分指数 NDSI = (Red − NIR) / (Red + NIR)，值越大盐渍化越重")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result.ndsi_salt, cmap="YlOrRd", vmin=-0.3, vmax=0.4)
            ax.set_title(f"NDSI 盐分指数 — {area_name} ({main_date})", fontsize=13)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDSI")
            st.pyplot(fig)
            plt.close(fig)

    # Tab: 盐分指数 SI
    if use_si:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("盐分指数 SI = sqrt(Blue × Red)，盐渍土亮度高 → SI 值大")
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(result.si, cmap="magma", vmin=0, vmax=np.nanpercentile(result.si, 98))
                ax.set_title("SI 盐分指数", fontsize=12)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close(fig)
            with col_b:
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(result.si2, cmap="magma", vmin=0, vmax=np.nanpercentile(result.si2, 98))
                ax.set_title("SI2 盐分指数", fontsize=12)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close(fig)

    # Tab: 亮度指数 BI
    if use_bi:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("亮度指数 BI = sqrt(Red² + NIR²)，用于辅助识别盐壳")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result.bi, cmap="gray", vmin=0, vmax=np.nanpercentile(result.bi, 98))
            ax.set_title(f"亮度指数 BI — {area_name} ({main_date})", fontsize=13)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="BI")
            st.pyplot(fig)
            plt.close(fig)

    # Tab: 分级统计
    with tabs[tab_idx]:
        tab_idx += 1
        st.caption("各盐渍化等级面积与占比统计")

        df = pd.DataFrame([
            {
                "等级": s_["name"],
                "像元数": s_["pixel_count"],
                "占比": f"{s_['ratio']*100:.2f}%",
                "面积(km²)": s_["area_km2"],
                "风险": s_["risk"],
                "说明": s_["description"],
            }
            for s_ in result.stats
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 面积占比横向柱状图
        names = [s_["name"] for s_ in result.stats]
        ratios = [s_["ratio"] * 100 for s_ in result.stats]
        colors = [s_["color"] for s_ in result.stats]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(names[::-1], ratios[::-1], color=colors[::-1])
        for bar, r in zip(bars, ratios[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}%", va="center", fontsize=9)
        ax.set_xlabel("面积占比 (%)")
        ax.set_title(f"盐渍化等级面积占比 — {area_name} ({main_date})", fontsize=13)
        ax.set_xlim(0, max(ratios) * 1.2)
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ---- 多时相提示 ----
    if n_selected > 1:
        st.divider()
        st.info(
            f"💡 已选择 {n_selected} 景影像，以上为主影像 ({main_date}) 的详细分析。"
            "如需多时相盐渍化动态对比，可分别选择不同日期影像重新分析。"
        )
