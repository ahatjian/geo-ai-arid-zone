"""
地表温度 (LST) 反演页面 — 热环境分析
======================================
基于 Landsat Collection 2 Level-2 地表温度产品 (ST_B10/ST_B6)
功能: LST 反演 (K→℃) + 热环境 5 级分级 + LST-NDVI 关系
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
from utils.aoi import render_aoi_selector
from utils.pc_data import search_images, get_rgb_preview_cached, download_multiband
from utils.lst import (
    THERMAL_LEVELS, DEFAULT_THERMAL_THRESHOLDS,
    read_lst_array, assess_thermal,
    compute_lst_ndvi_relation,
)
from utils.indices import calc_ndvi
from utils.visualization import render_classification

st.set_page_config(page_title="地表温度反演", page_icon="🌡️", layout="wide")

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
.lst-stat-box {
    background: #fdeaea;
    border: 1px solid #f0c4c4;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.lst-stat-box .label { font-size: 11px; color: #b0413e; margin-bottom: 4px; }
.lst-stat-box .value { font-size: 20px; font-weight: 700; color: #7a1f1a; }
</style>
""", unsafe_allow_html=True)

# Landsat 系列 (含热红外波段)
LANDSAT_KEYS = [k for k in COLLECTIONS.keys() if k.startswith("Landsat")]

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🌡️ 地表温度设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="吐鲁番盆地",
        help_text="推荐吐鲁番盆地/塔里木盆地 (夏季地表温度极高)",
        key_prefix="lst",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据 (仅 Landsat 含热红外)",
        LANDSAT_KEYS,
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
        help="Landsat 8/9 使用 ST_B10, Landsat 4/5/7 使用 ST_B6 地表温度产品",
    )
    st.caption(COLLECTIONS[satellite]["description"])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=120))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 20,
        help="LST 反演建议低云量影像")
    max_items = st.slider("最大影像数", 1, 15, 5)

    st.divider()

    st.subheader("分析选项")
    use_lst_map = st.checkbox("地表温度图 (LST)", value=True)
    use_thermal_class = st.checkbox("热环境分级 (5级)", value=True)
    use_lst_ndvi = st.checkbox("LST-NDVI 关系分析", value=True,
        help="热环境与植被覆盖的关系 (散点/分箱)")

    st.divider()

    with st.expander("⚙️ 高级设置"):
        t1 = st.number_input("较低温区阈值 (°C)", value=float(DEFAULT_THERMAL_THRESHOLDS[0]), step=1.0)
        t2 = st.number_input("常温区阈值 (°C)", value=float(DEFAULT_THERMAL_THRESHOLDS[1]), step=1.0)
        t3 = st.number_input("较高温区阈值 (°C)", value=float(DEFAULT_THERMAL_THRESHOLDS[2]), step=1.0)
        t4 = st.number_input("高温区阈值 (°C)", value=float(DEFAULT_THERMAL_THRESHOLDS[3]), step=1.0)
        pixel_size = st.number_input("像元大小 (m)", value=30.0, min_value=1.0)

    thresholds = [t1, t2, t3, t4]

    search_clicked = st.button(
        "🔍 搜索影像 & 分析", type="primary", width="stretch"
    )

# ============================================================
# 主页面
# ============================================================
st.title("🌡️ 地表温度 (LST) 反演与热环境分析")
st.markdown(
    f"**研究区: {area_name}** | "
    f"数据源: {satellite} | "
    f"日期: {start_date} → {end_date}"
)
st.markdown(
    "基于 USGS Collection 2 Level-2 地表温度产品反演地表温度 (K→℃)，"
    "实现热环境 5 级分级与 LST-NDVI 关系分析，适用于干旱区热岛/热环境监测。"
)

if search_clicked:
    # bbox 已由研究区/AOI 选择组件提供 (支持预设 + 自定义 AOI)

    # ---- Step 1: 搜索影像 ----
    with st.spinner("🔍 正在搜索 Landsat 影像..."):
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
                    '<div style="width:100%;height:120px;background:#f5e5e5;'
                    'border:1px dashed #e0c0c0;border-radius:6px;display:flex;'
                    'align-items:center;justify-content:center;color:#999">'
                    '无预览</div>',
                    unsafe_allow_html=True,
                )

            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_lst_{r['id']}",
                value=i < 3,
            )
            if selected:
                selected_items.append(r)

    n_selected = len(selected_items)
    if n_selected == 0:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    st.info(f"📊 已选择 **{n_selected}** 景影像")

    # ---- Step 3: 读取 LST & 分析 ----
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_lst = []
    all_ndvi = []
    image_labels = []

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / n_selected
        progress_bar.progress(pct)
        status_text.text(f"⬇️ 反演 LST... [{idx+1}/{n_selected}] {item['datetime']}")

        try:
            # 读取地表温度产品
            lst_kelvin = read_lst_array(item["item"], collection=satellite)
            all_lst.append(lst_kelvin)

            # 读取 red + nir 计算 NDVI (用于 LST-NDVI 关系)
            if use_lst_ndvi:
                try:
                    tmp_path = os.path.join(tempfile.gettempdir(), f"lst_ndvi_{item['id'][:12]}.tif")
                    tif_path = download_multiband(
                        item["item"], tmp_path, collection=satellite,
                        band_names=["red", "nir"],
                    )
                    if tif_path and os.path.exists(tif_path):
                        import rasterio
                        with rasterio.open(tif_path) as src:
                            bands = src.read().astype(np.float32)
                        red = bands[0]
                        nir = bands[1]
                        if np.nanmedian(red) > 10:
                            red = red / 10000.0
                            nir = nir / 10000.0
                        ndvi = calc_ndvi(red, nir)
                        # 对齐尺寸 (ST 与多光谱分辨率可能不同)
                        if ndvi.shape != lst_kelvin.shape:
                            from skimage.transform import resize
                            ndvi = resize(ndvi, lst_kelvin.shape, preserve_range=True)
                        all_ndvi.append(ndvi)
                        try:
                            os.remove(tif_path)
                        except Exception:
                            pass
                except Exception:
                    all_ndvi.append(None)
            else:
                all_ndvi.append(None)

            image_labels.append(item["datetime"])
        except Exception as e:
            st.warning(f"⚠️ {item['datetime']} 反演失败: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 反演完成")

    if not all_lst:
        st.error("❌ 没有成功反演的影像")
        st.stop()

    st.success(f"✅ 成功反演 **{len(all_lst)}** 景影像的地表温度")

    # ====================================================
    # 结果展示 (取最新一景作为主分析影像)
    # ====================================================
    lst_main = all_lst[0]
    ndvi_main = all_ndvi[0] if all_ndvi else None
    main_date = image_labels[0]

    with st.spinner("⏳ 热环境评估中..."):
        with StreamlitErrorBoundary("热环境评估", st=st, show_traceback=False):
            result = assess_thermal(lst_main, pixel_size_m=pixel_size, thresholds=thresholds)

    # 写入 session_state 供报告导出页自动采集
    st.session_state["lst_stats"] = {
        "area": area_name,
        "date": main_date,
        "satellite": satellite,
        "summary": {
            "mean_lst_c": float(result.summary["mean_lst_c"]),
            "max_lst_c": float(result.summary["max_lst_c"]),
            "min_lst_c": float(result.summary["min_lst_c"]),
            "hot_ratio": float(result.summary["hot_ratio"]),
            "dominant_level": result.summary["dominant_level"],
        },
        "stats": result.stats,
    }

    # ---- 汇总卡片 ----
    st.subheader("📊 热环境评估汇总")
    s = result.summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均地表温度 🌡️", f"{s['mean_lst_c']} °C")
    with col2:
        st.metric("最高地表温度 🔥", f"{s['max_lst_c']} °C")
    with col3:
        st.metric("高温区占比 (>30°C)", f"{s['hot_ratio']*100:.1f}%")
    with col4:
        st.metric("主导温区", s["dominant_level"])

    st.divider()

    # ---- 标签页 ----
    tab_names = []
    if use_lst_map:
        tab_names.append("地表温度图")
    if use_thermal_class:
        tab_names.append("热环境分级")
    if use_lst_ndvi:
        tab_names.append("LST-NDVI 关系")
    tab_names.append("分级统计")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # Tab: 地表温度图
    if use_lst_map:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("地表温度 (LST) 反演结果，单位 °C")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result.lst_celsius, cmap="RdYlBu_r",
                           vmin=np.nanpercentile(result.lst_celsius, 2),
                           vmax=np.nanpercentile(result.lst_celsius, 98))
            ax.set_title(f"地表温度 LST — {area_name} ({main_date})", fontsize=13)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="LST (°C)")
            st.pyplot(fig)
            plt.close(fig)

    # Tab: 热环境分级
    if use_thermal_class:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("5 级热环境分级 — 蓝(低温) → 红(高温)")
            class_names = [lv["name"] for lv in THERMAL_LEVELS]
            class_colors = [lv["color"] for lv in THERMAL_LEVELS]
            fig_img = render_classification(
                result.category,
                class_names=class_names,
                class_colors=class_colors,
                title=f"热环境分级 — {area_name} ({main_date})",
            )
            st.image(fig_img, width="stretch")

    # Tab: LST-NDVI 关系
    if use_lst_ndvi:
        with tabs[tab_idx]:
            tab_idx += 1
            if ndvi_main is not None:
                st.caption("地表温度与植被覆盖的关系 — 绿洲降温效应")
                relation = compute_lst_ndvi_relation(result.lst_celsius, ndvi_main)

                fig, axes = plt.subplots(1, 2, figsize=(13, 5))

                # 散点图 (抽样)
                lst_flat = result.lst_celsius.flatten()
                ndvi_flat = ndvi_main.flatten()
                mask = np.isfinite(lst_flat) & np.isfinite(ndvi_flat)
                lst_s = lst_flat[mask]
                ndvi_s = ndvi_flat[mask]
                if len(lst_s) > 5000:
                    idx = np.random.choice(len(lst_s), 5000, replace=False)
                    lst_s, ndvi_s = lst_s[idx], ndvi_s[idx]

                axes[0].scatter(ndvi_s, lst_s, s=3, alpha=0.3, c="#d2691e")
                axes[0].set_xlabel("NDVI")
                axes[0].set_ylabel("LST (°C)")
                axes[0].set_title(f"LST-NDVI 散点 (r={relation['correlation']})")
                axes[0].grid(True, alpha=0.3)

                # 分箱折线
                axes[1].plot(relation["ndvi_bins"], relation["mean_lst"], "o-", color="#b2182b")
                axes[1].set_xlabel("NDVI 分箱中心")
                axes[1].set_ylabel("平均 LST (°C)")
                axes[1].set_title("LST 随 NDVI 变化 (绿洲降温趋势)")
                axes[1].grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close(fig)

                if relation["correlation"] is not None:
                    if relation["correlation"] < -0.3:
                        st.info("🌿 相关系数显著为负，说明植被覆盖对地表有显著降温效应（绿洲冷岛效应）。")
                    else:
                        st.info("相关性较弱，该区域地表温度受植被影响不显著。")
            else:
                st.warning("⚠️ 未能读取 NDVI 波段，无法进行 LST-NDVI 关系分析。")

    # Tab: 分级统计
    with tabs[tab_idx]:
        tab_idx += 1
        st.caption("各热环境等级面积与占比统计")

        df = pd.DataFrame([
            {
                "等级": s_["name"],
                "像元数": s_["pixel_count"],
                "占比": f"{s_['ratio']*100:.2f}%",
                "面积(km²)": s_["area_km2"],
                "均温(°C)": s_["mean_lst_c"],
                "风险": s_["risk"],
            }
            for s_ in result.stats
        ])
        st.dataframe(df, width="stretch", hide_index=True)

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
        ax.set_title(f"热环境等级面积占比 — {area_name} ({main_date})", fontsize=13)
        ax.set_xlim(0, max(ratios) * 1.2)
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ---- 结果一键入库 ----
    st.divider()
    st.subheader("💾 保存分析结果")
    from utils.save_ui import render_save_button

    save_meta = {"研究区": area_name, "模块": "地表温度LST", "日期": main_date}
    render_save_button(
        default_name=f"{area_name}_地表温度_{main_date[:10]}",
        data=result.lst_celsius,
        kind="npy",
        meta={**save_meta, "说明": "地表温度 (°C)"},
        key_suffix="lst_temp",
    )
    render_save_button(
        default_name=f"{area_name}_热环境分级_{main_date[:10]}",
        data=result.category,
        kind="npy",
        meta={**save_meta, "说明": "热环境5级分级 (1低温~5高温)"},
        key_suffix="lst_cat",
    )
    render_save_button(
        default_name=f"{area_name}_热环境统计_{main_date[:10]}",
        data=result.stats,
        kind="csv",
        meta={**save_meta, "说明": "各热环境等级面积占比统计"},
        key_suffix="lst_stats",
    )
    st.caption("💡 保存后可前往「📦 数据下载中心」统一管理、下载或打包全部结果")

    # ---- 多时相提示 ----
    if n_selected > 1:
        st.divider()
        st.info(
            f"💡 已选择 {n_selected} 景影像，以上为主影像 ({main_date}) 的详细分析。"
            "如需多时相地表温度动态对比，可分别选择不同日期影像重新分析。"
        )
