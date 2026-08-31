"""
蒸散发 (ET) 估算页面 — SEBAL 简化能量平衡法
==============================================
基于 Landsat LST + 反照率 + NDVI 的地表蒸散发估算
能量平衡: Rn = H + LE + G → ET
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
from utils.lst import read_lst_array
from utils.desertification import calc_albedo_s2, calc_albedo_landsat
from utils.indices import calc_ndvi
from utils.evapotranspiration import (
    ET_LEVELS, DEFAULT_ET_THRESHOLDS, assess_et,
)
from utils.visualization import render_classification

st.set_page_config(page_title="蒸散发估算", page_icon="💨", layout="wide")

# Landsat 系列 (含热红外波段, ET 需要 LST)
LANDSAT_KEYS = [k for k in COLLECTIONS.keys() if k.startswith("Landsat")]

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("💨 蒸散发设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="河西走廊",
        help_text="推荐河西走廊/塔里木盆地 (绿洲农业蒸散发显著)",
        key_prefix="et",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据 (仅 Landsat 含热红外)",
        LANDSAT_KEYS,
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=120))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 15)
    max_items = st.slider("最大影像数", 1, 15, 5)

    st.divider()

    st.subheader("🌤️ 气象参数")
    st.caption("SEBAL 能量平衡需要的气象输入")
    rs_down = st.number_input("太阳短波辐射 Rs↓ (W/m²)", value=600.0, min_value=0.0, step=50.0,
        help="夏季晴天约 500-800 W/m²")
    ta_celsius = st.number_input("空气温度 Ta (°C)", value=25.0, min_value=-30.0, max_value=50.0, step=0.5)
    wind_speed = st.number_input("风速 u (m/s)", value=2.0, min_value=0.0, step=0.5)
    transmittance = st.number_input("大气短波透过率 τ", value=0.75, min_value=0.1, max_value=1.0, step=0.05)

    st.divider()

    with st.expander("⚙️ 高级设置"):
        t1 = st.number_input("低蒸散发阈值 (mm/day)", value=float(DEFAULT_ET_THRESHOLDS[0]), step=0.1)
        t2 = st.number_input("中等蒸散发阈值 (mm/day)", value=float(DEFAULT_ET_THRESHOLDS[1]), step=0.1)
        t3 = st.number_input("高蒸散发阈值 (mm/day)", value=float(DEFAULT_ET_THRESHOLDS[2]), step=0.1)
        t4 = st.number_input("极高蒸散发阈值 (mm/day)", value=float(DEFAULT_ET_THRESHOLDS[3]), step=0.1)
        pixel_size = st.number_input("像元大小 (m)", value=30.0, min_value=1.0)

    thresholds = [t1, t2, t3, t4]

    search_clicked = st.button(
        "🔍 搜索影像 & 估算蒸散发", type="primary", use_container_width=True
    )

# ============================================================
# 主页面
# ============================================================
st.title("💨 地表蒸散发 (ET) 估算")
st.markdown(f"**研究区: {area_name}** | 数据源: {satellite} | 日期: {start_date} → {end_date}")
st.markdown(
    "基于 SEBAL 简化能量平衡法 (Rn = H + LE + G)，结合 Landsat 地表温度、反照率与 NDVI "
    "估算地表蒸散发，适用于干旱区水资源平衡与绿洲耗水评估。"
)

if search_clicked:
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
            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_et_{r['id']}",
                value=i < 3,
            )
            if selected:
                selected_items.append(r)

    n_selected = len(selected_items)
    if n_selected == 0:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    st.info(f"📊 已选择 **{n_selected}** 景影像")

    # ---- Step 3: 读取 LST + 多光谱, 估算 ET ----
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_et_results = []
    image_labels = []

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / n_selected
        progress_bar.progress(pct)
        status_text.text(f"⬇️ 估算蒸散发... [{idx+1}/{n_selected}] {item['datetime']}")

        try:
            # 1. 读取地表温度
            lst_kelvin = read_lst_array(item["item"], collection=satellite)

            # 2. 读取多光谱波段 (反照率 + NDVI)
            tmp_path = os.path.join(tempfile.gettempdir(), f"et_ms_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)
            import rasterio
            with rasterio.open(tif_path) as src:
                bands = src.read().astype(np.float32)
            if os.path.exists(tif_path):
                os.remove(tif_path)

            blue, green, red, nir, swir1 = bands[0], bands[1], bands[2], bands[3], bands[4]
            swir2 = bands[5] if bands.shape[0] > 5 else swir1

            # 对齐尺寸到 LST
            if bands.shape[1:] != lst_kelvin.shape:
                from skimage.transform import resize
                def _rs(arr):
                    return resize(arr, lst_kelvin.shape, preserve_range=True, anti_aliasing=True)
                blue, green, red, nir, swir1, swir2 = map(_rs, [blue, green, red, nir, swir1, swir2])

            # 3. 反照率 (DN 自动缩放)
            if "Landsat" in satellite:
                albedo = calc_albedo_landsat(blue, red, nir, swir1, swir2)
            else:
                albedo = calc_albedo_s2(blue, red, nir, swir1, swir2)

            # 4. NDVI
            ndvi = calc_ndvi(red, nir)

            # 5. 蒸散发估算
            result = assess_et(
                albedo=albedo, ndvi=ndvi, lst_kelvin=lst_kelvin,
                pixel_size_m=pixel_size,
                rs_down=rs_down, ta_celsius=ta_celsius,
                wind_speed=wind_speed, transmittance=transmittance,
                thresholds=thresholds,
            )

            all_et_results.append(result)
            image_labels.append(item["datetime"])

        except Exception as e:
            st.warning(f"⚠️ {item['datetime']} 估算失败: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 估算完成")

    if not all_et_results:
        st.error("❌ 没有成功估算的影像")
        st.stop()

    st.success(f"✅ 成功估算 **{len(all_et_results)}** 景影像的蒸散发")

    # ====================================================
    # 结果展示 (取最新一景)
    # ====================================================
    result = all_et_results[0]
    main_date = image_labels[0]

    # ---- 汇总卡片 ----
    st.subheader("📊 蒸散发估算汇总")
    s = result.summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均蒸散发 💨", f"{s['mean_et']} mm/day")
    with col2:
        st.metric("平均净辐射 ☀️", f"{s['mean_rn']} W/m²")
    with col3:
        st.metric("平均潜热通量", f"{s['mean_le']} W/m²")
    with col4:
        st.metric("主导等级", s["dominant_level"])

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("平均感热通量", f"{s['mean_h']} W/m²")
    with col6:
        st.metric("平均土壤热通量", f"{s['mean_g']} W/m²")
    with col7:
        st.metric("空气温度", f"{s['ta_celsius']} °C")
    with col8:
        st.metric("太阳辐射", f"{s['rs_down']} W/m²")

    st.divider()

    # ---- 标签页 ----
    tab_et, tab_energy, tab_class, tab_stats = st.tabs(
        ["蒸散发 ET", "能量平衡分量", "ET 分级", "分级统计"]
    )

    with tab_et:
        st.caption("日蒸散发 ET (mm/day)")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(result.et_daily, cmap="YlGnBu",
                       vmin=np.nanpercentile(result.et_daily, 2),
                       vmax=np.nanpercentile(result.et_daily, 98))
        ax.set_title(f"日蒸散发 ET — {area_name} ({main_date})", fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ET (mm/day)")
        st.pyplot(fig)
        plt.close(fig)

    with tab_energy:
        st.caption("能量平衡四分量 (W/m²): Rn = H + LE + G")
        components = [
            ("净辐射 Rn", result.net_radiation, "RdYlBu_r"),
            ("感热通量 H", result.sensible_heat, "YlOrRd"),
            ("潜热通量 LE", result.latent_heat, "YlGnBu"),
            ("土壤热通量 G", result.soil_heat_flux, "YlOrBr"),
        ]
        for idx_comp in range(0, 4, 2):
            c1, c2 = st.columns(2)
            for col_i, col in enumerate([c1, c2]):
                i = idx_comp + col_i
                if i >= 4:
                    break
                name, arr, cmap = components[i]
                with col:
                    fig, ax = plt.subplots(figsize=(6.5, 5.5))
                    vmax = np.nanpercentile(arr, 98) if np.isfinite(arr).any() else 1
                    vmin = np.nanpercentile(arr, 2) if np.isfinite(arr).any() else 0
                    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(name, fontsize=12)
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="W/m²")
                    st.pyplot(fig)
                    plt.close(fig)

    with tab_class:
        st.caption("5 级蒸散发分级 — 红(极低) → 蓝(极高)")
        class_names = [lv["name"] for lv in ET_LEVELS]
        class_colors = [lv["color"] for lv in ET_LEVELS]
        fig_img = render_classification(
            result.category,
            class_names=class_names,
            class_colors=class_colors,
            title=f"蒸散发分级 — {area_name} ({main_date})",
        )
        st.image(fig_img, use_container_width=True)

    with tab_stats:
        st.caption("各蒸散发等级面积与占比")
        df = pd.DataFrame([
            {
                "等级": s_["name"],
                "像元数": s_["pixel_count"],
                "占比": f"{s_['ratio']*100:.2f}%",
                "面积(km²)": s_["area_km2"],
                "平均ET": s_["mean_et"],
                "风险": s_["risk"],
            }
            for s_ in result.stats
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        names = [s_["name"] for s_ in result.stats]
        ratios = [s_["ratio"] * 100 for s_ in result.stats]
        colors = [s_["color"] for s_ in result.stats]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(names[::-1], ratios[::-1], color=colors[::-1])
        for bar, r in zip(bars, ratios[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}%", va="center", fontsize=9)
        ax.set_xlabel("面积占比 (%)")
        ax.set_title(f"蒸散发等级面积占比 — {area_name} ({main_date})", fontsize=13)
        ax.set_xlim(0, max(ratios) * 1.2)
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ---- 方法说明 ----
    st.divider()
    with st.expander("📖 SEBAL 简化方法说明"):
        st.markdown("""
        **能量平衡**: `Rn = H + LE + G`
        
        1. **净辐射 Rn** = (1−albedo)·Rs↓ + εa·σ·Ta⁴ − ε·σ·Ts⁴
        2. **土壤热通量 G** = Rn·(Ts−273.15)/albedo·(0.0038·albedo+0.0074·albedo²)·(1−0.98·NDVI⁴)
        3. **感热通量 H** = ρ·cp·(Ts−Ta)/ra
        4. **潜热通量 LE** = Rn − G − H
        5. **日蒸散发 ET** = LE × 86400 / λ
        
        **简化假设**: 感热通量用空气动力学阻力经验式 (ra=208/u)，未迭代求解 Monin-Obukhov 稳定度；
        气象参数需用户提供 (太阳辐射/气温/风速)。适用于区域蒸散发空间格局分析。
        """)

    # ---- 结果一键入库 ----
    st.divider()
    st.subheader("💾 保存分析结果")
    from utils.save_ui import render_save_button

    save_meta = {"研究区": area_name, "模块": "蒸散发ET", "日期": main_date}
    render_save_button(
        default_name=f"{area_name}_蒸散发ET_{main_date[:10]}",
        data=result.et_daily,
        kind="npy",
        meta={**save_meta, "说明": "日蒸散发 (mm/day)"},
        key_suffix="et_daily",
    )
    render_save_button(
        default_name=f"{area_name}_ET分级_{main_date[:10]}",
        data=result.category,
        kind="npy",
        meta={**save_meta, "说明": "ET 5级分级"},
        key_suffix="et_cat",
    )
    render_save_button(
        default_name=f"{area_name}_ET统计_{main_date[:10]}",
        data=result.stats,
        kind="csv",
        meta={**save_meta, "说明": "各ET等级面积占比统计"},
        key_suffix="et_stats",
    )
    st.caption("💡 保存后可前往「📦 数据下载中心」统一管理、下载或打包全部结果")

    # ---- 多时相提示 ----
    if n_selected > 1:
        st.info(
            f"💡 已选择 {n_selected} 景影像，以上为主影像 ({main_date}) 的详细分析。"
            "如需多时相蒸散发动态对比，可分别选择不同日期影像重新分析。"
        )
