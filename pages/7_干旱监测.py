"""
干旱监测页面 — 多指数遥感干旱分析 + 预测 + 沙漠化
==================================================
干旱指数: VCI / NDDI / NDVI Anomaly / TVDI
预测方法: SARIMA / LSTM / Holt-Winters
沙漠化:   Albedo / TGSI / NDMI / DDI + Albedo-NDVI 特征空间法
数据源:   Sentinel-2 / Landsat via Planetary Computer
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS, COLORMAPS, CACHE_CONFIG
from utils.pc_data import (
    search_images, get_rgb_preview_cached,
    download_multiband, get_ndvi_preview_cached,
)

st.set_page_config(page_title="干旱监测", page_icon="🏜️", layout="wide")

# ============================================
# 样式
# ============================================
st.markdown("""
<style>
.drought-stat-card {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.drought-stat-card .label { font-size: 12px; color: #999; margin-bottom: 4px; }
.drought-stat-card .value { font-size: 22px; font-weight: 700; }
.drought-cat-bar {
    display: inline-block;
    height: 16px;
    border-radius: 3px;
    margin-right: 2px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# 侧边栏
# ============================================
with st.sidebar:
    st.title("🏜️ 干旱监测设置")

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
        start_date = st.date_input("开始日期", date.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 15)
    max_items = st.slider("最大影像数", 3, 30, 10,
        help="至少需要 3 景影像用于时序分析 (VCI/NDVI距平)")

    st.divider()

    st.subheader("干旱指数")
    st.caption("勾选需要计算的指数")

    use_vci = st.checkbox("VCI 植被状态指数", value=True,
        help="基于 NDVI 时序的植被状态 (0-100)，值越低越干旱")
    use_nddi = st.checkbox("NDDI 归一化干旱指数", value=True,
        help="基于 NDVI + NDWI，值 >0.5 为干旱")
    use_anomaly = st.checkbox("NDVI 标准化距平", value=True,
        help="当前 NDVI 偏离历史均值的程度")
    use_tvdi = st.checkbox("TVDI 温度植被干旱指数", value=False,
        help="需要 LST 地表温度数据 (Sentinel-2 不可用，需 Landsat)")

    st.divider()

    st.subheader("高级分析")
    use_forecast = st.checkbox("🔮 干旱趋势预测", value=True,
        help="基于 NDVI 时序预测未来趋势 (SARIMA/LSTM/Holt-Winters)")
    use_desertification = st.checkbox("🏜️ 沙漠化评估", value=True,
        help="Albedo/TGSI/NDMI/DDI 综合沙漠化等级评估")

    st.divider()

    search_clicked = st.button(
        "🔍 搜索影像 & 分析", type="primary", use_container_width=True
    )

    # 高级设置
    with st.expander("⚙️ 高级设置"):
        spi_scale = st.selectbox("SPI 时间尺度 (月)", [1, 3, 6, 12], index=1,
            help="需要气象数据支持")
        vhi_weight = st.slider("VHI 中 VCI 权重", 0.0, 1.0, 0.5,
            help="干旱区建议 0.4 (温度更重要)")
        pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)


# ============================================
# 主页面标题
# ============================================
st.title("🏜️ 遥感干旱监测分析")
st.markdown(
    f"**研究区: {area_name}** | "
    f"数据源: {satellite} | "
    f"日期: {start_date} → {end_date}"
)

# ============================================
# 核心分析流程
# ============================================
if search_clicked:
    bbox = area_info["bbox"]
    center = area_info["center"]

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
        st.warning("⚠️ 未找到符合条件的影像，请调整日期范围或云量阈值。")
        st.stop()

    st.success(f"✅ 找到 **{len(results)}** 景影像")

    # ---- Step 2: 影像选择 ----
    st.subheader("📋 选择分析影像")
    st.caption("勾选至少 3 景用于时序分析 (VCI/NDVI距平)，建议选不同季节的清晰影像")

    # 影像卡片网格
    cols = st.columns(3)
    selected_items = []
    for i, r in enumerate(results):
        with cols[i % 3]:
            # 加载 RGB 预览
            try:
                preview = get_rgb_preview_cached(
                    r["id"], collection=satellite, width=256
                )
            except Exception:
                preview = None

            if preview:
                st.image(preview, use_container_width=True)
            else:
                st.markdown(
                    f'<div style="width:100%;height:140px;background:#1a1a2e;'
                    f'border:1px dashed #444;border-radius:6px;display:flex;'
                    f'align-items:center;justify-content:center;color:#666">'
                    f'无预览</div>',
                    unsafe_allow_html=True,
                )

            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_{r['id']}",
                value=i < 5,  # 默认选前 5 景
            )
            if selected:
                selected_items.append(r)

    n_selected = len(selected_items)
    if n_selected < 3:
        st.warning(f"⚠️ 请至少选择 3 景影像进行时序分析（当前: {n_selected}）")
        st.stop()

    st.info(f"📊 已选择 **{n_selected}** 景影像进行多指数干旱分析")

    # ---- Step 3: 下载波段数据 ----
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_ndvi = []       # (H, W) × T
    all_ndwi = []       # for NDDI
    image_labels = []   # 日期标签
    latest_bands_data = None  # 最新影像的完整波段 (用于沙漠化)

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / n_selected
        progress_bar.progress(pct)
        status_text.text(f"⬇️ 下载中... [{idx+1}/{n_selected}] {item['datetime']}")

        try:
            # 下载多波段 GeoTIFF
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"drought_{item['id'][:12]}.tif"
            )
            tif_path = download_multiband(
                item["item"], tmp_path, collection=satellite
            )

            if tif_path and os.path.exists(tif_path):
                # 读取波段并计算指数
                import rasterio
                with rasterio.open(tif_path) as src:
                    # Sentinel-2 6波段: [B02, B03, B04, B08, B11, B12]
                    #             索引:   0    1    2    3    4    5
                    bands_data = src.read()

                # 判断卫星类型确定波段索引
                if "Sentinel" in satellite:
                    # Sentinel-2: blue=0, green=1, red=2, nir=3, swir1=4, swir2=5
                    red = bands_data[2].astype(np.float32)
                    nir = bands_data[3].astype(np.float32)
                    green = bands_data[1].astype(np.float32)
                else:
                    # Landsat: blue=0, green=1, red=2, nir=3, swir1=4, swir2=5
                    red = bands_data[2].astype(np.float32)
                    nir = bands_data[3].astype(np.float32)
                    green = bands_data[1].astype(np.float32)

                # NDVI
                ndvi_denom = nir + red
                ndvi = np.where(
                    ndvi_denom > 1e-6,
                    (nir - red) / ndvi_denom,
                    0.0
                )
                # 裁剪到合理范围
                ndvi = np.clip(ndvi, -1.0, 1.0)

                # NDWI (Green-NIR 版本, 用于 NDDI)
                ndwi_denom = green + nir
                ndwi = np.where(
                    ndwi_denom > 1e-6,
                    (green - nir) / ndwi_denom,
                    0.0
                )
                ndwi = np.clip(ndwi, -1.0, 1.0)

                all_ndvi.append(ndvi)
                all_ndwi.append(ndwi)
                image_labels.append(item["datetime"])

                # 保留最新影像的完整波段数据 (用于沙漠化评估)
                if idx == 0:
                    latest_bands_data = bands_data.astype(np.float32)

                # 清理临时文件
                try:
                    os.remove(tif_path)
                except Exception:
                    pass
            else:
                st.warning(f"⚠️ {item['datetime']} 下载失败，跳过")

        except Exception as e:
            st.warning(f"⚠️ {item['datetime']} 处理失败: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 数据下载完成")

    actual_count = len(all_ndvi)
    if actual_count < 2:
        st.error("❌ 有效影像不足，无法进行分析")
        st.stop()

    st.success(f"✅ 成功加载 **{actual_count}** 景影像数据")

    # ---- Step 4: 计算干旱指数 ----
    st.divider()
    st.subheader("📊 干旱指数分析结果")

    # 堆叠为时序立方体 (H, W, T)
    ndvi_stack = np.stack(all_ndvi, axis=-1)
    ndwi_stack = np.stack(all_ndwi, axis=-1)

    H, W, T = ndvi_stack.shape

    # 延迟导入干旱模块 (避免 Streamlit 缓存警告)
    from utils.drought import (
        calc_vci_pixelwise, calc_nddi, classify_vci,
        calc_ndvi_anomaly_pixelwise,
        compute_drought_index_stats, compute_drought_stats,
        DROUGHT_CATEGORIES,
    )

    tab_names = []
    tab_contents = {}

    # 4.1 VCI
    if use_vci and T >= 2:
        tab_names.append("VCI")
        try:
            vci_result = calc_vci_pixelwise(ndvi_stack)
            vci_latest = vci_result["vci_latest"]
            vci_cat = vci_result["drought_category"]
            vci_stats = compute_drought_index_stats(vci_latest, "VCI")
            cat_stats = compute_drought_stats(vci_cat, pixel_size_m=pixel_size)
            tab_contents["VCI"] = {
                "data": vci_latest,
                "category": vci_cat,
                "stats": vci_stats,
                "cat_stats": cat_stats,
                "vmin": 0, "vmax": 100,
                "cmap": "YlOrRd_r",
                "label": "VCI (0-100)",
                "description": "植被状态指数 — 值越低表示干旱越严重",
            }
        except Exception as e:
            st.warning(f"VCI 计算失败: {e}")

    # 4.2 NDDI
    if use_nddi:
        tab_names.append("NDDI")
        try:
            # NDDI 使用最新时相
            ndvi_latest_2d = ndvi_stack[:, :, -1]
            ndwi_latest_2d = ndwi_stack[:, :, -1]
            nddi = calc_nddi(ndvi_latest_2d, ndwi_latest_2d)
            nddi_stats = compute_drought_index_stats(nddi, "NDDI")

            # NDDI 分类 (自定义阈值)
            nddi_cat = np.full(nddi.shape, 0, dtype=np.int8)
            nddi_cat[nddi > 0.7] = 3   # 严重干旱
            nddi_cat[(nddi > 0.5) & (nddi <= 0.7)] = 2   # 中等干旱
            nddi_cat[(nddi > 0.3) & (nddi <= 0.5)] = 1   # 轻度干旱
            nddi_cat[(nddi > 0.1) & (nddi <= 0.3)] = 0   # 正常
            nddi_cat[nddi <= 0.1] = -1  # 湿润

            cat_stats_nddi = compute_drought_stats(nddi_cat, pixel_size_m=pixel_size)
            tab_contents["NDDI"] = {
                "data": nddi,
                "category": nddi_cat,
                "stats": nddi_stats,
                "cat_stats": cat_stats_nddi,
                "vmin": -0.5, "vmax": 1.0,
                "cmap": "YlOrRd",
                "label": "NDDI",
                "description": "归一化干旱指数 — 值 >0.5 表示干旱",
            }
        except Exception as e:
            st.warning(f"NDDI 计算失败: {e}")

    # 4.3 NDVI 距平
    if use_anomaly and T >= 2:
        tab_names.append("NDVI距平")
        try:
            anomaly_result = calc_ndvi_anomaly_pixelwise(ndvi_stack)
            anomaly_latest = anomaly_result["anomaly_latest"]
            anomaly_stats = compute_drought_index_stats(anomaly_latest, "NDVI_Anomaly")

            # 距平分类
            anom_cat = np.full(anomaly_latest.shape, 0, dtype=np.int8)
            anom_cat[anomaly_latest < -2.0] = 4
            anom_cat[(anomaly_latest >= -2.0) & (anomaly_latest < -1.5)] = 3
            anom_cat[(anomaly_latest >= -1.5) & (anomaly_latest < -1.0)] = 2
            anom_cat[(anomaly_latest >= -1.0) & (anomaly_latest < -0.5)] = 1
            anom_cat[(anomaly_latest >= 0.5) & (anomaly_latest < 1.0)] = -1
            anom_cat[anomaly_latest >= 1.0] = -2

            cat_stats_anom = compute_drought_stats(anom_cat, pixel_size_m=pixel_size)
            tab_contents["NDVI距平"] = {
                "data": anomaly_latest,
                "category": anom_cat,
                "stats": anomaly_stats,
                "cat_stats": cat_stats_anom,
                "vmin": -2.5, "vmax": 2.5,
                "cmap": "RdBu_r",
                "label": "标准化距平",
                "description": "NDVI 偏离历史均值的标准差倍数",
            }
        except Exception as e:
            st.warning(f"NDVI 距平计算失败: {e}")

    if not tab_contents:
        st.warning("⚠️ 没有成功计算的干旱指数，请检查数据")
        st.stop()

    # ---- Step 5: 结果展示 ----
    tabs = st.tabs(tab_names)

    for tab_name, tab_obj in zip(tab_names, tabs):
        content = tab_contents[tab_name]

        with tab_obj:
            st.caption(content["description"])

            # 统计卡片
            s = content["stats"]
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("有效像元", f"{s['valid_pixels']:,}")
            with col2:
                st.metric("均值", f"{s['mean']:.3f}")
            with col3:
                st.metric("标准差", f"{s['std']:.3f}")
            with col4:
                st.metric("最小值", f"{s['min']:.3f}")
            with col5:
                st.metric("最大值", f"{s['max']:.3f}")

            # 地图 + 分类柱状图
            col_map, col_chart = st.columns([3, 2])

            with col_map:
                # 渲染干旱指数图
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(
                    content["data"],
                    cmap=content["cmap"],
                    vmin=content["vmin"],
                    vmax=content["vmax"],
                )
                ax.set_title(f"{tab_name} — {area_name} ({image_labels[-1]})", fontsize=13)
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                cbar.set_label(content["label"], fontsize=10)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf), use_container_width=True)

            with col_chart:
                # 干旱等级分布柱状图
                cat_data = content["cat_stats"]
                if cat_data:
                    categories = [c["name"] for c in cat_data]
                    ratios = [c["ratio"] * 100 for c in cat_data]
                    colors = [c["color"] for c in cat_data]

                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    bars = ax2.barh(categories, ratios, color=colors, edgecolor="#333")
                    ax2.set_xlabel("面积占比 (%)", fontsize=10)
                    ax2.set_title("干旱等级分布", fontsize=12)
                    # 添加数值标签
                    for bar, val in zip(bars, ratios):
                        if val > 3:
                            ax2.text(
                                bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va="center", fontsize=9
                            )
                    plt.tight_layout()
                    buf2 = BytesIO()
                    plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                    plt.close()
                    buf2.seek(0)
                    st.image(Image.open(buf2), use_container_width=True)

            # 详细统计表
            with st.expander("📋 详细统计表"):
                if content["cat_stats"]:
                    df_cat = pd.DataFrame(content["cat_stats"])
                    df_cat = df_cat.rename(columns={
                        "name": "等级名称",
                        "pixel_count": "像元数",
                        "ratio": "占比",
                        "area_km2": "面积(km²)",
                    })
                    st.dataframe(
                        df_cat[["等级名称", "像元数", "占比", "面积(km²)"]],
                        use_container_width=True,
                        hide_index=True,
                    )

    # ---- Step 6: 时序趋势图 (所有指数共用) ----
    st.divider()
    st.subheader("📈 多时相趋势分析")

    # 计算各时相的区域均值 (用于趋势)
    times = np.arange(T)

    trend_data = {"时相": image_labels}

    # 逐时相计算 NDVI/VCI 均值
    mean_ndvi_series = []
    for t in range(T):
        mean_ndvi_series.append(np.nanmean(ndvi_stack[:, :, t]))
    trend_data["NDVI 均值"] = mean_ndvi_series

    # 如果有 VCI, 加 VCI 均值时序
    if "VCI" in tab_contents:
        vci_ts = tab_contents["VCI"]["data"]
        # VCI 最新的单时相是我们已经有的
        # 但完整 VCI 时序需要重新计算 ... 这里用简化的单时相离散分析
        # 展示 NDVI 时序代替
        pass

    df_trend = pd.DataFrame(trend_data)

    col_trend, col_slope = st.columns([2, 1])

    with col_trend:
        # 绘制 NDVI 均值时间序列
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(
            range(T), mean_ndvi_series,
            "o-", color="#2ecc71", linewidth=2, markersize=8,
            label="NDVI 均值"
        )
        ax3.fill_between(range(T), mean_ndvi_series, alpha=0.15, color="#2ecc71")
        ax3.set_xticks(range(T))
        ax3.set_xticklabels(image_labels, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("NDVI", fontsize=11)
        ax3.set_title(f"{area_name} NDVI 时序变化", fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 添加趋势线
        from utils.trend import theil_sen_slope
        try:
            slope = theil_sen_slope(np.array(mean_ndvi_series))
            intercept = np.median(np.array(mean_ndvi_series) - slope * np.arange(T))
            trend_line = intercept + slope * np.arange(T)
            ax3.plot(
                range(T), trend_line, "--", color="#e74c3c",
                linewidth=1.5, alpha=0.7,
                label=f"Sen 趋势 ({slope:.4f}/步)"
            )
            ax3.legend()
        except Exception:
            pass

        plt.tight_layout()
        buf3 = BytesIO()
        plt.savefig(buf3, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        buf3.seek(0)
        st.image(Image.open(buf3), use_container_width=True)

    with col_slope:
        st.subheader("趋势统计")
        arr_mean = np.array(mean_ndvi_series)
        st.metric("NDVI 均值", f"{np.mean(arr_mean):.4f}")
        st.metric("标准差", f"{np.std(arr_mean):.4f}")
        st.metric("最小值", f"{np.min(arr_mean):.4f}")
        st.metric("最大值", f"{np.max(arr_mean):.4f}")

        try:
            slope_val = theil_sen_slope(arr_mean)
            delta = slope_val > 0
            st.metric(
                "Sen 趋势",
                f"{slope_val:.4f}/步",
                delta=f"{'↗' if delta else '↘'} {'改善' if delta else '退化'}",
            )
        except Exception:
            st.metric("Sen 趋势", "N/A")

    # ---- Step 7: 干旱趋势预测 ----
    if use_forecast:
        st.divider()
        st.subheader("🔮 干旱趋势预测")

        from utils.forecast import (
            forecast_drought_trend, plot_forecast, plot_forecast_comparison,
            prepare_ndvi_timeseries, ForecastResult,
        )

        # 预测设置
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            forecast_method = st.selectbox(
                "预测方法",
                ["sarima", "holt_winters", "lstm", "all"],
                format_func=lambda x: {
                    "sarima": "SARIMA (统计)",
                    "holt_winters": "Holt-Winters (指数平滑)",
                    "lstm": "LSTM (深度学习)",
                    "all": "三种对比",
                }[x],
            )
        with col_f2:
            forecast_steps = st.slider("预测步数", 3, 24,
                min(12, max(3, T)), help="预测未来多少个时相")
        with col_f3:
            test_ratio = st.slider("测试比例", 0.0, 0.4, 0.2, 0.05,
                help="用于评估预测精度的回测比例")

        run_forecast = st.button("🚀 运行预测", type="primary", key="btn_forecast")

        if run_forecast:
            with st.spinner("⏳ 预测中..."):
                try:
                    forecast_result = forecast_drought_trend(
                        ndvi_stack,
                        forecast_steps=forecast_steps,
                        method=forecast_method,
                        test_ratio=test_ratio,
                    )

                    st.success(f"✅ 预测完成")

                    if forecast_method == "all":
                        results = forecast_result
                        # 对比图
                        fig = plot_forecast_comparison(
                            results,
                            title=f"{area_name} 多方法干旱趋势预测对比",
                            return_fig=True,
                        )
                        buf_f = BytesIO()
                        fig.savefig(buf_f, format="png", dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        buf_f.seek(0)
                        st.image(Image.open(buf_f), use_container_width=True)

                        # 各方法指标
                        cols_f = st.columns(min(3, len(results)))
                        for i, (method, r) in enumerate(results.items()):
                            with cols_f[i % 3]:
                                if r.metrics:
                                    st.metric(
                                        f"{method} MAE",
                                        f"{r.metrics.get('MAE', 'N/A')}",
                                    )
                                    st.metric(
                                        f"{method} R²",
                                        f"{r.metrics.get('R2', 'N/A')}",
                                    )
                                if len(r.forecast_values) > 0:
                                    st.caption(
                                        f"最后一期预测: {r.forecast_values[-1]:.4f}"
                                    )
                    else:
                        r = forecast_result
                        fig = plot_forecast(r, title=f"{area_name} 干旱趋势预测 ({r.method})",
                                           return_fig=True)
                        buf_f = BytesIO()
                        fig.savefig(buf_f, format="png", dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        buf_f.seek(0)
                        st.image(Image.open(buf_f), use_container_width=True)

                        # 指标
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("预测方法", r.method)
                        with col_m2:
                            st.metric("MAE", f"{r.metrics.get('MAE', 'N/A')}")
                        with col_m3:
                            st.metric("R²", f"{r.metrics.get('R2', 'N/A')}")
                        with col_m4:
                            last_pred = r.forecast_values[-1] if len(r.forecast_values) > 0 else 0
                            delta = last_pred - np.mean(mean_ndvi_series)
                            st.metric(
                                "末期预测",
                                f"{last_pred:.4f}",
                                delta=f"{delta:+.4f} vs 历史均值",
                            )

                        # 模型参数
                        with st.expander("🔧 模型参数"):
                            st.json(r.model_params)

                except Exception as e:
                    st.error(f"❌ 预测失败: {e}")
                    st.info("💡 提示：请确保选择 6 景以上影像以获得可靠预测。")
        else:
            st.info("👆 设置预测参数后点击「运行预测」")

    # ---- Step 8: 沙漠化评估 ----
    if use_desertification:
        st.divider()
        st.subheader("🏜️ 沙漠化评估")

        from utils.desertification import (
            assess_desertification, get_desertification_colormap,
            DESERTIFICATION_LEVELS, calc_albedo_s2, calc_tgsi,
            calc_ndmi, calc_ddi, compute_desertification_stats,
            analyze_desertification_trend,
        )

        run_desert = st.button("🚀 评估沙漠化", type="primary", key="btn_desert")

        if run_desert:
            if latest_bands_data is None:
                st.error("❌ 缺少波段数据，请重新搜索影像")
            else:
                with st.spinner("⏳ 沙漠化评估中..."):
                    try:
                        result = assess_desertification(
                            latest_bands_data,
                            satellite=satellite,
                            pixel_size_m=pixel_size,
                        )

                        # ---- 汇总卡片 ----
                        s = result.summary
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric(
                                "沙漠化总面积占比",
                                f"{s['total_desertification_ratio']*100:.1f}%",
                            )
                        with col_s2:
                            st.metric(
                                "重度以上占比",
                                f"{s['severe_desertification_ratio']*100:.1f}%",
                            )
                        with col_s3:
                            st.metric("主导等级", s["dominant_level"])
                        with col_s4:
                            st.metric("地表反照率均值", f"{s['albedo_mean']:.4f}")

                        # ---- 4 指数 + 分类 标签页 ----
                        d_tabs = st.tabs([
                            "🗺️ 沙漠化分类", "☀️ Albedo", "🪨 TGSI",
                            "💧 NDMI", "📊 DDI"
                        ])

                        # Tab 1: 分类图
                        with d_tabs[0]:
                            col_c1, col_c2 = st.columns([3, 2])
                            with col_c1:
                                fig_d, ax_d = plt.subplots(figsize=(10, 8))
                                ax_d.imshow(result.category_colors)
                                ax_d.set_title(f"沙漠化等级 — {area_name}", fontsize=13)
                                ax_d.axis("off")
                                # 图例
                                legend_patches = []
                                for level in DESERTIFICATION_LEVELS:
                                    from matplotlib.patches import Patch
                                    legend_patches.append(Patch(
                                        color=level["color"],
                                        label=level["name"],
                                    ))
                                ax_d.legend(
                                    handles=legend_patches, loc="lower right",
                                    fontsize=8, framealpha=0.9,
                                )
                                buf_d = BytesIO()
                                plt.tight_layout()
                                plt.savefig(buf_d, format="png", dpi=120, bbox_inches="tight")
                                plt.close()
                                buf_d.seek(0)
                                st.image(Image.open(buf_d), use_container_width=True)

                            with col_c2:
                                # 分级柱状图
                                names = [s["name"] for s in result.stats]
                                ratios = [s["ratio"] * 100 for s in result.stats]
                                colors = [s["color"] for s in result.stats]

                                fig_d2, ax_d2 = plt.subplots(figsize=(5, 4))
                                bars = ax_d2.barh(names, ratios, color=colors, edgecolor="#333")
                                ax_d2.set_xlabel("面积占比 (%)", fontsize=10)
                                ax_d2.set_title("沙漠化等级分布", fontsize=12)
                                for bar, val in zip(bars, ratios):
                                    if val > 3:
                                        ax_d2.text(
                                            bar.get_width() + 0.3,
                                            bar.get_y() + bar.get_height()/2,
                                            f"{val:.1f}%", va="center", fontsize=9,
                                        )
                                plt.tight_layout()
                                buf_d2 = BytesIO()
                                plt.savefig(buf_d2, format="png", dpi=100, bbox_inches="tight")
                                plt.close()
                                buf_d2.seek(0)
                                st.image(Image.open(buf_d2), use_container_width=True)

                            # 详细统计表
                            with st.expander("📋 沙漠化分级详细统计"):
                                df_desert = pd.DataFrame(result.stats)
                                df_desert = df_desert.rename(columns={
                                    "name": "等级", "pixel_count": "像元数",
                                    "ratio": "占比", "area_km2": "面积(km²)",
                                    "risk": "风险", "description": "描述",
                                })
                                st.dataframe(
                                    df_desert[["等级", "像元数", "占比", "面积(km²)", "风险"]],
                                    use_container_width=True, hide_index=True,
                                )

                        # Tab 2-5: 各指数图
                        index_configs = [
                            ("Albedo", result.albedo, "YlOrRd", 0.1, 0.5, "地表反照率"),
                            ("TGSI", result.tgsi, "Oranges", -0.3, 0.6, "表土粒度指数"),
                            ("NDMI", result.ndmi, "RdYlBu", -0.5, 0.8, "归一化水分指数"),
                            ("DDI", result.ddi, "YlOrRd", 0, 8, "沙漠化差异指数"),
                        ]

                        for i, (name, data, cmap, vmin, vmax, label) in enumerate(index_configs):
                            with d_tabs[i + 1]:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                                ax.set_title(
                                    f"{name} — {area_name} ({image_labels[0]})",
                                    fontsize=13,
                                )
                                ax.axis("off")
                                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                                cbar.set_label(label, fontsize=10)
                                buf = BytesIO()
                                plt.tight_layout()
                                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                                plt.close()
                                buf.seek(0)
                                st.image(Image.open(buf), use_container_width=True)

                                # 统计
                                valid = data[np.isfinite(data)]
                                col_s1, col_s2, col_s3 = st.columns(3)
                                col_s1.metric("均值", f"{np.mean(valid):.4f}")
                                col_s2.metric("标准差", f"{np.std(valid):.4f}")
                                col_s3.metric("有效像元", f"{len(valid):,}")

                        # 沙漠化趋势 (如有时序数据)
                        if T >= 3:
                            st.divider()
                            st.subheader("📈 沙漠化时序趋势")
                            try:
                                trend_result = analyze_desertification_trend(
                                    ndvi_stack,
                                    albedo_stack=ndvi_stack,  # 用 NDVI 近似
                                )
                                col_t1, col_t2, col_t3 = st.columns(3)
                                with col_t1:
                                    st.metric("NDVI 趋势", trend_result["ndvi_trend"])
                                with col_t2:
                                    st.metric("反照率趋势", trend_result["albedo_trend"])
                                with col_t3:
                                    st.metric(
                                        "沙漠化综合趋势",
                                        trend_result["desertification_trend"],
                                    )
                            except Exception as e:
                                st.warning(f"趋势分析失败: {e}")

                    except Exception as e:
                        st.error(f"❌ 沙漠化评估失败: {e}")
                        st.info("💡 提示：请确保已下载有效的多波段影像数据。")
        else:
            st.info("👆 点击「评估沙漠化」开始分析")

    # ---- Step 9: 下载选项 ----
    st.divider()
    st.subheader("📥 结果导出")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # 导出统计为 CSV
        all_stats_rows = []
        for name, content in tab_contents.items():
            row = {"指数": name}
            row.update(content["stats"])
            all_stats_rows.append(row)

        df_export = pd.DataFrame(all_stats_rows)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📊 下载统计摘要 CSV",
            data=csv,
            file_name=f"drought_stats_{area_name}_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        st.info(
            "💡 **提示**: 完整 GeoTIFF 导出请使用「报告导出」页面。\n\n"
            "如需 **SPI/SPEI** (标准化降水/蒸散指数)，需要额外提供月降水与气温数据。\n"
            "可通过 ERA5-Land 或 CHIRPS 数据源获取。"
        )

else:
    # 初始状态 — 展示使用说明
    st.info("👈 在左侧设置参数，点击「搜索影像 & 分析」开始干旱监测")

    with st.expander("📖 干旱指数说明"):
        st.markdown("""
        ### 遥感干旱指数一览

        | 指数 | 全称 | 所需数据 | 范围 | 干旱判据 |
        |------|------|----------|------|----------|
        | **VCI** | 植被状态指数 | NDVI 时序 | 0-100 | <35 干旱 |
        | **NDDI** | 归一化干旱指数 | NDVI + NDWI | -1~1 | >0.5 干旱 |
        | **NDVI距平** | 标准化距平 | NDVI 时序 | 无界 | <-1 偏干 |
        | **TVDI** | 温度植被干旱指数 | NDVI + LST | 0-1 | >0.6 干旱 |

        ### 高级分析模块

        | 模块 | 功能 | 方法 |
        |------|------|------|
        | **🔮 干旱预测** | NDVI 趋势预测 | SARIMA / LSTM / Holt-Winters |
        | **🏜️ 沙漠化** | 沙漠化等级评估 | Albedo / TGSI / NDMI / DDI + 特征空间法 |

        ### 使用流程
        1. 选择研究区和时间范围 → 2. 勾选指数和分析模块
        3. 搜索影像，勾选至少 3 景 → 4. 查看指数结果
        5. 运行干旱预测 (可选) → 6. 评估沙漠化 (可选)
        7. 导出 CSV 统计结果
        """)
