"""
植被分析页面 — NDVI / EVI 计算 + 植被覆盖统计 + Sen+MK 趋势分析
支持单/多时相 GeoTIFF 输入
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
from utils.error_handler import StreamlitErrorBoundary

st.set_page_config(page_title="植被分析", page_icon="🌿", layout="wide")

# ============================================
# 侧边栏
# ============================================
with st.sidebar:
    st.title("🌿 植被分析设置")

    st.subheader("数据源")
    data_mode = st.radio(
        "数据来源",
        ["📁 上传单景 GeoTIFF (单时相)", "📁 上传多景 GeoTIFF (多时相趋势)"],
    )

    st.divider()

    st.subheader("植被指数")
    index_type = st.radio(
        "选择指数",
        ["NDVI (推荐)", "EVI"],
        help="NDVI 适合干旱区，EVI 可避免植被饱和"
    )

    st.divider()

    st.subheader("像元参数")
    pixel_size = st.number_input(
        "像元尺寸 (米)",
        value=10.0, min_value=1.0, max_value=100.0,
    )

# ============================================
# 主页面
# ============================================
st.title("🌿 植被覆盖分析")
st.markdown("NDVI / EVI 自动计算 + 植被覆盖统计 + Sen+MK 趋势分析")

# ============================================
# 数据加载
# ============================================
if "单景" in data_mode:
    st.subheader("📁 上传 GeoTIFF")

    uploaded = st.file_uploader(
        "上传多波段 GeoTIFF (至少包含 Red, NIR 波段)",
        type=["tif", "tiff"],
        key="veg_single",
    )

    if uploaded:
        tmp_dir = tempfile.gettempdir()
        geotiff_path = os.path.join(tmp_dir, f"veg_{uploaded.name}")
        with open(geotiff_path, "wb") as f:
            f.write(uploaded.getvalue())
        st.success(f"✅ 已加载: {uploaded.name}")
    else:
        geotiff_path = None
        st.info("👆 请上传 GeoTIFF。推荐先在「数据浏览」页面下载全波段数据。")

else:
    st.subheader("📁 上传多景 GeoTIFF (时序分析)")

    uploaded_files = st.file_uploader(
        "上传多景多波段 GeoTIFF (按时间排序)",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="veg_multi",
    )

    if uploaded_files:
        st.success(f"✅ 已加载 {len(uploaded_files)} 景影像")
        st.info("请确保文件按时间顺序排列 (可拖动重排)")

        # 手动排序
        file_names = [f.name for f in uploaded_files]
        st.write("文件列表:")
        for i, name in enumerate(file_names):
            st.caption(f"  {i+1}. {name}")
    else:
        st.info("👆 上传多景不同时期的 GeoTIFF 进行趋势分析")

    uploaded_files_list = uploaded_files or []
    geotiff_path = None

# ============================================
# 单景分析
# ============================================
if "单景" in data_mode and geotiff_path:
    import rasterio

    with rasterio.open(geotiff_path) as src:
        n_bands = src.count
        img_shape = (src.height, src.width)

    st.divider()
    st.subheader("📊 影像信息")

    cols = st.columns(4)
    with cols[0]:
        st.metric("波段数", n_bands)
    with cols[1]:
        st.metric("尺寸", f"{img_shape[1]} × {img_shape[0]}")
    with cols[2]:
        st.metric("像元尺寸", f"{pixel_size:.0f}m")
    with cols[3]:
        area_km2 = img_shape[0] * img_shape[1] * pixel_size * pixel_size / 1e6
        st.metric("覆盖面积", f"{area_km2:.1f} km²")

    # 波段映射
    st.subheader("🔧 波段映射")

    col1, col2, col3 = st.columns(3)
    with col1:
        band_blue = st.number_input("Blue", 1, max(1, n_bands), 1)
        band_green = st.number_input("Green", 1, max(1, n_bands), 2)
    with col2:
        band_red = st.number_input("Red", 1, max(1, n_bands), 3)
        band_nir = st.number_input("NIR", 1, max(1, n_bands), 4)
    with col3:
        band_swir1 = st.number_input("SWIR1", 1, max(1, n_bands), 5)
        band_swir2 = st.number_input("SWIR2", 1, max(1, n_bands), 6)

    # 计算
    if st.button("🔬 计算植被指数", type="primary"):
        with st.spinner("正在计算植被指数..."):
            with StreamlitErrorBoundary("植被指数计算", st=st, show_traceback=True):
                from utils.indices import (
                    calc_ndvi,
                    calc_evi,
                    calc_vegetation_stats,
                    save_index_geotiff,
                )
                from utils.visualization import render_ndvi, render_evi, plot_histogram

                # 读取波段
                with rasterio.open(geotiff_path) as src:
                    red_arr = src.read(band_red).astype(np.float32)
                    nir_arr = src.read(band_nir).astype(np.float32)
                    if index_type == "EVI":
                        blue_arr = src.read(band_blue).astype(np.float32)

                # 计算指数
                if index_type == "NDVI (推荐)":
                    index_arr = calc_ndvi(red_arr, nir_arr)
                else:
                    index_arr = calc_evi(blue_arr, red_arr, nir_arr)

                # 统计
                stats = calc_vegetation_stats(index_arr, pixel_size_m=pixel_size)

                # 保存
                st.session_state["veg_index"] = index_arr
                st.session_state["veg_stats"] = stats
                st.session_state["veg_geotiff"] = geotiff_path
                st.session_state["veg_index_type"] = index_type

                # ===== 结果展示 =====
                st.divider()
                st.subheader("📊 植被覆盖统计")

                cols = st.columns(4)
                with cols[0]:
                    st.metric("平均 NDVI" if "NDVI" in index_type else "平均 EVI", f"{stats['mean']:.4f}")
                with cols[1]:
                    st.metric("最大", f"{stats['max']:.4f}")
                with cols[2]:
                    st.metric("密植被占比 (>0.6)", f"{stats['dense_veg_ratio']*100:.1f}%")
                with cols[3]:
                    st.metric("裸地占比 (<0.1)", f"{stats['bare_ratio']*100:.1f}%")

                cols2 = st.columns(2)
                with cols2[0]:
                    st.metric("稀疏植被占比 (0.2-0.6)", f"{stats['sparse_veg_ratio']*100:.1f}%")
                with cols2[1]:
                    st.metric("最小值", f"{stats['min']:.4f}")

                # 可视化
                st.divider()
                st.subheader("🗺️ 指数空间分布")

                viz_col1, viz_col2 = st.columns([3, 2])

                with viz_col1:
                    if "NDVI" in index_type:
                        img = render_ndvi(index_arr)
                    else:
                        img = render_evi(index_arr)
                    st.image(img)

                with viz_col2:
                    hist_fig = plot_histogram(
                        index_arr,
                        bins=80,
                        x_label=index_type.split(" ")[0],
                        title=f"{index_type.split(' ')[0]} 分布直方图",
                    )
                    st.plotly_chart(hist_fig)

                # 导出
                st.divider()
                st.subheader("💾 结果导出")

                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    index_name = "NDVI" if "NDVI" in index_type else "EVI"
                    tmp_dir = tempfile.gettempdir()
                    index_tif = os.path.join(tmp_dir, f"{index_name}.tif")
                    save_index_geotiff(index_arr.astype(np.float32), geotiff_path, index_tif)
                    with open(index_tif, "rb") as f:
                        st.download_button(
                            f"⬇️ {index_name} GeoTIFF",
                            f, file_name=f"{index_name}.tif",
                            mime="image/tiff",
                            width="stretch",
                        )

                with exp_col2:
                    csv_buf = BytesIO()
                    pd.DataFrame([stats]).to_csv(csv_buf, index=False, encoding="utf-8-sig")
                    csv_buf.seek(0)
                    st.download_button(
                        "⬇️ 统计结果 CSV",
                        csv_buf, file_name=f"{index_name}_stats.csv",
                        mime="text/csv",
                        width="stretch",
                    )

# ============================================
# 多时相趋势分析
# ============================================
elif "多景" in data_mode and len(uploaded_files_list) >= 2:
    import rasterio

    st.subheader("🔧 波段映射与日期标注")

    # 先检查首景波段数
    first_path = os.path.join(tempfile.gettempdir(), f"veg_multi_{uploaded_files_list[0].name}")
    with open(first_path, "wb") as f:
        f.write(uploaded_files_list[0].getvalue())

    with rasterio.open(first_path) as src:
        n_bands = src.count

    col1, col2 = st.columns(2)
    with col1:
        band_red_idx = st.number_input("Red 波段索引", 1, max(1, n_bands), 3, key="mt_red")
        band_nir_idx = st.number_input("NIR 波段索引", 1, max(1, n_bands), 4, key="mt_nir")
    with col2:
        if "EVI" in index_type:
            band_blue_idx = st.number_input("Blue 波段索引", 1, max(1, n_bands), 1, key="mt_blue")

    # 日期输入
    st.markdown("**为每景影像标注日期** (用于趋势分析)")
    dates = []
    for i, f in enumerate(uploaded_files_list):
        default_date = pd.Timestamp(f"2024-{((i*3)%12)+1:02d}-01")
        d = st.date_input(f"{i+1}. {f.name}", value=default_date, key=f"date_{i}")
        dates.append(d.strftime("%Y-%m-%d"))

    # 执行趋势分析
    if st.button("📈 执行时序趋势分析", type="primary"):
        with st.spinner("正在处理多时相影像..."):
            with StreamlitErrorBoundary("Sen+MK 趋势分析", st=st, show_traceback=True):
                from utils.indices import calc_ndvi, calc_evi
                from utils.visualization import (
                    plot_time_series,
                    plot_trend_scatter,
                    plot_histogram,
                )

                # 读取所有影像并计算指数
                mean_values = []
                ndvi_arrays = []
                for i, uf in enumerate(uploaded_files_list):
                    tmp_path = os.path.join(tempfile.gettempdir(), f"veg_multi_{uf.name}")
                    with open(tmp_path, "wb") as f_write:
                        f_write.write(uf.getvalue())

                    with rasterio.open(tmp_path) as src:
                        red_arr = src.read(band_red_idx).astype(np.float32)
                        nir_arr = src.read(band_nir_idx).astype(np.float32)

                    if "NDVI" in index_type:
                        ndvi_arr = calc_ndvi(red_arr, nir_arr)
                    else:
                        with rasterio.open(tmp_path) as src:
                            blue_arr = src.read(band_blue_idx).astype(np.float32)
                        ndvi_arr = calc_evi(blue_arr, red_arr, nir_arr)

                    mean_val = float(np.nanmean(ndvi_arr))
                    mean_values.append(mean_val)
                    ndvi_arrays.append(ndvi_arr)

                # ===== 月度合成 (MVC) =====
                st.markdown("**🗓️ 月度合成 (MVC 最大值合成)**")
                m_col1, m_col2 = st.columns([1, 3])
                with m_col1:
                    use_composite = st.toggle("启用月度合成", value=True,
                                              help="按月份取 NDVI 最大值, 消除云噪声和观测缺失")
                if use_composite:
                    from utils.composite import composite_series_by_month
                    monthly_means, month_labels = composite_series_by_month(
                        ndvi_arrays, dates, method="max"
                    )
                    st.caption(
                        f"✅ 已合成 {len(ndvi_arrays)} 景 → {len(month_labels)} 个月 "
                        f"({'、'.join(month_labels)})"
                    )
                    # 用月度合成均值作为趋势分析输入
                    y_original = monthly_means
                    composite_dates = month_labels
                else:
                    y_original = np.array(mean_values)
                    composite_dates = dates

                # ===== Savitzky-Golay 平滑 (可选) =====
                st.markdown("**🛰️ 时序平滑 (Savitzky-Golay)**")
                s_col1, s_col2, s_col3 = st.columns([1, 1, 2])
                with s_col1:
                    use_smoothing = st.toggle("启用 S-G 平滑", value=True,
                                              help="去除传感器噪声和云污染残留，保留季节趋势")
                with s_col2:
                    sg_window = st.number_input("窗口", 3, 15, 5, step=2,
                                                help="窗口越大越平滑")
                y = y_original
                if use_smoothing:
                    from utils.trend import savgol_smooth
                    y = savgol_smooth(y_original, window=int(sg_window), polyorder=2)
                    st.caption(f"✅ 已平滑 (窗口={int(sg_window)}, 多项式=2) — 原始 vs 平滑 R²={np.corrcoef(y_original, y)[0,1]:.4f}")

                # ===== Sen+MK 趋势分析 =====
                from scipy.stats import theilslopes
                try:
                    import pymannkendall as mk
                except ImportError:
                    st.warning("pymannkendall 未安装，仅使用 Sen 斜率。安装: pip install pymannkendall")
                    mk = None

                x = np.arange(len(composite_dates))

                # Sen 斜率
                slope, intercept, lo_slope, up_slope = theilslopes(y, x, 0.95)

                # Mann-Kendall 检验
                if mk:
                    mk_result = mk.original_test(y)
                    trend = mk_result.trend
                    p_value = mk_result.p
                else:
                    p_value = 0.5
                    trend = "increasing" if slope > 0 else "decreasing"

                # ===== 结果展示 =====
                st.divider()
                st.subheader("📊 趋势分析结果")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sen 斜率", f"{slope:.6f}/期")
                with col2:
                    trend_label = {"increasing": "📈 增加", "decreasing": "📉 减少", "no trend": "➖ 无趋势"}
                    st.metric("趋势方向", trend_label.get(trend, trend))
                with col3:
                    if mk:
                        sig = "显著" if p_value < 0.05 else "不显著"
                        st.metric("显著性 (p)", f"{p_value:.4f} ({sig})")
                    else:
                        st.metric("显著性", "未检测")
                with col4:
                    st.metric("影像数", len(mean_values))

                # 时序曲线 (原始 vs 平滑)
                st.divider()
                st.subheader("📈 时序变化曲线")

                index_label = "NDVI" if "NDVI" in index_type else "EVI"
                if use_smoothing:
                    import plotly.graph_objects as go
                    fig1 = plot_time_series(
                        composite_dates, y_original,
                        y_label=f"平均 {index_label}",
                        title=f"{index_label} 时序变化 — 原始 (n={len(mean_values)})",
                    )
                    fig1.add_trace(
                        go.Scatter(x=composite_dates, y=y, mode="lines+markers",
                                   name="S-G 平滑",
                                   line=dict(width=3, color="#e67e22"))
                    )
                else:
                    fig1 = plot_time_series(
                        composite_dates, y,
                        y_label=f"平均 {index_label}",
                        title=f"{index_label} 时序变化 (n={len(mean_values)})",
                    )
                st.plotly_chart(fig1)

                # 趋势散点
                st.subheader("📉 Sen + Mann-Kendall 趋势检验")

                fig2 = plot_trend_scatter(
                    list(x), list(y),
                    slope=slope,
                    p_value=p_value,
                    trend=trend,
                    x_label="影像序号",
                    y_label=f"平均 {index_label}",
                )
                st.plotly_chart(fig2)

                # 导出
                st.divider()
                st.subheader("💾 结果导出")

                results_df = pd.DataFrame({
                    "日期": composite_dates,
                    f"平均{index_label}": y_original,
                })
                if use_smoothing:
                    results_df[f"{index_label} S-G平滑"] = y
                results_df.loc["趋势"] = [
                    f"斜率={slope:.6f}, p={p_value:.4f}",
                    "",
                ]

                csv_buf = BytesIO()
                results_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                csv_buf.seek(0)
                st.download_button(
                    "⬇️ 时序数据 + 趋势结果 CSV",
                    csv_buf,
                    file_name=f"{index_label}_trend_analysis.csv",
                    mime="text/csv",
                )

else:
    # 显示说明
    st.divider()
    st.subheader("📖 植被指数说明")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### NDVI 归一化植被指数

        $$
        NDVI = \\frac{NIR - Red}{NIR + Red}
        $$

        | NDVI 范围 | 含义 |
        |-----------|------|
        | > 0.6 | 茂密植被 |
        | 0.2 - 0.6 | 草地/农田 |
        | 0.1 - 0.2 | 稀疏植被 |
        | < 0.1 | 裸地/水体 |

        **优点**: 简单稳定，广泛使用
        """)

    with col2:
        st.markdown("""
        ### EVI 增强植被指数

        $$
        EVI = 2.5 \\times \\frac{NIR - Red}{NIR + 6Red - 7.5Blue + 1}
        $$

        **优点**:
        - 不易饱和 (高覆盖区更准确)
        - 受土壤背景影响小
        - 受大气影响小

        **适用**: 干旱区植被茂密绿洲
        """)

    st.info("""
    **操作步骤**:
    1. 在「数据浏览」页面下载 GeoTIFF
    2. **单景**: 上传 1 个文件，计算 NDVI/EVI 空间分布
    3. **多景**: 上传多期影像，进行 Sen+MK 趋势分析
    4. 查看结果 → 导出 GeoTIFF / CSV
    """)
