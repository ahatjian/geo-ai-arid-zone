"""
辐射定标与大气校正页面 — 第 23 模块
======================================
遥感预处理链补全: DOS 暗像元法大气校正 + 辐射定标。

功能:
  - DOS 暗像元法大气校正 (Chavez 1988/1996): 自动估计大气路径辐射并去除
  - 辐射定标: DN → 辐亮度 → TOA 反射率 (用户输入定标参数)
  - 相对辐射归一化: 多时相影像辐射水平匹配
  - 校正前后对比 + 质量评估报告
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="辐射定标与大气校正", page_icon="🌤️", layout="wide")

st.title("🌤️ 辐射定标与大气校正")
st.markdown(
    "**DOS 暗像元法 · 辐射定标 · 相对归一化** — 遥感影像预处理链的核心环节"
)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🌤️ 预处理设置")

    st.subheader("📁 数据源")
    data_source = st.radio(
        "影像来源",
        ["📤 上传 GeoTIFF", "🗺️ 数据浏览页共享影像"],
    )

    geotiff_path = None
    if "📤" in data_source:
        uploaded = st.file_uploader("上传多波段 GeoTIFF", type=["tif", "tiff"])
        if uploaded:
            tmp_dir = tempfile.gettempdir()
            geotiff_path = os.path.join(tmp_dir, f"atmo_{uploaded.name}")
            with open(geotiff_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"✅ 已加载: {uploaded.name}")
    else:
        results = st.session_state.get("search_results", None)
        if results:
            st.caption(f"📡 使用数据浏览页的 {len(results)} 景影像")
        else:
            st.info("👈 请先在「数据浏览」页搜索影像")

    st.divider()

    st.subheader("🧪 处理选项")
    do_dos = st.toggle("DOS 暗像元法大气校正", value=True,
                       help="自动估计每波段大气路径辐射并去除 (无需大气参数)")
    dark_pct = st.slider("暗像元百分位 (%)", 0.1, 5.0, 1.0, 0.1,
                         disabled=not do_dos)
    stretch = st.toggle("校正后拉伸到 0-1", value=True,
                        help="校正后重新归一化反射率范围")

# ============================================================
# 主面板
# ============================================================

def load_bands(path: str) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float64)


if geotiff_path:
    bands = load_bands(geotiff_path)
    st.success(f"✅ 已加载 {bands.shape[0]} 波段 ({bands.shape[1]}×{bands.shape[2]})")
    st.divider()

    tab_dos, tab_cal, tab_norm = st.tabs(
        ["🌤️ DOS 大气校正", "📐 辐射定标", "🔗 相对归一化"]
    )

    # ============================================
    # Tab 1: DOS 大气校正
    # ============================================
    with tab_dos:
        st.subheader("🌤️ DOS 暗像元法大气校正")
        st.caption(
            "原理: 假设影像含地表反射率≈0 的暗像元（深水/阴影），"
            "其观测值 = 大气路径辐射，逐波段减去即可去除大气散射影响"
        )

        if st.button("🌤️ 执行大气校正", type="primary"):
            with st.spinner("大气校正中..."):
                from utils.atmospheric import dos_correction, dos_quality_report
                result = dos_correction(bands, dark_percentile=dark_pct,
                                        stretch_after=stretch)

            corrected = result["corrected"]

            # 质量报告
            st.subheader("📋 校正质量报告")
            band_names = [f"波段{i+1}" for i in range(bands.shape[0])]
            report = dos_quality_report(bands, corrected, band_names)
            st.dataframe(report, width="stretch", hide_index=True)

            # 大气路径辐射可视化
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(band_names, result["dark_values"],
                   color=[plt.cm.viridis(i / max(len(band_names) - 1, 1))
                          for i in range(len(band_names))])
            ax.set_ylabel("大气路径辐射估计")
            ax.set_title("各波段大气路径辐射 (DOS 估计)")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            # 前后对比 (取前 3 波段 RGB)
            if bands.shape[0] >= 3:
                st.subheader("👁️ 校正前后对比")
                def _display(arr):
                    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)  # R,G,B
                    if np.nanmedian(rgb) > 10:
                        rgb = rgb / 10000.0
                    rgb = np.clip(rgb, 0, 1)
                    # 百分比拉伸
                    for c in range(3):
                        ch = rgb[:, :, c]
                        lo = np.nanpercentile(ch, 2)
                        hi = np.nanpercentile(ch, 98)
                        if hi > lo:
                            rgb[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
                    return (rgb * 255).astype(np.uint8)

                fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].imshow(_display(bands))
                axes[0].set_title("校正前")
                axes[0].axis("off")
                axes[1].imshow(_display(corrected))
                axes[1].set_title("DOS 校正后")
                axes[1].axis("off")
                st.pyplot(fig2)
                plt.close(fig2)

            # 导出
            st.divider()
            st.subheader("💾 导出")
            out_tif = os.path.join(tempfile.gettempdir(), "dos_corrected.tif")
            import rasterio
            with rasterio.open(geotiff_path) as src:
                meta = src.meta.copy()
            meta.update(dtype="float64", count=corrected.shape[0])
            with rasterio.open(out_tif, "w", **meta) as dst:
                dst.write(corrected)
            with open(out_tif, "rb") as f:
                st.download_button("⬇️ 校正后 GeoTIFF", f,
                                   file_name="dos_corrected.tif",
                                   mime="image/tiff", width="stretch")

            # 存 session 供后续页面使用
            st.session_state["dos_corrected_bands"] = corrected
            st.session_state["dos_dark_values"] = result["dark_values"]

            # 一键保存到数据下载中心 (持久化闭环)
            st.divider()
            st.subheader("💾 保存到数据下载中心")
            from utils.save_ui import render_save_button
            render_save_button(
                default_name=f"DOS大气校正结果_{len(corrected)}波段",
                data=corrected,
                kind="npy",
                meta={
                    "模块": "辐射定标与大气校正",
                    "暗像元百分位": dark_pct,
                    "大气路径辐射": [round(float(v), 4) for v in result["dark_values"]],
                    "说明": "DOS 暗像元法校正后的反射率 (0-1)",
                },
                key_suffix="dos_result",
            )

    # ============================================
    # Tab 2: 辐射定标
    # ============================================
    with tab_cal:
        st.subheader("📐 辐射定标 (DN → TOA 反射率)")
        st.caption(
            "将原始 DN 值转换为 TOA 反射率。适用于 L1 级数据。"
            "Sentinel-2 L2A / Landsat C2 L2 已是地表反射率，无需此步骤。"
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            solar_zenith = st.number_input("太阳天顶角 θz (°)", 0.0, 90.0, 30.0,
                                           help="Landsat 元数据 SUN_ELEVATION, θz=90-高度角")
        with col_c2:
            d_es = st.number_input("日地距离 (AU)", 0.9, 1.1, 1.0, 0.001,
                                   help="Landsat 元数据 EARTH_SUN_DISTANCE")

        st.markdown("**定标参数 (每波段 gain/bias/ESUN)**")
        st.caption("Landsat 8: 默认反射率系数 (REFLECTANCE_MULT≈2e-5, ADD≈-0.1)")
        use_default = st.checkbox("使用 Landsat 8 默认反射率系数", value=True)

        if st.button("📐 执行辐射定标", type="primary"):
            with st.spinner("辐射定标中..."):
                from utils.atmospheric import toa_reflectance
                n = bands.shape[0]
                if use_default:
                    mult = [2e-5] * n
                    add = [-0.1] * n
                else:
                    mult_input = st.text_input("每波段 Mρ (逗号分隔)", "2e-5,2e-5,2e-5,2e-5,2e-5,2e-5")
                    add_input = st.text_input("每波段 Aρ (逗号分隔)", "-0.1,-0.1,-0.1,-0.1,-0.1,-0.1")
                    mult = [float(x) for x in mult_input.split(",")]
                    add = [float(x) for x in add_input.split(",")]

                toa = toa_reflectance(bands, mult, add, solar_zenith)

            st.success(f"✅ 辐射定标完成 — 反射率范围 {toa.min():.4f} ~ {toa.max():.4f}")
            st.info("📌 下一步建议: 在「DOS 大气校正」Tab 中对该结果执行大气校正")

            # 导出
            out_tif2 = os.path.join(tempfile.gettempdir(), "toa_reflectance.tif")
            import rasterio
            with rasterio.open(geotiff_path) as src:
                meta = src.meta.copy()
            meta.update(dtype="float64", count=toa.shape[0])
            with rasterio.open(out_tif2, "w", **meta) as dst:
                dst.write(toa)
            with open(out_tif2, "rb") as f:
                st.download_button("⬇️ TOA 反射率 GeoTIFF", f,
                                   file_name="toa_reflectance.tif",
                                   mime="image/tiff", width="stretch")
            st.session_state["toa_bands"] = toa

    # ============================================
    # Tab 3: 相对归一化
    # ============================================
    with tab_norm:
        st.subheader("🔗 相对辐射归一化")
        st.caption(
            "将目标影像的辐射水平匹配到参考影像，消除多时相间的大气/光照差异，"
            "常用于时序分析前的数据一致性处理。"
        )

        ref_upload = st.file_uploader("上传参考影像 GeoTIFF (同波段数)", type=["tif", "tiff"],
                                      key="norm_ref")
        norm_method = st.selectbox("归一化方法", ["linear", "hist_match"],
                                   format_func=lambda x: {
                                       "linear": "线性回归匹配 (推荐)",
                                       "hist_match": "直方图匹配",
                                   }[x])

        if ref_upload:
            ref_tmp = os.path.join(tempfile.gettempdir(), f"norm_ref_{ref_upload.name}")
            with open(ref_tmp, "wb") as f:
                f.write(ref_upload.getvalue())
            ref_bands = load_bands(ref_tmp)

            if ref_bands.shape != bands.shape:
                st.warning(f"⚠️ 波段数/尺寸不一致: 目标 {bands.shape} vs 参考 {ref_bands.shape}")
            elif st.button("🔗 执行归一化", type="primary"):
                with st.spinner("归一化中..."):
                    from utils.atmospheric import relative_normalization
                    normalized = relative_normalization(bands, ref_bands, norm_method)
                st.success(f"✅ 归一化完成 ({norm_method})")
                st.caption(f"目标均值: {np.nanmean(bands):.4f} → 归一化后: {np.nanmean(normalized):.4f} (参考: {np.nanmean(ref_bands):.4f})")
                st.session_state["normalized_bands"] = normalized

else:
    st.info("👈 请在左侧上传 GeoTIFF 影像开始预处理")
    with st.expander("📖 原理说明"):
        st.markdown("""
        ### 🌤️ DOS 暗像元法 (Chavez 1988/1996)
        大气校正的经典轻量方法：
        1. 假设影像中存在地表反射率≈0 的暗像元（深水、浓密阴影）
        2. 暗像元的观测值完全来自大气路径辐射
        3. 对每个波段，取直方图低端 1% 分位作为暗像元值并减去
        - 优点: 无需大气参数（气溶胶、水汽等），适用任何传感器
        - 局限: 假设影像确实存在暗像元（干旱区需注意水域/阴影）

        ### 📐 辐射定标
        - 辐亮度: L = gain × DN + bias
        - TOA 反射率: ρ = π × L × d² / (ESUN × cosθz)
        - Landsat C2 简化: ρ = (Mρ × DN + Aρ) / cosθz

        ### 🔗 相对辐射归一化
        - 线性回归: 用最小二乘拟合目标↔参考波段关系 y=ax+b
        - 直方图匹配: 使目标影像直方图分布与参考一致
        """)
