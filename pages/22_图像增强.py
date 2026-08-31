"""
图像增强与变换页面 — 第 22 模块
==================================
专业遥感平台核心图像处理能力:

  - PCA 主成分分析: 波段去相关 + 信息浓缩 + 前3分量RGB合成
  - 空间滤波: 均值/中值/高斯/锐化/边缘检测
  - 对比度增强: 百分比拉伸/直方图均衡化/伽马校正/CLAHE
  - IHS 融合: 全色锐化多光谱

支持: 上传 GeoTIFF / 使用数据浏览页共享的影像
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="图像增强与变换", page_icon="🎨", layout="wide")

st.title("🎨 图像增强与变换")
st.markdown(
    "**PCA 主成分分析 · 空间滤波 · 对比度增强 · IHS 融合** — "
    "专业遥感平台的影像处理核心能力"
)

# ============================================================
# 数据加载
# ============================================================
st.sidebar.title("🎨 图像处理设置")

with st.sidebar:
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
            geotiff_path = os.path.join(tmp_dir, f"imgproc_{uploaded.name}")
            with open(geotiff_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"✅ 已加载: {uploaded.name}")
    else:
        # 从数据浏览页共享的搜索影像
        results = st.session_state.get("search_results", None)
        if results:
            st.caption(f"📡 使用数据浏览页的 {len(results)} 景影像")
        else:
            st.info("👈 请先在「数据浏览」页搜索影像")

    st.divider()

    # 处理模式选择
    st.subheader("🧰 处理功能")
    proc_mode = st.radio(
        "选择处理",
        ["📊 PCA 主成分分析", "🌀 空间滤波", "☀️ 对比度增强", "🔗 IHS 融合"],
    )

if geotiff_path is None and "📤" not in data_source:
    # 共享影像模式: 尝试从 session 取
    pass

# ============================================================
# 主面板
# ============================================================

def load_bands(path: str) -> np.ndarray:
    """读取 GeoTIFF 波段数组。"""
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float64)


def load_rgb(geotiff_path: str, band_order=None):
    """读取 RGB 显示数组 (0-1)。"""
    import rasterio
    with rasterio.open(geotiff_path) as src:
        n = src.count
        if band_order is None:
            # 默认: B=1, G=2, R=3 (或按可用波段)
            b_idx, g_idx, r_idx = 1, 2, 3
            if n >= 6:  # 标准6波段: B,G,R,NIR,SWIR1,SWIR2
                b_idx, g_idx, r_idx = 1, 2, 3
            elif n == 4:  # R,G,B,NIR
                r_idx, g_idx, b_idx = 1, 2, 3
        else:
            r_idx, g_idx, b_idx = band_order
        red = src.read(r_idx).astype(np.float64)
        green = src.read(g_idx).astype(np.float64)
        blue = src.read(b_idx).astype(np.float64)
    if np.nanmedian(red) > 10:
        red /= 10000.0
        green /= 10000.0
        blue /= 10000.0
    return np.stack([red, green, blue], axis=-1)


# ---- 数据准备 ----
if geotiff_path:
    bands = load_bands(geotiff_path)
    st.success(f"✅ 已加载 {bands.shape[0]} 波段影像 ({bands.shape[1]}×{bands.shape[2]})")
    st.divider()

    # 原始 RGB 参考
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.subheader("📷 原始影像")
        try:
            rgb = load_rgb(geotiff_path)
            rgb_show = np.clip(rgb, 0, 1)
            st.image((rgb_show * 255).astype(np.uint8), caption="RGB 真彩色")
        except Exception as e:
            st.warning(f"RGB 预览不可用: {e}")

    with col_proc:
        # ============================================
        # PCA 主成分分析
        # ============================================
        if "PCA" in proc_mode:
            st.subheader("📊 PCA 主成分分析")
            from utils.image_processing import pca_transform, pca_rgb_composite

            n_comp = st.slider("保留分量数", 1, min(bands.shape[0], 6), min(3, bands.shape[0]))
            if st.button("🔬 执行 PCA", type="primary"):
                with st.spinner("计算主成分..."):
                    pca = pca_transform(bands, n_components=n_comp)

                # 贡献率
                ratios = pca["explained_variance_ratio"]
                cum = np.cumsum(ratios)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(1, len(ratios) + 1), ratios, color="#1f77b4", label="单分量")
                ax.plot(range(1, len(ratios) + 1), cum, "ro-", label="累计")
                ax.set_xlabel("主成分")
                ax.set_ylabel("方差贡献率")
                ax.set_title(f"前3分量累计贡献率: {cum[:3].sum()*100:.1f}%")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    f"💡 前 3 个主成分集中了 **{cum[:3].sum()*100:.1f}%** 的波段信息"
                    "（>90% 说明信息浓缩有效，可用于分类前降维）"
                )

                # 前3分量 RGB 合成
                if n_comp >= 3:
                    comp_rgb = pca_rgb_composite(pca)
                    st.image(comp_rgb, caption="前 3 主成分 RGB 合成（信息最丰富）")

        # ============================================
        # 空间滤波
        # ============================================
        elif "滤波" in proc_mode:
            st.subheader("🌀 空间滤波")
            from utils.image_processing import spatial_filter

            filter_type = st.selectbox(
                "滤波类型",
                ["mean", "median", "gaussian", "sharpen", "edge"],
                format_func=lambda x: {
                    "mean": "均值滤波 (平滑)",
                    "median": "中值滤波 (去椒盐噪声)",
                    "gaussian": "高斯滤波 (平滑)",
                    "sharpen": "锐化 (增强细节)",
                    "edge": "边缘检测 (Sobel)",
                }[x],
            )
            kernel = st.slider("窗口大小", 3, 9, 3, step=2)
            band_sel = st.selectbox("处理波段", list(range(1, bands.shape[0] + 1)),
                                    format_func=lambda b: f"波段 {b}")
            if st.button("🌀 执行滤波", type="primary"):
                with st.spinner("滤波中..."):
                    out = spatial_filter(bands[band_sel - 1], filter_type, kernel)
                # 显示增强对比
                show = contrast_enhance(bands[band_sel - 1], "percentile")
                show_out = contrast_enhance(out, "percentile")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(show, cmap="gray")
                axes[0].set_title("原始")
                axes[0].axis("off")
                axes[1].imshow(show_out, cmap="gray")
                axes[1].set_title(f"{filter_type} (窗口{kernel})")
                axes[1].axis("off")
                st.pyplot(fig)
                plt.close(fig)

        # ============================================
        # 对比度增强
        # ============================================
        elif "对比度" in proc_mode:
            st.subheader("☀️ 对比度增强")
            from utils.image_processing import contrast_enhance

            method = st.selectbox(
                "增强方法",
                ["percentile", "hist_eq", "gamma", "clahe"],
                format_func=lambda x: {
                    "percentile": "百分比截断拉伸 (2%-98%)",
                    "hist_eq": "直方图均衡化",
                    "gamma": "伽马校正",
                    "clahe": "CLAHE 自适应均衡化",
                }[x],
            )
            if method == "percentile":
                pct = st.slider("截断百分位", 0.5, 10.0, 2.0)
            elif method == "gamma":
                gamma = st.slider("伽马值 (<1 变亮, >1 变暗)", 0.3, 3.0, 1.0, 0.1)
            band_sel = st.selectbox("处理波段", list(range(1, bands.shape[0] + 1)),
                                    format_func=lambda b: f"波段 {b}", key="ce_band")
            if st.button("☀️ 执行增强", type="primary"):
                with st.spinner("增强中..."):
                    if method == "percentile":
                        out = contrast_enhance(bands[band_sel - 1], method, percentile=pct)
                    elif method == "gamma":
                        out = contrast_enhance(bands[band_sel - 1], method, gamma=gamma)
                    else:
                        out = contrast_enhance(bands[band_sel - 1], method)
                orig = contrast_enhance(bands[band_sel - 1], "percentile")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(orig, cmap="gray")
                axes[0].set_title("原始")
                axes[0].axis("off")
                axes[1].imshow(out, cmap="gray")
                axes[1].set_title(f"增强: {method}")
                axes[1].axis("off")
                st.pyplot(fig)
                plt.close(fig)

                # 直方图对比
                fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3))
                axes2[0].hist(orig.flatten(), bins=50, color="#1f77b4", alpha=0.7)
                axes2[0].set_title("原始直方图")
                axes2[1].hist(out.flatten(), bins=50, color="#2ca02c", alpha=0.7)
                axes2[1].set_title("增强后直方图")
                st.pyplot(fig2)
                plt.close(fig2)

        # ============================================
        # IHS 融合
        # ============================================
        elif "IHS" in proc_mode:
            st.subheader("🔗 IHS 图像融合 (全色锐化)")
            from utils.image_processing import ihs_fusion

            if bands.shape[0] < 4:
                st.warning("IHS 融合需要至少 4 波段 (3 个 RGB + 1 个全色/NIR)")
            else:
                strength = st.slider("融合强度", 0.0, 1.0, 0.5, 0.05)
                pan_band = st.selectbox("全色波段 (用于锐化)", list(range(1, bands.shape[0] + 1)),
                                        index=3, format_func=lambda b: f"波段 {b}")
                if st.button("🔗 执行融合", type="primary"):
                    with st.spinner("融合中..."):
                        rgb = load_rgb(geotiff_path)
                        pan = bands[pan_band - 1]
                        fused = ihs_fusion(rgb, pan, pan_strength=strength)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(np.clip(rgb, 0, 1))
                    axes[0].set_title("原始 RGB")
                    axes[0].axis("off")
                    axes[1].imshow(fused)
                    axes[1].set_title(f"IHS 融合 (强度 {strength})")
                    axes[1].axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

else:
    st.info("👈 请在左侧上传 GeoTIFF 影像开始处理")
    with st.expander("📖 功能说明"):
        st.markdown("""
        ### 📊 PCA 主成分分析
        通过协方差矩阵特征分解，将相关的多波段转换为互不相关的主成分。
        前 3 个分量通常集中 90%+ 的信息，可用于分类前降维和数据压缩。

        ### 🌀 空间滤波
        - **均值/高斯**: 平滑噪声，突出区域特征
        - **中值**: 去除椒盐噪声，保留边缘
        - **锐化**: 增强地物边界细节
        - **边缘检测**: Sobel 算子提取地物轮廓

        ### ☀️ 对比度增强
        - **百分比拉伸**: 截断异常值，突出主要地物差异
        - **直方图均衡化**: 拉伸直方图分布，增强对比
        - **伽马校正**: 调整影像亮度
        - **CLAHE**: 自适应局部均衡化，避免过度增强

        ### 🔗 IHS 融合
        将 RGB 转换到亮度-色调-饱和度空间，用高分辨率全色波段替换亮度，
        再转回 RGB —— 经典的全色锐化方法。
        """)
