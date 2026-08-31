"""
图像增强与变换模块 — 专业遥感平台核心图像处理能力
====================================================
提供 ENVI 等专业平台的常用图像处理功能:

  - PCA 主成分分析: 波段去相关、信息浓缩 (前 3 分量 RGB 合成)
  - 空间滤波: 均值/中值/高斯/锐化/边缘检测
  - 对比度增强: 线性拉伸/百分比截断/直方图均衡化/CLAHE
  - 图像融合 (简化 IHS 融合): 高分辨率全色 + 多光谱

适用于:
  - 数据浏览页的影像增强展示
  - 分类前预处理 (PCA 降维)
  - 视觉解译辅助
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# PCA 主成分分析
# ============================================================

def pca_transform(
    bands_data: np.ndarray,
    n_components: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    对多波段影像做主成分分析 (PCA)。

    原理: 通过协方差矩阵特征分解, 将相关波段转换为互不相关的
          主成分, 前几个分量集中了绝大部分信息。

    参数:
        bands_data: (B, H, W) 多波段数组
        n_components: 保留分量数 (None=全部)

    返回:
        dict: {
            "pca": (K, H, W) 主成分图像,
            "explained_variance_ratio": (K,) 各分量方差贡献率,
            "eigenvalues": (B,) 特征值,
            "components": (K, B) 特征向量 (载荷),
            "means": (B,) 各波段均值
        }
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.ndim != 3:
        raise ValueError(f"需要 3D 数组 (B,H,W), 实际: {bands_data.ndim}D")
    B, H, W = bands_data.shape

    # 展平为 (B, N)
    flat = bands_data.reshape(B, -1)
    # 处理 NaN: 用波段均值填充
    means = np.nanmean(flat, axis=1)
    nan_mask = ~np.isfinite(flat)
    if nan_mask.any():
        flat = flat.copy()
        flat[nan_mask] = np.repeat(means[:, None], flat.shape[1], axis=1)[nan_mask]

    # 中心化
    centered = flat - means[:, None]

    # 协方差矩阵特征分解 (B×B, 比 N×N 高效)
    cov = np.cov(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 降序排列
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if n_components is None:
        n_components = B
    n_components = min(n_components, B)

    # 投影
    comps = eigenvectors[:, :n_components]
    pca = comps.T @ centered  # (K, N)

    # 方差贡献率
    total_var = np.sum(eigenvalues)
    explained = eigenvalues[:n_components] / max(total_var, 1e-12)

    return {
        "pca": pca.reshape(n_components, H, W),
        "explained_variance_ratio": explained,
        "eigenvalues": eigenvalues,
        "components": comps.T,
        "means": means,
    }


def pca_rgb_composite(
    pca_result: Dict[str, np.ndarray],
    stretch_percentile: float = 2.0,
) -> np.ndarray:
    """
    将前 3 个主成分合成为 RGB 显示图 (uint8)。

    参数:
        pca_result: pca_transform 的返回
        stretch_percentile: 截断百分位数 (百分比拉伸)

    返回:
        (H, W, 3) uint8 RGB 合成图
    """
    pca = pca_result["pca"]
    n = min(3, pca.shape[0])
    rgb = np.zeros((pca.shape[1], pca.shape[2], 3), dtype=np.float64)
    for i in range(n):
        comp = pca[i]
        lo = np.nanpercentile(comp, stretch_percentile)
        hi = np.nanpercentile(comp, 100 - stretch_percentile)
        if hi <= lo:
            hi = lo + 1e-6
        rgb[:, :, i] = np.clip((comp - lo) / (hi - lo), 0, 1) * 255
    return rgb.astype(np.uint8)


# ============================================================
# 空间滤波
# ============================================================

def spatial_filter(
    arr: np.ndarray,
    filter_type: str = "median",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    空间滤波 (均值/中值/高斯/锐化/边缘)。

    参数:
        arr: (H, W) 或 (B, H, W) 数组
        filter_type: "mean" | "median" | "gaussian" | "sharpen" | "edge"
        kernel_size: 滤波器窗口大小 (3/5/7...)

    返回:
        滤波后数组 (同输入形状)
    """
    from scipy import ndimage

    arr = np.asarray(arr, dtype=np.float64)
    kernel_size = max(3, int(kernel_size) | 1)  # 强制奇数

    def _apply(a: np.ndarray) -> np.ndarray:
        if filter_type == "mean":
            return ndimage.uniform_filter(a, size=kernel_size, mode="nearest")
        if filter_type == "median":
            return ndimage.median_filter(a, size=kernel_size, mode="nearest")
        if filter_type == "gaussian":
            sigma = kernel_size / 3.0
            return ndimage.gaussian_filter(a, sigma=sigma, mode="nearest")
        if filter_type == "sharpen":
            # 锐化 = 原图 + (原图 - 高斯模糊) × 强度
            sigma = kernel_size / 3.0
            blurred = ndimage.gaussian_filter(a, sigma=sigma, mode="nearest")
            return np.clip(a + 1.2 * (a - blurred), np.nanmin(a), np.nanmax(a))
        if filter_type == "edge":
            # Sobel 边缘强度
            sobel_x = ndimage.sobel(a, axis=0)
            sobel_y = ndimage.sobel(a, axis=1)
            return np.hypot(sobel_x, sobel_y)
        raise ValueError(f"未知滤波类型: {filter_type}")

    if arr.ndim == 2:
        return _apply(arr)
    return np.stack([_apply(b) for b in arr])


# ============================================================
# 对比度增强
# ============================================================

def contrast_enhance(
    arr: np.ndarray,
    method: str = "percentile",
    percentile: float = 2.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    对比度增强。

    参数:
        arr: (H, W) 或 (B, H, W) 数组
        method: "percentile" (百分比截断拉伸) | "hist_eq" (直方图均衡化)
                | "gamma" (伽马校正) | "clahe" (自适应直方图均衡化)
        percentile: 截断百分位 (percentile 方法)
        gamma: 伽马值 (gamma 方法, <1 变亮, >1 变暗)

    返回:
        (H, W, 3) 或单波段 uint8 增强图
    """
    arr = np.asarray(arr, dtype=np.float64)

    def _enhance_single(a: np.ndarray) -> np.ndarray:
        valid = a[np.isfinite(a)]
        if valid.size == 0:
            return np.zeros(a.shape, dtype=np.uint8)

        if method == "percentile":
            lo = np.nanpercentile(valid, percentile)
            hi = np.nanpercentile(valid, 100 - percentile)
            if hi <= lo:
                hi = lo + 1e-6
            out = np.clip((a - lo) / (hi - lo), 0, 1)
        elif method == "hist_eq":
            # 直方图均衡化 (256 bins)
            hist, bins = np.histogram(valid, bins=256, range=(np.nanmin(valid), np.nanmax(valid)))
            cdf = np.cumsum(hist) / max(hist.sum(), 1)
            out = np.interp(a, bins[:-1], cdf)
        elif method == "gamma":
            lo = np.nanmin(valid)
            hi = np.nanmax(valid)
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((a - lo) / (hi - lo), 0, 1)
            out = np.power(norm, gamma)
        elif method == "clahe":
            try:
                from skimage.exposure import equalize_adapthist
                out = equalize_adapthist(a, clip_limit=0.03)
            except ImportError:
                # 降级为普通直方图均衡化
                hist, bins = np.histogram(valid, bins=256)
                cdf = np.cumsum(hist) / max(hist.sum(), 1)
                out = np.interp(a, bins[:-1], cdf)
        else:
            raise ValueError(f"未知增强方法: {method}")

        out = np.clip(out, 0, 1)
        out[~np.isfinite(a)] = 0
        return (out * 255).astype(np.uint8)

    if arr.ndim == 2:
        return _enhance_single(arr)
    return np.stack([_enhance_single(b) for b in arr])


# ============================================================
# 简化 IHS 图像融合 (全色锐化)
# ============================================================

def ihs_fusion(
    ms_rgb: np.ndarray,
    pan: np.ndarray,
    pan_strength: float = 0.5,
) -> np.ndarray:
    """
    简化 IHS 融合: 用高分辨率全色波段锐化多光谱 RGB。

    原理: 将 RGB 转到亮度-色调-饱和度空间, 用全色替换亮度分量,
          再转回 RGB。

    参数:
        ms_rgb: (H, W, 3) 多光谱 RGB (float 0-1 或 uint8)
        pan: (H, W) 全色波段
        pan_strength: 融合强度 (0=不融合, 1=完全替换亮度)

    返回:
        (H, W, 3) 融合后 RGB (uint8)
    """
    ms = np.asarray(ms_rgb, dtype=np.float64)
    if ms.max() > 1.0:
        ms = ms / 255.0
    pan = np.asarray(pan, dtype=np.float64)
    if pan.max() > 1.0:
        pan = pan / 255.0

    # IHS 变换 (简化矩阵)
    r, g, b = ms[:, :, 0], ms[:, :, 1], ms[:, :, 2]
    intensity = (r + g + b) / 3.0
    # 饱和度 (简化为标准差的近似)
    saturation = 1.0 - 3.0 * np.minimum(np.minimum(r, g), b) / (r + g + b + 1e-6)
    # 色调 (简化)
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)

    # 用全色替换亮度
    pan_norm = (pan - np.nanmin(pan)) / (np.nanmax(pan) - np.nanmin(pan) + 1e-6)
    new_intensity = (1 - pan_strength) * intensity + pan_strength * pan_norm

    # 逆变换
    h = hue
    s = np.clip(saturation, 0, 1)
    i = np.clip(new_intensity, 0, 1)

    r_out = i * (1 + 2 * s * np.cos(h - np.pi / 3) / np.sqrt(3))
    g_out = i * (1 - s * np.cos(h - np.pi / 3) / np.sqrt(3) + s * np.sin(h - np.pi / 3))
    b_out = i * (1 - s * np.cos(h - np.pi / 3) / np.sqrt(3) - s * np.sin(h - np.pi / 3))

    out = np.stack([r_out, g_out, b_out], axis=-1)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# ============================================================
# 综合工具
# ============================================================

def enhance_report(
    bands_data: np.ndarray,
    filter_types: List[str] = ("mean", "median", "gaussian", "sharpen"),
) -> Dict[str, np.ndarray]:
    """批量生成各滤波结果 (用于对比展示)。"""
    return {ft: spatial_filter(bands_data, ft) for ft in filter_types}
