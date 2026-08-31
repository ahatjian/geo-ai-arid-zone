"""
纹理分析模块 — GLCM 灰度共生矩阵 (Haralick et al. 1973)
=========================================================
基于灰度共生矩阵 (Gray-Level Co-occurrence Matrix) 的纹理特征提取,
是遥感影像分类的重要辅助特征。

核心能力:
  - glcm_texture_maps(): 局部纹理图 (滑窗 GLCM → 6 种纹理特征图)
  - glcm_global_stats(): 全局纹理统计 (整幅图 GLCM)
  - build_texture_stack(): 多波段纹理特征栈 (分类用)

6 种标准纹理特征 (Haralick):
  contrast      对比度 — 局部灰度差异 (越大纹理越粗糙)
  dissimilarity 非相似性 — 对比度的线性变体
  homogeneity   同质性 — 局部灰度均匀程度
  asm           角二阶矩 — 灰度分布均匀性
  energy        能量 — ASM 的平方根
  correlation   相关性 — 灰度线性相关性 (方向性纹理)

GLCM 参数:
  distance: 像素间距 (常用 1)
  angles: 方向 (0°/45°/90°/135°)
  levels: 灰度量化级数 (16/32/64, 越小越快)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from skimage.feature import graycomatrix, graycoprops

# 纹理特征属性名 (graycoprops 支持的属性)
TEXTURE_PROPS = ["contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation"]
TEXTURE_NAMES_CN = {
    "contrast": "对比度",
    "dissimilarity": "非相似性",
    "homogeneity": "同质性",
    "asm": "角二阶矩",
    "energy": "能量",
    "correlation": "相关性",
}


def _quantize(band: np.ndarray, levels: int) -> np.ndarray:
    """将连续值波段量化为 0..levels-1 整数 (GLCM 需要)。"""
    band = np.asarray(band, dtype=np.float64)
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    vmin, vmax = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    quantized = np.clip((band - vmin) / (vmax - vmin) * (levels - 1), 0, levels - 1)
    quantized[~np.isfinite(band)] = 0
    return quantized.astype(np.uint8)


def glcm_texture_maps(
    band: np.ndarray,
    distance: int = 1,
    angles: List[float] = (0, 45, 90, 135),
    levels: int = 32,
    window_size: int = 9,
    props: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    生成局部 GLCM 纹理特征图 (滑窗逐像元计算)。

    每个像元取其 window_size×window_size 邻域计算 GLCM,
    提取各方向平均的纹理特征值。

    参数:
        band: (H, W) 单波段数组
        distance: 像素间距
        angles: 方向列表 (度)
        levels: 灰度量化级数
        window_size: 滑窗大小 (奇数, 越大纹理尺度越粗)
        props: 要计算的纹理属性 (None=全部 6 种)

    返回:
        dict: {prop_name: (H, W) 纹理特征图}
    """
    band = np.asarray(band, dtype=np.float64)
    H, W = band.shape
    if props is None:
        props = TEXTURE_PROPS

    quantized = _quantize(band, levels)
    # 边界填充 (reflect 保持纹理连续性)
    pad = window_size // 2
    padded = np.pad(quantized, pad, mode="reflect")

    angles_rad = [np.radians(a) for a in angles]

    # 滑窗计算 (逐像元, 中小影像可接受; 大影像用步长采样)
    maps = {p: np.full((H, W), np.nan) for p in props}

    # 对每个像元计算邻域 GLCM
    for i in range(H):
        for j in range(W):
            window = padded[i:i + window_size, j:j + window_size]
            glcm = graycomatrix(
                window, distances=[distance], angles=angles_rad,
                levels=levels, symmetric=True, normed=True,
            )
            for p in props:
                try:
                    vals = graycoprops(glcm, p)  # (1, n_angles)
                    maps[p][i, j] = float(np.mean(vals))
                except Exception:
                    maps[p][i, j] = 0.0

    return maps


def glcm_texture_maps_fast(
    band: np.ndarray,
    distance: int = 1,
    angles: List[float] = (0, 45, 90, 135),
    levels: int = 32,
    window_size: int = 9,
    step: int = 2,
    props: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    快速版纹理图: 按步长采样计算 + 双线性插值恢复全尺寸。

    大影像 (≥512px) 用此函数替代 glcm_texture_maps, 速度提升 step² 倍。

    参数:
        同 glcm_texture_maps, 额外:
        step: 采样步长 (2=每4像元算1个, 再插值)

    返回:
        dict: {prop_name: (H, W) 纹理特征图}
    """
    band = np.asarray(band, dtype=np.float64)
    H, W = band.shape
    if props is None:
        props = TEXTURE_PROPS

    # 下采样
    small = band[::step, ::step]
    sh, sw = small.shape
    maps_small = glcm_texture_maps(
        small, distance=distance, angles=angles, levels=levels,
        window_size=window_size, props=props,
    )

    # 双线性插值恢复
    from scipy.ndimage import zoom
    maps = {}
    for p, m in maps_small.items():
        factors = (H / sh, W / sw)
        maps[p] = zoom(m, factors, order=1, mode="nearest")
    return maps


def glcm_global_stats(
    band: np.ndarray,
    distance: int = 1,
    angles: List[float] = (0, 45, 90, 135),
    levels: int = 32,
) -> Dict[str, float]:
    """
    全局 GLCM 纹理统计 (整幅图一个 GLCM, 各方向平均)。

    适用于: 分类样本的纹理特征 / 区域纹理描述。

    返回:
        dict: {prop: 均值}
    """
    band = np.asarray(band, dtype=np.float64)
    quantized = _quantize(band, levels)
    angles_rad = [np.radians(a) for a in angles]

    glcm = graycomatrix(
        quantized, distances=[distance], angles=angles_rad,
        levels=levels, symmetric=True, normed=True,
    )
    stats = {}
    for p in TEXTURE_PROPS:
        try:
            stats[p] = float(np.mean(graycoprops(glcm, p)))
        except Exception:
            stats[p] = 0.0
    return stats


def build_texture_stack(
    bands_data: np.ndarray,
    distance: int = 1,
    levels: int = 32,
    window_size: int = 9,
    props: Optional[List[str]] = None,
    step: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """
    构建多波段纹理特征栈 (分类辅助特征)。

    对每个输入波段生成 6 种纹理图, 堆叠为特征栈。

    参数:
        bands_data: (B, H, W) 波段数组
        distance/levels/window_size: GLCM 参数
        props: 纹理属性 (None=全部)
        step: 采样步长 (大影像加速)

    返回:
        (texture_stack, names)
        - texture_stack: (B × n_props, H, W) 纹理特征栈
        - names: ["band1_contrast", ...] 特征名
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    B, H, W = bands_data.shape
    if props is None:
        props = TEXTURE_PROPS

    use_fast = H * W > 512 * 512
    stack = []
    names = []

    for b in range(B):
        band = bands_data[b]
        if use_fast:
            maps = glcm_texture_maps_fast(
                band, distance=distance, levels=levels,
                window_size=window_size, step=step, props=props,
            )
        else:
            maps = glcm_texture_maps(
                band, distance=distance, levels=levels,
                window_size=window_size, props=props,
            )
        for p in props:
            stack.append(maps[p])
            names.append(f"band{b+1}_{p}")

    return np.stack(stack), names


def summarize_texture(
    band: np.ndarray,
    band_name: str = "波段1",
) -> Dict[str, Dict[str, float]]:
    """生成纹理统计摘要 (页面展示用)。"""
    stats = glcm_global_stats(band)
    return {
        "波段": band_name,
        "统计": {TEXTURE_NAMES_CN.get(k, k): v for k, v in stats.items()},
    }
