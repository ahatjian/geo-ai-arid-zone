"""
辐射定标与大气校正模块 — DOS 暗像元法
========================================
遥感影像预处理链的核心环节:

  - dos_correction(): DOS 暗像元法大气校正 (Chavez 1988/1996)
  - radiometric_calibration(): 辐射定标 (DN → 辐亮度 → TOA 反射率)
  - toa_reflectance(): 经典 TOA 反射率计算
  - relative_normalization(): 相对辐射归一化 (多时相匹配)

原理说明:
  大气校正的目标是去除大气散射/吸收对地表反射率的干扰。
  DOS (Dark Object Subtraction) 假设影像中存在"暗像元"
  (深水/浓密阴影, 地表反射率≈0), 其观测值完全来自大气路径辐射,
  从每个波段中减去该值即可近似去除大气效应。

  适用于: 无大气参数的场景 (如 Landsat/Sentinel L1 数据),
  是 QUAC/FLAASH 等复杂模型的轻量替代。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# DOS 暗像元法大气校正
# ============================================================

def estimate_dark_pixel(
    band: np.ndarray,
    dark_percentile: float = 1.0,
    histogram_bins: int = 256,
) -> Tuple[float, float]:
    """
    自动估计暗像元值 (每波段独立)。

    方法 (Chavez 1996): 取直方图低端 dark_percentile 分位的反射率
    作为大气路径辐射估计。

    参数:
        band: (H, W) 单波段数组 (反射率 0-1 或 DN)
        dark_percentile: 暗像元百分位 (默认 1%)
        histogram_bins: 直方图分箱数

    返回:
        (dark_value, dark_ratio)
        - dark_value: 暗像元值 (从该波段减去)
        - dark_ratio: 暗像元占有效像元比例
    """
    band = np.asarray(band, dtype=np.float64)
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        raise ValueError("波段无有效像元")

    dark_value = float(np.nanpercentile(valid, dark_percentile))
    dark_ratio = float(np.sum(valid <= dark_value) / valid.size)
    return dark_value, dark_ratio


def dos_correction(
    bands_data: np.ndarray,
    dark_percentile: float = 1.0,
    stretch_after: bool = True,
) -> Dict[str, np.ndarray]:
    """
    DOS 暗像元法大气校正 (逐波段独立)。

    公式: corrected = band - dark_value
    可选拉伸: corrected = (band - dark) × (1 - dark) / (max - dark)
              (校正后重新归一化到 0-1)

    参数:
        bands_data: (B, H, W) 多波段反射率 (0-1) 或 DN
        dark_percentile: 暗像元百分位
        stretch_after: 是否校正后拉伸到 0-1

    返回:
        dict: {
            "corrected": (B, H, W) 校正后数组,
            "dark_values": (B,) 各波段暗像元值 (大气路径辐射估计),
            "dark_ratios": (B,) 各波段暗像元比例
        }
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.ndim != 3:
        raise ValueError(f"需要 3D 数组 (B,H,W), 实际: {bands_data.ndim}D")

    corrected = np.empty_like(bands_data)
    dark_values = np.zeros(bands_data.shape[0])
    dark_ratios = np.zeros(bands_data.shape[0])

    for i in range(bands_data.shape[0]):
        band = bands_data[i]
        dark, ratio = estimate_dark_pixel(band, dark_percentile)
        dark_values[i] = dark
        dark_ratios[i] = ratio

        corr = band - dark
        if stretch_after:
            valid = corr[np.isfinite(corr)]
            vmax = np.nanmax(valid) if valid.size > 0 else 1.0
            if vmax > 0:
                corr = np.clip(corr / vmax, 0, 1)
        corrected[i] = corr

    return {
        "corrected": corrected,
        "dark_values": dark_values,
        "dark_ratios": dark_ratios,
    }


def dos_quality_report(
    bands_data: np.ndarray,
    corrected: np.ndarray,
    band_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    DOS 校正质量评估: 每波段校正前后统计对比。

    参数:
        bands_data: 原始波段 (B, H, W)
        corrected: 校正后波段 (B, H, W)
        band_names: 波段名列表 (可选)

    返回:
        list[dict]: 每波段 {波段, 校正前均值, 校正后均值, 均值变化, 大气路径辐射}
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)

    report = []
    for i in range(bands_data.shape[0]):
        before = np.nanmean(bands_data[i])
        after = np.nanmean(corrected[i])
        dark = float(np.nanmean(bands_data[i] - corrected[i]))
        name = band_names[i] if band_names and i < len(band_names) else f"波段{i+1}"
        report.append({
            "波段": name,
            "校正前均值": round(float(before), 4),
            "校正后均值": round(float(after), 4),
            "均值变化": round(float(before - after), 4),
            "大气路径辐射": round(dark, 4),
        })
    return report


# ============================================================
# 辐射定标 (DN → TOA 反射率)
# ============================================================

def radiometric_calibration(
    dn_bands: np.ndarray,
    gains: List[float],
    biases: List[float],
    solar_zenith_deg: float,
    esun: List[float],
    earth_sun_distance: float = 1.0,
) -> np.ndarray:
    """
    辐射定标: DN → 辐亮度 → TOA 反射率。

    公式:
      L = gain × DN + bias                    (辐亮度 W/m²/sr/μm)
      ρ = π × L × d² / (ESUN × cos(θz))       (TOA 反射率)

    参数:
        dn_bands: (B, H, W) DN 值数组
        gains: 每波段增益 (gain)
        biases: 每波段偏置 (bias)
        solar_zenith_deg: 太阳天顶角 (度)
        esun: 每波段太阳光谱辐照度 ESUN (W/m²/μm)
        earth_sun_distance: 日地距离 (AU, 1.0=平均)

    返回:
        (B, H, W) TOA 反射率 (0-1)
    """
    dn_bands = np.asarray(dn_bands, dtype=np.float64)
    B = dn_bands.shape[0]
    if len(gains) != B or len(biases) != B or len(esun) != B:
        raise ValueError("gains/biases/esun 长度必须等于波段数")

    cos_theta = np.cos(np.radians(solar_zenith_deg))
    if cos_theta < 1e-10:
        raise ValueError(f"太阳天顶角无效: {solar_zenith_deg}°")

    reflectance = np.empty_like(dn_bands)
    for i in range(B):
        radiance = gains[i] * dn_bands[i] + biases[i]
        reflectance[i] = (
            np.pi * radiance * earth_sun_distance ** 2
            / (esun[i] * cos_theta)
        )
    return np.clip(reflectance, 0, 1.5)


def toa_reflectance(
    dn_bands: np.ndarray,
    reflectance_mult: List[float],
    reflectance_add: List[float],
    solar_zenith_deg: float,
) -> np.ndarray:
    """
    Landsat Collection 2 风格 TOA 反射率 (用元数据系数)。

    公式: ρ = (Mρ × DN + Aρ) / cos(θz)

    Landsat C2 元数据提供 REFLECTANCE_MULT_BAND_x / REFLECTANCE_ADD_BAND_x。

    参数:
        dn_bands: (B, H, W) DN
        reflectance_mult: 每波段反射率乘数 Mρ
        reflectance_add: 每波段反射率加数 Aρ
        solar_zenith_deg: 太阳天顶角

    返回:
        (B, H, W) TOA 反射率
    """
    dn_bands = np.asarray(dn_bands, dtype=np.float64)
    B = dn_bands.shape[0]
    cos_theta = np.cos(np.radians(solar_zenith_deg))
    if cos_theta < 1e-10:
        raise ValueError(f"太阳天顶角无效: {solar_zenith_deg}°")

    result = np.empty_like(dn_bands)
    for i in range(B):
        mult = reflectance_mult[i] if i < len(reflectance_mult) else 2e-5
        add = reflectance_add[i] if i < len(reflectance_add) else -0.1
        result[i] = (mult * dn_bands[i] + add) / cos_theta
    return np.clip(result, 0, 1.5)


# ============================================================
# 相对辐射归一化 (多时相匹配)
# ============================================================

def relative_normalization(
    target_bands: np.ndarray,
    reference_bands: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    相对辐射归一化: 将目标影像辐射水平匹配到参考影像。

    方法:
      "linear": 最小二乘线性回归 y = a·x + b (逐波段)
      "hist_match": 直方图匹配 (逐波段)

    参数:
        target_bands: (B, H, W) 待归一化影像
        reference_bands: (B, H, W) 参考影像
        method: "linear" | "hist_match"

    返回:
        (B, H, W) 归一化后影像
    """
    target = np.asarray(target_bands, dtype=np.float64)
    reference = np.asarray(reference_bands, dtype=np.float64)
    if target.shape != reference.shape:
        raise ValueError(f"影像尺寸不一致: {target.shape} vs {reference.shape}")

    B = target.shape[0]
    normalized = np.empty_like(target)

    for i in range(B):
        t = target[i]
        r = reference[i]
        valid = np.isfinite(t) & np.isfinite(r)

        if method == "linear":
            # 最小二乘: y = a·x + b
            x = t[valid]
            y = r[valid]
            if x.size < 10:
                normalized[i] = t
                continue
            a, b = np.polyfit(x, y, 1)
            normalized[i] = a * t + b
        elif method == "hist_match":
            from skimage.exposure import match_histograms
            normalized[i] = match_histograms(t, r)
        else:
            raise ValueError(f"未知归一化方法: {method}")

    return normalized


# ============================================================
# 综合预处理链
# ============================================================

def preprocess_pipeline(
    bands_data: np.ndarray,
    do_dos: bool = True,
    do_normalize: bool = True,
    dark_percentile: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    综合预处理链: (可选) DOS 大气校正 → (可选) 0-1 归一化。

    参数:
        bands_data: (B, H, W) 输入波段
        do_dos: 是否执行 DOS 大气校正
        do_normalize: 是否归一化到 0-1 (DN 自动缩放)
        dark_percentile: DOS 暗像元百分位

    返回:
        dict: {"processed": (B,H,W) 处理后数组, "dark_values": ..., "steps": [处理步骤]}
    """
    bands = np.asarray(bands_data, dtype=np.float64)
    steps = []

    # DN → 0-1 归一化 (反射率)
    if do_normalize and np.nanmedian(bands) > 10:
        bands = bands / 10000.0
        steps.append("DN → 反射率 (÷10000)")

    dark_values = None
    if do_dos:
        result = dos_correction(bands, dark_percentile=dark_percentile)
        bands = result["corrected"]
        dark_values = result["dark_values"]
        steps.append(f"DOS 暗像元法大气校正 (percentile={dark_percentile}%)")

    return {
        "processed": bands,
        "dark_values": dark_values,
        "steps": steps,
    }
