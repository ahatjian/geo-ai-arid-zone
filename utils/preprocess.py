"""
影像预处理模块 — 云掩膜 + 重采样
================================
提供中等平台必备的影像预处理能力:

  - apply_s2_cloud_mask(): Sentinel-2 SCL 场景分类云掩膜
  - apply_landsat_cloud_mask(): Landsat QA_PIXEL 位掩码云掩膜
  - mask_clouds(): 通用云掩膜入口 (按卫星自动选择)
  - resample_array(): 数组重采样 (最近邻/双线性/立方卷积)
  - normalize_to_uint8(): 反射率归一化到 8bit 显示

云掩膜定义:
  Sentinel-2 SCL: 3=云影, 7=低云/不确定, 8=中云, 9=高云, 10=薄卷云
  Landsat QA_PIXEL: bit3=云, bit4=云影
"""

import numpy as np
from typing import Optional, Tuple

# ---- 常量 ----
# Sentinel-2 SCL 场景分类值 (ESA Level-2A)
SCL_CLOUD_VALUES = {3, 7, 8, 9, 10}  # 云影/低云/中云/高云/薄卷云
SCL_CLEAR_VALUES = {4, 5, 6}         # 植被/裸地/水 (明确晴空)
SCL_SNOW_ICE = 11                     # 冰雪 (干旱区高山积雪, 可按需掩膜)

# Landsat Collection 2 QA_PIXEL 位定义
QA_CLOUD_BIT = 3    # 云
QA_CLOUD_SHADOW_BIT = 4  # 云影
QA_SNOW_BIT = 5     # 雪


def apply_s2_cloud_mask(
    bands_data: np.ndarray,
    scl: np.ndarray,
    mask_shadow: bool = True,
    mask_cirrus: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sentinel-2 SCL 云掩膜: 将云/云影像元置 NaN。

    参数:
        bands_data: (B, H, W) 多波段反射率数组
        scl: (H, W) SCL 场景分类层 (0-11)
        mask_shadow: 是否掩膜云影 (SCL=3)
        mask_cirrus: 是否掩膜薄卷云 (SCL=10)

    返回:
        (masked_bands, cloud_mask)
        - masked_bands: (B, H, W), 云/云影处为 NaN
        - cloud_mask: (H, W) bool, True=云/云影像元
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    scl = np.asarray(scl, dtype=np.int16)

    if scl.shape != bands_data.shape[1:]:
        raise ValueError(
            f"SCL 尺寸 {scl.shape} 与波段 {bands_data.shape[1:]} 不匹配"
        )

    bad = set(SCL_CLOUD_VALUES)
    if not mask_shadow:
        bad.discard(3)
    if not mask_cirrus:
        bad.discard(10)

    cloud_mask = np.isin(scl, list(bad))
    masked = bands_data.copy()
    masked[:, cloud_mask] = np.nan
    return masked, cloud_mask


def apply_landsat_cloud_mask(
    bands_data: np.ndarray,
    qa_pixel: np.ndarray,
    mask_shadow: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Landsat Collection 2 QA_PIXEL 云掩膜。

    参数:
        bands_data: (B, H, W) 多波段反射率数组
        qa_pixel: (H, W) QA_PIXEL 位掩码
        mask_shadow: 是否掩膜云影 (bit 4)

    返回:
        (masked_bands, cloud_mask)
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    qa_pixel = np.asarray(qa_pixel, dtype=np.uint16)

    if qa_pixel.shape != bands_data.shape[1:]:
        raise ValueError("QA_PIXEL 尺寸与波段不匹配")

    # bit 3 = 云
    cloud_mask = (qa_pixel >> QA_CLOUD_BIT) & 1 == 1
    if mask_shadow:
        cloud_mask |= (qa_pixel >> QA_CLOUD_SHADOW_BIT) & 1 == 1

    masked = bands_data.copy()
    masked[:, cloud_mask] = np.nan
    return masked, cloud_mask


def mask_clouds(
    bands_data: np.ndarray,
    scl_or_qa: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    mask_shadow: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """按卫星自动选择云掩膜方法。"""
    if satellite.startswith("Sentinel"):
        return apply_s2_cloud_mask(bands_data, scl_or_qa, mask_shadow=mask_shadow)
    return apply_landsat_cloud_mask(bands_data, scl_or_qa, mask_shadow=mask_shadow)


def resample_array(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    method: str = "bilinear",
) -> np.ndarray:
    """
    数组重采样到目标尺寸。

    参数:
        arr: (H, W) 或 (B, H, W) 数组
        target_shape: (H_out, W_out)
        method: "nearest" | "bilinear" | "cubic"

    返回:
        重采样后的数组
    """
    from scipy.ndimage import zoom

    arr = np.asarray(arr, dtype=np.float64)
    h, w = arr.shape[-2:]
    t_h, t_w = target_shape

    if (h, w) == (t_h, t_w):
        return arr.copy()

    # 计算缩放因子 (scipy zoom 逐轴)
    factors = (t_h / h, t_w / w)

    order = {"nearest": 0, "bilinear": 1, "cubic": 3}.get(method, 1)

    if arr.ndim == 2:
        return zoom(arr, factors, order=order, mode="nearest")
    # 多波段: 逐波段
    return np.stack([zoom(b, factors, order=order, mode="nearest") for b in arr])


def normalize_to_uint8(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    反射率/指数归一化到 0-255 (显示用)。

    参数:
        arr: 输入数组 (含 NaN)
        vmin/vmax: 截断范围 (None=自动 2%~98% 分位数)

    返回:
        uint8 数组 (NaN → 0)
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    if vmin is None:
        vmin = float(np.nanpercentile(valid, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(valid, 98))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    out = np.clip((arr - vmin) / (vmax - vmin), 0, 1) * 255
    out[~np.isfinite(arr)] = 0
    return out.astype(np.uint8)


def cloud_cover_fraction(cloud_mask: np.ndarray) -> float:
    """统计云覆盖占比 (0-1)。"""
    cloud_mask = np.asarray(cloud_mask, dtype=bool)
    total = cloud_mask.size
    return float(cloud_mask.sum()) / max(total, 1)


def mask_stats(masked: np.ndarray) -> dict:
    """掩膜后有效像元统计。"""
    masked = np.asarray(masked)
    valid = np.isfinite(masked)
    return {
        "valid_ratio": round(float(valid.mean()), 4),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(masked.size),
    }
