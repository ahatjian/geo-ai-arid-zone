"""
遥感指数计算模块
支持从 numpy 数组或 GeoTIFF 文件计算 NDVI、EVI、MNDWI、AWEIsh
所有计算函数返回 numpy 数组，可选保存为 GeoTIFF
"""

import numpy as np
import os

# 从统一配置导入缓存 TTL
from config import CACHE_CONFIG

# ---- 条件缓存装饰器 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    """条件缓存装饰器"""
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=CACHE_CONFIG.get("show_spinner", True))
    else:
        def noop_decorator(func):
            return func
        return noop_decorator


# ============================================
# 基础指数计算 (输入: numpy 数组)
# ============================================

def calc_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算 NDVI 植被指数

    NDVI = (NIR - Red) / (NIR + Red)

    参数:
        red: 红波段数组 (float)
        nir: 近红外波段数组 (float)

    返回:
        NDVI 数组, 范围 [-1, 1]
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = nir + red
    # 避免除零
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
    return np.clip(ndvi, -1, 1)


def calc_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算 EVI 增强植被指数

    EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)

    参数:
        blue: 蓝波段数组 (float)
        red: 红波段数组 (float)
        nir: 近红外波段数组 (float)

    返回:
        EVI 数组, 范围约 [-1, 1]
    """
    blue = blue.astype(np.float32)
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, 0.0)
    return np.clip(evi, -1, 1)


def calc_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    计算 MNDWI 修正归一化水体指数

    MNDWI = (Green - SWIR1) / (Green + SWIR1)

    参数:
        green: 绿波段数组 (float)
        swir1: 短波红外1波段数组 (float)

    返回:
        MNDWI 数组, 范围 [-1, 1]
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    denominator = green + swir1
    mndwi = np.where(denominator != 0, (green - swir1) / denominator, 0.0)
    return np.clip(mndwi, -1, 1)


def calc_aweish(green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    计算 AWEIsh 自动水体提取指数 (阴影/暗表面版本)

    AWEIsh = 4 × (Green - SWIR2) - 0.25 × NIR + 2.75 × SWIR1

    参数:
        green: 绿波段数组 (float)
        nir: 近红外波段数组 (float)
        swir1: 短波红外1波段数组 (float)
        swir2: 短波红外2波段数组 (float)

    返回:
        AWEIsh 数组
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    swir2 = swir2.astype(np.float32)

    aweish = 4.0 * (green - swir2) - 0.25 * nir + 2.75 * swir1
    return aweish


# ============================================
# 水体提取 (阈值分割)
# ============================================

def extract_water_mndwi(mndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    基于 MNDWI 阈值提取水体

    参数:
        mndwi: MNDWI 数组
        threshold: 阈值，默认 0.0

    返回:
        二值数组: 1=水体, 0=非水体
    """
    return (mndwi > threshold).astype(np.uint8)


def extract_water_aweish(aweish: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    基于 AWEIsh 阈值提取水体

    参数:
        aweish: AWEIsh 数组
        threshold: 阈值，默认 0.0

    返回:
        二值数组: 1=水体, 0=非水体
    """
    return (aweish > threshold).astype(np.uint8)


# ============================================
# 面积统计
# ============================================

def calc_water_area(water_mask: np.ndarray, pixel_size_m: float = 10.0) -> dict:
    """
    计算水体面积统计

    参数:
        water_mask: 二值水体掩膜 (1=水体, 0=非水体)
        pixel_size_m: 像元尺寸 (米), Sentinel-2=10, Landsat=30

    返回:
        dict: {
            "water_pixels": 水体像元数,
            "total_pixels": 总像元数,
            "water_ratio": 水体占比,
            "water_area_km2": 水体面积 (km²),
            "pixel_area_m2": 单像元面积 (m²)
        }
    """
    water_pixels = int(np.sum(water_mask == 1))
    total_pixels = int(water_mask.size)
    pixel_area_m2 = pixel_size_m * pixel_size_m
    water_area_km2 = water_pixels * pixel_area_m2 / 1e6

    return {
        "water_pixels": water_pixels,
        "total_pixels": total_pixels,
        "water_ratio": water_pixels / total_pixels if total_pixels > 0 else 0,
        "water_area_km2": round(water_area_km2, 4),
        "pixel_area_m2": pixel_area_m2,
    }


def calc_vegetation_stats(ndvi: np.ndarray, pixel_size_m: float = 10.0) -> dict:
    """
    计算植被覆盖统计

    参数:
        ndvi: NDVI 数组
        pixel_size_m: 像元尺寸 (米)

    返回:
        dict: {
            "mean": 平均 NDVI,
            "max": 最大 NDVI,
            "min": 最小 NDVI,
            "dense_veg_ratio": 密植被占比 (NDVI>0.6),
            "sparse_veg_ratio": 稀疏植被占比 (0.2<NDVI<0.6),
            "bare_ratio": 裸地占比 (NDVI<0.1),
        }
    """
    total = ndvi.size
    return {
        "mean": round(float(np.nanmean(ndvi)), 4),
        "max": round(float(np.nanmax(ndvi)), 4),
        "min": round(float(np.nanmin(ndvi)), 4),
        "dense_veg_ratio": round(float(np.sum(ndvi > 0.6)) / total, 4),
        "sparse_veg_ratio": round(float(np.sum((ndvi > 0.2) & (ndvi <= 0.6))) / total, 4),
        "bare_ratio": round(float(np.sum(ndvi < 0.1)) / total, 4),
    }


# ============================================
# GeoTIFF 保存
# ============================================

def save_index_geotiff(
    data: np.ndarray,
    reference_path: str,
    output_path: str,
):
    """
    将指数计算结果保存为 GeoTIFF，使用参考影像的空间信息

    参数:
        data: 指数数组 (2D)
        reference_path: 参考影像路径 (用于获取 CRS 和 transform)
        output_path: 输出 GeoTIFF 路径
    """
    import rasterio

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=data.dtype,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)


def save_mask_geotiff(
    mask: np.ndarray,
    reference_path: str,
    output_path: str,
):
    """
    将二值掩膜保存为 GeoTIFF

    参数:
        mask: 二值掩膜数组 (2D, uint8)
        reference_path: 参考影像路径
        output_path: 输出路径
    """
    import rasterio

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            compress="lzw",
            nodata=255,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)


# ============================================
# 从 GeoTIFF 文件批量读取波段并计算
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def load_bands_from_geotiff(
    geotiff_path: str,
    band_indices: list = None,
) -> dict:
    """
    从多波段 GeoTIFF 加载指定波段

    参数:
        geotiff_path: GeoTIFF 文件路径
        band_indices: 波段索引列表 (0-based, e.g. [0,1,2,3,4,5] for 6-bands)。
                      默认 None → 加载所有波段 (返回键为 1-based 的 rasterio 索引)。

    返回:
        dict: {
            "bands": {idx: 数组, ...},  # 键 = 传入的 band_indices (0-based) 或默认 1-based
            "profile": rasterio profile,
            "crs": CRS,
            "transform": transform,
            "shape": (height, width),
        }
    """
    import rasterio

    with rasterio.open(geotiff_path) as src:
        if band_indices is None:
            # 默认: 加载所有波段, 键和 rasterio 索引都是 1-based
            band_indices = list(range(1, src.count + 1))
            zero_based = False
        else:
            # 显式传入的 band_indices 视为 0-based (调用方惯例)
            zero_based = True

        bands = {}
        for idx in band_indices:
            # rasterio 始终使用 1-based 索引读取
            # - 显式传入: idx 是 0-based → +1 转为 rasterio 1-based
            # - 默认路径: idx 已是 1-based → 直接使用
            rasterio_idx = idx + 1 if zero_based else idx
            bands[idx] = src.read(rasterio_idx).astype(np.float32)

        return {
            "bands": bands,
            "profile": src.profile,
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
        }


@_cache(ttl=CACHE_CONFIG["ttl_short"])
def compute_all_indices(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> dict:
    """
    一次性计算所有支持的遥感指数

    参数:
        blue, green, red, nir, swir1, swir2: 波段数组

    返回:
        dict: {
            "NDVI": ndarray,
            "EVI": ndarray,
            "MNDWI": ndarray,
            "AWEIsh": ndarray,
        }
    """
    return {
        "NDVI": calc_ndvi(red, nir),
        "EVI": calc_evi(blue, red, nir),
        "MNDWI": calc_mndwi(green, swir1),
        "AWEIsh": calc_aweish(green, nir, swir1, swir2),
    }
