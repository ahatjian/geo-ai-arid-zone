"""
土壤盐渍化监测模块
====================
基于 Sentinel-2/Landsat 多光谱数据的土壤盐渍化评估

核心方法: 盐分指数组合 + NDSI 盐分阈值分级 (西北干旱区绿洲盐渍化主流方法)

指标:
  - SI   (盐分指数 Salinity Index)        — SI = sqrt(Blue × Red)
  - SI1  (盐分指数1)                      — SI1 = sqrt(Green × Red)
  - SI2  (盐分指数2)                      — SI2 = sqrt(Green² + Red² + NIR²)
  - NDSI (归一化盐分指数)                 — NDSI = (Red − NIR) / (Red + NIR)
  - BI   (亮度指数 Brightness Index)      — BI = sqrt(Red² + NIR²)
  - SAVI (土壤调节植被指数, 用于掩膜)     — 排除植被干扰

分级: 5 级 (非/轻度/中度/重度/极重度盐渍化)

参考:
  - Allbed & Kumar (2013) 盐渍化遥感监测综述
  - Khan et al. (2005) 盐分指数阈值分级
  - 王飞等 (2010) 新疆干旱区土壤盐渍化遥感监测
  - 塔里木河流域绿洲盐渍化研究

依赖: numpy
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ---- 条件缓存 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=True)
    else:
        def noop(func):
            return func
        return noop


# ============================================================
# 盐渍化等级定义 (5 级, 与沙漠化模块配色一致)
# ============================================================

SALINITY_LEVELS = [
    {
        "code": 0, "name": "非盐渍化",
        "ndsi_range": (-np.inf, 0.0),
        "description": "土壤盐分含量低，适宜耕作与植被生长",
        "color": "#2ecc71",   # 绿色
        "risk": "安全",
    },
    {
        "code": 1, "name": "轻度盐渍化",
        "ndsi_range": (0.0, 0.08),
        "description": "表层出现轻微盐霜，部分作物生长受抑制",
        "color": "#f1c40f",   # 黄色
        "risk": "关注",
    },
    {
        "code": 2, "name": "中度盐渍化",
        "ndsi_range": (0.08, 0.16),
        "description": "盐斑明显，耐盐作物为主，需改良",
        "color": "#e67e22",   # 橙色
        "risk": "预警",
    },
    {
        "code": 3, "name": "重度盐渍化",
        "ndsi_range": (0.16, 0.24),
        "description": "大面积盐壳/盐霜，多数作物难以存活",
        "color": "#e74c3c",   # 红色
        "risk": "紧急",
    },
    {
        "code": 4, "name": "极重度盐渍化",
        "ndsi_range": (0.24, np.inf),
        "description": "盐壳裸露、盐结皮，几乎无植被，弃耕",
        "color": "#8e44ad",   # 紫色
        "risk": "危急",
    },
]

# NDSI 盐分指数分级阈值 (轻度/中度/重度/极重度分界)
DEFAULT_NDSI_THRESHOLDS = [0.0, 0.08, 0.16, 0.24]


@dataclass
class SalinityResult:
    """土壤盐渍化评估结果"""
    # 指数图
    si: Optional[np.ndarray] = None            # 盐分指数 SI
    si1: Optional[np.ndarray] = None           # 盐分指数 SI1
    si2: Optional[np.ndarray] = None           # 盐分指数 SI2
    ndsi_salt: Optional[np.ndarray] = None     # 归一化盐分指数
    bi: Optional[np.ndarray] = None            # 亮度指数
    ndvi: Optional[np.ndarray] = None          # NDVI (掩膜用)
    # 分类
    category: Optional[np.ndarray] = None      # (H, W) 0-4
    category_colors: Optional[np.ndarray] = None  # (H, W, 3) RGB
    # 统计
    stats: Optional[List[Dict]] = None         # 分级统计
    summary: Dict = field(default_factory=dict)


# ============================================================
# 指数计算
# ============================================================

def _scale_reflectance(arr: np.ndarray) -> np.ndarray:
    """若为 DN 值 (中位数 > 10) 则缩放到 0-1 反射率"""
    arr = np.asarray(arr, dtype=np.float64)
    if np.nanmedian(arr) > 10:
        arr = arr / 10000.0
    return arr


def calc_si(blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    计算盐分指数 SI (Salinity Index)

    SI = sqrt(Blue × Red)

    盐渍土在蓝、红波段反射率高 → SI 值大

    参数:
        blue: 蓝波段 (H, W)
        red: 红波段 (H, W)

    返回:
        si: (H, W) 盐分指数 (>=0)
    """
    blue = _scale_reflectance(blue)
    red = _scale_reflectance(red)
    si = np.sqrt(np.clip(blue * red, 0.0, None))
    return si.astype(np.float32)


def calc_si1(green: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    计算盐分指数 SI1

    SI1 = sqrt(Green × Red)

    参数:
        green: 绿波段 (H, W)
        red: 红波段 (H, W)

    返回:
        si1: (H, W)
    """
    green = _scale_reflectance(green)
    red = _scale_reflectance(red)
    si1 = np.sqrt(np.clip(green * red, 0.0, None))
    return si1.astype(np.float32)


def calc_si2(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算盐分指数 SI2

    SI2 = sqrt(Green² + Red² + NIR²)

    参数:
        green: 绿波段 (H, W)
        red: 红波段 (H, W)
        nir: 近红外 (H, W)

    返回:
        si2: (H, W)
    """
    green = _scale_reflectance(green)
    red = _scale_reflectance(red)
    nir = _scale_reflectance(nir)
    si2 = np.sqrt(green ** 2 + red ** 2 + nir ** 2)
    return si2.astype(np.float32)


def calc_ndsi_salinity(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算归一化盐分指数 NDSI (Normalized Difference Salinity Index)

    NDSI = (Red − NIR) / (Red + NIR)

    盐渍土在红波段反射增强、近红外相对降低 → NDSI 值为正
    注意: 与积雪 NDSI (green/swir1) 不同, 本指数针对土壤盐分

    参数:
        red: 红波段 (H, W)
        nir: 近红外 (H, W)

    返回:
        ndsi_salt: (H, W), 范围 [-1, 1]
    """
    red = _scale_reflectance(red)
    nir = _scale_reflectance(nir)
    denom = red + nir
    ndsi = np.where(denom > 1e-6, (red - nir) / denom, 0.0)
    return np.clip(ndsi, -1.0, 1.0).astype(np.float32)


def calc_bi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算亮度指数 BI (Brightness Index)

    BI = sqrt(Red² + NIR²)

    盐渍土地表亮度高 → BI 值大, 用于辅助识别盐壳

    参数:
        red: 红波段 (H, W)
        nir: 近红外 (H, W)

    返回:
        bi: (H, W)
    """
    red = _scale_reflectance(red)
    nir = _scale_reflectance(nir)
    bi = np.sqrt(red ** 2 + nir ** 2)
    return bi.astype(np.float32)


def calc_ndvi_mask(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算 NDVI (掩膜用)"""
    red = _scale_reflectance(red)
    nir = _scale_reflectance(nir)
    denom = nir + red
    ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


# ============================================================
# 分级
# ============================================================

def classify_salinity(
    ndsi_salt: np.ndarray,
    ndvi: Optional[np.ndarray] = None,
    mndwi: Optional[np.ndarray] = None,
    method: str = "composite",
    thresholds: Optional[List[float]] = None,
    veg_threshold: float = 0.40,
) -> np.ndarray:
    """
    土壤盐渍化分级 (5 级)

    方法:
      "composite": 先掩膜植被 (NDVI>veg_threshold) 和水体 (MNDWI>0),
                   剩余区域按 NDSI 盐分指数阈值分级 (推荐)
      "ndsi": 仅按 NDSI 盐分指数阈值分级

    参数:
        ndsi_salt: 归一化盐分指数 (H, W)
        ndvi: NDVI 数组 (composite 方法需要)
        mndwi: MNDWI 数组 (composite 方法可选, 掩膜水体)
        method: "composite" | "ndsi"
        thresholds: NDSI 分界阈值 [t1, t2, t3, t4] (轻度/中度/重度/极重度)
        veg_threshold: 植被阈值 (NDVI 高于此值视为植被, 判为非盐渍化)

    返回:
        category: (H, W) uint8, 0-4
    """
    if thresholds is None:
        thresholds = DEFAULT_NDSI_THRESHOLDS
    t1, t2, t3, t4 = thresholds

    ndsi_salt = np.asarray(ndsi_salt, dtype=np.float64)
    H, W = ndsi_salt.shape

    # 基础分级
    category = np.zeros((H, W), dtype=np.uint8)
    category[(ndsi_salt >= t1) & (ndsi_salt < t2)] = 1
    category[(ndsi_salt >= t2) & (ndsi_salt < t3)] = 2
    category[(ndsi_salt >= t3) & (ndsi_salt < t4)] = 3
    category[ndsi_salt >= t4] = 4

    if method == "composite":
        # 掩膜植被 → 非盐渍化
        if ndvi is not None:
            ndvi = np.asarray(ndvi, dtype=np.float64)
            category[ndvi > veg_threshold] = 0
        # 掩膜水体 → 非盐渍化
        if mndwi is not None:
            mndwi = np.asarray(mndwi, dtype=np.float64)
            category[mndwi > 0.0] = 0
        # 掩膜无效值 (NaN)
        invalid = ~np.isfinite(ndsi_salt)
        category[invalid] = 0

    return category


def compute_salinity_stats(
    category: np.ndarray,
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """
    计算盐渍化分级统计

    参数:
        category: (H, W) 分类图 (0-4)
        pixel_size_m: 像元大小 (m)

    返回:
        stats: 每级统计列表
    """
    category = np.asarray(category, dtype=np.int8)
    total_valid = np.sum(np.isfinite(category))

    stats = []
    for level in SALINITY_LEVELS:
        code = level["code"]
        count = np.sum(category == code)
        ratio = count / max(total_valid, 1)
        area = count * (pixel_size_m ** 2) / 1e6  # km²

        stats.append({
            "code": code,
            "name": level["name"],
            "pixel_count": int(count),
            "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": level["color"],
            "risk": level["risk"],
            "description": level["description"],
        })

    return stats


def get_salinity_colormap(category: np.ndarray) -> np.ndarray:
    """
    生成盐渍化分级 RGB 着色图

    参数:
        category: (H, W) 分类图 (0-4)

    返回:
        rgb: (H, W, 3) uint8
    """
    category = np.asarray(category, dtype=np.uint8)
    H, W = category.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for level in SALINITY_LEVELS:
        code = level["code"]
        hex_color = level["color"].lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        mask = category == code
        rgb[mask] = [r, g, b]

    return rgb


# ============================================================
# 一站式评估
# ============================================================

def assess_salinity(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    pixel_size_m: float = 10.0,
    method: str = "composite",
    thresholds: Optional[List[float]] = None,
    veg_threshold: float = 0.40,
) -> SalinityResult:
    """
    一站式土壤盐渍化评估

    参数:
        bands_data: (B, H, W) 多波段数据
          Sentinel-2: [B02, B03, B04, B08, B11, B12]
          Landsat:    [B2, B3, B4, B5, B6, B7]
        satellite: 卫星名称
        pixel_size_m: 像元大小 (m)
        method: 分级方法 "composite" | "ndsi"
        thresholds: NDSI 分界阈值
        veg_threshold: 植被掩膜阈值

    返回:
        SalinityResult
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)

    if bands_data.shape[0] < 4:
        raise ValueError(f"至少需要 4 个波段 (B,G,R,NIR), 实际: {bands_data.shape[0]}")

    # 提取波段 (统一 Sentinel/Landsat 索引: [B, G, R, NIR, SWIR1, SWIR2])
    blue = bands_data[0]
    green = bands_data[1]
    red = bands_data[2]
    nir = bands_data[3]
    swir1 = bands_data[4] if bands_data.shape[0] > 4 else None

    # 1. 各盐分指数
    si = calc_si(blue, red)
    si1 = calc_si1(green, red)
    si2 = calc_si2(green, red, nir)
    ndsi_salt = calc_ndsi_salinity(red, nir)
    bi = calc_bi(red, nir)
    ndvi = calc_ndvi_mask(red, nir)

    # 2. MNDWI (掩膜水体, 若有 SWIR1)
    mndwi = None
    if swir1 is not None:
        g_scaled = _scale_reflectance(green)
        s_scaled = _scale_reflectance(swir1)
        denom = g_scaled + s_scaled
        mndwi = np.where(denom > 1e-6, (g_scaled - s_scaled) / denom, 0.0)

    # 3. 分类
    category = classify_salinity(
        ndsi_salt, ndvi=ndvi, mndwi=mndwi,
        method=method, thresholds=thresholds, veg_threshold=veg_threshold,
    )

    # 4. 统计
    stats = compute_salinity_stats(category, pixel_size_m=pixel_size_m)
    colors = get_salinity_colormap(category)

    # 5. 汇总
    total_saline = sum(s["ratio"] for s in stats if s["code"] >= 1)
    severe_saline = sum(s["ratio"] for s in stats if s["code"] >= 3)

    return SalinityResult(
        si=si,
        si1=si1,
        si2=si2,
        ndsi_salt=ndsi_salt,
        bi=bi,
        ndvi=ndvi,
        category=category,
        category_colors=colors,
        stats=stats,
        summary={
            "total_salinization_ratio": round(total_saline, 4),
            "severe_salinization_ratio": round(severe_saline, 4),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"] if stats else "未知",
            "ndsi_salt_mean": round(float(np.nanmean(ndsi_salt)), 4),
            "si_mean": round(float(np.nanmean(si)), 4),
            "bi_mean": round(float(np.nanmean(bi)), 4),
            "pixel_size_m": pixel_size_m,
            "method": method,
        },
    )


@_cache(ttl=600)
def assess_salinity_cached(
    bands_hash: str,
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    pixel_size_m: float = 10.0,
    **kwargs,
) -> SalinityResult:
    """带缓存的盐渍化评估 (bands_hash 用于区分不同影像)"""
    return assess_salinity(bands_data, satellite=satellite, pixel_size_m=pixel_size_m, **kwargs)
