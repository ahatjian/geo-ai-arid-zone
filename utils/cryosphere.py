"""
冰冻圈分析模块
================
基于 Sentinel-2/Landsat 多光谱数据的冰冻圈要素分析

指标:
  - NDSI  (归一化积雪指数)  — 积雪识别 (Dozier 1989)
  - NDSI2 (NIR版积雪指数)   — 备选积雪检测
  - Snow Cover (积雪覆盖)   — 4级分类 (积雪/冰/混合/裸地)
  - Glacier Boundary (冰川边界) — NIR/SWIR 比值 + NDVI 双重过滤
  - Snow Line (雪线高度)      — DEM 辅助雪线估计
  - Frozen Ground (冻土活动层) — 地表状态分析

参考:
  - Dozier (1989) NDSI snow mapping
  - Hall et al. (2002) MODIS snow-cover mapping
  - 国家冰川冻土沙漠科学数据中心 GCE 平台
  - 青海生态之窗 青藏高原冰冻圈监测
  - Paul et al. (2015) glacier mapping guidelines

依赖: numpy, rasterio (可选)
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
# 积雪/冰川分类定义
# ============================================================

SNOW_COVER_CLASSES = [
    {"code": 0, "name": "裸地/水体", "ndsi_range": (-np.inf, 0.15),
     "description": "无积雪覆盖", "color": "#8B4513", "risk": "无"},
    {"code": 1, "name": "薄雪/混合像元", "ndsi_range": (0.15, 0.40),
     "description": "少量积雪或雪-裸地混合", "color": "#87CEEB", "risk": "低"},
    {"code": 2, "name": "积雪", "ndsi_range": (0.40, 0.70),
     "description": "明显积雪覆盖", "color": "#FFFFFF", "risk": "中"},
    {"code": 3, "name": "冰川/粒雪", "ndsi_range": (0.70, np.inf),
     "description": "冰川或高反射粒雪", "color": "#E0E0FF", "risk": "关注"},
]

GLACIER_CLASSES = [
    {"code": 0, "name": "非冰川区", "color": "#8B4513"},
    {"code": 1, "name": "冰川消融区", "color": "#ADD8E6"},
    {"code": 2, "name": "冰川积累区", "color": "#FFFFFF"},
]


@dataclass
class CryosphereResult:
    """冰冻圈分析结果"""
    ndsi: Optional[np.ndarray] = None          # (H, W) NDSI
    ndsi_nir: Optional[np.ndarray] = None      # (H, W) NDSI-NIR版本
    snow_cover: Optional[np.ndarray] = None    # (H, W) 积雪分类 0-3
    glacier_mask: Optional[np.ndarray] = None  # (H, W) 冰川掩膜 0-2
    ndvi: Optional[np.ndarray] = None          # (H, W)
    stats: Optional[List[Dict]] = None         # 分级统计
    summary: Dict = field(default_factory=dict)


# ============================================================
# 指数计算
# ============================================================

def calc_ndsi(
    green: np.ndarray,
    swir1: np.ndarray,
) -> np.ndarray:
    """
    计算归一化积雪指数 (Normalized Difference Snow Index)

    公式 (Dozier 1989, 适用于 Sentinel-2/Landsat):
      NDSI = (Green - SWIR1) / (Green + SWIR1)

    原理: 积雪在可见光波段高反射，在短波红外强吸收
          → NDSI > 0.4 表示积雪

    参数:
        green: 绿波段 (H, W), 反射率 0-1
        swir1: 短波红外1 (H, W)

    返回:
        ndsi: (H, W), -1~1, >0.4 为积雪
    """
    green = np.asarray(green, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)

    # 自动缩放
    if np.nanmedian(green) > 10:
        green /= 10000.0
        swir1 /= 10000.0

    denom = green + swir1
    ndsi = np.where(denom > 1e-6, (green - swir1) / denom, 0.0)
    return np.clip(ndsi, -1.0, 1.0).astype(np.float32)


def calc_ndsi_nir(
    green: np.ndarray,
    nir: np.ndarray,
) -> np.ndarray:
    """
    计算 NIR 版本积雪指数 (备选)

    公式: NDSI_NIR = (Green - NIR) / (Green + NIR)

    适用场景: SWIR1 波段不可用时

    参数:
        green: 绿波段 (H, W)
        nir: 近红外 (H, W)

    返回:
        ndsi_nir: (H, W)
    """
    green = np.asarray(green, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)

    if np.nanmedian(green) > 10:
        green /= 10000.0
        nir /= 10000.0

    denom = green + nir
    ndsi_nir = np.where(denom > 1e-6, (green - nir) / denom, 0.0)
    return np.clip(ndsi_nir, -1.0, 1.0).astype(np.float32)


def calc_snow_cover(
    ndsi: np.ndarray,
    method: str = "simple",
    threshold: float = 0.4,
) -> np.ndarray:
    """
    积雪覆盖分类

    方法:
      "simple"     — 单阈值二值化 (NDSI > threshold)
      "classified" — 4级分类 (裸地/薄雪/积雪/冰川)

    参数:
        ndsi: (H, W) NDSI
        method: 分类方法
        threshold: "simple" 方法的阈值

    返回:
        (H, W) 分类图
    """
    ndsi = np.asarray(ndsi, dtype=np.float64)

    if method == "simple":
        return (ndsi > threshold).astype(np.int8)
    else:
        category = np.zeros(ndsi.shape, dtype=np.int8)
        for cls in SNOW_COVER_CLASSES:
            lo, hi = cls["ndsi_range"]
            if cls["code"] > 0:
                category[(ndsi >= lo) & (ndsi < hi)] = cls["code"]
        return category


# ============================================================
# 冰川边界提取
# ============================================================

def extract_glacier_mask(
    ndsi: np.ndarray,
    ndvi: np.ndarray,
    nir: Optional[np.ndarray] = None,
    swir1: Optional[np.ndarray] = None,
    method: str = "ratio",
) -> np.ndarray:
    """
    冰川边界提取

    方法:
      "ndsi_only"  — 仅用 NDSI > 0.7
      "ratio"      — NIR/SWIR 比值 + NDVI + NDSI 三重过滤 (推荐)
      "combined"   — 综合多条件

    三重条件:
      1. NDSI > 0.4 (高反射积雪/冰)
      2. NDVI < 0.1 (几乎无植被)
      3. NIR/SWIR > 1.5 (冰川在 NIR 比 SWIR 更亮)

    返回:
        glacier_mask: (H, W), 0=非冰川, 1=消融区, 2=积累区
    """
    ndsi = np.asarray(ndsi, dtype=np.float64)
    ndvi = np.asarray(ndvi, dtype=np.float64)

    if method == "ndsi_only":
        # 简化版: NDSI > 0.7 = 冰川
        accumulation = (ndsi > 0.7) & (ndvi < 0.05)
        ablation = (ndsi > 0.4) & (ndsi <= 0.7) & (ndvi < 0.15)

        glacier = np.zeros(ndsi.shape, dtype=np.int8)
        glacier[ablation] = 1
        glacier[accumulation] = 2
        return glacier

    elif method == "ratio" and nir is not None and swir1 is not None:
        nir = np.asarray(nir, dtype=np.float64)
        swir1 = np.asarray(swir1, dtype=np.float64)

        if np.nanmedian(nir) > 10:
            nir /= 10000.0
            swir1 /= 10000.0

        # NIR/SWIR 比值
        ratio = np.where(swir1 > 1e-6, nir / swir1, 0.0)

        # 三重条件
        cond_snow = ndsi > 0.4
        cond_veg = ndvi < 0.1
        cond_ratio = ratio > 1.5

        # 积累区: NDSI 很高, 几乎无融化
        accumulation = (ndsi > 0.7) & (ndvi < 0.03) & cond_ratio

        # 消融区: NDSI 中等, 有裸冰暴露
        ablation = cond_snow & cond_veg & (~accumulation) & cond_ratio

        glacier = np.zeros(ndsi.shape, dtype=np.int8)
        glacier[ablation] = 1
        glacier[accumulation] = 2
        return glacier

    else:
        # combined: NDSI + NDVI only
        accumulation = (ndsi > 0.65) & (ndvi < 0.05)
        ablation = (ndsi > 0.35) & (ndsi <= 0.65) & (ndvi < 0.15)

        glacier = np.zeros(ndsi.shape, dtype=np.int8)
        glacier[ablation] = 1
        glacier[accumulation] = 2
        return glacier


# ============================================================
# 雪线高度估计
# ============================================================

def estimate_snow_line(
    ndsi: np.ndarray,
    dem: Optional[np.ndarray] = None,
    method: str = "percentile",
) -> Dict:
    """
    雪线高度估计

    方法:
      "percentile" — 取 NDSI > 0.4 的中位高程
      "gradient"   — 基于 NDSI 垂直梯度突变点

    参数:
        ndsi: (H, W) NDSI
        dem: (H, W) DEM (可选, 无则返回占位值)
        method: 估计方法

    返回:
        dict: {"snow_line_elevation": float, "method": str,
               "snow_cover_ratio": float, ...}
    """
    ndsi = np.asarray(ndsi, dtype=np.float64)
    snow_mask = ndsi > 0.4
    snow_ratio = snow_mask.sum() / max(snow_mask.size, 1)

    if dem is None:
        return {
            "snow_line_elevation": None,
            "method": method,
            "snow_cover_ratio": round(snow_ratio, 4),
            "note": "需要 DEM 数据计算雪线高度",
            "available": False,
        }

    dem = np.asarray(dem, dtype=np.float64)

    if method == "percentile":
        snow_elevations = dem[snow_mask]
        if len(snow_elevations) > 100:
            # 取积雪区第 10 百分位高程作为雪线近似
            snow_line = np.percentile(snow_elevations, 10)
        else:
            snow_line = None

    elif method == "gradient":
        # 沿高程梯度找 NDSI 突变点
        valid = np.isfinite(dem) & np.isfinite(ndsi)
        dem_valid = dem[valid]
        ndsi_valid = ndsi[valid]

        if len(dem_valid) < 100:
            snow_line = None
        else:
            # 按高程分 bin
            n_bins = min(30, len(dem_valid) // 100)
            elev_bins = np.linspace(dem_valid.min(), dem_valid.max(), n_bins + 1)
            bin_ndsi = []
            bin_elev = []

            for i in range(n_bins):
                bm = (dem_valid >= elev_bins[i]) & (dem_valid < elev_bins[i + 1])
                if bm.sum() > 5:
                    bin_elev.append((elev_bins[i] + elev_bins[i + 1]) / 2)
                    bin_ndsi.append(np.mean(ndsi_valid[bm]))

            if len(bin_ndsi) >= 4:
                # NDSI 随高程梯度最大处 = 雪线
                bin_ndsi = np.array(bin_ndsi)
                gradients = np.abs(np.diff(bin_ndsi))
                max_grad_idx = np.argmax(gradients)
                snow_line = bin_elev[max_grad_idx + 1]
            else:
                snow_line = None
    else:
        snow_line = None

    return {
        "snow_line_elevation": round(snow_line, 1) if snow_line else None,
        "method": method,
        "snow_cover_ratio": round(snow_ratio, 4),
        "snow_pixels": int(snow_mask.sum()),
        "total_pixels": int(snow_mask.size),
        "available": snow_line is not None,
    }


# ============================================================
# 冻土活动层分析
# ============================================================

def analyze_frozen_ground(
    ndvi: np.ndarray,
    ndsi: np.ndarray,
    lst: Optional[np.ndarray] = None,
) -> Dict:
    """
    冻土活动层状态分析 (简化的遥感替代方案)

    使用 NDVI + NDSI + 可选 LST 进行冻土状态推断:
      - 高 NDSI + 低 NDVI = 冻土区 (积雪覆盖)
      - 中 NDVI + 无雪 = 活动层融化中
      - 高 NDVI = 完全融化

    参数:
        ndvi: (H, W) NDVI
        ndsi: (H, W) NDSI
        lst: (H, W) 地表温度 (可选, 提高精度)

    返回:
        dict: 冻土状态统计
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    ndsi = np.asarray(ndsi, dtype=np.float64)

    # 冻土分类 (简化)
    frozen_mask = (ndsi > 0.4) & (ndvi < 0.1)  # 积雪覆盖 → 冻土
    thawing_mask = (ndsi < 0.15) & (ndvi > 0.1) & (ndvi < 0.3)  # 融化中
    active_mask = (~frozen_mask) & (~thawing_mask) & np.isfinite(ndvi)

    frozen_ratio = frozen_mask.sum() / max(frozen_mask.size, 1)
    thawing_ratio = thawing_mask.sum() / max(thawing_mask.size, 1)
    active_ratio = active_mask.sum() / max(active_mask.size, 1)

    # 如有 LST, 细化
    if lst is not None:
        lst = np.asarray(lst, dtype=np.float64)
        frozen_lst = float(np.nanmean(lst[frozen_mask])) if frozen_mask.any() else None
        thawing_lst = float(np.nanmean(lst[thawing_mask])) if thawing_mask.any() else None
        active_lst = float(np.nanmean(lst[active_mask])) if active_mask.any() else None
    else:
        frozen_lst = thawing_lst = active_lst = None

    # 冻土状态判断
    if frozen_ratio > 0.5:
        state = "🧊 大范围冻土"
    elif frozen_ratio > 0.2:
        state = "❄️ 部分冻土"
    elif thawing_ratio > 0.3:
        state = "💧 活动层融化"
    else:
        state = "🌿 完全融化"

    return {
        "state": state,
        "frozen_ratio": round(frozen_ratio, 4),
        "thawing_ratio": round(thawing_ratio, 4),
        "active_ratio": round(active_ratio, 4),
        "frozen_pixels": int(frozen_mask.sum()),
        "thawing_pixels": int(thawing_mask.sum()),
        "active_pixels": int(active_mask.sum()),
        "frozen_lst": round(frozen_lst, 2) if frozen_lst else None,
        "thawing_lst": round(thawing_lst, 2) if thawing_lst else None,
        "active_lst": round(active_lst, 2) if active_lst else None,
    }


# ============================================================
# 统计计算
# ============================================================

def compute_snow_cover_stats(
    snow_category: np.ndarray,
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """
    积雪覆盖分级统计

    参数:
        snow_category: (H, W) 0-3 积雪分类
        pixel_size_m: 像元大小

    返回:
        stats: 每级统计
    """
    total = np.sum(np.isfinite(snow_category))
    stats = []
    for cls in SNOW_COVER_CLASSES:
        code = cls["code"]
        count = int(np.sum(snow_category == code))
        ratio = count / max(total, 1)
        area = count * (pixel_size_m ** 2) / 1e6
        stats.append({
            "code": code, "name": cls["name"],
            "pixel_count": count, "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": cls["color"],
            "description": cls["description"],
        })
    return stats


def compute_glacier_stats(
    glacier_mask: np.ndarray,
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """冰川区域统计"""
    total = np.sum(np.isfinite(glacier_mask))
    stats = []
    for cls in GLACIER_CLASSES:
        code = cls["code"]
        count = int(np.sum(glacier_mask == code))
        ratio = count / max(total, 1)
        area = count * (pixel_size_m ** 2) / 1e6
        stats.append({
            "code": code, "name": cls["name"],
            "pixel_count": count, "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": cls["color"],
        })
    return stats


# ============================================================
# 一站式评估
# ============================================================

def assess_cryosphere(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    pixel_size_m: float = 10.0,
    snow_threshold: float = 0.4,
    extract_glacier: bool = True,
    dem: Optional[np.ndarray] = None,
) -> CryosphereResult:
    """
    一站式冰冻圈评估

    参数:
        bands_data: (B, H, W) 多波段数据
        satellite: 卫星名称
        pixel_size_m: 像元大小
        snow_threshold: NDSI 积雪阈值
        extract_glacier: 是否提取冰川边界
        dem: DEM 数据 (可选, 用于雪线估计)

    返回:
        CryosphereResult
    """
    bands = np.asarray(bands_data, dtype=np.float64)
    if bands.shape[0] < 5:
        raise ValueError(f"至少需要 5 个波段, 实际: {bands.shape[0]}")

    green = bands[1]
    red = bands[2]
    nir = bands[3]
    swir1 = bands[4]

    # 1. NDSI
    ndsi = calc_ndsi(green, swir1)

    # 2. NDSI NIR 版本
    ndsi_nir = calc_ndsi_nir(green, nir)

    # 3. NDVI
    denom = nir + red
    ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # 4. 积雪分类
    snow_category = calc_snow_cover(ndsi, method="classified", threshold=snow_threshold)
    snow_stats = compute_snow_cover_stats(snow_category, pixel_size_m)

    # 5. 冰川边界
    glacier_mask = None
    if extract_glacier:
        glacier_mask = extract_glacier_mask(
            ndsi, ndvi, nir=nir, swir1=swir1, method="ratio"
        )

    # 6. 雪线
    snow_line = estimate_snow_line(ndsi, dem=dem)

    # 7. 冻土状态
    frozen = analyze_frozen_ground(ndvi, ndsi)

    # 8. 汇总
    total_snow_ratio = sum(s["ratio"] for s in snow_stats if s["code"] >= 2)

    return CryosphereResult(
        ndsi=ndsi.astype(np.float32),
        ndsi_nir=ndsi_nir.astype(np.float32),
        snow_cover=snow_category,
        glacier_mask=glacier_mask,
        ndvi=ndvi.astype(np.float32),
        stats=snow_stats,
        summary={
            "total_snow_cover_ratio": round(total_snow_ratio, 4),
            "dominant_snow_class": max(snow_stats, key=lambda s: s["ratio"])["name"],
            "snow_line": snow_line,
            "frozen_ground": frozen,
            "ndsi_mean": round(float(np.nanmean(ndsi)), 4),
            "ndsi_std": round(float(np.nanstd(ndsi)), 4),
        },
    )
