"""
农业干旱分析模块
================
基于 Sentinel-2/Landsat 的农田干旱监测

指标:
  - CWSI  (作物水分胁迫指数) — NDVI 法简化版
  - SMI   (土壤水分指数)     — SWIR 反射率法
  - MPDI  (改进型垂直干旱指数) — 近红外-红光空间
  - ASWI  (农业土壤水分指数) — SWIR+NIR 综合
  - Irrigation Demand (灌溉需求) — 分区分级评估
  - TVDI  (温度植被干旱指数) — 复用 drought.py

核心思路: 农业干旱 = 植被水分胁迫 + 土壤水分亏缺

参考:
  - Jackson et al. (1981) CWSI
  - 民勤 AI 水肥一体化平台
  - 酒泉生态农业监测系统
  - 中国旱区农业遥感监测技术规范

依赖: numpy
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

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
# 农业干旱等级
# ============================================================

AGRI_DROUGHT_LEVELS = [
    {"code": 0, "name": "无干旱", "cwsi_range": (-np.inf, 0.2),
     "description": "水分充足，作物正常", "color": "#2ecc71", "irrigation": "无需灌溉"},
    {"code": 1, "name": "轻度干旱", "cwsi_range": (0.2, 0.4),
     "description": "水分轻度不足", "color": "#f1c40f", "irrigation": "建议灌溉"},
    {"code": 2, "name": "中度干旱", "cwsi_range": (0.4, 0.6),
     "description": "水分亏缺，影响产量", "color": "#e67e22", "irrigation": "需要灌溉"},
    {"code": 3, "name": "重度干旱", "cwsi_range": (0.6, 0.8),
     "description": "严重水分胁迫", "color": "#e74c3c", "irrigation": "急需灌溉"},
    {"code": 4, "name": "极度干旱", "cwsi_range": (0.8, np.inf),
     "description": "作物死亡风险", "color": "#8e44ad", "irrigation": "紧急灌溉"},
]


@dataclass
class AgriDroughtResult:
    """农业干旱评估结果"""
    cwsi: Optional[np.ndarray] = None
    smi: Optional[np.ndarray] = None
    mpdi: Optional[np.ndarray] = None
    ndvi: Optional[np.ndarray] = None
    category: Optional[np.ndarray] = None
    irrigation_demand: Optional[np.ndarray] = None
    stats: Optional[List[Dict]] = None
    summary: Dict = field(default_factory=dict)


# ============================================================
# 作物水分胁迫指数 (CWSI)
# ============================================================

def calc_cwsi_ndvi(
    ndvi: np.ndarray,
    ndvi_wet: Optional[float] = None,
    ndvi_dry: Optional[float] = None,
) -> np.ndarray:
    """
    基于 NDVI 的作物水分胁迫指数 (简化版 CWSI)

    原理: CWSI = 1 - (NDVI - NDVI_dry) / (NDVI_wet - NDVI_dry)
          其中 NDVI_wet = 充分灌溉区的 NDVI (高值)
              NDVI_dry = 干旱胁迫区的 NDVI (低值)

    如果没有指定 wet/dry，自动使用 5%/95% 百分位

    参数:
        ndvi: (H, W) NDVI
        ndvi_wet: 湿润参考 NDVI
        ndvi_dry: 干旱参考 NDVI

    返回:
        cwsi: (H, W), 0~1, 值越高胁迫越严重
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    valid = ndvi[np.isfinite(ndvi) & (ndvi > 0)]

    if ndvi_wet is None:
        ndvi_wet = np.percentile(valid, 95) if len(valid) > 10 else 0.8
    if ndvi_dry is None:
        ndvi_dry = np.percentile(valid, 5) if len(valid) > 10 else 0.1

    denom = ndvi_wet - ndvi_dry
    if denom < 0.01:
        denom = 0.01

    cwsi = 1.0 - (ndvi - ndvi_dry) / denom
    return np.clip(cwsi, 0.0, 1.0).astype(np.float32)


# ============================================================
# 土壤水分指数 (SMI)
# ============================================================

def calc_smi_swir(
    swir1: np.ndarray,
    swir1_max: Optional[float] = None,
    swir1_min: Optional[float] = None,
) -> np.ndarray:
    """
    基于 SWIR 的土壤水分指数 (SMI)

    原理: SWIR 反射率与土壤含水量负相关
          SMI = (SWIR_max - SWIR) / (SWIR_max - SWIR_min)

    参数:
        swir1: (H, W) SWIR1 波段反射率
        swir1_max: 干土 SWIR 值 (无水分)
        swir1_min: 湿土 SWIR 值 (饱和)

    返回:
        smi: (H, W), 0~1, 值越高越湿润
    """
    swir1 = np.asarray(swir1, dtype=np.float64)
    if np.nanmedian(swir1) > 10:
        swir1 /= 10000.0

    valid = swir1[np.isfinite(swir1)]

    if swir1_max is None:
        swir1_max = np.percentile(valid, 95) if len(valid) > 10 else 0.5
    if swir1_min is None:
        swir1_min = np.percentile(valid, 5) if len(valid) > 10 else 0.05

    denom = swir1_max - swir1_min
    if denom < 0.01:
        denom = 0.01

    smi = (swir1_max - swir1) / denom
    return np.clip(smi, 0.0, 1.0).astype(np.float32)


def calc_smi_combined(
    ndvi: np.ndarray,
    swir1: np.ndarray,
    ndvi_weight: float = 0.6,
) -> np.ndarray:
    """
    综合土壤水分指数 (NDVI + SWIR 加权)

    参数:
        ndvi: (H, W)
        swir1: (H, W)
        ndvi_weight: NDVI 权重 (0~1)

    返回:
        smi_combined: (H, W)
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)

    # 归一化
    ndvi_norm = np.clip((ndvi + 0.2) / 1.0, 0, 1)  # NDVI -0.2~0.8
    smi_swir = calc_smi_swir(swir1)

    smi = ndvi_weight * ndvi_norm + (1 - ndvi_weight) * smi_swir
    return np.clip(smi, 0.0, 1.0).astype(np.float32)


# ============================================================
# 改进型垂直干旱指数 (MPDI)
# ============================================================

def calc_mpdi(
    red: np.ndarray,
    nir: np.ndarray,
) -> np.ndarray:
    """
    改进型垂直干旱指数 (Modified Perpendicular Drought Index)

    基于 NIR-Red 光谱空间:
      MPDI = (NIR + M * Red) / sqrt(1 + M²)

    其中 M 为土壤线斜率 (NIR_soil / Red_soil)

    参数:
        red: 红波段 (H, W)
        nir: 近红外 (H, W)

    返回:
        mpdi: (H, W), 值越高越干旱
    """
    red = np.asarray(red, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)

    if np.nanmedian(red) > 10:
        red /= 10000.0
        nir /= 10000.0

    # 自动估计土壤线斜率 (取低 NDVI 区的 Red-NIR 回归)
    ndvi_raw = np.where((nir + red) > 1e-6, (nir - red) / (nir + red), 0.0)
    soil_mask = (ndvi_raw > -0.1) & (ndvi_raw < 0.2) & np.isfinite(red) & np.isfinite(nir)

    if soil_mask.sum() > 100:
        red_soil = red[soil_mask]
        nir_soil = nir[soil_mask]
        M = np.polyfit(red_soil, nir_soil, 1)[0]
        M = max(0.5, min(M, 2.0))  # 约束
    else:
        M = 1.0

    mpdi = (nir + M * red) / np.sqrt(1 + M * M)
    return mpdi.astype(np.float32)


# ============================================================
# 分类与灌溉需求
# ============================================================

def classify_agri_drought(
    cwsi: np.ndarray,
    smi: Optional[np.ndarray] = None,
    method: str = "cwsi",
) -> np.ndarray:
    """
    农业干旱等级分类

    方法:
      "cwsi"  — 仅用 CWSI
      "composite" — CWSI + SMI 综合

    返回:
        category: (H, W) int8, 0-4
    """
    cwsi = np.asarray(cwsi, dtype=np.float64)
    category = np.zeros(cwsi.shape, dtype=np.int8)

    for level in AGRI_DROUGHT_LEVELS:
        lo, hi = level["cwsi_range"]
        if level["code"] > 0:
            category[(cwsi >= lo) & (cwsi < hi)] = level["code"]

    # 如有 SMI, 微调
    if smi is not None and method == "composite":
        smi_arr = np.asarray(smi, dtype=np.float64)
        # 极湿 → 降1级
        wet_mask = (smi_arr > 0.7) & (category >= 2)
        category[wet_mask] = np.maximum(category[wet_mask] - 1, 0)
        # 极干 → 升1级
        dry_mask = (smi_arr < 0.15) & (category <= 2)
        category[dry_mask] = np.minimum(category[dry_mask] + 1, 4)

    return category


def estimate_irrigation_demand(
    cwsi: np.ndarray,
    smi: Optional[np.ndarray] = None,
    crop_type: str = "general",
) -> Dict:
    """
    灌溉需求评估

    基于 CWSI 等级估算灌溉优先级和大致需水量

    参数:
        cwsi: (H, W) 作物水分胁迫指数
        smi: (H, W) 土壤水分指数 (可选)
        crop_type: 作物类型 ("general"/"wheat"/"corn"/"cotton")

    返回:
        dict: 灌溉需求统计
    """
    cwsi = np.asarray(cwsi, dtype=np.float64)

    # 作物系数 (简化的需水基准 mm/day)
    kc_map = {
        "general": 5.0,
        "wheat": 4.5,
        "corn": 6.0,
        "cotton": 5.5,
    }
    kc = kc_map.get(crop_type, 5.0)

    # 分区统计
    cwsi_valid = cwsi[np.isfinite(cwsi)]
    mean_cwsi = float(np.nanmean(cwsi_valid))

    # 各级面积
    level_areas = {}
    for level in AGRI_DROUGHT_LEVELS:
        mask = (cwsi >= level["cwsi_range"][0]) & (cwsi < level["cwsi_range"][1])
        level_areas[level["name"]] = float(mask.sum() / max(cwsi.size, 1))

    # 灌溉需求量估算 (简化: 日需水 = kc * CWSI)
    daily_demand = kc * mean_cwsi  # mm/day
    urgent_ratio = level_areas.get("重度干旱", 0) + level_areas.get("极度干旱", 0)

    # 灌溉建议
    if mean_cwsi < 0.2:
        advice = "水分充足，无需灌溉"
        priority = "低"
    elif mean_cwsi < 0.4:
        advice = "轻度缺水，建议按计划灌溉"
        priority = "中"
    elif mean_cwsi < 0.6:
        advice = "中度缺水，需增加灌溉频次"
        priority = "高"
    else:
        advice = "严重缺水，立即启动应急灌溉"
        priority = "紧急"

    return {
        "mean_cwsi": round(mean_cwsi, 4),
        "daily_water_demand_mm": round(daily_demand, 2),
        "urgent_area_ratio": round(urgent_ratio, 4),
        "irrigation_priority": priority,
        "irrigation_advice": advice,
        "crop_type": crop_type,
        "level_ratios": {k: round(v, 4) for k, v in level_areas.items()},
    }


# ============================================================
# 统计
# ============================================================

def compute_agri_stats(
    category: np.ndarray,
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """农业干旱分级统计"""
    total = np.sum(np.isfinite(category))
    stats = []
    for level in AGRI_DROUGHT_LEVELS:
        count = int(np.sum(category == level["code"]))
        stats.append({
            "code": level["code"], "name": level["name"],
            "pixel_count": count,
            "ratio": round(count / max(total, 1), 4),
            "area_km2": round(count * pixel_size_m ** 2 / 1e6, 2),
            "color": level["color"],
            "irrigation": level["irrigation"],
        })
    return stats


# ============================================================
# 一站式评估
# ============================================================

def assess_agri_drought(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    pixel_size_m: float = 10.0,
    crop_type: str = "general",
    ndvi: Optional[np.ndarray] = None,
) -> AgriDroughtResult:
    """
    一站式农业干旱评估

    参数:
        bands_data: (B, H, W) 6波段数据
        satellite: 卫星名称
        pixel_size_m: 像元大小
        crop_type: 作物类型
        ndvi: 预计算 NDVI

    返回:
        AgriDroughtResult
    """
    bands = np.asarray(bands_data, dtype=np.float64)
    red = bands[2]
    nir = bands[3]
    swir1 = bands[4]

    # NDVI
    if ndvi is None:
        denom = nir + red
        ndvi_val = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
        ndvi_val = np.clip(ndvi_val, -1.0, 1.0)
    else:
        ndvi_val = np.asarray(ndvi, dtype=np.float64)

    # CWSI
    cwsi = calc_cwsi_ndvi(ndvi_val)

    # SMI
    smi = calc_smi_swir(swir1)

    # MPDI
    mpdi = calc_mpdi(red, nir)

    # 分类
    category = classify_agri_drought(cwsi, smi=smi, method="composite")

    # 灌溉需求
    irrigation = estimate_irrigation_demand(cwsi, smi=smi, crop_type=crop_type)

    # 统计
    stats = compute_agri_stats(category, pixel_size_m)

    # 汇总
    drought_ratio = sum(s["ratio"] for s in stats if s["code"] >= 2)

    return AgriDroughtResult(
        cwsi=cwsi.astype(np.float32),
        smi=smi.astype(np.float32),
        mpdi=mpdi.astype(np.float32),
        ndvi=ndvi_val.astype(np.float32),
        category=category,
        irrigation_demand=None,
        stats=stats,
        summary={
            "mean_cwsi": round(float(np.nanmean(cwsi)), 4),
            "mean_smi": round(float(np.nanmean(smi)), 4),
            "moderate_plus_ratio": round(drought_ratio, 4),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"],
            "irrigation": irrigation,
        },
    )
