"""
生态安全评估模块
================
基于 PSR (压力-状态-响应) 框架的干旱区生态安全评价

PSR 指标:
  压力 (Pressure):  人类活动强度 (NDVI 减少 + 建设用地 + 干旱)
  状态 (State):      生态系统现状 (植被覆盖 + 水体面积 + 土壤水分)
  响应 (Response):   生态恢复能力 (NDVI 增加趋势 + 绿洲稳定性)

综合: ESI (生态安全指数) = w_p * P + w_s * S + w_r * R

参考:
  - OECD (1993) PSR 框架
  - 青海生态之窗 生态安全评估体系
  - 干旱区生态安全评价技术规范

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


ECO_SECURITY_LEVELS = [
    {"code": 0, "name": "安全", "esi_range": (0.7, np.inf),
     "description": "生态系统健康稳定", "color": "#2ecc71", "status": "优秀"},
    {"code": 1, "name": "较安全", "esi_range": (0.55, 0.7),
     "description": "生态系统基本稳定", "color": "#27ae60", "status": "良好"},
    {"code": 2, "name": "预警", "esi_range": (0.4, 0.55),
     "description": "生态压力增大", "color": "#f1c40f", "status": "关注"},
    {"code": 3, "name": "较不安全", "esi_range": (0.25, 0.4),
     "description": "生态系统退化中", "color": "#e67e22", "status": "警戒"},
    {"code": 4, "name": "不安全", "esi_range": (-np.inf, 0.25),
     "description": "生态系统严重退化", "color": "#e74c3c", "status": "危急"},
]


@dataclass
class EcoSecurityResult:
    psi: Optional[np.ndarray] = None       # 压力指数
    ssi: Optional[np.ndarray] = None       # 状态指数
    rsi: Optional[np.ndarray] = None       # 响应指数
    esi: Optional[np.ndarray] = None       # 综合生态安全指数
    category: Optional[np.ndarray] = None  # 等级分类
    stats: Optional[List[Dict]] = None
    summary: Dict = field(default_factory=dict)


def calc_psi(
    ndvi: np.ndarray,
    ndvi_decline: Optional[np.ndarray] = None,
    nddi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    压力指数 (Pressure Stress Index)

    PSI = 1 - 0.5 * NDVI_norm - 0.3 * Stability + 0.2 * Drought

    高 PSI = 高生态压力 (植被少/不稳定/干旱)

    参数:
        ndvi: (H,W) NDVI
        ndvi_decline: (H,W) NDVI 下降幅度
        nddi: (H,W) 干旱指数

    返回:
        psi: (H,W) 0~1
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    # NDVI 越低 → 压力越大
    ndvi_norm = 1.0 - np.clip((ndvi + 0.2) / 1.2, 0, 1)

    if ndvi_decline is not None:
        decline = np.clip(np.asarray(ndvi_decline, dtype=np.float64) / 0.3, 0, 1)
    else:
        decline = 0.3

    if nddi is not None:
        drought = np.clip(np.asarray(nddi, dtype=np.float64), 0, 1)
    else:
        drought = 0.2

    psi = 0.5 * ndvi_norm + 0.3 * decline + 0.2 * drought
    return np.clip(psi, 0, 1).astype(np.float32)


def calc_ssi(
    ndvi: np.ndarray,
    ndmi: Optional[np.ndarray] = None,
    water_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    状态指数 (State Security Index)

    SSI = 0.6 * NDVI_norm + 0.2 * Moisture + 0.2 * Water

    高 SSI = 生态状态好

    参数:
        ndvi: (H,W) NDVI
        ndmi: (H,W) 水分指数
        water_ratio: 水体面积占比

    返回:
        ssi: (H,W) 0~1
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    ndvi_norm = np.clip((ndvi + 0.2) / 1.2, 0, 1)

    if ndmi is not None:
        moisture = np.clip((np.asarray(ndmi, dtype=np.float64) + 0.3) / 0.8, 0, 1)
    else:
        moisture = 0.5

    wr = water_ratio if water_ratio else 0.1

    ssi = 0.6 * ndvi_norm + 0.2 * moisture + 0.2 * wr
    return np.clip(ssi, 0, 1).astype(np.float32)


def calc_rsi(
    ndvi_trend: np.ndarray,
    stability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    响应指数 (Response Security Index)

    RSI = 0.5 * Trend_norm + 0.5 * Stability

    高 RSI = 生态恢复能力强

    参数:
        ndvi_trend: (H,W) NDVI 变化趋势 (正=改善)
        stability: (H,W) 稳定性 (NDVI 标准差倒数)

    返回:
        rsi: (H,W) 0~1
    """
    trend = np.asarray(ndvi_trend, dtype=np.float64)
    trend_norm = np.clip((trend + 0.1) / 0.2, 0, 1)

    if stability is not None:
        stab = np.asarray(stability, dtype=np.float64)
        stab_norm = 1.0 - np.clip(stab / 0.15, 0, 1)
    else:
        stab_norm = 0.5

    rsi = 0.5 * trend_norm + 0.5 * stab_norm
    return np.clip(rsi, 0, 1).astype(np.float32)


def calc_esi(
    psi: np.ndarray,
    ssi: np.ndarray,
    rsi: np.ndarray,
    weights: Tuple[float, float, float] = (0.35, 0.40, 0.25),
) -> np.ndarray:
    """
    综合生态安全指数 (Ecological Security Index)

    ESI = w_p * (1-PSI) + w_s * SSI + w_r * RSI

    原理: PSI 是负向指标 (高=差), 需取反

    返回:
        esi: (H,W) 0~1, 值越高越安全
    """
    psi = np.asarray(psi, dtype=np.float64)
    ssi = np.asarray(ssi, dtype=np.float64)
    rsi = np.asarray(rsi, dtype=np.float64)

    wp, ws, wr = weights
    esi = wp * (1.0 - psi) + ws * ssi + wr * rsi
    return np.clip(esi, 0, 1).astype(np.float32)


def classify_eco_security(esi: np.ndarray) -> np.ndarray:
    """生态安全等级分类"""
    esi = np.asarray(esi, dtype=np.float64)
    cat = np.zeros(esi.shape, dtype=np.int8)
    for level in ECO_SECURITY_LEVELS:
        lo, hi = level["esi_range"]
        if level["code"] > 0:
            cat[(esi >= lo) & (esi < hi)] = level["code"]
    return cat


def compute_eco_stats(category: np.ndarray, pixel_size_m: float = 10.0) -> List[Dict]:
    """生态安全统计"""
    total = np.sum(np.isfinite(category))
    stats = []
    for lvl in ECO_SECURITY_LEVELS:
        ct = int(np.sum(category == lvl["code"]))
        stats.append({
            "code": lvl["code"], "name": lvl["name"],
            "pixel_count": ct, "ratio": round(ct / max(total, 1), 4),
            "area_km2": round(ct * pixel_size_m**2 / 1e6, 2),
            "color": lvl["color"], "status": lvl["status"],
        })
    return stats


def assess_eco_security(
    ndvi: np.ndarray,
    ndmi: Optional[np.ndarray] = None,
    nddi: Optional[np.ndarray] = None,
    ndvi_trend: Optional[np.ndarray] = None,
    water_ratio: float = 0.0,
    pixel_size_m: float = 10.0,
    weights: Tuple[float, float, float] = (0.35, 0.40, 0.25),
) -> EcoSecurityResult:
    """一站式 PSR 生态安全评估"""
    ndvi = np.asarray(ndvi, dtype=np.float64)
    H, W = ndvi.shape

    psi = calc_psi(ndvi, nddi=nddi)
    ssi = calc_ssi(ndvi, ndmi=ndmi, water_ratio=water_ratio)
    rsi = calc_rsi(ndvi_trend if ndvi_trend is not None else np.zeros_like(ndvi))
    esi = calc_esi(psi, ssi, rsi, weights=weights)
    category = classify_eco_security(esi)
    stats = compute_eco_stats(category, pixel_size_m)

    safe_ratio = sum(s["ratio"] for s in stats if s["code"] <= 1)
    unsafe_ratio = sum(s["ratio"] for s in stats if s["code"] >= 3)

    return EcoSecurityResult(
        psi=psi, ssi=ssi, rsi=rsi, esi=esi,
        category=category, stats=stats,
        summary={
            "esi_mean": round(float(np.nanmean(esi)), 4),
            "safe_ratio": round(safe_ratio, 4),
            "unsafe_ratio": round(unsafe_ratio, 4),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"],
            "overall_status": "安全" if safe_ratio > 0.7 else ("预警" if safe_ratio > 0.4 else "危急"),
        },
    )
