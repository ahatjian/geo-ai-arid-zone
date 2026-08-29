"""
蒸散发 (ET) 估算模块
=====================
基于 SEBAL 简化能量平衡法的地表蒸散发估算 (干旱区水资源平衡核心)

能量平衡: Rn = H + LE + G
  Rn: 净辐射 (Net Radiation)
  H:  感热通量 (Sensible Heat Flux)
  LE: 潜热通量 (Latent Heat Flux, 即蒸散发能量)
  G:  土壤热通量 (Soil Heat Flux)

核心流程:
  反照率 + NDVI + LST → 发射率 → 净辐射 → 土壤热通量 → 感热通量 → 潜热通量 → 日蒸散发

参考:
  - Bastiaanssen et al. (1998) SEBAL 遥感能量平衡算法
  - Allen et al. (2007) METRIC 算法
  - Liang (2001) 宽带反照率
  - FAO-56 蒸散发理论

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
# 物理常量
# ============================================================

STEFAN_BOLTZMANN = 5.67e-8          # σ, W/(m²·K⁴)
AIR_DENSITY = 1.15                   # 空气密度 ρ, kg/m³
AIR_SPECIFIC_HEAT = 1004.0           # 空气定压比热 cp, J/(kg·K)
LATENT_HEAT_VAPOR = 2.45e6           # 汽化潜热 λ, J/kg
SECONDS_PER_DAY = 86400              # 秒/天

# ============================================================
# 蒸散发等级定义 (5 级)
# ============================================================

ET_LEVELS = [
    {
        "code": 0, "name": "极低蒸散发",
        "et_range": (-np.inf, 0.5),
        "description": "ET < 0.5 mm/day，沙漠/裸地/戈壁",
        "color": "#8b0000",   # 深红
        "risk": "缺水",
    },
    {
        "code": 1, "name": "低蒸散发",
        "et_range": (0.5, 1.5),
        "description": "0.5-1.5 mm/day，稀疏植被/荒漠草原",
        "color": "#e67e22",   # 橙
        "risk": "较缺水",
    },
    {
        "code": 2, "name": "中等蒸散发",
        "et_range": (1.5, 3.0),
        "description": "1.5-3.0 mm/day，草地/农田",
        "color": "#f1c40f",   # 黄
        "risk": "正常",
    },
    {
        "code": 3, "name": "高蒸散发",
        "et_range": (3.0, 5.0),
        "description": "3.0-5.0 mm/day，灌溉农田/绿洲",
        "color": "#2ecc71",   # 绿
        "risk": "湿润",
    },
    {
        "code": 4, "name": "极高蒸散发",
        "et_range": (5.0, np.inf),
        "description": "ET > 5.0 mm/day，水体/湿地/茂密植被",
        "color": "#0066cc",   # 蓝
        "risk": "丰水",
    },
]

# ET 分级阈值 (mm/day): [低/中等/高/极高分界]
DEFAULT_ET_THRESHOLDS = [0.5, 1.5, 3.0, 5.0]


@dataclass
class ETResult:
    """蒸散发估算结果"""
    albedo: Optional[np.ndarray] = None         # 地表反照率
    ndvi: Optional[np.ndarray] = None           # NDVI
    emissivity: Optional[np.ndarray] = None     # 地表发射率
    lst_celsius: Optional[np.ndarray] = None    # 地表温度 (°C)
    net_radiation: Optional[np.ndarray] = None  # 净辐射 Rn (W/m²)
    soil_heat_flux: Optional[np.ndarray] = None # 土壤热通量 G (W/m²)
    sensible_heat: Optional[np.ndarray] = None  # 感热通量 H (W/m²)
    latent_heat: Optional[np.ndarray] = None    # 潜热通量 LE (W/m²)
    et_daily: Optional[np.ndarray] = None       # 日蒸散发 (mm/day)
    category: Optional[np.ndarray] = None       # (H, W) 0-4 ET 分级
    category_colors: Optional[np.ndarray] = None
    stats: Optional[List[Dict]] = None
    summary: Dict = field(default_factory=dict)


# ============================================================
# 中间参数计算
# ============================================================

def calc_fvc(ndvi: np.ndarray, ndvi_soil: float = 0.2, ndvi_veg: float = 0.5) -> np.ndarray:
    """
    计算植被覆盖度 FVC (Fractional Vegetation Cover)

    FVC = ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil))²

    参数:
        ndvi: NDVI 数组
        ndvi_soil: 裸土 NDVI 阈值 (默认 0.2)
        ndvi_veg: 全植被 NDVI 阈值 (默认 0.5)

    返回:
        fvc: (H, W), 范围 [0, 1]
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    fvc = ((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)) ** 2
    fvc = np.clip(fvc, 0.0, 1.0)
    return fvc.astype(np.float32)


def calc_emissivity(ndvi: np.ndarray, ndvi_soil: float = 0.2, ndvi_veg: float = 0.5) -> np.ndarray:
    """
    计算地表发射率 ε (Surface Emissivity)

    ε = 0.004 × FVC + 0.986 (裸土 ε≈0.986, 全植被 ε≈0.99)

    参数:
        ndvi: NDVI 数组

    返回:
        emissivity: (H, W)
    """
    fvc = calc_fvc(ndvi, ndvi_soil=ndvi_soil, ndvi_veg=ndvi_veg)
    emissivity = 0.004 * fvc + 0.986
    return emissivity.astype(np.float32)


def calc_atmospheric_emissivity(transmittance: float = 0.75) -> float:
    """
    计算大气有效发射率 εa (Bastiaanssen 1998 经验公式)

    εa = 1.08 × (-ln(τ_sw))^0.265

    参数:
        transmittance: 大气短波透过率 τ_sw (晴天 0.6-0.8, 默认 0.75)

    返回:
        eps_a: 大气发射率
    """
    tau = max(transmittance, 0.1)
    eps_a = 1.08 * (-np.log(tau)) ** 0.265
    return float(eps_a)


# ============================================================
# 能量平衡分量
# ============================================================

def calc_net_radiation(
    albedo: np.ndarray,
    lst_kelvin: np.ndarray,
    ndvi: np.ndarray,
    rs_down: float = 600.0,
    ta_kelvin: float = 298.15,
    transmittance: float = 0.75,
) -> np.ndarray:
    """
    计算净辐射 Rn (Net Radiation, W/m²)

    Rn = (1 - albedo) × Rs↓ + εa × σ × Ta⁴ - ε × σ × Ts⁴

    参数:
        albedo: 地表反照率 (0-1)
        lst_kelvin: 地表温度 (K)
        ndvi: NDVI (用于计算发射率)
        rs_down: 太阳短波入射辐射 Rs↓ (W/m², 默认 600)
        ta_kelvin: 空气温度 Ta (K, 默认 25°C = 298.15)
        transmittance: 大气短波透过率 (默认 0.75)

    返回:
        rn: (H, W) 净辐射 (W/m²)
    """
    albedo = np.asarray(albedo, dtype=np.float64)
    lst = np.asarray(lst_kelvin, dtype=np.float64)
    ndvi = np.asarray(ndvi, dtype=np.float64)

    emissivity = calc_emissivity(ndvi)
    eps_atm = calc_atmospheric_emissivity(transmittance)

    # 短波净辐射
    rn_short = (1.0 - albedo) * rs_down

    # 长波净辐射
    rl_in = eps_atm * STEFAN_BOLTZMANN * ta_kelvin ** 4
    rl_out = emissivity * STEFAN_BOLTZMANN * lst ** 4
    rn_long = rl_in - rl_out

    rn = rn_short + rn_long
    return rn.astype(np.float32)


def calc_soil_heat_flux(
    rn: np.ndarray,
    albedo: np.ndarray,
    ndvi: np.ndarray,
    lst_celsius: np.ndarray,
) -> np.ndarray:
    """
    计算土壤热通量 G (Soil Heat Flux, W/m²)

    G = Rn × (Ts - 273.15) / albedo × (0.0038 × albedo + 0.0074 × albedo²) × (1 - 0.98 × NDVI⁴)

    参数:
        rn: 净辐射 (W/m²)
        albedo: 地表反照率
        ndvi: NDVI
        lst_celsius: 地表温度 (°C)

    返回:
        g: (H, W) 土壤热通量 (W/m²)
    """
    rn = np.asarray(rn, dtype=np.float64)
    albedo = np.asarray(albedo, dtype=np.float64)
    ndvi = np.asarray(ndvi, dtype=np.float64)
    ts_c = np.asarray(lst_celsius, dtype=np.float64)

    albedo_safe = np.where(albedo < 0.05, 0.05, albedo)
    g = rn * (ts_c / albedo_safe) * (0.0038 * albedo_safe + 0.0074 * albedo_safe ** 2) * (1 - 0.98 * ndvi ** 4)
    return g.astype(np.float32)


def calc_sensible_heat(
    lst_kelvin: np.ndarray,
    ta_kelvin: float = 298.15,
    wind_speed: float = 2.0,
    aerodynamic_resistance: Optional[float] = None,
) -> np.ndarray:
    """
    计算感热通量 H (Sensible Heat Flux, W/m²)

    H = ρ × cp × (Ts - Ta) / ra

    参数:
        lst_kelvin: 地表温度 (K)
        ta_kelvin: 空气温度 (K, 默认 25°C)
        wind_speed: 风速 (m/s, 用于估算 ra)
        aerodynamic_resistance: 空气动力学阻力 ra (s/m), None=按风速估算 ra=208/u

    返回:
        h: (H, W) 感热通量 (W/m²)
    """
    lst = np.asarray(lst_kelvin, dtype=np.float64)

    if aerodynamic_resistance is None:
        # ra = 208 / u (Allen 2007 简化)
        u_safe = max(wind_speed, 0.5)
        ra = 208.0 / u_safe
    else:
        ra = aerodynamic_resistance

    h = AIR_DENSITY * AIR_SPECIFIC_HEAT * (lst - ta_kelvin) / ra
    return h.astype(np.float32)


def calc_latent_heat(rn: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    计算潜热通量 LE (Latent Heat Flux, W/m²)

    LE = Rn - G - H

    参数:
        rn: 净辐射 (W/m²)
        g: 土壤热通量 (W/m²)
        h: 感热通量 (W/m²)

    返回:
        le: (H, W) 潜热通量 (W/m²)
    """
    rn = np.asarray(rn, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    le = rn - g - h
    return le.astype(np.float32)


def calc_et_daily(le: np.ndarray, method: str = "instant") -> np.ndarray:
    """
    由潜热通量计算日蒸散发 ET (mm/day)

    ET = LE × 86400 / λ

    参数:
        le: 潜热通量 (W/m² = J/(s·m²))
        method: "instant" (瞬时外推日尺度, 默认)

    返回:
        et_daily: (H, W) 日蒸散发 (mm/day)
    """
    le = np.asarray(le, dtype=np.float64)
    # ET(mm/day) = LE(J/(s·m²)) × 86400(s) / λ(J/kg) = kg/m² = mm (1mm 水 = 1kg/m²)
    et = le * SECONDS_PER_DAY / LATENT_HEAT_VAPOR
    return et.astype(np.float32)


# ============================================================
# ET 分级
# ============================================================

def classify_et(
    et_daily: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> np.ndarray:
    """
    蒸散发分级 (5 级)

    参数:
        et_daily: 日蒸散发 (mm/day)
        thresholds: [t1, t2, t3, t4] 低/中等/高/极高分界, 默认 [0.5, 1.5, 3.0, 5.0]

    返回:
        category: (H, W) uint8, 0-4
    """
    if thresholds is None:
        thresholds = DEFAULT_ET_THRESHOLDS
    t1, t2, t3, t4 = thresholds

    et = np.asarray(et_daily, dtype=np.float64)
    H, W = et.shape

    category = np.zeros((H, W), dtype=np.uint8)
    category[(et >= t1) & (et < t2)] = 1
    category[(et >= t2) & (et < t3)] = 2
    category[(et >= t3) & (et < t4)] = 3
    category[et >= t4] = 4

    invalid = ~np.isfinite(et)
    category[invalid] = 0

    return category


def compute_et_stats(
    category: np.ndarray,
    et_daily: np.ndarray,
    pixel_size_m: float = 30.0,
) -> List[Dict]:
    """计算蒸散发分级统计"""
    category = np.asarray(category, dtype=np.int8)
    et = np.asarray(et_daily, dtype=np.float64)
    total_valid = np.sum(np.isfinite(et))

    stats = []
    for level in ET_LEVELS:
        code = level["code"]
        mask = category == code
        count = np.sum(mask)
        ratio = count / max(total_valid, 1)
        area = count * (pixel_size_m ** 2) / 1e6

        zone_mean = float(np.nanmean(et[mask])) if count > 0 else None

        stats.append({
            "code": code,
            "name": level["name"],
            "pixel_count": int(count),
            "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": level["color"],
            "risk": level["risk"],
            "description": level["description"],
            "mean_et": round(zone_mean, 2) if zone_mean is not None else None,
        })

    return stats


def get_et_colormap(category: np.ndarray) -> np.ndarray:
    """生成蒸散发分级 RGB 着色图"""
    category = np.asarray(category, dtype=np.uint8)
    H, W = category.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for level in ET_LEVELS:
        code = level["code"]
        hex_color = level["color"].lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        mask = category == code
        rgb[mask] = [r, g, b]

    return rgb


# ============================================================
# 一站式评估
# ============================================================

def assess_et(
    albedo: np.ndarray,
    ndvi: np.ndarray,
    lst_kelvin: np.ndarray,
    pixel_size_m: float = 30.0,
    rs_down: float = 600.0,
    ta_celsius: float = 25.0,
    wind_speed: float = 2.0,
    transmittance: float = 0.75,
    thresholds: Optional[List[float]] = None,
) -> ETResult:
    """
    一站式蒸散发估算 (SEBAL 简化能量平衡法)

    参数:
        albedo: 地表反照率 (0-1)
        ndvi: NDVI
        lst_kelvin: 地表温度 (K)
        pixel_size_m: 像元大小 (m)
        rs_down: 太阳短波入射辐射 (W/m², 默认 600)
        ta_celsius: 空气温度 (°C, 默认 25)
        wind_speed: 风速 (m/s, 默认 2)
        transmittance: 大气短波透过率 (默认 0.75)
        thresholds: ET 分级阈值 (mm/day)

    返回:
        ETResult
    """
    albedo = np.asarray(albedo, dtype=np.float64)
    ndvi = np.asarray(ndvi, dtype=np.float64)
    lst_k = np.asarray(lst_kelvin, dtype=np.float64)

    ta_kelvin = ta_celsius + 273.15
    lst_celsius = lst_k - 273.15

    emissivity = calc_emissivity(ndvi)

    # 能量平衡分量
    rn = calc_net_radiation(albedo, lst_k, ndvi, rs_down=rs_down,
                            ta_kelvin=ta_kelvin, transmittance=transmittance)
    g = calc_soil_heat_flux(rn, albedo, ndvi, lst_celsius)
    h = calc_sensible_heat(lst_k, ta_kelvin=ta_kelvin, wind_speed=wind_speed)
    le = calc_latent_heat(rn, g, h)
    et_daily = calc_et_daily(le)

    # 分级
    category = classify_et(et_daily, thresholds=thresholds)
    stats = compute_et_stats(category, et_daily, pixel_size_m=pixel_size_m)
    colors = get_et_colormap(category)

    # 汇总
    valid = et_daily[np.isfinite(et_daily)]
    mean_et = float(np.nanmean(valid)) if valid.size > 0 else None
    total_et_volume = float(np.nansum(et_daily) * pixel_size_m ** 2 / 1e6) if valid.size > 0 else None  # 万m³/day

    return ETResult(
        albedo=albedo.astype(np.float32),
        ndvi=ndvi.astype(np.float32),
        emissivity=emissivity.astype(np.float32),
        lst_celsius=lst_celsius.astype(np.float32),
        net_radiation=rn,
        soil_heat_flux=g,
        sensible_heat=h,
        latent_heat=le,
        et_daily=et_daily,
        category=category,
        category_colors=colors,
        stats=stats,
        summary={
            "mean_et": round(mean_et, 3) if mean_et is not None else None,
            "total_et_volume_10km3": round(total_et_volume / 1e4, 3) if total_et_volume else None,
            "mean_rn": round(float(np.nanmean(rn)), 1),
            "mean_le": round(float(np.nanmean(le)), 1),
            "mean_h": round(float(np.nanmean(h)), 1),
            "mean_g": round(float(np.nanmean(g)), 1),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"] if stats else "未知",
            "pixel_size_m": pixel_size_m,
            "rs_down": rs_down,
            "ta_celsius": ta_celsius,
            "wind_speed": wind_speed,
        },
    )


@_cache(ttl=600)
def assess_et_cached(
    et_hash: str,
    albedo: np.ndarray,
    ndvi: np.ndarray,
    lst_kelvin: np.ndarray,
    pixel_size_m: float = 30.0,
    **kwargs,
) -> ETResult:
    """带缓存的蒸散发估算"""
    return assess_et(albedo, ndvi, lst_kelvin, pixel_size_m=pixel_size_m, **kwargs)
