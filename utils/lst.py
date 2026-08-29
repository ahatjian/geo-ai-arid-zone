"""
地表温度 (LST) 反演与热环境分析模块
======================================
基于 Landsat Collection 2 Level-2 地表温度产品 (ST_B10 / ST_B6)

功能:
  - LST 反演 (DN → Kelvin → Celsius)
  - 热环境 5 级分级 (低温/较低温/常温/较高温/高温)
  - 热环境统计 (各温区面积、占比、均值)
  - LST-NDVI 关系 (热岛与植被响应)

参考:
  - USGS Collection 2 Level-2 Surface Temperature 产品规范
  - Qin et al. (2001) 单窗算法
  - 干旱区热环境 / 城市热岛遥感监测

依赖: numpy, rioxarray (读取 STAC 资产)
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
# 热环境等级定义 (5 级)
# ============================================================

THERMAL_LEVELS = [
    {
        "code": 0, "name": "低温区",
        "lst_range": (-np.inf, 10.0),
        "description": "地表温度 < 10°C，多为水体/高寒区",
        "color": "#2166ac",   # 深蓝
        "risk": "冷源",
    },
    {
        "code": 1, "name": "较低温区",
        "lst_range": (10.0, 20.0),
        "description": "10-20°C，植被覆盖较好的绿洲/农田",
        "color": "#74add1",   # 浅蓝
        "risk": "凉爽",
    },
    {
        "code": 2, "name": "常温区",
        "lst_range": (20.0, 30.0),
        "description": "20-30°C，一般裸地/稀疏植被",
        "color": "#fee090",   # 浅黄
        "risk": "正常",
    },
    {
        "code": 3, "name": "较高温区",
        "lst_range": (30.0, 40.0),
        "description": "30-40°C，裸露地表/荒漠边缘",
        "color": "#f46d43",   # 橙红
        "risk": "偏热",
    },
    {
        "code": 4, "name": "高温区",
        "lst_range": (40.0, np.inf),
        "description": "地表温度 > 40°C，戈壁/沙漠/热岛核心",
        "color": "#b2182b",   # 深红
        "risk": "极热",
    },
]

# 热环境分级阈值 (°C): [低温/较低温/常温/较高温分界]
DEFAULT_THERMAL_THRESHOLDS = [10.0, 20.0, 30.0, 40.0]


@dataclass
class LSTResult:
    """地表温度分析结果"""
    lst_kelvin: Optional[np.ndarray] = None      # (H, W) 开尔文
    lst_celsius: Optional[np.ndarray] = None     # (H, W) 摄氏度
    category: Optional[np.ndarray] = None        # (H, W) 0-4 热环境分级
    category_colors: Optional[np.ndarray] = None  # (H, W, 3) RGB
    stats: Optional[List[Dict]] = None           # 分级统计
    summary: Dict = field(default_factory=dict)


# ============================================================
# LST 读取与反演
# ============================================================

def read_lst_array(item, collection: str = "Landsat-8") -> np.ndarray:
    """
    从 STAC Item 读取 Landsat 地表温度产品并反演为开尔文温度

    参数:
        item: STAC Item 对象 (需含 ST 资产)
        collection: 数据集名称 (Landsat-8/9/7/4-5)

    返回:
        lst_kelvin: (H, W) 地表温度 (Kelvin), 无效像元为 NaN
    """
    from config import LANDSAT_ST_ASSET, LANDSAT_ST_SCALE, LANDSAT_ST_OFFSET
    import rioxarray

    asset_name = LANDSAT_ST_ASSET[collection]
    href = item.assets[asset_name].href

    data = rioxarray.open_rasterio(href).squeeze()
    dn = data.values.astype(np.float32)

    # 反演: Kelvin = DN * scale + offset
    lst_kelvin = dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET

    # 无数据像元 (DN == 0) 设为 NaN
    lst_kelvin = np.where(dn <= 0, np.nan, lst_kelvin)

    return lst_kelvin.astype(np.float32)


def kelvin_to_celsius(lst_kelvin: np.ndarray) -> np.ndarray:
    """开尔文转摄氏度"""
    return lst_kelvin - 273.15


# ============================================================
# 分级
# ============================================================

def classify_thermal(
    lst_celsius: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> np.ndarray:
    """
    地表热环境分级 (5 级)

    参数:
        lst_celsius: (H, W) 地表温度 (°C)
        thresholds: [t1, t2, t3, t4] 低温/较低温/常温/较高温分界, 默认 [10,20,30,40]

    返回:
        category: (H, W) uint8, 0-4
    """
    if thresholds is None:
        thresholds = DEFAULT_THERMAL_THRESHOLDS
    t1, t2, t3, t4 = thresholds

    lst = np.asarray(lst_celsius, dtype=np.float64)
    H, W = lst.shape

    category = np.zeros((H, W), dtype=np.uint8)
    category[(lst >= t1) & (lst < t2)] = 1
    category[(lst >= t2) & (lst < t3)] = 2
    category[(lst >= t3) & (lst < t4)] = 3
    category[lst >= t4] = 4

    # 无效值归为低温区 (0)
    invalid = ~np.isfinite(lst)
    category[invalid] = 0

    return category


def compute_lst_stats(
    category: np.ndarray,
    lst_celsius: np.ndarray,
    pixel_size_m: float = 30.0,
) -> List[Dict]:
    """
    计算热环境分级统计

    参数:
        category: (H, W) 分级图 (0-4)
        lst_celsius: (H, W) 地表温度 (°C)
        pixel_size_m: 像元大小 (m), Landsat=30

    返回:
        stats: 每级统计列表
    """
    category = np.asarray(category, dtype=np.int8)
    lst = np.asarray(lst_celsius, dtype=np.float64)
    total_valid = np.sum(np.isfinite(lst))

    stats = []
    for level in THERMAL_LEVELS:
        code = level["code"]
        mask = category == code
        count = np.sum(mask)
        ratio = count / max(total_valid, 1)
        area = count * (pixel_size_m ** 2) / 1e6  # km²

        # 该温区平均地表温度
        zone_mean = float(np.nanmean(lst[mask])) if count > 0 else None

        stats.append({
            "code": code,
            "name": level["name"],
            "pixel_count": int(count),
            "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": level["color"],
            "risk": level["risk"],
            "description": level["description"],
            "mean_lst_c": round(zone_mean, 2) if zone_mean is not None else None,
        })

    return stats


def get_thermal_colormap(category: np.ndarray) -> np.ndarray:
    """生成热环境分级 RGB 着色图"""
    category = np.asarray(category, dtype=np.uint8)
    H, W = category.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for level in THERMAL_LEVELS:
        code = level["code"]
        hex_color = level["color"].lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        mask = category == code
        rgb[mask] = [r, g, b]

    return rgb


# ============================================================
# 一站式评估
# ============================================================

def assess_thermal(
    lst_kelvin: np.ndarray,
    pixel_size_m: float = 30.0,
    thresholds: Optional[List[float]] = None,
) -> LSTResult:
    """
    一站式地表热环境评估

    参数:
        lst_kelvin: (H, W) 地表温度 (Kelvin)
        pixel_size_m: 像元大小 (m)
        thresholds: 热环境分级阈值 (°C)

    返回:
        LSTResult
    """
    lst_kelvin = np.asarray(lst_kelvin, dtype=np.float64)
    lst_celsius = lst_kelvin - 273.15

    category = classify_thermal(lst_celsius, thresholds=thresholds)
    stats = compute_lst_stats(category, lst_celsius, pixel_size_m=pixel_size_m)
    colors = get_thermal_colormap(category)

    # 汇总
    hot_ratio = sum(s["ratio"] for s in stats if s["code"] >= 3)
    cold_ratio = sum(s["ratio"] for s in stats if s["code"] <= 1)

    valid = lst_celsius[np.isfinite(lst_celsius)]
    mean_lst = float(np.nanmean(valid)) if valid.size > 0 else None
    max_lst = float(np.nanmax(valid)) if valid.size > 0 else None
    min_lst = float(np.nanmin(valid)) if valid.size > 0 else None

    return LSTResult(
        lst_kelvin=lst_kelvin.astype(np.float32),
        lst_celsius=lst_celsius.astype(np.float32),
        category=category,
        category_colors=colors,
        stats=stats,
        summary={
            "mean_lst_c": round(mean_lst, 2) if mean_lst is not None else None,
            "max_lst_c": round(max_lst, 2) if max_lst is not None else None,
            "min_lst_c": round(min_lst, 2) if min_lst is not None else None,
            "hot_ratio": round(hot_ratio, 4),
            "cold_ratio": round(cold_ratio, 4),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"] if stats else "未知",
            "pixel_size_m": pixel_size_m,
        },
    )


# ============================================================
# LST-NDVI 关系 (热岛与植被响应)
# ============================================================

def compute_lst_ndvi_relation(
    lst_celsius: np.ndarray,
    ndvi: np.ndarray,
    bins: int = 20,
) -> Dict:
    """
    计算 LST 与 NDVI 的关系 (热环境与植被覆盖)

    参数:
        lst_celsius: (H, W) 地表温度 (°C)
        ndvi: (H, W) NDVI
        bins: 分箱数

    返回:
        dict: {
            "ndvi_bins": 各分箱 NDVI 中心值,
            "mean_lst": 各分箱平均 LST,
            "correlation": 相关系数,
        }
    """
    lst = np.asarray(lst_celsius, dtype=np.float64).flatten()
    nd = np.asarray(ndvi, dtype=np.float64).flatten()

    mask = np.isfinite(lst) & np.isfinite(nd)
    lst = lst[mask]
    nd = nd[mask]

    if len(lst) < 10:
        return {"ndvi_bins": [], "mean_lst": [], "correlation": None}

    # 分箱统计
    edges = np.linspace(-0.2, 1.0, bins + 1)
    ndvi_bins = []
    mean_lst = []
    for i in range(bins):
        sel = (nd >= edges[i]) & (nd < edges[i + 1])
        if np.sum(sel) > 0:
            ndvi_bins.append((edges[i] + edges[i + 1]) / 2)
            mean_lst.append(float(np.mean(lst[sel])))

    # 相关系数
    corr = float(np.corrcoef(lst, nd)[0, 1]) if len(lst) > 2 else None

    return {
        "ndvi_bins": ndvi_bins,
        "mean_lst": mean_lst,
        "correlation": round(corr, 4) if corr is not None else None,
    }


@_cache(ttl=600)
def assess_thermal_cached(
    lst_hash: str,
    lst_kelvin: np.ndarray,
    pixel_size_m: float = 30.0,
    **kwargs,
) -> LSTResult:
    """带缓存的热环境评估 (lst_hash 用于区分不同影像)"""
    return assess_thermal(lst_kelvin, pixel_size_m=pixel_size_m, **kwargs)
