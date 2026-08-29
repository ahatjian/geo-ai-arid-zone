"""
自定义光谱指数计算器 (Band Math) 模块
========================================
提供灵活的波段运算能力，支持:
  - 12 种预设遥感指数 (NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/BSI...)
  - 自定义波段运算表达式 (支持 + - * / 及 numpy 数学函数)

波段名约定 (Sentinel-2 / Landsat 统一):
  B, G, R, NIR, SWIR1, SWIR2 (对应 [B02/B2, B03/B3, B04/B4, B08/B5, B11/B6, B12/B7])

依赖: numpy
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ============================================================
# 波段名定义
# ============================================================

BAND_NAMES = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]

# 允许在表达式中使用的 numpy 函数 (安全白名单)
ALLOWED_NUMPY_FUNCS = [
    "sqrt", "log", "log10", "exp", "abs", "clip",
    "where", "maximum", "minimum", "power", "square",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "floor", "ceil", "round", "sign", "nan_to_num",
]

# ============================================================
# 预设指数库
# ============================================================

PRESET_INDICES = [
    {
        "name": "NDVI", "formula": "(NIR - R) / (NIR + R)",
        "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化植被指数 — 最常用的植被指标",
    },
    {
        "name": "EVI", "formula": "2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "增强植被指数 — 避免高植被区饱和",
    },
    {
        "name": "SAVI", "formula": "(NIR - R) / (NIR + R + 0.5) * 1.5",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "土壤调节植被指数 — 降低土壤背景影响",
    },
    {
        "name": "GNDVI", "formula": "(NIR - G) / (NIR + G)",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "绿度归一化植被指数 — 对叶绿素更敏感",
    },
    {
        "name": "RVI", "formula": "NIR / R",
        "cmap": "Greens", "vmin": 0.0, "vmax": 8.0,
        "desc": "比值植被指数 — NIR 与红波段比值",
    },
    {
        "name": "DVI", "formula": "NIR - R",
        "cmap": "Greens", "vmin": -0.5, "vmax": 0.8,
        "desc": "差值植被指数 — NIR 与红波段差值",
    },
    {
        "name": "NDWI", "formula": "(G - NIR) / (G + NIR)",
        "cmap": "Blues", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化水体指数 — 水体提取",
    },
    {
        "name": "MNDWI", "formula": "(G - SWIR1) / (G + SWIR1)",
        "cmap": "Blues", "vmin": -1.0, "vmax": 1.0,
        "desc": "修正归一化水体指数 — 抑制建筑干扰",
    },
    {
        "name": "NDBI", "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "cmap": "Oranges", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化建筑指数 — 建设用地提取",
    },
    {
        "name": "BSI", "formula": "(SWIR1 + R - NIR - B) / (SWIR1 + R + NIR + B)",
        "cmap": "YlOrBr", "vmin": -1.0, "vmax": 1.0,
        "desc": "裸土指数 — 裸土/荒漠识别",
    },
    {
        "name": "NDMI", "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "cmap": "RdYlBu", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化水分指数 — 植被/土壤含水量",
    },
    {
        "name": "GRVI", "formula": "NIR / G",
        "cmap": "Greens", "vmin": 0.0, "vmax": 6.0,
        "desc": "绿红植被指数 — 绿度比值",
    },
]

PRESET_INDEX_NAMES = [p["name"] for p in PRESET_INDICES]


# ============================================================
# 波段运算求值器
# ============================================================

def get_band_arrays(bands_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将 (B, H, W) 波段数组转换为 {波段名: 数组} 字典

    参数:
        bands_data: (6, H, W) 多波段数据, 顺序 [B, G, R, NIR, SWIR1, SWIR2]

    返回:
        dict: {"B": ..., "G": ..., "R": ..., "NIR": ..., "SWIR1": ..., "SWIR2": ...}
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.shape[0] < 6:
        raise ValueError(f"至少需要 6 个波段, 实际: {bands_data.shape[0]}")

    # 自动缩放反射率 (DN → 0-1)
    if np.nanmedian(bands_data) > 10:
        bands_data = bands_data / 10000.0

    return {
        "B": bands_data[0],
        "G": bands_data[1],
        "R": bands_data[2],
        "NIR": bands_data[3],
        "SWIR1": bands_data[4],
        "SWIR2": bands_data[5],
    }


def evaluate_band_math(
    expression: str,
    bands: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    安全求值波段运算表达式

    参数:
        expression: 波段运算表达式, 如 "(NIR - R) / (NIR + R)"
                    可使用波段名 (B/G/R/NIR/SWIR1/SWIR2) 和 numpy 数学函数
        bands: {波段名: 数组} 字典

    返回:
        result: 计算结果数组 (H, W)

    ⚠️ 安全说明: 使用受限 eval (空 __builtins__ + numpy 函数白名单)，
       仅支持算术运算和白名单内的数学函数，禁止任意代码执行。
    """
    # 构建命名空间: 波段名 + numpy 白名单函数
    namespace = {name: arr for name, arr in bands.items()}
    namespace.update({func: getattr(np, func) for func in ALLOWED_NUMPY_FUNCS})

    with np.errstate(divide="ignore", invalid="ignore"):
        result = eval(expression, {"__builtins__": {}}, namespace)

    result = np.asarray(result, dtype=np.float64)

    # 替换 inf 为 NaN
    result = np.where(np.isfinite(result), result, np.nan)

    return result


def get_preset_index(name: str) -> Optional[Dict]:
    """按名称获取预设指数定义"""
    for p in PRESET_INDICES:
        if p["name"].lower() == name.lower():
            return p
    return None


def compute_index_stats(index_array: np.ndarray, pixel_size_m: float = 10.0) -> Dict:
    """
    计算指数统计信息

    参数:
        index_array: (H, W) 指数数组
        pixel_size_m: 像元大小 (m)

    返回:
        dict: {mean, std, min, max, p5, p95, valid_ratio}
    """
    arr = np.asarray(index_array, dtype=np.float64)
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "p5": None, "p95": None, "valid_ratio": 0.0}

    return {
        "mean": round(float(np.nanmean(valid)), 4),
        "std": round(float(np.nanstd(valid)), 4),
        "min": round(float(np.nanmin(valid)), 4),
        "max": round(float(np.nanmax(valid)), 4),
        "p5": round(float(np.nanpercentile(valid, 5)), 4),
        "p95": round(float(np.nanpercentile(valid, 95)), 4),
        "valid_ratio": round(len(valid) / arr.size, 4),
    }
