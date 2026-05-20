"""
Sen+MK 趋势分析模块
支持逐像元时序趋势分析、均值时序趋势分析
依赖: numpy, scipy, pymannkendall (可选)
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


# ============================================
# 核心: Sen 斜率计算 (无需 pymannkendall)
# ============================================

def theil_sen_slope(ts: np.ndarray, times: Optional[np.ndarray] = None) -> float:
    """
    Theil-Sen 斜率估计 (对一维时序)

    参数:
        ts: 一维时序数组
        times: 时间索引 (None=等距 0..n-1)

    返回:
        float: Theil-Sen 斜率
    """
    ts = np.asarray(ts, dtype=np.float64).flatten()
    n = len(ts)

    if n < 3:
        return 0.0

    if times is None:
        times = np.arange(n, dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64).flatten()

    # 过滤 NaN
    mask = np.isfinite(ts) & np.isfinite(times)
    ts = ts[mask]
    times = times[mask]
    n = len(ts)

    if n < 3:
        return 0.0

    # 计算所有 pairwise 斜率
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if dt != 0:
                slopes.append((ts[j] - ts[i]) / dt)

    if not slopes:
        return 0.0

    return float(np.median(slopes))


# ============================================
# 逐像元 Sen+MK 趋势分析
# ============================================

def calc_sen_mk_trend_pixelwise(
    ts_stack: np.ndarray,
    times: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    对多维时序 (height, width, time) 逐像元计算 Sen 斜率 + Mann-Kendall 检验

    参数:
        ts_stack: (height, width, time) 时序影像堆叠
        times: 时间索引 (None=等距)
        alpha: 显著性水平 (默认 0.05)

    返回:
        dict: {
            "slope": (H, W) Sen 斜率,
            "trend": (H, W) 趋势方向 {-1=减少, 0=不变, 1=增加},
            "p_value": (H, W) MK p 值,
            "significant": (H, W) bool 是否显著,
        }
    """
    H, W, T = ts_stack.shape

    if times is None:
        times = np.arange(T, dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64)

    # 预分配
    slope = np.full((H, W), np.nan, dtype=np.float32)
    trend = np.zeros((H, W), dtype=np.int8)
    p_value = np.full((H, W), np.nan, dtype=np.float32)
    significant = np.zeros((H, W), dtype=bool)

    try:
        import pymannkendall as mk
        has_mk = True
    except ImportError:
        has_mk = False

    for i in range(H):
        for j in range(W):
            ts = ts_stack[i, j, :].astype(np.float64)

            # 过滤无效值
            valid = np.isfinite(ts)
            if valid.sum() < 3:
                continue

            ts_valid = ts[valid]
            times_valid = times[valid]

            # Theil-Sen 斜率
            s = theil_sen_slope(ts_valid, times_valid)
            slope[i, j] = s

            # Mann-Kendall 检验
            if has_mk:
                try:
                    result = mk.original_test(ts_valid)
                    p_value[i, j] = result.p
                    if result.trend == "increasing":
                        trend[i, j] = 1
                    elif result.trend == "decreasing":
                        trend[i, j] = -1
                    else:
                        trend[i, j] = 0
                    significant[i, j] = result.p < alpha
                except Exception:
                    # MK 失败时回退到仅用斜率判断
                    trend[i, j] = 1 if s > 0 else (-1 if s < 0 else 0)
                    p_value[i, j] = np.nan
            else:
                # 无 pymannkendall，仅用斜率判断
                trend[i, j] = 1 if s > 0 else (-1 if s < 0 else 0)
                p_value[i, j] = np.nan

    return {
        "slope": slope,
        "trend": trend,
        "p_value": p_value,
        "significant": significant,
    }


# ============================================
# 均值时序 Sen+MK (简化版，用于统计面板)
# ============================================

def calc_sen_mk_trend(
    ts: np.ndarray,
    times: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    对一维时序 (如研究区平均 NDVI 时序) 计算 Sen 斜率 + MK 检验

    参数:
        ts: 一维时序数组
        times: 时间索引 (None=等距)
        alpha: 显著性水平

    返回:
        dict: {"slope", "trend", "p_value", "significant", "intercept"}
    """
    ts = np.asarray(ts, dtype=np.float64).flatten()

    if times is None:
        times = np.arange(len(ts), dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64).flatten()

    # 过滤 NaN
    mask = np.isfinite(ts) & np.isfinite(times)
    ts = ts[mask]
    times = times[mask]

    n = len(ts)
    if n < 3:
        return {
            "slope": 0.0,
            "trend": "no trend",
            "p_value": np.nan,
            "significant": False,
            "intercept": np.nan,
            "n_valid": n,
        }

    # Theil-Sen 斜率
    slope = theil_sen_slope(ts, times)

    # 截距 = 中位数 (ts - slope * times)
    intercept = float(np.median(ts - slope * times))

    # Mann-Kendall
    try:
        import pymannkendall as mk
        result = mk.original_test(ts)
        return {
            "slope": slope,
            "trend": result.trend,
            "p_value": result.p,
            "significant": result.p < alpha,
            "intercept": intercept,
            "n_valid": n,
        }
    except ImportError:
        trend_str = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "no trend")
        return {
            "slope": slope,
            "trend": trend_str,
            "p_value": np.nan,
            "significant": False,
            "intercept": intercept,
            "n_valid": n,
        }


# ============================================
# 均值时序趋势分析 (多时相研究区均值)
# ============================================

def calc_mean_timeseries_trend(
    ts_stack: np.ndarray,
    times: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    计算研究区均值时序的 Sen+MK 趋势

    参数:
        ts_stack: (height, width, time) 时序影像
        times: 时间索引
        alpha: 显著性水平

    返回:
        dict: {"slope", "trend", "p_value", "significant", "mean_ts", "n"}
    """
    # 每个时相的均值和有效像元数
    T = ts_stack.shape[2]
    mean_ts = np.full(T, np.nan, dtype=np.float64)
    n_valid = np.zeros(T, dtype=int)

    for t in range(T):
        band = ts_stack[:, :, t]
        valid = band[np.isfinite(band)]
        mean_ts[t] = valid.mean() if len(valid) > 0 else np.nan
        n_valid[t] = len(valid)

    if times is None:
        times = np.arange(T, dtype=np.float64)

    result = calc_sen_mk_trend(mean_ts, times, alpha)
    result["mean_ts"] = mean_ts
    result["n_valid"] = n_valid

    return result
