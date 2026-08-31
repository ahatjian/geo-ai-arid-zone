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


# ============================================================
# Savitzky-Golay 时序平滑 (NDVI 时序重建标准方法)
# ============================================================

def savgol_smooth(
    values: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
    mode: str = "interp",
) -> np.ndarray:
    """
    Savitzky-Golay 滤波平滑 NDVI 时序。

    原理: 用滑动窗口内的多项式最小二乘拟合替代原值，
          去除传感器噪声和云污染残留，保留季节趋势。

    参数:
        values: 一维时序 (T,), 允许 NaN (mode="interp" 时插值填充)
        window: 窗口大小 (奇数, 默认 7, 对应半月合成)
        polyorder: 多项式阶数 (默认 2, 不宜超过 window-1)
        mode: scipy.signal.savgol_filter 边界模式 ("interp" 推荐)

    返回:
        smoothed: (T,) 平滑后时序 (与输入同长)

    说明:
        窗口需为奇数且 ≤ 时序长度; polyorder < window。
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"savgol_smooth 仅支持一维时序, 实际: {values.ndim}D")

    n = len(values)
    if n < 5:
        return values.copy()

    if window % 2 == 0:
        window += 1  # 强制奇数
    window = min(window, n if n % 2 == 1 else n - 1)
    polyorder = min(polyorder, window - 1)
    if polyorder < 1:
        polyorder = 1

    # 缺失值填充: 线性插值, 两端回退到邻近有效值
    work = values.copy()
    nan_mask = ~np.isfinite(work)
    if nan_mask.all():
        return values.copy()
    if nan_mask.any():
        idx = np.arange(n)
        valid_idx = idx[~nan_mask]
        work = np.interp(idx, valid_idx, work[valid_idx])

    try:
        from scipy.signal import savgol_filter
        return savgol_filter(work, window_length=window, polyorder=polyorder, mode=mode)
    except ImportError:
        # scipy 不可用时回退到滑动平均
        kernel = np.ones(window) / window
        return np.convolve(work, kernel, mode="same")


def smooth_ndvi_stack(
    ndvi_stack: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """
    对 NDVI 时序立方体 (H, W, T) 逐像元 S-G 平滑。

    参数:
        ndvi_stack: (H, W, T) NDVI 时序
        window/polyorder: S-G 滤波参数

    返回:
        smoothed: (H, W, T)
    """
    ndvi_stack = np.asarray(ndvi_stack, dtype=np.float64)
    if ndvi_stack.ndim != 3:
        raise ValueError(f"需要 3D 数组 (H,W,T), 实际: {ndvi_stack.ndim}D")

    H, W, T = ndvi_stack.shape
    if T < 5:
        return ndvi_stack.copy()

    flat = ndvi_stack.reshape(-1, T)
    smoothed = np.vstack([
        savgol_smooth(row, window=window, polyorder=polyorder) for row in flat
    ])
    return smoothed.reshape(H, W, T)


# ============================================================
# STL 时序分解 (趋势/季节/残差)
# ============================================================

def stl_decompose(
    values: np.ndarray,
    seasonal_period: int = 12,
    robust: bool = True,
) -> Dict[str, np.ndarray]:
    """
    STL (Seasonal-Trend decomposition using Loess) 时序分解。

    将 NDVI 时序分解为: 趋势 + 季节 + 残差 三部分。
    这是时间序列分析的经典方法 (Cleveland et al. 1990),
    可分离长期趋势与季节性波动。

    参数:
        values: 一维时序 (T,), 允许 NaN
        seasonal_period: 季节周期 (月度数据=12, 半月=24, 周=52)
        robust: 是否用稳健迭代 (抗离群)

    返回:
        dict: {
            "trend": (T,) 长期趋势,
            "seasonal": (T,) 季节分量,
            "resid": (T,) 残差,
            "seasonal_strength": float 季节强度 (0-1),
            "trend_strength": float 趋势强度 (0-1)
        }
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < seasonal_period * 2 + 1:
        # 数据太短无法分解: 返回全趋势
        return {
            "trend": values.copy(),
            "seasonal": np.zeros_like(values),
            "resid": np.zeros_like(values),
            "seasonal_strength": 0.0,
            "trend_strength": 0.0,
        }

    # 缺失值插值
    work = values.copy()
    nan_mask = ~np.isfinite(work)
    if nan_mask.any():
        idx = np.arange(n)
        valid_idx = idx[~nan_mask]
        if len(valid_idx) >= 2:
            work = np.interp(idx, valid_idx, work[valid_idx])
        else:
            work = np.nan_to_num(work, nan=np.nanmean(work))

    try:
        from statsmodels.tsa.seasonal import STL
        result = STL(work, period=seasonal_period, robust=robust).fit()
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
    except Exception:
        # statsmodels 不可用/失败 → 移动平均分解
        trend = pd_rolling_mean(work, window=seasonal_period)
        seasonal = work - trend
        resid = np.zeros_like(work)

    # 分量强度 (Wang et al. 2006)
    var_resid = np.nanvar(resid) if np.isfinite(resid).any() else 0.0
    var_seasonal = np.nanvar(seasonal) if np.isfinite(seasonal).any() else 0.0
    var_trend = np.nanvar(trend) if np.isfinite(trend).any() else 0.0
    total = var_trend + var_seasonal + var_resid
    seasonal_strength = max(0.0, min(1.0, var_seasonal / max(total, 1e-12)))
    trend_strength = max(0.0, min(1.0, var_trend / max(total, 1e-12)))

    return {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
        "seasonal_strength": round(float(seasonal_strength), 4),
        "trend_strength": round(float(trend_strength), 4),
    }


def pd_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """移动平均 (无 pandas 依赖的轻量实现)。"""
    values = np.asarray(values, dtype=np.float64)
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")
