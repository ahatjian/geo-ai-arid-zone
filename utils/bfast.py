"""
BFAST 风格时序断点检测模块 — 植被突变检测
============================================
BFAST (Breaks For Additive Seasonal and Trend, Verbesselt et al. 2010)
是遥感时序突变检测的经典方法, 用于识别植被覆盖的突变事件
(森林砍伐、火灾、干旱灾害、生态恢复等)。

实现流程:
  1. STL 分解: 分离 趋势/季节/残差 (复用 trend.stl_decompose)
  2. 断点检测: 对去季节序列用 PELT 算法 (ruptures) 找突变点
  3. 显著性检验: Chow 检验确认断点统计显著性 (p < 0.05)
  4. 突变评估: 幅度/方向/前后均值差 (负向突变 = 植被退化)

干旱区应用:
  - 绿洲退化/扩张的起始时间识别
  - 荒漠化事件的突变检测
  - 生态恢复工程的成效评估 (正向突变)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def detect_breaks(
    values: np.ndarray,
    seasonal_period: int = 12,
    min_segment: int = 6,
    penalty: Optional[float] = None,
    max_breaks: int = 3,
    alpha: float = 0.05,
    robust: bool = True,
) -> Dict:
    """
    BFAST 风格断点检测主函数。

    流程: STL 去季节 → PELT 断点检测 → Chow 显著性检验。

    参数:
        values: 一维时序 (T,), 允许 NaN
        seasonal_period: 季节周期 (月度=12, 半月=24)
        min_segment: 最小分段长度 (断点间最小间隔)
        penalty: PELT 惩罚项 (None=自动, 越大断点越少)
        max_breaks: 最多检测断点数
        alpha: 显著性水平 (Chow 检验)

    返回:
        dict: {
            "break_indices": [int, ...] 断点索引,
            "break_dates": [str, ...] 断点日期标签 (若提供 dates),
            "significance": [float, ...] 各断点 Chow 检验 p 值,
            "magnitudes": [float, ...] 各断点突变幅度,
            "directions": ["负向突变"/"正向突变", ...],
            "trend": (T,) 趋势分量,
            "seasonal": (T,) 季节分量,
            "deseasonalized": (T,) 去季节序列,
            "method": str
        }
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < min_segment * 2 + seasonal_period:
        return _empty_result(n, "时序过短无法检测")

    # 1. STL 分解去季节
    from utils.trend import stl_decompose
    stl = stl_decompose(values, seasonal_period=seasonal_period, robust=robust)
    trend = stl["trend"]
    seasonal = stl["seasonal"]
    # 去季节序列 = 趋势 + 残差 (去除季节性波动)
    deseasonalized = trend + stl["resid"]

    # 2. 缺失值处理
    work = deseasonalized.copy()
    nan_mask = ~np.isfinite(work)
    if nan_mask.any():
        idx = np.arange(n)
        valid_idx = idx[~nan_mask]
        if len(valid_idx) >= 2:
            work = np.interp(idx, valid_idx, work[valid_idx])
        else:
            work = np.nan_to_num(work, nan=0.0)

    # 3. PELT 断点检测
    try:
        import ruptures as rpt

        signal = work.reshape(-1, 1)
        if penalty is None:
            # 自动惩罚: 基于信号方差 (BIC 风格)
            sigma2 = np.var(signal)
            penalty = 2 * sigma2 * np.log(n) / max(n, 1)

        algo = rpt.Pelt(model="rbf", min_size=min_segment, jump=1).fit(signal)
        bkps = algo.predict(pen=penalty)

        # 转换断点为索引 (去掉序列末尾)
        raw_indices = [b - 1 for b in bkps[:-1]]
        # 过滤: 最多 max_breaks 个
        raw_indices = raw_indices[:max_breaks]
    except ImportError:
        # ruptures 不可用 → 回退到滚动均值差检测
        raw_indices = _rolling_break_detection(work, min_segment, max_breaks)
    except Exception:
        raw_indices = _rolling_break_detection(work, min_segment, max_breaks)

    # 4. Chow 显著性检验 + 突变评估
    break_indices = []
    p_values = []
    magnitudes = []
    directions = []

    for b_idx in raw_indices:
        if b_idx < min_segment or b_idx >= n - min_segment:
            continue
        try:
            p_val = chow_test(work, b_idx, min_segment=min_segment)
        except Exception:
            p_val = 1.0

        if p_val < alpha:
            break_indices.append(int(b_idx))
            p_values.append(round(float(p_val), 6))
            # 突变幅度 = 断点前后线性趋势拟合值之差
            mag, direction = _break_magnitude(work, b_idx, min_segment)
            magnitudes.append(round(float(mag), 4))
            directions.append(direction)

    # 按时间排序 (PELT 返回的已有序, 保险起见)
    order = np.argsort(break_indices) if break_indices else []
    break_indices = [break_indices[i] for i in order]
    p_values = [p_values[i] for i in order]
    magnitudes = [magnitudes[i] for i in order]
    directions = [directions[i] for i in order]

    return {
        "break_indices": break_indices,
        "break_dates": [],
        "significance": p_values,
        "magnitudes": magnitudes,
        "directions": directions,
        "trend": trend,
        "seasonal": seasonal,
        "deseasonalized": deseasonalized,
        "method": "BFAST (STL + PELT + Chow)",
    }


def _rolling_break_detection(
    values: np.ndarray,
    min_segment: int,
    max_breaks: int,
) -> List[int]:
    """回退方案: 滚动窗口均值差检测 (无 ruptures 时)。"""
    n = len(values)
    window = min_segment
    scores = []
    for i in range(window, n - window):
        left_mean = np.mean(values[i - window:i])
        right_mean = np.mean(values[i:i + window])
        scores.append(abs(right_mean - left_mean))
    if not scores:
        return []
    # 找前 max_breaks 个局部峰值
    scores = np.array(scores)
    threshold = np.percentile(scores, 90)
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i] > threshold and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            peaks.append(i + window)
    return peaks[:max_breaks]


def chow_test(
    values: np.ndarray,
    break_idx: int,
    min_segment: int = 6,
) -> float:
    """
    Chow 结构突变检验。

    原假设: 断点前后两个子序列可用同一回归模型描述 (无突变)。
    备择假设: 存在结构突变。

    实现: 用常数/线性模型比较 分段回归 SSE 与整体回归 SSE,
          构造 F 统计量并计算 p 值。

    参数:
        values: 去季节时序
        break_idx: 断点索引
        min_segment: 最小分段长度

    返回:
        float: p 值 (< 0.05 表示存在显著突变)
    """
    from scipy import stats

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if break_idx < min_segment or break_idx >= n - min_segment:
        return 1.0

    x = np.arange(n, dtype=np.float64)
    left = slice(0, break_idx)
    right = slice(break_idx, n)

    # 整体线性回归
    slope_all, intercept_all, _, _, _ = stats.linregress(x, values)
    pred_all = slope_all * x + intercept_all
    sse_all = np.sum((values - pred_all) ** 2)

    # 分段回归 (各自线性拟合)
    def _fit_segment(idx_slice):
        xs = x[idx_slice]
        ys = values[idx_slice]
        if len(xs) < 3 or np.all(xs == xs[0]):
            return np.full(len(xs), np.mean(ys))
        slope, intercept, _, _, _ = stats.linregress(xs, ys)
        return slope * xs + intercept

    pred_left = _fit_segment(left)
    pred_right = _fit_segment(right)
    sse_left = np.sum((values[left] - pred_left) ** 2)
    sse_right = np.sum((values[right] - pred_right) ** 2)
    sse_split = sse_left + sse_right

    # F 统计量 (2 个分段 × 2 参数 vs 整体 2 参数 → 额外 2 个自由度)
    df1 = 2
    df2 = n - 4
    if sse_split <= 1e-12 or df2 <= 0:
        return 0.0  # 完美分段拟合 → 显著突变
    f_stat = ((sse_all - sse_split) / df1) / (sse_split / df2)
    p_value = 1.0 - stats.f.cdf(max(f_stat, 0), df1, df2)
    return float(p_value)


def _break_magnitude(
    values: np.ndarray,
    break_idx: int,
    min_segment: int = 6,
) -> Tuple[float, str]:
    """
    突变幅度与方向评估。

    幅度 = 断点前后趋势延伸值之差 (在断点处评估)。
    负值 = 负向突变 (退化), 正值 = 正向突变 (恢复)。
    """
    n = len(values)
    x = np.arange(n, dtype=np.float64)

    def _fit(xs, ys):
        from scipy import stats
        if len(xs) < 3:
            return np.mean(ys), 0.0
        slope, intercept, _, _, _ = stats.linregress(xs, ys)
        return slope, intercept

    left_x = x[:break_idx]
    right_x = x[break_idx:]
    slope_l, int_l = _fit(left_x, values[:break_idx])
    slope_r, int_r = _fit(right_x, values[break_idx:])

    # 在断点处评估
    pred_l = slope_l * break_idx + int_l
    pred_r = slope_r * break_idx + int_r
    magnitude = pred_r - pred_l

    direction = "正向突变" if magnitude > 0 else "负向突变"
    return magnitude, direction


def detect_vegetation_breaks(
    ndvi_series: np.ndarray,
    dates: Optional[List[str]] = None,
    seasonal_period: int = 12,
    **kwargs,
) -> Dict:
    """
    NDVI 时序植被突变检测 (BFAST 专用封装)。

    参数:
        ndvi_series: NDVI 一维时序 (T,)
        dates: 日期标签 (T,) 可选
        seasonal_period: 季节周期
        **kwargs: 传给 detect_breaks

    返回:
        detect_breaks 结果 + break_dates 填充
    """
    result = detect_breaks(ndvi_series, seasonal_period=seasonal_period, **kwargs)

    if dates is not None and len(dates) == len(ndvi_series):
        result["break_dates"] = [
            str(dates[i]) for i in result["break_indices"]
        ]
    return result


def _empty_result(n: int, reason: str) -> Dict:
    """空结果 (数据不足时)。"""
    return {
        "break_indices": [],
        "break_dates": [],
        "significance": [],
        "magnitudes": [],
        "directions": [],
        "trend": np.full(n, np.nan),
        "seasonal": np.full(n, np.nan),
        "deseasonalized": np.full(n, np.nan),
        "method": f"BFAST (未执行: {reason})",
    }


def summarize_breaks(result: Dict) -> List[Dict]:
    """将断点结果转为可读的摘要列表 (页面展示用)。"""
    rows = []
    for i, idx in enumerate(result["break_indices"]):
        rows.append({
            "断点序号": i + 1,
            "时间索引": idx,
            "日期": result["break_dates"][i] if i < len(result["break_dates"]) else f"t={idx}",
            "突变幅度": result["magnitudes"][i] if i < len(result["magnitudes"]) else None,
            "方向": result["directions"][i] if i < len(result["directions"]) else "",
            "显著性 p": result["significance"][i] if i < len(result["significance"]) else None,
        })
    return rows
