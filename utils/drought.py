"""
干旱指数计算模块
================================
支持多种干旱指数：
  - SPI  (Standardized Precipitation Index) 标准化降水指数
  - SPEI (Standardized Precip. Evapotranspiration Index) 标准化降水蒸散指数
  - VCI  (Vegetation Condition Index) 植被状态指数
  - TCI  (Temperature Condition Index) 温度状态指数
  - VHI  (Vegetation Health Index) 植被健康指数
  - NDDI (Normalized Difference Drought Index) 归一化干旱指数
  - NDVI_Anomaly  NDVI 距平
  - TVDI (Temperature Vegetation Dryness Index) 温度植被干旱指数

参考来源:
  - McKee et al. (1993) SPI
  - Vicente-Serrano et al. (2010) SPEI
  - Kogan (1995) VCI/TCI/VHI
  - 慧天干旱监测与预警平台 FYDI 设计思路
  - 西北干旱监测预测业务服务综合系统

依赖: numpy, scipy (用于分布拟合)
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple
from scipy import stats

warnings.filterwarnings("ignore")

# 从统一配置导入缓存 TTL
from config import CACHE_CONFIG

# ---- 条件缓存装饰器 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    """条件缓存装饰器"""
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=CACHE_CONFIG.get("show_spinner", True))
    else:
        def noop_decorator(func):
            return func
        return noop_decorator


# ============================================
# 干旱等级定义 (参考中国气象局 GB/T 20481-2017)
# ============================================

DROUGHT_CATEGORIES = {
    -3: {"name": "极端湿润", "color": "#00008B", "en": "Extremely Wet"},
    -2: {"name": "严重湿润", "color": "#0000CD", "en": "Severely Wet"},
    -1: {"name": "中等湿润", "color": "#4169E1", "en": "Moderately Wet"},
    0:  {"name": "正常",     "color": "#90EE90", "en": "Near Normal"},
    1:  {"name": "轻度干旱", "color": "#FFFF00", "en": "Mild Drought"},
    2:  {"name": "中等干旱", "color": "#FFA500", "en": "Moderate Drought"},
    3:  {"name": "严重干旱", "color": "#FF4500", "en": "Severe Drought"},
    4:  {"name": "极端干旱", "color": "#8B0000", "en": "Extreme Drought"},
}

# SPI/SPEI 等级阈值
SPI_THRESHOLDS = [
    (2.0,    -3),   # >= 2.0: 极端湿润
    (1.5,    -2),   # >= 1.5: 严重湿润
    (1.0,    -1),   # >= 1.0: 中等湿润
    (-1.0,    0),   # >= -1.0: 正常
    (-1.5,    1),   # >= -1.5: 轻度干旱
    (-2.0,    2),   # >= -2.0: 中等干旱
    (float("-inf"), 3),  # < -2.0: 严重干旱 (SPI通常不到-4)
]

# VCI 等级阈值
VCI_THRESHOLDS = [
    (90,    -3),    # >= 90: 极端湿润
    (70,    -2),    # >= 70: 严重湿润
    (50,    -1),    # >= 50: 中等湿润
    (40,     0),    # >= 40: 正常
    (30,     1),    # >= 30: 轻度干旱
    (20,     2),    # >= 20: 中等干旱
    (10,     3),    # >= 10: 严重干旱
    (float("-inf"), 4),  # < 10: 极端干旱
]


def classify_drought(index_value, thresholds: List[Tuple[float, int]]):
    """
    根据阈值表对干旱指数进行分类 (支持标量和数组)

    参数:
        index_value: 指数值 (float 或 np.ndarray)
        thresholds: [(threshold, category), ...] 从高到低排列

    返回:
        int 或 np.ndarray: 干旱等级 (-3 到 4)
    """
    index_value = np.asarray(index_value)
    if index_value.ndim == 0:
        # 标量
        for threshold, category in thresholds:
            if float(index_value) >= threshold:
                return category
        return thresholds[-1][1]
    else:
        # 数组: 向量化分类 (阈值从高到低，首次匹配优先)
        result = np.full(index_value.shape, thresholds[-1][1], dtype=int)
        unassigned = np.ones(index_value.shape, dtype=bool)
        for threshold, category in thresholds:
            match = unassigned & (index_value >= threshold)
            result[match] = category
            unassigned[match] = False
        return result


def classify_spi(spi):
    """SPI/SPEI 值 → 干旱等级 (支持标量和数组)"""
    return classify_drought(spi, SPI_THRESHOLDS)


def classify_vci(vci):
    """VCI 值 → 干旱等级 (支持标量和数组)"""
    return classify_drought(vci, VCI_THRESHOLDS)


# ============================================
# 1. SPI — 标准化降水指数
# ============================================

def _fit_gamma_params(data: np.ndarray) -> Tuple[float, float, float]:
    """
    对非零降水量拟合 Gamma 分布

    参数:
        data: 一维降水数组

    返回:
        (alpha, beta, prob_zero) — Gamma 形状/尺度参数 + 零值概率
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]

    if len(data) < 3:
        return 1.0, 1.0, 0.0

    # 处理零值 (McKee方法: 混合分布)
    non_zero = data[data > 1e-8]
    prob_zero = 1.0 - len(non_zero) / len(data) if len(data) > 0 else 0.0

    if len(non_zero) < 3:
        return 1.0, 1.0, prob_zero

    try:
        # Gamma 分布 MLE 拟合
        shape, loc, scale = stats.gamma.fit(non_zero, floc=0)
        alpha = shape
        beta = scale
        # 验证参数合理性
        if not (np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0):
            alpha, beta = 1.0, non_zero.mean()
    except Exception:
        # 回退: 矩估计
        mu = non_zero.mean()
        var = non_zero.var()
        if var > 0:
            alpha = (mu ** 2) / var
            beta = var / mu
        else:
            alpha, beta = 1.0, mu

    return max(alpha, 0.01), max(beta, 1e-8), prob_zero


def _gamma_cdf(x: float, alpha: float, beta: float, prob_zero: float) -> float:
    """带零值处理的 Gamma CDF"""
    if x <= 1e-8:
        return prob_zero
    return prob_zero + (1.0 - prob_zero) * stats.gamma.cdf(x, a=alpha, scale=beta)


def _spi_from_cdf(x: float, alpha: float, beta: float, prob_zero: float) -> float:
    """从 Gamma CDF 转换为标准正态分位数 (SPI)"""
    cdf = _gamma_cdf(x, alpha, beta, prob_zero)
    # 限制范围避免极端值
    cdf = np.clip(cdf, 0.0001, 0.9999)
    return float(stats.norm.ppf(cdf))


def calc_spi(precip_ts: np.ndarray,
             scale: int = 3,
             times: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算标准化降水指数 (SPI)

    参数:
        precip_ts: 一维降水时序 (mm), 建议至少 30 个时相
        scale: 时间尺度 (月数), 常用 1/3/6/12
        times: 时间索引, None=等距

    返回:
        np.ndarray: SPI 时间序列, 前 scale-1 个值为 NaN

    示例:
        monthly_precip = np.array([20, 45, 10, ...])  # 逐月降水
        spi3 = calc_spi(monthly_precip, scale=3)     # 3个月 SPI
    """
    precip = np.asarray(precip_ts, dtype=np.float64).flatten()
    n = len(precip)

    if n < scale + 2:
        return np.full(n, np.nan)

    # 计算 scale 月滑动和
    rolling_sum = np.full(n, np.nan)
    for i in range(scale - 1, n):
        rolling_sum[i] = np.sum(precip[i - scale + 1:i + 1])

    # 对每个窗口位置拟合 Gamma 分布
    # 简化: 使用全部非NaN滚动和拟合 (适用于中等长度序列)
    valid_sums = rolling_sum[np.isfinite(rolling_sum)]
    alpha, beta, p0 = _fit_gamma_params(valid_sums)

    spi = np.full(n, np.nan)
    for i in range(scale - 1, n):
        if np.isfinite(rolling_sum[i]):
            spi[i] = _spi_from_cdf(rolling_sum[i], alpha, beta, p0)

    return spi


def calc_spi_pixelwise(ts_stack: np.ndarray,
                       scale: int = 3,
                       times: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    逐像元 SPI 计算 (对立方体时序数据)

    参数:
        ts_stack: (height, width, time) 降水时序立方体
        scale: SPI 时间尺度
        times: 时间索引

    返回:
        dict: {
            "spi": (H, W, T) SPI 时序,
            "spi_latest": (H, W) 最新时相 SPI,
            "drought_category": (H, W) 干旱等级
        }
    """
    H, W, T = ts_stack.shape
    spi_stack = np.full((H, W, T), np.nan, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            ts = ts_stack[i, j, :]
            spi_stack[i, j, :] = calc_spi(ts, scale=scale, times=times)

    # 最新时相
    spi_latest = spi_stack[:, :, -1].copy()
    drought_cat = np.zeros((H, W), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            if np.isfinite(spi_latest[i, j]):
                drought_cat[i, j] = classify_spi(spi_latest[i, j])

    return {
        "spi": spi_stack,
        "spi_latest": spi_latest,
        "drought_category": drought_cat,
    }


# ============================================
# 2. SPEI — 标准化降水蒸散指数
# ============================================

def calc_pet_thornthwaite(temp_c: np.ndarray,
                          lat: float = 40.0,
                          month_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Thornthwaite 方法估算潜在蒸散 (PET)

    参数:
        temp_c: 月均气温 (°C)
        lat: 纬度 (用于日长校正)
        month_indices: 月份索引 (1-12), None=假设从1月开始

    返回:
        np.ndarray: PET (mm/month), 与输入同形

    参考: Thornthwaite (1948)
    """
    temp = np.asarray(temp_c, dtype=np.float64).flatten()
    n = len(temp)
    pet = np.full(n, np.nan)

    if month_indices is None:
        month_indices = np.array([(i % 12) + 1 for i in range(n)])

    # 日长校正因子 (北半球近似, 按纬度)
    # 简化: 使用常数值 (西北干旱区 lat ~35-45)
    day_length_factor = _day_length_correction(lat, month_indices)

    for i in range(n):
        t = temp[i]
        if not np.isfinite(t) or t <= 0:
            pet[i] = 0.0
            continue

        # 热量指数
        heat_index = _calc_heat_index(temp, month_indices)

        if heat_index <= 0:
            pet[i] = 0.0
            continue

        # 未校正 PET
        pet_uncorrected = 16.0 * (10.0 * t / heat_index) ** _thornthwaite_exponent(heat_index)
        # 日长 + 每月天数校正
        days_in_month = _days_in_month(int(month_indices[i]))
        pet[i] = pet_uncorrected * (days_in_month / 30.0) * (day_length_factor[i] / 12.0)

    return pet


def _calc_heat_index(temp_c: np.ndarray, month_indices: np.ndarray) -> float:
    """年热量指数"""
    annual_i = 0.0
    for i in range(1, 13):
        mask = month_indices == i
        if mask.any():
            t_avg = np.nanmean(temp_c[mask])
            if t_avg > 0:
                annual_i += (t_avg / 5.0) ** 1.514
    return annual_i


def _thornthwaite_exponent(heat_index: float) -> float:
    """Thornthwaite 指数 a"""
    hi = heat_index
    return (6.75e-7 * hi ** 3 - 7.71e-5 * hi ** 2 + 1.792e-2 * hi + 0.49239)


def _day_length_correction(lat: float, month_indices: np.ndarray) -> np.ndarray:
    """日长校正因子 (月平均日照时数 / 12)"""
    # 北半球各月近似日长校正 (纬度 40°N 左右)
    # 实际应用中可查表，这里用简化值
    monthly_dl = np.array([0.85, 0.90, 1.03, 1.10, 1.21, 1.23,
                           1.24, 1.16, 1.03, 0.94, 0.83, 0.79])
    # 纬度调整
    lat_factor = 1.0 + 0.01 * (lat - 40.0)  # 简单线性调整
    dl = monthly_dl * lat_factor

    result = np.zeros(len(month_indices))
    for i, m in enumerate(month_indices):
        idx = int(m) - 1
        if 0 <= idx < 12:
            result[i] = dl[idx]
        else:
            result[i] = 1.0
    return result


def _days_in_month(month: int) -> int:
    """每月天数"""
    if month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 28
    else:
        return 31


def calc_spei(precip_ts: np.ndarray,
              temp_ts: np.ndarray,
              scale: int = 3,
              lat: float = 40.0,
              month_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算标准化降水蒸散指数 (SPEI)

    参数:
        precip_ts: 降水时序 (mm)
        temp_ts: 气温时序 (°C)
        scale: 时间尺度 (月)
        lat: 纬度
        month_indices: 月份索引

    返回:
        np.ndarray: SPEI 时序, 前 scale-1 个值为 NaN

    参考: Vicente-Serrano et al. (2010)
    """
    precip = np.asarray(precip_ts, dtype=np.float64).flatten()
    temp = np.asarray(temp_ts, dtype=np.float64).flatten()
    n = min(len(precip), len(temp))

    if n < scale + 5:
        return np.full(n, np.nan)

    # 计算 PET
    pet = calc_pet_thornthwaite(temp[:n], lat=lat, month_indices=month_indices)

    # 水量平衡 D = P - PET
    water_balance = precip[:n] - pet

    # 计算 scale 月累积水量平衡
    rolling_wb = np.full(n, np.nan)
    for i in range(scale - 1, n):
        rolling_wb[i] = np.sum(water_balance[i - scale + 1:i + 1])

    # 拟合 Log-Logistic 分布 (3参数)
    valid_wb = rolling_wb[np.isfinite(rolling_wb)]

    if len(valid_wb) < 5:
        return np.full(n, np.nan)

    try:
        # Log-logistic 分布拟合
        shape, loc, scale_param = stats.fisk.fit(valid_wb, floc=0)
        if not (np.isfinite(shape) and shape > 0 and np.isfinite(scale_param) and scale_param > 0):
            shape, scale_param = 2.0, np.std(valid_wb) if np.std(valid_wb) > 0 else 1.0
    except Exception:
        shape, scale_param = 2.0, max(np.std(valid_wb), 1e-6)

    loc = 0.0  # log-logistic 的 loc

    spei = np.full(n, np.nan)
    for i in range(scale - 1, n):
        if np.isfinite(rolling_wb[i]):
            try:
                cdf = stats.fisk.cdf(rolling_wb[i], c=shape, loc=loc, scale=scale_param)
                cdf = np.clip(cdf, 0.0001, 0.9999)
                spei[i] = float(stats.norm.ppf(cdf))
            except Exception:
                spei[i] = np.nan

    return spei


# ============================================
# 3. VCI — 植被状态指数
# ============================================

def calc_vci(ndvi_ts: np.ndarray,
             ndvi_min: Optional[float] = None,
             ndvi_max: Optional[float] = None) -> np.ndarray:
    """
    计算植被状态指数 (VCI)

    VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) × 100

    参数:
        ndvi_ts: NDVI 时间序列
        ndvi_min: 时序 NDVI 最小值 (None=自动从数据中取)
        ndvi_max: 时序 NDVI 最大值 (None=自动从数据中取)

    返回:
        np.ndarray: VCI 值 (0-100), 越高表示植被状态越好

    参考: Kogan (1995)
    """
    ndvi = np.asarray(ndvi_ts, dtype=np.float64).flatten()

    if ndvi_min is None:
        valid = ndvi[np.isfinite(ndvi)]
        ndvi_min = np.percentile(valid, 5) if len(valid) > 0 else 0.0
    if ndvi_max is None:
        valid = ndvi[np.isfinite(ndvi)]
        ndvi_max = np.percentile(valid, 95) if len(valid) > 0 else 1.0

    denom = ndvi_max - ndvi_min
    if denom < 1e-8:
        return np.full_like(ndvi, 50.0)

    vci = (ndvi - ndvi_min) / denom * 100.0
    return np.clip(vci, 0.0, 100.0)


def calc_vci_pixelwise(ts_stack: np.ndarray,
                       times: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    逐像元 VCI 计算

    对每个像元的 NDVI 时序求 min/max，计算 VCI

    参数:
        ts_stack: (height, width, time) NDVI 时序立方体
        times: 时间索引

    返回:
        dict: {
            "vci": (H, W, T) VCI 时序,
            "vci_latest": (H, W) 最新时相 VCI,
            "ndvi_min": (H, W) 像元历史最小值,
            "ndvi_max": (H, W) 像元历史最大值,
            "drought_category": (H, W) 干旱等级
        }
    """
    H, W, T = ts_stack.shape

    # 逐像元最小/最大值 (排除 NaN)
    ndvi_min = np.nanmin(ts_stack, axis=2)
    ndvi_max = np.nanmax(ts_stack, axis=2)

    # 避免除零
    denom = ndvi_max - ndvi_min
    denom[denom < 1e-8] = 1.0

    vci_stack = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        vci_stack[:, :, t] = np.clip(
            (ts_stack[:, :, t] - ndvi_min) / denom * 100.0, 0.0, 100.0
        )

    vci_latest = vci_stack[:, :, -1].copy()

    # 干旱等级
    drought_cat = np.zeros((H, W), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            if np.isfinite(vci_latest[i, j]):
                drought_cat[i, j] = classify_vci(vci_latest[i, j])

    return {
        "vci": vci_stack,
        "vci_latest": vci_latest,
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "drought_category": drought_cat,
    }


# ============================================
# 4. TCI — 温度状态指数
# ============================================

def calc_tci(lst_ts: np.ndarray,
             lst_min: Optional[float] = None,
             lst_max: Optional[float] = None) -> np.ndarray:
    """
    计算温度状态指数 (TCI)

    TCI = (LST_max - LST) / (LST_max - LST_min) × 100

    参数:
        lst_ts: 地表温度时间序列 (K 或 °C)
        lst_min: 时序 LST 最小值
        lst_max: 时序 LST 最大值

    返回:
        np.ndarray: TCI 值 (0-100), 越高表示温度胁迫越小

    参考: Kogan (1995)
    """
    lst = np.asarray(lst_ts, dtype=np.float64).flatten()

    if lst_min is None:
        valid = lst[np.isfinite(lst)]
        lst_min = np.percentile(valid, 5) if len(valid) > 0 else 0.0
    if lst_max is None:
        valid = lst[np.isfinite(lst)]
        lst_max = np.percentile(valid, 95) if len(valid) > 0 else 100.0

    denom = lst_max - lst_min
    if denom < 1e-8:
        return np.full_like(lst, 50.0)

    tci = (lst_max - lst) / denom * 100.0
    return np.clip(tci, 0.0, 100.0)


def calc_tci_pixelwise(ts_stack: np.ndarray) -> Dict[str, np.ndarray]:
    """
    逐像元 TCI 计算

    参数:
        ts_stack: (H, W, T) LST 时序立方体

    返回:
        dict: {"tci": ..., "tci_latest": ..., "lst_min": ..., "lst_max": ..., "drought_category": ...}
    """
    H, W, T = ts_stack.shape

    lst_min = np.nanmin(ts_stack, axis=2)
    lst_max = np.nanmax(ts_stack, axis=2)

    denom = lst_max - lst_min
    denom[denom < 1e-8] = 1.0

    tci_stack = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        tci_stack[:, :, t] = np.clip(
            (lst_max - ts_stack[:, :, t]) / denom * 100.0, 0.0, 100.0
        )

    tci_latest = tci_stack[:, :, -1].copy()

    drought_cat = np.zeros((H, W), dtype=np.int8)
    # TCI 也用 VCI 的阈值体系
    for i in range(H):
        for j in range(W):
            if np.isfinite(tci_latest[i, j]):
                drought_cat[i, j] = classify_vci(tci_latest[i, j])

    return {
        "tci": tci_stack,
        "tci_latest": tci_latest,
        "lst_min": lst_min,
        "lst_max": lst_max,
        "drought_category": drought_cat,
    }


# ============================================
# 5. VHI — 植被健康指数
# ============================================

def calc_vhi(vci: np.ndarray, tci: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """
    计算植被健康指数 (VHI)

    VHI = a × VCI + (1-a) × TCI

    参数:
        vci: VCI 数组 (0-100)
        tci: TCI 数组 (0-100)
        weight: VCI 权重 a (默认 0.5), 干旱区建议 0.4 (温度更重要)

    返回:
        np.ndarray: VHI 值 (0-100)

    参考: Kogan (2001)
    """
    vci = np.asarray(vci, dtype=np.float32)
    tci = np.asarray(tci, dtype=np.float32)
    return np.clip(weight * vci + (1.0 - weight) * tci, 0.0, 100.0)


def calc_vhi_pixelwise(vci_stack: np.ndarray,
                       tci_stack: np.ndarray,
                       weight: float = 0.5) -> Dict[str, np.ndarray]:
    """
    逐像元 VHI 计算

    参数:
        vci_stack: (H, W, T) VCI 立方体
        tci_stack: (H, W, T) TCI 立方体
        weight: VCI 权重

    返回:
        dict: {"vhi": ..., "vhi_latest": ..., "drought_category": ...}
    """
    H, W, T = vci_stack.shape
    vhi_stack = calc_vhi(vci_stack, tci_stack, weight)

    vhi_latest = vhi_stack[:, :, -1].copy()
    drought_cat = np.zeros((H, W), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            if np.isfinite(vhi_latest[i, j]):
                drought_cat[i, j] = classify_vci(vhi_latest[i, j])

    return {
        "vhi": vhi_stack,
        "vhi_latest": vhi_latest,
        "drought_category": drought_cat,
    }


# ============================================
# 6. NDDI — 归一化干旱指数
# ============================================

def calc_nddi(ndvi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    计算归一化干旱指数 (NDDI)

    NDDI = (NDVI - NDWI) / (NDVI + NDWI)

    参数:
        ndvi: NDVI 数组
        ndwi: NDWI 数组 (使用 Green-NIR 版本: (Green-NIR)/(Green+NIR))

    返回:
        np.ndarray: NDDI 值，值越大表示越干旱
        NDDI > 0.5: 干旱
        NDDI > 0.7: 严重干旱

    参考: Gu et al. (2007), Sentinel-2 适用
    """
    ndvi = np.asarray(ndvi, dtype=np.float32)
    ndwi = np.asarray(ndwi, dtype=np.float32)

    denom = ndvi + ndwi
    nddi = np.where(np.abs(denom) > 1e-8, (ndvi - ndwi) / denom, 0.0)
    return nddi


# Sentinel-2 版本 NDWI (用于 NDDI)
def calc_ndwi_s2(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Sentinel-2 NDWI (Green-NIR 版本)

    NDWI = (Green - NIR) / (Green + NIR)

    用于 NDDI 计算，值越高水体越多，值越低越干旱
    """
    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    denom = green + nir
    ndwi = np.where(np.abs(denom) > 1e-8, (green - nir) / denom, 0.0)
    return np.clip(ndwi, -1.0, 1.0)


# ============================================
# 7. NDVI 距平
# ============================================

def calc_ndvi_anomaly(ndvi_ts: np.ndarray,
                      baseline_mean: Optional[float] = None,
                      baseline_std: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    计算 NDVI 标准化距平

    Anomaly = (NDVI - mean_NDVI) / std_NDVI

    参数:
        ndvi_ts: NDVI 时间序列
        baseline_mean: 基线均值 (None=从数据中计算)
        baseline_std: 基线标准差 (None=从数据中计算)

    返回:
        dict: {
            "anomaly": 标准化距平数组,
            "raw_anomaly": 原始距平 (NDVI - mean),
            "mean": 基线均值,
            "std": 基线标准差
        }
    """
    ndvi = np.asarray(ndvi_ts, dtype=np.float64).flatten()
    valid = ndvi[np.isfinite(ndvi)]

    if baseline_mean is None:
        baseline_mean = np.mean(valid) if len(valid) > 0 else 0.0
    if baseline_std is None:
        baseline_std = np.std(valid) if len(valid) > 0 else 1.0

    if baseline_std < 1e-8:
        baseline_std = 1e-8

    raw_anomaly = ndvi - baseline_mean
    anomaly = raw_anomaly / baseline_std

    return {
        "anomaly": anomaly,
        "raw_anomaly": raw_anomaly,
        "mean": baseline_mean,
        "std": baseline_std,
    }


def calc_ndvi_anomaly_pixelwise(ts_stack: np.ndarray) -> Dict[str, np.ndarray]:
    """
    逐像元 NDVI 距平

    参数:
        ts_stack: (H, W, T) NDVI 时序

    返回:
        dict: {
            "anomaly": (H, W, T) 标准化距平,
            "anomaly_latest": (H, W) 最新时相距平,
            "mean_ndvi": (H, W) 像元均值,
            "std_ndvi": (H, W) 像元标准差
        }
    """
    H, W, T = ts_stack.shape

    mean_ndvi = np.nanmean(ts_stack, axis=2)
    std_ndvi = np.nanstd(ts_stack, axis=2)
    std_ndvi[std_ndvi < 1e-8] = 1e-8

    anomaly_stack = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        anomaly_stack[:, :, t] = (ts_stack[:, :, t] - mean_ndvi) / std_ndvi

    return {
        "anomaly": anomaly_stack,
        "anomaly_latest": anomaly_stack[:, :, -1],
        "mean_ndvi": mean_ndvi,
        "std_ndvi": std_ndvi,
    }


# ============================================
# 8. TVDI — 温度植被干旱指数
# ============================================

def calc_tvdi(ndvi: np.ndarray, lst: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    计算温度植被干旱指数 (TVDI)

    基于 NDVI-LST 特征空间的干湿边拟合

    参数:
        ndvi: NDVI 数组 (2D)
        lst: 地表温度数组 (2D), 同形状

    返回:
        (tvdi, info): TVDI 数组 (0-1, 越大越干) + 拟合信息

    参考: Sandholt et al. (2002)
    """
    ndvi = np.asarray(ndvi, dtype=np.float64).flatten()
    lst = np.asarray(lst, dtype=np.float64).flatten()

    # 过滤无效值
    mask = np.isfinite(ndvi) & np.isfinite(lst) & (ndvi > 0.05) & (ndvi < 0.95)
    ndvi_v = ndvi[mask]
    lst_v = lst[mask]

    if len(ndvi_v) < 30:
        return np.full_like(ndvi.reshape(-1), np.nan), {"error": "有效像元不足"}

    # 按 NDVI 分箱求干湿边
    n_bins = min(20, len(np.unique(np.round(ndvi_v, 2))))
    bins = np.linspace(ndvi_v.min(), ndvi_v.max(), n_bins + 1)

    dry_edge_ndvi = []
    dry_edge_lst = []
    wet_edge_ndvi = []
    wet_edge_lst = []

    for k in range(n_bins):
        bin_mask = (ndvi_v >= bins[k]) & (ndvi_v < bins[k + 1])
        if bin_mask.sum() < 5:
            continue
        bin_ndvi = ndvi_v[bin_mask]
        bin_lst = lst_v[bin_mask]
        # 干边: 99分位数
        dry_edge_ndvi.append(np.mean(bin_ndvi))
        dry_edge_lst.append(np.percentile(bin_lst, 98))
        # 湿边: 5分位数
        wet_edge_ndvi.append(np.mean(bin_ndvi))
        wet_edge_lst.append(np.percentile(bin_lst, 5))

    if len(dry_edge_ndvi) < 3:
        return np.full_like(ndvi.reshape(-1), np.nan), {"error": "干湿边拟合点不足"}

    dry_edge_ndvi = np.array(dry_edge_ndvi)
    dry_edge_lst = np.array(dry_edge_lst)
    wet_edge_ndvi = np.array(wet_edge_ndvi)
    wet_edge_lst = np.array(wet_edge_lst)

    # 线性拟合干湿边
    dry_slope, dry_intercept = np.polyfit(dry_edge_ndvi, dry_edge_lst, 1)
    wet_slope, wet_intercept = np.polyfit(wet_edge_ndvi, wet_edge_lst, 1)

    # TVDI 计算
    def _lst_dry(ndvi_val):
        return dry_slope * ndvi_val + dry_intercept

    def _lst_wet(ndvi_val):
        return wet_slope * ndvi_val + wet_intercept

    ndvi_all = ndvi.reshape(-1)
    lst_all = lst.reshape(-1)

    lst_dry_vals = _lst_dry(ndvi_all)
    lst_wet_vals = _lst_wet(ndvi_all)

    delta = lst_dry_vals - lst_wet_vals
    delta[np.abs(delta) < 1e-6] = 1e-6

    tvdi = np.clip((lst_all - lst_wet_vals) / delta, 0.0, 1.0)
    tvdi[~np.isfinite(ndvi_all) | ~np.isfinite(lst_all)] = np.nan

    info = {
        "dry_slope": dry_slope,
        "dry_intercept": dry_intercept,
        "wet_slope": wet_slope,
        "wet_intercept": wet_intercept,
        "n_bins": n_bins,
        "n_valid_pixels": len(ndvi_v),
    }

    return tvdi, info


# ============================================
# 9. 综合干旱评估
# ============================================

def calc_composite_drought_index(indices: Dict[str, np.ndarray],
                                  weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    多指数综合干旱评估

    将多个干旱指数按权重融合为综合干旱指数 (CDI)

    参数:
        indices: {"vci": array, "tci": array, "spi": array, ...}
        weights: {"vci": 0.3, "tci": 0.2, "spi": 0.3, ...}
                 None=等权重

    返回:
        np.ndarray: 综合干旱指数 (0-100, 越低越干旱)
    """
    if not indices:
        raise ValueError("至少需要输入一个指数")

    if weights is None:
        w = 1.0 / len(indices)
        weights = {k: w for k in indices}

    # 归一化各指数到 0-100
    normalized = {}
    for name, arr in indices.items():
        arr = np.asarray(arr, dtype=np.float64)
        if name in ("vci", "tci", "vhi"):
            # 已经是 0-100
            normalized[name] = arr
        elif name in ("spi", "spei"):
            # SPI [-3, 3] → [0, 100], 映射: SPI=0→50, SPI=-2→10, SPI=2→90
            normalized[name] = 50.0 + arr * 20.0
        elif name == "nddi":
            # NDDI [-1, 1] → [100, 0] (反转, 越高越湿润)
            normalized[name] = (1.0 - arr) * 50.0
        else:
            # 默认 min-max 归一化
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if vmax - vmin > 1e-8:
                    normalized[name] = (arr - vmin) / (vmax - vmin) * 100.0
                else:
                    normalized[name] = np.full_like(arr, 50.0)
            else:
                normalized[name] = arr

    # 加权融合
    first_key = list(normalized.keys())[0]
    shape = normalized[first_key].shape
    cdi = np.zeros(shape, dtype=np.float64)

    for name, arr in normalized.items():
        wgt = weights.get(name, 0.0)
        cdi += wgt * arr

        # 补 NaN
        cdi[~np.isfinite(arr)] = np.nan

    return np.clip(cdi, 0.0, 100.0)


# ============================================
# 10. 统计与分类 (公共)
# ============================================

def compute_drought_stats(drought_array: np.ndarray,
                          categories: Dict[int, Dict] = None,
                          pixel_size_m: float = 10.0) -> List[Dict]:
    """
    计算干旱分类统计

    参数:
        drought_array: 干旱等级数组 (整数 -3 到 4)
        categories: 干旱等级定义字典 (默认 DROUGHT_CATEGORIES)
        pixel_size_m: 像元大小

    返回:
        list[dict]: 统计信息列表
    """
    if categories is None:
        categories = DROUGHT_CATEGORIES

    arr = np.asarray(drought_array).flatten()
    total = len(arr)
    stats_list = []

    for cat, info in sorted(categories.items()):
        count = int(np.sum(arr == cat))
        ratio = count / total if total > 0 else 0
        area_km2 = count * (pixel_size_m ** 2) / 1e6

        stats_list.append({
            "category": cat,
            "name": info["name"],
            "en_name": info["en"],
            "color": info["color"],
            "pixel_count": count,
            "ratio": round(ratio, 4),
            "area_km2": round(area_km2, 4),
        })

    return stats_list


def compute_drought_index_stats(index_array: np.ndarray,
                                name: str = "index") -> Dict:
    """
    计算干旱指数的统计摘要

    参数:
        index_array: 指数值数组
        name: 指数名称

    返回:
        dict: 统计摘要
    """
    arr = np.asarray(index_array, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return {"name": name, "valid_pixels": 0, "error": "无有效数据"}

    return {
        "name": name,
        "valid_pixels": len(valid),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p5": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
    }


# ============================================
# 11. 一站式干旱分析
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_short"])
def analyze_drought_remote(ndvi_ts: np.ndarray,
                           ndwi_ts: Optional[np.ndarray] = None,
                           lst_ts: Optional[np.ndarray] = None,
                           precip_ts: Optional[np.ndarray] = None,
                           temp_ts: Optional[np.ndarray] = None,
                           spi_scale: int = 3,
                           vhi_weight: float = 0.5) -> Dict:
    """
    一站式遥感干旱分析 (单像元时序)

    参数:
        ndvi_ts: NDVI 时间序列
        ndwi_ts: NDWI 时间序列 (用于 NDDI)
        lst_ts: LST 时间序列 (用于 TCI/TVDI)
        precip_ts: 降水时间序列 (用于 SPI)
        temp_ts: 气温时间序列 (用于 SPEI)
        spi_scale: SPI/SPEI 时间尺度
        vhi_weight: VHI 中 VCI 的权重

    返回:
        dict: 所有可计算的干旱指数结果
    """
    result = {"success": True, "indices": {}, "stats": {}, "categories": {}}

    ndvi = np.asarray(ndvi_ts, dtype=np.float64).flatten()

    try:
        # VCI
        vci = calc_vci(ndvi)
        result["indices"]["vci"] = vci
        result["stats"]["vci"] = compute_drought_index_stats(vci, "VCI")
        # 分类 (使用最新值)
        result["categories"]["vci"] = classify_vci(vci[-1]) if len(vci) > 0 else 0

        # NDDI
        if ndwi_ts is not None:
            ndwi = np.asarray(ndwi_ts, dtype=np.float64).flatten()
            # 对齐长度
            min_len = min(len(ndvi), len(ndwi))
            nddi = calc_nddi(ndvi[:min_len], ndwi[:min_len])
            result["indices"]["nddi"] = nddi
            result["stats"]["nddi"] = compute_drought_index_stats(nddi, "NDDI")

        # TCI (如果有 LST)
        if lst_ts is not None:
            lst = np.asarray(lst_ts, dtype=np.float64).flatten()
            tci = calc_tci(lst)
            result["indices"]["tci"] = tci
            result["stats"]["tci"] = compute_drought_index_stats(tci, "TCI")
            result["categories"]["tci"] = classify_vci(tci[-1]) if len(tci) > 0 else 0

            # VHI
            if "tci" in result["indices"]:
                min_len = min(len(vci), len(tci))
                vhi = calc_vhi(vci[:min_len], tci[:min_len], vhi_weight)
                result["indices"]["vhi"] = vhi
                result["stats"]["vhi"] = compute_drought_index_stats(vhi, "VHI")
                result["categories"]["vhi"] = classify_vci(vhi[-1]) if len(vhi) > 0 else 0

        # SPI
        if precip_ts is not None:
            precip = np.asarray(precip_ts, dtype=np.float64).flatten()
            spi = calc_spi(precip, scale=spi_scale)
            result["indices"]["spi"] = spi
            result["stats"]["spi"] = compute_drought_index_stats(
                spi[np.isfinite(spi)], "SPI"
            )
            result["categories"]["spi"] = classify_spi(spi[-1]) if len(spi) > 0 else 0

        # SPEI
        if precip_ts is not None and temp_ts is not None:
            precip = np.asarray(precip_ts, dtype=np.float64).flatten()
            temp = np.asarray(temp_ts, dtype=np.float64).flatten()
            spei = calc_spei(precip, temp, scale=spi_scale)
            result["indices"]["spei"] = spei
            result["stats"]["spei"] = compute_drought_index_stats(
                spei[np.isfinite(spei)], "SPEI"
            )
            result["categories"]["spei"] = classify_spi(spei[-1]) if len(spei) > 0 else 0

        # NDVI 距平
        anomaly_result = calc_ndvi_anomaly(ndvi)
        result["indices"]["ndvi_anomaly"] = anomaly_result["anomaly"]
        result["stats"]["ndvi_anomaly"] = compute_drought_index_stats(
            anomaly_result["anomaly"], "NDVI_Anomaly"
        )

        # 综合干旱评估 (如果至少有两种指数)
        composite_indices = {}
        composite_weights = {}
        if "vci" in result["indices"]:
            composite_indices["vci"] = result["indices"]["vci"]
            composite_weights["vci"] = 0.4
        if "tci" in result["indices"]:
            composite_indices["tci"] = result["indices"]["tci"]
            composite_weights["tci"] = 0.3
        if "nddi" in result["indices"]:
            composite_indices["nddi"] = result["indices"]["nddi"]
            composite_weights["nddi"] = 0.3

        if len(composite_indices) >= 2:
            # 对齐长度
            min_len = min(len(arr) for arr in composite_indices.values())
            aligned = {k: v[:min_len] for k, v in composite_indices.items()}
            cdi = calc_composite_drought_index(aligned, composite_weights)
            result["indices"]["cdi"] = cdi
            result["stats"]["cdi"] = compute_drought_index_stats(cdi, "CDI")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result
