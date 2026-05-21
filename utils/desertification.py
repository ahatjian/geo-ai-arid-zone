"""
沙漠化监测模块
================
基于 Sentinel-2/Landsat 多光谱数据的沙漠化评估

核心方法: Albedo-NDVI 特征空间法 (中国沙漠化研究主流方法)

指标:
  - Albedo (地表反照率)     — Liang 2001 宽带反照率公式
  - TGSI  (表土粒度指数)    — Xiao et al. 2006, 土壤粗化程度
  - NDMI  (归一化水分指数)   — 土壤湿度
  - DDI   (沙漠化差异指数)   — Albedo-NDVI 特征空间综合
  - SDI   (沙化程度指数)     — 5级分类 (非/轻度/中度/重度/极重度)

参考:
  - Verstraete & Pinty (1996) Albedo-NDVI 特征空间
  - 曾永年等 (2006) 基于 Albedo-NDVI 的沙漠化遥感监测
  - 国家冰川冻土沙漠科学数据中心 沙漠化分级标准
  - 西北干旱监测预测业务服务综合系统
  - 北方干旱综合监测平台

依赖: numpy, scipy
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
# 沙漠化等级定义
# ============================================================

DESERTIFICATION_LEVELS = [
    {
        "code": 0, "name": "非沙漠化",
        "ddi_range": (-np.inf, 3.0),
        "ndvi_range": (0.4, np.inf),
        "albedo_range": (-np.inf, 0.25),
        "description": "植被茂密，地表稳定",
        "color": "#2ecc71",   # 绿色
        "risk": "安全",
    },
    {
        "code": 1, "name": "轻度沙漠化",
        "ddi_range": (3.0, 4.0),
        "ndvi_range": (0.25, 0.4),
        "albedo_range": (0.25, 0.30),
        "description": "植被开始退化，土壤轻微沙化",
        "color": "#f1c40f",   # 黄色
        "risk": "关注",
    },
    {
        "code": 2, "name": "中度沙漠化",
        "ddi_range": (4.0, 5.0),
        "ndvi_range": (0.15, 0.25),
        "albedo_range": (0.30, 0.35),
        "description": "斑块状流沙出现，植被盖度下降",
        "color": "#e67e22",   # 橙色
        "risk": "预警",
    },
    {
        "code": 3, "name": "重度沙漠化",
        "ddi_range": (5.0, 6.0),
        "ndvi_range": (0.05, 0.15),
        "albedo_range": (0.35, 0.40),
        "description": "半流动沙丘为主，植被稀疏",
        "color": "#e74c3c",   # 红色
        "risk": "紧急",
    },
    {
        "code": 4, "name": "极重度沙漠化",
        "ddi_range": (6.0, np.inf),
        "ndvi_range": (-np.inf, 0.05),
        "albedo_range": (0.40, np.inf),
        "description": "流动沙丘，几乎无植被",
        "color": "#8e44ad",   # 紫色
        "risk": "危急",
    },
]


@dataclass
class DesertificationResult:
    """沙漠化评估结果"""
    # 指数图
    albedo: Optional[np.ndarray] = None        # (H, W)
    tgsi: Optional[np.ndarray] = None          # (H, W)
    ndmi: Optional[np.ndarray] = None          # (H, W)
    ddi: Optional[np.ndarray] = None           # (H, W) 沙漠化差异指数
    ndvi: Optional[np.ndarray] = None          # (H, W)
    # 分类
    category: Optional[np.ndarray] = None      # (H, W) 0-4
    category_colors: Optional[np.ndarray] = None  # (H, W, 3) RGB
    # 统计
    stats: Optional[List[Dict]] = None         # 分级统计
    summary: Dict = field(default_factory=dict)


# ============================================================
# 指数计算
# ============================================================

def calc_albedo_s2(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    method: str = "liang",
) -> np.ndarray:
    """
    计算 Sentinel-2 宽带地表反照率

    公式 (Liang 2001, 针对 Sentinel-2 调整):
      Albedo = 0.356 * B2 + 0.130 * B4 + 0.373 * B8 + 0.085 * B11 + 0.072 * B12 - 0.0018

    参数:
        blue, red, nir, swir1, swir2: (H, W) 反射率 (0-1 或 0-10000)
        method: "liang" (默认) 或 "simple" (简化为 NIR + SWIR1 加权)

    返回:
        albedo: (H, W) 地表反照率
    """
    blue = np.asarray(blue, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    swir2 = np.asarray(swir2, dtype=np.float64)

    # 自动缩放: 如果是 DN 值 (>10) 则缩放到 0-1
    if np.nanmedian(blue) > 10:
        blue /= 10000.0
        red /= 10000.0
        nir /= 10000.0
        swir1 /= 10000.0
        swir2 /= 10000.0

    if method == "simple":
        # 简化版: NIR 和 SWIR1 权重
        albedo = 0.5 * nir + 0.5 * swir1
    else:
        albedo = (
            0.356 * blue + 0.130 * red + 0.373 * nir
            + 0.085 * swir1 + 0.072 * swir2 - 0.0018
        )

    # 约束范围
    albedo = np.clip(albedo, 0.0, 1.0)
    return albedo.astype(np.float32)


def calc_albedo_landsat(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    """
    计算 Landsat 宽带反照率

    公式 (Liang 2001, Landsat OLI):
      Albedo = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018

    参数同 calc_albedo_s2
    """
    return calc_albedo_s2(blue, red, nir, swir1, swir2, method="liang")


def calc_tgsi(
    red: np.ndarray,
    swir1: np.ndarray,
    blue: Optional[np.ndarray] = None,
    version: int = 2,
) -> np.ndarray:
    """
    计算表土粒度指数 (Topsoil Grain Size Index)

    原理: 土壤越粗化 (沙化), 短波红外反射越强, TGSI 值越高

    Version 1 (Xiao et al. 2006 原始):
      TGSI = (SWIR1 - Red) / (SWIR1 + Red)

    Version 2 (改进, 加入 Blue 波段):
      TGSI = (SWIR1 - Red) / (SWIR1 + Red + Blue)

    参数:
        red: 红波段 (H, W)
        swir1: 短波红外1 (H, W)
        blue: 蓝波段 (H, W), 仅 version 2 需要
        version: 1 或 2

    返回:
        tgsi: (H, W), 值越大土壤越粗
    """
    red = np.asarray(red, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)

    if version == 2 and blue is not None:
        blue = np.asarray(blue, dtype=np.float64)
        denom = swir1 + red + blue
    else:
        denom = swir1 + red

    tgsi = np.where(
        denom > 1e-6,
        (swir1 - red) / denom,
        0.0,
    )
    return np.clip(tgsi, -1.0, 1.0).astype(np.float32)


def calc_ndmi(
    nir: np.ndarray,
    swir1: np.ndarray,
) -> np.ndarray:
    """
    计算归一化水分指数 (Normalized Difference Moisture Index)

    公式: NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    沙漠区水分低 → NDMI 低值(甚至负值)

    参数:
        nir: 近红外 (H, W)
        swir1: 短波红外1 (H, W)

    返回:
        ndmi: (H, W), 范围 -1~1
    """
    nir = np.asarray(nir, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    denom = nir + swir1
    ndmi = np.where(denom > 1e-6, (nir - swir1) / denom, 0.0)
    return np.clip(ndmi, -1.0, 1.0).astype(np.float32)


# ============================================================
# Albedo-NDVI 特征空间法 (DDI)
# ============================================================

def calc_ddi(
    ndvi: np.ndarray,
    albedo: np.ndarray,
    method: str = "auto",
) -> np.ndarray:
    """
    计算沙漠化差异指数 (Desertification Difference Index)

    基于 Albedo-NDVI 特征空间:
      - 在 NDVI-Albedo 散点图中，上边界代表"干边"(dry edge)
      - DDI = k * NDVI - Albedo  (或 Albedo - k * NDVI)
      - DDI 值越高表示越接近干边 (沙漠化越严重)

    方法:
      "auto"  — 自动线性拟合干边
      "fixed" — 使用固定参数 k=1.0 (适用于干旱区)
      "regression" — 对 NDVI 分 bin 取上包络线

    参数:
        ndvi: (H, W) NDVI, 范围 -1~1
        albedo: (H, W) 反照率, 范围 0~1
        method: 特征空间方法

    返回:
        ddi: (H, W) DDI 值, 越大沙漠化越严重
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    albedo = np.asarray(albedo, dtype=np.float64)

    # 获取 k 值 (干边斜率)
    if method == "fixed":
        k = 1.0
    elif method == "regression":
        k = _fit_dry_edge_regression(ndvi, albedo)
    else:  # auto
        k = _fit_dry_edge_auto(ndvi, albedo)

    # DDI = a * Albedo - k * NDVI
    # 参考曾永年等 (2006): DDI = (1/√(1+k²)) * (k * NDVI - Albedo)
    # 简化为 a * Albedo - NDVI 便于解释

    # 使用归一化版本
    ddi = k * (1.0 - ndvi) + albedo

    # 缩放到合理范围
    ddi = np.clip(ddi, 0.0, 10.0)

    return ddi.astype(np.float32)


def _fit_dry_edge_auto(ndvi: np.ndarray, albedo: np.ndarray) -> float:
    """自动估计干边斜率 k"""
    # 取 NDVI > 0.05 的有效像元
    mask = (np.isfinite(ndvi) & np.isfinite(albedo)
            & (ndvi > -1) & (ndvi < 1)
            & (albedo > 0) & (albedo < 1))

    ndvi_valid = ndvi[mask]
    albedo_valid = albedo[mask]

    if len(ndvi_valid) < 100:
        return 1.0

    # 分 bin 取上包络线 (95 百分位)
    n_bins = min(20, max(5, len(ndvi_valid) // 500))
    ndvi_bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
    upper_ndvi, upper_albedo = [], []

    for i in range(n_bins):
        bin_mask = (ndvi_valid >= ndvi_bins[i]) & (ndvi_valid < ndvi_bins[i + 1])
        if bin_mask.sum() > 10:
            upper_ndvi.append(np.median(ndvi_valid[bin_mask]))
            upper_albedo.append(np.percentile(albedo_valid[bin_mask], 95))

    if len(upper_ndvi) < 3:
        return 1.0

    upper_ndvi = np.array(upper_ndvi)
    upper_albedo = np.array(upper_albedo)

    # 线性回归: Albedo = a - b * NDVI → k = b
    try:
        coeffs = np.polyfit(upper_ndvi, upper_albedo, 1)
        slope = -coeffs[0]  # b = -slope of Albedo vs NDVI
        k = max(0.3, min(slope, 2.5))  # 约束
        return k
    except Exception:
        return 1.0


def _fit_dry_edge_regression(ndvi: np.ndarray, albedo: np.ndarray) -> float:
    """回归法估计干边斜率"""
    return _fit_dry_edge_auto(ndvi, albedo)


# ============================================================
# 沙漠化分类
# ============================================================

def classify_desertification(
    ddi: np.ndarray,
    ndvi: Optional[np.ndarray] = None,
    albedo: Optional[np.ndarray] = None,
    tgsi: Optional[np.ndarray] = None,
    method: str = "ddi",
) -> np.ndarray:
    """
    沙漠化等级分类 (5级)

    参数:
        ddi: 沙漠化差异指数 (H, W)
        ndvi: NDVI (可选, 用于双重验证)
        albedo: 反照率 (可选)
        tgsi: TGSI (可选, 精度增强)
        method: "ddi" | "composite"

    返回:
        category: (H, W) int8, 0-4
    """
    ddi = np.asarray(ddi, dtype=np.float64)

    if method == "composite" and ndvi is not None and albedo is not None:
        return _classify_composite(ddi, ndvi, albedo, tgsi)
    else:
        return _classify_by_ddi(ddi)


def _classify_by_ddi(ddi: np.ndarray) -> np.ndarray:
    """基于 DDI 阈值分类"""
    ddi = np.asarray(ddi, dtype=np.float64)
    category = np.full(ddi.shape, 0, dtype=np.int8)

    for level in DESERTIFICATION_LEVELS:
        lo, hi = level["ddi_range"]
        mask = (ddi >= lo) & (ddi < hi)
        # 跳过第一个 (非沙漠化), 从高到低覆盖
        if level["code"] > 0:
            category[mask] = level["code"]

    return category


def _classify_composite(
    ddi: np.ndarray,
    ndvi: np.ndarray,
    albedo: np.ndarray,
    tgsi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """综合多指标分类 (更精确)"""
    ndvi = np.asarray(ndvi, dtype=np.float64)
    albedo = np.asarray(albedo, dtype=np.float64)
    category = np.full(ddi.shape, 0, dtype=np.int8)

    for level in DESERTIFICATION_LEVELS:
        code = level["code"]
        if code == 0:
            continue

        ndvi_lo, ndvi_hi = level["ndvi_range"]
        alb_lo, alb_hi = level["albedo_range"]

        mask = ((ndvi >= ndvi_lo) & (ndvi < ndvi_hi)
                | (albedo >= alb_lo) & (albedo < alb_hi))

        # TGSI 增强 (可选)
        if tgsi is not None:
            tgsi_arr = np.asarray(tgsi, dtype=np.float64)
            if code >= 3:
                mask = mask & (tgsi_arr > 0.1)

        category[mask] = code

    return category


def get_desertification_colormap(category: np.ndarray) -> np.ndarray:
    """
    将分类结果转为 RGB 色彩图

    返回:
        colormap: (H, W, 3) uint8 RGB 数组
    """
    H, W = category.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    color_map = {
        0: [46, 204, 113],   # 绿: 非沙漠化
        1: [241, 196, 15],   # 黄: 轻度
        2: [230, 126, 34],   # 橙: 中度
        3: [231, 76, 60],    # 红: 重度
        4: [142, 68, 173],   # 紫: 极重度
    }

    for code, color in color_map.items():
        rgb[category == code] = color

    return rgb


# ============================================================
# 统计计算
# ============================================================

def compute_desertification_stats(
    category: np.ndarray,
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """
    计算沙漠化分级统计

    参数:
        category: (H, W) 分类图 (0-4)
        pixel_size_m: 像元大小 (m)

    返回:
        stats: 每级统计列表
    """
    category = np.asarray(category, dtype=np.int8)
    total_valid = np.sum(np.isfinite(category))

    stats = []
    for level in DESERTIFICATION_LEVELS:
        code = level["code"]
        count = np.sum(category == code)
        ratio = count / max(total_valid, 1)
        area = count * (pixel_size_m ** 2) / 1e6  # km²

        stats.append({
            "code": code,
            "name": level["name"],
            "pixel_count": int(count),
            "ratio": round(ratio, 4),
            "area_km2": round(area, 2),
            "color": level["color"],
            "risk": level["risk"],
            "description": level["description"],
        })

    return stats


def compute_index_stats(
    index_data: np.ndarray,
    index_name: str = "",
) -> Dict:
    """
    计算指数统计

    返回:
        dict: {mean, std, min, max, p5, p95, valid_pixels}
    """
    data = np.asarray(index_data, dtype=np.float64)
    valid = data[np.isfinite(data)]

    if len(valid) == 0:
        return {"index": index_name, "valid_pixels": 0}

    return {
        "index": index_name,
        "mean": round(float(np.mean(valid)), 4),
        "std": round(float(np.std(valid)), 4),
        "min": round(float(np.min(valid)), 4),
        "max": round(float(np.max(valid)), 4),
        "p5": round(float(np.percentile(valid, 5)), 4),
        "p95": round(float(np.percentile(valid, 95)), 4),
        "valid_pixels": int(len(valid)),
    }


# ============================================================
# 一站式评估
# ============================================================

def assess_desertification(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    pixel_size_m: float = 10.0,
    ndvi: Optional[np.ndarray] = None,
    tgsi_version: int = 2,
) -> DesertificationResult:
    """
    一站式沙漠化评估

    参数:
        bands_data: (B, H, W) 多波段数据
          Sentinel-2: [B02, B03, B04, B08, B11, B12]
          Landsat:    [B2, B3, B4, B5, B6, B7]
        satellite: 卫星名称
        pixel_size_m: 像元大小
        ndvi: 预计算的 NDVI (可选, 节省计算)
        tgsi_version: TGSI 版本 (1 或 2)

    返回:
        DesertificationResult
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)

    if bands_data.shape[0] < 5:
        raise ValueError(f"至少需要 5 个波段, 实际: {bands_data.shape[0]}")

    # 提取波段 (统一 Sentinel/Landsat 索引)
    blue = bands_data[0]
    green = bands_data[1]
    red = bands_data[2]
    nir = bands_data[3]
    swir1 = bands_data[4]
    swir2 = bands_data[5] if bands_data.shape[0] > 5 else swir1

    # 1. 计算 NDVI (如未提供)
    if ndvi is None:
        ndvi_denom = nir + red
        ndvi = np.where(ndvi_denom > 1e-6, (nir - red) / ndvi_denom, 0.0)
        ndvi = np.clip(ndvi, -1.0, 1.0)

    # 2. Albedo
    if "Landsat" in satellite:
        albedo = calc_albedo_landsat(blue, red, nir, swir1, swir2)
    else:
        albedo = calc_albedo_s2(blue, red, nir, swir1, swir2)

    # 3. TGSI
    tgsi = calc_tgsi(red, swir1, blue=blue, version=tgsi_version)

    # 4. NDMI
    ndmi = calc_ndmi(nir, swir1)

    # 5. DDI
    ddi = calc_ddi(ndvi, albedo, method="auto")

    # 6. 分类
    category = classify_desertification(
        ddi, ndvi=ndvi, albedo=albedo,
        tgsi=tgsi, method="composite",
    )

    # 7. 统计
    stats = compute_desertification_stats(category, pixel_size_m=pixel_size_m)
    colors = get_desertification_colormap(category)

    # 8. 汇总
    total_desert = sum(s["ratio"] for s in stats if s["code"] >= 1)
    severe_desert = sum(s["ratio"] for s in stats if s["code"] >= 3)

    return DesertificationResult(
        albedo=albedo.astype(np.float32),
        tgsi=tgsi,
        ndmi=ndmi,
        ddi=ddi,
        ndvi=ndvi.astype(np.float32),
        category=category,
        category_colors=colors,
        stats=stats,
        summary={
            "total_desertification_ratio": round(total_desert, 4),
            "severe_desertification_ratio": round(severe_desert, 4),
            "dominant_level": max(stats, key=lambda s: s["ratio"])["name"] if stats else "未知",
            "albedo_mean": round(float(np.nanmean(albedo)), 4),
            "tgsi_mean": round(float(np.nanmean(tgsi)), 4),
            "ndmi_mean": round(float(np.nanmean(ndmi)), 4),
            "pixel_size_m": pixel_size_m,
        },
    )


# ============================================================
# 缓存版 (用于 Streamlit)
# ============================================================

@_cache(ttl=600)
def assess_desertification_cached(
    bands_tuple: Tuple,
    satellite: str,
    pixel_size_m: float = 10.0,
    ndvi_tuple: Optional[Tuple] = None,
) -> Dict:
    """
    缓存版沙漠化评估 (绕过 numpy 数组不可哈希问题)

    bands_tuple: tuple of tuples — np.array → tuple of tuples
    ndvi_tuple: 预计算 NDVI 的 tuple 形式

    返回:
        dict (DesertificationResult 可序列化版本)
    """
    bands_data = np.array(bands_tuple, dtype=np.float64)
    ndvi = np.array(ndvi_tuple, dtype=np.float64) if ndvi_tuple else None

    result = assess_desertification(
        bands_data, satellite=satellite,
        pixel_size_m=pixel_size_m, ndvi=ndvi,
    )

    return {
        "albedo": result.albedo,
        "tgsi": result.tgsi,
        "ndmi": result.ndmi,
        "ddi": result.ddi,
        "category": result.category,
        "stats": result.stats,
        "summary": result.summary,
    }


# ============================================================
# 沙化趋势分析 (时序)
# ============================================================

def analyze_desertification_trend(
    ndvi_stack: np.ndarray,
    albedo_stack: np.ndarray,
) -> Dict:
    """
    沙漠化时序趋势分析

    用 NDVI 和 Albedo 的时序判断沙漠化演变方向

    参数:
        ndvi_stack: (H, W, T) NDVI 时序
        albedo_stack: (H, W, T) 反照率时序

    返回:
        dict: {
            "ndvi_trend": "改善/退化/稳定",
            "albedo_trend": "降低/升高/稳定",
            "desertification_trend": "逆转/加剧/稳定",
            "slope_ndvi": float,
            "slope_albedo": float,
        }
    """
    from utils.trend import theil_sen_slope

    ndvi_stack = np.asarray(ndvi_stack, dtype=np.float64)
    albedo_stack = np.asarray(albedo_stack, dtype=np.float64)

    T = ndvi_stack.shape[-1]
    # 逐时相取均值
    ndvi_ts = np.array([np.nanmean(ndvi_stack[:, :, t]) for t in range(T)])
    albedo_ts = np.array([np.nanmean(albedo_stack[:, :, t]) for t in range(T)])

    # Sen 斜率
    slope_ndvi = theil_sen_slope(ndvi_ts)
    slope_albedo = theil_sen_slope(albedo_ts)

    # 趋势判断
    if slope_ndvi > 0.002:
        ndvi_trend = "🌿 植被改善"
    elif slope_ndvi < -0.002:
        ndvi_trend = "🏜️ 植被退化"
    else:
        ndvi_trend = "➡️ 稳定"

    if slope_albedo > 0.002:
        albedo_trend = "⬆️ 反照率升高 (沙化加剧)"
    elif slope_albedo < -0.002:
        albedo_trend = "⬇️ 反照率降低 (植被恢复)"
    else:
        albedo_trend = "➡️ 稳定"

    # 综合沙漠化趋势
    if slope_ndvi < -0.002 and slope_albedo > 0.002:
        desert_trend = "🔴 沙漠化加剧"
    elif slope_ndvi > 0.002 and slope_albedo < -0.002:
        desert_trend = "🟢 沙漠化逆转"
    else:
        desert_trend = "🟡 稳定/波动"

    return {
        "ndvi_trend": ndvi_trend,
        "albedo_trend": albedo_trend,
        "desertification_trend": desert_trend,
        "slope_ndvi": round(slope_ndvi, 6),
        "slope_albedo": round(slope_albedo, 6),
        "ndvi_timeseries": ndvi_ts.tolist(),
        "albedo_timeseries": albedo_ts.tolist(),
    }
