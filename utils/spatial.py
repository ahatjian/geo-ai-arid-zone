"""
空间邻域分析模块 — Geo AI 差异化空间分析
==========================================
与传统 GIS 的空间分析不同, 本模块面向平台已有分析结果的
"智能空间分析", 与 AI 解读深度结合:

  - raster_buffer(): 栅格缓冲区 (基于距离变换)
  - buffer_zone_stats(): 缓冲区内指标统计 (缓冲带生态分析)
  - neighborhood_stats(): 邻域统计 (均值/标准差/变异系数)
  - overlay_crosstab(): 双栅格叠加交叉表 (分类 × 指数)
  - smart_buffer_analysis(): AI 智能缓冲分析 (一站式)

差异化设计:
  1. 输入 = 平台分析结果 (盐渍化/LST/NDVI/分类图), 无需外部 GIS
  2. 输出 = 统计表 + 结构化结果, 供 DeepSeek AI 解读
  3. 智能目标: 自动识别目标地物类 (如"水体"/"绿洲") 并缓冲
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# 栅格缓冲区
# ============================================================

def raster_buffer(
    target_mask: np.ndarray,
    distance_m: float,
    pixel_size_m: float = 10.0,
    include_target: bool = True,
) -> np.ndarray:
    """
    栅格缓冲区: 对目标像元集合生成距离缓冲带。

    原理: 距离变换 (欧氏距离) → 阈值截断。

    参数:
        target_mask: (H, W) bool, True=目标区域
        distance_m: 缓冲距离 (米)
        pixel_size_m: 像元大小 (米)
        include_target: 是否包含目标区域本身 (False=仅缓冲带)

    返回:
        (H, W) uint8: 0=外部, 1=缓冲带, 2=目标区域 (include_target=True 时)
    """
    target_mask = np.asarray(target_mask, dtype=bool)
    from scipy import ndimage

    # 无目标像元 → 直接返回全 0 (距离变换对全背景输入返回全 0, 需提前处理)
    if not target_mask.any():
        return np.zeros(target_mask.shape, dtype=np.uint8)

    # 欧氏距离变换: 每个像元到最近目标像元的距离 (像元单位)
    dist = ndimage.distance_transform_edt(~target_mask)
    buffer_px = max(1, int(round(distance_m / pixel_size_m)))

    buffer_zone = (dist <= buffer_px) & ~target_mask

    result = np.zeros(target_mask.shape, dtype=np.uint8)
    result[buffer_zone] = 1
    if include_target:
        result[target_mask] = 2
    return result


def buffer_zone_stats(
    value_array: np.ndarray,
    buffer_map: np.ndarray,
    target_value: int = 2,
    buffer_value: int = 1,
) -> Dict:
    """
    缓冲带统计: 目标区与缓冲带的指标对比。

    返回:
        dict: {
            "target_mean": 目标区指标均值,
            "buffer_mean": 缓冲带指标均值,
            "outer_mean": 外部区指标均值,
            "target_area_km2": 目标区面积,
            "buffer_area_km2": 缓冲带面积,
            "gradient": 目标→缓冲→外部 梯度 (正=递减),
            "target_pixels": ..., "buffer_pixels": ...
        }
    """
    value_array = np.asarray(value_array, dtype=np.float64)
    buffer_map = np.asarray(buffer_map, dtype=np.uint8)

    def _mean_of(mask: np.ndarray) -> Optional[float]:
        vals = value_array[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size > 0 else None

    target_mask = buffer_map == target_value
    buffer_mask = buffer_map == buffer_value
    outer_mask = ~target_mask & ~buffer_mask

    t_mean = _mean_of(target_mask)
    b_mean = _mean_of(buffer_mask)
    o_mean = _mean_of(outer_mask)

    # 梯度: 正=目标区高于外围 (如绿洲 NDVI 高于荒漠)
    gradient = None
    if t_mean is not None and o_mean is not None:
        gradient = t_mean - o_mean

    return {
        "target_mean": t_mean,
        "buffer_mean": b_mean,
        "outer_mean": o_mean,
        "gradient": gradient,
        "target_pixels": int(target_mask.sum()),
        "buffer_pixels": int(buffer_mask.sum()),
    }


def smart_buffer_analysis(
    classification: np.ndarray,
    value_array: np.ndarray,
    target_class: int,
    distance_m: float = 3000.0,
    pixel_size_m: float = 10.0,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    AI 智能缓冲分析 (一站式): 分类图中选目标地物 → 缓冲 → 指标对比。

    典型应用: 绿洲 (植被类) 周边 3km 的 NDVI 衰减分析;
              水体周边盐渍化风险带分析。

    参数:
        classification: (H, W) 分类图
        value_array: (H, W) 分析指标 (NDVI/LST/盐分指数等)
        target_class: 目标地物类别值
        distance_m: 缓冲距离 (米)
        pixel_size_m: 像元大小
        class_names: 类别名列表 (供结果标注)

    返回:
        dict: {
            "target_name": 目标地物名,
            "buffer_distance_km": float,
            "stats": buffer_zone_stats 结果,
            "analysis_text": 结构化文本 (供 AI 解读)
        }
    """
    classification = np.asarray(classification, dtype=np.int16)
    value_array = np.asarray(value_array, dtype=np.float64)

    target_mask = classification == target_class
    buffer_map = raster_buffer(target_mask, distance_m, pixel_size_m)

    stats = buffer_zone_stats(value_array, buffer_map)

    target_name = "目标地物"
    if class_names and target_class < len(class_names):
        target_name = class_names[target_class]

    # 生成结构化文本供 AI 解读 (None 均值 → "无数据")
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "无数据"

    analysis_text = (
        f"对分类图中的「{target_name}」(类别 {target_class}) 做 {distance_m/1000:.1f} km 缓冲区分析。"
        f"目标区指标均值: {_fmt(stats['target_mean'])}；"
        f"缓冲带均值: {_fmt(stats['buffer_mean'])}；"
        f"外部区均值: {_fmt(stats['outer_mean'])}；"
        f"目标-外部梯度: {_fmt(stats['gradient'])}。"
        f"目标区面积约 {stats['target_pixels']*pixel_size_m**2/1e6:.1f} km², "
        f"缓冲带面积约 {stats['buffer_pixels']*pixel_size_m**2/1e6:.1f} km²。"
    )

    return {
        "target_name": target_name,
        "target_class": int(target_class),
        "buffer_distance_km": round(distance_m / 1000.0, 2),
        "stats": stats,
        "analysis_text": analysis_text,
    }


# ============================================================
# 邻域统计
# ============================================================

def neighborhood_stats(
    band: np.ndarray,
    kernel_size: int = 3,
    stat: str = "mean",
) -> np.ndarray:
    """
    邻域统计: 每个像元取其 kernel×kernel 邻域的统计量。

    参数:
        band: (H, W) 指标数组
        kernel_size: 邻域大小 (奇数)
        stat: "mean" | "std" | "cv" (变异系数) | "median"

    返回:
        (H, W) 邻域统计图 (边界用 reflect 填充)
    """
    from scipy import ndimage

    band = np.asarray(band, dtype=np.float64)
    kernel_size = max(3, int(kernel_size) | 1)

    if stat == "mean":
        return ndimage.uniform_filter(band, size=kernel_size, mode="reflect")
    if stat == "std":
        mean = ndimage.uniform_filter(band, size=kernel_size, mode="reflect")
        sq_mean = ndimage.uniform_filter(band ** 2, size=kernel_size, mode="reflect")
        return np.sqrt(np.clip(sq_mean - mean ** 2, 0, None))
    if stat == "cv":
        mean = ndimage.uniform_filter(band, size=kernel_size, mode="reflect")
        sq_mean = ndimage.uniform_filter(band ** 2, size=kernel_size, mode="reflect")
        std = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None))
        return std / (np.abs(mean) + 1e-6)
    if stat == "median":
        return ndimage.median_filter(band, size=kernel_size, mode="reflect")
    raise ValueError(f"未知邻域统计: {stat}")


# ============================================================
# 叠加交叉表
# ============================================================

def overlay_crosstab(
    array_a: np.ndarray,
    array_b: np.ndarray,
    names_a: Optional[List[str]] = None,
    names_b: Optional[List[str]] = None,
    pixel_size_m: float = 10.0,
) -> Dict:
    """
    双栅格叠加交叉表 (分类 × 指标分级/分类)。

    典型应用: 土地覆盖 × 干旱分级 → "哪类地物正经历重度干旱";
              盐渍化 × 植被覆盖 → "盐渍化与植被退化的空间关系"。

    参数:
        array_a: (H, W) 栅格 A (整数类别)
        array_b: (H, W) 栅格 B (整数类别或分级)
        names_a: A 的类别名列表 (索引=类别值)
        names_b: B 的类别名列表
        pixel_size_m: 像元大小

    返回:
        dict: {
            "crosstab": (nA, nB) 交叉计数矩阵,
            "areas_km2": (nA, nB) 交叉面积矩阵,
            "rows": [(名称A, 名称B, 面积km², 占比%), ...] 排序后,
            "top_relations": [(名称A, 名称B, 面积), ...] 前5关系
        }
    """
    array_a = np.asarray(array_a, dtype=np.int16)
    array_b = np.asarray(array_b, dtype=np.int16)

    if array_a.shape != array_b.shape:
        raise ValueError(f"栅格尺寸不一致: {array_a.shape} vs {array_b.shape}")

    # 有效像元
    valid = np.isfinite(array_a) & np.isfinite(array_b)
    if not valid.any():
        raise ValueError("无有效叠加像元")

    vals_a = array_a[valid]
    vals_b = array_b[valid]

    # 自动类别范围
    max_a = int(max(vals_a.max(), 0))
    max_b = int(max(vals_b.max(), 0))

    # 交叉计数
    crosstab = np.zeros((max_a + 1, max_b + 1), dtype=np.int64)
    np.add.at(crosstab, (vals_a, vals_b), 1)

    area_per_pixel = pixel_size_m ** 2 / 1e6  # km²
    areas = crosstab * area_per_pixel
    total = int(valid.sum())

    def _name(names, idx):
        if names and idx < len(names):
            return str(names[idx])
        return f"类{idx}"

    # 行列表 (按面积降序)
    rows = []
    for i in range(max_a + 1):
        for j in range(max_b + 1):
            if crosstab[i, j] > 0:
                rows.append({
                    "A": _name(names_a, i),
                    "B": _name(names_b, j),
                    "面积_km2": round(float(areas[i, j]), 2),
                    "占比_pct": round(float(crosstab[i, j]) / total * 100, 2),
                    "a_val": i,
                    "b_val": j,
                })
    rows.sort(key=lambda r: r["面积_km2"], reverse=True)

    return {
        "crosstab": crosstab,
        "areas_km2": areas,
        "rows": rows,
        "top_relations": [
            {"A": r["A"], "B": r["B"], "面积_km2": r["面积_km2"]}
            for r in rows[:5]
        ],
    }


def overlay_analysis_text(result: Dict) -> str:
    """将叠加分析结果转为结构化文本 (供 AI 解读)。"""
    if not result["top_relations"]:
        return "叠加分析未发现显著的空间关系。"
    top = result["top_relations"][0]
    text = (
        f"叠加分析发现面积最大的空间关系是「{top['A']} × {top['B']}」"
        f"({top['面积_km2']:.2f} km²)。"
    )
    if len(result["top_relations"]) > 1:
        others = "、".join(
            f"{r['A']}×{r['B']}({r['面积_km2']:.1f}km²)"
            for r in result["top_relations"][1:3]
        )
        text += f" 其次为 {others}。"
    return text
