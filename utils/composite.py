"""
月度时序合成模块 — 最大值/均值/中值合成
========================================
将逐景 NDVI 时序合成到月度尺度, 消除云噪声和观测缺失。

常用方法:
  - MAX (最大合成 MVC): 月内每像元取 NDVI 最大值, 最常用,
    有效抑制云污染 (云通常降低 NDVI)
  - MEAN: 月内均值 (简单)
  - MEDIAN: 月内中值 (稳健, 抗离群)

适合:
  - 植被分析页的多景影像 → 月度 NDVI 曲线
  - 干旱监测的 VCI/NDVI 距平时序
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def composite_ndvi_monthly(
    ndvi_list: List[np.ndarray],
    dates: List[str],
    method: str = "max",
) -> Tuple[np.ndarray, List[str]]:
    """
    将多景 NDVI 按月份合成。

    参数:
        ndvi_list: 每景 NDVI 数组列表 [(H,W), ...]
        dates: 每景日期列表 ["YYYY-MM-DD", ...], 与 ndvi_list 对齐
        method: "max" | "mean" | "median"

    返回:
        (monthly_ndvi, month_labels)
        - monthly_ndvi: (H, W, M) 合成结果, M=月份数
        - month_labels: ["YYYY-MM", ...]
    """
    if len(ndvi_list) != len(dates):
        raise ValueError(f"ndvi_list ({len(ndvi_list)}) 与 dates ({len(dates)}) 长度不一致")
    if len(ndvi_list) == 0:
        raise ValueError("ndvi_list 为空")

    # 按月份分组
    months: Dict[str, List[np.ndarray]] = {}
    for ndvi, date_str in zip(ndvi_list, dates):
        month_key = date_str[:7]  # "YYYY-MM"
        months.setdefault(month_key, []).append(np.asarray(ndvi, dtype=np.float64))

    month_labels = sorted(months.keys())
    H, W = months[month_labels[0]][0].shape

    composites = []
    for label in month_labels:
        stack = np.stack(months[label])  # (n, H, W)
        if method == "max":
            with np.errstate(invalid="ignore"):
                comp = np.nanmax(stack, axis=0)
        elif method == "median":
            with np.errstate(invalid="ignore"):
                comp = np.nanmedian(stack, axis=0)
        else:  # mean
            with np.errstate(invalid="ignore"):
                comp = np.nanmean(stack, axis=0)
        composites.append(comp)

    return np.stack(composites, axis=-1), month_labels


def composite_series_by_month(
    ndvi_list: List[np.ndarray],
    dates: List[str],
    method: str = "max",
) -> Tuple[np.ndarray, List[str]]:
    """
    区域均值时序按月份合成 (轻量版, 用于趋势曲线)。

    参数:
        ndvi_list: 每景 NDVI 数组 [(H,W), ...]
        dates: 日期列表
        method: "max" | "mean" | "median"

    返回:
        (monthly_means, month_labels)
        - monthly_means: (M,) 每月区域均值
        - month_labels: ["YYYY-MM", ...]
    """
    monthly, labels = composite_ndvi_monthly(ndvi_list, dates, method=method)
    means = np.array([np.nanmean(monthly[:, :, m]) for m in range(monthly.shape[2])])
    return means, labels


def merge_ndvi_max(
    ndvi_list: List[np.ndarray],
) -> np.ndarray:
    """所有景逐像元取最大值 (全期 MVC)。"""
    if not ndvi_list:
        raise ValueError("ndvi_list 为空")
    stack = np.stack([np.asarray(n, dtype=np.float64) for n in ndvi_list])
    with np.errstate(invalid="ignore"):
        return np.nanmax(stack, axis=0)
