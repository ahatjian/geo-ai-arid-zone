"""月度时序合成测试 — composite 模块"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_ndvi_series(n=6, h=20, w=20, seed=0):
    """生成含云噪声的 NDVI 时序 (仅部分景有云, 云处值显著偏低)"""
    rng = np.random.default_rng(seed)
    series = []
    for i in range(n):
        base = 0.3 + 0.05 * i / n
        arr = base + rng.normal(0, 0.02, (h, w))
        # 仅奇数景有云污染 (模拟偶发云), 偶数景晴空
        if i % 2 == 0:
            arr[5:8, 5:8] = 0.05
        series.append(arr)
    return series


class TestCompositeNDVIMonthly:
    def test_basic_composite(self):
        from utils.composite import composite_ndvi_monthly
        ndvi_list = make_ndvi_series(6)
        dates = ["2025-01-05", "2025-01-15", "2025-02-01", "2025-02-15", "2025-03-01", "2025-03-15"]
        monthly, labels = composite_ndvi_monthly(ndvi_list, dates, method="max")
        assert monthly.shape == (20, 20, 3)
        assert labels == ["2025-01", "2025-02", "2025-03"]

    def test_max_removes_cloud_noise(self):
        """MVC 应消除云污染 (云处值偏低)"""
        from utils.composite import composite_ndvi_monthly
        ndvi_list = make_ndvi_series(6)
        dates = ["2025-01-05", "2025-01-15"] * 3
        monthly, _ = composite_ndvi_monthly(ndvi_list, dates, method="max")
        # 云污染区 (5:8, 5:8) 合成后应接近正常值而非 0.05
        assert monthly[6, 6, 0] > 0.15

    def test_methods_comparable(self):
        from utils.composite import composite_ndvi_monthly
        ndvi_list = make_ndvi_series(4)
        dates = ["2025-01-05", "2025-01-15", "2025-01-20", "2025-01-25"]
        m_max, _ = composite_ndvi_monthly(ndvi_list, dates, method="max")
        m_mean, _ = composite_ndvi_monthly(ndvi_list, dates, method="mean")
        m_med, _ = composite_ndvi_monthly(ndvi_list, dates, method="median")
        # 最大值 ≥ 均值 ≥ 中值 (一般情况)
        assert np.nanmean(m_max) >= np.nanmean(m_mean) >= np.nanmean(m_med) - 0.02

    def test_length_mismatch_raises(self):
        from utils.composite import composite_ndvi_monthly
        with pytest.raises(ValueError):
            composite_ndvi_monthly(make_ndvi_series(3), ["2025-01-01"], method="max")

    def test_empty_raises(self):
        from utils.composite import composite_ndvi_monthly
        with pytest.raises(ValueError):
            composite_ndvi_monthly([], [], method="max")


class TestSeriesByMonth:
    def test_monthly_means(self):
        from utils.composite import composite_series_by_month
        ndvi_list = make_ndvi_series(6)
        dates = ["2025-01-05", "2025-01-15", "2025-02-01", "2025-02-15", "2025-03-01", "2025-03-15"]
        means, labels = composite_series_by_month(ndvi_list, dates, method="mean")
        assert means.shape == (3,)
        assert labels == ["2025-01", "2025-02", "2025-03"]
        assert np.all(np.isfinite(means))

    def test_increasing_trend_preserved(self):
        """月份间上升趋势在合成后保留"""
        from utils.composite import composite_series_by_month
        rng = np.random.default_rng(3)
        ndvi_list = [0.2 + 0.1 * i + rng.normal(0, 0.01, (10, 10)) for i in range(4)]
        dates = [f"2025-0{i+1}-10" for i in range(4)]
        means, _ = composite_series_by_month(ndvi_list, dates, method="max")
        assert means[-1] > means[0]


class TestMergeMax:
    def test_merge_ndvi_max(self):
        from utils.composite import merge_ndvi_max
        ndvi_list = make_ndvi_series(4)
        merged = merge_ndvi_max(ndvi_list)
        assert merged.shape == (20, 20)
        # 合并后每个像元 ≥ 任一单景对应像元
        assert np.all(merged >= ndvi_list[0] - 1e-9)
