"""BFAST 时序断点检测测试 — 植被突变检测"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_mutated_series(n=36, break_idx=20, magnitude=-0.12, seed=42,
                        seasonal=0.05, trend=0.001):
    """生成含季节 + 趋势 + 突变的模拟 NDVI 时序"""
    t = np.arange(n)
    base = 0.35 + seasonal * np.sin(2 * np.pi * t / 12) + trend * t
    mutated = base.copy()
    mutated[break_idx:] += magnitude
    noise = np.random.default_rng(seed).normal(0, 0.01, n)
    return mutated + noise, t


class TestDetectBreaks:
    def test_detects_major_break(self):
        """应检出主要突变点 (误差 ±3 期)"""
        from utils.bfast import detect_breaks
        values, _ = make_mutated_series()
        result = detect_breaks(values, seasonal_period=12)
        assert result["break_indices"], "应检出断点"
        nearest = min(result["break_indices"], key=lambda i: abs(i - 20))
        assert abs(nearest - 20) <= 3

    def test_break_direction_negative(self):
        """负向突变应被识别为负向"""
        from utils.bfast import detect_breaks
        values, _ = make_mutated_series(magnitude=-0.12)
        result = detect_breaks(values, seasonal_period=12)
        nearest_idx = min(result["break_indices"], key=lambda i: abs(i - 20))
        i = result["break_indices"].index(nearest_idx)
        assert result["directions"][i] == "负向突变"
        assert result["magnitudes"][i] < 0

    def test_positive_break_detected(self):
        """正向突变 (生态恢复) 应被检出"""
        from utils.bfast import detect_breaks
        values, _ = make_mutated_series(magnitude=+0.15, seed=7)
        result = detect_breaks(values, seasonal_period=12)
        nearest_idx = min(result["break_indices"], key=lambda i: abs(i - 20))
        i = result["break_indices"].index(nearest_idx)
        assert result["directions"][i] == "正向突变"
        assert result["magnitudes"][i] > 0

    def test_no_break_series(self):
        """无突变时序不应检出显著断点"""
        from utils.bfast import detect_breaks
        t = np.arange(36)
        values = 0.35 + 0.05 * np.sin(2 * np.pi * t / 12) + 0.001 * t
        values += np.random.default_rng(1).normal(0, 0.005, 36)
        result = detect_breaks(values, seasonal_period=12, alpha=0.01)
        # 可能检出小断点, 但幅度应小
        if result["break_indices"]:
            assert max(abs(m) for m in result["magnitudes"]) < 0.05

    def test_short_series_graceful(self):
        from utils.bfast import detect_breaks
        result = detect_breaks(np.array([0.3, 0.4, 0.5, 0.4, 0.6]),
                               seasonal_period=12)
        assert result["break_indices"] == []
        assert "未执行" in result["method"]

    def test_nan_handling(self):
        from utils.bfast import detect_breaks
        values, _ = make_mutated_series()
        values[15] = np.nan
        result = detect_breaks(values, seasonal_period=12)
        assert result["break_indices"], "NaN 应被插值处理"


class TestChowTest:
    def test_significant_at_break(self):
        from utils.bfast import chow_test
        values, _ = make_mutated_series()
        p = chow_test(values, 20, min_segment=6)
        assert p < 0.05

    def test_not_significant_elsewhere(self):
        from utils.bfast import chow_test
        values, _ = make_mutated_series()
        p = chow_test(values, 5, min_segment=6)
        assert p > 0.05

    def test_invalid_index(self):
        from utils.bfast import chow_test
        values, _ = make_mutated_series()
        assert chow_test(values, 2, min_segment=6) == 1.0
        assert chow_test(values, 34, min_segment=6) == 1.0


class TestVegetationBreaks:
    def test_dates_filled(self):
        from utils.bfast import detect_vegetation_breaks
        values, _ = make_mutated_series()
        dates = [f"2023-{m+1:02d}" for m in range(36)]
        result = detect_vegetation_breaks(values, dates, seasonal_period=12)
        assert len(result["break_dates"]) == len(result["break_indices"])
        if result["break_indices"]:
            assert result["break_dates"][0].startswith("2023")

    def test_summarize(self):
        from utils.bfast import detect_breaks, summarize_breaks
        values, _ = make_mutated_series()
        result = detect_breaks(values, seasonal_period=12)
        rows = summarize_breaks(result)
        assert len(rows) == len(result["break_indices"])
        if rows:
            assert "突变幅度" in rows[0]
            assert "方向" in rows[0]
