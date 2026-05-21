"""trend.py 单元测试 — Sen+MK 趋势分析"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from utils.trend import theil_sen_slope, calc_sen_mk_trend, calc_mean_timeseries_trend

class TestTrend:
    def test_theil_sen_perfect_linear(self):
        ts = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        slope = theil_sen_slope(ts)
        assert slope == pytest.approx(2.0)

    def test_theil_sen_short_series(self):
        from utils.trend import theil_sen_slope
        ts = np.array([0.3, 0.31])
        # Short series (< 3) returns 0.0
        slope = theil_sen_slope(ts)
        assert slope == pytest.approx(0.0, abs=0.01)

    def test_theil_sen_all_nan(self):
        ts = np.full(10, np.nan)
        slope = theil_sen_slope(ts)
        assert slope == pytest.approx(0.0)

    def test_theil_sen_negative_trend(self):
        ts = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        slope = theil_sen_slope(ts)
        assert slope < 0

    def test_calc_sen_mk_trend(self):
        ts = np.random.randn(30).cumsum() * 0.01 + 0.4
        result = calc_sen_mk_trend(ts)
        assert "slope" in result
        assert "trend" in result
        assert "p_value" in result

    def test_calc_mean_timeseries_trend(self):
        ndvi_stack = np.random.rand(20, 20, 12) * 0.3 + 0.25
        result = calc_mean_timeseries_trend(ndvi_stack)
        assert "slope" in result
        assert "trend" in result
