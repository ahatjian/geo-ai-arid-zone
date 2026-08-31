"""预测与土地覆盖模块单元测试 — forecast / landcover"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class TestForecast:
    """干旱预测 — 时序准备/评估/简单预测"""

    def test_prepare_ndvi_timeseries_mean(self):
        from utils.forecast import prepare_ndvi_timeseries
        rng = np.random.default_rng(0)
        stack = rng.uniform(0.1, 0.6, (20, 20, 12))
        values, times = prepare_ndvi_timeseries(stack, method="mean")
        assert values.shape == (12,)
        assert times.shape == (12,)
        assert np.all(np.isfinite(values))

    def test_prepare_ndvi_timeseries_3d_required(self):
        from utils.forecast import prepare_ndvi_timeseries
        with pytest.raises(ValueError):
            prepare_ndvi_timeseries(np.zeros((5, 5)))

    def test_prepare_ndvi_timeseries_nan_filled(self):
        """全 NaN 帧应被插值填充"""
        from utils.forecast import prepare_ndvi_timeseries
        stack = np.full((10, 10, 6), 0.3)
        stack[:, :, 3] = np.nan  # 第 4 帧全无效
        values, _ = prepare_ndvi_timeseries(stack, method="mean")
        assert np.all(np.isfinite(values))

    def test_split_train_test(self):
        from utils.forecast import split_train_test
        values = np.linspace(0.2, 0.6, 36)
        train, test = split_train_test(values, test_ratio=0.2)
        assert len(train) + len(test) == 36
        assert len(train) > len(test)

    def test_evaluate_forecast_perfect(self):
        from utils.forecast import evaluate_forecast
        y = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        result = evaluate_forecast(y, y)  # 完美预测
        assert result["RMSE"] == 0.0
        assert result["MAE"] == 0.0
        assert result["N"] == 5

    def test_evaluate_forecast_imperfect(self):
        from utils.forecast import evaluate_forecast
        y = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        pred = y + 0.1
        result = evaluate_forecast(y, pred)
        assert result["RMSE"] > 0
        assert result["N"] == 5

    def test_holt_winters_forecast(self):
        """Holt-Winters 简单预测应输出 forecast_values"""
        from utils.forecast import forecast_holt_winters
        values = 0.4 + 0.02 * np.sin(np.arange(24) / 2.0) + 0.01 * np.arange(24)
        result = forecast_holt_winters(values, forecast_steps=4)
        pred = np.asarray(result.forecast_values)
        assert len(pred) == 4
        assert np.all(np.isfinite(pred))

    def test_forecast_drought_trend(self):
        from utils.forecast import forecast_drought_trend
        rng = np.random.default_rng(1)
        stack = rng.uniform(0.2, 0.5, (10, 10, 24))
        result = forecast_drought_trend(stack, method="holt_winters", forecast_steps=3)
        pred = np.asarray(result.forecast_values)
        assert len(pred) == 3


class TestLandcover:
    """土地覆盖 — ESA/ESRI 类别映射与统计"""

    def test_esa_to_arid6_mapping(self):
        from utils.landcover import esa_to_arid6
        # ESA 类别 40 (农田) → 干旱区 4 (农田)
        arr = np.full((5, 5), 40, dtype=np.uint8)
        result = esa_to_arid6(arr)
        assert np.all(result == 4)
        # ESA 类别 80 (水域) → 干旱区 5 (水体)
        arr_water = np.full((5, 5), 80, dtype=np.uint8)
        assert np.all(esa_to_arid6(arr_water) == 5)

    def test_esri_to_arid6_mapping(self):
        from utils.landcover import esri_to_arid6
        # ESRI 类别 1 (水体) → 干旱区 5 (水体)
        arr = np.full((5, 5), 1, dtype=np.uint8)
        result = esri_to_arid6(arr)
        assert np.all(result == 5)

    def test_esa_to_arid6_unknown_stays_zero(self):
        from utils.landcover import esa_to_arid6
        arr = np.array([[99]], dtype=np.uint8)  # 未知类别
        result = esa_to_arid6(arr)
        assert result[0, 0] == 0

    def test_esa_classes_definition(self):
        from utils.landcover import ESA_CLASSES, ARID6_CLASSES
        assert len(ESA_CLASSES) == 12  # ESA 原始 11 类 + 无数据(0)
        assert len(ARID6_CLASSES) == 6  # 干旱区 6 类

    def test_compute_landcover_stats(self):
        from utils.landcover import compute_landcover_stats, ARID6_CLASSES
        rng = np.random.default_rng(5)
        classes = rng.integers(0, 6, (50, 50)).astype(np.uint8)
        class_names = {k: v["name"] for k, v in ARID6_CLASSES.items()}
        class_colors = {k: v["color"] for k, v in ARID6_CLASSES.items()}
        stats = compute_landcover_stats(classes, class_names, class_colors, pixel_size_m=10.0)
        assert stats, "统计不应为空"
        total_ratio = sum(s["pixel_ratio"] for s in stats)
        assert abs(total_ratio - 1.0) < 0.01

    def test_esa_tiles_for_bbox(self):
        """塔里木盆地 bbox 应生成 ESA 瓦片名"""
        from utils.landcover import _esa_tiles_for_bbox
        tiles = _esa_tiles_for_bbox([76, 37, 88, 42])
        assert tiles, "应生成至少一个瓦片"
        for t in tiles:
            assert len(t) >= 6  # e.g. "N36E075"
            assert t[0] in "NS" and "E" in t or "W" in t

    def test_get_landcover_for_study_area(self):
        """真实调用 ESA 土地覆盖 (网络可用时)"""
        from utils.landcover import get_landcover_for_study_area
        from config import STUDY_AREAS
        try:
            data, meta = get_landcover_for_study_area("塔里木盆地", STUDY_AREAS)
            assert data is not None
            assert data.ndim == 2
        except Exception:
            pytest.skip("ESA 数据源不可用, 跳过网络测试")
