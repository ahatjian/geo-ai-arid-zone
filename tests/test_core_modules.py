"""
核心计算模块单元测试
测试: drought, desertification, forecast, cryosphere, ecology, agri_drought
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_bands():
    """模拟 Sentinel-2 6波段数据"""
    np.random.seed(42)
    H, W = 50, 50
    blue = 0.08 + np.random.randn(H, W) * 0.01
    green = 0.12 + np.random.randn(H, W) * 0.02
    red = 0.15 + np.random.randn(H, W) * 0.02
    nir = 0.30 + np.random.randn(H, W) * 0.03
    swir1 = 0.25 + np.random.randn(H, W) * 0.03
    swir2 = 0.22 + np.random.randn(H, W) * 0.02
    return np.stack([blue, green, red, nir, swir1, swir2])


@pytest.fixture
def mock_ndvi_stack():
    """模拟 NDVI 时序立方体 (H,W,T)"""
    np.random.seed(42)
    H, W, T = 40, 40, 24
    stack = np.zeros((H, W, T))
    for t in range(T):
        stack[:, :, t] = 0.3 + 0.15 * np.sin(2*np.pi*t/12) + np.random.randn(H, W)*0.03
    return stack


# ============================================================
# Drought 模块测试
# ============================================================

class TestDrought:
    def test_calc_nddi(self):
        from utils.drought import calc_nddi
        # Use correlated data (NDVI ≈ 0.3, NDWI ≈ -0.2) to avoid division blowup
        ndvi = np.full((20, 20), 0.3) + np.random.rand(20, 20) * 0.1
        ndwi = np.full((20, 20), -0.2) + np.random.rand(20, 20) * 0.1
        nddi = calc_nddi(ndvi, ndwi)
        assert nddi.shape == (20, 20)
        assert np.all(np.isfinite(nddi))

    def test_calc_vci_pixelwise(self, mock_ndvi_stack):
        from utils.drought import calc_vci_pixelwise
        result = calc_vci_pixelwise(mock_ndvi_stack)
        assert "vci_latest" in result
        assert "drought_category" in result
        assert result["vci_latest"].shape == (40, 40)
        assert 0 <= np.nanmean(result["vci_latest"]) <= 100

    def test_compute_drought_stats(self):
        from utils.drought import compute_drought_stats
        cat = np.random.randint(-1, 5, (30, 30)).astype(np.int8)
        stats = compute_drought_stats(cat, pixel_size_m=10)
        assert len(stats) > 0
        assert "name" in stats[0]
        total_px = sum(s["pixel_count"] for s in stats)
        assert total_px == 900


# ============================================================
# Desertification 模块测试
# ============================================================

class TestDesertification:
    def test_calc_albedo(self, mock_bands):
        from utils.desertification import calc_albedo_s2
        albedo = calc_albedo_s2(mock_bands[0], mock_bands[2],
                                mock_bands[3], mock_bands[4], mock_bands[5])
        assert albedo.shape == (50, 50)
        assert 0 <= np.nanmean(albedo) <= 1

    def test_calc_tgsi(self, mock_bands):
        from utils.desertification import calc_tgsi
        tgsi = calc_tgsi(mock_bands[2], mock_bands[4], blue=mock_bands[0])
        assert tgsi.shape == (50, 50)
        assert -1 <= np.nanmean(tgsi) <= 1

    def test_assess_desertification(self, mock_bands):
        from utils.desertification import assess_desertification
        result = assess_desertification(mock_bands, pixel_size_m=10)
        assert result.category is not None
        assert result.albedo is not None
        assert result.ddi is not None
        assert len(result.stats) == 5
        assert "total_desertification_ratio" in result.summary


# ============================================================
# Forecast 模块测试
# ============================================================

class TestForecast:
    def test_holt_winters(self):
        from utils.forecast import forecast_holt_winters
        ts = 0.35 + 0.15 * np.sin(2*np.pi*np.arange(24)/12)
        result = forecast_holt_winters(ts, forecast_steps=6)
        assert len(result.forecast_values) == 6
        assert result.method.startswith("Holt-Winters")

    def test_forecast_drought_trend(self, mock_ndvi_stack):
        from utils.forecast import forecast_drought_trend
        result = forecast_drought_trend(mock_ndvi_stack, forecast_steps=4, method="sarima")
        assert result.forecast_steps == 4
        assert "SARIMA" in result.method

    def test_evaluate_forecast(self):
        from utils.forecast import evaluate_forecast
        actual = np.array([0.3, 0.35, 0.4, 0.38, 0.42])
        predicted = actual + 0.02
        metrics = evaluate_forecast(actual, predicted)
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert metrics["RMSE"] < 0.1


# ============================================================
# Cryosphere 模块测试
# ============================================================

class TestCryosphere:
    def test_calc_ndsi(self, mock_bands):
        from utils.cryosphere import calc_ndsi
        ndsi = calc_ndsi(mock_bands[1], mock_bands[4])
        assert ndsi.shape == (50, 50)
        assert -1 <= np.nanmean(ndsi) <= 1

    def test_calc_snow_cover(self):
        from utils.cryosphere import calc_snow_cover
        ndsi = np.random.rand(30, 30) * 0.9
        cat = calc_snow_cover(ndsi, method="classified")
        assert cat.shape == (30, 30)
        assert cat.max() <= 3
        assert cat.min() >= 0

    def test_assess_cryosphere(self, mock_bands):
        from utils.cryosphere import assess_cryosphere
        result = assess_cryosphere(mock_bands, pixel_size_m=10)
        assert result.ndsi is not None
        assert result.snow_cover is not None
        assert result.glacier_mask is not None
        assert "total_snow_cover_ratio" in result.summary


# ============================================================
# Ecology 模块测试
# ============================================================

class TestEcology:
    def test_calc_esi(self):
        from utils.ecology import calc_psi, calc_ssi, calc_rsi, calc_esi
        ndvi = np.random.rand(20, 20) * 0.6 + 0.1
        psi = calc_psi(ndvi)
        ssi = calc_ssi(ndvi)
        rsi = calc_rsi(np.zeros_like(ndvi))
        esi = calc_esi(psi, ssi, rsi)
        assert esi.shape == (20, 20)
        assert 0 <= np.nanmean(esi) <= 1

    def test_classify_eco_security(self):
        from utils.ecology import classify_eco_security
        esi = np.random.rand(30, 30)
        cat = classify_eco_security(esi)
        assert cat.min() >= 0
        assert cat.max() <= 4

    def test_assess_eco_security(self):
        from utils.ecology import assess_eco_security
        ndvi = np.random.rand(20, 20) * 0.6 + 0.1
        result = assess_eco_security(ndvi)
        assert result.esi is not None
        assert len(result.stats) == 5


# ============================================================
# Agri Drought 模块测试
# ============================================================

class TestAgriDrought:
    def test_calc_cwsi(self):
        from utils.agri_drought import calc_cwsi_ndvi
        ndvi = np.random.rand(30, 30) * 0.6 + 0.1
        cwsi = calc_cwsi_ndvi(ndvi)
        assert cwsi.shape == (30, 30)
        assert 0 <= np.nanmean(cwsi) <= 1

    def test_calc_smi(self, mock_bands):
        from utils.agri_drought import calc_smi_swir
        smi = calc_smi_swir(mock_bands[4])
        assert smi.shape == (50, 50)
        assert 0 <= np.nanmean(smi) <= 1

    def test_assess_agri_drought(self, mock_bands):
        from utils.agri_drought import assess_agri_drought
        result = assess_agri_drought(mock_bands, pixel_size_m=10)
        assert result.cwsi is not None
        assert result.smi is not None
        assert len(result.stats) == 5
        assert "irrigation" in result.summary


# ============================================================
# LLM 模块测试
# ============================================================

class TestLLM:
    def test_fallback_parse(self):
        from utils.llm import fallback_parse
        result = fallback_parse("帮我分析塔里木盆地2024年的植被变化和干旱情况")
        assert result["study_area"] == "塔里木盆地"
        assert result["year_start"] == 2024
        assert len(result["modules"]) >= 1
        assert result["method"] == "fallback"

    def test_fallback_parse_default(self):
        from utils.llm import fallback_parse
        result = fallback_parse("看看最近的水体变化")
        assert result["study_area"] == "塔里木盆地"
        assert len(result["modules"]) >= 1

    def test_fallback_parse_glacier(self):
        from utils.llm import fallback_parse
        result = fallback_parse("天山的冰川消融情况")
        assert "冰冻圈分析" in [m["name"] for m in result["modules"]]

    def test_is_llm_available(self):
        from utils.llm import is_llm_available
        available = is_llm_available()
        assert isinstance(available, bool)
