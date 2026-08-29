"""
土壤盐渍化模块单元测试
测试: salinity (SI/NDSI/BI 指数 + 分级 + 评估)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_bands():
    """模拟 Sentinel-2 6波段数据 (含盐渍土特征: 红波段高、近红外相对低)"""
    np.random.seed(42)
    H, W = 50, 50
    blue = 0.20 + np.random.randn(H, W) * 0.03
    green = 0.24 + np.random.randn(H, W) * 0.03
    red = 0.30 + np.random.randn(H, W) * 0.04
    nir = 0.26 + np.random.randn(H, W) * 0.04
    swir1 = 0.30 + np.random.randn(H, W) * 0.04
    swir2 = 0.28 + np.random.randn(H, W) * 0.04
    return np.stack([blue, green, red, nir, swir1, swir2])


class TestSalinityIndex:
    def test_calc_si(self, mock_bands):
        from utils.salinity import calc_si
        si = calc_si(mock_bands[0], mock_bands[2])
        assert si.shape == (50, 50)
        assert np.all(np.isfinite(si))
        assert np.nanmin(si) >= 0

    def test_calc_ndsi_salinity(self, mock_bands):
        from utils.salinity import calc_ndsi_salinity
        ndsi = calc_ndsi_salinity(mock_bands[2], mock_bands[3])
        assert ndsi.shape == (50, 50)
        assert -1 <= np.nanmean(ndsi) <= 1

    def test_calc_bi(self, mock_bands):
        from utils.salinity import calc_bi
        bi = calc_bi(mock_bands[2], mock_bands[3])
        assert bi.shape == (50, 50)
        assert np.nanmin(bi) >= 0

    def test_calc_si1_si2(self, mock_bands):
        from utils.salinity import calc_si1, calc_si2
        si1 = calc_si1(mock_bands[1], mock_bands[2])
        si2 = calc_si2(mock_bands[1], mock_bands[2], mock_bands[3])
        assert si1.shape == (50, 50)
        assert si2.shape == (50, 50)
        assert np.nanmin(si1) >= 0
        assert np.nanmin(si2) >= 0


class TestSalinityClassification:
    def test_classify_salinity_ndsi(self):
        from utils.salinity import classify_salinity
        ndsi = np.array([[-0.2, 0.0, 0.04], [0.12, 0.20, 0.30]])
        cat = classify_salinity(ndsi, method="ndsi")
        assert cat.shape == (2, 3)
        assert cat.min() >= 0 and cat.max() <= 4

    def test_classify_salinity_composite(self):
        from utils.salinity import classify_salinity
        ndsi = np.full((10, 10), 0.20)  # 重度盐渍化
        ndvi = np.full((10, 10), 0.5)   # 但全是植被 → 应判为非盐渍化
        cat = classify_salinity(ndsi, ndvi=ndvi, method="composite")
        assert np.all(cat == 0)

    def test_compute_salinity_stats(self):
        from utils.salinity import compute_salinity_stats
        cat = np.random.randint(0, 5, (30, 30)).astype(np.int8)
        stats = compute_salinity_stats(cat, pixel_size_m=10)
        assert len(stats) == 5
        total_px = sum(s["pixel_count"] for s in stats)
        assert total_px == 900

    def test_get_salinity_colormap(self):
        from utils.salinity import get_salinity_colormap
        cat = np.zeros((5, 5), dtype=np.uint8)
        cat[1, 1] = 4
        rgb = get_salinity_colormap(cat)
        assert rgb.shape == (5, 5, 3)
        assert rgb.dtype == np.uint8


class TestAssessSalinity:
    def test_assess_salinity(self, mock_bands):
        from utils.salinity import assess_salinity
        result = assess_salinity(mock_bands, satellite="Sentinel-2 L2A", pixel_size_m=10)
        assert result.si is not None
        assert result.ndsi_salt is not None
        assert result.bi is not None
        assert result.category is not None
        assert len(result.stats) == 5
        assert "total_salinization_ratio" in result.summary
        assert "dominant_level" in result.summary

    def test_assess_salinity_composite(self, mock_bands):
        from utils.salinity import assess_salinity
        result = assess_salinity(mock_bands, method="composite", veg_threshold=0.4)
        assert result.summary["method"] == "composite"
        assert 0 <= result.summary["total_salinization_ratio"] <= 1
