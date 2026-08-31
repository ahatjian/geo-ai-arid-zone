"""核心专题模块单元测试 — 沙漠化/冰冻圈/生态/农业干旱"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_bands(h=40, w=40, seed=42):
    """生成模拟 6 波段反射率 (0-1), 顺序 [B,G,R,NIR,SWIR1,SWIR2]"""
    rng = np.random.default_rng(seed)
    return np.stack([
        0.10 + rng.random((h, w)) * 0.05,  # B
        0.15 + rng.random((h, w)) * 0.05,  # G
        0.18 + rng.random((h, w)) * 0.05,  # R
        0.35 + rng.random((h, w)) * 0.10,  # NIR
        0.25 + rng.random((h, w)) * 0.05,  # SWIR1
        0.20 + rng.random((h, w)) * 0.05,  # SWIR2
    ])


class TestDesertification:
    """沙漠化评估 — 反照率/DDI/TGSI/分级"""

    def test_calc_albedo_s2_range(self):
        from utils.desertification import calc_albedo_s2
        b = make_bands()
        albedo = calc_albedo_s2(b[0], b[2], b[3], b[4], b[5])
        assert albedo.shape == (40, 40)
        assert albedo.min() >= 0.0 and albedo.max() <= 1.0

    def test_calc_albedo_landsat_same_as_s2(self):
        from utils.desertification import calc_albedo_landsat, calc_albedo_s2
        b = make_bands()
        assert np.allclose(
            calc_albedo_landsat(b[0], b[2], b[3], b[4], b[5]),
            calc_albedo_s2(b[0], b[2], b[3], b[4], b[5]),
        )

    def test_calc_albedo_dn_autoscale(self):
        """DN 值 (0-10000) 应自动缩放到 0-1"""
        from utils.desertification import calc_albedo_s2
        b = make_bands() * 10000
        albedo = calc_albedo_s2(b[0], b[2], b[3], b[4], b[5])
        assert albedo.max() <= 1.0

    def test_calc_ddi_finite(self):
        from utils.desertification import calc_ddi
        rng = np.random.default_rng(1)
        albedo = rng.uniform(0.1, 0.5, (40, 40))
        ndvi = rng.uniform(0.0, 0.5, (40, 40))
        ddi = calc_ddi(albedo, ndvi)
        assert ddi.shape == (40, 40)
        assert np.all(np.isfinite(ddi))

    def test_classify_by_ddi_extreme(self):
        from utils.desertification import classify_desertification
        # 高 DDI (>6) = 重度沙漠化
        ddi = np.full((10, 10), 6.5)
        cat = classify_desertification(ddi, method="ddi")
        assert cat.max() == 4
        # 低 DDI (<3) = 非沙漠化
        ddi_low = np.full((10, 10), 1.0)
        cat_low = classify_desertification(ddi_low, method="ddi")
        assert cat_low.min() == 0

    def test_classify_composite_method(self):
        from utils.desertification import classify_desertification
        # 高反照率 + 低 NDVI → 重度
        albedo = np.full((10, 10), 0.45)
        ndvi = np.full((10, 10), 0.05)
        ddi = np.full((10, 10), 6.5)
        cat = classify_desertification(ddi, ndvi=ndvi, albedo=albedo, method="composite")
        assert cat.max() >= 3

    def test_assess_desertification_stats(self):
        from utils.desertification import assess_desertification
        result = assess_desertification(make_bands(), satellite="Sentinel-2 L2A")
        assert result.category.shape == (40, 40)
        assert result.stats, "统计不应为空"
        ratios = {s["name"]: s["ratio"] for s in result.stats}
        assert abs(sum(ratios.values()) - 1.0) < 0.01


class TestCryosphere:
    """冰冻圈分析 — NDSI/雪盖/冰川"""

    def test_calc_ndsi_high_for_snow(self):
        from utils.cryosphere import calc_ndsi
        green = np.full((10, 10), 0.8)
        swir1 = np.full((10, 10), 0.05)
        ndsi = calc_ndsi(green, swir1)
        assert np.all(ndsi > 0.4)

    def test_calc_ndsi_low_for_soil(self):
        from utils.cryosphere import calc_ndsi
        green = np.full((10, 10), 0.2)
        swir1 = np.full((10, 10), 0.3)
        ndsi = calc_ndsi(green, swir1)
        assert np.all(ndsi < 0.1)

    def test_calc_ndsi_zero_denominator(self):
        from utils.cryosphere import calc_ndsi
        ndsi = calc_ndsi(np.zeros((5, 5)), np.zeros((5, 5)))
        assert ndsi.shape == (5, 5)

    def test_calc_snow_cover_threshold(self):
        from utils.cryosphere import calc_ndsi, calc_snow_cover
        rng = np.random.default_rng(1)
        ndsi = rng.uniform(-0.5, 0.9, (30, 30))
        snow = calc_snow_cover(ndsi, threshold=0.4)
        assert set(np.unique(snow)) <= {0, 1, 2}
        assert snow.shape == (30, 30)

    def test_compute_snow_cover_stats(self):
        from utils.cryosphere import compute_snow_cover_stats
        snow = np.zeros((50, 50), dtype=np.int8)
        snow[:25, :] = 1  # 一半积雪
        stats = compute_snow_cover_stats(snow, pixel_size_m=10.0)
        assert stats, "统计不应为空"
        total = sum(s["area_km2"] for s in stats)
        assert total > 0

    def test_assess_cryosphere(self):
        from utils.cryosphere import assess_cryosphere
        result = assess_cryosphere(make_bands(40, 40, seed=2))
        assert result.ndsi.shape == (40, 40)


class TestEcology:
    """生态评估 — PSR 模型"""

    def test_calc_psi_low_ndvi_high_pressure(self):
        from utils.ecology import calc_psi
        # 低 NDVI → 高压力
        psi = calc_psi(np.full((10, 10), 0.05))
        assert np.all(psi > 0.5)
        # 高 NDVI → 低压力
        psi2 = calc_psi(np.full((10, 10), 0.6))
        assert np.all(psi2 < 0.4)

    def test_calc_psi_range(self):
        from utils.ecology import calc_psi
        rng = np.random.default_rng(2)
        psi = calc_psi(rng.uniform(-0.1, 0.8, (30, 30)))
        assert psi.min() >= 0.0 and psi.max() <= 1.0

    def test_calc_ssi_high_ndvi_good_state(self):
        from utils.ecology import calc_ssi
        ssi = calc_ssi(np.full((10, 10), 0.7), water_ratio=0.2)
        assert np.all(ssi > 0.5)
        ssi_low = calc_ssi(np.full((10, 10), 0.0))
        assert np.all(ssi_low < 0.4)

    def test_calc_rsi(self):
        from utils.ecology import calc_rsi
        rsi = calc_rsi(np.full((10, 10), 0.08))  # 正趋势 → 恢复好
        assert np.all(rsi > 0.5)
        rsi_neg = calc_rsi(np.full((10, 10), -0.08))
        assert np.all(rsi_neg < 0.4)

    def test_calc_esi_weighted(self):
        from utils.ecology import calc_esi
        psi = np.full((10, 10), 0.8)   # 高压力
        ssi = np.full((10, 10), 0.2)   # 差状态
        rsi = np.full((10, 10), 0.2)   # 弱响应
        esi = calc_esi(psi, ssi, rsi)
        assert 0 <= esi.min() <= 0.5

    def test_classify_eco_security_levels(self):
        from utils.ecology import classify_eco_security
        esi = np.array([0.9, 0.5, 0.25])
        cat = classify_eco_security(esi)
        assert cat[0] < cat[2]  # 高 ESI → 低等级 (更安全)

    def test_assess_eco_security(self):
        from utils.ecology import assess_eco_security
        rng = np.random.default_rng(3)
        ndvi = rng.uniform(0.1, 0.7, (30, 30))
        result = assess_eco_security(ndvi)
        assert result.esi.shape == (30, 30)
        assert result.category.shape == (30, 30)


class TestAgriDrought:
    """农业干旱 — CWSI/SMI/MPDI"""

    def test_calc_cwsi_ndvi_range(self):
        from utils.agri_drought import calc_cwsi_ndvi
        ndvi = np.array([0.1, 0.3, 0.5, 0.7])
        cwsi = calc_cwsi_ndvi(ndvi)
        assert np.all(np.isfinite(cwsi))
        # 干旱时 NDVI 低 → CWSI 高
        assert cwsi[0] > cwsi[-1]

    def test_calc_smi_swir_range(self):
        from utils.agri_drought import calc_smi_swir
        swir1 = np.array([0.2, 0.3, 0.4])
        smi = calc_smi_swir(swir1)
        assert np.all(np.isfinite(smi))

    def test_calc_mpdi_two_band(self):
        from utils.agri_drought import calc_mpdi
        b = make_bands(20, 20, seed=7)
        mpdi = calc_mpdi(b[2], b[3])  # red, nir
        assert mpdi.shape == (20, 20)
        assert np.all(np.isfinite(mpdi))

    def test_classify_agri_drought_levels(self):
        from utils.agri_drought import classify_agri_drought
        cat = classify_agri_drought(np.array([0.15, 0.5, 0.9]))
        assert cat[0] == 0  # 低 CWSI → 无干旱
        assert cat[-1] == 4  # 高 CWSI → 极度干旱
        assert cat[0] < cat[-1]

    def test_assess_agri_drought(self):
        from utils.agri_drought import assess_agri_drought
        result = assess_agri_drought(make_bands(25, 25, seed=8))
        assert result.category.shape == (25, 25)
        assert result.stats
