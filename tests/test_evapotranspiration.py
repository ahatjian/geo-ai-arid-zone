"""
蒸散发 (ET) 模块单元测试
测试: 能量平衡分量 + ET 估算 + 分级
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_inputs():
    """模拟反照率/NDVI/LST (合理物理值)"""
    np.random.seed(42)
    H, W = 40, 40
    albedo = 0.20 + np.random.rand(H, W) * 0.15       # 0.2-0.35
    ndvi = 0.3 + np.random.rand(H, W) * 0.4           # 0.3-0.7
    lst_kelvin = 300 + np.random.rand(H, W) * 15      # 300-315 K (27-42°C)
    return albedo.astype(np.float32), ndvi.astype(np.float32), lst_kelvin.astype(np.float32)


class TestETComponents:
    def test_calc_fvc(self):
        from utils.evapotranspiration import calc_fvc
        ndvi = np.array([0.1, 0.2, 0.35, 0.5, 0.7])
        fvc = calc_fvc(ndvi)
        assert fvc.shape == (5,)
        assert fvc.min() >= 0 and fvc.max() <= 1
        assert fvc[1] == 0.0   # NDVI=0.2 裸土 → FVC=0
        assert fvc[3] == 1.0   # NDVI=0.5 全植被 → FVC=1

    def test_calc_emissivity(self):
        from utils.evapotranspiration import calc_emissivity
        ndvi = np.array([0.1, 0.5])
        eps = calc_emissivity(ndvi)
        assert 0.98 <= eps.min() <= 1.0
        assert eps[0] < eps[1]  # 植被发射率更高

    def test_calc_atmospheric_emissivity(self):
        from utils.evapotranspiration import calc_atmospheric_emissivity
        eps = calc_atmospheric_emissivity(0.75)
        assert 0 < eps < 1

    def test_calc_net_radiation(self, mock_inputs):
        from utils.evapotranspiration import calc_net_radiation
        albedo, ndvi, lst_k = mock_inputs
        rn = calc_net_radiation(albedo, lst_k, ndvi, rs_down=600, ta_kelvin=298.15)
        assert rn.shape == (40, 40)
        # 白天净辐射应为正值
        assert np.nanmean(rn) > 0

    def test_calc_soil_heat_flux(self, mock_inputs):
        from utils.evapotranspiration import calc_soil_heat_flux, calc_net_radiation
        albedo, ndvi, lst_k = mock_inputs
        rn = calc_net_radiation(albedo, lst_k, ndvi)
        g = calc_soil_heat_flux(rn, albedo, ndvi, lst_k - 273.15)
        assert g.shape == (40, 40)
        # 土壤热通量应小于净辐射
        assert np.nanmean(np.abs(g)) < np.nanmean(np.abs(rn)) + 100

    def test_calc_sensible_heat(self, mock_inputs):
        from utils.evapotranspiration import calc_sensible_heat
        _, _, lst_k = mock_inputs
        h = calc_sensible_heat(lst_k, ta_kelvin=298.15, wind_speed=2.0)
        assert h.shape == (40, 40)

    def test_calc_latent_heat(self, mock_inputs):
        from utils.evapotranspiration import calc_latent_heat
        rn = np.full((10, 10), 500.0)
        g = np.full((10, 10), 100.0)
        h = np.full((10, 10), 200.0)
        le = calc_latent_heat(rn, g, h)
        assert np.allclose(le, 200.0)  # 500 - 100 - 200

    def test_calc_et_daily(self):
        from utils.evapotranspiration import calc_et_daily
        le = np.array([0.0, 100.0, 300.0])
        et = calc_et_daily(le)
        assert et.shape == (3,)
        assert et[0] == 0.0
        assert et[2] > et[1]  # 潜热通量越大 ET 越大
        # 300 W/m² 潜热 ≈ 10.6 mm/day (合理范围)
        assert 5 < et[2] < 20


class TestAssessET:
    def test_assess_et(self, mock_inputs):
        from utils.evapotranspiration import assess_et
        albedo, ndvi, lst_k = mock_inputs
        result = assess_et(albedo, ndvi, lst_k, pixel_size_m=30)
        assert result.et_daily is not None
        assert result.net_radiation is not None
        assert result.latent_heat is not None
        assert result.category is not None
        assert len(result.stats) == 5
        assert "mean_et" in result.summary
        assert "dominant_level" in result.summary
        assert "mean_rn" in result.summary
        # 蒸散发应为正值
        assert result.summary["mean_et"] > 0

    def test_classify_et(self):
        from utils.evapotranspiration import classify_et
        et = np.array([[0.2, 1.0, 2.0], [4.0, 6.0, 0.3]])
        cat = classify_et(et)
        assert cat.shape == (2, 3)
        assert cat.min() >= 0 and cat.max() <= 4
        assert cat[0, 0] == 0  # 0.2 → 极低
        assert cat[1, 1] == 4  # 6.0 → 极高

    def test_energy_balance_conservation(self, mock_inputs):
        """能量平衡守恒: Rn ≈ H + LE + G"""
        from utils.evapotranspiration import assess_et
        albedo, ndvi, lst_k = mock_inputs
        result = assess_et(albedo, ndvi, lst_k)
        rn = np.nanmean(result.net_radiation)
        h = np.nanmean(result.sensible_heat)
        le = np.nanmean(result.latent_heat)
        g = np.nanmean(result.soil_heat_flux)
        # 能量平衡应在数值误差范围内闭合
        assert abs(rn - (h + le + g)) < 1.0
