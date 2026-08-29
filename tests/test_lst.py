"""
地表温度 (LST) 模块单元测试
测试: lst (K→℃ 反演 + 热环境分级 + 统计 + LST-NDVI 关系)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_lst_kelvin():
    """模拟地表温度 (Kelvin)，含热环境梯度"""
    np.random.seed(42)
    H, W = 50, 50
    # 生成 280-330 K (约 7-57 °C) 的温度场
    base = np.linspace(280, 330, W)
    lst = np.tile(base, (H, 1)) + np.random.randn(H, W) * 2
    # 加入一些无效像元
    lst[0, 0] = np.nan
    return lst.astype(np.float32)


class TestLST:
    def test_kelvin_to_celsius(self):
        from utils.lst import kelvin_to_celsius
        k = np.array([273.15, 300.0, 373.15])
        c = kelvin_to_celsius(k)
        assert np.allclose(c, [0.0, 26.85, 100.0])

    def test_classify_thermal(self, mock_lst_kelvin):
        from utils.lst import classify_thermal
        c = mock_lst_kelvin - 273.15
        cat = classify_thermal(c)
        assert cat.shape == (50, 50)
        assert cat.min() >= 0 and cat.max() <= 4

    def test_compute_lst_stats(self, mock_lst_kelvin):
        from utils.lst import compute_lst_stats, classify_thermal
        c = mock_lst_kelvin - 273.15
        cat = classify_thermal(c)
        stats = compute_lst_stats(cat, c, pixel_size_m=30)
        assert len(stats) == 5
        assert all("mean_lst_c" in s for s in stats)

    def test_assess_thermal(self, mock_lst_kelvin):
        from utils.lst import assess_thermal
        result = assess_thermal(mock_lst_kelvin, pixel_size_m=30)
        assert result.lst_celsius is not None
        assert result.category is not None
        assert len(result.stats) == 5
        assert "mean_lst_c" in result.summary
        assert "hot_ratio" in result.summary
        assert "dominant_level" in result.summary

    def test_get_thermal_colormap(self, mock_lst_kelvin):
        from utils.lst import get_thermal_colormap, classify_thermal
        c = mock_lst_kelvin - 273.15
        cat = classify_thermal(c)
        rgb = get_thermal_colormap(cat)
        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8


class TestLSTNDVIRelation:
    def test_compute_lst_ndvi_relation(self, mock_lst_kelvin):
        from utils.lst import compute_lst_ndvi_relation
        lst_c = mock_lst_kelvin - 273.15
        # 模拟 NDVI 与 LST 负相关 (植被越多温度越低):
        # LST 从左到右递增 (280→330K), NDVI 从左到右递减 (0.8→0.0)
        H, W = lst_c.shape
        ndvi = np.linspace(0.8, 0.0, W)
        ndvi = np.tile(ndvi, (H, 1))
        rel = compute_lst_ndvi_relation(lst_c, ndvi)
        assert "ndvi_bins" in rel
        assert "mean_lst" in rel
        assert "correlation" in rel
        assert rel["correlation"] is not None
        # 构造的负相关数据应得到负相关系数
        assert rel["correlation"] < 0
