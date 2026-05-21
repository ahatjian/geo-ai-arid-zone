"""indices.py 单元测试 — 遥感指数计算"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from utils.indices import calc_ndvi, calc_mndwi, calc_evi, calc_aweish, load_bands_from_geotiff

class TestIndices:
    def test_calc_ndvi_normal(self):
        red = np.full((5, 5), 0.15)
        nir = np.full((5, 5), 0.30)
        ndvi = calc_ndvi(red, nir)
        assert np.allclose(ndvi, 0.3333, atol=0.01)

    def test_calc_ndvi_zero_division(self):
        zero = np.zeros((10, 10))
        ndvi = calc_ndvi(zero, zero)
        assert not np.any(np.isinf(ndvi))

    def test_calc_ndvi_nan_input(self):
        nan_arr = np.full((10, 10), np.nan)
        ndvi = calc_ndvi(nan_arr, nan_arr)
        assert np.all(np.isnan(ndvi))

    def test_calc_mndwi(self):
        green = np.full((5, 5), 0.12)
        swir1 = np.full((5, 5), 0.25)
        mndwi = calc_mndwi(green, swir1)
        assert np.all(np.isfinite(mndwi))
        assert -1 <= np.nanmean(mndwi) <= 1

    def test_calc_evi(self):
        blue = np.full((5, 5), 0.08)
        red = np.full((5, 5), 0.15)
        nir = np.full((5, 5), 0.30)
        evi = calc_evi(blue, red, nir)
        assert np.all(np.isfinite(evi))

    def test_calc_aweish(self):
        green = np.full((5, 5), 0.12)
        swir1 = np.full((5, 5), 0.25)
        nir = np.full((5, 5), 0.30)
        swir2 = np.full((5, 5), 0.22)
        result = calc_aweish(green=green, nir=nir, swir1=swir1, swir2=swir2)
        assert np.all(np.isfinite(result))
