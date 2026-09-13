"""干旱指数模块测试 — drought

覆盖 SPI/SPEI/PET、VCI/TCI/VHI、NDDI/NDWI、NDVI 距平、TVDI 与综合评估。

这个模块此前**没有独立测试文件** —— tests/test_arid_core.py 覆盖的是
desertification / cryosphere / ecology / agri_drought, 而"干旱监测"页面
真正依赖的这批指数一直没被测, 覆盖率仅 19%。

断言尽量落在**数学性质**上 (值域 / 单调性 / 边界 / 公式 / 退化输入),
而不是实现细节 —— 这样即使将来换拟合算法, 这些测试依然有效。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


# ============================================================
# 数据构造
# ============================================================

def make_precip(n=120, seed=0, mean=30.0):
    """含季节性的月降水序列 (mm)"""
    rng = np.random.default_rng(seed)
    months = np.arange(n) % 12
    seasonal = 20.0 * np.sin(2 * np.pi * months / 12)
    return np.clip(mean + seasonal + rng.normal(0, 5, n), 0, None)


def make_ndvi_ts(n=120, seed=1):
    """含物候周期的 NDVI 时序"""
    rng = np.random.default_rng(seed)
    months = np.arange(n) % 12
    return 0.35 + 0.15 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 0.02, n)


def make_lst_ts(n=120, seed=2):
    """含季节周期的地表温度时序 (K)"""
    rng = np.random.default_rng(seed)
    months = np.arange(n) % 12
    return 295.0 + 12.0 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 1.0, n)


# ============================================================
# 等级分类
# ============================================================

class TestClassifyDrought:
    def test_spi_threshold_boundaries(self):
        """阈值判定是 >=, 边界值应归入较高等级"""
        from utils.drought import classify_spi
        assert classify_spi(2.0) == -3      # 极端湿润
        assert classify_spi(1.99) == -2
        assert classify_spi(1.5) == -2
        assert classify_spi(1.0) == -1
        assert classify_spi(0.0) == 0       # 正常
        assert classify_spi(-1.0) == 0      # 边界属于"正常"
        assert classify_spi(-1.01) == 1
        assert classify_spi(-1.5) == 1
        assert classify_spi(-2.0) == 2
        assert classify_spi(-2.01) == 3

    def test_vci_threshold_boundaries(self):
        from utils.drought import classify_vci
        assert classify_vci(90) == -3
        assert classify_vci(70) == -2
        assert classify_vci(50) == -1
        assert classify_vci(40) == 0
        assert classify_vci(30) == 1
        assert classify_vci(20) == 2
        assert classify_vci(10) == 3
        assert classify_vci(9.99) == 4      # 极端干旱

    def test_array_matches_scalar(self):
        """向量化分类必须与逐元素标量分类结果一致"""
        from utils.drought import classify_spi
        values = np.array([2.5, 1.7, 1.2, 0.0, -1.2, -1.7, -2.5])
        assert np.array_equal(
            classify_spi(values),
            np.array([classify_spi(v) for v in values]),
        )

    def test_scalar_returns_python_int(self):
        from utils.drought import classify_spi
        assert isinstance(classify_spi(-2.5), (int, np.integer))

    def test_category_covers_full_range(self):
        """DROUGHT_CATEGORIES 的色标应覆盖所有可能等级"""
        from utils.drought import DROUGHT_CATEGORIES, classify_spi, classify_vci
        cats = set(DROUGHT_CATEGORIES)
        assert classify_spi(-99.0) in cats      # 最干旱
        assert classify_vci(-99.0) in cats
        assert classify_vci(999.0) in cats      # 最湿润


# ============================================================
# VCI — 植被状态指数
# ============================================================

class TestCalcVCI:
    def test_explicit_bounds_formula(self):
        """显式给定 min/max 时公式可精确验证: (NDVI-min)/(max-min)*100"""
        from utils.drought import calc_vci
        ndvi = np.array([0.2, 0.4, 0.6, 0.8])
        vci = calc_vci(ndvi, ndvi_min=0.2, ndvi_max=0.8)
        assert np.allclose(vci, [0.0, 100 / 3, 200 / 3, 100.0])

    def test_range_within_0_100(self):
        from utils.drought import calc_vci
        vci = calc_vci(make_ndvi_ts())
        valid = vci[np.isfinite(vci)]
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_monotonic_in_ndvi(self):
        from utils.drought import calc_vci
        ndvi = np.linspace(0.1, 0.8, 20)
        vci = calc_vci(ndvi, ndvi_min=0.1, ndvi_max=0.8)
        assert np.all(np.diff(vci) >= -1e-9)

    def test_constant_series_does_not_divide_by_zero(self):
        """全常量序列使 max-min=0, 不应产生 inf/nan"""
        from utils.drought import calc_vci
        vci = calc_vci(np.full(24, 0.35))
        assert np.all(np.isfinite(vci))

    def test_nan_input_preserved(self):
        from utils.drought import calc_vci
        ndvi = make_ndvi_ts(48)
        ndvi[5] = np.nan
        vci = calc_vci(ndvi)
        assert np.isnan(vci[5])


# ============================================================
# TCI — 温度状态指数
# ============================================================

class TestCalcTCI:
    def test_inverse_relation_to_lst(self):
        """TCI 与温度反向: 越热 → 指数越低 (胁迫越大)"""
        from utils.drought import calc_tci
        lst = np.array([280.0, 290.0, 300.0, 310.0])
        tci = calc_tci(lst, lst_min=280.0, lst_max=310.0)
        assert np.allclose(tci, [100.0, 200 / 3, 100 / 3, 0.0])
        assert np.all(np.diff(tci) < 0)

    def test_range_within_0_100(self):
        from utils.drought import calc_tci
        tci = calc_tci(make_lst_ts())
        valid = tci[np.isfinite(tci)]
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_auto_bounds_use_percentiles(self):
        """默认边界取 5/95 百分位, 因此极值会被裁剪到 0-100 内"""
        from utils.drought import calc_tci
        lst = np.concatenate([make_lst_ts(100), [240.0, 360.0]])  # 两个离群值
        tci = calc_tci(lst)
        assert np.nanmin(tci) >= 0.0
        assert np.nanmax(tci) <= 100.0


# ============================================================
# VHI — 植被健康指数
# ============================================================

class TestCalcVHI:
    def test_weight_one_equals_vci(self):
        from utils.drought import calc_vhi
        vci = np.array([20.0, 60.0])
        tci = np.array([80.0, 40.0])
        assert np.allclose(calc_vhi(vci, tci, weight=1.0), vci)

    def test_weight_zero_equals_tci(self):
        from utils.drought import calc_vhi
        vci = np.array([20.0, 60.0])
        tci = np.array([80.0, 40.0])
        assert np.allclose(calc_vhi(vci, tci, weight=0.0), tci)

    def test_weighted_formula(self):
        from utils.drought import calc_vhi
        vci, tci = np.array([40.0]), np.array([80.0])
        out = calc_vhi(vci, tci, weight=0.25)
        assert np.allclose(out, [0.25 * 40.0 + 0.75 * 80.0])

    def test_default_weight_is_half(self):
        from utils.drought import calc_vhi
        out = calc_vhi(np.array([0.0]), np.array([100.0]))
        assert np.allclose(out, [50.0])

    def test_clipped_to_0_100(self):
        from utils.drought import calc_vhi
        assert calc_vhi(np.array([-50.0]), np.array([0.0]), weight=1.0)[0] == 0.0
        assert calc_vhi(np.array([200.0]), np.array([0.0]), weight=1.0)[0] == 100.0


# ============================================================
# NDDI / NDWI
# ============================================================

class TestCalcNDDI:
    def test_formula(self):
        from utils.drought import calc_nddi
        ndvi = np.array([0.6, 0.2, 0.4])
        ndwi = np.array([0.2, 0.4, 0.4])
        assert np.allclose(calc_nddi(ndvi, ndwi), (ndvi - ndwi) / (ndvi + ndwi))

    def test_zero_denominator_guarded(self):
        """NDVI=NDWI=0 时分母为零, 应回落到 0 而非 inf/nan"""
        from utils.drought import calc_nddi
        out = calc_nddi(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(out[0])

    def test_positive_when_vegetation_dominates(self):
        from utils.drought import calc_nddi
        assert calc_nddi(np.array([0.7]), np.array([0.1]))[0] > 0


class TestCalcNDWIS2:
    def test_formula(self):
        from utils.drought import calc_ndwi_s2
        green = np.array([0.3, 0.1])
        nir = np.array([0.1, 0.3])
        assert np.allclose(calc_ndwi_s2(green, nir), [0.5, -0.5])

    def test_clipped_to_minus_one_one(self):
        from utils.drought import calc_ndwi_s2
        rng = np.random.default_rng(0)
        out = calc_ndwi_s2(rng.random(200), rng.random(200))
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_water_is_positive(self):
        """水体在绿-近红外上应为正值"""
        from utils.drought import calc_ndwi_s2
        assert calc_ndwi_s2(np.array([0.15]), np.array([0.05]))[0] > 0


# ============================================================
# NDVI 距平
# ============================================================

class TestCalcNDVIAnomaly:
    def test_standardized_against_own_baseline(self):
        """以自身为基线时, 距平应严格零均值、单位标准差"""
        from utils.drought import calc_ndvi_anomaly
        ts = make_ndvi_ts(120)
        result = calc_ndvi_anomaly(ts)
        anom = result["anomaly"]
        valid = anom[np.isfinite(anom)]
        assert abs(np.mean(valid)) < 1e-6
        assert abs(np.std(valid) - 1.0) < 1e-6

    def test_explicit_baseline_formula(self):
        from utils.drought import calc_ndvi_anomaly
        result = calc_ndvi_anomaly(np.array([0.4, 0.2]),
                                   baseline_mean=0.3, baseline_std=0.1)
        assert np.allclose(result["anomaly"], [1.0, -1.0])

    def test_reports_baseline(self):
        from utils.drought import calc_ndvi_anomaly
        result = calc_ndvi_anomaly(np.array([0.3, 0.5]))
        assert result["mean"] == pytest.approx(0.4)
        assert result["std"] > 0


# ============================================================
# SPI / PET
# ============================================================

class TestCalcSPI:
    def test_leading_nan_for_scale(self):
        """前 scale-1 个时相凑不满滑动窗口, 应为 NaN"""
        from utils.drought import calc_spi
        spi = calc_spi(make_precip(120), scale=3)
        assert np.all(np.isnan(spi[:2]))
        assert np.isfinite(spi[2:]).any()

    def test_short_series_returns_all_nan(self):
        from utils.drought import calc_spi
        spi = calc_spi(np.array([10.0, 20.0, 30.0]), scale=3)
        assert np.all(np.isnan(spi))

    def test_dry_period_yields_negative_spi(self):
        """降水骤降应产生负 SPI (干旱)"""
        from utils.drought import calc_spi
        precip = make_precip(150)
        precip[120:130] = 0.0
        spi = calc_spi(precip, scale=3)
        assert np.nanmean(spi[122:130]) < 0.0

    def test_wet_period_yields_positive_spi(self):
        from utils.drought import calc_spi
        precip = make_precip(150)
        precip[120:130] = 200.0
        spi = calc_spi(precip, scale=3)
        assert np.nanmean(spi[122:130]) > 0.0

    def test_roughly_standardized(self):
        """SPI 的定义即标准正态化 —— 长序列上均值接近 0、标准差接近 1

        容差放得较宽: 简化实现对整段序列只拟合一组 Gamma 参数,
        不像标准算法那样逐月拟合。
        """
        from utils.drought import calc_spi
        spi = calc_spi(make_precip(300), scale=3)
        valid = spi[np.isfinite(spi)]
        assert abs(np.mean(valid)) < 0.8
        assert 0.4 < np.std(valid) < 1.8

    @pytest.mark.parametrize("scale", [1, 3, 6, 12])
    def test_various_scales_produce_finite_output(self, scale):
        from utils.drought import calc_spi
        spi = calc_spi(make_precip(180), scale=scale)
        assert np.isfinite(spi[scale:]).any()


class TestCalcPETThornthwaite:
    def test_non_negative(self):
        from utils.drought import calc_pet_thornthwaite
        temp = np.tile(np.array([5.0, 10.0, 20.0, 25.0, 15.0, 5.0]), 4)
        pet = calc_pet_thornthwaite(temp, lat=40.0)
        assert np.all(pet[np.isfinite(pet)] >= 0.0)

    def test_increases_with_temperature(self):
        from utils.drought import calc_pet_thornthwaite
        cold = calc_pet_thornthwaite(np.full(24, 5.0), lat=40.0)
        warm = calc_pet_thornthwaite(np.full(24, 25.0), lat=40.0)
        assert np.nanmean(warm) > np.nanmean(cold)

    def test_latitude_affects_day_length_correction(self):
        """Thornthwaite 含日长订正项, 纬度应影响结果"""
        from utils.drought import calc_pet_thornthwaite
        temp = np.tile(np.array([5.0, 10.0, 20.0, 25.0, 15.0, 5.0]), 4)
        low = calc_pet_thornthwaite(temp, lat=10.0)
        high = calc_pet_thornthwaite(temp, lat=60.0)
        assert not np.allclose(low, high, equal_nan=True)


# ============================================================
# 逐像元版本
# ============================================================

class TestPixelwise:
    def test_spi_pixelwise_keys_and_shapes(self):
        from utils.drought import calc_spi_pixelwise
        stack = np.random.default_rng(0).uniform(5, 60, (4, 4, 40))
        out = calc_spi_pixelwise(stack, scale=3)
        assert set(out) >= {"spi", "spi_latest", "drought_category"}
        assert out["spi"].shape == (4, 4, 40)
        assert out["spi_latest"].shape == (4, 4)
        assert out["drought_category"].shape == (4, 4)

    def test_vci_pixelwise_matches_1d_version(self):
        """逐像元 VCI 应与把同一序列喂给一维版本一致"""
        from utils.drought import calc_vci_pixelwise, calc_vci
        stack = np.random.default_rng(1).uniform(0.15, 0.75, (3, 3, 36))
        out = calc_vci_pixelwise(stack)
        ts = stack[1, 2, :]
        expected = calc_vci(ts, ndvi_min=np.nanmin(ts), ndvi_max=np.nanmax(ts))
        assert np.allclose(out["vci"][1, 2, :], expected, atol=1e-4)

    def test_vci_pixelwise_reports_bounds(self):
        from utils.drought import calc_vci_pixelwise
        stack = np.random.default_rng(2).uniform(0.1, 0.8, (3, 3, 24))
        out = calc_vci_pixelwise(stack)
        assert out["ndvi_min"].shape == (3, 3)
        assert np.all(out["ndvi_max"] >= out["ndvi_min"])

    def test_tci_pixelwise_range(self):
        from utils.drought import calc_tci_pixelwise
        stack = np.random.default_rng(3).uniform(280, 320, (4, 4, 36))
        out = calc_tci_pixelwise(stack)
        assert out["tci"].shape == (4, 4, 36)
        assert np.nanmin(out["tci"]) >= 0.0
        assert np.nanmax(out["tci"]) <= 100.0

    def test_vhi_pixelwise_weighted_formula(self):
        from utils.drought import calc_vhi_pixelwise
        vci = np.random.default_rng(4).uniform(0, 100, (3, 3, 12))
        tci = np.random.default_rng(5).uniform(0, 100, (3, 3, 12))
        out = calc_vhi_pixelwise(vci, tci, weight=0.5)
        assert np.allclose(out["vhi"], np.clip(0.5 * vci + 0.5 * tci, 0, 100), atol=1e-3)

    def test_ndvi_anomaly_pixelwise(self):
        from utils.drought import calc_ndvi_anomaly_pixelwise
        stack = np.random.default_rng(6).uniform(0.2, 0.6, (3, 3, 30))
        out = calc_ndvi_anomaly_pixelwise(stack)
        assert out["anomaly"].shape == (3, 3, 30)
        assert out["anomaly_latest"].shape == (3, 3)


# ============================================================
# TVDI
# ============================================================

class TestCalcTVDI:
    def test_returns_tuple_with_info(self):
        from utils.drought import calc_tvdi
        rng = np.random.default_rng(7)
        ndvi = rng.uniform(0.1, 0.9, 400)
        lst = 300.0 - 30.0 * ndvi + rng.normal(0, 2, 400)   # 负相关特征空间
        result = calc_tvdi(ndvi, lst)
        assert isinstance(result, tuple) and len(result) == 2
        tvdi, info = result
        assert isinstance(info, dict)

    def test_insufficient_pixels_reports_error(self):
        """有效像元太少时应返回 error 说明, 而非崩溃"""
        from utils.drought import calc_tvdi
        tvdi, info = calc_tvdi(np.array([0.5]), np.array([300.0]))
        assert "error" in info

    def test_tvdi_range_when_computable(self):
        from utils.drought import calc_tvdi
        rng = np.random.default_rng(8)
        ndvi = rng.uniform(0.1, 0.9, 2000)
        lst = 320.0 - 40.0 * ndvi + rng.normal(0, 3, 2000)
        tvdi, info = calc_tvdi(ndvi, lst)
        if "error" not in info:
            valid = tvdi[np.isfinite(tvdi)]
            # TVDI 理论值域 0-1 (可因拟合略越界)
            assert valid.min() > -0.5
            assert valid.max() < 1.5


# ============================================================
# 综合指数与统计
# ============================================================

class TestCompositeDroughtIndex:
    def test_default_equal_weights(self):
        from utils.drought import calc_composite_drought_index
        cdi = calc_composite_drought_index({
            "vci": np.array([60.0]),
            "tci": np.array([40.0]),
        })
        assert np.allclose(cdi, [50.0])

    def test_explicit_weights(self):
        from utils.drought import calc_composite_drought_index
        cdi = calc_composite_drought_index(
            {"vci": np.array([100.0]), "tci": np.array([0.0])},
            weights={"vci": 0.25, "tci": 0.75},
        )
        assert np.allclose(cdi, [25.0])

    def test_empty_raises(self):
        from utils.drought import calc_composite_drought_index
        with pytest.raises(ValueError):
            calc_composite_drought_index({})

    def test_output_range(self):
        from utils.drought import calc_composite_drought_index
        rng = np.random.default_rng(9)
        cdi = calc_composite_drought_index({
            "vci": rng.uniform(0, 100, 50),
            "tci": rng.uniform(0, 100, 50),
        })
        assert cdi.min() >= 0.0 and cdi.max() <= 100.0


class TestComputeDroughtStats:
    def test_area_and_ratio(self):
        """面积 = 像元数 × 像元面积; 10m 像元 → 100 m²"""
        from utils.drought import compute_drought_stats
        arr = np.zeros((10, 10), dtype=int)
        arr[:3, :] = 3                     # 30 个极端干旱像元
        stats = compute_drought_stats(arr, pixel_size_m=10.0)
        severe = next(s for s in stats if s["category"] == 3)
        assert severe["pixel_count"] == 30
        assert severe["ratio"] == pytest.approx(0.3)
        assert severe["area_km2"] == pytest.approx(30 * 100 / 1e6)

    def test_covers_all_categories(self):
        from utils.drought import compute_drought_stats
        arr = np.array([[-3, -2, -1], [0, 1, 2], [3, 4, 0]])
        stats = compute_drought_stats(arr)
        assert len(stats) == 8              # -3..4 共 8 级
        assert sum(s["pixel_count"] for s in stats) == arr.size

    def test_ratio_sums_to_one(self):
        from utils.drought import compute_drought_stats
        rng = np.random.default_rng(10)
        arr = rng.integers(-3, 5, (20, 20))
        stats = compute_drought_stats(arr)
        assert sum(s["ratio"] for s in stats) == pytest.approx(1.0, abs=1e-3)

    def test_names_from_category_table(self):
        from utils.drought import compute_drought_stats, DROUGHT_CATEGORIES
        stats = compute_drought_stats(np.array([[3]]))
        severe = next(s for s in stats if s["category"] == 3)
        assert severe["name"] == DROUGHT_CATEGORIES[3]["name"]


class TestComputeDroughtIndexStats:
    def test_summary_fields(self):
        from utils.drought import compute_drought_index_stats
        arr = np.random.default_rng(11).uniform(-3, 3, (20, 20))
        out = compute_drought_index_stats(arr, "SPI3")
        assert out["name"] == "SPI3"
        assert out["valid_pixels"] == 400
        for key in ("mean", "median", "std", "min", "max", "p5", "p95"):
            assert key in out

    def test_percentile_ordering(self):
        from utils.drought import compute_drought_index_stats
        arr = np.random.default_rng(12).uniform(-3, 3, (30, 30))
        out = compute_drought_index_stats(arr, "VCI")
        assert out["p5"] <= out["p25"] <= out["p75"] <= out["p95"]

    def test_all_nan_reports_error(self):
        from utils.drought import compute_drought_index_stats
        out = compute_drought_index_stats(np.full((5, 5), np.nan), "SPI")
        assert "error" in out


# ============================================================
# 一站式入口
# ============================================================

class TestAnalyzeDroughtRemote:
    def test_full_pipeline_runs(self):
        """完整分析链应跑通并返回结构化结果"""
        from utils.drought import analyze_drought_remote
        n = 120
        result = analyze_drought_remote(
            ndvi_ts=make_ndvi_ts(n),
            ndwi_ts=np.random.default_rng(13).uniform(-0.3, 0.2, n),
            lst_ts=make_lst_ts(n),
            precip_ts=make_precip(n),
            temp_ts=np.tile(np.array([2.0, 6.0, 12.0, 20.0, 26.0, 24.0,
                                      18.0, 12.0, 5.0, -2.0, -8.0, -4.0]), 10),
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert "indices" in result

    def test_handles_short_series_gracefully(self):
        """序列过短时不应抛异常"""
        from utils.drought import analyze_drought_remote
        result = analyze_drought_remote(
            ndvi_ts=np.array([0.3, 0.35]),
            ndwi_ts=np.array([-0.1, -0.2]),
            lst_ts=np.array([295.0, 296.0]),
            precip_ts=np.array([10.0, 12.0]),
            temp_ts=np.array([5.0, 6.0]),
        )
        assert isinstance(result, dict)
