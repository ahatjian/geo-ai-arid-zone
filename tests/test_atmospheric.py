"""辐射定标与大气校正测试 — DOS 暗像元法/定标/归一化"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_observed_image(seed=0, h=80, w=80):
    """生成含真实暗像元 + 大气路径辐射的模拟影像"""
    rng = np.random.default_rng(seed)
    true_surface = rng.uniform(0.2, 0.4, (6, h, w))
    true_surface[:, 60:80, 60:80] = 0.01  # 深水暗像元区
    atmo = np.array([0.05, 0.04, 0.03, 0.02, 0.015, 0.01])[:, None, None]
    return true_surface + atmo, atmo


class TestDOSCorrection:
    def test_shape_preserved(self):
        from utils.atmospheric import dos_correction
        observed, _ = make_observed_image()
        result = dos_correction(observed)
        assert result["corrected"].shape == observed.shape

    def test_dark_value_estimate_accurate(self):
        """DOS 估计的大气路径辐射应接近真实值 (误差 < 0.02)"""
        from utils.atmospheric import dos_correction
        observed, true_atmo = make_observed_image()
        result = dos_correction(observed, dark_percentile=1.0, stretch_after=False)
        errs = np.abs(result["dark_values"] - true_atmo.flatten())
        assert np.mean(errs) < 0.02

    def test_dark_pixel_region_recovered(self):
        """暗像元区校正后应接近地表真实值 (≈0)"""
        from utils.atmospheric import dos_correction
        observed, _ = make_observed_image()
        result = dos_correction(observed, dark_percentile=1.0, stretch_after=False)
        water = result["corrected"][:, 65:75, 65:75]
        assert np.nanmean(water) < 0.05

    def test_stretch_mode(self):
        from utils.atmospheric import dos_correction
        observed, _ = make_observed_image()
        result = dos_correction(observed, stretch_after=True)
        assert result["corrected"].min() >= 0
        assert result["corrected"].max() <= 1.0

    def test_estimate_dark_pixel(self):
        from utils.atmospheric import estimate_dark_pixel
        observed, _ = make_observed_image()
        dark, ratio = estimate_dark_pixel(observed[0])
        assert dark > 0
        assert ratio < 0.1  # 暗像元占比合理 (含模拟的深水区)

    def test_invalid_input(self):
        from utils.atmospheric import dos_correction
        with pytest.raises(ValueError):
            dos_correction(np.zeros((5, 5)))

    def test_quality_report(self):
        from utils.atmospheric import dos_correction, dos_quality_report
        observed, _ = make_observed_image()
        result = dos_correction(observed, stretch_after=False)
        report = dos_quality_report(observed, result["corrected"])
        assert len(report) == 6
        assert "大气路径辐射" in report[0]
        assert report[0]["校正前均值"] > report[0]["校正后均值"]


class TestRadiometricCalibration:
    def test_toa_reflectance_range(self):
        from utils.atmospheric import toa_reflectance
        dn = np.full((6, 20, 20), 5000.0)
        toa = toa_reflectance(dn, [2e-5] * 6, [-0.1] * 6, solar_zenith_deg=30)
        # ρ = (2e-5×5000 - 0.1)/cos(30°) = 0.0
        assert abs(np.nanmean(toa)) < 0.05

    def test_solar_zenith_effect(self):
        """天顶角越大 → 反射率越大 (斜射路径更长)"""
        from utils.atmospheric import toa_reflectance
        dn = np.full((1, 10, 10), 10000.0)
        r0 = toa_reflectance(dn, [2e-5], [-0.1], 30)
        r60 = toa_reflectance(dn, [2e-5], [-0.1], 60)
        assert np.nanmean(r60) > np.nanmean(r0)

    def test_invalid_zenith(self):
        from utils.atmospheric import toa_reflectance
        with pytest.raises(ValueError):
            toa_reflectance(np.zeros((1, 5, 5)), [1], [0], 90)
    def test_radiometric_calibration(self):
        from utils.atmospheric import radiometric_calibration
        dn = np.full((3, 10, 10), 1000.0)
        refl = radiometric_calibration(dn, [0.1] * 3, [-10] * 3,
                                       solar_zenith_deg=45, esun=[2000] * 3)
        assert np.all(np.isfinite(refl))
        assert refl.max() <= 1.5

    def test_length_mismatch(self):
        from utils.atmospheric import radiometric_calibration
        with pytest.raises(ValueError):
            radiometric_calibration(np.zeros((3, 5, 5)), [0.1], [0], 30, [2000])


class TestRelativeNormalization:
    def test_linear_aligns_means(self):
        """线性归一化后目标均值应接近参考均值"""
        from utils.atmospheric import relative_normalization
        rng = np.random.default_rng(1)
        target = rng.uniform(0.1, 0.4, (3, 30, 30))
        reference = target * 0.8 + 0.05
        normalized = relative_normalization(target, reference, "linear")
        assert abs(np.nanmean(normalized) - np.nanmean(reference)) < 0.02

    def test_hist_match(self):
        from utils.atmospheric import relative_normalization
        rng = np.random.default_rng(2)
        target = rng.uniform(0.1, 0.4, (1, 30, 30))
        reference = rng.uniform(0.3, 0.6, (1, 30, 30))
        normalized = relative_normalization(target, reference, "hist_match")
        assert normalized.shape == target.shape
        assert np.all(np.isfinite(normalized))

    def test_shape_mismatch(self):
        from utils.atmospheric import relative_normalization
        with pytest.raises(ValueError):
            relative_normalization(np.zeros((3, 10, 10)), np.zeros((3, 20, 20)))


class TestPreprocessPipeline:
    def test_dn_autoscale(self):
        """DN 值 (0-10000) 应自动缩放 + DOS 校正"""
        from utils.atmospheric import preprocess_pipeline
        rng = np.random.default_rng(3)
        dn = (rng.uniform(0.1, 0.4, (6, 40, 40)) * 10000).astype(np.float64)
        dn[:, 30:40, 30:40] = 100  # 暗像元
        result = preprocess_pipeline(dn)
        assert result["processed"].shape == dn.shape
        assert result["processed"].max() <= 1.0
        assert "DOS" in " ".join(result["steps"])

    def test_pipeline_no_dos(self):
        from utils.atmospheric import preprocess_pipeline
        bands = np.random.default_rng(4).uniform(0.1, 0.4, (6, 20, 20))
        result = preprocess_pipeline(bands, do_dos=False)
        assert result["dark_values"] is None
        assert result["processed"].shape == bands.shape
