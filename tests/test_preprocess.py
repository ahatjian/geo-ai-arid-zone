"""影像预处理模块测试 — 云掩膜 / 重采样 / 归一化 / S-G 平滑"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_bands(h=40, w=40, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.6, (6, h, w))


def make_scl(h=40, w=40, seed=2):
    """生成含云 (8/9) 和晴空 (4/5/6) 的 SCL 层"""
    rng = np.random.default_rng(seed)
    scl = rng.integers(4, 7, (h, w)).astype(np.int16)
    scl[5:15, 5:15] = 8   # 中云块
    scl[15:20, 15:20] = 9  # 高云块
    scl[0:3, 0:3] = 3      # 云影块
    return scl


class TestS2CloudMask:
    def test_mask_returns_same_shape(self):
        from utils.preprocess import apply_s2_cloud_mask
        bands = make_bands()
        scl = make_scl()
        masked, cloud_mask = apply_s2_cloud_mask(bands, scl)
        assert masked.shape == bands.shape
        assert cloud_mask.shape == (40, 40)

    def test_cloud_pixels_are_nan(self):
        from utils.preprocess import apply_s2_cloud_mask
        bands = make_bands()
        scl = make_scl()
        masked, cloud_mask = apply_s2_cloud_mask(bands, scl)
        # 云块区域应被掩膜
        assert cloud_mask[10, 10]   # SCL=8 云
        assert cloud_mask[17, 17]   # SCL=9 云
        assert np.isnan(masked[0, 10, 10])  # 该像元所有波段为 NaN

    def test_clear_pixels_unchanged(self):
        from utils.preprocess import apply_s2_cloud_mask
        bands = make_bands()
        scl = make_scl()
        masked, cloud_mask = apply_s2_cloud_mask(bands, scl)
        # 晴空区 (SCL=4~6) 不被掩膜
        assert not cloud_mask[30, 30]
        assert masked[0, 30, 30] == bands[0, 30, 30]

    def test_shadow_option(self):
        from utils.preprocess import apply_s2_cloud_mask
        bands = make_bands()
        scl = make_scl()
        _, cm_with_shadow = apply_s2_cloud_mask(bands, scl, mask_shadow=True)
        _, cm_no_shadow = apply_s2_cloud_mask(bands, scl, mask_shadow=False)
        # 关闭云影掩膜后, 云影块 (0-3) 不再被掩
        assert cm_with_shadow[1, 1]
        assert not cm_no_shadow[1, 1]

    def test_shape_mismatch_raises(self):
        from utils.preprocess import apply_s2_cloud_mask
        with pytest.raises(ValueError):
            apply_s2_cloud_mask(make_bands(), make_scl(30, 30))


class TestLandsatCloudMask:
    def test_qa_bit3_cloud(self):
        from utils.preprocess import apply_landsat_cloud_mask
        bands = make_bands(20, 20, seed=3)
        qa = np.zeros((20, 20), dtype=np.uint16)
        qa[5, 5] = 1 << 3  # bit3 = 云
        qa[10, 10] = 1 << 4  # bit4 = 云影
        masked, cloud_mask = apply_landsat_cloud_mask(bands, qa)
        assert cloud_mask[5, 5]
        assert cloud_mask[10, 10]
        assert not cloud_mask[0, 0]
        assert np.isnan(masked[0, 5, 5])

    def test_mask_clouds_dispatch(self):
        from utils.preprocess import mask_clouds
        bands = make_bands(20, 20, seed=4)
        scl = make_scl(20, 20)
        masked_s2, _ = mask_clouds(bands, scl, satellite="Sentinel-2 L2A")
        assert np.isnan(masked_s2[0, 10, 10])
        qa = np.zeros((20, 20), dtype=np.uint16)
        qa[10, 10] = 1 << 3
        masked_ls, _ = mask_clouds(bands, qa, satellite="Landsat-8")
        assert np.isnan(masked_ls[0, 10, 10])


class TestResample:
    def test_same_shape_returns_copy(self):
        from utils.preprocess import resample_array
        arr = np.random.default_rng(0).random((10, 10))
        out = resample_array(arr, (10, 10))
        assert out.shape == (10, 10)
        assert np.allclose(out, arr)

    def test_upscale_2d(self):
        from utils.preprocess import resample_array
        arr = np.random.default_rng(0).random((10, 10))
        out = resample_array(arr, (20, 20), method="nearest")
        assert out.shape == (20, 20)

    def test_downscale_multiband(self):
        from utils.preprocess import resample_array
        arr = np.random.default_rng(0).random((3, 20, 20))
        out = resample_array(arr, (10, 10), method="bilinear")
        assert out.shape == (3, 10, 10)

    def test_upscale_different_methods(self):
        from utils.preprocess import resample_array
        arr = np.random.default_rng(0).random((8, 8))
        for method in ["nearest", "bilinear", "cubic"]:
            out = resample_array(arr, (16, 16), method=method)
            assert out.shape == (16, 16)


class TestNormalize:
    def test_uint8_range(self):
        from utils.preprocess import normalize_to_uint8
        arr = np.random.default_rng(0).uniform(-0.2, 0.8, (20, 20))
        out = normalize_to_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_nan_handling(self):
        from utils.preprocess import normalize_to_uint8
        arr = np.full((10, 10), 0.5)
        arr[2, 2] = np.nan
        out = normalize_to_uint8(arr)
        assert out[2, 2] == 0  # NaN → 0

    def test_cloud_cover_fraction(self):
        from utils.preprocess import cloud_cover_fraction
        cm = np.zeros((100, 100), dtype=bool)
        cm[:50, :] = True
        assert cloud_cover_fraction(cm) == pytest.approx(0.5)

    def test_mask_stats(self):
        from utils.preprocess import mask_stats
        masked = np.full((10, 10), 1.0)
        masked[0, 0] = np.nan
        stats = mask_stats(masked)
        assert stats["valid_ratio"] == pytest.approx(0.99)
        assert stats["valid_pixels"] == 99


class TestSavgolSmooth:
    def test_smooth_preserves_shape(self):
        from utils.trend import savgol_smooth
        rng = np.random.default_rng(0)
        values = 0.4 + 0.05 * np.sin(np.arange(30) / 4.0) + rng.normal(0, 0.01, 30)
        out = savgol_smooth(values, window=7, polyorder=2)
        assert out.shape == (30,)
        assert np.all(np.isfinite(out))

    def test_smooth_reduces_noise(self):
        from utils.trend import savgol_smooth
        rng = np.random.default_rng(1)
        signal = 0.3 + 0.1 * np.sin(np.arange(40) / 3.0)
        noisy = signal + rng.normal(0, 0.05, 40)
        smoothed = savgol_smooth(noisy, window=9, polyorder=3)
        # 平滑后噪声应显著降低
        noise_raw = np.std(noisy - signal)
        noise_sm = np.std(smoothed - signal)
        assert noise_sm < noise_raw * 0.8

    def test_nan_filled_by_interpolation(self):
        from utils.trend import savgol_smooth
        values = np.array([0.3, 0.4, np.nan, 0.6, 0.7, 0.8, 0.9])
        out = savgol_smooth(values, window=5, polyorder=2)
        assert np.all(np.isfinite(out))

    def test_short_series_returns_copy(self):
        from utils.trend import savgol_smooth
        values = np.array([0.3, 0.4, 0.5])
        out = savgol_smooth(values)
        assert np.array_equal(out, values)

    def test_smooth_ndvi_stack(self):
        from utils.trend import smooth_ndvi_stack
        rng = np.random.default_rng(2)
        stack = rng.uniform(0.2, 0.5, (8, 8, 24))
        out = smooth_ndvi_stack(stack, window=7, polyorder=2)
        assert out.shape == (8, 8, 24)

    def test_smooth_stack_too_short(self):
        from utils.trend import smooth_ndvi_stack
        stack = np.random.default_rng(3).random((5, 5, 3))
        out = smooth_ndvi_stack(stack)
        assert out.shape == (5, 5, 3)  # 不崩溃, 原样返回


class TestPdfReport:
    def test_generate_pdf_bytes(self):
        from utils.pdf_report import generate_report_pdf
        pdf = generate_report_pdf(
            title="测试报告",
            meta_lines=["研究区域: 塔里木盆地", "分析日期: 2025-06-01"],
            sections=[
                {"heading": "植被分析", "content": "NDVI 均值 0.35，植被覆盖良好。"},
                {"heading": "指标表", "table": (["指标", "数值"], [("NDVI", "0.35"), ("覆盖率", "42%")])},
            ],
            footer="测试页脚",
        )
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_pdf_without_table(self):
        from utils.pdf_report import generate_report_pdf
        pdf = generate_report_pdf("标题", ["元信息"], [{"heading": "章节", "content": "正文内容"}])
        assert pdf is not None and pdf[:4] == b"%PDF"

    def test_pdf_empty_sections(self):
        from utils.pdf_report import generate_report_pdf
        pdf = generate_report_pdf("空报告", ["无章节"], [])
        assert pdf is not None and pdf[:4] == b"%PDF"


class TestPdfFontFallback:
    """PDF 中文字体的三级降级。

    这一段是 CI 首次运行抓出来的: 本地 Windows 有微软雅黑, 测试全绿;
    GitHub runner (以及任何 Linux 服务器) 没装中文字体, 字体查找返回
    None, PDF 直接返回 None —— 线上"PDF 科研报告"功能静默失效。
    """

    def test_survives_without_any_system_font(self, monkeypatch):
        """模拟 Linux/Docker: 系统无中文字体时仍须产出 PDF"""
        import utils.pdf_report as pr
        monkeypatch.setattr(pr, "_find_chinese_font", lambda: None)

        pdf = pr.generate_report_pdf(
            "塔里木盆地植被分析报告",
            ["研究区: 塔里木盆地"],
            [{"heading": "植被状况", "content": "NDVI 均值 0.32。"}],
        )
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_cid_font_used_as_fallback(self, monkeypatch):
        """无系统字体时应回落到 reportlab 内置 CID 字体"""
        import utils.pdf_report as pr
        monkeypatch.setattr(pr, "_find_chinese_font", lambda: None)
        assert pr._register_cjk_font() == "STSong-Light"

    def test_system_font_preferred_when_present(self):
        """有系统字体时优先用系统字体 (字形覆盖更完整)"""
        import utils.pdf_report as pr
        if not pr._find_chinese_font():
            pytest.skip("本机未安装系统中文字体")
        assert pr._register_cjk_font() == "CJKFont"

    def test_registration_failure_falls_back_not_raises(self, monkeypatch):
        """系统字体注册抛异常时也要降级, 而不是让 PDF 生成失败"""
        import utils.pdf_report as pr
        monkeypatch.setattr(pr, "_find_chinese_font", lambda: "/nonexistent/font.ttf")
        assert pr._register_cjk_font() == "STSong-Light"


# ============================================================
# 云掩膜资产配置 — 页面据此自动获取 SCL / QA_PIXEL
# ============================================================

class TestCloudAssetConfig:
    """第 23 页的云掩膜靠 COLLECTIONS[*]["cloud_asset"] 自动下载图层。

    缺这个字段不会报错, 只会静默跳过掩膜 —— 所以用测试把它钉死。
    """

    def test_every_collection_declares_cloud_asset(self):
        from config import COLLECTIONS
        for name, cfg in COLLECTIONS.items():
            assert "cloud_asset" in cfg, f"{name} 未声明 cloud_asset"
            assert cfg["cloud_asset"], f"{name} 的 cloud_asset 为空"

    def test_sentinel2_uses_scl(self):
        from config import COLLECTIONS
        assert COLLECTIONS["Sentinel-2 L2A"]["cloud_asset"] == "SCL"

    @pytest.mark.parametrize("sat", ["Landsat-8", "Landsat-9", "Landsat-7", "Landsat-4-5"])
    def test_landsat_uses_qa_pixel(self, sat):
        from config import COLLECTIONS
        assert COLLECTIONS[sat]["cloud_asset"] == "QA_PIXEL"

    def test_cloud_asset_not_mixed_into_bands(self):
        """cloud_asset 必须是独立字段。

        若混进 bands, 所有以 len(bands) 判断波段数的代码 (下载、校验、
        页面提示) 都会把掩膜层当成反射率波段。
        """
        from config import COLLECTIONS
        for name, cfg in COLLECTIONS.items():
            assert cfg["cloud_asset"] not in cfg["bands"].values(), name

    def test_band_count_still_six(self):
        from config import COLLECTIONS
        for name, cfg in COLLECTIONS.items():
            assert len(cfg["bands"]) == 6, name

    def test_cloud_asset_resolvable_via_download_asset(self):
        """资产名要能被 pc_data.download_asset 直接使用 (即 STAC 资产键)"""
        from config import COLLECTIONS
        known = {"SCL", "QA_PIXEL"}
        for name, cfg in COLLECTIONS.items():
            assert cfg["cloud_asset"] in known, name


class TestMaskCloudsDispatch:
    """mask_clouds 按卫星名分派到 S2 / Landsat 两套位掩码逻辑"""

    def test_sentinel_name_dispatches_to_scl(self):
        from utils.preprocess import mask_clouds
        bands = make_bands()
        scl = make_scl()
        masked, cm = mask_clouds(bands, scl, satellite="Sentinel-2 L2A")
        # SCL=8/9 是云, 应被掩膜
        assert cm[8, 8]     # 中云块
        assert cm[17, 17]   # 高云块
        assert np.isnan(masked[:, 8, 8]).all()

    @pytest.mark.parametrize("sat", ["Landsat-8", "Landsat-9"])
    def test_landsat_names_dispatch_to_qa_pixel(self, sat):
        from utils.preprocess import mask_clouds
        bands = make_bands()
        qa = np.zeros((40, 40), dtype=np.uint16)
        qa[10:20, 10:20] |= 1 << 3   # bit 3 = 云
        masked, cm = mask_clouds(bands, qa, satellite=sat)
        assert cm[15, 15]
        assert not cm[0, 0]

    def test_landsat_shadow_option_respected(self):
        from utils.preprocess import mask_clouds
        bands = make_bands()
        qa = np.zeros((40, 40), dtype=np.uint16)
        qa[5:10, 5:10] |= 1 << 4   # bit 4 = 云影, 非云

        _, with_shadow = mask_clouds(bands, qa, satellite="Landsat-8", mask_shadow=True)
        _, no_shadow = mask_clouds(bands, qa, satellite="Landsat-8", mask_shadow=False)
        assert with_shadow[7, 7]
        assert not no_shadow[7, 7]

    def test_dimension_mismatch_raises(self):
        """图层尺寸不符要显式报错 —— 页面据此提示用户而非静默算错"""
        from utils.preprocess import mask_clouds
        with pytest.raises(ValueError):
            mask_clouds(make_bands(40, 40), make_scl(30, 30), satellite="Sentinel-2 L2A")
