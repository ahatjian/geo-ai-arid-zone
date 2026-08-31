"""图像处理 + 非监督分类 + STL 分解测试 — 专业平台能力验证"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_related_bands(h=40, w=40, seed=0):
    """生成高度相关的多波段 (模拟真实遥感波段相关性)"""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.4, (h, w))
    return np.stack([
        base + rng.normal(0, 0.02, (h, w)),
        base * 0.9 + rng.normal(0, 0.02, (h, w)),
        base * 0.8 + rng.normal(0, 0.02, (h, w)),
        base * 1.5 + rng.normal(0, 0.02, (h, w)),
        base * 0.6 + rng.normal(0, 0.02, (h, w)),
        base * 0.5 + rng.normal(0, 0.02, (h, w)),
    ])


class TestPCA:
    """主成分分析"""

    def test_pca_shape_and_info_concentration(self):
        from utils.image_processing import pca_transform
        bands = make_related_bands()
        pca = pca_transform(bands)
        assert pca["pca"].shape == (6, 40, 40)
        # 相关波段 → 前3分量应集中 >90% 信息
        assert pca["explained_variance_ratio"][:3].sum() > 0.9

    def test_pca_n_components(self):
        from utils.image_processing import pca_transform
        pca = pca_transform(make_related_bands(), n_components=3)
        assert pca["pca"].shape == (3, 40, 40)
        assert len(pca["explained_variance_ratio"]) == 3

    def test_pca_rgb_composite(self):
        from utils.image_processing import pca_transform, pca_rgb_composite
        pca = pca_transform(make_related_bands())
        rgb = pca_rgb_composite(pca)
        assert rgb.shape == (40, 40, 3)
        assert rgb.dtype == np.uint8

    def test_pca_invalid_input(self):
        from utils.image_processing import pca_transform
        with pytest.raises(ValueError):
            pca_transform(np.zeros((5, 5)))


class TestSpatialFilter:
    """空间滤波"""

    @pytest.mark.parametrize("ft", ["mean", "median", "gaussian", "sharpen", "edge"])
    def test_all_filters(self, ft):
        from utils.image_processing import spatial_filter
        arr = make_related_bands()[0]
        out = spatial_filter(arr, ft, 3)
        assert out.shape == arr.shape
        assert np.all(np.isfinite(out))

    def test_multiband_filter(self):
        from utils.image_processing import spatial_filter
        bands = make_related_bands()
        out = spatial_filter(bands, "median", 3)
        assert out.shape == bands.shape

    def test_invalid_filter(self):
        from utils.image_processing import spatial_filter
        with pytest.raises(ValueError):
            spatial_filter(np.zeros((5, 5)), "nonexistent")


class TestContrastEnhance:
    """对比度增强"""

    @pytest.mark.parametrize("method", ["percentile", "hist_eq", "gamma", "clahe"])
    def test_all_methods(self, method):
        from utils.image_processing import contrast_enhance
        arr = make_related_bands()[0]
        out = contrast_enhance(arr, method)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_gamma_parameter(self):
        from utils.image_processing import contrast_enhance
        arr = make_related_bands()[0]
        out = contrast_enhance(arr, "gamma", gamma=0.5)
        assert out.dtype == np.uint8

    def test_nan_handling(self):
        from utils.image_processing import contrast_enhance
        arr = np.full((10, 10), 0.5)
        arr[2, 2] = np.nan
        out = contrast_enhance(arr, "percentile")
        assert out[2, 2] == 0  # NaN → 0


class TestIHSFusion:
    def test_fusion_shape(self):
        from utils.image_processing import ihs_fusion
        rng = np.random.default_rng(1)
        rgb = rng.uniform(0.1, 0.5, (20, 20, 3))
        pan = rng.uniform(0.1, 0.5, (20, 20))
        fused = ihs_fusion(rgb, pan)
        assert fused.shape == (20, 20, 3)
        assert fused.dtype == np.uint8

    def test_fusion_uint8_input(self):
        from utils.image_processing import ihs_fusion
        rng = np.random.default_rng(2)
        rgb = (rng.uniform(0.1, 0.5, (20, 20, 3)) * 255).astype(np.uint8)
        pan = (rng.uniform(0.1, 0.5, (20, 20)) * 255).astype(np.uint8)
        fused = ihs_fusion(rgb, pan)
        assert fused.shape == (20, 20, 3)


class TestUnsupervised:
    """KMeans 非监督分类"""

    def _make_3class_image(self):
        rng = np.random.default_rng(0)
        H, W = 60, 60
        bands = rng.uniform(0.1, 0.3, (6, H, W))
        bands[3, 10:30, 10:30] = 0.45   # NIR 高 → 植被
        bands[0:3, 30:50, 30:50] = 0.05  # 低反射 → 水体
        bands[4, 50:60, 50:60] = 0.5     # SWIR 高 → 裸地
        return bands

    def test_kmeans_classify(self):
        from utils.unsupervised import kmeans_classify
        bands = self._make_3class_image()
        result = kmeans_classify(bands, n_classes=3)
        assert result["classification"].shape == (60, 60)
        assert result["sizes"].sum() == 60 * 60
        assert "KMeans" in result["method"]

    def test_kmeans_separates_regions(self):
        """三类模拟地物应被聚到不同类"""
        from utils.unsupervised import kmeans_classify
        bands = self._make_3class_image()
        result = kmeans_classify(bands, n_classes=3)
        cls = result["classification"]
        veg = set(np.unique(cls[10:30, 10:30]).tolist())
        water = set(np.unique(cls[30:50, 30:50]).tolist())
        bare = set(np.unique(cls[50:60, 50:60]).tolist())
        # 三个区域应归属不同类别
        assert veg.isdisjoint(water)
        assert veg.isdisjoint(bare)
        assert water.isdisjoint(bare)

    def test_kmeans_feature_stack(self):
        from utils.unsupervised import kmeans_feature_stack
        bands = self._make_3class_image()
        stack = kmeans_feature_stack(bands)
        assert stack.shape[0] == 9  # 6 波段 + 3 指数
        stack2 = kmeans_feature_stack(bands, add_indices=False)
        assert stack2.shape[0] == 6

    def test_auto_describe_classes(self):
        from utils.unsupervised import auto_describe_classes
        # 高 NIR → 植被, 低反射 → 水体, 高 SWIR → 裸地
        centers = np.array([
            [0.1, 0.1, 0.1, 0.45, 0.1, 0.1],  # 植被
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 水体
            [0.2, 0.2, 0.2, 0.2, 0.45, 0.4],  # 裸地
        ])
        names = auto_describe_classes(centers)
        assert any("植被" in n for n in names)
        assert any("水体" in n for n in names)
        assert any("裸地" in n for n in names)


class TestSTLDecompose:
    """STL 时序分解"""

    def test_decompose_shape(self):
        from utils.trend import stl_decompose
        t = np.arange(36)
        values = 0.3 + 0.003 * t + 0.08 * np.sin(2 * np.pi * t / 12)
        result = stl_decompose(values, seasonal_period=12)
        assert result["trend"].shape == (36,)
        assert result["seasonal"].shape == (36,)
        assert result["resid"].shape == (36,)

    def test_trend_recovery(self):
        """趋势分量应高度还原真实趋势"""
        from utils.trend import stl_decompose
        t = np.arange(36)
        trend_true = 0.3 + 0.003 * t
        seasonal_true = 0.08 * np.sin(2 * np.pi * t / 12)
        values = trend_true + seasonal_true + np.random.default_rng(0).normal(0, 0.01, 36)
        result = stl_decompose(values, seasonal_period=12)
        corr = np.corrcoef(result["trend"], trend_true)[0, 1]
        assert corr > 0.95

    def test_seasonal_recovery(self):
        from utils.trend import stl_decompose
        t = np.arange(36)
        trend_true = 0.3 + 0.003 * t
        seasonal_true = 0.08 * np.sin(2 * np.pi * t / 12)
        values = trend_true + seasonal_true
        result = stl_decompose(values, seasonal_period=12)
        corr = np.corrcoef(result["seasonal"], seasonal_true)[0, 1]
        assert corr > 0.9

    def test_short_series_graceful(self):
        """短时序应优雅降级 (返回全趋势)"""
        from utils.trend import stl_decompose
        values = np.array([0.3, 0.4, 0.5, 0.4, 0.6])
        result = stl_decompose(values, seasonal_period=12)
        assert result["trend"].shape == (5,)

    def test_nan_input(self):
        from utils.trend import stl_decompose
        t = np.arange(36)
        values = 0.3 + 0.003 * t + 0.08 * np.sin(2 * np.pi * t / 12)
        values[10] = np.nan
        result = stl_decompose(values, seasonal_period=12)
        assert np.all(np.isfinite(result["trend"]))
