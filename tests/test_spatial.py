"""空间邻域分析测试 — 缓冲区/邻域统计/叠加交叉"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_oasis_scene(h=80, w=80, seed=0):
    """模拟绿洲场景: 中心植被 (类1) + 周边荒漠 + NDVI 梯度"""
    rng = np.random.default_rng(seed)
    classification = np.zeros((h, w), dtype=np.int16)
    classification[30:50, 30:50] = 1  # 绿洲
    ndvi = rng.uniform(0.05, 0.2, (h, w))
    ndvi[30:50, 30:50] = rng.uniform(0.35, 0.5, (20, 20))
    return classification, ndvi


class TestRasterBuffer:
    def test_buffer_shapes(self):
        from utils.spatial import raster_buffer
        cls, _ = make_oasis_scene()
        buf = raster_buffer(cls == 1, 200, 10)
        assert buf.shape == cls.shape
        assert set(np.unique(buf)) <= {0, 1, 2}

    def test_buffer_has_three_zones(self):
        """应有目标区/缓冲带/外部区三个分区"""
        from utils.spatial import raster_buffer
        cls, _ = make_oasis_scene()
        buf = raster_buffer(cls == 1, 200, 10)
        assert (buf == 2).sum() > 0    # 目标
        assert (buf == 1).sum() > 0    # 缓冲带
        assert (buf == 0).sum() > 0    # 外部

    def test_larger_buffer_bigger_zone(self):
        from utils.spatial import raster_buffer
        cls, _ = make_oasis_scene()
        buf_small = raster_buffer(cls == 1, 100, 10)
        buf_large = raster_buffer(cls == 1, 300, 10)
        assert (buf_large == 1).sum() > (buf_small == 1).sum()

    def test_no_target(self):
        from utils.spatial import raster_buffer
        cls, _ = make_oasis_scene()
        buf = raster_buffer(np.zeros_like(cls, dtype=bool), 200, 10)
        assert (buf == 1).sum() == 0  # 无目标 → 无缓冲带


class TestBufferZoneStats:
    def test_oasis_ndvi_gradient(self):
        """绿洲 NDVI 应高于缓冲带和外部 (绿洲-荒漠梯度)"""
        from utils.spatial import raster_buffer, buffer_zone_stats
        cls, ndvi = make_oasis_scene()
        buf = raster_buffer(cls == 1, 200, 10)
        stats = buffer_zone_stats(ndvi, buf)
        assert stats["target_mean"] > stats["buffer_mean"]
        assert stats["target_mean"] > stats["outer_mean"]
        assert stats["gradient"] > 0

    def test_stats_fields(self):
        from utils.spatial import raster_buffer, buffer_zone_stats
        cls, ndvi = make_oasis_scene()
        buf = raster_buffer(cls == 1, 200, 10)
        stats = buffer_zone_stats(ndvi, buf)
        for key in ["target_mean", "buffer_mean", "outer_mean", "gradient",
                    "target_pixels", "buffer_pixels"]:
            assert key in stats


class TestSmartBufferAnalysis:
    def test_analysis_text(self):
        from utils.spatial import smart_buffer_analysis
        cls, ndvi = make_oasis_scene()
        result = smart_buffer_analysis(cls, ndvi, target_class=1,
                                       distance_m=200, class_names=["背景", "绿洲"])
        assert result["target_name"] == "绿洲"
        assert result["buffer_distance_km"] == pytest.approx(0.2)
        assert "绿洲" in result["analysis_text"]
        assert "缓冲区" in result["analysis_text"]

    def test_invalid_class_graceful(self):
        """不存在的类别应正常返回 (无目标)"""
        from utils.spatial import smart_buffer_analysis
        cls, ndvi = make_oasis_scene()
        result = smart_buffer_analysis(cls, ndvi, target_class=9, distance_m=200)
        assert result["stats"]["target_pixels"] == 0


class TestNeighborhoodStats:
    def test_all_stats(self):
        from utils.spatial import neighborhood_stats
        cls, ndvi = make_oasis_scene()
        for stat in ["mean", "std", "cv", "median"]:
            out = neighborhood_stats(ndvi, 5, stat)
            assert out.shape == ndvi.shape
            assert np.all(np.isfinite(out[5:-5, 5:-5]))

    def test_mean_smooths(self):
        from utils.spatial import neighborhood_stats
        rng = np.random.default_rng(1)
        noisy = rng.normal(0.3, 0.05, (40, 40))
        smoothed = neighborhood_stats(noisy, 7, "mean")
        assert np.std(smoothed) < np.std(noisy)

    def test_cv_range(self):
        from utils.spatial import neighborhood_stats
        cls, ndvi = make_oasis_scene()
        cv = neighborhood_stats(ndvi, 5, "cv")
        assert np.nanmax(cv[5:-5, 5:-5]) > 0


class TestOverlayCrosstab:
    def test_crosstab_basic(self):
        from utils.spatial import overlay_crosstab
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, (40, 40)).astype(np.int16)
        b = rng.integers(0, 2, (40, 40)).astype(np.int16)
        result = overlay_crosstab(a, b, names_a=["A0", "A1", "A2"], names_b=["B0", "B1"])
        assert result["crosstab"].shape == (3, 2)
        assert result["crosstab"].sum() == 40 * 40
        assert result["areas_km2"].shape == (3, 2)
        assert result["rows"]

    def test_areas_sum(self):
        from utils.spatial import overlay_crosstab
        rng = np.random.default_rng(1)
        a = rng.integers(0, 2, (50, 50)).astype(np.int16)
        b = rng.integers(0, 2, (50, 50)).astype(np.int16)
        result = overlay_crosstab(a, b, pixel_size_m=10)
        total_area = sum(r["面积_km2"] for r in result["rows"])
        assert total_area == pytest.approx(50 * 50 * 100 / 1e6, abs=0.01)

    def test_top_relations(self):
        from utils.spatial import overlay_crosstab
        rng = np.random.default_rng(2)
        a = rng.integers(0, 2, (40, 40)).astype(np.int16)
        b = rng.integers(0, 2, (40, 40)).astype(np.int16)
        result = overlay_crosstab(a, b)
        assert len(result["top_relations"]) <= 5
        # 面积降序
        areas = [r["面积_km2"] for r in result["rows"]]
        assert areas == sorted(areas, reverse=True)

    def test_analysis_text(self):
        from utils.spatial import overlay_crosstab, overlay_analysis_text
        rng = np.random.default_rng(3)
        a = rng.integers(0, 2, (40, 40)).astype(np.int16)
        b = rng.integers(0, 2, (40, 40)).astype(np.int16)
        result = overlay_crosstab(a, b, names_a=["裸地", "植被"], names_b=["正常", "干旱"])
        text = overlay_analysis_text(result)
        assert "叠加分析" in text

    def test_shape_mismatch(self):
        from utils.spatial import overlay_crosstab
        with pytest.raises(ValueError):
            overlay_crosstab(np.zeros((10, 10), dtype=np.int16),
                             np.zeros((20, 20), dtype=np.int16))


class TestSecurity:
    """空间分析安全加固测试"""

    def test_oversized_class_rejected(self):
        """恶意超大类别值应被拒绝 (内存 DoS 防护)"""
        from utils.spatial import overlay_crosstab
        a = np.array([[0, 1], [2, 3]], dtype=np.int32)
        b = np.full((2, 2), 999999, dtype=np.int32)  # 恶意类别值
        with pytest.raises(ValueError):
            overlay_crosstab(a, b)

    def test_negative_class_rejected(self):
        from utils.spatial import overlay_crosstab
        a = np.array([[0, 1]], dtype=np.int16)
        b = np.array([[-5, 0]], dtype=np.int16)
        with pytest.raises(ValueError):
            overlay_crosstab(a, b)

    def test_sparse_classes_compact(self):
        """稀疏类别 (如 0 和 100) 应紧凑分配, 不产生大数组"""
        from utils.spatial import overlay_crosstab
        a = np.array([[0, 100]], dtype=np.int16)
        b = np.array([[0, 1]], dtype=np.int16)
        result = overlay_crosstab(a, b)
        # 形状 = 实际类别数 (2×2), 而非 101×2
        assert result["crosstab"].shape == (2, 2)
        assert result["crosstab"].sum() == 2

    def test_sparse_values_mapped(self):
        """稀疏类别的面积统计应正确映射回原始值"""
        from utils.spatial import overlay_crosstab
        a = np.array([[0, 100]], dtype=np.int16)
        b = np.array([[0, 1]], dtype=np.int16)
        result = overlay_crosstab(a, b, pixel_size_m=10)
        # 每个组合 1 像元 = 100 m² = 0.0001 km²
        assert result["rows"][0]["面积_km2"] == pytest.approx(0.0001)
        # 原始类别值保留在行记录中
        vals = {(r["a_val"], r["b_val"]) for r in result["rows"]}
        assert (0, 0) in vals and (100, 1) in vals
