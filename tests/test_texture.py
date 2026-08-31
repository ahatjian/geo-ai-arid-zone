"""GLCM 纹理分析测试 — 纹理特征提取与分类辅助"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_texture_image(h=60, w=60, seed=0):
    """上半平滑 / 下半粗糙的纹理测试图"""
    rng = np.random.default_rng(seed)
    smooth = rng.normal(0.3, 0.01, (h // 2, w))
    rough = rng.normal(0.3, 0.08, (h // 2, w))
    return np.concatenate([smooth, rough])


class TestGLCMTextures:
    def test_texture_maps_shape(self):
        from utils.texture import glcm_texture_maps
        band = make_texture_image()
        maps = glcm_texture_maps(band, window_size=9)
        assert set(maps.keys()) == {"contrast", "dissimilarity", "homogeneity",
                                    "asm", "energy", "correlation"}
        for m in maps.values():
            assert m.shape == band.shape

    def test_contrast_distinguishes_texture(self):
        """粗糙区对比度应显著高于平滑区"""
        from utils.texture import glcm_texture_maps
        band = make_texture_image()
        maps = glcm_texture_maps(band, window_size=9)
        rough_mean = np.nanmean(maps["contrast"][30:])
        smooth_mean = np.nanmean(maps["contrast"][:30])
        assert rough_mean > smooth_mean * 1.5

    def test_fast_version_same_shape(self):
        from utils.texture import glcm_texture_maps_fast
        band = make_texture_image()
        maps = glcm_texture_maps_fast(band, window_size=9, step=2)
        assert maps["contrast"].shape == band.shape

    def test_global_stats(self):
        from utils.texture import glcm_global_stats
        band = make_texture_image()
        stats = glcm_global_stats(band)
        assert set(stats.keys()) == {"contrast", "dissimilarity", "homogeneity",
                                     "asm", "energy", "correlation"}
        assert stats["contrast"] > 0

    def test_texture_stack(self):
        from utils.texture import build_texture_stack
        band = make_texture_image(40, 40)
        stack, names = build_texture_stack(np.stack([band, band]), window_size=9, step=2)
        assert stack.shape[0] == 12  # 2 波段 × 6 纹理
        assert len(names) == 12
        assert names[0] == "band1_contrast"

    def test_quantize_edge_case(self):
        """常量波段不应崩溃"""
        from utils.texture import glcm_texture_maps
        band = np.full((20, 20), 0.5)
        maps = glcm_texture_maps(band, window_size=5)
        assert maps["contrast"].shape == (20, 20)


class TestTextureClassification:
    def test_texture_improves_accuracy(self):
        """纹理特征应提升分类精度 (粗糙/平滑区分)"""
        from utils.supervised import assess_supervised_classification
        rng = np.random.default_rng(0)
        bands_dict = {b: rng.uniform(0.05, 0.4, (30, 30))
                      for b in ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]}
        bands_dict["NIR"][:15] += rng.normal(0, 0.08, (15, 30))

        train_idx, train_lbl = [], []
        for i in range(0, 30, 3):
            for j in range(0, 30, 6):
                train_idx.append([i, j])
                train_lbl.append(0 if i < 15 else 1)

        r_base = assess_supervised_classification(
            bands_dict, np.array(train_idx), np.array(train_lbl),
            classifier="random_forest", add_indices=True, add_texture=False)
        r_tex = assess_supervised_classification(
            bands_dict, np.array(train_idx), np.array(train_lbl),
            classifier="random_forest", add_indices=True, add_texture=True)

        # 纹理版本特征数 = 基础 + 6
        assert r_tex.summary["n_features"] == r_base.summary["n_features"] + 6
        # 纹理应提升精度 (至少不降低)
        assert r_tex.accuracy["overall_accuracy"] >= r_base.accuracy["overall_accuracy"] - 0.05

    def test_texture_stack_in_supervised(self):
        """监督分类支持 add_texture 参数"""
        from utils.supervised import build_feature_stack
        rng = np.random.default_rng(1)
        bands_dict = {b: rng.uniform(0.05, 0.4, (20, 20))
                      for b in ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]}
        stack = build_feature_stack(bands_dict, add_indices=True, add_texture=True)
        assert stack.shape[2] == 15  # 6 波段 + 3 指数 + 6 纹理
