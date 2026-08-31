"""AI 视觉理解与质量诊断测试"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def make_scene(seed=0, h=60, w=60):
    """模拟植被+水体+裸地影像"""
    rng = np.random.default_rng(seed)
    bands = rng.uniform(0.08, 0.3, (6, h, w))
    bands[3, 10:30, 10:30] = 0.45    # NIR 高 → 植被
    bands[0:3, 35:45, 35:45] = 0.06   # 低反射 → 水体
    bands[4, 50:60, 50:60] = 0.45     # SWIR 高 → 裸地
    return bands


class TestImageFingerprint:
    def test_fingerprint_fields(self):
        from utils.ai_vision import image_fingerprint
        fp = image_fingerprint(make_scene())
        assert "spectral" in fp
        assert "indices" in fp
        assert "composition" in fp
        assert "texture_roughness" in fp

    def test_composition_identifies_water(self):
        """水体区域应被识别"""
        from utils.ai_vision import image_fingerprint
        fp = image_fingerprint(make_scene())
        assert fp["composition"]["水体"] > 0.05

    def test_indices_computed(self):
        from utils.ai_vision import image_fingerprint
        fp = image_fingerprint(make_scene())
        assert set(fp["indices"].keys()) == {"NDVI", "MNDWI", "NDBI", "NDSI"}

    def test_insufficient_bands(self):
        from utils.ai_vision import image_fingerprint
        with pytest.raises(ValueError):
            image_fingerprint(np.zeros((3, 10, 10)))

    def test_dn_autoscale(self):
        """DN 值 (0-10000) 应自动缩放"""
        from utils.ai_vision import image_fingerprint
        bands = make_scene() * 10000
        fp = image_fingerprint(bands)
        assert 0 <= fp["spectral"]["红"]["mean"] <= 1.5


class TestVisionDescribe:
    def test_fallback_description(self):
        """无 API Key 时应生成规则描述"""
        from utils.ai_vision import vision_describe
        result = vision_describe(make_scene(), api_key="")
        assert result["description"]
        assert result["success"] is False
        assert result["fingerprint"]

    def test_description_mentions_composition(self):
        from utils.ai_vision import vision_describe
        result = vision_describe(make_scene(), api_key="")
        assert "%" in result["description"]


class TestQualityDiagnose:
    def test_clean_image_good_grade(self):
        """干净影像应评级较高"""
        from utils.ai_vision import quality_diagnose
        result = quality_diagnose(make_scene())
        assert result["grade"] in ("优", "良")
        assert result["score"] >= 70

    def test_cloudy_image_lower_score(self):
        """含云影像评级应低于干净影像"""
        from utils.ai_vision import quality_diagnose
        clean = make_scene()
        cloudy = clean.copy()
        cloudy[:, 0:15, 0:15] = 0.55  # 云区 (宽谱高反射)
        r_clean = quality_diagnose(clean)
        r_cloudy = quality_diagnose(cloudy)
        assert r_cloudy["score"] < r_clean["score"]
        assert r_cloudy["metrics"]["cloud_ratio"] > r_clean["metrics"]["cloud_ratio"]

    def test_metrics_fields(self):
        from utils.ai_vision import quality_diagnose
        result = quality_diagnose(make_scene())
        for key in ["cloud_ratio", "noise_level", "anomaly_ratio", "usable_ratio"]:
            assert key in result["metrics"]
            assert 0 <= result["metrics"][key] <= 1

    def test_diagnosis_text(self):
        """质量诊断应生成可读文本"""
        from utils.ai_vision import quality_diagnose
        result = quality_diagnose(make_scene(), api_key="")
        assert result["ai_diagnosis"]
        assert "质量" in result["ai_diagnosis"]

    def test_insufficient_bands(self):
        from utils.ai_vision import quality_diagnose
        with pytest.raises(ValueError):
            quality_diagnose(np.zeros((3, 10, 10)))
