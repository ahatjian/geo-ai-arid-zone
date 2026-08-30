"""
自定义光谱指数计算器 (spectral) 模块单元测试
测试: 波段运算求值器 + 预设指数
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_bands_dict():
    """模拟 6 波段数据字典"""
    np.random.seed(42)
    H, W = 40, 40
    return {
        "B": 0.10 + np.random.randn(H, W) * 0.02,
        "G": 0.15 + np.random.randn(H, W) * 0.03,
        "R": 0.18 + np.random.randn(H, W) * 0.03,
        "NIR": 0.35 + np.random.randn(H, W) * 0.04,
        "SWIR1": 0.25 + np.random.randn(H, W) * 0.04,
        "SWIR2": 0.20 + np.random.randn(H, W) * 0.03,
    }


class TestBandMath:
    def test_get_band_arrays(self):
        from utils.spectral import get_band_arrays
        bands = np.stack([
            np.full((10, 10), 0.1), np.full((10, 10), 0.2),
            np.full((10, 10), 0.3), np.full((10, 10), 0.4),
            np.full((10, 10), 0.5), np.full((10, 10), 0.6),
        ])
        d = get_band_arrays(bands)
        assert set(d.keys()) == {"B", "G", "R", "NIR", "SWIR1", "SWIR2"}
        assert d["NIR"].shape == (10, 10)

    def test_evaluate_ndvi(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        result = evaluate_band_math("(NIR - R) / (NIR + R)", mock_bands_dict)
        assert result.shape == (40, 40)
        assert np.all(np.isfinite(result))
        assert -1 <= np.nanmean(result) <= 1

    def test_evaluate_with_function(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        result = evaluate_band_math("sqrt(NIR * R)", mock_bands_dict)
        assert result.shape == (40, 40)
        assert np.nanmin(result) >= 0

    def test_evaluate_division_by_zero(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        # 构造除零场景
        bands = dict(mock_bands_dict)
        bands["R"] = -bands["NIR"]  # NIR + R = 0
        result = evaluate_band_math("(NIR - R) / (NIR + R)", bands)
        # 除零位置应为 NaN 而非 inf
        assert not np.any(np.isinf(result))

    def test_get_preset_index(self):
        from utils.spectral import get_preset_index, PRESET_INDICES
        p = get_preset_index("NDVI")
        assert p is not None
        assert p["name"] == "NDVI"
        assert "formula" in p
        # 所有预设指数公式都应能求值
        assert len(PRESET_INDICES) == 12


class TestPresetIndices:
    def test_all_presets_evaluable(self, mock_bands_dict):
        from utils.spectral import PRESET_INDICES, evaluate_band_math
        for p in PRESET_INDICES:
            result = evaluate_band_math(p["formula"], mock_bands_dict)
            assert result.shape == (40, 40), f"{p['name']} 求值失败"
            assert np.all(np.isfinite(result)), f"{p['name']} 含 inf/nan"

    def test_compute_index_stats(self, mock_bands_dict):
        from utils.spectral import compute_index_stats, evaluate_band_math
        ndvi = evaluate_band_math("(NIR - R) / (NIR + R)", mock_bands_dict)
        stats = compute_index_stats(ndvi)
        assert "mean" in stats
        assert "std" in stats
        assert stats["valid_ratio"] > 0.9


class TestBandMathSecurity:
    """AST 白名单求值器安全测试 (RCE 防护)"""

    MALICIOUS_EXPRESSIONS = [
        "__import__('os').system('id')",
        "().__class__.__bases__",
        "(NIR).__class__.__mro__",
        "open('C:/Windows/win.ini').read()",
        "[x for x in ()]",
        "globals()",
        "locals()",
        "getattr(__builtins__, 'eval')",
    ]

    def test_malicious_expressions_rejected(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        for expr in self.MALICIOUS_EXPRESSIONS:
            with pytest.raises((ValueError, SyntaxError, TypeError)):
                evaluate_band_math(expr, mock_bands_dict)

    def test_attribute_access_rejected(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        with pytest.raises(ValueError):
            evaluate_band_math("(NIR).shape", mock_bands_dict)

    def test_unknown_function_rejected(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        with pytest.raises(ValueError):
            evaluate_band_math("exec('pass')", mock_bands_dict)

    def test_unknown_identifier_rejected(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        with pytest.raises((ValueError, NameError)):
            evaluate_band_math("X + 1", mock_bands_dict)

    def test_legit_expressions_still_work(self, mock_bands_dict):
        """安全表达式不应被误杀"""
        from utils.spectral import evaluate_band_math
        for expr in [
            "(NIR - R) / (NIR + R)",
            "sqrt((NIR - R) / (NIR + R + 0.5))",
            "where(NIR > R, 1, 0)",
            "clip((NIR - R) * 100, -1, 1)",
            "1.5 * (NIR - R) / (NIR + R + 0.5)",
            "NIR / (R + SWIR1)",
        ]:
            result = evaluate_band_math(expr, mock_bands_dict)
            assert result.shape == (40, 40)

    def test_overlong_expression_rejected(self, mock_bands_dict):
        from utils.spectral import evaluate_band_math
        with pytest.raises(ValueError):
            evaluate_band_math("R + R + R + R" * 1000, mock_bands_dict)
