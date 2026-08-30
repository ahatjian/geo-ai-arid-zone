"""AI 解读引擎单元测试 — 安全降级与规则解读验证"""

import pytest
import utils.ai_insight as ai


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """测试期间强制无 API Key, 走规则降级路径。"""
    monkeypatch.setattr(ai, "_get_api_key", lambda: "")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


class TestRuleBasedInsight:
    def test_returns_text_without_api(self):
        text = ai.generate_ai_insight("植被分析", {"NDVI均值": 0.3})
        assert isinstance(text, str) and len(text) > 10

    def test_ndvi_low_classification(self):
        text = ai.generate_ai_insight("植被分析", {"NDVI均值": 0.05})
        assert "极低" in text or "裸地" in text

    def test_ndvi_medium_classification(self):
        text = ai.generate_ai_insight("植被分析", {"NDVI均值": 0.25})
        assert "稀疏" in text or "低" in text

    def test_ndvi_high_classification(self):
        text = ai.generate_ai_insight("植被分析", {"NDVI均值": 0.65})
        assert "较高" in text or "良好" in text

    def test_vegetation_ratio_converted_to_percent(self):
        text = ai.generate_ai_insight("植被分析", {"植被覆盖率": 0.35})
        assert "35.0%" in text

    def test_water_ratio(self):
        text = ai.generate_ai_insight("水体监测", {"水体覆盖率": 0.05})
        assert "5.0%" in text

    def test_lst_hot_region(self):
        text = ai.generate_ai_insight("地表温度", {"LST均值": 42.5})
        assert "42.5" in text and ("热环境" in text or "风险" in text)

    def test_empty_metrics_uses_generic_notes(self):
        text = ai.generate_ai_insight("综合评估", {})
        assert len(text) > 10

    def test_study_area_and_period_included(self):
        text = ai.generate_ai_insight("植被分析", {"NDVI均值": 0.3}, "河西走廊", "2024-2025")
        assert "河西走廊" in text
        assert "2024-2025" in text

    def test_output_length_limited(self):
        text = ai.generate_ai_insight("分析", {"指标1": 0.1, "指标2": 0.2, "指标3": 0.3})
        assert len(text) <= ai.MAX_OUTPUT_CHARS + 50


class TestSummarize:
    def test_short_text_unchanged(self):
        assert ai.summarize_insight("短文") == "短文"

    def test_long_text_truncated(self):
        long_text = "。".join(["这是一个用于测试摘要功能的较长句子内容"] * 30)
        summary = ai.summarize_insight(long_text, max_len=50)
        assert len(summary) <= 60
        assert summary.endswith("…")

    def test_empty(self):
        assert ai.summarize_insight("") == ""


class TestExplainMetrics:
    def test_ndvi_explained(self):
        text = ai.explain_metrics(["NDVI"])
        assert "NDVI" in text

    def test_unknown_metric_falls_back(self):
        text = ai.explain_metrics(["CUSTOM_METRIC_XYZ"])
        assert "自定义" in text or "说明" in text

    def test_multiple_metrics(self):
        text = ai.explain_metrics(["NDVI", "MNDWI"])
        assert "NDVI" in text and "MNDWI" in text


class TestHistory:
    def test_empty_history(self):
        assert ai.get_insight_history() == []

    def test_is_ai_available(self):
        # 无 key 时不可用
        assert ai.is_ai_available() is False
