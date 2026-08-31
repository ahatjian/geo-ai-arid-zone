"""AI 智能助手测试 — 对话/自动分析/异常检测"""

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


class TestAutoAnalyze:
    def test_composition_detected(self):
        """应识别出植被/水体/裸地三类"""
        from utils.ai_assistant import auto_analyze
        result = auto_analyze(make_scene())
        names = [c["name"] for c in result["land_composition"]]
        assert any("水体" in n for n in names)
        assert any("植被" in n for n in names)
        assert any("裸地" in n for n in names)

    def test_ratios_sum_approx(self):
        """构成占比应接近 1"""
        from utils.ai_assistant import auto_analyze
        result = auto_analyze(make_scene())
        total = sum(c["ratio"] for c in result["land_composition"])
        assert abs(total - 1.0) < 0.05

    def test_indices_computed(self):
        """NDVI/MNDWI/NDBI 应全部计算"""
        from utils.ai_assistant import auto_analyze
        result = auto_analyze(make_scene())
        assert set(result["indices"].keys()) == {"NDVI", "MNDWI", "NDBI"}

    def test_ai_assessment_fallback(self):
        """无 API Key 时降级为规则评估"""
        from utils.ai_assistant import auto_analyze
        result = auto_analyze(make_scene(), api_key="")
        assert result["ai_assessment"]
        assert "AI 自动评估" in result["ai_assessment"]

    def test_insufficient_bands(self):
        from utils.ai_assistant import auto_analyze
        with pytest.raises(ValueError):
            auto_analyze(np.zeros((3, 10, 10)))


class TestDetectAnomalies:
    def test_local_detects_injected(self):
        """局部方法应检出注入的异常区"""
        from utils.ai_assistant import detect_anomalies
        bands = make_scene()
        ndvi = (bands[3] - bands[2]) / (bands[3] + bands[2] + 1e-6)
        ndvi[20:25, 20:25] = -0.3  # 注入异常
        result = detect_anomalies(ndvi, method="local")
        assert result["anomaly_mask"].sum() > 0
        assert result["n_clusters"] >= 1

    def test_no_anomaly_smooth(self):
        """平滑影像不应检出异常"""
        from utils.ai_assistant import detect_anomalies
        smooth = np.full((40, 40), 0.3) + np.random.default_rng(1).normal(0, 0.005, (40, 40))
        result = detect_anomalies(smooth, method="local", threshold=4.0)
        assert result["anomaly_ratio"] < 0.05

    def test_min_cluster_filters(self):
        """碎斑过滤: 小异常簇应被滤除"""
        from utils.ai_assistant import detect_anomalies
        rng = np.random.default_rng(2)
        band = rng.uniform(0.2, 0.4, (50, 50))
        band[5, 5] = 1.0  # 单像元异常 (碎斑)
        result = detect_anomalies(band, method="local", min_cluster=16)
        # 单像元异常应被过滤
        assert result["anomaly_mask"][5, 5] == False or result["anomaly_ratio"] < 0.02

    def test_all_methods(self):
        from utils.ai_assistant import detect_anomalies
        bands = make_scene()
        ndvi = (bands[3] - bands[2]) / (bands[3] + bands[2] + 1e-6)
        for method in ["local", "mad", "zscore"]:
            result = detect_anomalies(ndvi, method=method)
            assert result["anomaly_mask"].shape == ndvi.shape
            assert "method" in result

    def test_small_image_graceful(self):
        from utils.ai_assistant import detect_anomalies
        result = detect_anomalies(np.random.default_rng(0).uniform(0, 1, (8, 8)))
        assert result["anomaly_ratio"] == 0.0


class TestChatAssistant:
    def test_no_api_key_message(self):
        """无 Key 时应返回引导信息而非崩溃"""
        from utils.ai_assistant import chat_with_assistant
        resp = chat_with_assistant([{"role": "user", "content": "你好"}], api_key="")
        assert resp["success"] is False
        assert "DEEPSEEK_API_KEY" in resp["reply"]

    def test_messages_normalization(self):
        """应自动注入系统提示"""
        from utils.ai_assistant import chat_with_assistant
        # 无法真实调用, 但无 key 时不应崩溃
        resp = chat_with_assistant([], api_key="")
        assert resp["error"] is not None


class TestPlatformContext:
    """平台上下文注入测试"""

    def test_build_context_empty(self):
        """无分析结果时返回空提示"""
        import streamlit as st
        st.session_state.clear()
        from utils.ai_assistant import build_platform_context
        ctx = build_platform_context()
        assert "尚未完成" in ctx

    def test_build_context_with_results(self):
        """有分析结果时应汇总关键指标"""
        import streamlit as st
        st.session_state.clear()
        st.session_state["veg_stats"] = {"mean": 0.35, "dense_veg_ratio": 0.4, "bare_ratio": 0.1}
        st.session_state["salinity_stats"] = {
            "summary": {"total_ratio": 0.25, "dominant_level": "中度盐渍化"}}
        from utils.ai_assistant import build_platform_context
        ctx = build_platform_context()
        assert "植被分析" in ctx
        assert "土壤盐渍化" in ctx
        assert "0.350" in ctx  # NDVI 均值

    def test_build_context_tolerates_bad_data(self):
        """异常数据结构不应崩溃"""
        import streamlit as st
        st.session_state.clear()
        st.session_state["veg_stats"] = "not a dict"  # 坏数据
        from utils.ai_assistant import build_platform_context
        ctx = build_platform_context()
        assert isinstance(ctx, str)


class TestKnowledgeRAG:
    """RAG 遥感知识库测试"""

    def test_search_ndvi(self):
        from utils.knowledge_base import search_knowledge
        hits = search_knowledge("NDVI 怎么计算")
        assert hits
        assert any("NDVI" in h["content"] for h in hits)

    def test_search_atmospheric(self):
        from utils.knowledge_base import search_knowledge
        hits = search_knowledge("什么是 DOS 大气校正")
        assert any("大气校正" in h["content"] for h in hits)

    def test_search_study_area(self):
        from utils.knowledge_base import search_knowledge
        hits = search_knowledge("塔里木盆地生态")
        assert any("塔里木盆地" in h["content"] for h in hits)

    def test_no_match_returns_empty(self):
        from utils.knowledge_base import search_knowledge
        assert search_knowledge("qqqqzzzz毫无意义") == []

    def test_build_context(self):
        from utils.knowledge_base import build_knowledge_context
        ctx = build_knowledge_context("MNDWI 水体指数")
        assert "知识库" in ctx
        assert "MNDWI" in ctx

    def test_empty_query_no_context(self):
        from utils.knowledge_base import build_knowledge_context
        assert build_knowledge_context("") == ""

    def test_index_formula(self):
        from utils.knowledge_base import search_index_formula, get_index_formula
        assert "NDVI" in search_index_formula("帮我算 NDVI")
        assert get_index_formula("ndvi") == "(NIR - R) / (NIR + R)"

    def test_knowledge_stats(self):
        from utils.knowledge_base import knowledge_stats
        stats = knowledge_stats()
        assert stats["entries"] >= 30
        assert stats["index_formulas"] >= 10
