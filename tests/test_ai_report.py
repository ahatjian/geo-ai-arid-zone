"""AI 智能报告全文测试"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


def make_sources():
    """模拟报告导出页采集的完整指标"""
    return {
        "water_area": 12.5, "water_pct": 3.2,
        "veg_mean": 0.35, "veg_coverage": 42.5, "veg_index": "NDVI", "veg_trend": "不显著",
        "change_increase": 5.2, "change_decrease": 3.1,
        "salinity_total_ratio": 0.25, "salinity_severe_ratio": 0.08, "salinity_dominant": "中度盐渍化",
        "lst_mean": 32.5, "lst_max": 45.2, "lst_hot_ratio": 0.3,
        "et_mean": 2.8, "et_rn": 210.5,
        "sup_oa": 0.873, "sup_kappa": 0.812, "sup_f1": 0.855,
        "bfast_n": 2, "bfast_neg": 1, "bfast_pos": 1,
        "trans_change": 125.3, "trans_t1": "2020", "trans_t2": "2021",
    }


class TestSectionsFromSources:
    def test_all_modules_extracted(self):
        from utils.ai_report import sections_from_sources
        sections = sections_from_sources(make_sources())
        assert len(sections) >= 8  # 9 个模块
        names = [s["name"] for s in sections]
        assert "水体监测" in names
        assert "植被分析" in names
        assert "BFAST 断点检测" in names

    def test_empty_sources(self):
        from utils.ai_report import sections_from_sources
        assert sections_from_sources({}) == []

    def test_water_zero_skipped(self):
        from utils.ai_report import sections_from_sources
        src = make_sources()
        src["water_area"] = 0
        sections = sections_from_sources(src)
        assert not any(s["name"] == "水体监测" for s in sections)


class TestGenerateFullReport:
    def test_template_fallback(self):
        """无 API Key 时生成模板报告"""
        from utils.ai_report import generate_full_report
        from utils.ai_report import sections_from_sources
        result = generate_full_report(
            study_area="测试区",
            sections=sections_from_sources(make_sources()),
            api_key="",
        )
        assert result["report"]
        assert result["method"] == "template"
        assert result["sections_used"] >= 8
        # 报告结构
        for marker in ["执行摘要", "分项分析", "综合结论", "对策与建议"]:
            assert marker in result["report"]

    def test_data_table_included(self):
        from utils.ai_report import generate_full_report
        from utils.ai_report import sections_from_sources
        result = generate_full_report(
            study_area="测试区",
            sections=sections_from_sources(make_sources()),
            api_key="",
            include_data_table=True,
        )
        assert "指标数据表" in result["report"]
        assert "| 模块 | 指标 | 数值 |" in result["report"]

    def test_no_data_table_when_off(self):
        from utils.ai_report import generate_full_report
        result = generate_full_report(
            study_area="测试区",
            sections=[{"name": "植被", "metrics": {"NDVI": 0.3}}],
            api_key="",
            include_data_table=False,
        )
        assert "指标数据表" not in result["report"]

    def test_empty_sections(self):
        """无指标时仍应生成报告 (不崩溃)"""
        from utils.ai_report import generate_full_report
        result = generate_full_report(study_area="测试区", sections=[], api_key="")
        assert result["report"]
        assert result["sections_used"] == 0

    def test_template_mentions_metrics(self):
        """模板报告应引用指标值"""
        from utils.ai_report import generate_full_report
        from utils.ai_report import sections_from_sources
        result = generate_full_report(
            study_area="塔里木盆地",
            sections=sections_from_sources(make_sources()),
            api_key="",
        )
        assert "NDVI" in result["report"] or "0.35" in result["report"]
