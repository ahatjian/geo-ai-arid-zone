"""LLM 模块完整性测试 — 模块清单与提示词同步验证"""

import utils.llm as llm


class TestModuleDefinitions:
    def test_all_20_modules_registered(self):
        """平台 20 个分析模块应全部注册到 LLM 模块清单。"""
        assert len(llm.MODULE_DEFINITIONS) == 20

    def test_known_modules_present(self):
        names = {m["name"] for m in llm.MODULE_DEFINITIONS}
        for expected in ["数据浏览", "智能工作流", "土壤盐渍化", "地表温度",
                         "指数计算器", "土地转移", "蒸散发", "监督分类",
                         "矢量导出", "数据下载中心"]:
            assert expected in names, f"缺少模块: {expected}"

    def test_pages_unique(self):
        pages = [m["page"] for m in llm.MODULE_DEFINITIONS]
        assert len(pages) == len(set(pages)), "页面存在重复"

    def test_required_fields(self):
        for m in llm.MODULE_DEFINITIONS:
            for field in ["name", "page", "icon", "keywords", "desc"]:
                assert field in m, f"模块 {m.get('name')} 缺少字段 {field}"
            assert m["keywords"], f"模块 {m['name']} 关键词为空"


class TestSystemPrompt:
    def test_prompt_covers_all_modules(self):
        """提示词应自动包含全部模块, 保证 LLM 能推荐新模块。"""
        for m in llm.MODULE_DEFINITIONS:
            assert m["name"] in llm.SYSTEM_PROMPT, f"提示词缺少模块 {m['name']}"

    def test_prompt_mentions_study_areas(self):
        for area in llm.STUDY_AREA_NAMES:
            assert area in llm.SYSTEM_PROMPT

    def test_prompt_dynamic_rebuild(self):
        """提示词由模块清单自动生成, 模块增删后提示词同步。"""
        assert llm.SYSTEM_PROMPT == llm.build_system_prompt()


class TestFallbackParse:
    def test_study_area_detection(self):
        r = llm.fallback_parse("帮我分析天山北坡的冰川")
        assert r["study_area"] == "天山北坡"

    def test_default_study_area(self):
        r = llm.fallback_parse("分析植被情况")
        assert r["study_area"] == "塔里木盆地"

    def test_year_extraction(self):
        r = llm.fallback_parse("分析2023到2025年干旱情况")
        assert r["year_start"] == 2023
        assert r["year_end"] == 2025

    def test_module_matching(self):
        r = llm.fallback_parse("帮我看看湖泊的水体面积变化")
        names = [m["name"] for m in r["modules"]]
        assert "水体监测" in names

    def test_salinity_keyword(self):
        r = llm.fallback_parse("盐渍化情况怎么样")
        names = [m["name"] for m in r["modules"]]
        assert "土壤盐渍化" in names

    def test_no_match_falls_to_browse(self):
        r = llm.fallback_parse("随便看看")
        assert r["modules"], "至少推荐一个模块"

    def test_method_flag(self):
        assert llm.fallback_parse("测试")["method"] == "fallback"
