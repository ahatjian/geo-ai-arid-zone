"""
LLM 智能查询模块 — DeepSeek API 集成
====================================
为智能工作流提供自然语言理解能力

支持:
  - 自然语言 → 分析模块匹配 (增强版)
  - 研究区名称实体识别
  - 时间范围提取
  - 分析建议生成

配置:
  设置环境变量 DEEPSEEK_API_KEY 或通过 config 传入
  API Base: https://api.deepseek.com/v1

依赖: requests (已安装)
"""

import os
import re
import json
from typing import Optional, Dict, List
from datetime import datetime

# ---- 模块定义 (与 pages/12 共用) ----
MODULE_DEFINITIONS = [
    {"name": "数据浏览", "page": "1_数据浏览", "icon": "🗺️",
     "keywords": ["搜索", "预览", "下载", "查看影像", "最新"],
     "desc": "STAC 搜索 + RGB/NDVI/MNDWI 预览"},
    {"name": "水体监测", "page": "2_水体监测", "icon": "💧",
     "keywords": ["水体", "湖泊", "mndwi", "水面", "水库", "水域", "河流"],
     "desc": "MNDWI/AWEIsh + AI 水体分割"},
    {"name": "植被分析", "page": "3_植被分析", "icon": "🌿",
     "keywords": ["ndvi", "植被", "绿洲", "覆盖", "草地", "森林", "绿化"],
     "desc": "NDVI/EVI + Sen+MK 趋势"},
    {"name": "AI分类", "page": "4_AI分类", "icon": "🤖",
     "keywords": ["分类", "地物", "土地", "覆盖类型", "AI", "分割", "模型"],
     "desc": "ESA/ESRI + 深度学习地物分割"},
    {"name": "变化检测", "page": "5_变化检测", "icon": "🔄",
     "keywords": ["变化", "检测", "对比", "前后", "差异", "演变"],
     "desc": "双时相 7 级变化分类"},
    {"name": "报告导出", "page": "6_报告导出", "icon": "📄",
     "keywords": ["报告", "导出", "汇总", "生成报告", "综合"],
     "desc": "HTML 综合分析报告"},
    {"name": "干旱监测", "page": "7_干旱监测", "icon": "🏜️",
     "keywords": ["干旱", "vci", "nddi", "缺水", "距平", "预测", "沙漠化"],
     "desc": "多指数干旱 + 预测 + 沙漠化"},
    {"name": "冰冻圈分析", "page": "8_冰冻圈分析", "icon": "❄️",
     "keywords": ["冰川", "雪", "ndsi", "冻土", "冰", "积雪", "融雪"],
     "desc": "NDSI 雪盖 + 冰川边界 + 冻土"},
    {"name": "农业干旱", "page": "9_农业干旱", "icon": "🌾",
     "keywords": ["农业", "作物", "灌溉", "cwsi", "农田", "土壤", "棉花", "小麦"],
     "desc": "CWSI + 土壤水分 + 灌溉需求"},
    {"name": "时序动画", "page": "10_时序动画", "icon": "🎬",
     "keywords": ["动画", "gif", "视频", "时序", "动态", "演变过程"],
     "desc": "NDVI/水体/雪盖 GIF 动画"},
    {"name": "生态评估", "page": "11_生态评估", "icon": "🌍",
     "keywords": ["生态", "安全", "psr", "评估", "环境", "脆弱", "保护"],
     "desc": "PSR 生态安全评价"},
    {"name": "土壤盐渍化", "page": "13_土壤盐渍化", "icon": "🧂",
     "keywords": ["盐渍化", "盐分", "盐碱", "盐渍", "盐霜", "盐壳", "salinity", "次生盐化"],
     "desc": "SI/NDSI 盐分指数 + 5级盐渍化评估"},
    {"name": "地表温度", "page": "14_LST", "icon": "🌡️",
     "keywords": ["温度", "地表温度", "热环境", "热岛", "lst", "高温", "热红外"],
     "desc": "Landsat 热红外 LST 反演 + 热环境分级"},
    {"name": "指数计算器", "page": "15_指数计算器", "icon": "🧮",
     "keywords": ["指数计算", "自定义指数", "波段运算", "band math", "自定义公式"],
     "desc": "预设指数 + 自定义波段运算"},
    {"name": "土地转移", "page": "16_土地转移", "icon": "🔀",
     "keywords": ["转移矩阵", "土地利用变化", "土地覆盖变化", "转移", "用地变化", "转化"],
     "desc": "双时相土地覆盖转移矩阵 + 净变化"},
    {"name": "蒸散发", "page": "17_蒸散发", "icon": "💨",
     "keywords": ["蒸散发", "蒸发", "蒸腾", "et", "耗水", "水资源", "能量平衡", "sebal"],
     "desc": "SEBAL 能量平衡蒸散发估算"},
    {"name": "监督分类", "page": "18_监督分类", "icon": "🎯",
     "keywords": ["监督分类", "训练", "分类器", "随机森林", "svm", "样本", "机器学习", "训练样本"],
     "desc": "自定义样本训练分类模型 (RF/SVM/KNN/MLP)"},
    {"name": "矢量导出", "page": "19_矢量导出", "icon": "🗺️",
     "keywords": ["矢量", "导出", "shapefile", "geojson", "kml", "矢量化", "面要素", "多边形", "arcgis", "qgis", "边界"],
     "desc": "分类结果矢量化导出 (GeoJSON/Shapefile/KML)"},
]

STUDY_AREA_NAMES = [
    "塔里木盆地", "柴达木盆地", "河西走廊",
    "吐鲁番盆地", "天山北坡", "准噶尔盆地",
]

SYSTEM_PROMPT = """你是一个西北干旱区遥感分析助手。用户用中文描述分析需求，你要解析并返回 JSON。

任务：
1. 识别用户想分析的研究区 (study_area)
2. 识别时间范围 (year_start, year_end)
3. 推荐分析模块 (modules)，从以下列表中选择：
   [数据浏览, 水体监测, 植被分析, AI分类, 变化检测, 报告导出, 
    干旱监测, 冰冻圈分析, 农业干旱, 时序动画, 生态评估]
4. 生成简短解释 (explanation)

只返回 JSON，不要其他文字。格式：
{"study_area": "塔里木盆地", "year_start": 2024, "year_end": 2025, 
 "modules": ["植被分析", "干旱监测"], "explanation": "帮你分析..."}

如果没有明确研究区，默认"塔里木盆地"。如果没有明确年份，默认今年。"""


def _get_api_key() -> str:
    """从多个来源获取 API Key"""
    # 1. 传入参数
    # 2. Streamlit secrets
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception as e:
        import logging
        logging.debug(f"LLM st.secrets/query error: {e}")
    # 3. 环境变量
    return os.environ.get("DEEPSEEK_API_KEY", "")


def query_deepseek(prompt: str, api_key: Optional[str] = None) -> Dict:
    """
    调用 DeepSeek API 解析自然语言查询

    参数:
        prompt: 用户自然语言查询
        api_key: DeepSeek API Key (None=从环境变量读取)

    返回:
        dict: 解析结果
    """
    if api_key is None:
        api_key = _get_api_key()

    if not api_key:
        # 无 API Key → 降级到模板匹配
        return fallback_parse(prompt)

    try:
        import requests
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 300,
            },
            timeout=15,
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # 提取 JSON
        json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return _validate_result(parsed)
    except Exception as e:
        import logging
        logging.debug(f"LLM st.secrets/query error: {e}")

    return fallback_parse(prompt)


def fallback_parse(prompt: str) -> Dict:
    """
    降级方案: 基于模板匹配的查询解析 (无需 LLM)
    与 pages/12 共用模板逻辑
    """
    prompt_lower = prompt.lower()

    # 研究区识别
    study_area = "塔里木盆地"
    for area in STUDY_AREA_NAMES:
        if area in prompt:
            study_area = area
            break

    # 年份提取
    years = re.findall(r'(20\d{2})', prompt)
    year_start = int(years[0]) if len(years) >= 1 else datetime.now().year
    year_end = int(years[1]) if len(years) >= 2 else year_start

    # 模块匹配
    modules = []
    for mod in MODULE_DEFINITIONS:
        score = sum(1 for kw in mod["keywords"] if kw in prompt_lower)
        if score > 0:
            modules.append({"name": mod["name"], "page": mod["page"],
                           "icon": mod["icon"], "desc": mod["desc"], "score": score})
    modules.sort(key=lambda x: x["score"], reverse=True)

    # 生成解释
    if modules:
        mod_names = [m["name"] for m in modules[:3]]
        explanation = f"在 {study_area}（{year_start}年），推荐使用{'、'.join(mod_names)}模块进行分析。"
    else:
        explanation = f"建议从数据浏览开始，查看 {study_area} 的卫星影像。"

    return {
        "study_area": study_area,
        "year_start": year_start,
        "year_end": year_end,
        "modules": modules[:5],
        "explanation": explanation,
        "method": "fallback",
    }


def _validate_result(result: Dict) -> Dict:
    """验证和补全 LLM 返回结果"""
    if "study_area" not in result or result["study_area"] not in STUDY_AREA_NAMES:
        result["study_area"] = "塔里木盆地"

    for field in ["year_start", "year_end"]:
        if field not in result or not isinstance(result.get(field), int):
            result[field] = datetime.now().year

    if "modules" not in result:
        result["modules"] = []
    else:
        # 将字符串模块名转为完整信息
        enriched = []
        for m in result["modules"]:
            mod_name = m if isinstance(m, str) else m.get("name", "")
            for defn in MODULE_DEFINITIONS:
                if defn["name"] == mod_name:
                    enriched.append({**defn, "score": 5})
                    break
        result["modules"] = enriched if enriched else result["modules"]

    result["explanation"] = result.get("explanation", "已为你匹配相关分析模块。")
    result["method"] = "deepseek"
    return result


def is_llm_available() -> bool:
    """检查 LLM 是否可用"""
    return bool(_get_api_key())
