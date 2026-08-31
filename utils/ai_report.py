"""
AI 智能报告全文生成模块
========================
基于平台已采集的各模块分析指标, DeepSeek 一键生成完整科研报告:
执行摘要 → 分章节解读 → 综合结论 → 对策建议。

与 utils/ai_insight.py 的区别:
  ai_insight: 单模块指标 → 单段解读
  本模块:    全平台指标 → 结构化完整报告 (可下载 Markdown)

报告结构 (AI 生成):
  # 标题
  ## 一、执行摘要
  ## 二、分项分析 (水体/植被/盐渍化/... 每个有数据的模块)
  ## 三、综合结论
  ## 四、对策与建议
  (附) 指标数据表
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 2

REPORT_SYSTEM_PROMPT = """你是干旱区遥感与生态研究专家。用户提供研究区多模块遥感分析指标，请生成一份完整的科研分析报告。

报告结构要求:
# {研究区} 遥感综合分析报告

## 一、执行摘要
200 字以内概括研究区整体状况与关键发现。

## 二、分项分析
对每个有数据的分析模块写 2-4 句话:
- 指标数值含义 (结合干旱区背景)
- 生态/环境信号解读
- 与其他模块的关联 (如有)

## 三、综合结论
150 字以内, 基于所有模块给出研究区综合评估。

## 四、对策与建议
给出 3-5 条具体可执行的监测/管理建议 (带优先级)。

要求: 专业、客观、数据驱动; 只使用提供的数据, 不编造指标; 中文输出。"""


def _get_api_key() -> str:
    """从 st.secrets 或环境变量读取 DeepSeek API Key。"""
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception as e:
        logger.debug(f"ai_report st.secrets error: {e}")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def is_ai_available() -> bool:
    return bool(_get_api_key())


def _format_sections(sections: List[Dict]) -> str:
    """将各模块指标格式化为 prompt 文本块。"""
    lines = []
    for sec in sections:
        name = sec.get("name", "")
        metrics = sec.get("metrics", {})
        if not metrics:
            continue
        lines.append(f"### {name}")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines) if lines else "（无指标数据）"


def generate_full_report(
    study_area: str = "研究区",
    sections: Optional[List[Dict]] = None,
    time_range: str = "",
    api_key: Optional[str] = None,
    include_data_table: bool = True,
) -> Dict:
    """
    生成完整科研报告。

    参数:
        study_area: 研究区名称
        sections: [{"name": "水体监测", "metrics": {"水体占比": 0.05, ...}}, ...]
        time_range: 时间范围 (可选)
        api_key: DeepSeek Key (None=自动)
        include_data_table: 报告末尾是否附指标数据表

    返回:
        dict: {
            "report": 完整报告文本 (Markdown),
            "success": bool,
            "method": "deepseek" | "template",
            "sections_used": int 有数据的模块数
        }
    """
    if api_key is None:
        api_key = _get_api_key()

    sections = sections or []
    used = [s for s in sections if s.get("metrics")]
    metric_text = _format_sections(used)
    period = f"，时间范围 {time_range}" if time_range else ""

    user_prompt = (
        f"研究区: {study_area}{period}\n\n"
        f"分析指标数据:\n{metric_text}"
    )

    if api_key and used:
        try:
            import requests
            last_error: Optional[Exception] = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                            "messages": [
                                {"role": "system", "content": REPORT_SYSTEM_PROMPT.format(
                                    研究区=study_area)},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.4,
                            "max_tokens": 2500,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"API 状态码 {resp.status_code}")
                    report = resp.json()["choices"][0]["message"]["content"].strip()
                    if report:
                        return _finalize(report, study_area, used,
                                         include_data_table, "deepseek")
                except Exception as e:
                    last_error = e
                    logger.warning(f"报告生成失败 (第 {attempt+1} 次): {e}")
                    if attempt < MAX_RETRIES:
                        import time
                        time.sleep(1.0 * (attempt + 1))
            logger.debug(f"报告生成全部失败, 模板降级: {last_error}")
        except ImportError:
            pass

    return _finalize(_template_report(study_area, used, period),
                     study_area, used, include_data_table, "template")


def _finalize(
    body: str,
    study_area: str,
    used: List[Dict],
    include_data_table: bool,
    method: str,
) -> Dict:
    """组装最终报告 (加标题/数据表/页脚)。"""
    report = body.strip()

    if include_data_table and used:
        table = "\n\n## 附: 指标数据表\n\n| 模块 | 指标 | 数值 |\n|------|------|------|\n"
        for sec in used:
            name = sec["name"]
            for k, v in sec.get("metrics", {}).items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                table += f"| {name} | {k} | {val} |\n"
        report += table

    report += f"\n\n---\n*本报告由 Geo AI 平台 AI 自动生成 (方法: {method})*"
    return {
        "report": report,
        "success": True,
        "method": method,
        "sections_used": len(used),
    }


def _template_report(
    study_area: str,
    used: List[Dict],
    period: str,
) -> str:
    """规则模板报告 (无 API Key 时)。"""
    lines = [f"# {study_area} 遥感综合分析报告{period}\n"]
    lines.append("## 一、执行摘要")
    lines.append("基于平台多模块遥感分析，对研究区生态状况进行综合评估。")
    lines.append("")

    lines.append("## 二、分项分析")
    for sec in used:
        name = sec["name"]
        lines.append(f"### {name}")
        metrics = sec.get("metrics", {})
        descs = []
        for k, v in list(metrics.items())[:3]:
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            descs.append(f"{k}为{val}")
        if descs:
            lines.append(f"{name}分析显示，{'、'.join(descs)}。")
        lines.append("")

    lines.append("## 三、综合结论")
    lines.append("研究区整体状况需结合多模块指标综合研判，建议补充时序数据验证趋势。")
    lines.append("")

    lines.append("## 四、对策与建议")
    lines.append("1. 建议持续监测关键生态指标（植被/水体/干旱）的时序变化；")
    lines.append("2. 对指标异常区域进行现场核查；")
    lines.append("3. 结合气象与水文数据交叉验证分析结论。")
    return "\n".join(lines)


def sections_from_sources(sources: Dict) -> List[Dict]:
    """
    从报告导出页的 collected["sources"] 转换为 sections 结构。

    参数:
        sources: 报告导出页自动采集的指标字典

    返回:
        list[dict]: [{"name": "模块名", "metrics": {...}}, ...]
    """
    sections = []

    if "water_area" in sources and sources["water_area"] > 0:
        sections.append({"name": "水体监测", "metrics": {
            "水体面积(km²)": sources["water_area"],
            "水体占比(%)": sources["water_pct"],
        }})

    if "veg_mean" in sources and sources["veg_mean"] != 0:
        sections.append({"name": "植被分析", "metrics": {
            f"{sources.get('veg_index', 'NDVI')}均值": sources["veg_mean"],
            "植被覆盖度(%)": sources["veg_coverage"],
            "植被趋势": sources.get("veg_trend", ""),
        }})

    if "change_increase" in sources:
        net = sources.get("change_increase", 0) - sources.get("change_decrease", 0)
        sections.append({"name": "变化检测", "metrics": {
            "增加面积(km²)": sources["change_increase"],
            "减少面积(km²)": sources["change_decrease"],
            "净变化(km²)": net,
        }})

    if "ai_classes" in sources:
        sections.append({"name": "AI 分类", "metrics": {
            "分类模型": sources.get("ai_model", ""),
            "类别数": sources["ai_classes"],
        }})

    if "salinity_total_ratio" in sources:
        sections.append({"name": "土壤盐渍化", "metrics": {
            "盐渍化总面积占比": sources["salinity_total_ratio"],
            "重度及以上占比": sources["salinity_severe_ratio"],
            "主导等级": sources.get("salinity_dominant", ""),
        }})

    if "lst_mean" in sources:
        sections.append({"name": "地表温度 LST", "metrics": {
            "平均温度(°C)": sources["lst_mean"],
            "最高温度(°C)": sources["lst_max"],
            "高温区占比": sources["lst_hot_ratio"],
        }})

    if "et_mean" in sources:
        sections.append({"name": "蒸散发 ET", "metrics": {
            "平均蒸散发(mm/day)": sources["et_mean"],
            "净辐射(W/m²)": sources["et_rn"],
        }})

    if "sup_oa" in sources:
        sections.append({"name": "监督分类", "metrics": {
            "总体精度OA": sources["sup_oa"],
            "Kappa": sources["sup_kappa"],
            "宏平均F1": sources["sup_f1"],
        }})

    if "bfast_n" in sources and sources.get("bfast_n", 0) > 0:
        sections.append({"name": "BFAST 断点检测", "metrics": {
            "突变事件数": sources["bfast_n"],
            "负向突变(退化)": sources["bfast_neg"],
            "正向突变(恢复)": sources["bfast_pos"],
        }})

    if "trans_change" in sources:
        sections.append({"name": "土地转移", "metrics": {
            "总变化面积(km²)": sources["trans_change"],
            "时相": f"{sources.get('trans_t1','')}→{sources.get('trans_t2','')}",
        }})

    return sections
