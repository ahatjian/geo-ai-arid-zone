"""AI-led Geo AI research copilot.

This module turns a short research goal into a plan, an executable quick
scan, and a written scientific conclusion. It is intentionally designed to
work in two modes:

- DeepSeek API available: LLM parsing and report writing.
- No API key: deterministic module matching plus rule-based reporting.

The copilot does not replace the specialised analysis pages; it gives users
an AI-led starting point and links each step back to the platform module.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from utils.llm import query_deepseek, is_llm_available  # noqa: E402
from utils.ai_assistant import auto_analyze, build_platform_context  # noqa: E402
from utils.knowledge_base import build_knowledge_context  # noqa: E402

REQUEST_TIMEOUT = 35
MAX_RETRIES = 2

_MODULE_ACTIONS = {
    "数据浏览": "检索研究区最新低云量影像，确认数据时空覆盖",
    "水体监测": "计算 MNDWI/AWEIsh，提取水体面积与空间分布",
    "植被分析": "计算 NDVI/EVI 时序，检测绿洲植被趋势",
    "AI分类": "获取土地覆盖产品或执行深度/非监督分类",
    "变化检测": "选择双时相影像，识别土地利用与生态变化",
    "干旱监测": "计算 VCI/NDDI/TVDI 并评估干旱等级",
    "冰冻圈分析": "计算 NDSI 雪盖、冰川边界与冻土活动层",
    "农业干旱": "计算 CWSI/土壤水分，评估灌溉需求",
    "时序动画": "生成 NDVI/水体/雪盖时序动画",
    "生态评估": "基于 PSR 模型计算生态安全指数",
    "土壤盐渍化": "计算 SI/NDSI，识别盐渍化等级与范围",
    "地表温度": "反演 Landsat LST，分析热环境格局",
    "蒸散发": "基于 SEBAL 估算地表蒸散发与耗水",
    "监督分类": "采集样本并训练 RF/SVM/KNN/MLP 分类器",
    "土地转移": "计算双时相土地覆盖转移矩阵",
    "指数计算器": "按研究目标自定义波段指数",
    "矢量导出": "将栅格结果转为 GeoJSON/Shapefile/KML",
    "空间邻域分析": "对目标地物执行缓冲区与叠加空间分析",
}

_MODULE_OUTPUTS = {
    "数据浏览": "研究区可用影像清单与真彩色预览",
    "水体监测": "水体分布图、面积统计与变化结论",
    "植被分析": "NDVI/EVI 分布图、趋势显著性与突变事件",
    "AI分类": "土地覆盖分类图与各类面积统计",
    "变化检测": "变化分类图、变化面积与空间热点",
    "干旱监测": "干旱指数图、等级统计与预测结果",
    "冰冻圈分析": "雪盖/冰川/冻土分级结果与变化统计",
    "农业干旱": "作物水分胁迫图与灌溉建议",
    "时序动画": "时序动态 GIF 与变化曲线",
    "生态评估": "生态安全指数图与 PSR 分项得分",
    "土壤盐渍化": "盐渍化分级图与面积统计",
    "地表温度": "LST 分布图与热环境分级",
    "蒸散发": "ET 空间分布与能量分量统计",
    "监督分类": "分类器模型、分类图与精度评价",
    "土地转移": "土地转移矩阵与净变化方向",
    "指数计算器": "自定义指数图层与统计结果",
    "矢量导出": "GeoJSON/Shapefile/KML 矢量成果",
    "空间邻域分析": "缓冲区统计、交叉表与空间格局解读",
}


def _get_api_key() -> str:
    try:
        import streamlit as st

        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _normalize_bands(bands_data: np.ndarray) -> np.ndarray:
    bands = np.asarray(bands_data, dtype=np.float64)
    if bands.ndim != 3 or bands.shape[0] < 4:
        raise ValueError("至少需要 4 个波段")
    if np.nanmedian(bands) > 10:
        bands = bands / 10000.0
    return bands


def _safe_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-6, (a - b) / denom, 0.0)


def generate_research_plan(prompt: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Turn a research goal into a structured, executable module plan."""
    parsed = query_deepseek(prompt, api_key=api_key)
    module_items = parsed.get("modules", [])

    modules = []
    for item in module_items[:5]:
        name = item.get("name") if isinstance(item, dict) else str(item)
        if name and name not in modules:
            modules.append(name)

    if not modules:
        modules = ["数据浏览", "植被分析", "干旱监测"]

    steps = []
    for idx, name in enumerate(modules, start=1):
        steps.append(
            {
                "step": idx,
                "module": name,
                "action": _MODULE_ACTIONS.get(name, f"进入「{name}」模块执行相关分析"),
                "output": _MODULE_OUTPUTS.get(name, "对应模块的分析成果与统计结果"),
            }
        )

    knowledge = build_knowledge_context(prompt, top_k=3)
    platform_context = build_platform_context()
    return {
        "study_area": parsed.get("study_area", "塔里木盆地"),
        "year_start": parsed.get("year_start"),
        "year_end": parsed.get("year_end"),
        "modules": modules,
        "steps": steps,
        "explanation": parsed.get("explanation", ""),
        "knowledge": knowledge,
        "platform_context": platform_context,
        "method": parsed.get("method", "fallback"),
        "llm_available": is_llm_available(),
    }


def quick_scan(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a fast, dependency-light spectral scan for copilot reporting."""
    modules = modules or []
    base = auto_analyze(bands_data, satellite=satellite, api_key="")
    bands = _normalize_bands(bands_data)

    scan = {
        "ndvi_mean": base["indices"]["NDVI"]["mean"],
        "mndwi_mean": base["indices"]["MNDWI"]["mean"],
        "ndbi_mean": base["indices"]["NDBI"]["mean"],
        "water_ratio": base["indices"]["MNDWI"]["water_ratio"],
        "vegetation_ratio": base["indices"]["NDVI"]["veg_ratio"],
        "dominant_landcover": base["dominant"],
        "land_composition": base["land_composition"],
    }

    g, r, nir = bands[1], bands[2], bands[3]
    swir1 = bands[4] if bands.shape[0] > 4 else nir

    if "土壤盐渍化" in modules or "盐渍化" in str(modules):
        ndsi_salt = _safe_idx(r, swir1)
        scan["ndsi_salinity_mean"] = float(np.nanmean(ndsi_salt))
        scan["salinity_ratio"] = float(np.mean(ndsi_salt > 0))

    if "干旱监测" in modules or "干旱" in str(modules):
        ndwi = _safe_idx(g, nir)
        ndvi = _safe_idx(nir, r)
        nddi = _safe_idx(ndvi, ndwi)
        scan["nddi_mean"] = float(np.nanmean(nddi))
        scan["drought_risk_ratio"] = float(np.mean(nddi > 0.3))

    if "冰冻圈分析" in modules or "冰冻圈" in str(modules):
        ndsi_snow = _safe_idx(g, swir1)
        scan["ndsi_snow_mean"] = float(np.nanmean(ndsi_snow))
        scan["snow_ratio"] = float(np.mean(ndsi_snow > 0.4))

    return scan


def _rule_copilot_report(plan, scan):
    """Deterministic fallback report that remains useful without an LLM."""
    study_area = plan.get("study_area", "研究区")
    lines = [f"# {study_area} Geo AI 快速研究报告", ""]

    lines.append("## 一、研究目标")
    lines.append(plan.get("explanation", "基于用户目标开展多模块遥感分析。"))
    lines.append("")

    lines.append("## 二、AI 推荐方案")
    for step in plan.get("steps", []):
        lines.append(f"{step['step']}. {step['module']}：{step['action']}；预期产出：{step['output']}")
    lines.append("")

    if scan:
        lines.append("## 三、快速扫描发现")
        lines.append(
            f"- 区域主导地物为{scan['dominant_landcover']}，"
            f"植被覆盖约 {scan['vegetation_ratio']*100:.1f}%，"
            f"水体覆盖约 {scan['water_ratio']*100:.1f}%。"
        )
        lines.append(f"- NDVI 均值 {scan['ndvi_mean']:.3f}，MNDWI 均值 {scan['mndwi_mean']:.3f}。")
        if scan.get("nddi_mean") is not None:
            lines.append(f"- NDDI 均值 {scan['nddi_mean']:.3f}，干旱风险像元占比约 {scan['drought_risk_ratio']*100:.1f}%。")
        if scan.get("ndsi_salinity_mean") is not None:
            lines.append(f"- 盐分指数均值 {scan['ndsi_salinity_mean']:.3f}，疑似盐渍化像元占比约 {scan['salinity_ratio']*100:.1f}%。")

        lines.append("")
        lines.append("## 四、初步判断")
        ndvi = scan["ndvi_mean"]
        if ndvi < 0.1:
            lines.append("研究区植被覆盖极低，表现出明显的干旱区荒漠/裸地特征。")
        elif ndvi < 0.3:
            lines.append("研究区植被覆盖偏低，可能处于绿洲外围或荒漠-绿洲过渡带。")
        else:
            lines.append("研究区存在较稳定的植被覆盖，可能对应绿洲或灌溉农田。")

        if scan.get("nddi_mean") is not None and scan["nddi_mean"] > 0.3:
            lines.append("干旱风险信号较明显，建议结合 VCI/TVDI 和气象数据进一步验证。")
        if scan.get("ndsi_salinity_mean") is not None and scan["ndsi_salinity_mean"] > 0:
            lines.append("盐分指数偏高，建议核查绿洲外围是否存在次生盐渍化。")
    else:
        lines.append("## 三、数据状态")
        lines.append("本次未提供影像，已生成研究方案；建议在数据浏览页检索影像或启用离线演示模式后重新执行。")

    lines.append("")
    lines.append("## 五、下一步建议")
    lines.append("1. 按上述推荐方案完成专业模块分析；")
    lines.append("2. 对核心指标执行时间序列趋势检验；")
    lines.append("3. 将结果保存至数据下载中心并生成 HTML/PDF 报告。")
    return "\n".join(lines)

def generate_copilot_report(
    plan: Dict[str, Any],
    scan: Optional[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> str:
    """Write a concise scientific conclusion from plan and scan results."""
    if api_key is None:
        api_key = _get_api_key()

    module_text = " → ".join(plan.get("modules", []))
    plan_text = "\n".join(
        f"{s['step']}. {s['module']}：{s['action']}" for s in plan.get("steps", [])
    )

    if scan:
        metric_text = (
            f"NDVI均值={scan['ndvi_mean']:.4f}；"
            f"MNDWI均值={scan['mndwi_mean']:.4f}；"
            f"植被覆盖={scan['vegetation_ratio']*100:.1f}%；"
            f"水体覆盖={scan['water_ratio']*100:.1f}%；"
            f"主导地物={scan['dominant_landcover']}。"
        )
        if "ndsi_salinity_mean" in scan:
            metric_text += f"盐分指数均值={scan['ndsi_salinity_mean']:.4f}；盐渍化风险={scan['salinity_ratio']*100:.1f}%。"
        if "nddi_mean" in scan:
            metric_text += f"NDDI均值={scan['nddi_mean']:.4f}；干旱风险={scan['drought_risk_ratio']*100:.1f}%。"
    else:
        metric_text = "尚未提供影像，快速扫描未执行。"

    platform_context = plan.get("platform_context", "")
    user_prompt = (
        f"研究目标：{plan.get('explanation', '') or module_text}\n"
        f"研究区：{plan.get('study_area', '研究区')}\n"
        f"推荐模块：{module_text}\n"
        f"执行计划：\n{plan_text}\n"
        f"快速扫描结果：{metric_text}\n"
        f"已完成分析：{platform_context}\n"
        f"请输出研究结论与下一步建议。"
    )

    if api_key:
        try:
            import requests

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
                                {
                                    "role": "system",
                                    "content": (
                                        "你是西北干旱区遥感 Geo AI 研究助理。"
                                        "基于用户研究目标、推荐模块和快速扫描结果，"
                                        "输出 250-450 字结论：1) 核心发现；"
                                        "2) 区域生态/资源含义；3) 建议的验证与下一步。"
                                    ),
                                },
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.4,
                            "max_tokens": 700,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        text = resp.json()["choices"][0]["message"]["content"].strip()
                        if text:
                            return text
                except Exception as exc:
                    logger.warning("copilot report attempt failed: %s", exc)
                    if attempt < MAX_RETRIES:
                        import time

                        time.sleep(0.8 * (attempt + 1))
        except ImportError:
            pass

    return _rule_copilot_report(plan, scan)


def run_copilot(
    prompt: str,
    bands_data: Optional[np.ndarray] = None,
    satellite: str = "Sentinel-2 L2A",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end AI-led research copilot."""
    plan = generate_research_plan(prompt, api_key=api_key)
    scan = quick_scan(bands_data, satellite, plan.get("modules", [])) if bands_data is not None else None
    report = generate_copilot_report(plan, scan, api_key=api_key)
    return {
        "prompt": prompt,
        "plan": plan,
        "scan": scan,
        "report": report,
        "llm_available": is_llm_available(),
    }
