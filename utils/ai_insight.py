"""
AI 智能解读引擎 — DeepSeek 生成专业分析解读
============================================
将平台各模块的分析指标输入 DeepSeek，生成面向科研/汇报的专业文字解读。

核心能力:
  - generate_ai_insight(): 通用解读 (分析类型 + 指标字典 → 解读文本)
  - explain_metrics(): 核心遥感指标解释 (面向非遥感专业读者)
  - summarize_insight(): 长解读压缩为 2-3 行摘要
  - get_insight_history(): 本会话解读历史记录

安全设计:
  - 无 API Key / 请求失败 → 自动降级为规则模板解读 (fallback)
  - 不抛出异常, 所有失败路径都返回可用文本
  - 输出截断到 MAX_OUTPUT_CHARS, 防止超长响应

配置:
  DeepSeek API Key 来源与 utils/llm.py 一致 (st.secrets → 环境变量)
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---- 常量 ----
MAX_OUTPUT_CHARS = 800
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2

SYSTEM_PROMPT = """你是一名西北干旱区遥感与生态学分析专家。用户会提供遥感分析指标，请用中文输出 150-300 字的专业解读。

要求:
1. 解释指标数值反映的生态/环境含义
2. 结合西北干旱区（绿洲-荒漠、水资源约束）的背景特征进行分析
3. 给出 1-2 条可执行的监测或管理建议
4. 语气专业、客观，适合写进科研报告

只输出解读正文，不要标题、不要列表、不要客套话。"""


def _get_api_key() -> str:
    """从 st.secrets 或环境变量读取 DeepSeek API Key。"""
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception as e:
        logger.debug(f"ai_insight st.secrets error: {e}")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def is_ai_available() -> bool:
    """检查 DeepSeek AI 是否可用 (仅需有 API Key)。"""
    return bool(_get_api_key())


def _format_metrics(metrics: Dict) -> str:
    """把指标字典格式化为 prompt 中的文本块。"""
    if not metrics:
        return "（无具体指标数据）"
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        elif isinstance(value, (int, str)):
            lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def generate_ai_insight(
    analysis_type: str,
    metrics: Dict,
    study_area: str = "",
    time_range: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    生成专业分析解读。

    参数:
        analysis_type: 分析类型 (如 "植被分析"、"水体监测")
        metrics: 指标字典 {"NDVI均值": 0.35, ...}
        study_area: 研究区名称 (可选)
        time_range: 时间范围描述 (可选)
        api_key: DeepSeek API Key (None=自动获取)

    返回:
        str: 解读文本 (API 失败时返回规则模板解读)
    """
    if api_key is None:
        api_key = _get_api_key()

    location = study_area or "研究区"
    period = f"，时间范围 {time_range}" if time_range else ""
    metric_text = _format_metrics(metrics)
    user_prompt = (
        f"分析类型：{analysis_type}\n"
        f"研究区：{location}{period}\n"
        f"遥感指标：\n{metric_text}\n"
        f"请输出专业解读。"
    )

    if api_key:
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
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.5,
                            "max_tokens": 600,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"API 状态码 {resp.status_code}")
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    if content:
                        return content[:MAX_OUTPUT_CHARS]
                    raise ValueError("空响应")
                except Exception as e:
                    last_error = e
                    logger.warning(f"AI 解读请求失败 (第 {attempt + 1} 次): {e}")
                    if attempt < MAX_RETRIES:
                        import time
                        time.sleep(1.0 * (attempt + 1))
            logger.debug(f"AI 解读请求全部失败, 使用规则解读: {last_error}")
        except ImportError:
            logger.warning("requests 未安装, 使用规则解读")

    return _rule_based_insight(analysis_type, metrics, location, period)


# ------------------------------------------------------------
# 规则模板解读 (降级方案, 无 API 时也保证可用)
# ------------------------------------------------------------
def _rule_based_insight(
    analysis_type: str, metrics: Dict, location: str, period: str
) -> str:
    """基于指标阈值的规则解读, 无需网络。"""
    scope = f"{location}（时间范围 {period}）" if period else location
    parts = [f"{scope}的{analysis_type}结果显示："]
    notes = []

    # 植被类指标
    if "NDVI均值" in metrics or "ndvi_mean" in metrics:
        ndvi = metrics.get("NDVI均值", metrics.get("ndvi_mean", 0))
        if ndvi < 0.1:
            notes.append("NDVI 均值极低（<0.1），地表以裸地/荒漠为主，植被覆盖稀疏")
        elif ndvi < 0.3:
            notes.append("NDVI 均值偏低，植被覆盖度为低-中等，反映干旱区绿洲外围的稀疏植被格局")
        elif ndvi < 0.5:
            notes.append("NDVI 均值处于中等水平，表明有较稳定的植被覆盖（绿洲或灌草地）")
        else:
            notes.append("NDVI 均值较高，植被覆盖良好，可能存在高密度绿洲植被")

    if "植被覆盖率" in metrics:
        ratio = metrics.get("植被覆盖率", metrics.get("vegetation_ratio", 0))
        if isinstance(ratio, float) and ratio <= 1.0:
            pct = ratio * 100
        else:
            pct = float(ratio)
        notes.append(f"研究区植被覆盖率约 {pct:.1f}%，符合干旱区植被以绿洲斑块状分布的特征")

    # 水体类指标
    if "MNDWI均值" in metrics or "水体覆盖率" in metrics:
        water = metrics.get("水体覆盖率", metrics.get("water_ratio", None))
        if water is not None:
            if isinstance(water, float) and water <= 1.0:
                w_pct = water * 100
            else:
                w_pct = float(water)
            notes.append(f"水体覆盖率约 {w_pct:.1f}%，水资源分布格局值得持续监测")

    # 温度类指标
    if "LST均值" in metrics or "lst_mean" in metrics:
        lst = metrics.get("LST均值", metrics.get("lst_mean", 0))
        if lst > 40:
            notes.append(f"地表温度均值约 {lst:.1f}℃，存在较强热环境压力，需关注热岛/荒漠化加剧风险")
        elif lst > 25:
            notes.append(f"地表温度均值约 {lst:.1f}℃，处于干旱区夏季典型水平")

    if not notes:
        stats = "；".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(metrics.items())[:5])
        notes.append(f"核心指标为 {stats}，总体处于可分析区间")

    parts.append("；".join(notes))
    parts.append("建议结合时间序列趋势分析进一步确认演变方向，必要时与气象和水文资料交叉验证。")
    return "。".join(parts)


# ------------------------------------------------------------
# 指标解释
# ------------------------------------------------------------
INDEX_EXPLANATIONS = {
    "NDVI": "归一化植被指数，基于近红外与红波段的反射差异，反映植被覆盖和生长活力，值域 -1~1，越高植被越茂盛",
    "EVI": "增强型植被指数，校正了大气与土壤背景影响，在高植被覆盖区比 NDVI 更不易饱和",
    "MNDWI": "改进型归一化差异水体指数，用绿波段替代红波段，能有效抑制建筑物噪声，提取水体更准确",
    "AWEIsh": "阴影水体指数，通过多波段线性组合增强水体和阴影的区分度，适合复杂地物环境",
    "NDSI": "归一化差异盐分指数，基于盐分对短波红外的反射特征，值越高盐渍化程度越重",
    "NDDI": "归一化差异干旱指数，由 NDVI 与 NDWI 组合而成，值越大干旱程度越高",
    "LST": "地表温度，由热红外波段反演，是地表能量平衡和热环境状况的直接体现",
    "NDSI雪": "归一化差异积雪指数，利用可见光与短波红外的强反差识别积雪覆盖",
    "CWSI": "作物水分胁迫指数，反映作物蒸腾受阻程度，值越大缺水越严重",
}


def explain_metrics(metric_names: List[str]) -> str:
    """解释常用遥感指数含义 (面向非专业读者)。"""
    found = []
    for name in metric_names:
        key = name.strip().upper()
        for known, text in INDEX_EXPLANATIONS.items():
            if known.upper() in key or key in known.upper():
                found.append(f"**{name}**: {text}")
                break
    if not found:
        return "该指标为平台自定义计算参数，建议结合分析页面的说明理解其含义。"
    return "\n\n".join(found)


# ------------------------------------------------------------
# 摘要与历史
# ------------------------------------------------------------
def summarize_insight(insight: str, max_len: int = 120) -> str:
    """将解读文本压缩为短摘要 (截断到句号)。"""
    insight = (insight or "").strip()
    if not insight:
        return ""
    if len(insight) <= max_len:
        return insight
    cut = insight[:max_len]
    last_dot = max(cut.rfind("。"), cut.rfind(". "), cut.rfind("，"))
    if last_dot > max_len * 0.4:
        return cut[: last_dot + 1] + "…"
    return cut + "…"


def get_insight_history() -> List[Dict]:
    """返回本会话的 AI 解读历史 (列表形式)。"""
    try:
        import streamlit as st
        return list(st.session_state.get("ai_insight_history", []))
    except Exception:
        return []


# ============================================================
# Streamlit 渲染组件 (各分析页面统一接入 AI 解读)
# ============================================================

def render_ai_insight_block(
    analysis_type: str,
    metrics: Dict,
    study_area: str = "",
    time_range: str = "",
    key_suffix: str = "",
    show_button: bool = True,
    default_expanded: bool = True,
) -> Optional[str]:
    """
    在分析页面渲染统一的 AI 解读区块。

    用法 (各页面一行接入):
        from utils.ai_insight import render_ai_insight_block
        render_ai_insight_block(
            analysis_type="土壤盐渍化分析",
            metrics={"盐渍化总面积占比": 0.35, "重度及以上占比": 0.12},
            study_area=area_name,
            key_suffix="salinity",
        )

    特性:
      - 自动检查 API 可用性 (无 key 时提示)
      - 解读结果缓存于 session_state (避免重复调用)
      - LLM 输出转义 (防 XSS)
      - 显示解读来源 (DeepSeek AI / 规则模板)

    参数:
        analysis_type: 分析类型 (如 "土壤盐渍化分析")
        metrics: 指标字典 {"指标名": 值, ...}
        study_area: 研究区 (可选)
        time_range: 时间范围 (可选)
        key_suffix: 唯一标识 (避免多区块 key 冲突)
        show_button: 是否用按钮触发 (False=自动生成)
        default_expanded: 默认展开

    返回:
        str | None: 解读文本 (未触发时 None)
    """
    import streamlit as st
    import html as _html

    cache_key = f"ai_insight_{key_suffix}"
    ai_ok = is_ai_available()

    with st.expander("🤖 AI 智能解读", expanded=default_expanded):
        if not ai_ok:
            st.caption("🔑 配置 DEEPSEEK_API_KEY 后由 DeepSeek AI 解读，当前使用规则模板")
        else:
            st.caption("🧠 DeepSeek AI 基于本次分析指标生成专业解读")

        # 已缓存 → 直接显示
        if cache_key in st.session_state:
            st.markdown(
                f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                f"{_html.escape(st.session_state[cache_key])}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"来源: {'DeepSeek AI' if ai_ok else '规则模板'}")
            return st.session_state[cache_key]

        # 按钮或自动生成
        if show_button:
            clicked = st.button("🧠 生成 AI 解读", key=f"ai_btn_{key_suffix}")
            if not clicked:
                return None

        with st.spinner("AI 解读中..."):
            insight = generate_ai_insight(
                analysis_type=analysis_type,
                metrics=metrics,
                study_area=study_area,
                time_range=time_range,
            )

        st.session_state[cache_key] = insight
        st.markdown(
            f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
            f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
            f"{_html.escape(insight)}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"来源: {'DeepSeek AI' if ai_ok else '规则模板'}")
        return insight
