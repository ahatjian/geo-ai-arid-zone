"""
AI 智能助手模块 — Geo AI 平台 AI 主导核心
==========================================
三大 AI 主导能力:

  1. chat_with_assistant(): 多轮对话助手
     - 遥感知识问答 / 平台操作引导 / 分析结果解读
     - 保留对话历史 (messages 传入)

  2. auto_analyze(): AI 一键分析
     - 上传影像 → AI 自动决定分析方案 (指数/分类)
     - 自动执行 → 结构化结果 → AI 综合解读

  3. detect_anomalies(): AI 异常检测
     - 自动扫描影像中与周边显著不同的区域
     - 统计方法 (z-score / MAD) + AI 成因解读

与 utils/llm.py 的区别:
  llm.py 只做"自然语言 → 模块匹配" (结构化 JSON)
  本模块做"深度对话 + 自动分析 + 智能解读" (AI 主导决策)
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# DeepSeek 配置 (与 llm.py 保持一致)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2


def _get_api_key() -> str:
    """从 st.secrets 或环境变量读取 DeepSeek API Key。"""
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception as e:
        logger.debug(f"ai_assistant st.secrets error: {e}")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def is_ai_available() -> bool:
    """检查 DeepSeek AI 是否可用。"""
    return bool(_get_api_key())


# ============================================================
# 1. 多轮对话助手
# ============================================================

ASSISTANT_SYSTEM_PROMPT = """你是「Geo AI 西北干旱区遥感智能分析平台」的 AI 助手。你的职责：

1. 回答遥感/GIS/干旱区生态相关专业问题 (NDVI、MNDWI、LST、盐渍化、干旱指数、BFAST 等)
2. 引导用户使用平台功能: 平台有 24 个分析模块, 包括
   数据浏览/水体监测/植被分析/AI分类/变化检测/报告导出/干旱监测/冰冻圈分析/
   农业干旱/时序动画/生态评估/智能工作流/土壤盐渍化/LST/指数计算器/土地转移/
   蒸散发/监督分类/矢量导出/数据下载中心/系统状态/图像增强/大气校正/空间邻域分析
3. 解读用户的遥感分析结果 (数据驱动, 不编造)

回答要求:
- 专业、简洁、面向科研场景
- 涉及平台功能时给出具体操作路径 (如"进入「植被分析」页 → 上传多景影像 → 执行时序趋势分析")
- 不编造数据; 用户问分析结果时, 如果没提供数据, 引导其先完成分析
- 用中文回答"""


def build_platform_context() -> str:
    """
    构建平台当前分析上下文 (注入对话, 让 AI 感知用户已完成的分析)。

    从 session_state 汇总用户当前的分析结果摘要, 使 AI 能回答
    "我刚才的分析说明了什么?" 之类的问题。

    返回:
        str: 上下文文本 (无结果时返回空提示)
    """
    context_parts = []
    try:
        import streamlit as st
    except ImportError:
        return ""

    # 各模块的分析结果摘要
    ctx_sources = [
        ("植被分析", "veg_stats", lambda v: f"NDVI均值 {v.get('mean', 0):.3f}, "
                                            f"密植被占比 {v.get('dense_veg_ratio', 0)*100:.1f}%, "
                                            f"裸地占比 {v.get('bare_ratio', 0)*100:.1f}%"),
        ("水体监测", "water_stats", lambda v: f"水体面积 {v.get('water_area_km2', 0):.2f} km², "
                                              f"水体占比 {v.get('water_ratio', 0)*100:.1f}%"),
        ("土壤盐渍化", "salinity_stats", lambda v: f"盐渍化总面积占比 {v['summary'].get('total_ratio', 0)*100:.1f}%, "
                                                   f"主导等级 {v['summary'].get('dominant_level', '')}"),
        ("地表温度LST", "lst_stats", lambda v: f"平均温度 {v['summary'].get('mean_lst_c', 0):.1f}°C, "
                                               f"高温区占比 {v['summary'].get('hot_ratio', 0)*100:.1f}%"),
        ("蒸散发ET", "et_stats", lambda v: f"平均蒸散发 {v['summary'].get('mean_et', 0):.2f} mm/day"),
        ("监督分类", "supervised_stats", lambda v: f"OA {v['accuracy'].get('oa', 0)*100:.1f}%, "
                                                   f"Kappa {v['accuracy'].get('kappa', 0):.3f}"),
        ("BFAST断点", "bfast_result", lambda v: f"检测到 {v.get('n_breaks', 0)} 次突变 "
                                                f"({v.get('n_negative', 0)} 负向/ {v.get('n_positive', 0)} 正向)"),
        ("变化检测", "cd_stats", lambda v: f"T1指数均值 {v.get('idx_t1_mean', 0):.4f} → "
                                           f"T2 {v.get('idx_t2_mean', 0):.4f}, "
                                           f"方法 {v.get('method', '')}"),
        ("土地转移", "transition_stats", lambda v: f"总变化面积 {v['summary'].get('total_change_km2', 0):.2f} km², "
                                                   f"主要转移 {len(v.get('major_transitions', []))} 条"),
        ("KMeans聚类", "km_class_result", lambda v: f"{int(np.max(v)) + 1 if np.size(v) else 0} 类, "
                                                    f"推断 {', '.join(map(str, st.session_state.get('km_class_names', [])[:3]))}"),
    ]

    for label, key, fmt in ctx_sources:
        data = st.session_state.get(key)
        if data is not None:
            try:
                context_parts.append(f"【{label}】{fmt(data)}")
            except Exception:
                pass

    if not context_parts:
        return "（用户尚未完成任何分析）"

    return "用户当前已完成的分析结果摘要:\n" + "\n".join(context_parts)


def chat_with_assistant(
    messages: List[Dict],
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 1000,
    include_context: bool = False,
) -> Dict:
    """
    多轮对话 (DeepSeek)。

    参数:
        messages: [{"role": "system"|"user"|"assistant", "content": ...}, ...]
                  含历史的完整消息列表 (AI 主导的多轮交互)
        api_key: DeepSeek Key (None=自动获取)
        temperature: 采样温度
        max_tokens: 最大输出长度
        include_context: 是否注入平台当前分析上下文。
            默认 False (隐私保护: 不自动把用户分析数据发给第三方)。
            仅当用户明确要求解读其分析时由页面显式开启。

    返回:
        dict: {"reply": 助手回复, "success": bool, "error": str|None,
               "model": str, "usage": dict|None}
    """
    if api_key is None:
        api_key = _get_api_key()

    if not api_key:
        return {
            "reply": "⚠️ 未配置 DEEPSEEK_API_KEY，AI 助手不可用。\n"
                     "配置方法: 复制 .streamlit/secrets.toml.template 为 secrets.toml 并填入 Key。",
            "success": False,
            "error": "no_api_key",
            "model": None,
            "usage": None,
        }

    # 确保系统提示存在
    full_messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]

    # RAG 知识库注入: 从用户最新问题检索相关遥感知识 (提高准确性, 减少幻觉)
    try:
        from utils.knowledge_base import build_knowledge_context, search_index_formula
        latest_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if latest_user:
            rag_parts = []
            formula = search_index_formula(str(latest_user))
            if formula:
                rag_parts.append(f"[指数公式] {formula}")
            kb_ctx = build_knowledge_context(str(latest_user), top_k=3)
            if kb_ctx:
                rag_parts.append(kb_ctx)
            if rag_parts:
                full_messages.append({
                    "role": "system",
                    "content": "\n".join(rag_parts),
                })
    except ImportError:
        pass  # 知识库不可用时静默跳过
    except Exception as e:
        logger.debug(f"RAG 知识库注入失败: {e}")

    # 注入平台当前分析上下文 (AI 感知用户已完成的分析)
    if include_context:
        platform_ctx = build_platform_context()
        if platform_ctx:
            full_messages.append({"role": "system", "content": platform_ctx})

    for m in messages:
        if m.get("role") != "system":
            full_messages.append({"role": m["role"], "content": str(m.get("content", ""))})

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
                        "messages": full_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"API 状态码 {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                return {
                    "reply": data["choices"][0]["message"]["content"].strip(),
                    "success": True,
                    "error": None,
                    "model": data.get("model", "deepseek-chat"),
                    "usage": data.get("usage"),
                }
            except Exception as e:
                last_error = e
                logger.warning(f"对话请求失败 (第 {attempt + 1} 次): {e}")
                if attempt < MAX_RETRIES:
                    import time
                    time.sleep(1.0 * (attempt + 1))

        return {
            "reply": f"⚠️ AI 请求失败: {last_error}",
            "success": False,
            "error": str(last_error),
            "model": None,
            "usage": None,
        }
    except ImportError:
        return {"reply": "⚠️ requests 未安装", "success": False,
                "error": "no_requests", "model": None, "usage": None}


# ============================================================
# 2. AI 一键分析
# ============================================================

def auto_analyze(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    api_key: Optional[str] = None,
) -> Dict:
    """
    AI 一键分析: 自动决定分析方案并执行。

    流程 (AI 主导):
      1. 自动计算核心光谱指数 (NDVI/MNDWI/NDBI)
      2. 根据指数统计自动判断地物构成 (AI 决策规则)
      3. 生成结构化分析结果
      4. DeepSeek 综合解读

    参数:
        bands_data: (B, H, W) 标准 6 波段 [B,G,R,NIR,SWIR1,SWIR2]
        satellite: 卫星名 (仅用于标注)
        api_key: DeepSeek Key

    返回:
        dict: {
            "indices": {指数名: 统计},
            "land_composition": [{"name": ..., "ratio": ...}],
            "ai_assessment": str (AI 解读),
            "summary": str (结构化摘要)
        }
    """
    bands = np.asarray(bands_data, dtype=np.float64)
    if bands.shape[0] < 4:
        raise ValueError(f"至少需要 4 波段, 实际: {bands.shape[0]}")

    # 自动缩放
    if np.nanmedian(bands) > 10:
        bands = bands / 10000.0

    B, G, R, NIR = bands[0], bands[1], bands[2], bands[3]
    SWIR1 = bands[4] if bands.shape[0] > 4 else NIR

    # ---- 1. 光谱指数 ----
    def _idx(a, b):
        denom = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 1e-6, (a - b) / denom, 0.0)

    ndvi = _idx(NIR, R)
    mndwi = _idx(G, SWIR1)
    ndbi = _idx(SWIR1, NIR)

    indices = {
        "NDVI": {"mean": float(np.nanmean(ndvi)), "std": float(np.nanstd(ndvi)),
                 "veg_ratio": float(np.mean(ndvi > 0.2))},
        "MNDWI": {"mean": float(np.nanmean(mndwi)), "std": float(np.nanstd(mndwi)),
                  "water_ratio": float(np.mean(mndwi > 0))},
        "NDBI": {"mean": float(np.nanmean(ndbi)), "std": float(np.nanstd(ndbi)),
                 "built_ratio": float(np.mean(ndbi > 0))},
    }

    # ---- 2. 地物构成 (AI 决策规则, 互斥分配) ----
    # 优先级: 水体 > 植被 > 建设用地 > 裸地 (每像元只属一类)
    water_mask = mndwi > 0
    veg_mask = (~water_mask) & (ndvi > 0.2)
    built_mask = (~water_mask) & (~veg_mask) & (ndbi > 0.05)
    bare_mask = ~(water_mask | veg_mask | built_mask)

    composition = [
        {"name": "水体", "ratio": round(float(np.mean(water_mask)), 4)},
        {"name": "植被覆盖", "ratio": round(float(np.mean(veg_mask)), 4)},
        {"name": "疑似建设用地", "ratio": round(float(np.mean(built_mask)), 4)},
        {"name": "裸地/低植被", "ratio": round(float(np.mean(bare_mask)), 4)},
    ]
    composition.sort(key=lambda x: x["ratio"], reverse=True)
    dominant = composition[0]["name"]

    # ---- 3. 结构化摘要 (供 AI 解读) ----
    veg = float(np.mean(veg_mask))
    water = float(np.mean(water_mask))
    bare = float(np.mean(bare_mask))
    summary = (
        f"基于 {satellite} 影像的 AI 自动分析："
        f"研究区以「{dominant}」为主（占 {composition[0]['ratio']*100:.1f}%）。"
        f"植被覆盖 {veg*100:.1f}% (NDVI均值 {indices['NDVI']['mean']:.3f})，"
        f"水体占比 {water*100:.1f}% (MNDWI均值 {indices['MNDWI']['mean']:.3f})，"
        f"裸地占比 {bare*100:.1f}%。"
    )

    # ---- 4. DeepSeek 综合解读 ----
    insight = _auto_analyze_insight(summary, composition, api_key)

    return {
        "indices": indices,
        "land_composition": composition,
        "dominant": dominant,
        "ai_assessment": insight,
        "summary": summary,
    }


def _auto_analyze_insight(
    summary: str,
    composition: List[Dict],
    api_key: Optional[str] = None,
) -> str:
    """AI 综合解读 (无 key 时用规则解读)。"""
    if api_key is None:
        api_key = _get_api_key()

    if api_key:
        try:
            import requests
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                    "messages": [
                        {"role": "system", "content":
                         "你是干旱区遥感专家。基于影像自动分析结果，输出 120-200 字综合评估："
                         "1) 区域生态状况判断 2) 突出的环境信号 3) 建议的进一步分析。"},
                        {"role": "user", "content": summary},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 400,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"自动分析解读失败: {e}")

    # 规则降级
    top = composition[0]
    second = composition[1] if len(composition) > 1 else None
    text = f"AI 自动评估：区域以{top['name']}为主（{top['ratio']*100:.1f}%）。"
    if second and second["ratio"] > 0.1:
        text += f"其次为{second['name']}（{second['ratio']*100:.1f}%）。"
    text += "建议结合时序趋势与干旱指数进行深入分析。"
    return text


# ============================================================
# 3. AI 异常检测
# ============================================================

def detect_anomalies(
    band: np.ndarray,
    method: str = "local",
    threshold: float = 3.0,
    min_cluster: int = 16,
    kernel: int = 7,
) -> Dict:
    """
    AI 异常检测: 自动扫描影像中与周边显著不同的区域。

    方法:
      "local": 局部异常 — 与邻域中位数比较 (对全局分布不敏感,
               最适合遥感异常检测: 突变/孤立斑块)
      "mad": 全局 MAD (中位数绝对偏差, 稳健)
      "zscore": 全局 z-score (常规)

    参数:
        band: (H, W) 指标数组 (NDVI/LST/盐分等)
        method: "local" | "mad" | "zscore"
        threshold: 异常阈值 (MAD/z 倍数或局部偏离倍数)
        min_cluster: 最小异常像元簇 (过滤碎斑)
        kernel: 局部邻域大小 (local 方法)

    返回:
        dict: {
            "anomaly_mask": (H, W) bool 异常区域,
            "anomaly_ratio": float 异常占比,
            "n_clusters": int 异常簇数,
            "anomaly_stats": {"high": 高值异常占比, "low": 低值异常占比},
            "method": str
        }
    """
    band = np.asarray(band, dtype=np.float64)
    valid = band[np.isfinite(band)]
    if valid.size < 100:
        return {
            "anomaly_mask": np.zeros(band.shape, dtype=bool),
            "anomaly_ratio": 0.0, "n_clusters": 0,
            "anomaly_stats": {"high": 0.0, "low": 0.0}, "method": method,
        }

    from scipy import ndimage

    if method == "local":
        # 局部异常: 与邻域中位数比较
        kernel = max(3, int(kernel) | 1)
        local_median = ndimage.median_filter(band, size=kernel, mode="reflect")
        diff = band - local_median
        # 局部尺度估计: 邻域绝对偏离的中位数 (局部 MAD)
        local_mad = ndimage.median_filter(np.abs(diff), size=kernel, mode="reflect")
        local_mad = np.maximum(local_mad, 1e-9)
        z = diff / local_mad
        high_mask = z > threshold
        low_mask = z < -threshold
        label = f"局部异常 (邻域{kernel}×{kernel}, 阈值 {threshold}×)"
    elif method == "mad":
        median = np.nanmedian(valid)
        mad = np.nanmedian(np.abs(valid - median))
        if mad < 1e-9:
            mad = np.nanstd(valid) + 1e-9
        high_mask = (band - median) > threshold * mad
        low_mask = (median - band) > threshold * mad
        label = f"MAD (阈值 {threshold}×)"
    else:
        mean = np.nanmean(valid)
        std = np.nanstd(valid)
        if std < 1e-9:
            std = 1e-9
        high_mask = (band - mean) > threshold * std
        low_mask = (mean - band) > threshold * std
        label = f"Z-score (阈值 {threshold}σ)"

    anomaly = high_mask | low_mask

    # 连通域分析 (过滤碎斑)
    labeled, n = ndimage.label(anomaly, structure=np.ones((3, 3)))
    sizes = ndimage.sum(anomaly, labeled, range(1, n + 1))
    keep = np.zeros_like(anomaly)
    for i, s in enumerate(sizes, start=1):
        if s >= min_cluster:
            keep |= labeled == i
    anomaly = keep
    n_clusters = int(ndimage.label(anomaly)[1])

    total = float(np.isfinite(band).sum())
    return {
        "anomaly_mask": anomaly,
        "anomaly_ratio": round(float(anomaly.sum()) / max(total, 1), 6),
        "n_clusters": n_clusters,
        "anomaly_stats": {
            "high": round(float(high_mask.sum()) / max(total, 1), 6),
            "low": round(float(low_mask.sum()) / max(total, 1), 6),
        },
        "method": label,
    }


def anomaly_insight(
    band: np.ndarray,
    anomaly_result: Dict,
    band_name: str = "指标",
    api_key: Optional[str] = None,
) -> str:
    """异常区域 AI 成因解读。"""
    if api_key is None:
        api_key = _get_api_key()

    if api_key:
        try:
            import requests
            band = np.asarray(band, dtype=np.float64)
            vals = band[np.isfinite(band)]
            anomaly_vals = band[anomaly_result["anomaly_mask"]]
            context = (
                f"对「{band_name}」影像做异常检测："
                f"发现 {anomaly_result['n_clusters']} 个异常簇 (占 {anomaly_result['anomaly_ratio']*100:.2f}%)。"
                f"全图均值 {np.nanmean(vals):.4f}，"
                f"异常区均值 {np.nanmean(anomaly_vals):.4f} (差异 {np.nanmean(anomaly_vals)-np.nanmean(vals):+.4f})。"
                f"方法: {anomaly_result['method']}。"
            )
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                    "messages": [
                        {"role": "system", "content":
                         "你是干旱区遥感异常检测专家。基于检测结果，判断异常的可能成因"
                         "(如水体变化/火灾迹地/盐渍化/云影残留/传感器噪声)，"
                         "输出 100-180 字，给出验证建议。"},
                        {"role": "user", "content": context},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 300,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"异常解读失败: {e}")

    # 规则降级
    return (
        f"检测到 {anomaly_result['n_clusters']} 个异常区域（占 {anomaly_result['anomaly_ratio']*100:.2f}%），"
        f"异常区{band_name}与全图均值差异显著。建议叠加土地利用图核验成因，"
        "并检查是否包含云影或传感器噪声。"
    )
