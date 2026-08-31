"""
AI 影像视觉理解与质量诊断模块
================================
两大 AI 增强能力:

  1. vision_describe(): AI 影像视觉理解 (看图说话)
     光谱指纹提取 (波段统计/指数/纹理/地物构成) → LLM 生成影像综合描述,
     模拟 AI "看懂"遥感影像。

  2. quality_diagnose(): AI 数据质量诊断
     云覆盖/噪声/异常像元/可用像元自动评估 → 质量评级 + AI 解读,
     指导数据是否可用、如何处理。
"""

import os
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


def _get_api_key() -> str:
    """从 st.secrets 或环境变量读取 DeepSeek API Key。"""
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception as e:
        logger.debug(f"ai_vision st.secrets error: {e}")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _normalize(bands_data: np.ndarray) -> np.ndarray:
    """DN → 反射率 (0-1)。"""
    bands = np.asarray(bands_data, dtype=np.float64)
    if np.nanmedian(bands) > 10:
        bands = bands / 10000.0
    return bands


def _safe_idx(a, b):
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-6, (a - b) / denom, 0.0)


# ============================================================
# 光谱指纹
# ============================================================

def image_fingerprint(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
) -> Dict:
    """
    提取影像光谱指纹: 波段统计 + 指数 + 纹理 + 地物构成。

    参数:
        bands_data: (B, H, W) 标准 6 波段 [B,G,R,NIR,SWIR1,SWIR2]
        satellite: 卫星名

    返回:
        dict: 结构化指纹
    """
    bands = _normalize(bands_data)
    if bands.shape[0] < 4:
        raise ValueError("至少需要 4 波段")

    B, G, R, NIR = bands[0], bands[1], bands[2], bands[3]
    SWIR1 = bands[4] if bands.shape[0] > 4 else NIR
    SWIR2 = bands[5] if bands.shape[0] > 5 else SWIR1

    ndvi = _safe_idx(NIR, R)
    mndwi = _safe_idx(G, SWIR1)
    ndbi = _safe_idx(SWIR1, NIR)
    ndsi = _safe_idx(G, SWIR2)

    # 波段统计
    band_names = ["蓝", "绿", "红", "近红外", "短波红外1", "短波红外2"]
    spectral = {}
    for i, bn in enumerate(band_names[:bands.shape[0]]):
        spectral[bn] = {
            "mean": round(float(np.nanmean(bands[i])), 4),
            "std": round(float(np.nanstd(bands[i])), 4),
        }

    # 指数
    indices = {
        "NDVI": round(float(np.nanmean(ndvi)), 4),
        "MNDWI": round(float(np.nanmean(mndwi)), 4),
        "NDBI": round(float(np.nanmean(ndbi)), 4),
        "NDSI": round(float(np.nanmean(ndsi)), 4),
    }

    # 纹理粗糙度
    from scipy import ndimage
    nir_smooth = ndimage.uniform_filter(NIR, size=5)
    texture_roughness = round(float(np.nanstd(NIR - nir_smooth)), 4)

    # 地物构成 (互斥)
    water_mask = mndwi > 0
    veg_mask = (~water_mask) & (ndvi > 0.2)
    built_mask = (~water_mask) & (~veg_mask) & (ndbi > 0.05)
    bare_mask = ~(water_mask | veg_mask | built_mask)
    composition = {
        "水体": round(float(np.mean(water_mask)), 4),
        "植被": round(float(np.mean(veg_mask)), 4),
        "建设用地": round(float(np.mean(built_mask)), 4),
        "裸地": round(float(np.mean(bare_mask)), 4),
    }

    return {
        "satellite": satellite,
        "spectral": spectral,
        "indices": indices,
        "texture_roughness": texture_roughness,
        "composition": composition,
    }


# ============================================================
# AI 视觉理解 (看图说话)
# ============================================================

def vision_describe(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    api_key: Optional[str] = None,
) -> Dict:
    """
    AI 影像视觉理解: 基于光谱指纹生成影像综合描述。

    流程: 指纹提取 → LLM "看图"描述 (景观类型/生态状况/空间特征)。

    返回:
        dict: {"description": AI 描述, "fingerprint": 指纹, "success": bool}
    """
    fp = image_fingerprint(bands_data, satellite)
    if api_key is None:
        api_key = _get_api_key()

    comp_desc = "、".join(
        f"{k} {v*100:.1f}%" for k, v in
        sorted(fp["composition"].items(), key=lambda x: -x[1]) if v > 0.01
    )
    context = (
        f"卫星: {satellite}\n"
        f"光谱均值: " + ", ".join(f"{k}={v['mean']}" for k, v in fp["spectral"].items()) + "\n"
        f"指数: NDVI={fp['indices']['NDVI']}, MNDWI={fp['indices']['MNDWI']}, "
        f"NDBI={fp['indices']['NDBI']}, NDSI={fp['indices']['NDSI']}\n"
        f"纹理粗糙度: {fp['texture_roughness']}\n"
        f"地物构成: {comp_desc}"
    )

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
                         "你是遥感影像解译专家。基于影像的光谱指纹数据，像'看图'一样描述这幅遥感影像："
                         "1) 景观类型判断（绿洲/荒漠/水域/农田/城市等） 2) 生态状况 3) 突出的空间特征。"
                         "输出 120-200 字，专业、具象。"},
                        {"role": "user", "content": context},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 400,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return {
                    "description": resp.json()["choices"][0]["message"]["content"].strip(),
                    "fingerprint": fp,
                    "success": True,
                }
        except Exception as e:
            logger.warning(f"视觉理解失败: {e}")

    # 规则降级
    top = sorted(fp["composition"].items(), key=lambda x: -x[1])[0]
    desc = (
        f"该影像以{top[0]}为主（{top[1]*100:.1f}%），"
        f"NDVI={fp['indices']['NDVI']:.3f}，MNDWI={fp['indices']['MNDWI']:.3f}。"
    )
    if fp["indices"]["NDVI"] < 0.15:
        desc += "植被覆盖稀疏，干旱特征明显。"
    elif fp["indices"]["NDVI"] > 0.35:
        desc += "植被覆盖良好，可能存在绿洲或农田。"
    return {"description": desc, "fingerprint": fp, "success": False}


# ============================================================
# AI 数据质量诊断
# ============================================================

def quality_diagnose(
    bands_data: np.ndarray,
    satellite: str = "Sentinel-2 L2A",
    api_key: Optional[str] = None,
) -> Dict:
    """
    AI 影像质量诊断: 云/噪声/异常像元评估 + 质量评级 + AI 解读。

    指标:
      cloud_ratio  云覆盖估计 (亮像元 + 低 NIR)
      noise_level  噪声水平 (中值滤波差异)
      anomaly_ratio 异常像元占比 (局部 MAD)
      usable_ratio 可用像元占比

    返回:
        dict: {"metrics", "ai_diagnosis", "grade", "score"}
    """
    bands = _normalize(bands_data)
    if bands.shape[0] < 4:
        raise ValueError("至少需要 4 波段")

    B, G, R, NIR = bands[0], bands[1], bands[2], bands[3]
    from scipy import ndimage

    # 1. 云覆盖: 宽谱高反射 (云在可见光+NIR 均高反射)
    brightness = (B + G + R) / 3.0
    cloud_ratio = float(np.mean((brightness > 0.4) & (NIR > 0.35)))

    # 2. 噪声: 3×3 中值滤波差异
    nir_smooth = ndimage.median_filter(NIR, size=3)
    noise = float(np.nanstd(NIR - nir_smooth))

    # 3. 异常像元: 局部 MAD
    local_median = ndimage.median_filter(NIR, size=7)
    local_mad = ndimage.median_filter(np.abs(NIR - local_median), size=7)
    local_mad = np.maximum(local_mad, 1e-9)
    z = (NIR - local_median) / local_mad
    anomaly_ratio = float(np.mean(np.abs(z) > 4.0))

    # 4. 可用像元
    usable_ratio = float(np.mean(np.isfinite(bands).all(axis=0)))

    metrics = {
        "cloud_ratio": round(cloud_ratio, 4),
        "noise_level": round(noise, 4),
        "anomaly_ratio": round(anomaly_ratio, 4),
        "usable_ratio": round(usable_ratio, 4),
    }

    # 质量评级
    score = 100
    if cloud_ratio > 0.2:
        score -= 30
    elif cloud_ratio > 0.1:
        score -= 15
    if noise > 0.02:
        score -= 15
    elif noise > 0.01:
        score -= 8
    if anomaly_ratio > 0.02:
        score -= 10
    if usable_ratio < 0.9:
        score -= 10
    grade = "优" if score >= 85 else ("良" if score >= 70 else ("中" if score >= 55 else "差"))

    return {
        "metrics": metrics,
        "ai_diagnosis": _quality_insight(metrics, grade, satellite, api_key),
        "grade": grade,
        "score": score,
    }


def _quality_insight(
    metrics: Dict,
    grade: str,
    satellite: str,
    api_key: Optional[str] = None,
) -> str:
    """质量诊断 AI 解读 (无 Key 时规则降级)。"""
    if api_key is None:
        api_key = _get_api_key()

    context = (
        f"影像质量诊断 ({satellite}): 云覆盖 {metrics['cloud_ratio']*100:.1f}%, "
        f"噪声水平 {metrics['noise_level']:.4f}, 异常像元 {metrics['anomaly_ratio']*100:.2f}%, "
        f"可用像元 {metrics['usable_ratio']*100:.1f}%, 综合评级 {grade}。"
    )

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
                         "你是遥感数据质量评估专家。基于诊断指标输出 80-150 字："
                         "1) 数据可用性结论 2) 主要质量问题 3) 处理建议（云掩膜/去噪/重选时相）。"},
                        {"role": "user", "content": context},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 250,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"质量解读失败: {e}")

    if grade == "优":
        return f"影像质量评级「优」，云覆盖低（{metrics['cloud_ratio']*100:.1f}%），噪声可控，适合直接分析。"
    if grade == "差":
        return (f"影像质量评级「差」，云覆盖达 {metrics['cloud_ratio']*100:.1f}%，"
                "建议重新选择低云量时相，或先执行云掩膜预处理（平台「大气校正」模块）。")
    return (f"影像质量评级「{grade}」，云覆盖 {metrics['cloud_ratio']*100:.1f}%、"
            f"异常像元 {metrics['anomaly_ratio']*100:.2f}%。建议云掩膜后使用。")
