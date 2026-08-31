"""
RAG 遥感知识库模块 — 内置指数公式/术语/方法库
==============================================
为 AI 对话提供检索增强 (Retrieval-Augmented Generation):
AI 回答遥感问题时, 先从本地知识库检索相关条目注入上下文,
减少幻觉、提高专业准确性。

知识库内容 (约 30 条):
  - 遥感指数公式与阈值 (NDVI/EVI/MNDWI/NDSI/NDDI/LST...)
  - 专业术语解释 (辐射定标/大气校正/监督分类/纹理...)
  - 干旱区研究常识 (绿洲-荒漠/水资源/盐渍化...)
  - 平台方法说明 (SEBAL/PSR/BFAST/S-G...)

检索策略 (轻量, 无需 embedding):
  关键词重叠打分 → 返回 top-K 相关条目
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 知识库条目
# ============================================================

KNOWLEDGE_ENTRIES: List[Dict] = [
    # ---- 遥感指数 ----
    {"keywords": ["ndvi", "植被指数", "归一化植被指数"],
     "content": "NDVI = (NIR - R) / (NIR + R)，值域 -1~1。>0.6 茂密植被，0.2-0.4 草地/农田，<0.1 裸地/水体。干旱区 NDVI 低是正常现象，需结合季节对比。"},
    {"keywords": ["evi", "增强植被指数"],
     "content": "EVI = 2.5×(NIR-R)/(NIR+6R-7.5B+1)，校正大气和土壤背景，高植被区不易饱和，适合绿洲农田监测。"},
    {"keywords": ["mndwi", "水体指数", "改进型归一化差异水体指数"],
     "content": "MNDWI = (G - SWIR1) / (G + SWIR1)，>0 为水体。用绿波段替代红波段，抑制建筑噪声，干旱区湖泊/水库提取首选。"},
    {"keywords": ["aweish", "阴影水体"],
     "content": "AWEIsh 通过多波段线性组合增强水体与阴影区分度，适合含山体阴影的干旱区水域提取。"},
    {"keywords": ["ndsi", "盐分指数", "盐渍化"],
     "content": "NDSI(盐分) = (R - SWIR1)/(R + SWIR1)，值越高盐渍化越重。干旱区绿洲外围次生盐渍化监测的核心指数。"},
    {"keywords": ["nddi", "干旱指数", "归一化差异干旱指数"],
     "content": "NDDI = (NDVI - NDWI)/(NDVI + NDWI)，值越大干旱程度越高，>0.5 显著干旱。结合 VCI 使用更可靠。"},
    {"keywords": ["lst", "地表温度", "热红外"],
     "content": "LST 由热红外波段反演。Landsat C2 L2 提供 ST_B10/ST_B6 产品，DN×0.00341802+149 = 开尔文。干旱区夏季 LST 40°C+ 常见。"},
    {"keywords": ["vci", "植被状态指数"],
     "content": "VCI = (NDVI - NDVI_min)/(NDVI_max - NDVI_min)×100，用历史极值归一化，消除区域差异，<30 干旱，>70 湿润。"},
    {"keywords": ["spi", "spei", "标准化降水"],
     "content": "SPI/SPEI 为气象干旱指数，SPI 基于降水概率，SPEI 考虑蒸散发。<-1.5 严重干旱，<-2 极端干旱。"},
    {"keywords": ["cwsi", "作物水分胁迫"],
     "content": "CWSI 基于冠层温度-气温差，0-1，>0.6 严重胁迫。干旱区农业灌溉决策的核心指标。"},
    {"keywords": ["tgs1", "tgsi", "荒漠化指数", "沙漠化"],
     "content": "TGSI = (R - B)/(R + B + G)，值越大土壤裸化越重；DDI = f(albedo, NDVI) 组合判断荒漠化等级。"},

    # ---- 专业术语 ----
    {"keywords": ["辐射定标", "radiometric"],
     "content": "辐射定标将 DN 转换为物理量（辐亮度/反射率）。Landsat C2: ρ=(Mρ×DN+Aρ)/cosθz。Sentinel-2 L2A 已是地表反射率。"},
    {"keywords": ["大气校正", "dos", "暗像元", "flaash"],
     "content": "大气校正去除大气散射/吸收。DOS 暗像元法假设影像含反射率≈0 的暗像元，其观测值即大气路径辐射，逐波段减去。无需大气参数，适合无观测数据的场景。FLAASH/QUAC 为专业替代。"},
    {"keywords": ["云掩膜", "scl", "qa_pixel"],
     "content": "云掩膜剔除云污染像元。Sentinel-2 用 SCL 层（8/9/10=云，3=云影），Landsat 用 QA_PIXEL 位掩码（bit3 云/bit4 云影）。分析前云掩膜可显著提升指数精度。"},
    {"keywords": ["重采样", "resample"],
     "content": "重采样将栅格对齐到统一分辨率。方法：最近邻（分类图）、双线性（指数图）、三次卷积（高质量）。Sentinel-2 20m 波段重采样到 10m 常用于多波段分析。"},
    {"keywords": ["监督分类", "random forest", "随机森林", "svm", "knn", "mlp"],
     "content": "监督分类需训练样本。平台支持随机森林/SVM/KNN/MLP 四种分类器，特征可加 NDVI/MNDWI/NDBI 指数和 GLCM 纹理（实测纹理提升 OA 20 个百分点）。"},
    {"keywords": ["非监督分类", "kmeans", "聚类"],
     "content": "KMeans 无需样本自动聚类。平台自动加入光谱指数特征栈提升可分性，并根据聚类中心光谱自动推断地物类型（植被/水体/裸地等）。"},
    {"keywords": ["纹理", "glcm", "灰度共生矩阵"],
     "content": "GLCM 灰度共生矩阵提取纹理特征：对比度/同质性/能量/相关性等 6 种，量化地物结构差异，是分类的重要辅助特征。"},
    {"keywords": ["segmentation", "分割", "unet", "深度学习"],
     "content": "深度学习分割用 UNet 等网络逐像元分类。平台水体分割基于 OmniWaterMask 预训练模型（4波段输入），无模型时自动降级为 Otsu 自适应阈值。"},

    # ---- 时序方法 ----
    {"keywords": ["sen", "mk", "mann-kendall", "theil-sen", "趋势"],
     "content": "Theil-Sen 斜率 + Mann-Kendall 检验是遥感时序趋势分析标准方法：Sen 斜率估计变化幅度，MK 检验显著性（p<0.05 显著）。"},
    {"keywords": ["s-g", "savitzky", "平滑"],
     "content": "Savitzky-Golay 滤波用滑动窗口多项式拟合平滑时序，去除云污染和传感器噪声，保留季节趋势。窗口 7-9、多项式 2-3 常用。"},
    {"keywords": ["stl", "分解", "季节"],
     "content": "STL 时序分解将序列分离为趋势+季节+残差，季节强度>0.5 表明明显季节性（干旱区绿洲植被典型）。"},
    {"keywords": ["bfast", "断点", "突变"],
     "content": "BFAST 检测时序突变事件（砍伐/火灾/退化/恢复）。流程：STL 去季节 → PELT 断点检测 → Chow 检验显著性。负向突变=植被退化。"},
    {"keywords": ["sarima", "预测", "lstm", "holt"],
     "content": "干旱预测方法：SARIMA（统计）、LSTM（深度学习，需较长时序）、Holt-Winters（指数平滑）。均支持置信区间。"},
    {"keywords": ["pca", "主成分"],
     "content": "PCA 主成分分析对相关波段去相关，前 3 分量通常浓缩 90%+ 信息，可用于分类前降维和数据压缩。"},
    {"keywords": ["合成", "mvc", "月度最大"],
     "content": "月度最大合成 MVC 取月内 NDVI 最大值，有效消除云污染（云通常降低 NDVI），是时序分析的标准预处理。"},

    # ---- 干旱区常识 ----
    {"keywords": ["绿洲", "oasis", "过渡带"],
     "content": "绿洲-荒漠过渡带是干旱区生态最敏感区域：绿洲中心 NDVI 高，向荒漠递减，过渡带宽度和梯度反映生态稳定性。"},
    {"keywords": ["塔里木盆地"],
     "content": "塔里木盆地：中国最大内陆盆地，年均降水<100mm，绿洲沿塔里木河分布，胡杨林与盐渍化是核心生态问题。"},
    {"keywords": ["柴达木", "盐湖"],
     "content": "柴达木盆地：盐湖密布，钾盐资源丰富。干涸湖床是粉尘源，水体面积变化监测是重点。"},
    {"keywords": ["河西走廊"],
     "content": "河西走廊：祁连山融水滋养的绿洲农业走廊，水资源约束下存在上游截水导致下游绿洲退化的风险。"},
    {"keywords": ["准噶尔", "天山北坡", "冰川"],
     "content": "天山北坡冰川是绿洲水源的重要补给，冰川退缩直接威胁下游水资源安全，需持续监测雪线与冰川面积。"},
    {"keywords": ["蒸散发", "et", "sebal", "能量平衡"],
     "content": "SEBAL 能量平衡法估算蒸散发：Rn = H + LE + G，ET = LE×86400/λ。需要 LST+反照率+NDVI 与气象参数，是干旱区水资源平衡的核心。"},
    {"keywords": ["psr", "生态安全", "生态评估"],
     "content": "PSR 模型：压力-状态-响应三指数加权得生态安全指数 ESI。PSI 高=生态压力大，SSI 高=状态好，RSI 高=恢复力强。"},
    {"keywords": ["盐渍化", "次生盐渍化"],
     "content": "干旱区次生盐渍化多由灌溉不当（大水漫灌/排水不畅）引起：蒸发强烈使盐分表聚。NDSI 高+植被退化是典型信号。"},
    {"keywords": ["艾丁湖", "吐鲁番"],
     "content": "吐鲁番盆地艾丁湖为中国陆地最低点（-154m），极端干旱，湖泊退缩是气候变化响应的敏感指示。"},
]

# 常用指数公式速查 (供快速展示)
INDEX_FORMULAS = {
    "NDVI": "(NIR - R) / (NIR + R)",
    "EVI": "2.5×(NIR-R)/(NIR+6R-7.5B+1)",
    "MNDWI": "(G - SWIR1) / (G + SWIR1)",
    "NDWI": "(G - NIR) / (G + NIR)",
    "NDSI盐分": "(R - SWIR1) / (R + SWIR1)",
    "NDDI": "(NDVI - NDWI) / (NDVI + NDWI)",
    "SAVI": "(NIR-R)/(NIR+R+0.5)×1.5",
    "NDBI": "(SWIR1 - NIR) / (SWIR1 + NIR)",
    "BSI": "(SWIR1+R-NIR-B)/(SWIR1+R+NIR+B)",
    "CWSI": "1 - (T_canopy - T_air)/(T_max - T_air)",
    "VCI": "(NDVI-NDVI_min)/(NDVI_max-NDVI_min)×100",
    "TGSI": "(R - B) / (R + B + G)",
}


# ============================================================
# 检索
# ============================================================

def search_knowledge(query: str, top_k: int = 3) -> List[Dict]:
    """
    关键词重叠打分检索知识库。

    参数:
        query: 用户问题
        top_k: 返回条目数

    返回:
        list[dict]: [{"keywords": ..., "content": ..., "score": ...}, ...]
    """
    if not query:
        return []

    q_lower = query.lower()
    # 分词: 提取英文词 + 中文字符片段
    tokens_en = set(re.findall(r"[a-z][a-z0-9\-]+", q_lower))
    tokens_cn = set(re.findall(r"[一-鿿]{2,6}", query))

    scored = []
    for entry in KNOWLEDGE_ENTRIES:
        score = 0
        for kw in entry["keywords"]:
            kw_lower = kw.lower()
            if kw_lower in q_lower:
                score += 3  # 完整关键词命中
            else:
                # 部分匹配: 中文关键词片段
                for tc in tokens_cn:
                    if tc in kw and len(tc) >= 2:
                        score += 1
                for te in tokens_en:
                    if te in kw_lower and len(te) >= 3:
                        score += 1
        if score > 0:
            scored.append({**entry, "score": score})

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def build_knowledge_context(query: str, top_k: int = 3) -> str:
    """
    检索并格式化知识上下文 (注入对话)。

    参数:
        query: 用户问题
        top_k: 检索条目数

    返回:
        str: 知识上下文文本 (无命中返回空串)
    """
    hits = search_knowledge(query, top_k=top_k)
    if not hits:
        return ""

    lines = ["[遥感知识库参考]"]
    for h in hits:
        lines.append(f"- {h['content']}")
    return "\n".join(lines)


def get_index_formula(index_name: str) -> Optional[str]:
    """查询指数公式 (大小写不敏感)。"""
    for name, formula in INDEX_FORMULAS.items():
        if name.lower() == index_name.lower():
            return formula
        if index_name.lower() in name.lower() and len(index_name) >= 3:
            return formula
    return None


def search_index_formula(query: str) -> Optional[str]:
    """从问题中识别指数并返回公式。"""
    for name in INDEX_FORMULAS:
        if name.lower() in query.lower() or name in query:
            return f"{name}: {INDEX_FORMULAS[name]}"
    return None


def knowledge_stats() -> Dict:
    """知识库统计。"""
    return {
        "entries": len(KNOWLEDGE_ENTRIES),
        "index_formulas": len(INDEX_FORMULAS),
        "categories": {
            "指数/术语": sum(1 for e in KNOWLEDGE_ENTRIES[:15]),
            "时序方法": sum(1 for e in KNOWLEDGE_ENTRIES[15:21]),
            "干旱区常识": len(KNOWLEDGE_ENTRIES) - 21,
        },
    }
