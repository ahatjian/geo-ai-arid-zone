# TODO: 拆分超大文件 (793行) — 按功能模式拆为独立组件
"""
报告导出页面 — 汇总分析结果，生成结构化 HTML/Word 报告
支持自动从其他页面采集数据 (v1.4)
"""

import streamlit as st
import os
import sys
import base64
from datetime import datetime
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS
from utils.error_handler import StreamlitErrorBoundary
from utils.ai_insight import generate_ai_insight, is_ai_available

st.set_page_config(page_title="报告导出", page_icon="📄", layout="wide")

st.title("📄 分析报告导出")
st.markdown("汇总各模块分析结果，生成结构化研究报告")

# ============================================
# 自动采集引擎
# ============================================

def _safe_get(key, default=None):
    """安全读取 session_state，不存在返回 default"""
    return st.session_state.get(key, default)


def _auto_collect_data():
    """
    从 session_state 自动采集其他页面的分析结果。
    返回 dict 包含各模块可用数据。
    """
    collected = {
        "water": False, "veg": False, "change": False, "ai": False,
        "sources": {},
    }

    # ---- 水体数据 (来自 2_水体监测.py) ----
    ws = _safe_get("water_stats")
    if ws and isinstance(ws, dict):
        try:
            pixel_area = ws.get("pixel_size_m", 10.0)
            collected["sources"]["water_index"] = _safe_get("water_index_type", "MNDWI")
            collected["sources"]["water_area"] = ws.get("water_area_km2", 0.0)
            collected["sources"]["water_pct"] = ws.get("water_ratio", 0.0) * 100
            collected["sources"]["water_threshold"] = ws.get("threshold", 0.0)
            collected["sources"]["water_pixel"] = pixel_area
            collected["sources"]["water_trend"] = "稳定"  # 默认值
            collected["water"] = True
        except Exception:
            pass

    # ---- 植被数据 (来自 3_植被分析.py) ----
    vs = _safe_get("veg_stats")
    if vs and isinstance(vs, dict):
        try:
            collected["sources"]["veg_index"] = _safe_get("veg_index_type", "NDVI")
            collected["sources"]["veg_mean"] = vs.get("mean", 0.0)
            collected["sources"]["veg_max"] = vs.get("max", 0.0)
            collected["sources"]["veg_min"] = vs.get("min", 0.0)
            collected["sources"]["veg_std"] = vs.get("std", 0.0)
            # 覆盖度 = 密植被 + 稀疏植被比例
            dense = vs.get("dense_veg_ratio", 0.0)
            sparse = vs.get("sparse_veg_ratio", 0.0)
            collected["sources"]["veg_coverage"] = (dense + sparse) * 100
            collected["sources"]["veg_trend"] = "不显著"
            collected["sources"]["veg_slope"] = 0.0
            collected["veg"] = True
        except Exception:
            pass

    # ---- 变化检测数据 (来自 5_变化检测.py) ----
    cd_stats = _safe_get("cd_stats")
    cd_levels = _safe_get("cd_level_counts")
    if cd_stats and isinstance(cd_stats, dict) and cd_levels:
        try:
            pixel_area = cd_stats.get("pixel_area_km2", 0.0)
            collected["sources"]["change_method"] = cd_stats.get("method", "差值法")
            collected["sources"]["change_date1"] = _safe_get("cd_t1_date", "T1")
            collected["sources"]["change_date2"] = _safe_get("cd_t2_date", "T2")

            # 从 level_counts 计算各类面积
            inc_total = sum(cd_levels.get(l, 0) for l in [1, 2, 3]) * pixel_area
            dec_total = sum(cd_levels.get(l, 0) for l in [-1, -2, -3]) * pixel_area
            stable = cd_levels.get(0, 0) * pixel_area

            collected["sources"]["change_increase"] = inc_total
            collected["sources"]["change_decrease"] = dec_total
            collected["sources"]["change_stable"] = stable

            # 检测指数来自 cd_index_t1/t2 的类型推断
            collected["sources"]["change_index"] = "NDVI"  # 默认
            collected["change"] = True
        except Exception:
            pass

    # ---- AI 分类数据 (来自 4_AI分类.py) ----
    ai_result = _safe_get("ai_class_result")
    ai_names = _safe_get("ai_class_names")
    if ai_result is not None and ai_names:
        try:
            import numpy as np
            arr = np.asarray(ai_result)
            # 计算各类像素占比
            total = arr.size
            class_areas_dict = {}
            for idx, name in enumerate(ai_names):
                cnt = np.sum(arr == idx)
                area_km2 = (cnt * 100) / 1e6  # 假设 10m 分辨率: 100m²/pixel
                class_areas_dict[name] = area_km2

            collected["sources"]["ai_model"] = "UNet + ResNet50 (AI)"
            collected["sources"]["ai_classes"] = len(ai_names)
            collected["sources"]["ai_oa"] = 0.0
            collected["sources"]["ai_kappa"] = 0.0
            collected["sources"]["ai_tilesize"] = 512
            collected["sources"]["ai_class_areas"] = class_areas_dict
            collected["ai"] = True
        except Exception:
            pass

    return collected


def _format_source_info(tab_name, available_sources):
    """生成自动采集来源的提示信息"""
    if not available_sources:
        return None
    keys_str = ", ".join(list(available_sources.keys())[:5])
    if len(available_sources) > 5:
        keys_str += f" 等 {len(available_sources)} 项"
    return f"📡 已从 **{tab_name}** 页面自动采集 {len(available_sources)} 项数据 ({keys_str})"

# ============================================
# 侧边栏 - 报告配置
# ============================================
with st.sidebar:
    st.header("📋 报告配置")

    # ---- 自动采集开关 ----
    st.subheader("🔄 数据来源")
    auto_collect = st.toggle(
        "自动从分析页面采集数据",
        value=True,
        help="开启后自动读取水体监测、植被分析、变化检测、AI分类页面的结果数据",
    )
    if not auto_collect:
        st.caption("⚠️ 手动模式：请在各 Tab 中手工输入数据")

    # 自动采集状态栏
    if auto_collect:
        collected = _auto_collect_data()
        available_count = sum([collected["water"], collected["veg"], collected["change"], collected["ai"]])
        if available_count > 0:
            st.success(f"✅ 已采集 {available_count}/4 个模块的数据")
            details = []
            if collected["water"]:
                details.append("💧 水体")
            if collected["veg"]:
                details.append("🌿 植被")
            if collected["change"]:
                details.append("🔄 变化")
            if collected["ai"]:
                details.append("🤖 AI")
            st.caption(" | ".join(details))
        else:
            st.warning("⚠️ 未检测到分析数据，请先在对应页面执行分析")

    st.divider()

    # ---- AI 智能解读 ----
    st.subheader("🤖 AI 智能解读")
    if is_ai_available():
        st.caption("🧠 DeepSeek AI 已连接，可生成专业解读")
    else:
        st.caption("未检测到 API Key，将使用规则模板解读")
    ai_insight_enabled = st.toggle(
        "生成 AI 智能解读",
        value=True,
        help="基于各模块分析指标，调用 DeepSeek 生成专业生态/环境解读，并嵌入报告",
    )

    # ---- 元数据 ----
    # 自动填充研究区
    auto_area = _safe_get("selected_area", "")
    area_names_map = {name: info for name, info in STUDY_AREAS.items()}
    area_names_list = list(area_names_map.keys())

    if auto_collect and auto_area and auto_area in area_names_list:
        default_idx = area_names_list.index(auto_area) + 1  # +1 因为索引0是"自定义"
    else:
        default_idx = 0

    # 报告标题
    default_title = f"干旱区遥感分析报告 — {datetime.now().strftime('%Y年%m月%d日')}"
    if auto_collect and auto_area:
        default_title = f"{auto_area}遥感分析报告 — {datetime.now().strftime('%Y年%m月%d日')}"

    report_title = st.text_input("报告标题", value=default_title)

    options = ["自定义"] + area_names_list
    selected_area = st.selectbox("研究区域", options, index=default_idx)

    if selected_area != "自定义":
        area_info = area_names_map[selected_area]
        st.caption(f"中心: ({area_info['center'][0]}, {area_info['center'][1]})")
        report_area = selected_area
    else:
        report_area = st.text_input("输入研究区名称", placeholder="如：塔里木盆地南部")

    # 分析日期
    analysis_date = st.date_input("分析日期", value=datetime.today())

    # 作者
    author = st.text_input("作者/单位", value="", placeholder="可选")

# ============================================
# 主面板 - 分析内容
# ============================================
st.subheader("📊 分析内容")

tab1, tab2, tab3, tab4 = st.tabs([
    "💧 水体监测结果",
    "🌿 植被分析结果",
    "🔄 变化检测结果",
    "🤖 AI 分类结果",
])

# 自动采集数据源 (如已开启)
if auto_collect:
    src = collected["sources"]
else:
    src = {}

# ---- 水体监测 ----
with tab1:
    st.markdown("### 水体监测分析")

    # 自动采集提示
    if auto_collect and collected["water"]:
        st.info(_format_source_info("水体监测", src_water := {k: v for k, v in src.items() if k.startswith("water_")}))
        disabled_w = True
    else:
        disabled_w = False

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_idx_val = src.get("water_index") if disabled_w else None
        water_index = st.selectbox(
            "水体指数", ["MNDWI", "AWEIsh"],
            index=0 if (not w_idx_val or w_idx_val == "MNDWI") else 1,
            key="w_idx", disabled=disabled_w,
        )
        if disabled_w and w_idx_val:
            water_index = w_idx_val  # 强制使用自动值

        w_area_val = src.get("water_area") if disabled_w else 0.0
        water_area = st.number_input(
            "水体面积 (km²)", min_value=0.0,
            value=float(w_area_val) if disabled_w else 0.0,
            step=0.1, key="w_area", disabled=disabled_w,
        )

        w_pct_val = src.get("water_pct") if disabled_w else 0.0
        water_pct = st.number_input(
            "水体占比 (%)", min_value=0.0, max_value=100.0,
            value=float(w_pct_val) if disabled_w else 0.0,
            step=0.1, key="w_pct", disabled=disabled_w,
        )
    with col_w2:
        w_thr_val = src.get("water_threshold") if disabled_w else 0.0
        water_threshold = st.number_input(
            "提取阈值", min_value=-1.0, max_value=1.0,
            value=float(w_thr_val) if disabled_w else 0.0,
            step=0.05, key="w_thr", disabled=disabled_w,
        )

        w_trend_val = src.get("water_trend") if disabled_w else "稳定"
        trend_options = ["稳定", "增加", "减少", "波动"]
        trend_idx = trend_options.index(w_trend_val) if w_trend_val in trend_options else 0
        water_trend = st.selectbox("变化趋势", trend_options, index=trend_idx, key="w_trend", disabled=disabled_w)
        if disabled_w:
            water_trend = w_trend_val

        w_px_val = src.get("water_pixel") if disabled_w else 10.0
        water_pixel = st.number_input(
            "像素分辨率 (m)", min_value=1.0, max_value=100.0,
            value=float(w_px_val) if disabled_w else 10.0,
            step=1.0, key="w_px", disabled=disabled_w,
        )

    water_notes = st.text_area("水体分析备注", placeholder="如：主要为季节性湖泊扩张，东南部新增水体约3.2 km²...",
                                key="w_notes", height=80)
    water_fig = st.file_uploader("上传水体分析截图/图表", type=["png", "jpg", "jpeg", "pdf"],
                                  accept_multiple_files=True, key="w_fig")
    if water_fig:
        st.caption(f"已上传 {len(water_fig)} 个附件")

# ---- 植被分析 ----
with tab2:
    st.markdown("### 植被分析")

    if auto_collect and collected["veg"]:
        st.info(_format_source_info("植被分析", {k: v for k, v in src.items() if k.startswith("veg_")}))
        disabled_v = True
    else:
        disabled_v = False

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v_idx_val = src.get("veg_index") if disabled_v else "NDVI"
        veg_index = st.selectbox(
            "植被指数", ["NDVI", "EVI"],
            index=0 if v_idx_val == "NDVI" else 1,
            key="v_idx", disabled=disabled_v,
        )
        if disabled_v:
            veg_index = v_idx_val

        veg_mean = st.number_input(
            "均值", min_value=-1.0, max_value=1.0,
            value=float(src.get("veg_mean", 0.0)) if disabled_v else 0.0,
            step=0.01, key="v_mean", disabled=disabled_v,
        )
        veg_std = st.number_input(
            "标准差", min_value=0.0, max_value=1.0,
            value=float(src.get("veg_std", 0.0)) if disabled_v else 0.0,
            step=0.01, key="v_std", disabled=disabled_v,
        )
    with col_v2:
        veg_max = st.number_input(
            "最大值", min_value=-1.0, max_value=1.0,
            value=float(src.get("veg_max", 0.0)) if disabled_v else 0.0,
            step=0.01, key="v_max", disabled=disabled_v,
        )
        veg_min = st.number_input(
            "最小值", min_value=-1.0, max_value=1.0,
            value=float(src.get("veg_min", 0.0)) if disabled_v else 0.0,
            step=0.01, key="v_min", disabled=disabled_v,
        )
        veg_coverage = st.number_input(
            "植被覆盖度 (%)", min_value=0.0, max_value=100.0,
            value=float(src.get("veg_coverage", 0.0)) if disabled_v else 0.0,
            step=0.1, key="v_cov", disabled=disabled_v,
        )

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        v_trend_val = src.get("veg_trend") if disabled_v else "不显著"
        trend_v_opts = ["不显著", "显著增加", "显著减少", "稳定"]
        trend_v_idx = trend_v_opts.index(v_trend_val) if v_trend_val in trend_v_opts else 0
        veg_trend = st.selectbox("趋势方向", trend_v_opts, index=trend_v_idx, key="v_trend", disabled=disabled_v)
        if disabled_v:
            veg_trend = v_trend_val
    with col_v4:
        veg_slope = st.number_input(
            "Sen 斜率 (×10⁻³/yr)",
            value=float(src.get("veg_slope", 0.0)) if disabled_v else 0.0,
            step=0.1, key="v_slope", disabled=disabled_v,
        )

    veg_notes = st.text_area("植被分析备注", placeholder="如：NDVI 均值 0.32，绿洲区植被状况良好，南部荒漠区稳定...",
                              key="v_notes", height=80)
    veg_fig = st.file_uploader("上传植被分析截图/图表", type=["png", "jpg", "jpeg", "pdf"],
                                accept_multiple_files=True, key="v_fig")

# ---- 变化检测 ----
with tab3:
    st.markdown("### 变化检测")

    if auto_collect and collected["change"]:
        st.info(_format_source_info("变化检测", {k: v for k, v in src.items() if k.startswith("change_")}))
        disabled_c = True
    else:
        disabled_c = False

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        c_idx_val = src.get("change_index") if disabled_c else "NDVI"
        change_index = st.selectbox(
            "检测指数", ["NDVI", "MNDWI", "EVI"],
            index=["NDVI", "MNDWI", "EVI"].index(c_idx_val) if c_idx_val in ["NDVI", "MNDWI", "EVI"] else 0,
            key="c_idx", disabled=disabled_c,
        )
        if disabled_c:
            change_index = c_idx_val

        c_method_val = src.get("change_method") if disabled_c else "差值法"
        method_opts = ["差值法", "比值法"]
        method_idx = method_opts.index(c_method_val) if c_method_val in method_opts else 0
        change_method = st.selectbox("检测方法", method_opts, index=method_idx, key="c_method", disabled=disabled_c)
        if disabled_c:
            change_method = c_method_val
    with col_c2:
        change_date1 = st.text_input(
            "T1 日期", placeholder="2020-06-15",
            value=str(src.get("change_date1", "")) if disabled_c and src.get("change_date1") else "",
            key="c_d1", disabled=disabled_c,
        )
        change_date2 = st.text_input(
            "T2 日期", placeholder="2024-06-15",
            value=str(src.get("change_date2", "")) if disabled_c and src.get("change_date2") else "",
            key="c_d2", disabled=disabled_c,
        )

    col_c3, col_c4, col_c5 = st.columns(3)
    with col_c3:
        change_increase = st.number_input(
            "增加面积 (km²)", min_value=0.0,
            value=float(src.get("change_increase", 0.0)) if disabled_c else 0.0,
            step=0.1, key="c_inc", disabled=disabled_c,
        )
    with col_c4:
        change_decrease = st.number_input(
            "减少面积 (km²)", min_value=0.0,
            value=float(src.get("change_decrease", 0.0)) if disabled_c else 0.0,
            step=0.1, key="c_dec", disabled=disabled_c,
        )
    with col_c5:
        change_stable = st.number_input(
            "稳定面积 (km²)", min_value=0.0,
            value=float(src.get("change_stable", 0.0)) if disabled_c else 0.0,
            step=0.1, key="c_stb", disabled=disabled_c,
        )

    change_notes = st.text_area("变化检测备注", placeholder="如：4年间植被增加12.5 km²，主要集中于绿洲边缘...",
                                 key="c_notes", height=80)
    change_fig = st.file_uploader("上传变化检测截图/图表", type=["png", "jpg", "jpeg", "pdf"],
                                   accept_multiple_files=True, key="c_fig")

# ---- AI 分类 ----
with tab4:
    st.markdown("### AI 地物分类")

    if auto_collect and collected["ai"]:
        st.info(_format_source_info("AI 分类", {k: v for k, v in src.items() if k.startswith("ai_")}))
        disabled_a = True
    else:
        disabled_a = False

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        a_model_val = src.get("ai_model") if disabled_a else "UNet + ResNet50"
        model_opts = ["UNet + ResNet50", "DeepLabV3+", "FPN", "指数阈值分类"]
        model_idx = model_opts.index(a_model_val) if a_model_val in model_opts else 0
        ai_model = st.selectbox("分类模型", model_opts, index=model_idx, key="a_model", disabled=disabled_a)
        if disabled_a:
            ai_model = a_model_val

        ai_classes = st.number_input(
            "分类类别数", min_value=2, max_value=20,
            value=int(src.get("ai_classes", 6)) if disabled_a else 6,
            key="a_cls", disabled=disabled_a,
        )
        ai_oa = st.number_input(
            "总体精度 OA (%)", min_value=0.0, max_value=100.0,
            value=float(src.get("ai_oa", 0.0)) if disabled_a else 0.0,
            step=0.1, key="a_oa", disabled=disabled_a,
        )
    with col_a2:
        ai_kappa = st.number_input(
            "Kappa 系数", min_value=0.0, max_value=1.0,
            value=float(src.get("ai_kappa", 0.0)) if disabled_a else 0.0,
            step=0.01, key="a_kp", disabled=disabled_a,
        )
        ai_tilesize = st.number_input(
            "Tile 大小", min_value=128, max_value=2048,
            value=int(src.get("ai_tilesize", 512)) if disabled_a else 512,
            step=128, key="a_ts", disabled=disabled_a,
        )

    st.markdown("**各类别面积统计**")
    class_cols = st.columns(3)
    class_areas = {}
    class_names = ["水体", "植被", "裸地", "建设用地", "农田", "矿区"]

    # 自动填充各类面积
    auto_class_areas = src.get("ai_class_areas", {}) if disabled_a else {}

    for i, cn in enumerate(class_names):
        with class_cols[i % 3]:
            default_val = auto_class_areas.get(cn, 0.0) if disabled_a else 0.0
            class_areas[cn] = st.number_input(
                f"{cn} (km²)", min_value=0.0,
                value=float(default_val),
                step=0.1, key=f"cls_{i}", disabled=disabled_a,
            )

    ai_notes = st.text_area("AI 分类备注", placeholder="如：整体精度 87.3%，裸地与建设用地存在一定混淆...",
                             key="a_notes", height=80)
    ai_fig = st.file_uploader("上传分类结果截图/图表", type=["png", "jpg", "jpeg", "pdf"],
                               accept_multiple_files=True, key="a_fig")

# ============================================
# 附加内容
# ============================================
st.divider()
st.subheader("📝 附加内容")

add_section = st.checkbox("添加自定义章节")

if add_section:
    section_title = st.text_input("章节标题", placeholder="如：讨论与结论")
    section_content = st.text_area("章节内容", placeholder="撰写分析讨论、结论建议等...", height=150)

# ============================================
# 报告预览与生成
# ============================================
st.divider()

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    preview = st.button("👁️ 预览报告", type="secondary")
with col_btn2:
    generate = st.button("📥 生成并下载报告", type="primary")

# ============================================
# HTML 报告模板
# ============================================
CSS_STYLE = """
<style>
    body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; background: #fff; }
    h1 { color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; font-size: 26px; }
    h2 { color: #2471a3; border-bottom: 2px solid #85c1e9; padding-bottom: 6px; margin-top: 30px; font-size: 20px; }
    h3 { color: #2e86c1; font-size: 16px; margin-top: 20px; }
    .meta { color: #666; font-size: 14px; margin-bottom: 20px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
    .kpi-card { background: #f0f8ff; border-left: 4px solid #2980b9; padding: 15px; border-radius: 5px; }
    .kpi-card .label { font-size: 12px; color: #666; text-transform: uppercase; }
    .kpi-card .value { font-size: 24px; font-weight: bold; color: #1a5276; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th, td { border: 1px solid #d4e6f1; padding: 10px 12px; text-align: left; font-size: 14px; }
    th { background: #2980b9; color: white; font-weight: 600; }
    tr:nth-child(even) { background: #f0f8ff; }
    .notes { background: #fef9e7; border-left: 4px solid #f1c40f; padding: 12px 15px; margin: 15px 0; border-radius: 3px; font-size: 14px; }
    .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 12px; text-align: center; }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }
    .auto-badge { background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 6px; }
</style>
"""


def build_report_html():
    """根据用户填写的表单数据构建完整 HTML 报告"""

    # 收集所有图片
    all_images = []
    for key, label in [
        ("w_fig", "水体监测"),
        ("v_fig", "植被分析"),
        ("c_fig", "变化检测"),
        ("a_fig", "AI 分类"),
    ]:
        files = st.session_state.get(key, [])
        for f in files:
            if f is not None:
                all_images.append((label, f))

    def b64img(file_obj):
        """将上传文件转为 base64 data URI"""
        try:
            data = file_obj.read()
            file_obj.seek(0)
            ext = file_obj.name.rsplit(".", 1)[-1].lower() if file_obj.name else "png"
            mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif") else "image/png"
            b64 = base64.b64encode(data).decode()
            return f"data:{mime};base64,{b64}"
        except Exception:
            return ""

    has_water = water_area > 0 or water_notes
    has_veg = veg_mean != 0 or veg_notes
    has_change = (change_increase > 0 or change_decrease > 0) or change_notes
    has_ai = any(v > 0 for v in class_areas.values()) or ai_notes

    sections = []

    # ---- 水体监测章节 ----
    if has_water:
        sec = '<h2>💧 水体监测分析</h2>\n'
        if auto_collect and collected["water"]:
            sec += '<p style="color:#27ae60;font-size:13px;">🤖 数据自动采集自「水体监测」页面</p>\n'
        sec += f'<div class="kpi-grid">\n'
        sec += f'<div class="kpi-card"><div class="label">水体指数</div><div class="value">{water_index}</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">水体面积</div><div class="value">{water_area:.2f} km²</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">水体占比</div><div class="value">{water_pct:.1f}%</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">变化趋势</div><div class="value">{water_trend}</div></div>\n'
        sec += f'</div>\n'
        sec += f'<table><tr><th>指标</th><th>值</th></tr>\n'
        sec += f'<tr><td>水体指数</td><td>{water_index}</td></tr>\n'
        sec += f'<tr><td>提取阈值</td><td>{water_threshold}</td></tr>\n'
        sec += f'<tr><td>水体面积</td><td>{water_area:.2f} km²</td></tr>\n'
        sec += f'<tr><td>水体占比</td><td>{water_pct:.1f}%</td></tr>\n'
        sec += f'<tr><td>像素分辨率</td><td>{water_pixel} m</td></tr>\n'
        sec += f'<tr><td>变化趋势</td><td>{water_trend}</td></tr>\n'
        sec += f'</table>\n'
        if water_notes:
            sec += f'<div class="notes"><strong>分析备注：</strong>{water_notes}</div>\n'
        for label, f in all_images:
            if label == "水体监测":
                img_src = b64img(f)
                if img_src:
                    sec += f'<img src="{img_src}" alt="{f.name}">\n'
        sections.append(sec)

    # ---- 植被分析章节 ----
    if has_veg:
        sec = '<h2>🌿 植被分析</h2>\n'
        if auto_collect and collected["veg"]:
            sec += '<p style="color:#27ae60;font-size:13px;">🤖 数据自动采集自「植被分析」页面</p>\n'
        sec += f'<div class="kpi-grid">\n'
        sec += f'<div class="kpi-card"><div class="label">{veg_index} 均值</div><div class="value">{veg_mean:.4f}</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">覆盖度</div><div class="value">{veg_coverage:.1f}%</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">趋势</div><div class="value">{veg_trend}</div></div>\n'
        sec += f'</div>\n'
        sec += f'<table><tr><th>指标</th><th>值</th></tr>\n'
        sec += f'<tr><td>植被指数</td><td>{veg_index}</td></tr>\n'
        sec += f'<tr><td>均值</td><td>{veg_mean:.4f}</td></tr>\n'
        sec += f'<tr><td>标准差</td><td>{veg_std:.4f}</td></tr>\n'
        sec += f'<tr><td>最大值</td><td>{veg_max:.4f}</td></tr>\n'
        sec += f'<tr><td>最小值</td><td>{veg_min:.4f}</td></tr>\n'
        sec += f'<tr><td>覆盖度</td><td>{veg_coverage:.1f}%</td></tr>\n'
        sec += f'<tr><td>趋势方向</td><td>{veg_trend}</td></tr>\n'
        sec += f'<tr><td>Sen 斜率</td><td>{veg_slope:.1f} × 10⁻³/yr</td></tr>\n'
        sec += f'</table>\n'
        if veg_notes:
            sec += f'<div class="notes"><strong>分析备注：</strong>{veg_notes}</div>\n'
        for label, f in all_images:
            if label == "植被分析":
                img_src = b64img(f)
                if img_src:
                    sec += f'<img src="{img_src}" alt="{f.name}">\n'
        sections.append(sec)

    # ---- 变化检测章节 ----
    if has_change:
        sec = '<h2>🔄 变化检测分析</h2>\n'
        if auto_collect and collected["change"]:
            sec += '<p style="color:#27ae60;font-size:13px;">🤖 数据自动采集自「变化检测」页面</p>\n'
        sec += f'<p style="color:#666;">时相: T1 ({change_date1}) → T2 ({change_date2}) | 方法: {change_method} ({change_index})</p>\n'
        sec += f'<div class="kpi-grid">\n'
        sec += f'<div class="kpi-card"><div class="label">增加面积</div><div class="value" style="color:#27ae60;">{change_increase:.2f} km²</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">减少面积</div><div class="value" style="color:#e74c3c;">{change_decrease:.2f} km²</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">稳定面积</div><div class="value">{change_stable:.2f} km²</div></div>\n'
        total_c = change_increase + change_decrease + change_stable
        net_c = change_increase - change_decrease
        sec += f'<div class="kpi-card"><div class="label">净变化</div><div class="value">{net_c:+.2f} km²</div></div>\n'
        sec += f'</div>\n'
        sec += f'<table><tr><th>指标</th><th>值</th></tr>\n'
        sec += f'<tr><td>检测指数</td><td>{change_index}</td></tr>\n'
        sec += f'<tr><td>检测方法</td><td>{change_method}</td></tr>\n'
        sec += f'<tr><td>T1 日期</td><td>{change_date1}</td></tr>\n'
        sec += f'<tr><td>T2 日期</td><td>{change_date2}</td></tr>\n'
        sec += f'<tr><td>增加面积</td><td>{change_increase:.2f} km²</td></tr>\n'
        sec += f'<tr><td>减少面积</td><td>{change_decrease:.2f} km²</td></tr>\n'
        sec += f'<tr><td>稳定面积</td><td>{change_stable:.2f} km²</td></tr>\n'
        sec += f'<tr><td>净变化</td><td>{net_c:+.2f} km²</td></tr>\n'
        sec += f'</table>\n'
        if change_notes:
            sec += f'<div class="notes"><strong>分析备注：</strong>{change_notes}</div>\n'
        for label, f in all_images:
            if label == "变化检测":
                img_src = b64img(f)
                if img_src:
                    sec += f'<img src="{img_src}" alt="{f.name}">\n'
        sections.append(sec)

    # ---- AI 分类章节 ----
    if has_ai:
        sec = '<h2>🤖 AI 地物分类</h2>\n'
        if auto_collect and collected["ai"]:
            sec += '<p style="color:#27ae60;font-size:13px;">🤖 数据自动采集自「AI 分类」页面</p>\n'
        sec += f'<div class="kpi-grid">\n'
        sec += f'<div class="kpi-card"><div class="label">分类模型</div><div class="value" style="font-size:16px;">{ai_model}</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">总体精度</div><div class="value">{ai_oa:.1f}%</div></div>\n'
        sec += f'<div class="kpi-card"><div class="label">Kappa</div><div class="value">{ai_kappa:.3f}</div></div>\n'
        sec += f'</div>\n'
        sec += f'<table><tr><th>类别</th><th>面积 (km²)</th></tr>\n'
        for cn in class_names:
            v = class_areas.get(cn, 0)
            if v > 0:
                sec += f'<tr><td>{cn}</td><td>{v:.2f}</td></tr>\n'
        sec += f'</table>\n'
        sec += f'<table><tr><th>参数</th><th>值</th></tr>\n'
        sec += f'<tr><td>模型</td><td>{ai_model}</td></tr>\n'
        sec += f'<tr><td>类别数</td><td>{ai_classes}</td></tr>\n'
        sec += f'<tr><td>Tile 大小</td><td>{ai_tilesize}</td></tr>\n'
        sec += f'<tr><td>OA</td><td>{ai_oa:.1f}%</td></tr>\n'
        sec += f'<tr><td>Kappa</td><td>{ai_kappa:.3f}</td></tr>\n'
        sec += f'</table>\n'
        if ai_notes:
            sec += f'<div class="notes"><strong>分析备注：</strong>{ai_notes}</div>\n'
        for label, f in all_images:
            if label == "AI 分类":
                img_src = b64img(f)
                if img_src:
                    sec += f'<img src="{img_src}" alt="{f.name}">\n'
        sections.append(sec)

    # ---- 自定义章节 ----
    if add_section and section_content:
        sec = f'<h2>📝 {section_title}</h2>\n'
        sec += f'<p style="line-height:1.8;">{section_content.replace(chr(10), "<br>")}</p>\n'
        sections.append(sec)

    # ---- AI 智能解读章节 ----
    ai_section_html = ""
    if ai_insight_enabled and sections:
        # 收集各模块指标, 生成 AI 解读
        ai_metrics = {}
        if has_water and water_area > 0:
            ai_metrics.update({
                "水体覆盖率": round(water_pct / 100.0, 4),
                "水体面积(km²)": round(water_area, 2),
            })
        if has_veg and veg_mean != 0:
            ai_metrics.update({
                "NDVI均值": round(veg_mean, 4),
                "植被覆盖率": round(veg_coverage / 100.0, 4),
            })
        if has_change and (change_increase + change_decrease) > 0:
            total_c = change_increase + change_decrease + change_stable
            if total_c > 0:
                ai_metrics.update({
                    "变化净增占比": round((change_increase - change_decrease) / total_c, 4),
                })

        if ai_metrics:
            analysis_type = []
            if has_water and water_area > 0:
                analysis_type.append("水体监测")
            if has_veg and veg_mean != 0:
                analysis_type.append("植被分析")
            if has_change and (change_increase + change_decrease) > 0:
                analysis_type.append("变化检测")
            type_str = "+".join(analysis_type) if analysis_type else "综合分析"

            insight = generate_ai_insight(
                analysis_type=type_str,
                metrics=ai_metrics,
                study_area=report_area,
                time_range=report_date,
            )
            ai_section_html = (
                f'<h2>🤖 AI 智能解读</h2>\n'
                f'<div class="ai-insight" style="background:#f0f8f4;border-left:4px solid #27ae60;'
                f'padding:16px 20px;border-radius:6px;line-height:1.9;font-size:14px;">'
                f'{insight.replace(chr(10), "<br>")}</div>\n'
                f'<p style="color:#999;font-size:12px;">本解读由 DeepSeek AI 基于上述分析指标自动生成'
                f'（{"DeepSeek AI" if is_ai_available() else "规则模板"}）</p>\n'
            )

    # 组装完整报告
    report_date = analysis_date.strftime("%Y年%m月%d日") if hasattr(analysis_date, "strftime") else str(analysis_date)
    author_line = f'<span>作者: {author}</span>' if author else ""

    body_sections = "\n".join(sections) if sections else "<p style='color:#999;text-align:center;padding:40px;'>请在左侧各 Tab 中填入分析数据后生成报告</p>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report_title}</title>
{CSS_STYLE}
</head>
<body>

<h1>{report_title}</h1>
<div class="meta">
    <span>研究区域: {report_area}</span> | 
    <span>分析日期: {report_date}</span>
    {author_line}
</div>

{body_sections}

{ai_section_html}

<div class="footer">
    <p>本报告由 Geo AI 干旱区遥感智能分析平台自动生成</p>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据来源: Sentinel-2 / Landsat</p>
</div>

</body>
</html>"""

    return html, ai_section_html


# ============================================
# 预览/生成逻辑
# ============================================
if preview or generate:
    with StreamlitErrorBoundary("报告生成", st=st, show_traceback=False):
        report_html, ai_section_html = build_report_html()

        if preview:
            st.subheader("👁️ 报告预览")
            st.html(report_html, height=800)

        if generate:
            st.subheader("📥 下载报告")
            html_bytes = report_html.encode("utf-8")
            st.download_button(
                label="⬇️ 下载 HTML 报告",
                data=html_bytes,
                file_name=f"遥感分析报告_{report_area}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True,
            )
            st.success(f"✅ 报告已生成 — 点击上方按钮下载")

            # AI 解读展示
            if ai_section_html and "AI 智能解读" in ai_section_html:
                st.divider()
                st.subheader("🤖 AI 智能解读")
                st.markdown("🧠 **DeepSeek AI** 基于本次分析指标自动生成：")
                st.markdown(
                    f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                    f"padding:16px 20px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                    f"{ai_section_html.split('<h2>🤖 AI 智能解读</h2>')[1].split('</div>')[0]}</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("📄 查看 HTML 源码"):
                st.code(report_html[:5000], language="html")
                if len(report_html) > 5000:
                    st.caption(f"... (共 {len(report_html):,} 字符)")

# ============================================
# 使用说明
# ============================================
if not preview and not generate:
    if auto_collect:
        st.info("""
        ### 📖 使用说明 v1.4 (自动采集模式)
        
        1. **自动采集已开启**: 报告页面会自动读取水体监测、植被分析、变化检测、AI分类页面的分析结果
        2. **按顺序操作**: 建议先在各分析页面完成分析 → 再进入报告页面，数据会自动填充
        3. **手动覆盖**: 如需修改自动采集的数据，关闭侧边栏的「自动采集」开关即可切换到手动模式
        4. **附加内容**: 可添加讨论、结论等自定义章节
        5. **预览 & 下载**: 点击「预览报告」查看效果，点击「生成并下载报告」导出 HTML 文件
        
        > 自动采集模式下，各 Tab 的输入框会被禁用。切换回手动模式即可编辑。
        """)
    else:
        st.info("""
        ### 📖 使用说明 (手动模式)
        
        1. **填写元数据**: 在左侧边栏设置报告标题、研究区、日期和作者
        2. **输入各模块结果**: 在四个 Tab 中填入各分析模块的输出数据（面积、统计值、趋势等）
        3. **上传截图**: 将各模块生成的可视化结果截图上传到对应 Tab
        4. **添加自定义内容**: 可添加讨论、结论等额外章节
        5. **预览 & 下载**: 点击「预览报告」查看效果，点击「生成并下载报告」导出 HTML 文件
        
        > 建议开启「自动采集」以获得最佳体验。HTML 报告可在任何浏览器中打开查看、打印或导出为 PDF。
        """)

st.divider()
st.caption("📄 报告导出页面 | Geo AI 干旱区遥感智能分析平台")
