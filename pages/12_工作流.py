"""
智能分析工作流 — 自然语言查询 + 一键分析 + 综合报告
====================================================
支持: 自然语言描述需求 → 自动匹配模块 → 一键执行 → 生成报告
"""
import streamlit as st
import os, sys, tempfile, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from io import BytesIO
from PIL import Image
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STUDY_AREAS, COLLECTIONS
from utils.pc_data import search_images, download_multiband

st.set_page_config(page_title="工作流", page_icon="⚡", layout="wide")

# ============================================================
# 智能查询引擎 (模板匹配, 无需 LLM)
# ============================================================
QUERY_TEMPLATES = [
    {"keywords": ["ndvi", "植被", "绿洲", "覆盖"], "module": "植被分析", "page": "3_植被分析",
     "icon": "🌿", "desc": "NDVI/EVI 计算 + Sen+MK 趋势分析"},
    {"keywords": ["水体", "湖泊", "mndwi", "水面", "水库", "面积"], "module": "水体监测", "page": "2_水体监测",
     "icon": "💧", "desc": "MNDWI/AWEIsh + AI 水体分割"},
    {"keywords": ["干旱", "vci", "nddi", "spi", "缺水", "距平"], "module": "干旱监测", "page": "7_干旱监测",
     "icon": "🏜️", "desc": "多指数干旱 + 预测 + 沙漠化"},
    {"keywords": ["分类", "地物", "土地", "覆盖", "ai", "分割"], "module": "AI分类", "page": "4_AI分类",
     "icon": "🤖", "desc": "ESA/ESRI + 深度学习地物分割"},
    {"keywords": ["变化", "检测", "对比", "前后"], "module": "变化检测", "page": "5_变化检测",
     "icon": "🔄", "desc": "双时相 7 级变化分类"},
    {"keywords": ["冰川", "雪", "ndsi", "冻土", "冰", "积雪"], "module": "冰冻圈分析", "page": "8_冰冻圈分析",
     "icon": "❄️", "desc": "NDSI 雪盖 + 冰川边界 + 冻土"},
    {"keywords": ["农业", "作物", "灌溉", "cwsi", "农田", "土壤水分"], "module": "农业干旱", "page": "9_农业干旱",
     "icon": "🌾", "desc": "CWSI + 土壤水分 + 灌溉需求"},
    {"keywords": ["动画", "gif", "视频", "时序", "动态"], "module": "时序动画", "page": "10_时序动画",
     "icon": "🎬", "desc": "NDVI/水体/雪盖 GIF 动画"},
    {"keywords": ["生态", "安全", "psr", "评估", "环境"], "module": "生态评估", "page": "11_生态评估",
     "icon": "🌍", "desc": "PSR 生态安全评价"},
    {"keywords": ["报告", "导出", "汇总"], "module": "报告导出", "page": "6_报告导出",
     "icon": "📄", "desc": "HTML 综合分析报告"},
]


def parse_query(query: str) -> list:
    """解析自然语言查询, 返回匹配的分析模块列表"""
    query_lower = query.lower()
    matched = []
    for tmpl in QUERY_TEMPLATES:
        score = sum(1 for kw in tmpl["keywords"] if kw in query_lower)
        if score > 0:
            matched.append({**tmpl, "score": score})
    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


# ============================================================
# 页面主体
# ============================================================
st.title("⚡ 智能分析工作流")

tab_query, tab_wizard, tab_report = st.tabs(["💬 自然语言查询", "🧭 分步向导", "📋 综合报告"])

# ---- Tab 1: 自然语言查询 ----
with tab_query:
    st.subheader("💬 告诉我你想分析什么")
    st.caption("用自然语言描述需求, 系统自动匹配分析模块")

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        query = st.text_input("描述你的分析需求",
                             placeholder='例如: "帮我分析塔里木盆地 2025 年的植被变化和干旱情况"',
                             label_visibility="collapsed")
    with col_btn:
        analyze_btn = st.button("🔍 分析需求", type="primary")

    if analyze_btn and query:
        matched = parse_query(query)

        if matched:
            st.success(f"🎯 匹配到 **{len(matched)}** 个相关分析模块")

            # 默认研究区提取
            detected_area = None
            for area_name in STUDY_AREAS:
                if area_name in query:
                    detected_area = area_name
                    break
            if detected_area:
                st.info(f"📍 检测到研究区: **{detected_area}**")

            # 年份提取
            years = re.findall(r'(20\d{2})', query)
            if years:
                st.info(f"📅 检测到年份: {', '.join(years)}")

            st.divider()
            st.subheader("📊 推荐分析方案")

            cols = st.columns(min(3, len(matched)))
            for i, m in enumerate(matched):
                with cols[i % 3]:
                    score_bar = "█" * min(m["score"], 5) + "░" * max(0, 5 - m["score"])
                    st.markdown(f"""
                    <div style="border:1px solid #444;border-radius:8px;padding:14px;
                    background:linear-gradient(135deg, #1a1a2e, #16213e);margin-bottom:8px">
                    <div style="font-size:24px">{m['icon']}</div>
                    <div style="font-weight:700;font-size:15px;margin:6px 0">{m['module']}</div>
                    <div style="font-size:12px;color:#999;margin-bottom:6px">{m['desc']}</div>
                    <div style="font-size:10px;color:#666">匹配度: {score_bar} ({m['score']})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"进入 {m['module']}", key=f"q_{i}"):
                        st.switch_page(f"pages/{m['page']}.py")

            # 快捷: 一键设置所有匹配模块
            st.divider()
            if st.button("⚡ 一键进入首个推荐模块", type="primary"):
                st.switch_page(f"pages/{matched[0]['page']}.py")
        else:
            st.warning("🔍 未匹配到相关模块, 试试: 植被 / 水体 / 干旱 / 冰川 / 农业 / 生态")

    with st.expander("📖 查询示例"):
        st.markdown("""
        | 查询 | 匹配模块 |
        |------|---------|
        | "塔里木 NDVI 变化趋势" | 植被分析 + 干旱监测 |
        | "天山冰川积雪消融" | 冰冻圈分析 + 时序动画 |
        | "河西走廊农业缺水" | 农业干旱 + 干旱监测 |
        | "柴达木湖泊面积变化" | 水体监测 + 变化检测 |
        | "生态安全评估" | 生态评估 + 植被分析 |
        | "做一份分析报告" | 报告导出 + 植被分析 |
        """)

# ---- Tab 2: 分步向导 ----
with tab_wizard:
    st.subheader("🧭 三步完成遥感分析")

    # Step 1: 研究区 + 数据源
    st.markdown("### 📍 Step 1: 选择研究区和数据")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        default_area = st.session_state.get("selected_area", "塔里木盆地")
        if default_area not in STUDY_AREAS:
            default_area = "塔里木盆地"
        wiz_area = st.selectbox("研究区", list(STUDY_AREAS.keys()),
                               index=list(STUDY_AREAS.keys()).index(default_area))
        st.session_state["selected_area"] = wiz_area
        wiz_info = STUDY_AREAS[wiz_area]
        st.caption(f"📌 {wiz_info['description']}")
    with col_a2:
        wiz_sat = st.selectbox("数据源", list(COLLECTIONS.keys()),
                              format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            wiz_start = st.date_input("开始", date.today() - timedelta(days=365))
        with col_d2:
            wiz_end = st.date_input("结束", date.today())

    # Step 2: 选择分析模块
    st.markdown("### 🔧 Step 2: 选择分析模块")

    cols_mod = st.columns(4)
    module_choices = {}
    all_modules = [
        ("植被分析", "🌿", "NDVI/EVI + 趋势"),
        ("水体监测", "💧", "MNDWI + AI分割"),
        ("干旱监测", "🏜️", "多指数 + 预测"),
        ("AI分类", "🤖", "地物分割"),
        ("变化检测", "🔄", "双时相变化"),
        ("冰冻圈分析", "❄️", "NDSI + 冰川"),
        ("农业干旱", "🌾", "CWSI + 灌溉"),
        ("时序动画", "🎬", "GIF 动画"),
        ("生态评估", "🌍", "PSR 评价"),
    ]

    for i, (name, icon, desc) in enumerate(all_modules):
        with cols_mod[i % 4]:
            module_choices[name] = st.checkbox(f"{icon} {name}", value=i < 3,
                                              help=desc)

    selected_modules = [m[0] for m in all_modules if module_choices[m[0]]]

    # Step 3: 执行
    st.markdown("### ⚡ Step 3: 执行分析")

    if not selected_modules:
        st.warning("请至少选择一个分析模块")
    else:
        st.info(f"📋 已选择 **{len(selected_modules)}** 个模块: {', '.join(selected_modules)}")

        wiz_cloud = st.slider("云量阈值 (%)", 0, 100, 15)
        wiz_items = st.slider("影像数", 1, 12, 3)

        run_wiz = st.button("🚀 一键执行分析", type="primary")

        if run_wiz:
            bbox = wiz_info["bbox"]
            results_summary = {}

            # 搜索影像
            with st.spinner("🔍 搜索影像..."):
                try:
                    search_results = search_images(
                        bbox=bbox, start_date=wiz_start.strftime("%Y-%m-%d"),
                        end_date=wiz_end.strftime("%Y-%m-%d"),
                        collection=wiz_sat, cloud_cover_max=wiz_cloud,
                        max_items=wiz_items)
                except Exception as e:
                    st.error(f"搜索失败: {e}")
                    search_results = []

            if not search_results:
                st.warning("⚠️ 无影像")
                st.stop()

            st.success(f"✅ 找到 {len(search_results)} 景影像")

            # 下载第一景作为代表
            progress = st.progress(0)
            status = st.empty()

            status.text(f"⬇️ 下载 {search_results[0]['datetime']}...")
            try:
                tmp = os.path.join(tempfile.gettempdir(), f"wiz_{search_results[0]['id'][:12]}.tif")
                tif = download_multiband(search_results[0]["item"], tmp, collection=wiz_sat)
                if tif and os.path.exists(tif):
                    import rasterio
                    with rasterio.open(tif) as src:
                        bands = src.read().astype(np.float64)
                    if np.nanmedian(bands[1]) > 10:
                        bands /= 10000.0
                    progress.progress(0.5)

                    # 计算各项指标
                    blue, green, red, nir, swir1 = bands[0], bands[1], bands[2], bands[3], bands[4]
                    ndvi = np.clip((nir - red) / (nir + red + 1e-6), -1, 1)

                    results_summary["study_area"] = wiz_area
                    results_summary["date"] = search_results[0]["datetime"]
                    results_summary["satellite"] = wiz_sat
                    results_summary["ndvi_mean"] = float(np.nanmean(ndvi))
                    results_summary["ndvi_std"] = float(np.nanstd(ndvi))

                    # 植被
                    status.text("🌿 植被分析...")
                    veg_px = int((ndvi > 0.2).sum())
                    results_summary["vegetation_ratio"] = round(veg_px / max(ndvi.size, 1), 4)

                    # 水体 (MNDWI 简算)
                    status.text("💧 水体分析...")
                    mndwi = np.clip((green - swir1) / (green + swir1 + 1e-6), -1, 1)
                    water_px = int((mndwi > 0).sum())
                    results_summary["water_ratio"] = round(water_px / max(mndwi.size, 1), 4)

                    progress.progress(1.0)
                    status.text("✅ 分析完成")

                    try:
                        os.remove(tif)
                    except Exception:
                        pass
                else:
                    st.error("❌ 下载失败")
                    st.stop()
            except Exception as e:
                st.error(f"❌ 处理失败: {e}")
                st.stop()

            # 展示结果
            st.divider()
            st.subheader("📊 分析结果概览")

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("NDVI 均值", f"{results_summary['ndvi_mean']:.4f}")
            with col_r2:
                st.metric("植被覆盖率", f"{results_summary['vegetation_ratio']*100:.1f}%")
            with col_r3:
                st.metric("水体覆盖率", f"{results_summary['water_ratio']*100:.1f}%")
            with col_r4:
                st.metric("分析时相", results_summary["date"])

            # 各模块跳转
            st.divider()
            st.subheader("🔗 进入详细分析")
            page_map = {
                "植被分析": "3_植被分析", "水体监测": "2_水体监测",
                "干旱监测": "7_干旱监测", "AI分类": "4_AI分类",
                "变化检测": "5_变化检测", "冰冻圈分析": "8_冰冻圈分析",
                "农业干旱": "9_农业干旱", "时序动画": "10_时序动画",
                "生态评估": "11_生态评估",
            }
            btn_cols = st.columns(min(4, len(selected_modules)))
            for i, mod_name in enumerate(selected_modules):
                with btn_cols[i % 4]:
                    page = page_map.get(mod_name)
                    if page:
                        if st.button(f"进入 {mod_name}", key=f"wiz_{i}"):
                            st.switch_page(f"pages/{page}.py")

            # 保存到 session
            st.session_state["workflow_results"] = results_summary

# ---- Tab 3: 综合报告 ----
with tab_report:
    st.subheader("📋 综合报告")

    wf_results = st.session_state.get("workflow_results", None)

    if wf_results is None:
        st.info("💡 请先在「分步向导」中执行分析，结果将自动汇总到此。")
    else:
        st.success(f"✅ 报告数据来源: {wf_results['study_area']} ({wf_results['date']})")

        # 生成 HTML 报告
        report_html = f"""
        <html><head><meta charset="utf-8">
        <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; max-width:800px; margin:40px auto; color:#333; }}
        .header {{ text-align:center; border-bottom:3px solid #2ecc71; padding-bottom:20px; margin-bottom:30px; }}
        .header h1 {{ color:#1a5276; font-size:24px; }}
        .header p {{ color:#666; font-size:14px; }}
        .card {{ border:1px solid #ddd; border-radius:8px; padding:20px; margin:16px 0; background:#fafafa; }}
        .card h3 {{ color:#1a5276; margin-top:0; }}
        .metric-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
        .metric {{ background:white; border-radius:6px; padding:15px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
        .metric .value {{ font-size:28px; font-weight:700; color:#2ecc71; }}
        .metric .label {{ font-size:12px; color:#999; margin-top:4px; }}
        .footer {{ text-align:center; color:#999; font-size:12px; margin-top:40px; border-top:1px solid #eee; padding-top:20px; }}
        </style></head><body>
        <div class="header">
        <h1>🛰️ {wf_results['study_area']} 遥感分析报告</h1>
        <p>数据源: {wf_results['satellite']} | 分析日期: {wf_results['date']} | 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <div class="card">
        <h3>📊 核心指标</h3>
        <div class="metric-grid">
        <div class="metric"><div class="value">{wf_results['ndvi_mean']:.4f}</div><div class="label">NDVI 均值</div></div>
        <div class="metric"><div class="value">{wf_results['vegetation_ratio']*100:.1f}%</div><div class="label">植被覆盖率</div></div>
        <div class="metric"><div class="value">{wf_results['water_ratio']*100:.1f}%</div><div class="label">水体覆盖率</div></div>
        <div class="metric"><div class="value">{wf_results['ndvi_std']:.4f}</div><div class="label">NDVI 标准差</div></div>
        </div>
        </div>

        <div class="card">
        <h3>🎯 分析建议</h3>
        <p>基于遥感初步分析, 建议关注以下方向:</p>
        <ul>
        <li>植被时空变化趋势监测</li>
        <li>水资源分布与季节波动</li>
        <li>干旱化程度评估</li>
        <li>生态安全状态跟踪</li>
        </ul>
        <p>💡 使用平台各专业分析模块获取详细图表和统计数据。</p>
        </div>

        <div class="footer">
        <p>Geo AI 干旱区遥感智能分析平台 v1.7 | 自动生成</p>
        </div>
        </body></html>
        """

        st.download_button("📥 下载 HTML 报告", report_html.encode("utf-8"),
                          f"report_{wf_results['study_area']}_{wf_results['date']}.html",
                          "text/html")

        st.divider()
        st.subheader("📄 报告预览")
        st.html(report_html, height=500)
