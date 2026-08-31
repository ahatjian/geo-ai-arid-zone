"""
智能分析工作流 — 自然语言查询 + 一键分析 + 综合报告
====================================================
支持: 自然语言描述需求 → 自动匹配模块 → 一键执行 → 生成报告
"""
import streamlit as st
import os, sys, tempfile, numpy as np, pandas as pd
import html
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from io import BytesIO
from PIL import Image
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.pc_data import search_images, download_multiband

st.set_page_config(page_title="工作流", page_icon="⚡", layout="wide")

# ============================================================
# 智能查询引擎 (LLM + 模板降级)
# ============================================================
from utils.llm import query_deepseek, fallback_parse, is_llm_available

def parse_query(query: str) -> list:
    """解析自然语言查询, 返回匹配的分析模块列表"""
    if is_llm_available():
        try:
            result = query_deepseek(query)
            modules = result.get("modules", [])
            if modules:
                # LLM 返回的模块可能只有 name, 补齐完整信息
                enriched = []
                for m in modules:
                    if isinstance(m, dict) and "page" in m:
                        enriched.append(m)
                    else:
                        enriched.append({"name": str(m), "icon": "🤖", "desc": "LLM 推荐", "page": "", "score": 5})
                return enriched
        except Exception:
            pass
    # 降级到模板匹配
    result = fallback_parse(query)
    return result.get("modules", [])

# LLM 状态提示
LLM_AVAILABLE = is_llm_available()


# ============================================================
# 页面主体
# ============================================================
st.title("⚡ 智能分析工作流")

tab_query, tab_wizard, tab_report = st.tabs(["💬 自然语言查询", "🧭 分步向导", "📋 综合报告"])

# ---- Tab 1: 自然语言查询 ----
with tab_query:
    st.subheader("💬 告诉我你想分析什么")
    if LLM_AVAILABLE:
        st.caption("🧠 DeepSeek AI 已激活 — 智能理解你的分析需求")
    else:
        st.caption("关键词匹配模式 — 设置 DEEPSEEK_API_KEY 启用 AI")

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
                    score = m.get("score", 1)
                    score_bar = "█" * min(score, 5) + "░" * max(0, 5 - score)
                    # LLM 返回的模块名/描述视为不可信, 转义后渲染
                    m_name_esc = html.escape(str(m.get("name", "")))
                    m_desc_esc = html.escape(str(m.get("desc", "")))
                    st.markdown(f"""
                    <div style="border:1px solid #444;border-radius:8px;padding:14px;
                    background:linear-gradient(135deg, #1a1a2e, #16213e);margin-bottom:8px">
                    <div style="font-size:24px">{m['icon']}</div>
                    <div style="font-weight:700;font-size:15px;margin:6px 0">{m_name_esc}</div>
                    <div style="font-size:12px;color:#999;margin-bottom:6px">{m_desc_esc}</div>
                    <div style="font-size:10px;color:#666">匹配度: {score_bar} ({m['score']})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"进入 {m['name']}", key=f"q_{i}"):
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
        use_custom = st.checkbox("自定义 AOI (bbox/GeoJSON)", value=False, key="wf_custom")
        if use_custom:
            from utils.aoi import parse_geojson_bbox
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                lon_min = st.number_input("西经", value=80.0, format="%.2f", key="wf_lon_min")
                lat_min = st.number_input("南纬", value=38.0, format="%.2f", key="wf_lat_min")
            with col_b2:
                lon_max = st.number_input("东经", value=90.0, format="%.2f", key="wf_lon_max")
                lat_max = st.number_input("北纬", value=44.0, format="%.2f", key="wf_lat_max")
            wiz_bbox = [lon_min, lat_min, lon_max, lat_max]
            wiz_area = "自定义AOI"
            geojson_file = st.file_uploader("或上传 GeoJSON", type=["geojson", "json"], key="wf_geojson")
            if geojson_file is not None:
                try:
                    wiz_bbox, _ = parse_geojson_bbox(geojson_file.getvalue())
                    st.success("✅ 已解析 GeoJSON 边界")
                except Exception as e:
                    st.error(f"GeoJSON 解析失败: {e}")
        else:
            default_area = st.session_state.get("selected_area", "塔里木盆地")
            if default_area not in STUDY_AREAS:
                default_area = "塔里木盆地"
            wiz_area = st.selectbox("研究区", list(STUDY_AREAS.keys()),
                                   index=list(STUDY_AREAS.keys()).index(default_area))
            st.session_state["selected_area"] = wiz_area
            wiz_info = STUDY_AREAS[wiz_area]
            wiz_bbox = wiz_info["bbox"]
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
        ("土壤盐渍化", "🧂", "SI/NDSI + 5级评估"),
        ("地表温度", "🌡️", "LST 反演 + 热环境"),
        ("蒸散发", "💨", "SEBAL 能量平衡"),
        ("监督分类", "🎯", "自定义样本训练"),
        ("土地转移", "🔀", "双时相转移矩阵"),
        ("指数计算器", "🧮", "自定义波段运算"),
        ("矢量导出", "🗺️", "GeoJSON/Shapefile/KML"),
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
            bbox = wiz_bbox
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
                    blue, green, red, nir, swir1, swir2 = bands[0], bands[1], bands[2], bands[3], bands[4], bands[5]
                    ndvi = np.clip((nir - red) / (nir + red + 1e-6), -1, 1)
                    mndwi = np.clip((green - swir1) / (green + swir1 + 1e-6), -1, 1)
                    evi = np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1e-6), -1, 1)
                    ndsi_salt = np.clip((red - swir1) / (red + swir1 + 1e-6), -1, 1)
                    nddi = np.clip((ndvi - mndwi) / (ndvi + mndwi + 1e-6), -1, 1)

                    results_summary["study_area"] = wiz_area
                    results_summary["date"] = search_results[0]["datetime"]
                    results_summary["satellite"] = wiz_sat
                    results_summary["ndvi_mean"] = float(np.nanmean(ndvi))
                    results_summary["ndvi_std"] = float(np.nanstd(ndvi))

                    # 按勾选模块动态计算 (向导快速评估)
                    progress.progress(0.6)
                    wiz_metrics = {}

                    # 植被分析
                    if "植被分析" in selected_modules:
                        status.text("🌿 植被分析...")
                        veg_px = int((ndvi > 0.2).sum())
                        results_summary["vegetation_ratio"] = round(veg_px / max(ndvi.size, 1), 4)
                        wiz_metrics["植被覆盖率"] = results_summary["vegetation_ratio"]
                        results_summary["evi_mean"] = float(np.nanmean(evi))
                        wiz_metrics["EVI 均值"] = results_summary["evi_mean"]

                    # 水体监测
                    if "水体监测" in selected_modules:
                        status.text("💧 水体分析...")
                        water_px = int((mndwi > 0).sum())
                        results_summary["water_ratio"] = round(water_px / max(mndwi.size, 1), 4)
                        wiz_metrics["水体覆盖率"] = results_summary["water_ratio"]

                    # 土壤盐渍化 (NDSI 盐分快速评估)
                    if "土壤盐渍化" in selected_modules:
                        status.text("🧂 盐渍化评估...")
                        salt_px = int((ndsi_salt > 0).sum())
                        results_summary["salinity_ratio"] = round(salt_px / max(ndsi_salt.size, 1), 4)
                        results_summary["ndsi_mean"] = float(np.nanmean(ndsi_salt))
                        wiz_metrics["NDSI 盐分均值"] = results_summary["ndsi_mean"]
                        wiz_metrics["疑似盐渍化占比"] = results_summary["salinity_ratio"]

                    # 干旱监测 (NDDI 快速评估)
                    if "干旱监测" in selected_modules:
                        status.text("🏜️ 干旱评估...")
                        dry_px = int((nddi > 0.3).sum())
                        results_summary["nddi_mean"] = float(np.nanmean(nddi))
                        results_summary["dry_ratio"] = round(dry_px / max(nddi.size, 1), 4)
                        wiz_metrics["NDDI 均值"] = results_summary["nddi_mean"]
                        wiz_metrics["干旱风险占比"] = results_summary["dry_ratio"]

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

            # 按勾选模块动态展示指标
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("NDVI 均值", f"{results_summary['ndvi_mean']:.4f}")
            if "植被分析" in selected_modules:
                with metric_cols[1]:
                    st.metric("植被覆盖率", f"{results_summary['vegetation_ratio']*100:.1f}%")
            if "水体监测" in selected_modules:
                with metric_cols[2]:
                    st.metric("水体覆盖率", f"{results_summary['water_ratio']*100:.1f}%")
            with metric_cols[3]:
                st.metric("分析时相", results_summary["date"])

            # 其余勾选模块的指标展示
            extra_metrics = {k: v for k, v in wiz_metrics.items() if k not in ("植被覆盖率", "水体覆盖率")}
            if extra_metrics:
                extra_cols = st.columns(min(4, len(extra_metrics)))
                for i, (k, v) in enumerate(extra_metrics.items()):
                    with extra_cols[i % 4]:
                        if isinstance(v, float) and v <= 1.0:
                            st.metric(k, f"{v:.4f}")
                        else:
                            st.metric(k, f"{v}")

            # 提示: 未支持向导快评的模块 → 跳转深入分析
            quick_modules = {"植被分析", "水体监测", "土壤盐渍化", "干旱监测"}
            deep_modules = [m for m in selected_modules if m not in quick_modules]
            if deep_modules:
                st.info(
                    "🔗 " + "、".join(deep_modules) + " 已为你准备好跳转入口"
                    "（下方「进入详细分析」），向导快评覆盖植被/水体/盐渍化/干旱，"
                    "其余模块请在对应页面用完整参数深入分析。"
                )

            # AI 智能解读 (基于实际勾选模块的指标)
            from utils.ai_insight import generate_ai_insight, is_ai_available as _ai_ok
            insight_metrics = {"NDVI均值": results_summary["ndvi_mean"]}
            insight_metrics.update(wiz_metrics)
            with st.spinner("🧠 DeepSeek AI 解读中..."):
                ai_text = generate_ai_insight(
                    analysis_type="+".join(selected_modules[:4]) + "综合分析",
                    metrics=insight_metrics,
                    study_area=results_summary["study_area"],
                    time_range=results_summary["date"],
                )
            # LLM 输出视为不可信, 转义后展示 (防 XSS)
            ai_text_esc = html.escape(ai_text or "")
            st.markdown(
                f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                f"🧠 **AI 解读**（{'DeepSeek AI' if _ai_ok() else '规则模板'}）：{ai_text_esc}</div>",
                unsafe_allow_html=True,
            )
            st.session_state["workflow_ai_insight"] = ai_text

            # 各模块跳转
            st.divider()
            st.subheader("🔗 进入详细分析")
            page_map = {
                "植被分析": "3_植被分析", "水体监测": "2_水体监测",
                "干旱监测": "7_干旱监测", "AI分类": "4_AI分类",
                "变化检测": "5_变化检测", "冰冻圈分析": "8_冰冻圈分析",
                "农业干旱": "9_农业干旱", "时序动画": "10_时序动画",
                "生态评估": "11_生态评估", "土壤盐渍化": "13_土壤盐渍化",
                "地表温度": "14_LST", "蒸散发": "17_蒸散发",
                "监督分类": "18_监督分类", "土地转移": "16_土地转移",
                "指数计算器": "15_指数计算器", "矢量导出": "19_矢量导出",
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

        # 合并 AI 解读 (如分步向导已生成) + HTML 转义 (防 XSS)
        wf_results["ai_insight"] = html.escape(str(st.session_state.get("workflow_ai_insight", "")))
        wf_results["study_area_esc"] = html.escape(str(wf_results["study_area"]))
        wf_results["satellite_esc"] = html.escape(str(wf_results["satellite"]))
        wf_results["date_esc"] = html.escape(str(wf_results["date"]))

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
        <h1>🛰️ {wf_results['study_area_esc']} 遥感分析报告</h1>
        <p>数据源: {wf_results['satellite_esc']} | 分析日期: {wf_results['date_esc']} | 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <div class="card">
        <h3>📊 核心指标</h3>
        <div class="metric-grid">
        <div class="metric"><div class="value">{wf_results['ndvi_mean']:.4f}</div><div class="label">NDVI 均值</div></div>
        <div class="metric"><div class="value">{wf_results.get('vegetation_ratio', 0)*100:.1f}%</div><div class="label">植被覆盖率</div></div>
        <div class="metric"><div class="value">{wf_results.get('water_ratio', 0)*100:.1f}%</div><div class="label">水体覆盖率</div></div>
        <div class="metric"><div class="value">{wf_results.get('ndvi_std', 0):.4f}</div><div class="label">NDVI 标准差</div></div>
        </div>
        </div>

        <div class="card">
        <h3>📊 扩展指标</h3>
        <div class="metric-grid">
        <div class="metric"><div class="value">{wf_results.get('evi_mean', 0):.4f}</div><div class="label">EVI 均值</div></div>
        <div class="metric"><div class="value">{wf_results.get('ndsi_mean', 0):.4f}</div><div class="label">NDSI 盐分均值</div></div>
        <div class="metric"><div class="value">{wf_results.get('nddi_mean', 0):.4f}</div><div class="label">NDDI 均值</div></div>
        <div class="metric"><div class="value">{wf_results.get('dry_ratio', 0)*100:.1f}%</div><div class="label">干旱风险占比</div></div>
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

        <div class="card" style="background:#f0f8f4;border-left:4px solid #27ae60;">
        <h3>🤖 AI 智能解读</h3>
        <p>{wf_results.get('ai_insight', '')}</p>
        </div>

        <div class="footer">
        <p>Geo AI 干旱区遥感智能分析平台 v1.20 | 自动生成</p>
        </div>
        </body></html>
        """

        st.download_button("📥 下载 HTML 报告", report_html.encode("utf-8"),
                          f"report_{wf_results['study_area']}_{wf_results['date']}.html",
                          "text/html")

        st.divider()
        st.subheader("📄 报告预览")
        st.html(report_html, width="stretch")
