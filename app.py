"""
Geo AI 干旱区遥感智能分析平台 — 首页入口
版本 v1.21 — 25 模块完整链路: 从数据浏览到智能工作流 + AI 解读
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import APP_TITLE, APP_ICON, APP_VERSION, STUDY_AREAS
from utils.error_handler import StreamlitErrorBoundary

# ============================================
# 页面配置
# ============================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# 自定义 CSS
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .area-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        background: linear-gradient(135deg, #f8fbff, #f0f7f0);
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .area-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .area-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .area-desc {
        font-size: 0.85rem;
        color: #555;
        line-height: 1.5;
    }
    .area-tags {
        margin-top: 0.5rem;
    }
    .area-tags span {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        background: #e8f4fd;
        color: #1f77b4;
        font-size: 0.75rem;
    }
    .nav-card {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-card:hover {
        border-color: #1f77b4;
        background: #f0f7fb;
    }
    .nav-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .nav-label {
        font-weight: 600;
        color: #333;
    }
    .nav-desc {
        font-size: 0.78rem;
        color: #888;
        margin-top: 0.3rem;
    }
    .stats-box {
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(135deg, #f0f7fb, #f0f7f0);
        border-radius: 8px;
        margin: 0.3rem;
    }
    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stats-label {
        font-size: 0.78rem;
        color: #888;
    }
    hr.divider {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e8e8e8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 标题区
# ============================================
st.markdown(f'<p class="main-title">{APP_ICON} {APP_TITLE}</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">免费卫星数据 + 遥感指数自动计算 = 不写代码做科研级遥感分析</p>',
    unsafe_allow_html=True,
)

# ============================================
# 侧边栏 - 平台信息
# ============================================
with st.sidebar:
    st.title("🛰️ 平台面板")

    # 快速统计
    st.subheader("平台能力")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">6</div><div class="stats-label">预设研究区</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">5</div><div class="stats-label">卫星数据源</div></div>',
            unsafe_allow_html=True,
        )

    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">12+</div><div class="stats-label">遥感指数</div></div>',
            unsafe_allow_html=True,
        )
    with cols2[1]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">AI</div><div class="stats-label">DeepSeek + 深度学习</div></div>',
            unsafe_allow_html=True,
        )

    cols3 = st.columns(2)
    with cols3[0]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">25</div><div class="stats-label">分析模块</div></div>',
            unsafe_allow_html=True,
        )
    with cols3[1]:
        st.markdown(
            '<div class="stats-box"><div class="stats-number">FREE</div><div class="stats-label">完全免费</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("当前研究区")
    if "selected_area" in st.session_state:
        area = st.session_state["selected_area"]
        if area in STUDY_AREAS:
            st.info(f"📍 **{area}**\n\n{STUDY_AREAS[area]['description']}")
        else:
            bbox = st.session_state.get("selected_bbox", None)
            if bbox:
                st.info(f"📍 **{area}**\n\nbbox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
            else:
                st.info(f"📍 **{area}**")
    else:
        st.info("👈 请在下方选择研究区")

    st.divider()
    st.caption("📡 数据源: Microsoft Planetary Computer")
    st.caption("🧠 AI 引擎: DeepSeek AI + PyTorch")
    st.caption("🚀 部署: Streamlit Cloud (免费)")

# ============================================
# Tab1: 功能导航
# ============================================
tab_intro, tab_areas, tab_about = st.tabs(["🚀 快速开始", "🌏 研究区选择", "📖 关于平台"])

with tab_intro:
    # ============================================
    # AI 快速对话入口 (首页即 AI)
    # ============================================
    st.markdown("### 🤖 问我任何遥感问题")
    ai_q_col1, ai_q_col2 = st.columns([4, 1])
    with ai_q_col1:
        ai_quick_q = st.text_input(
            "向 AI 提问",
            placeholder='例如: "什么是 NDVI？" / "如何分析塔里木盆地盐渍化？" / "帮我解读分析结果"',
            label_visibility="collapsed",
            key="home_ai_question",
        )
    with ai_q_col2:
        ai_quick_go = st.button("🚀 去问 AI", type="primary", width="stretch")
    if ai_quick_go and ai_quick_q.strip():
        st.session_state["pending_ai_question"] = ai_quick_q.strip()
        st.switch_page("pages/25_AI助手.py")
    st.caption("AI 助手可回答遥感知识、引导平台操作、解读你的分析结果")
    st.divider()

    st.subheader("选择功能模块开始分析")

    col1, col2, col3, col4 = st.columns(4)

    nav_items = [
        {
            "icon": "🤖",
            "label": "AI 智能助手",
            "desc": "AI 对话\n一键分析/异常检测",
            "page": "25_AI助手",
        },
        {
            "icon": "🗺️",
            "label": "数据浏览",
            "desc": "搜索卫星影像\n预览与下载",
            "page": "1_数据浏览",
        },
        {
            "icon": "💧",
            "label": "水体监测",
            "desc": "MNDWI/AWEIsh\n水体面积统计",
            "page": "2_水体监测",
        },
        {
            "icon": "🌿",
            "label": "植被分析",
            "desc": "NDVI/EVI计算\n趋势分析",
            "page": "3_植被分析",
        },
        {
            "icon": "🤖",
            "label": "AI 分类",
            "desc": "深度学习\n地物分割",
            "page": "4_AI分类",
        },
        {
            "icon": "🔄",
            "label": "变化检测",
            "desc": "双时相对比\n变化识别",
            "page": "5_变化检测",
        },
        {
            "icon": "📄",
            "label": "报告导出",
            "desc": "汇总结果\n生成报告",
            "page": "6_报告导出",
        },
        {
            "icon": "🏜️",
            "label": "干旱监测",
            "desc": "VCI/NDDI/距平\n多指数干旱分析",
            "page": "7_干旱监测",
        },
        {
            "icon": "❄️",
            "label": "冰冻圈分析",
            "desc": "NDSI/雪盖/冰川\n冻土活动层分析",
            "page": "8_冰冻圈分析",
        },
        {
            "icon": "🌾",
            "label": "农业干旱",
            "desc": "CWSI/土壤水分\n灌溉需求评估",
            "page": "9_农业干旱",
        },
        {
            "icon": "🎬",
            "label": "时序动画",
            "desc": "NDVI/水体/雪盖\n年际变化 GIF",
            "page": "10_时序动画",
        },
        {
            "icon": "🌍",
            "label": "生态评估",
            "desc": "PSR 压力-状态-响应\n生态安全评价",
            "page": "11_生态评估",
        },
        {
            "icon": "⚡",
            "label": "智能工作流",
            "desc": "自然语言查询\n一键分析 + 报告",
            "page": "12_工作流",
        },
        {
            "icon": "🧂",
            "label": "土壤盐渍化",
            "desc": "SI/NDSI 盐分指数\n5级盐渍化评估",
            "page": "13_土壤盐渍化",
        },
        {
            "icon": "🌡️",
            "label": "地表温度 LST",
            "desc": "Landsat 热红外反演\n5级热环境分级",
            "page": "14_LST",
        },
        {
            "icon": "🧮",
            "label": "指数计算器",
            "desc": "预设指数 + 自定义\n波段运算 (Band Math)",
            "page": "15_指数计算器",
        },
        {
            "icon": "🔀",
            "label": "土地转移矩阵",
            "desc": "双时相土地覆盖对比\n转移方向与净变化",
            "page": "16_土地转移",
        },
        {
            "icon": "💨",
            "label": "蒸散发估算",
            "desc": "SEBAL 能量平衡\n地表蒸散发 ET",
            "page": "17_蒸散发",
        },
        {
            "icon": "🎯",
            "label": "监督分类",
            "desc": "自定义样本训练\nRF/SVM/KNN/MLP",
            "page": "18_监督分类",
        },
        {
            "icon": "🗺️",
            "label": "矢量导出",
            "desc": "分类结果矢量化\nGeoJSON/Shapefile/KML",
            "page": "19_矢量导出",
        },
        {
            "icon": "📦",
            "label": "数据下载中心",
            "desc": "结果持久化\n统一下载/打包/删除",
            "page": "20_数据下载中心",
        },
        {
            "icon": "🛠️",
            "label": "系统状态",
            "desc": "缓存管理\n环境健康检查",
            "page": "21_系统状态",
        },
        {
            "icon": "🎨",
            "label": "图像增强",
            "desc": "PCA/滤波/增强\nIHS 融合",
            "page": "22_图像增强",
        },
        {
            "icon": "🌤️",
            "label": "大气校正",
            "desc": "DOS 暗像元法\n辐射定标",
            "page": "23_辐射定标大气校正",
        },
        {
            "icon": "🗺️",
            "label": "空间邻域分析",
            "desc": "AI 智能缓冲区\n叠加分析",
            "page": "24_空间邻域分析",
        },
    ]

    # 第一行: 4 cards
    for i, col in enumerate([col1, col2, col3, col4]):
        with col:
            item = nav_items[i]
            st.markdown(
                f"""<div class="nav-card">
                <div class="nav-icon">{item['icon']}</div>
                <div class="nav-label">{item['label']}</div>
                <div class="nav-desc">{item['desc'].replace(chr(10),'<br>')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"进入 {item['label']}", key=f"nav_{i}"):
                st.switch_page(f"pages/{item['page']}.py")

    # 第二行: 4 cards
    col5, col6, col7, col8 = st.columns(4)
    for j, col in enumerate([col5, col6, col7, col8]):
        with col:
            item = nav_items[4 + j]
            st.markdown(
                f"""<div class="nav-card">
                <div class="nav-icon">{item['icon']}</div>
                <div class="nav-label">{item['label']}</div>
                <div class="nav-desc">{item['desc'].replace(chr(10),'<br>')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"进入 {item['label']}", key=f"nav_{4 + j}"):
                st.switch_page(f"pages/{item['page']}.py")

    # 第三行: 4 cards
    col9, col10, col11, col12 = st.columns(4)
    for k, col in enumerate([col9, col10, col11, col12]):
        with col:
            item = nav_items[8 + k]
            st.markdown(
                f"""<div class="nav-card">
                <div class="nav-icon">{item['icon']}</div>
                <div class="nav-label">{item['label']}</div>
                <div class="nav-desc">{item['desc'].replace(chr(10),'<br>')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"进入 {item['label']}", key=f"nav_{8 + k}"):
                st.switch_page(f"pages/{item['page']}.py")

    # 第四行起: 剩余 cards (第 13 个模块起), 每 4 个一行
    remaining = nav_items[12:]
    for row_start in range(0, len(remaining), 4):
        row_items = remaining[row_start:row_start + 4]
        cols4 = st.columns(4)
        for m, col in enumerate(cols4):
            if m >= len(row_items):
                break
            with col:
                item = row_items[m]
                st.markdown(
                    f"""<div class="nav-card">
                    <div class="nav-icon">{item['icon']}</div>
                    <div class="nav-label">{item['label']}</div>
                    <div class="nav-desc">{item['desc'].replace(chr(10),'<br>')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(f"进入 {item['label']}", key=f"nav_{12 + row_start + m}"):
                    st.switch_page(f"pages/{item['page']}.py")

    st.divider()

    st.subheader("🔄 工作流程")
    st.markdown("""
    ```
    ① 选择研究区 → ② 搜索卫星影像 → ③ 计算遥感指数 → ④ AI/统计分析 → ⑤ AI 解读 + 导出结果
    ```
    """)

    st.info("💡 **提示**: 所有功能模块共享同一研究区和影像搜索结果，选择后可在各页面间自由切换。")

with tab_areas:
    st.subheader("研究区快速选择")

    # 2行3列布局
    areas = list(STUDY_AREAS.items())
    for row_idx in range(0, len(areas), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            idx = row_idx + col_idx
            if idx >= len(areas):
                break
            name, info = areas[idx]
            with cols[col_idx]:
                tags_html = " ".join(
                    f"<span>{t}</span>" for t in info["keywords"]
                )
                st.markdown(
                    f"""<div class="area-card">
                    <div class="area-name">📍 {name}</div>
                    <div class="area-desc">{info['description']}</div>
                    <div class="area-tags">{tags_html}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"选择 {name}",
                    key=f"area_{idx}",
                    width="stretch",
                ):
                    st.session_state["selected_area"] = name
                    st.session_state["selected_bbox"] = info["bbox"]
                    st.session_state["selected_center"] = info["center"]
                    st.rerun()

    st.divider()

    # 概览地图 (leafmap 失败自动降级为静态示意图)
    st.subheader("研究区概览")
    with StreamlitErrorBoundary("研究区概览地图", st=st, show_traceback=False):
        from utils.map_utils import draw_area_schematic
        try:
            import leafmap
            from shapely.geometry import box
            import geopandas as gpd

            m = leafmap.Map(center=[40, 90], zoom=4, height=400)

            # 添加所有研究区边界
            for name, info in STUDY_AREAS.items():
                bbox = info["bbox"]
                bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
                gdf = gpd.GeoDataFrame(
                    {"name": [name]}, geometry=[bbox_geom], crs="EPSG:4326"
                )
                m.add_gdf(
                    gdf,
                    layer_name=name,
                    style={"color": "blue", "fillOpacity": 0.08, "weight": 1.5},
                )

            m.add_basemap("Esri.WorldImagery")
            m.to_streamlit(height=400)
        except Exception:
            # 降级: 静态 6 研究区示意图
            img = draw_area_schematic("塔里木盆地", STUDY_AREAS["塔里木盆地"]["bbox"],
                                      all_areas=STUDY_AREAS, figsize=(9, 6))
            if img:
                st.image(img, caption="6 大研究区位置示意", width="stretch")
            else:
                st.warning("地图组件加载失败，请刷新重试")

with tab_about:
    st.subheader("关于本平台")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 💡 核心价值
        
        **免费卫星数据 + AI 自动分析 = 不写代码做科研级遥感分析**
        
        - 🛰️ **真实卫星数据**: Sentinel-2 (10m) + Landsat-8/9 (30m)
        - 🧮 **自动指数计算**: NDVI, EVI, MNDWI, AWEIsh 一键生成
        - 🤖 **AI 深度学习**: UNet/DeepLabV3+ 地物分类与水體分割
        - 📊 **专业分析**: Sen+MK 趋势分析、年际变化检测
        - 💾 **结果导出**: GeoTIFF / CSV / PNG 标准格式
        
        ### 🎯 适用场景
        - 干旱区水资源监测 (水体 + 干旱)
        - 绿洲植被覆盖变化 (NDVI 时序 + 动画)
        - 冰川/积雪消融追踪 (冰冻圈)
        - 农业灌溉需求评估 (农业干旱)
        - 土地覆盖分类制图 (AI 分类)
        - 沙漠化监测与生态安全 (沙漠化 + PSR)
        """)

    with col2:
        st.markdown("""
        ### 🛠️ 技术栈
        
        | 层级 | 技术 |
        |------|------|
        | 前端 | Streamlit + leafmap |
        | 数据 | Planetary Computer STAC |
        | 分析 | NumPy + SciPy + Rasterio |
        | AI | PyTorch + smp |
        | 可视化 | Plotly + Matplotlib |
        | 部署 | Streamlit Cloud (免费) |
        
        ### 📡 数据源
        
        | 卫星 | 分辨率 | 重访周期 | 时序 |
        |------|--------|----------|------|
        | Sentinel-2 | 10m | 5天 | 2015-至今 |
        | Landsat-8/9 | 30m | 16天 | 2013-至今 |
        | Landsat-4-5/7 | 30m | 16天 | 1982-至今 |
        
        ### 🔬 预设研究区
        """)

        for name in STUDY_AREAS:
            st.markdown(f"- 📍 {name}")

    st.divider()

    st.markdown(f"""
    <div style='text-align: center; color: gray; font-size: 0.85rem;'>
        <p>Geo AI 干旱区遥感分析平台 {APP_VERSION} | Powered by Microsoft Planetary Computer + DeepSeek AI</p>
        <p>🛰️ Sentinel-2 (10m) | Landsat 4-9 (30m, 1982-至今) | 完全免费</p>
    </div>
    """, unsafe_allow_html=True)
