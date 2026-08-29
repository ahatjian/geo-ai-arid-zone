"""
土地覆盖转移矩阵页面 — 双时相变化分析
========================================
基于 ESA WorldCover (2020 v100 / 2021 v200) 双时相土地覆盖对比
生成土地利用转移矩阵、净变化、主要转移方向
"""
import streamlit as st
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS
from utils.error_handler import StreamlitErrorBoundary
from utils.aoi import render_aoi_selector
from utils.landcover import get_esa_landcover, esa_to_arid6, ARID6_CLASSES, ESA_CLASSES
from utils.transition import (
    analyze_transition, plot_transition_heatmap, plot_net_change_bar,
)

st.set_page_config(page_title="土地覆盖转移矩阵", page_icon="🔀", layout="wide")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🔀 转移矩阵设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="河西走廊",
        help_text="推荐河西走廊/塔里木盆地 (绿洲农业变化显著)",
        key_prefix="trans",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("数据源")
    st.markdown(
        "**ESA WorldCover** (10m)\n\n"
        "🕐 **T1**: 2020 年 (v100)\n\n"
        "🕑 **T2**: 2021 年 (v200)"
    )

    st.divider()

    st.subheader("分类体系")
    class_scheme = st.radio(
        "分类体系",
        ["干旱区6类", "ESA 原始11类"],
        index=0,
        help="干旱区6类: 植被/裸地/建设用地/农田/水体 (推荐, 更清晰); ESA 11类: 完整类别",
    )

    pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)

    analyze_clicked = st.button(
        "🔀 计算转移矩阵", type="primary", use_container_width=True
    )

# ============================================================
# 主页面
# ============================================================
st.title("🔀 土地覆盖转移矩阵分析")
st.markdown(f"**研究区: {area_name}** | 数据源: ESA WorldCover (2020 → 2021)")
st.markdown(
    "对比两个时相的土地覆盖分类，生成土地利用转移矩阵，"
    "识别植被退化、农田扩张、建设用地增长等变化方向。"
)

if analyze_clicked:
    # bbox 已由研究区/AOI 选择组件提供 (支持预设 + 自定义 AOI)

    # ---- Step 1: 获取两个时相土地覆盖 ----
    with st.spinner("⬇️ 获取 2020 年土地覆盖数据..."):
        with StreamlitErrorBoundary("获取 2020 数据", st=st, show_traceback=True):
            class_t1_raw, meta1 = get_esa_landcover(bbox, version="v100", year="2020")

    with st.spinner("⬇️ 获取 2021 年土地覆盖数据..."):
        with StreamlitErrorBoundary("获取 2021 数据", st=st, show_traceback=True):
            class_t2_raw, meta2 = get_esa_landcover(bbox, version="v200", year="2021")

    # ---- Step 2: 分类体系处理 ----
    if class_scheme == "干旱区6类":
        class_t1 = esa_to_arid6(class_t1_raw)
        class_t2 = esa_to_arid6(class_t2_raw)
        class_names = [ARID6_CLASSES[i]["name"] for i in range(6)]
        class_colors = [ARID6_CLASSES[i]["color"] for i in range(6)]
    else:
        # ESA 原始类别 (保留存在的类别)
        class_t1 = class_t1_raw.astype(np.int16)
        class_t2 = class_t2_raw.astype(np.int16)
        class_names = [ESA_CLASSES[k]["name"] for k in sorted(ESA_CLASSES.keys())]
        class_colors = [ESA_CLASSES[k]["color"] for k in sorted(ESA_CLASSES.keys())]
        # 重新映射类别值到连续索引
        valid_keys = sorted(ESA_CLASSES.keys())
        remap = {v: i for i, v in enumerate(valid_keys)}
        class_t1 = np.vectorize(lambda x: remap.get(x, 0))(class_t1)
        class_t2 = np.vectorize(lambda x: remap.get(x, 0))(class_t2)

    # ---- Step 3: 对齐尺寸 ----
    h = min(class_t1.shape[0], class_t2.shape[0])
    w = min(class_t1.shape[1], class_t2.shape[1])
    class_t1 = class_t1[:h, :w]
    class_t2 = class_t2[:h, :w]

    # ---- Step 4: 计算转移矩阵 ----
    with st.spinner("🔀 计算转移矩阵..."):
        result = analyze_transition(
            class_t1, class_t2,
            class_names=class_names,
            pixel_size_m=pixel_size,
            top_n=10,
        )

    st.success("✅ 转移矩阵计算完成")

    # ---- KPI 卡片 ----
    st.subheader("📊 变化概览")
    total_change = result["total_change_km2"]
    total_area = float(np.sum(result["area_matrix"]))
    change_rate = (total_change / total_area * 100) if total_area > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总变化面积", f"{total_change:.2f} km²")
    with col2:
        st.metric("变化率", f"{change_rate:.2f}%")
    with col3:
        st.metric("分析总面积", f"{total_area:.2f} km²")
    with col4:
        top_trans = result["major_transitions"][0] if result["major_transitions"] else None
        st.metric("最大转移", top_trans["from"] + "→" + top_trans["to"] if top_trans else "无")

    st.divider()

    # ---- 标签页 ----
    tab_heat, tab_net, tab_table, tab_major = st.tabs(
        ["转移矩阵热力图", "净变化", "转移矩阵表", "主要转移方向"]
    )

    with tab_heat:
        fig = plot_transition_heatmap(
            result["area_matrix"], class_names,
            title=f"土地覆盖转移矩阵 — {area_name} (2020 → 2021)",
            return_fig=True,
        )
        st.pyplot(fig)

    with tab_net:
        fig = plot_net_change_bar(
            result["net_change"],
            title=f"土地类型净变化 — {area_name}",
            return_fig=True,
        )
        st.pyplot(fig)

        st.markdown("**净变化 = 转入 − 转出**，正值表示该类型面积增加，负值表示减少。")

    with tab_table:
        n = len(class_names)
        df = pd.DataFrame(
            result["area_matrix"],
            index=[f"{name} (T1)" for name in class_names],
            columns=[f"{name} (T2)" for name in class_names],
        )
        df = df.round(2)
        st.dataframe(df, use_container_width=True)
        st.caption("单位: km² | 行 = T1 (2020) 起始类型，列 = T2 (2021) 最终类型")

    with tab_major:
        if result["major_transitions"]:
            trans_df = pd.DataFrame(result["major_transitions"])
            trans_df.columns = ["起始类型", "转为类型", "面积(km²)"]
            st.dataframe(trans_df, use_container_width=True, hide_index=True)

            # 主要转移方向条形图
            import matplotlib.pyplot as plt
            labels = [f"{t['from']}→{t['to']}" for t in result["major_transitions"]]
            areas = [t["area_km2"] for t in result["major_transitions"]]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(labels[::-1], areas[::-1], color="#d2691e")
            for bar, a in zip(bars, areas[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{a:.1f} km²", va="center", fontsize=9)
            ax.set_xlabel("转移面积 (km²)")
            ax.set_title(f"主要土地转移方向 — {area_name}", fontsize=13)
            ax.set_xlim(0, max(areas) * 1.2)
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("两个时相土地覆盖无显著变化。")

    # ---- 图例 ----
    st.divider()
    st.subheader("🗺️ 分类图例")
    legend_cols = st.columns(3)
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        with legend_cols[i % 3]:
            st.markdown(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{color};border-radius:3px;margin-right:6px"></span>{name}',
                unsafe_allow_html=True,
            )
