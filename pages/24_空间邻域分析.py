"""
空间邻域分析页面 — 第 24 模块 (Geo AI 差异化)
================================================
与传统 GIS 空间分析不同, 本模块直接使用平台已有的分析结果
(分类图/指数图), 并与 DeepSeek AI 解读深度结合:

  - 🎯 AI 智能缓冲区: 分类图选目标地物 → 缓冲 → 指标梯度分析 → AI 解读
  - 📐 邻域统计: 指数图邻域均值/标准差/变异系数
  - 🔀 叠加分析: 分类 × 指数交叉表 → AI 结论

差异化: 无需外部 GIS 数据, 输入即平台分析结果, 输出即 AI 解读。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="空间邻域分析", page_icon="🗺️", layout="wide")

st.title("🗺️ 空间邻域分析（Geo AI）")
st.markdown(
    "**AI 智能缓冲区 · 邻域统计 · 叠加分析** — "
    "直接分析平台结果，DeepSeek 自动解读空间格局"
)

# ============================================================
# 侧边栏: 数据源
# ============================================================
with st.sidebar:
    st.title("🗺️ 空间分析设置")

    st.subheader("📦 输入数据")
    st.caption("使用平台已有分析结果（非外部 GIS 数据）")

    # 数据源: session 中的分析结果 或 上传
    available = []
    if "km_class_result" in st.session_state:
        available.append("KMeans 聚类结果 (AI分类页)")
    if "ai_class_result" in st.session_state:
        available.append("AI 分类结果 (AI分类页)")
    if "veg_index" in st.session_state:
        available.append("植被指数 NDVI (植被分析页)")
    if "dos_corrected_bands" in st.session_state:
        available.append("大气校正波段 (预处理页)")

    src_option = "📤 上传 GeoTIFF"
    if available:
        src_option = st.selectbox("数据来源", ["📤 上传 GeoTIFF"] + available)

    data_mode = "upload"
    geotiff_path = None
    if src_option == "📤 上传 GeoTIFF":
        uploaded = st.file_uploader("上传分类图 或 指数图 GeoTIFF", type=["tif", "tiff"])
        if uploaded:
            geotiff_path = save_upload_tmp(uploaded)
            st.success(f"✅ 已加载: {uploaded.name}")
    else:
        data_mode = "session"
        if "聚类" in src_option:
            st.session_state["spatial_class"] = st.session_state["km_class_result"]
            st.session_state["spatial_class_names"] = st.session_state.get("km_class_names", None)
        elif "AI 分类" in src_option:
            st.session_state["spatial_class"] = st.session_state["ai_class_result"]
            st.session_state["spatial_class_names"] = st.session_state.get("ai_class_names", None)
        elif "植被" in src_option:
            st.session_state["spatial_value"] = st.session_state["veg_index"]
        elif "大气" in src_option:
            st.session_state["spatial_value"] = st.session_state["dos_corrected_bands"][2]  # R 波段

    st.divider()
    st.caption("💡 提示: 先在其他页面完成分析，结果会自动出现在数据源列表")

# ============================================================
# 主面板
# ============================================================

# 栅格读取安全限制 (防超大文件内存 DoS)
MAX_RASTER_PIXELS = 50_000_000  # 50 MP


def load_raster(path: str) -> np.ndarray:
    """读取单波段栅格 (带尺寸限制)。"""
    import rasterio
    with rasterio.open(path) as src:
        if src.width * src.height > MAX_RASTER_PIXELS:
            raise ValueError(
                f"栅格过大 ({src.width}×{src.height} = {src.width*src.height/1e6:.0f} MP, "
                f"上限 {MAX_RASTER_PIXELS/1e6:.0f} MP)，请裁剪后上传"
            )
        return src.read(1).astype(np.float64)


def save_upload_tmp(uploaded) -> str:
    """安全保存上传文件到临时目录 (唯一名, 使用后清理)。"""
    import uuid as _uuid
    safe_name = f"spatial_{_uuid.uuid4().hex[:12]}.tif"
    geotiff_path = os.path.join(tempfile.gettempdir(), safe_name)
    with open(geotiff_path, "wb") as f:
        f.write(uploaded.getvalue())
    return geotiff_path


# 确定输入
input_arr = None
input_kind = None  # "class" | "value"
if data_mode == "upload" and geotiff_path:
    input_arr = load_raster(geotiff_path)
    input_kind = "value"
    st.sidebar.caption("上传数据按指数图处理（如需分类图请选择平台结果）")
elif data_mode == "session":
    if "spatial_class" in st.session_state:
        input_arr = st.session_state["spatial_class"]
        input_kind = "class"
    elif "spatial_value" in st.session_state:
        input_arr = st.session_state["spatial_value"]
        input_kind = "value"

if input_arr is None:
    st.info("👈 请上传 GeoTIFF 或先在其他页面完成分析（分类/植被/盐渍化等）")
    with st.expander("📖 Geo AI 空间分析说明"):
        st.markdown("""
        ### 与传统 GIS 的区别
        传统 GIS（ArcGIS/QGIS）空间分析需要：
        - 外部矢量/栅格数据 + 手工图层操作
        - 输出几何结果，用户自行解读

        本平台的 **Geo AI 空间分析**：
        - 输入 = 平台内部分析结果（分类图/指数图），零数据准备
        - 分析 = 缓冲区梯度 / 邻域统计 / 叠加交叉，一键执行
        - 输出 = 统计 + **DeepSeek AI 专业解读**（空间格局的生态含义）

        ### 典型场景
        - 绿洲周边 3km 缓冲带的 NDVI 衰减分析（绿洲-荒漠过渡带）
        - 水体缓冲带盐渍化风险识别
        - 土地覆盖 × 干旱分级的空间叠加诊断
        """)
    st.stop()

st.success(f"✅ 已加载 {'分类图' if input_kind == 'class' else '指标图'} ({input_arr.shape[1]}×{input_arr.shape[0]})")
st.divider()

tab_buf, tab_nb, tab_ov = st.tabs([
    "🎯 AI 智能缓冲区",
    "📐 邻域统计",
    "🔀 叠加分析",
])

# ============================================================
# Tab 1: AI 智能缓冲区
# ============================================================
with tab_buf:
    st.subheader("🎯 AI 智能缓冲区分析")
    st.caption("从分类图选择目标地物 → 缓冲带指标梯度 → DeepSeek 解读空间格局")

    if input_kind != "class":
        st.warning("⚠️ 缓冲区分析需要分类图 — 请选择平台分类结果（KMeans/AI分类）或上传分类 GeoTIFF")
    else:
        # 类别选择
        unique_classes = np.unique(input_arr[input_arr >= 0])
        class_names = st.session_state.get("spatial_class_names", None)
        cls_options = {int(c): (class_names[int(c)] if class_names and int(c) < len(class_names) else f"类别{int(c)}")
                       for c in unique_classes}

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            target_cls = st.selectbox("目标地物类别", list(cls_options.keys()),
                                      format_func=lambda c: f"{c}: {cls_options[c]}")
        with col_b2:
            buf_dist = st.number_input("缓冲距离 (米)", 100, 50000, 3000, 500)
        with col_b3:
            pixel_size = st.number_input("像元大小 (米)", 1, 100, 10)

        # 指标图选择
        st.markdown("**📊 分析指标**")
        metric_source = st.radio(
            "指标来源",
            ["📤 上传指标 GeoTIFF", "🗺️ 用分类图本身", "🌿 植被指数 (session)"],
            horizontal=True,
        )
        value_arr = None
        if "上传" in metric_source:
            metric_file = st.file_uploader("上传指标图 (NDVI/LST/盐分等)", type=["tif", "tiff"],
                                           key="buf_metric")
            if metric_file:
                mtmp = save_upload_tmp(metric_file)
                value_arr = load_raster(mtmp)
        elif "session" in metric_source and "veg_index" in st.session_state:
            value_arr = st.session_state["veg_index"]
            st.caption("✅ 使用植被分析页的 NDVI")
        else:
            value_arr = input_arr.astype(np.float64)

        if st.button("🎯 执行缓冲区分析", type="primary"):
            with st.spinner("缓冲区分析中..."):
                from utils.spatial import smart_buffer_analysis
                result = smart_buffer_analysis(
                    input_arr, value_arr, target_class=int(target_cls),
                    distance_m=buf_dist, pixel_size_m=pixel_size,
                    class_names=[cls_options[c] for c in sorted(cls_options)],
                )

            stats = result["stats"]
            st.divider()
            st.subheader(f"📊 「{result['target_name']}」{result['buffer_distance_km']}km 缓冲区分析")

            # 指标卡片
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("目标区均值", f"{stats['target_mean']:.4f}" if stats['target_mean'] is not None else "—")
            with c2:
                st.metric("缓冲带均值", f"{stats['buffer_mean']:.4f}" if stats['buffer_mean'] is not None else "—")
            with c3:
                st.metric("外部区均值", f"{stats['outer_mean']:.4f}" if stats['outer_mean'] is not None else "—")
            with c4:
                st.metric("目标-外部梯度", f"{stats['gradient']:+.4f}" if stats['gradient'] is not None else "—")

            # 缓冲区可视化
            from utils.spatial import raster_buffer
            buf_map = raster_buffer(input_arr == int(target_cls), buf_dist, pixel_size)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].imshow(input_arr, cmap="tab20", interpolation="nearest")
            axes[0].set_title(f"分类图 (目标={cls_options[target_cls]})")
            axes[0].axis("off")
            im = axes[1].imshow(buf_map, cmap="RdYlBu_r", interpolation="nearest")
            axes[1].set_title(f"{result['buffer_distance_km']}km 缓冲区")
            axes[1].axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # AI 解读 (差异化核心)
            st.divider()
            st.subheader("🤖 DeepSeek AI 空间解读")
            from utils.ai_insight import generate_ai_insight, is_ai_available
            with st.spinner("AI 解读空间格局..."):
                insight = generate_ai_insight(
                    analysis_type=f"「{result['target_name']}」缓冲区空间分析",
                    metrics={
                        "目标区指标均值": stats["target_mean"],
                        "缓冲带均值": stats["buffer_mean"],
                        "外部区均值": stats["outer_mean"],
                        "目标-外部梯度": stats["gradient"],
                    },
                    study_area="研究区",
                )
            ai_esc = __import__("html").escape(insight or "")
            st.markdown(
                f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                f"🧠 **AI 解读**（{'DeepSeek AI' if is_ai_available() else '规则模板'}）：{ai_esc}</div>",
                unsafe_allow_html=True,
            )

# ============================================================
# Tab 2: 邻域统计
# ============================================================
with tab_nb:
    st.subheader("📐 邻域统计")
    st.caption("每个像元取其邻域的统计量 — 识别局部异常/均质区域")

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        nb_stat = st.selectbox("统计量", ["mean", "std", "cv", "median"],
                               format_func=lambda x: {
                                   "mean": "均值 (局部平滑)",
                                   "std": "标准差 (局部变异)",
                                   "cv": "变异系数 (相对变异)",
                                   "median": "中值 (稳健)",
                               }[x])
    with col_n2:
        nb_kernel = st.slider("邻域大小", 3, 15, 5, step=2)

    if st.button("📐 计算邻域统计", type="primary"):
        with st.spinner("邻域计算中..."):
            from utils.spatial import neighborhood_stats
            nb_map = neighborhood_stats(input_arr, nb_kernel, nb_stat)

        st.divider()
        st.subheader(f"📊 邻域{nb_stat} ({nb_kernel}×{nb_kernel})")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im0 = axes[0].imshow(input_arr, cmap="viridis", interpolation="nearest")
        axes[0].set_title("原始")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        im1 = axes[1].imshow(nb_map, cmap="viridis", interpolation="nearest")
        axes[1].set_title(f"邻域 {nb_stat}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)
        st.pyplot(fig)
        plt.close(fig)

        # 统计摘要
        valid = nb_map[np.isfinite(nb_map)]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("均值", f"{np.nanmean(valid):.4f}")
        with c2:
            st.metric("标准差", f"{np.nanstd(valid):.4f}")
        with c3:
            st.metric("范围", f"{np.nanmin(valid):.3f} ~ {np.nanmax(valid):.3f}")

# ============================================================
# Tab 3: 叠加分析
# ============================================================
with tab_ov:
    st.subheader("🔀 叠加分析（分类 × 指数）")
    st.caption("将分类图与分级/指数图叠加，生成交叉表 — 空间关系诊断")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        overlay_type = st.selectbox(
            "叠加对象",
            ["干旱分级", "土地覆盖", "自定义上传"],
            help="与分类图叠加的第二个图层",
        )
    with col_o2:
        o_pixel = st.number_input("像元大小 (米)", 1, 100, 10)

    overlay_arr = None
    names_b = None
    if overlay_type == "干旱分级":
        overlay_arr = st.session_state.get("drought_category", None)
        names_b = ["正常", "轻旱", "中旱", "重旱", "特旱"]
        if overlay_arr is None:
            st.info("💡 提示: 请先在「干旱监测」页生成干旱分级结果")
    elif overlay_type == "土地覆盖":
        overlay_arr = st.session_state.get("lc_result", None)
        names_b = ["水体", "植被", "裸地", "建设用地", "农田", "其他"]
        if overlay_arr is None:
            st.info("💡 提示: 请先在「AI 分类」页生成分类结果")
    else:
        ov_file = st.file_uploader("上传第二图层 GeoTIFF", type=["tif", "tiff"], key="ov_file")
        if ov_file:
            otmp = save_upload_tmp(ov_file)
            with open(otmp, "wb") as f:
                f.write(ov_file.getvalue())
            overlay_arr = load_raster(otmp)

    if overlay_arr is not None:
        if overlay_arr.shape != input_arr.shape:
            st.warning(f"⚠️ 尺寸不一致: 主图 {input_arr.shape} vs 叠加 {overlay_arr.shape}")
        elif st.button("🔀 执行叠加分析", type="primary"):
            with st.spinner("叠加分析中..."):
                from utils.spatial import overlay_crosstab, overlay_analysis_text
                class_names_a = st.session_state.get("spatial_class_names", None)
                result = overlay_crosstab(
                    input_arr, overlay_arr.astype(np.int16),
                    names_a=class_names_a, names_b=names_b,
                    pixel_size_m=o_pixel,
                )

            st.divider()
            st.subheader("📊 空间叠加交叉表")

            # 表格
            import pandas as pd
            df = pd.DataFrame([
                {"图层A": r["A"], "图层B": r["B"],
                 "面积(km²)": r["面积_km2"], "占比(%)": r["占比_pct"]}
                for r in result["rows"][:15]
            ])
            st.dataframe(df, width="stretch", hide_index=True)

            # 前 5 关系
            st.subheader("🏆 主要空间关系")
            for i, r in enumerate(result["top_relations"][:5]):
                st.markdown(
                    f"{i+1}. **{r['A']} × {r['B']}** — {r['面积_km2']:.2f} km²"
                )

            # AI 解读
            st.divider()
            st.subheader("🤖 DeepSeek AI 叠加解读")
            from utils.ai_insight import generate_ai_insight, is_ai_available
            top = result["top_relations"]
            with st.spinner("AI 解读空间关系..."):
                insight = generate_ai_insight(
                    analysis_type="空间叠加分析",
                    metrics={
                        "最大空间关系面积": top[0]["面积_km2"] if top else 0,
                        "最大关系占比": max(r["占比_pct"] for r in result["rows"]) if result["rows"] else 0,
                        "主要关系数": len(result["top_relations"]),
                    },
                    study_area="研究区",
                )
            ai_esc = __import__("html").escape(insight or "")
            st.markdown(
                f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                f"🧠 **AI 解读**（{'DeepSeek AI' if is_ai_available() else '规则模板'}）：{ai_esc}</div>",
                unsafe_allow_html=True,
            )
