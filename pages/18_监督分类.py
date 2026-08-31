"""
监督分类训练页面 — 自定义样本训练模型
========================================
基于 scikit-learn 的传统机器学习监督分类
摆脱对 ESA/ESRI 现成产品的依赖, 训练自己的分类模型
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.aoi import render_aoi_selector
from utils.pc_data import search_images, get_rgb_preview_cached, download_multiband
from utils.supervised import (
    CLASSIFIERS, FEATURE_BANDS, INDEX_FEATURES,
    build_feature_stack, sample_from_reference,
    sample_from_geojson, sample_from_csv, coords_to_indices,
    assess_supervised_classification,
)
from utils.landcover import ARID6_CLASSES, esa_to_arid6, get_esa_landcover
from utils.visualization import render_classification

st.set_page_config(page_title="监督分类训练", page_icon="🎯", layout="wide")

ARID6_NAMES = [ARID6_CLASSES[i]["name"] for i in range(6)]
ARID6_COLORS = [ARID6_CLASSES[i]["color"] for i in range(6)]

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🎯 监督分类设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="塔里木盆地",
        key_prefix="sup",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据",
        list(COLLECTIONS.keys()),
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 15)
    max_items = st.slider("最大影像数", 1, 15, 5)

    st.divider()

    st.subheader("🏷️ 训练样本来源")
    sample_source = st.radio(
        "样本来源",
        ["从 ESA 参考产品采样", "上传 GeoJSON 样本", "上传 CSV 样本"],
        help="ESA 采样: 用现有产品自动生成训练样本 (最便捷)\n"
             "GeoJSON/CSV: 上传自己标注的样本",
    )

    if sample_source == "从 ESA 参考产品采样":
        n_samples = st.slider("每类采样数", 50, 2000, 500,
            help="每类随机采样多少像元作为训练样本")
    elif sample_source == "上传 GeoJSON 样本":
        geojson_file = st.file_uploader("上传 GeoJSON (点/面, 含 class 属性)", type=["geojson", "json"])
        class_field = st.text_input("类别属性字段", value="class")
    else:
        csv_file = st.file_uploader("上传 CSV (lon,lat,label 列)", type=["csv"])

    st.divider()

    st.subheader("🤖 分类器")
    classifier_key = st.selectbox(
        "选择分类器",
        list(CLASSIFIERS.keys()),
        format_func=lambda x: CLASSIFIERS[x]["name"],
    )
    st.caption(CLASSIFIERS[classifier_key]["desc"])

    with st.expander("⚙️ 分类器参数"):
        if classifier_key == "random_forest":
            n_estimators = st.slider("决策树数量", 50, 500, 200, step=50)
            classifier_params = {"n_estimators": n_estimators}
        elif classifier_key == "svm":
            svm_c = st.slider("C 参数", 0.1, 100.0, 10.0, step=0.5)
            classifier_params = {"C": svm_c}
        elif classifier_key == "knn":
            n_neighbors = st.slider("近邻数 K", 1, 15, 5)
            classifier_params = {"n_neighbors": n_neighbors}
        else:  # mlp
            classifier_params = {}

    st.divider()

    use_indices = st.checkbox("添加遥感指数特征 (NDVI/NDWI/NDBI)", value=True)
    test_ratio = st.slider("测试集比例", 0.0, 0.5, 0.3, 0.05,
        help="留出多少比例样本做精度评估")

    train_clicked = st.button(
        "🎯 训练 & 分类", type="primary", use_container_width=True
    )

# ============================================================
# 主页面
# ============================================================
st.title("🎯 监督分类训练")
st.markdown(f"**研究区: {area_name}** | 数据源: {satellite} | 日期: {start_date} → {end_date}")
st.markdown(
    "自定义训练样本，训练自己的分类模型，摆脱对 ESA/ESRI 现成产品的依赖。"
    "支持随机森林 / SVM / KNN / MLP 四种分类器。"
)

if train_clicked:
    # ---- Step 1: 搜索影像 ----
    with st.spinner("🔍 正在搜索影像..."):
        try:
            results = search_images(
                bbox=bbox,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                collection=satellite,
                cloud_cover_max=cloud_cover,
                max_items=max_items,
            )
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

    if not results:
        st.warning("⚠️ 未找到符合条件的影像。降低云量阈值试试？")
        st.stop()

    st.success(f"✅ 找到 **{len(results)}** 景影像")

    # ---- Step 2: 影像选择 ----
    st.subheader("📋 选择分析影像")
    cols = st.columns(3)
    selected_items = []
    for i, r in enumerate(results):
        with cols[i % 3]:
            try:
                preview = get_rgb_preview_cached(r["id"], collection=satellite, width=256)
            except Exception:
                preview = None
            if preview:
                st.image(preview)
            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_sup_{r['id']}",
                value=i == 0,
            )
            if selected:
                selected_items.append(r)

    if not selected_items:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    item = selected_items[0]
    st.info(f"📊 正在分析: **{item['datetime']}**")

    # ---- Step 3: 下载波段 ----
    with st.spinner("⬇️ 下载多波段数据..."):
        with StreamlitErrorBoundary("波段下载", st=st, show_traceback=False):
            tmp_path = os.path.join(tempfile.gettempdir(), f"sup_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)
            import rasterio
            with rasterio.open(tif_path) as src:
                bands_data = src.read().astype(np.float32)
                transform = src.transform
                img_crs = str(src.crs)
                img_h, img_w = src.height, src.width
            if os.path.exists(tif_path):
                os.remove(tif_path)

    bands_dict = {
        "B": bands_data[0], "G": bands_data[1], "R": bands_data[2],
        "NIR": bands_data[3], "SWIR1": bands_data[4],
        "SWIR2": bands_data[5] if bands_data.shape[0] > 5 else bands_data[4],
    }

    # ---- Step 4: 采集训练样本 ----
    st.subheader("🏷️ 训练样本采集")

    try:
        if sample_source == "从 ESA 参考产品采样":
            with st.spinner("⬇️ 获取 ESA 参考分类..."):
                esa_raw, _ = get_esa_landcover(bbox, version="v200", year="2021")
                reference_class = esa_to_arid6(esa_raw)
                # 对齐到影像尺寸
                if reference_class.shape != (img_h, img_w):
                    from skimage.transform import resize
                    reference_class = resize(
                        reference_class, (img_h, img_w), order=0,
                        preserve_range=True, anti_aliasing=False,
                    ).astype(np.int16)

            train_indices, train_labels = sample_from_reference(
                reference_class, n_samples_per_class=n_samples,
            )
            st.success(f"✅ 从 ESA 参考产品采样 **{len(train_labels)}** 个训练样本")

        elif sample_source == "上传 GeoJSON 样本":
            if geojson_file is None:
                st.error("⚠️ 请先上传 GeoJSON 样本文件")
                st.stop()
            coords, labels = sample_from_geojson(geojson_file.getvalue(), class_field=class_field)
            indices = coords_to_indices(coords, transform, crs=img_crs)
            if not indices:
                st.error("❌ 没有样本落在影像范围内，请检查坐标系")
                st.stop()
            train_indices = np.array(indices)
            train_labels = np.array(labels, dtype=np.int16)
            st.success(f"✅ 解析 **{len(train_labels)}** 个 GeoJSON 样本")

        else:  # CSV
            if csv_file is None:
                st.error("⚠️ 请先上传 CSV 样本文件")
                st.stop()
            coords, labels = sample_from_csv(csv_file.getvalue())
            indices = coords_to_indices(coords, transform, crs=img_crs)
            if not indices:
                st.error("❌ 没有样本落在影像范围内，请检查坐标")
                st.stop()
            train_indices = np.array(indices)
            train_labels = np.array(labels, dtype=np.int16)
            st.success(f"✅ 解析 **{len(train_labels)}** 个 CSV 样本")

    except Exception as e:
        st.error(f"样本采集失败: {e}")
        st.stop()

    # ---- Step 5: 训练 + 分类 ----
    st.subheader("🤖 训练 & 分类")
    with st.spinner(f"训练 {CLASSIFIERS[classifier_key]['name']} 中..."):
        with StreamlitErrorBoundary("监督分类", st=st, show_traceback=True):
            result = assess_supervised_classification(
                bands_dict=bands_dict,
                train_indices=train_indices,
                train_labels=train_labels,
                classifier=classifier_key,
                add_indices=use_indices,
                test_ratio=test_ratio,
                **classifier_params,
            )

    st.success("✅ 训练完成")

    # 写入 session_state 供报告导出页自动采集
    st.session_state["supervised_stats"] = {
        "area": area_name,
        "classifier": s["classifier"],
        "accuracy": {
            "oa": float(acc["overall_accuracy"]),
            "kappa": float(acc["kappa"]),
            "f1_macro": float(acc["f1_macro"]),
        },
        "summary": {
            "n_train": s["n_train_samples"],
            "n_test": s["n_test_samples"],
            "n_features": s["n_features"],
        },
    }

    # ---- 汇总卡片 ----
    st.subheader("📊 分类结果汇总")
    s = result.summary
    acc = result.accuracy
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总体精度 OA", f"{acc['overall_accuracy']*100:.2f}%")
    with col2:
        st.metric("Kappa 系数", f"{acc['kappa']:.3f}")
    with col3:
        st.metric("宏平均 F1", f"{acc['f1_macro']:.3f}")
    with col4:
        st.metric("训练样本数", s["n_train_samples"])

    st.caption(f"分类器: {s['classifier']} | 特征数: {s['n_features']} | 测试样本: {s['n_test_samples']}")

    st.divider()

    # ---- 标签页 ----
    tab_map, tab_cm, tab_f1 = st.tabs(["分类结果图", "混淆矩阵", "各类别精度"])

    with tab_map:
        st.caption("整景分类结果 (干旱区6类)")
        # 用实际预测的类别数
        used_classes = np.unique(result.prediction)
        fig_img = render_classification(
            result.prediction,
            class_names=ARID6_NAMES,
            class_colors=ARID6_COLORS,
            title=f"监督分类结果 — {area_name} ({item['datetime']})",
        )
        st.image(fig_img, use_container_width=True)

    with tab_cm:
        st.caption("混淆矩阵 (测试集)")
        cm = acc["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        n_cm = cm.shape[0]
        for i in range(n_cm):
            for j in range(n_cm):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        ax.set_xticks(range(n_cm))
        ax.set_yticks(range(n_cm))
        ax.set_xticklabels(ARID6_NAMES[:n_cm], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ARID6_NAMES[:n_cm], fontsize=9)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title("混淆矩阵", fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

    with tab_f1:
        st.caption("各类别 F1 分数")
        f1_data = list(acc["per_class_f1"].items())
        names = [x[0] for x in f1_data]
        f1s = [x[1] for x in f1_data]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(names, f1s, color="#1f77b4")
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylabel("F1 分数")
        ax.set_ylim(0, 1.1)
        ax.set_title("各类别分类精度 (F1)", fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ---- 方法说明 ----
    st.divider()
    st.info(
        f"💡 **提示**: 当前用 {s['classifier']} 训练，特征含 6 波段"
        f"{' + 3 遥感指数' if use_indices else ''}。"
        "可切换分类器或增加采样数提升精度；上传自己的 GeoJSON/CSV 样本可实现全自定义分类。"
    )
