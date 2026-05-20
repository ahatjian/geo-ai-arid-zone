"""
AI 地物分类页面 — 公开土地覆盖产品 + AI 深度学习推理
支持 ESA WorldCover (10m) / ESRI Land Cover (10m) 即开即用
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, AI_MODELS, MODELS_DIR, ONNX_MODELS, ONNX_CONFIG
from utils.error_handler import StreamlitErrorBoundary, safe_execute

st.set_page_config(page_title="AI 分类", page_icon="🤖", layout="wide")

# ============================================
# 侧边栏 - 模式与参数配置
# ============================================
with st.sidebar:
    st.title("🤖 地物分类设置")

    # ---- 运行模式 ----
    st.subheader("运行模式")
    run_mode = st.radio(
        "选择模式",
        [
            "🌍 公开土地覆盖产品 (ESA/ESRI)",
            "🧠 AI 深度学习推理 (模型+影像)",
        ],
        help="公开产品：ESA WorldCover / ESRI Land Cover 即开即用\nAI 推理：上传预训练模型 + Sentinel-2 影像",
    )

    st.divider()

    # ---- 研究区 ----
    if "公开" in run_mode:
        st.subheader("研究区")
        area_name = st.selectbox(
            "选择研究区",
            list(STUDY_AREAS.keys()),
            index=1,  # 默认柴达木盆地
        )
        area_info = STUDY_AREAS[area_name]
        bbox = area_info["bbox"]
        st.caption(f"范围: {bbox} | {area_info['description']}")

        st.divider()

        st.subheader("数据源")
        source = st.radio(
            "土地覆盖产品",
            ["ESA WorldCover (2021, 10m)", "ESRI Land Cover (2024, 10m)"],
            index=0,
        )
        source_key = "esa" if "ESA" in source else "esri"

        map_to_arid6 = st.checkbox("映射到干旱区6类", value=True,
                                   help="将 ESA 11类 / ESRI 9类 合并为 植被/裸地/建设用地/农田/水体/其他")

    else:
        st.subheader("模型配置")
        task_type = st.radio(
            "分类任务",
            ["🏷️ 土地覆盖 (6类)", "💧 水体分割 (2类)"],
        )
        if "土地" in task_type:
            model_cfg = AI_MODELS["land_cover"]
            onnx_default_model = ONNX_MODELS.get("land_cover", "models/landcover_unet_resnet50.onnx")
            onnx_num_classes = 6
            onnx_activation = "softmax"
        else:
            model_cfg = AI_MODELS["water_seg"]
            onnx_default_model = ONNX_MODELS.get("water_seg", "models/water_seg_unet_resnet34.onnx")
            onnx_num_classes = 2
            onnx_activation = "sigmoid"

        st.divider()

        # 模型格式选择
        st.subheader("🧠 模型格式")
        model_format = st.radio(
            "选择推理引擎",
            ["PyTorch 模型 (.pth, .pt)", "ONNX 推理加速 (.onnx)"],
            help="PyTorch: 使用 geoai-py + Torch 推理\nONNX: 使用 ONNX Runtime 独立推理 (更快、更轻量)",
            key="model_format_radio",
        )
        is_onnx_format = "ONNX" in model_format

        if is_onnx_format:
            # ONNX 配置参数
            st.subheader("⚡ ONNX 推理配置")

            # 检查 ONNX Runtime 可用性
            try:
                from utils.onnx_engine import check_onnx_available
                onnx_info = check_onnx_available()
                if onnx_info["onnx_available"]:
                    st.success(f"✅ ONNX Runtime {onnx_info['version']}")
                    if onnx_info["cuda_available"]:
                        st.success("🚀 CUDA 加速可用")
                    provider_list = ", ".join(onnx_info["providers"][:3])
                    st.caption(f"Provider: {provider_list}")
                else:
                    st.error("❌ ONNX Runtime 未安装")
                    st.code("pip install onnxruntime  # CPU\npip install onnxruntime-gpu  # GPU")
            except Exception as e:
                st.warning(f"ONNX 检测: {e}")

            # ONNX 模型来源
            onnx_model_source = st.radio(
                "模型来源",
                ["📦 上传 .onnx 模型", "📂 预设路径"],
                key="onnx_source_radio",
            )
            if "上传" in onnx_model_source:
                onnx_model_file = st.file_uploader(
                    "上传 ONNX 语义分割模型",
                    type=["onnx"],
                    key="ai_onnx_upload",
                    help="由 PyTorch 导出或公开的 ONNX 语义分割模型"
                )
                onnx_model_path = None
            else:
                onnx_model_path = st.text_input(
                    "ONNX 模型路径",
                    value=onnx_default_model,
                    help="服务器上的 ONNX 模型文件路径",
                    key="ai_onnx_path",
                )
                onnx_model_file = None
                if os.path.exists(onnx_model_path):
                    st.success(f"✅ 已找到: {os.path.basename(onnx_model_path)}")
                else:
                    st.warning(f"⚠️ 未找到: {onnx_model_path}")

            # ONNX Provider
            try:
                from utils.onnx_engine import check_onnx_available
                available_providers = check_onnx_available()["providers"]
                if not available_providers:
                    available_providers = ["CPUExecutionProvider"]
            except Exception:
                available_providers = ["CPUExecutionProvider"]
            onnx_provider = st.selectbox(
                "推理提供者",
                available_providers,
                index=0,
                help="CPUExecutionProvider: CPU 推理\nCUDAExecutionProvider: GPU 加速",
                key="ai_onnx_provider",
            )

            with st.expander("⚙️ ONNX 高级参数"):
                from config import ONNX_CONFIG
                onnx_window_size = st.slider(
                    "滑动窗口大小", 128, 2000, ONNX_CONFIG.get("window_size", 512), 64,
                    key="ai_onnx_ws",
                )
                onnx_overlap = st.slider(
                    "窗口重叠", 0, 500, ONNX_CONFIG.get("overlap", 256), 32,
                    key="ai_onnx_ol",
                )
                onnx_batch_size = st.slider(
                    "批次大小", 1, 8, ONNX_CONFIG.get("batch_size", 4),
                    key="ai_onnx_bs",
                )
        else:
            # PyTorch 推理参数
            st.subheader("PyTorch 推理参数")
            tile_size = st.number_input("Tile 大小", 128, 1024, 512, 128)
            overlap = st.number_input("重叠", 0, 512, 256, 64)

# ============================================
# 主页面
# ============================================
st.title("🤖 土地覆盖分类分析")
st.markdown(
    "*支持公开土地覆盖产品即开即用，也可上传模型进行 AI 深度学习推理*"
)

# ============================================
# 模式 A: 公开土地覆盖产品
# ============================================
if "公开" in run_mode:
    st.markdown("---")
    st.subheader(f"🌍 {source} — {area_name}")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("分辨率", "10m")
    with col_info2:
        st.metric("研究区经度", f"{bbox[0]:.1f}° ~ {bbox[2]:.1f}°")
    with col_info3:
        st.metric("研究区纬度", f"{bbox[1]:.1f}° ~ {bbox[3]:.1f}°")

    # ---- 切片信息 ----
    with st.expander("📐 覆盖的 ESA 切片", expanded=False):
        with StreamlitErrorBoundary("ESA 切片查询", st=st, show_traceback=False):
            from utils.landcover import get_tile_list_for_area
            tiles = get_tile_list_for_area(area_name, STUDY_AREAS)
            if tiles:
                tile_df = pd.DataFrame([
                    {"切片": t["name"], "边界": f"[{t['bbox'][0]}, {t['bbox'][1]}, {t['bbox'][2]}, {t['bbox'][3]}]"}
                    for t in tiles
                ])
                st.dataframe(tile_df, use_container_width=True)
                st.caption("COG 格式，只拉取需要的窗口区域，无需下载完整文件")
            else:
                st.info("无覆盖切片")

    # ---- 执行按钮 ----
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_btn = st.button("🚀 获取土地覆盖数据", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"正在从 {source} 拉取土地覆盖数据... (COG 远程读取，首次可能较慢)"):
            with StreamlitErrorBoundary("土地覆盖数据获取", st=st, show_traceback=True):
                from utils.landcover import get_landcover_for_study_area

                result = get_landcover_for_study_area(
                    bbox, source=source_key, map_to_arid6=map_to_arid6,
                )

                if not result["success"]:
                    st.error(f"❌ 数据获取失败: {result.get('error', '未知错误')}")
                    if "traceback" in result:
                        with st.expander("🔍 错误详情"):
                            st.code(result["traceback"])
                    st.stop()

                # 保存到 session_state
                st.session_state["lc_result"] = result
                st.session_state["lc_bbox"] = bbox
                st.session_state["lc_source"] = source_key
                st.session_state["lc_arid6"] = map_to_arid6

    # ---- 展示结果 ----
    if "lc_result" in st.session_state and st.session_state["lc_bbox"] == bbox:
        result = st.session_state["lc_result"]
        class_array = result["class_array"]
        meta = result["meta"]
        stats = result["stats"]
        class_names = result["class_names"]
        class_colors = result["class_colors"]

        st.markdown("---")
        st.subheader("📊 分类统计")

        # 统计表格
        stat_df = pd.DataFrame([
            {
                "类别": s["class_name"],
                "像元数": f"{s['pixel_count']:,}",
                "占比": f"{s['pixel_ratio']*100:.2f}%",
                "面积 (km²)": f"{s['area_km2']:.2f}",
            }
            for s in stats if s["pixel_count"] > 0
        ])
        st.dataframe(stat_df, use_container_width=True)

        # ---- 可视化: 柱状图 + 饼图 ----
        col_chart1, col_chart2 = st.columns(2)

        # 只取有数据的类别
        valid_labels = [s["class_name"] for s in stats if s["pixel_count"] > 0]
        valid_areas = [s["area_km2"] for s in stats if s["pixel_count"] > 0]
        valid_colors = [s["class_color"] for s in stats if s["pixel_count"] > 0]

        with col_chart1:
            # 面积柱状图
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=valid_labels,
                y=valid_areas,
                marker_color=valid_colors,
                text=[f"{v:.1f}" for v in valid_areas],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="各类别面积 (km²)",
                height=400,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="",
                yaxis_title="面积 (km²)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            # 饼图
            fig_pie = go.Figure()
            fig_pie.add_trace(go.Pie(
                labels=valid_labels,
                values=[s["pixel_count"] for s in stats if s["pixel_count"] > 0],
                marker_colors=valid_colors,
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                title="各类别占比",
                height=400,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ---- 分类地图 ----
        st.subheader("🗺️ 分类可视化地图")

        from utils.visualization import render_classification

        # 构建渲染用的名称和颜色列表（按类别值排序）
        sorted_classes = sorted(class_names.items())
        render_names = [name for _, name in sorted_classes]
        render_colors = [class_colors[k] for k, _ in sorted_classes]

        class_img = render_classification(
            class_array,
            class_names=render_names,
            class_colors=render_colors,
            title=f"{result['source_label']} | {area_name}",
        )
        st.image(class_img, use_container_width=True)

        # ---- 色块图例 ----
        color_html = '<div style="margin:10px 0;display:flex;flex-wrap:wrap;gap:8px;">'
        for cls_val, cls_name in sorted_classes:
            color = class_colors.get(cls_val, "#888888")
            color_html += (
                f'<span style="display:flex;align-items:center;margin-right:16px;">'
                f'<span style="display:inline-block;width:16px;height:16px;'
                f'background:{color};border-radius:3px;margin-right:6px;"></span>'
                f'<span style="font-size:0.85rem;">{cls_name}</span>'
                f'</span>'
            )
        color_html += '</div>'
        st.markdown(color_html, unsafe_allow_html=True)

        # ---- 导出 ----
        st.markdown("---")
        st.subheader("📥 导出结果")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            # GeoTIFF 导出
            if st.button("💾 导出分类 GeoTIFF", use_container_width=True):
                with StreamlitErrorBoundary("GeoTIFF 导出", st=st):
                    from utils.landcover import export_landcover_geotiff
                    import tempfile
                    fname = f"landcover_{area_name}_{source_key}.tif"
                    out_path = os.path.join(tempfile.gettempdir(), fname)
                    export_landcover_geotiff(
                        class_array,
                        out_path,
                        transform=meta.get("transform"),
                        crs=meta.get("crs", "EPSG:4326"),
                        class_names=class_names,
                    )
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇️ 下载 GeoTIFF",
                            f,
                            file_name=fname,
                            mime="image/tiff",
                            use_container_width=True,
                        )
                    st.success(f"✅ 已导出: {fname}")

        with col_exp2:
            # CSV 统计表导出
            csv_data = stat_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 下载统计表 (CSV)",
                csv_data,
                file_name=f"landcover_stats_{area_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ============================================
# 模式 B: AI 深度学习推理
# ============================================
else:
    st.markdown("---")

    # 分类体系预览
    st.subheader("📋 分类体系")
    if "土地" in task_type:
        class_names = model_cfg["class_names"]
        class_colors = model_cfg["class_colors"]
    else:
        class_names = model_cfg["class_names"]
        class_colors = model_cfg["class_colors"]

    color_html = ""
    for name, color in zip(class_names, class_colors):
        color_html += (
            f'<span style="display:inline-block;width:18px;height:18px;'
            f'background:{color};border-radius:4px;margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f'<span style="margin-right:20px;font-size:0.9rem;">{name}</span>'
        )
    st.markdown(f"<div style='margin:10px 0;'>{color_html}</div>", unsafe_allow_html=True)

    # 数据输入
    st.subheader("📁 输入数据")
    input_mode = st.radio(
        "数据来源",
        ["📁 上传 GeoTIFF (多波段)", "📡 从数据浏览页获取"],
        horizontal=True,
        key="ai_input_mode",
    )

    if "上传" in input_mode:
        uploaded = st.file_uploader(
            "上传多波段 GeoTIFF (建议 Sentinel-2 L2A, ≥6 波段)",
            type=["tif", "tiff"],
            key="ai_upload",
            help="波段顺序: Blue, Green, Red, NIR, SWIR1, SWIR2",
        )
        if uploaded:
            tmp_dir = tempfile.gettempdir()
            geotiff_path = os.path.join(tmp_dir, f"ai_class_{uploaded.name}")
            with open(geotiff_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"✅ 已加载: {uploaded.name}")

            import rasterio
            with rasterio.open(geotiff_path) as src:
                st.info(
                    f"影像: {src.width} × {src.height} px | "
                    f"波段: {src.count} | CRS: {src.crs} | "
                    f"数据类型: {src.dtypes[0]}"
                )
            st.session_state["ai_geotiff_path"] = geotiff_path
        else:
            geotiff_path = None
    else:
        st.info("👆 请先在「数据浏览」页面搜索并下载影像，然后回到本页选择「上传 GeoTIFF」")
        geotiff_path = None

    # 模型上传 (根据格式动态切换)
    st.subheader("🧠 模型文件")
    if is_onnx_format:
        # ONNX 模型已在侧边栏处理，此处显示状态
        if onnx_model_file is not None:
            model_file = onnx_model_file
            model_file_type = "onnx"
            st.success(f"✅ ONNX 模型已上传: {onnx_model_file.name}")
        elif onnx_model_path is not None:
            model_file = None  # 路径模式不需要上传对象
            model_file_type = "onnx_path"
            actual_onnx_path = onnx_model_path
        else:
            model_file = None
            model_file_type = None
            st.warning("⚠️ 请在左侧边栏选择或上传 ONNX 模型")
    else:
        model_file = st.file_uploader(
            "上传预训练模型 (.pth, .pt)",
            type=["pth", "pt"],
            key="ai_model",
            help="由 geoai-py 训练的分割模型，或从公开源下载的预训练权重",
        )
        model_file_type = "pytorch"

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        # 根据模型格式判断按钮是否可用
        if is_onnx_format:
            btn_disabled = geotiff_path is None or (onnx_model_file is None and not os.path.exists(onnx_model_path or ""))
            btn_label = "⚡ 运行 ONNX 推理"
        else:
            btn_disabled = geotiff_path is None or model_file is None
            btn_label = "🔮 运行 AI 推理"
        run_ai_btn = st.button(
            btn_label,
            type="primary",
            disabled=btn_disabled,
            use_container_width=True,
        )
    with col_btn2:
        run_baseline_btn = st.button(
            "📊 指数阈值分类 (基线)",
            disabled=geotiff_path is None,
            use_container_width=True,
        )

    # ---- AI 推理执行 ----
    if run_ai_btn and ((not is_onnx_format and model_file) or (is_onnx_format and (onnx_model_file or (onnx_model_path and os.path.exists(onnx_model_path))))) and geotiff_path:
        # ============================================
        # ONNX 推理分支
        # ============================================
        if is_onnx_format:
            with st.spinner("⚡ 初始化 ONNX 推理引擎..."):
                with StreamlitErrorBoundary("ONNX 深度学习推理", st=st, show_traceback=True):
                    from utils.onnx_engine import (
                        check_onnx_available, run_onnx_inference, inspect_onnx_model
                    )
                    from utils.visualization import render_classification
                    import rasterio

                    # 检查 ONNX Runtime
                    ort_info = check_onnx_available()
                    if not ort_info["onnx_available"]:
                        st.error("❌ ONNX Runtime 未安装。请运行: pip install onnxruntime")
                        st.stop()

                    # 准备模型路径
                    if onnx_model_file is not None:
                        tmp_model = os.path.join(MODELS_DIR, f"uploaded_{onnx_model_file.name}")
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        with open(tmp_model, "wb") as f:
                            f.write(onnx_model_file.getvalue())
                        actual_onnx_path = tmp_model
                        st.success(f"✅ ONNX 模型已加载: {onnx_model_file.name}")
                    else:
                        actual_onnx_path = onnx_model_path

                    if not os.path.exists(actual_onnx_path):
                        st.error(f"❌ ONNX 模型文件不存在: {actual_onnx_path}")
                        st.stop()

                    # 读取输入影像
                    with rasterio.open(geotiff_path) as src:
                        st.info(
                            f"📐 影像: {src.width}×{src.height} px | "
                            f"波段: {src.count} | CRS: {src.crs}\n\n"
                            f"🧠 ONNX: {os.path.basename(actual_onnx_path)} | "
                            f"Provider: {onnx_provider} | "
                            f"窗口: {onnx_window_size}px | 重叠: {onnx_overlap}px"
                        )
                        n_bands = min(src.count, 10)
                        bands = [src.read(i+1).astype(np.float32) for i in range(n_bands)]
                        input_array = np.stack(bands, axis=0)
                        profile = src.profile.copy()

                    # 数据标准化
                    for c in range(input_array.shape[0]):
                        c_min, c_max = np.percentile(input_array[c], [2, 98])
                        if c_max > c_min:
                            input_array[c] = np.clip((input_array[c] - c_min) / (c_max - c_min), 0, 1)

                    # 进度条
                    progress_bar = st.progress(0.0)
                    progress_text = st.empty()

                    def onnx_progress(completed, total):
                        pct = completed / max(total, 1)
                        progress_bar.progress(min(pct, 1.0))
                        progress_text.text(f"ONNX 推理中... {completed}/{total} 窗口")

                    # 执行 ONNX 推理
                    st.info(f"🚀 开始 ONNX 推理 | {onnx_num_classes} 类语义分割...")
                    onnx_result = run_onnx_inference(
                        onnx_model_path=actual_onnx_path,
                        input_array=input_array,
                        window_size=onnx_window_size,
                        overlap=onnx_overlap,
                        batch_size=onnx_batch_size,
                        num_classes=onnx_num_classes,
                        activation=onnx_activation,
                        progress_callback=onnx_progress,
                        provider=onnx_provider,
                    )

                    progress_bar.progress(1.0)
                    progress_text.empty()

                    class_array = onnx_result["output"]
                    inference_time = onnx_result["inference_time_s"]
                    num_windows = onnx_result.get("num_windows", 0)

                    st.success(
                        f"✅ ONNX 推理完成 | "
                        f"耗时: {inference_time:.1f}s | "
                        f"窗口: {num_windows} | "
                        f"{onnx_num_classes} 类"
                    )

                    # 保存结果到 session_state
                    st.session_state["ai_class_result"] = class_array
                    st.session_state["ai_class_names"] = model_cfg["class_names"]
                    st.session_state["ai_class_colors"] = model_cfg["class_colors"]
                    st.session_state["ai_inference_time"] = inference_time
                    st.session_state["ai_num_windows"] = num_windows
                    st.session_state["ai_is_onnx"] = True
                    st.session_state["ai_geotiff_profile"] = profile

                    # ONNX 模型检查
                    with st.expander("🔍 ONNX 模型信息"):
                        try:
                            model_info = inspect_onnx_model(actual_onnx_path)
                            if "error" not in model_info:
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric("IR 版本", model_info.get("ir_version", "N/A"))
                                with col_m2:
                                    st.metric("模型大小", f"{model_info.get('model_size_mb', 0):.1f} MB")
                                with col_m3:
                                    st.metric("算子节点", model_info.get("num_nodes", "N/A"))
                                st.text(f"输入: {model_info.get('inputs', [])}")
                                st.text(f"输出: {model_info.get('outputs', [])}")
                            else:
                                st.warning(model_info["error"])
                        except Exception as e:
                            st.warning(f"模型检查失败: {e}")

        # ============================================
        # PyTorch 推理分支 (原有逻辑)
        # ============================================
        else:
            with st.spinner("⚙️ 初始化 AI 推理引擎..."):
                st.info("""
                ### ⚠️ AI 推理需要完整依赖

                当前预期：
                - `geoai-py` 库中的 `semantic_segmentation()` / `segment_water()` 方法
                - 或 `torch` + `segmentation_models_pytorch` 的自定义推理管线

                **推荐操作路径**:
                1. **水体分割**: 使用 `geoai.segment_water()` 内置 OmniWaterMask 模型
                2. **土地覆盖**: 训练自定义 UNet 模型或使用公开产品 (切到 ESA/ESRI 模式)

                正在加载模型...
                """)

                with StreamlitErrorBoundary("AI 深度学习推理", st=st, show_traceback=True):
                    import torch
                    import geoai

                    # 临时保存用户上传的模型文件
                    model_path = os.path.join(MODELS_DIR, "uploaded_model.pth")
                    with open(model_path, "wb") as f:
                        f.write(model_file.getvalue())

                    st.success("✅ 模型文件已加载")

                    # 执行推理
                    if "水体" in task_type and hasattr(geoai, "segment_water"):
                        output_path = os.path.join(
                            tempfile.gettempdir(),
                            "water_mask_result.tif",
                        )
                        result = geoai.segment_water(
                            input_path=geotiff_path,
                            band_order="sentinel2",
                            output_raster=output_path,
                            patch_size=tile_size,
                            overlap_size=overlap,
                        )
                        st.success(f"✅ 水体分割完成 → {output_path}")

                    elif hasattr(geoai, "semantic_segmentation"):
                        output_path = os.path.join(
                            tempfile.gettempdir(),
                            "landcover_class_result.tif",
                        )
                        result = geoai.semantic_segmentation(
                            input_path=geotiff_path,
                            output_path=output_path,
                            model_path=model_path,
                            architecture="unet",
                            encoder_name="resnet34",
                            num_channels=6,
                            num_classes=model_cfg["num_classes"],
                            window_size=tile_size,
                            overlap=overlap,
                        )
                        st.success(f"✅ 语义分割完成 → {output_path}")

                    else:
                        st.warning("geoai-py 未提供推理接口，请使用下方基线方法或更新 geoai-py")

    # ---- ONNX 推理结果展示 ----
    if is_onnx_format and "ai_class_result" in st.session_state:
        class_array = st.session_state["ai_class_result"]
        class_names_list = st.session_state["ai_class_names"]
        class_colors_list = st.session_state["ai_class_colors"]
        inference_time = st.session_state.get("ai_inference_time", 0)
        num_windows = st.session_state.get("ai_num_windows", 0)

        st.markdown("---")
        st.subheader("📊 ONNX 推理结果")

        # KPI 卡片
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        total_pixels = int(class_array.size)
        with col_kpi1:
            st.metric("⚡ 推理耗时", f"{inference_time:.1f} s")
        with col_kpi2:
            st.metric("🪟 处理窗口", num_windows)
        with col_kpi3:
            st.metric("📐 总像元", f"{total_pixels:,}")
        with col_kpi4:
            st.metric("🏷️ 类别数", onnx_num_classes)

        # 分类统计
        stat_rows = []
        for cls_idx in range(onnx_num_classes):
            cnt = int(np.sum(class_array == cls_idx))
            pct = cnt / max(total_pixels, 1) * 100
            name = class_names_list[cls_idx] if cls_idx < len(class_names_list) else f"类{cls_idx}"
            color = class_colors_list[cls_idx] if cls_idx < len(class_colors_list) else "#888888"
            stat_rows.append({
                "类别": name,
                "像元数": f"{cnt:,}",
                "占比": f"{pct:.2f}%",
                "颜色": color,
            })
        stat_df = pd.DataFrame(stat_rows)

        # 只展示有数据的类别
        valid_rows = [r for r in stat_rows if int(r["像元数"].replace(",", "")) > 0]
        if valid_rows:
            st.dataframe(
                pd.DataFrame([{k: v for k, v in r.items() if k != "颜色"} for r in valid_rows]),
                use_container_width=True
            )

        # 可视化: 柱状图 + 饼图
        col_ch1, col_ch2 = st.columns(2)

        valid_labels = [r["类别"] for r in valid_rows]
        valid_counts = [int(r["像元数"].replace(",", "")) for r in valid_rows]
        valid_colors = [r["颜色"] for r in valid_rows]

        with col_ch1:
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=valid_labels,
                y=valid_counts,
                marker_color=valid_colors,
                text=[f"{v:,}" for v in valid_counts],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="各类别像元数",
                height=400,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_ch2:
            fig_pie = go.Figure()
            fig_pie.add_trace(go.Pie(
                labels=valid_labels,
                values=valid_counts,
                marker_colors=valid_colors,
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                title="各类别占比",
                height=400,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # 分类地图
        st.subheader("🗺️ ONNX 分类地图")
        from utils.visualization import render_classification
        class_img = render_classification(
            class_array,
            class_names=class_names_list[:onnx_num_classes],
            class_colors=class_colors_list[:onnx_num_classes],
            title=f"ONNX 语义分割 | {'土地覆盖' if '土地' in task_type else '水体分割'} | 耗时 {inference_time:.1f}s",
        )
        st.image(class_img, use_container_width=True)

        # 图例
        legend_html = '<div style="margin:10px 0;display:flex;flex-wrap:wrap;gap:8px;">'
        for cls_idx in range(onnx_num_classes):
            name = class_names_list[cls_idx] if cls_idx < len(class_names_list) else f"类{cls_idx}"
            color = class_colors_list[cls_idx] if cls_idx < len(class_colors_list) else "#888888"
            legend_html += (
                f'<span style="display:flex;align-items:center;margin-right:14px;">'
                f'<span style="width:14px;height:14px;background:{color};'
                f'border-radius:3px;margin-right:5px;"></span>'
                f'<span style="font-size:0.85rem;">{name}</span></span>'
            )
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

        # 导出
        st.markdown("---")
        st.subheader("📥 导出 ONNX 推理结果")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = stat_df[["类别", "像元数", "占比"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 下载统计表 (CSV)",
                csv_data,
                file_name=f"onnx_class_stats_{task_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_exp2:
            # 导出分类结果为 NumPy
            np_bytes = BytesIO()
            np.save(np_bytes, class_array)
            st.download_button(
                "💾 下载分类数组 (.npy)",
                np_bytes.getvalue(),
                file_name=f"onnx_class_array_{task_type.replace(' ', '_')}.npy",
                mime="application/octet-stream",
                use_container_width=True,
            )

    # ---- 基线指数阈值分类 ----
    if run_baseline_btn and geotiff_path:
        with st.spinner("正在计算指数并分类..."):
            with StreamlitErrorBoundary("基线指数阈值分类", st=st, show_traceback=True):
                from utils.indices import calc_ndvi, calc_mndwi
                from utils.visualization import render_classification
                import rasterio

                with rasterio.open(geotiff_path) as src:
                    bands_needed = 6
                    if src.count < bands_needed:
                        st.error(f"需要至少 {bands_needed} 个波段，当前: {src.count}")
                        st.stop()
                    blue = src.read(1).astype(np.float32)
                    green = src.read(2).astype(np.float32)
                    red = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                    swir1 = src.read(5).astype(np.float32)

                ndvi = calc_ndvi(red, nir)
                mndwi = calc_mndwi(green, swir1)

                # 干旱区6类阈值分类
                class_map = np.zeros_like(ndvi, dtype=np.uint8)
                class_map[mndwi > 0.05] = 5  # 水体 (MNDWI > 0.05)
                class_map[(ndvi > 0.5) & (class_map == 0)] = 1  # 密植被
                class_map[(ndvi > 0.25) & (class_map == 0)] = 4  # 农田/稀疏植被
                class_map[(ndvi > 0.1) & (class_map == 0)] = 1  # 稀疏植被 → 植被
                class_map[(ndvi < -0.1) & (class_map == 0)] = 2  # 裸地 (很低的 NDVI)
                # 其余默认为 2 (裸地，干旱区主体)

                # 替换 NaN
                class_map[~np.isfinite(ndvi)] = 0

                st.success("✅ 基线阈值分类完成")

                # 统计
                class_labels = {0: "其他", 1: "植被", 2: "裸地", 3: "建设用地", 4: "农田", 5: "水体"}
                class_color_map = {0: "#000000", 1: "#00CC44", 2: "#CCCCCC", 3: "#FF4444", 4: "#FFAA00", 5: "#0066FF"}
                unique, counts = np.unique(class_map, return_counts=True)
                total = class_map.size

                st.subheader("📊 分类统计")
                stat_rows = []
                for cls, cnt in zip(unique, counts):
                    stat_rows.append({
                        "类别": class_labels.get(cls, f"类{cls}"),
                        "像元数": f"{cnt:,}",
                        "占比": f"{cnt/total*100:.2f}%",
                    })
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

                # 可视化
                render_names = [class_labels[i] for i in range(6)]
                render_colors = [class_color_map[i] for i in range(6)]
                img = render_classification(
                    class_map,
                    class_names=render_names,
                    class_colors=render_colors,
                    title="指数阈值分类结果 (NDVI + MNDWI 基线)",
                )
                st.image(img, use_container_width=True)

                # 图例
                legend_html = '<div style="margin:10px 0;display:flex;flex-wrap:wrap;gap:8px;">'
                for i in range(6):
                    legend_html += (
                        f'<span style="display:flex;align-items:center;margin-right:14px;">'
                        f'<span style="width:14px;height:14px;background:{class_color_map[i]};'
                        f'border-radius:3px;margin-right:5px;"></span>'
                        f'<span style="font-size:0.85rem;">{class_labels[i]}</span></span>'
                    )
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

    # ---- 指南区 ----
    st.markdown("---")
    with st.expander("📖 AI 推理完整指南"):
        st.markdown("""
        ### 🧠 如何获得预训练模型

        **方案 A: geoai-py 水体分割 (推荐)**
        ```python
        import geoai
        # geoai.segment_water() 内置 OmniWaterMask，无需额外模型
        result = geoai.segment_water("sentinel2_image.tif")
        ```

        **方案 B: 训练自定义土地覆盖模型**
        ```python
        import geoai
        # train_segmentation_model 自动保存最佳模型到 output_dir
        geoai.train_segmentation_model(
            images_dir='data/training/images/',
            labels_dir='data/training/masks/',
            output_dir='models/',
            architecture='unet',
            encoder_name='resnet50',
            num_channels=6,
            num_classes=6,
            num_epochs=80
        )
        # 模型自动保存至: models/best_model.pth
        ```

        **方案 C: 公开数据集**
        - Zenodo `dset-s2`: 800+ 标注水体影像
        - GLOBELAND: 全球 10m 土地覆盖标签

        ---

        ### ⚡ ONNX 模型导出 (PyTorch → ONNX)

        训练完成后可将 PyTorch 模型导出为 ONNX 格式，获得 2-5x 推理加速：

        ```python
        from utils.onnx_engine import export_to_onnx
        import torch

        # 加载训练好的 PyTorch 模型
        model = torch.load('models/best_model.pth', map_location='cpu')
        model.eval()

        # 导出为 ONNX
        result = export_to_onnx(
            model=model,
            output_path='models/landcover_unet.onnx',
            input_shape=(1, 6, 512, 512),  # (batch, channels, H, W)
            opset_version=13,
            simplify=True,
            verify=True,
        )
        print(f"导出成功: {result['output_path']} ({result['model_size_mb']} MB)")
        ```

        **ONNX 推理（无需 PyTorch/geoai-py）:**
        ```python
        from utils.onnx_engine import run_onnx_inference
        import numpy as np

        # 直接使用 ONNX Runtime 推理
        img = np.load('sentinel2_patch.npy')  # (C, H, W)
        result = run_onnx_inference(
            onnx_model_path='models/landcover_unet.onnx',
            input_array=img,
            window_size=512,
            overlap=256,
            num_classes=6,
            activation='softmax',   # 多类别
            # activation='sigmoid', # 二分类(水体)
        )
        class_map = result['output']  # (H, W) 分类结果
        ```

        **性能对比:** ONNX Runtime 在 CPU 上比 PyTorch 快 2-3x，GPU 上快 3-5x。

        ---

        💡 **提示**: 没有模型时，可使用「🌍 公开土地覆盖产品」模式，
        直接获取 ESA WorldCover / ESRI Land Cover 官方分类结果。
        """)
