"""
AI 智能分析助手页面 — 第 25 模块 (Geo AI 核心差异化)
======================================================
AI 主导的三大能力:

  💬 AI 对话助手: 遥感知识问答 + 平台引导 + 结果解读 (多轮对话)
  🪄 AI 一键分析: 上传影像 → AI 自动决定分析方法 → 自动执行 → AI 报告
  🕵️ AI 异常检测: 自动扫描影像异常区域 → AI 成因解读
"""

import os
import sys
import tempfile
import uuid as _uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI 智能助手", page_icon="🤖", layout="wide")

st.title("🤖 AI 智能分析助手")
st.markdown(
    "**对话 · 自动分析 · 异常检测** — AI 主导的 Geo AI 核心体验"
)

# ============================================================
# 辅助
# ============================================================

MAX_RASTER_PIXELS = 50_000_000


def load_bands(path: str) -> np.ndarray:
    """读取多波段栅格 (尺寸限制含波段数, 防多波段绕过)。"""
    import rasterio
    with rasterio.open(path) as src:
        # 总像元 = 波段数 × 宽 × 高 (防恶意多波段文件)
        total_pixels = src.count * src.width * src.height
        if total_pixels > MAX_RASTER_PIXELS:
            raise ValueError(
                f"栅格过大 ({src.count}波段×{src.width}×{src.height} = "
                f"{total_pixels/1e6:.0f} MP, 上限 {MAX_RASTER_PIXELS/1e6:.0f} MP)"
            )
        # 只读前 6 个波段 (平台标准波段数, 限制分配)
        n_read = min(src.count, 6)
        return src.read(list(range(1, n_read + 1))).astype(np.float64)


def save_upload_tmp(uploaded) -> str:
    safe_name = f"ai_{_uuid.uuid4().hex[:12]}.tif"
    path = os.path.join(tempfile.gettempdir(), safe_name)
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    return path


from utils.ai_assistant import (
    chat_with_assistant, auto_analyze, detect_anomalies,
    anomaly_insight, is_ai_available,
)

AI_ON = is_ai_available()

tab_chat, tab_auto, tab_anom, tab_vision, tab_quality = st.tabs([
    "💬 AI 对话助手",
    "🪄 AI 一键分析",
    "🕵️ AI 异常检测",
    "👁️ AI 影像理解",
    "🏥 AI 质量诊断",
])

# ============================================================
# Tab 1: AI 对话助手
# ============================================================
with tab_chat:
    st.subheader("💬 AI 对话助手")
    st.caption("问遥感知识、问平台操作、解读分析结果 — AI 全程引导")

    if not AI_ON:
        st.warning("⚠️ 未配置 DEEPSEEK_API_KEY — 请参考 README 配置后使用")

    # 初始化对话历史
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # 首页带来的待回答问题 → 自动发起对话
    pending_q = st.session_state.pop("pending_ai_question", None)
    if pending_q:
        st.session_state["chat_history"].append({"role": "user", "content": pending_q})
        # 自动生成 AI 回复 (隐私保护: 仅明确要求解读分析时注入上下文)
        wants_context = any(k in pending_q for k in
                            ["解读我的分析", "我的分析", "分析结果说明", "我刚才"])
        resp = chat_with_assistant(
            st.session_state["chat_history"],
            include_context=wants_context,
        )
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": resp["reply"]}
        )

    # AI 感知状态: 显示当前已完成的分析
    from utils.ai_assistant import build_platform_context
    platform_ctx = build_platform_context()
    if "尚未完成" not in platform_ctx:
        with st.expander("🧠 AI 已感知你的分析状态", expanded=False):
            st.caption(platform_ctx.replace("\n", "  \n"))
            st.caption("💡 现在可以问 AI：'我刚才的分析说明了什么？'")

    # 快捷问题
    st.markdown("**📌 快捷提问**")
    q_cols = st.columns(3)
    quick_questions = [
        "NDVI 和 EVI 有什么区别？",
        "如何分析塔里木盆地的盐渍化？",
        "帮我解读：NDVI 均值 0.18 说明什么？",
    ]
    # 有分析结果时加入解读快捷问题
    if "尚未完成" not in platform_ctx:
        quick_questions[2] = "我刚才的分析结果说明了什么？"
    for i, q in enumerate(quick_questions):
        with q_cols[i]:
            if st.button(q, key=f"quick_q_{i}", width="stretch"):
                st.session_state["chat_history"].append({"role": "user", "content": q})

    st.divider()

    # 对话展示
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入
    user_input = st.chat_input("问我任何遥感/GIS 问题，或让我引导你分析...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                # 隐私保护: 仅当用户明确要求解读其分析时注入上下文
                wants_context = any(k in user_input for k in
                                    ["解读我的分析", "我的分析", "分析结果说明", "我刚才"])
                resp = chat_with_assistant(
                    st.session_state["chat_history"],
                    include_context=wants_context,
                )
            reply = resp["reply"]
            st.markdown(reply)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    if st.session_state["chat_history"]:
        if st.button("🗑️ 清空对话", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

# ============================================================
# Tab 2: AI 一键分析
# ============================================================
with tab_auto:
    st.subheader("🪄 AI 一键分析")
    st.caption("上传影像 → AI 自动计算指数/判断地物构成 → 生成综合评估")

    auto_file = st.file_uploader("上传多波段 GeoTIFF (6波段: B,G,R,NIR,SWIR1,SWIR2)",
                                 type=["tif", "tiff"], key="auto_upload")

    if auto_file:
        auto_path = save_upload_tmp(auto_file)
        try:
            bands = load_bands(auto_path)
            st.success(f"✅ 已加载 {bands.shape[0]} 波段 ({bands.shape[1]}×{bands.shape[2]})")
        except Exception as e:
            st.error(f"❌ 读取失败: {e}")
            bands = None
        finally:
            # 读取后清理临时文件 (防泄漏)
            try:
                os.remove(auto_path)
            except OSError:
                pass

        if bands is not None:
            if st.button("🪄 开始 AI 自动分析", type="primary"):
                with st.spinner("AI 分析中（指数计算 → 地物判断 → 综合解读）..."):
                    result = auto_analyze(bands)

                st.divider()
                st.subheader("📊 AI 自动分析结果")

                # 地物构成
                st.markdown("**🏞️ 地物构成 (AI 自动判断)**")
                comp_cols = st.columns(4)
                for i, c in enumerate(result["land_composition"][:4]):
                    with comp_cols[i]:
                        st.metric(c["name"], f"{c['ratio']*100:.1f}%")

                # 光谱指数
                st.markdown("**🧮 自动计算的指数**")
                idx_cols = st.columns(3)
                for i, (name, stats) in enumerate(result["indices"].items()):
                    with idx_cols[i]:
                        st.metric(f"{name} 均值", f"{stats['mean']:.4f}",
                                  delta=f"std {stats['std']:.4f}")

                # AI 综合评估
                st.markdown("**🤖 AI 综合评估**")
                ai_esc = __import__("html").escape(result["ai_assessment"] or "")
                st.markdown(
                    f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                    f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                    f"{ai_esc}</div>",
                    unsafe_allow_html=True,
                )

                # 保存到下载中心
                st.divider()
                from utils.save_ui import render_save_button
                render_save_button(
                    default_name="AI自动分析结果",
                    data=result["land_composition"],
                    kind="csv",
                    meta={"模块": "AI智能助手", "方法": "auto_analyze"},
                    key_suffix="auto_result",
                )

    else:
        st.info("👆 上传影像后 AI 将自动完成分析")

# ============================================================
# Tab 3: AI 异常检测
# ============================================================
with tab_anom:
    st.subheader("🕵️ AI 异常检测")
    st.caption("AI 自动扫描影像中与周边显著不同的区域（突变/退化/异常）")

    anom_source = st.radio(
        "分析对象",
        ["📤 上传指数 GeoTIFF", "🌿 植被指数 (session)", "🧂 盐渍化结果 (session)"],
        horizontal=True,
    )

    anom_band = None
    anom_name = "指标"
    if "上传" in anom_source:
        anom_file = st.file_uploader("上传指标图 (NDVI/LST/盐分等)", type=["tif", "tiff"],
                                     key="anom_upload")
        if anom_file:
            anom_path = save_upload_tmp(anom_file)
            try:
                anom_band = load_bands(anom_path)
                anom_name = os.path.basename(anom_file.name)
            except Exception as e:
                st.error(f"❌ 读取失败: {e}")
            finally:
                try:
                    os.remove(anom_path)
                except OSError:
                    pass
    elif "植被" in anom_source and "veg_index" in st.session_state:
        anom_band = st.session_state["veg_index"]
        anom_name = "NDVI"
        st.caption("✅ 使用植被分析页的 NDVI")
    elif "盐渍化" in anom_source and "salinity_stats" in st.session_state:
        st.info("💡 请先在「土壤盐渍化」页完成分析")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        anom_method = st.selectbox("检测方法", ["local", "mad", "zscore"],
                                   format_func=lambda x: {
                                       "local": "局部异常 (推荐)",
                                       "mad": "全局 MAD",
                                       "zscore": "全局 Z-score",
                                   }[x])
    with col_a2:
        anom_thresh = st.slider("异常阈值", 1.0, 6.0, 3.0, 0.5)
    with col_a3:
        anom_kernel = st.slider("邻域大小", 3, 15, 7, step=2)

    if anom_band is not None:
        if st.button("🕵️ 开始异常检测", type="primary"):
            with st.spinner("AI 扫描异常区域..."):
                anom_result = detect_anomalies(
                    anom_band, method=anom_method,
                    threshold=anom_thresh, kernel=anom_kernel,
                )

            st.divider()
            st.subheader("📊 异常检测结果")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("异常区域", f"{anom_result['anomaly_ratio']*100:.2f}%")
            with c2:
                st.metric("异常簇数", anom_result["n_clusters"])
            with c3:
                st.metric("方法", anom_result["method"].split(" ")[0])

            # 可视化
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            im0 = axes[0].imshow(anom_band, cmap="viridis", interpolation="nearest")
            axes[0].set_title(f"{anom_name} 原始")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], shrink=0.8)

            overlay = np.stack([
                np.full_like(anom_band, 0.9),  # R 高 → 红色标记
                np.full_like(anom_band, 0.1),
                np.full_like(anom_band, 0.1),
            ], axis=-1)
            norm_band = (anom_band - np.nanmin(anom_band)) / (
                np.nanmax(anom_band) - np.nanmin(anom_band) + 1e-6)
            gray = np.stack([norm_band] * 3, axis=-1)
            display = np.where(anom_result["anomaly_mask"][:, :, None], overlay, gray)
            axes[1].imshow(np.clip(display, 0, 1))
            axes[1].set_title("异常区域 (红色标记)")
            axes[1].axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # AI 成因解读
            st.subheader("🤖 AI 异常成因解读")
            with st.spinner("AI 分析异常成因..."):
                insight = anomaly_insight(anom_band, anom_result, anom_name)
            ai_esc = __import__("html").escape(insight or "")
            st.markdown(
                f"<div style='background:#fdf0f0;border-left:4px solid #e74c3c;"
                f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                f"🕵️ **AI 解读**：{ai_esc}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("👆 上传指标图或先在植被分析页生成 NDVI")

# ============================================================
# Tab 4: AI 影像理解 (看图说话)
# ============================================================
with tab_vision:
    st.subheader("👁️ AI 影像理解")
    st.caption("AI 基于光谱指纹'看图'——判断景观类型/生态状况/空间特征")

    vis_file = st.file_uploader("上传多波段 GeoTIFF (6波段)", type=["tif", "tiff"], key="vis_upload")
    if vis_file:
        vis_path = save_upload_tmp(vis_file)
        try:
            vis_bands = load_bands(vis_path)
            st.success(f"✅ 已加载 {vis_bands.shape[0]} 波段")
        except Exception as e:
            st.error(f"❌ 读取失败: {e}")
            vis_bands = None
        finally:
            try:
                os.remove(vis_path)
            except OSError:
                pass

        if vis_bands is not None:
            if st.button("👁️ AI 看图理解", type="primary"):
                with st.spinner("AI 提取光谱指纹并解读..."):
                    from utils.ai_vision import vision_describe
                    result = vision_describe(vis_bands)

                fp = result["fingerprint"]
                st.divider()
                st.subheader("📊 光谱指纹")

                # 地物构成
                comp_cols = st.columns(4)
                for i, (name, ratio) in enumerate(fp["composition"].items()):
                    with comp_cols[i]:
                        st.metric(name, f"{ratio*100:.1f}%")

                # 指数
                idx_cols = st.columns(4)
                for i, (name, val) in enumerate(fp["indices"].items()):
                    with idx_cols[i]:
                        st.metric(name, f"{val:.4f}")

                # AI 描述
                st.subheader("🤖 AI 影像描述")
                desc = result["description"]
                ai_esc = __import__("html").escape(desc or "")
                st.markdown(
                    f"<div style='background:#f0f8f4;border-left:4px solid #27ae60;"
                    f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                    f"👁️ **AI 解读**：{ai_esc}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("👆 上传影像，AI 将'看图'描述")

# ============================================================
# Tab 5: AI 质量诊断
# ============================================================
with tab_quality:
    st.subheader("🏥 AI 影像质量诊断")
    st.caption("AI 自动评估云覆盖/噪声/异常像元，判定数据是否可用")

    q_file = st.file_uploader("上传待诊断 GeoTIFF", type=["tif", "tiff"], key="q_upload")
    if q_file:
        q_path = save_upload_tmp(q_file)
        try:
            q_bands = load_bands(q_path)
            st.success(f"✅ 已加载 {q_bands.shape[0]} 波段")
        except Exception as e:
            st.error(f"❌ 读取失败: {e}")
            q_bands = None
        finally:
            try:
                os.remove(q_path)
            except OSError:
                pass

        if q_bands is not None:
            if st.button("🏥 开始质量诊断", type="primary"):
                with st.spinner("AI 质量评估中..."):
                    from utils.ai_vision import quality_diagnose
                    result = quality_diagnose(q_bands)

                st.divider()
                st.subheader("📊 诊断结果")

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("综合评级", result["grade"])
                with c2:
                    st.metric("质量分", f"{result['score']}/100")
                with c3:
                    st.metric("云覆盖", f"{result['metrics']['cloud_ratio']*100:.1f}%")
                with c4:
                    st.metric("噪声水平", f"{result['metrics']['noise_level']:.4f}")
                with c5:
                    st.metric("异常像元", f"{result['metrics']['anomaly_ratio']*100:.2f}%")

                # AI 解读
                st.subheader("🤖 AI 质量评估")
                diag = result["ai_diagnosis"]
                ai_esc = __import__("html").escape(diag or "")
                st.markdown(
                    f"<div style='background:#fdf0f0;border-left:4px solid #e74c3c;"
                    f"padding:14px 18px;border-radius:6px;line-height:1.9;font-size:14px;'>"
                    f"🏥 **AI 诊断**：{ai_esc}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("👆 上传影像，AI 将评估数据质量")
