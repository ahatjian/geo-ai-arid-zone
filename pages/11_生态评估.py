"""
生态安全评估页面 — PSR 压力-状态-响应模型
==========================================
"""
import streamlit as st
import os, sys, tempfile, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.aoi import render_aoi_selector
from utils.pc_data import search_images, get_rgb_preview_cached, download_multiband

st.set_page_config(page_title="生态评估", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.eco-card { background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:14px 18px; text-align:center; }
.eco-card .label { font-size:11px; color:#6cb4ee; margin-bottom:4px; }
.eco-card .value { font-size:20px; font-weight:700; }
.eco-safe { border-color:#2ecc71 !important; }
.eco-warn { border-color:#f1c40f !important; }
.eco-danger { border-color:#e74c3c !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🌍 生态评估设置")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="塔里木盆地",
        key_prefix="eco",
    )
    if bbox is None:
        st.stop()

    st.divider()
    satellite = st.selectbox("数据源", list(COLLECTIONS.keys()),
                            format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始", date.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("结束", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 15)
    max_items = st.slider("最大影像数", 1, 10, 3)

    st.divider()
    st.subheader("PSR 权重")
    wp = st.slider("压力 (P)", 0.0, 1.0, 0.35, 0.05)
    ws = st.slider("状态 (S)", 0.0, 1.0, 0.40, 0.05)
    wr = st.slider("响应 (R)", 0.0, 1.0, 0.25, 0.05)

    with st.expander("⚙️ 高级"):
        pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)

    search_clicked = st.button("🔍 评估生态安全", type="primary")

st.title("🌍 生态安全评估 (PSR 模型)")
st.markdown(f"**{area_name}** | {satellite} | {start_date} → {end_date}")

if search_clicked:
    # bbox 已由研究区/AOI 选择组件提供

    with st.spinner("🔍 搜索..."):
        try:
            results = search_images(bbox=bbox, start_date=start_date.strftime("%Y-%m-%d"),
                                    end_date=end_date.strftime("%Y-%m-%d"),
                                    collection=satellite, cloud_cover_max=cloud_cover, max_items=max_items)
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

    if not results:
        st.warning("⚠️ 无影像")
        st.stop()

    st.success(f"✅ {len(results)} 景")

    # 下载分析
    progress = st.progress(0)
    status = st.empty()
    all_ndvi, all_bands, image_labels = [], [], []

    for idx, r in enumerate(results):
        progress.progress((idx + 1) / len(results))
        status.text(f"⬇️ [{idx+1}/{len(results)}] {r['datetime']}")
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"eco_{r['id'][:12]}.tif")
            tif = download_multiband(r["item"], tmp, collection=satellite)
            if tif and os.path.exists(tif):
                import rasterio
                with rasterio.open(tif) as src:
                    b = src.read().astype(np.float64)
                if np.nanmedian(b[1]) > 10:
                    b /= 10000.0
                ndvi = np.clip((b[3]-b[2])/(b[3]+b[2]+1e-6), -1, 1)
                all_ndvi.append(ndvi.astype(np.float32))
                all_bands.append(b.astype(np.float32))
                image_labels.append(r["datetime"])
                try:
                    os.remove(tif)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"⚠️ {r['datetime']}: {e}")

    progress.progress(1.0)
    status.text("✅ 完成")

    if len(all_ndvi) < 1:
        st.error("❌ 无有效影像")
        st.stop()

    # 使用最新影像进行评估
    ndvi_main = all_ndvi[0]
    bands_main = all_bands[0]
    main_date = image_labels[0]

    # NDVI 趋势 (如有 3+ 景)
    ndvi_trend = np.zeros_like(ndvi_main)
    if len(all_ndvi) >= 3:
        ndvi_stack = np.stack(all_ndvi, axis=-1)
        from utils.trend import theil_sen_slope
        H, W, T = ndvi_stack.shape
        for i in range(H):
            for j in range(W):
                ts = ndvi_stack[i, j, :]
                if np.isfinite(ts).all():
                    ndvi_trend[i, j] = theil_sen_slope(ts)

    # NDMI (土壤水分)
    nir = bands_main[3]
    swir1 = bands_main[4]
    ndmi = np.clip((nir - swir1) / (nir + swir1 + 1e-6), -1, 1)

    # PSR 评估
    from utils.ecology import assess_eco_security, ECO_SECURITY_LEVELS

    result = assess_eco_security(
        ndvi=ndvi_main, ndmi=ndmi, ndvi_trend=ndvi_trend,
        pixel_size_m=pixel_size,
        weights=(wp, ws, wr),
    )

    # 汇总
    s = result.summary
    status_css = "eco-safe" if s["overall_status"] == "安全" else ("eco-warn" if s["overall_status"] == "预警" else "eco-danger")

    st.subheader("📊 生态安全评估结果")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="eco-card {status_css}"><div class="label">ESI 均值</div><div class="value">{s["esi_mean"]:.3f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="eco-card"><div class="label">安全区占比</div><div class="value">{s["safe_ratio"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="eco-card"><div class="label">不安全区占比</div><div class="value">{s["unsafe_ratio"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="eco-card {status_css}"><div class="label">综合状态</div><div class="value">{s["overall_status"]}</div></div>', unsafe_allow_html=True)

    # AI 智能解读 (统一组件)
    from utils.ai_insight import render_ai_insight_block
    render_ai_insight_block(
        analysis_type="生态安全评估 (PSR 模型)",
        metrics={
            "ESI生态安全指数": s["esi_mean"],
            "安全区占比": s["safe_ratio"],
            "不安全区占比": s["unsafe_ratio"],
            "综合状态": s["overall_status"],
        },
        study_area=area_name,
        key_suffix="eco_ai",
        show_button=True,
    )

    # Tab
    tabs = st.tabs(["ESI 综合", "PSI 压力", "SSI 状态", "RSI 响应"])

    tab_data = [("ESI 生态安全指数", result.esi, "RdYlGn", 0, 1),
                ("PSI 压力指数", result.psi, "YlOrRd", 0, 1),
                ("SSI 状态指数", result.ssi, "YlGn", 0, 1),
                ("RSI 响应指数", result.rsi, "PuBuGn", 0, 1)]

    for i, (tab, (title, data, cmap, vmin, vmax)) in enumerate(zip(tabs, tab_data)):
        with tab:
            st.caption(title)
            col_m, col_s = st.columns([3, 1])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title} — {area_name} ({main_date})", fontsize=13)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                buf = BytesIO()
                plt.tight_layout(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(); buf.seek(0)
                st.image(Image.open(buf))
            with col_s:
                valid = data[np.isfinite(data)]
                st.metric("均值", f"{np.nanmean(valid):.4f}")
                st.metric("标准差", f"{np.nanstd(valid):.4f}")
                st.metric("范围", f"{np.nanmin(valid):.3f} ~ {np.nanmax(valid):.3f}")

    # 分级统计
    st.divider()
    st.subheader("📋 生态安全分级")

    names = [s["name"] for s in result.stats]
    ratios = [s["ratio"] * 100 for s in result.stats]
    colors = [s["color"] for s in result.stats]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(names, ratios, color=colors, edgecolor="#333")
    ax2.set_ylabel("面积占比 (%)", fontsize=12)
    ax2.set_title(f"{area_name} 生态安全等级分布", fontsize=14)
    for bar, val in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)
    buf2 = BytesIO()
    plt.tight_layout(); plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
    plt.close(); buf2.seek(0)
    st.image(Image.open(buf2))

    with st.expander("📋 详细统计"):
        df = pd.DataFrame(result.stats)
        df = df.rename(columns={"name":"等级","pixel_count":"像元","ratio":"占比","area_km2":"面积km²","status":"状态"})
        st.dataframe(df[["等级","像元","占比","面积km²","状态"]], height=300, hide_index=True)

    # 导出
    st.divider()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📊 下载生态评估 CSV", csv, f"eco_{area_name}_{main_date}.csv", "text/csv")
    with col_d2:
        st.info(f"💡 PSR 权重: P={wp:.0%} S={ws:.0%} R={wr:.0%} | 基于 OECD PSR 框架")

else:
    st.info("👈 选择研究区，点击评估开始分析")

    with st.expander("📖 PSR 生态安全模型说明"):
        st.markdown("""
        ### PSR 框架 (OECD 1993)

        | 维度 | 指标 | 遥感代理 |
        |------|------|---------|
        | **P 压力** | 人类活动强度 | NDVI 低值 + 干旱指数 |
        | **S 状态** | 生态系统现状 | NDVI + 土壤水分 + 水体面积 |
        | **R 响应** | 恢复能力 | NDVI 趋势 + 稳定性 |

        ### ESI = 0.35×(1-PSI) + 0.40×SSI + 0.25×RSI

        - ESI > 0.70: 🟢 安全
        - ESI 0.55-0.70: 🟢 较安全
        - ESI 0.40-0.55: 🟡 预警
        - ESI 0.25-0.40: 🟠 较不安全
        - ESI < 0.25: 🔴 不安全
        """)
