"""
矢量导出页面 — 栅格分类结果 → 矢量 (GeoJSON / Shapefile / KML)
================================================================
将平台分析产出的分类/分级/掩膜结果矢量化，导出为
ArcGIS / QGIS / Google Earth 可直接打开的标准矢量格式。

结果来源:
  - ESA 土地覆盖 (干旱区6类)   — 纯在线, 最快
  - 水体掩膜 (MNDWI)            — Sentinel-2 影像
  - 沙漠化分级 (5级)            — Sentinel-2 影像
  - 盐渍化分级 (5级)            — Sentinel-2 影像
  - 上传本地 GeoTIFF            — 直接矢量化
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.aoi import render_aoi_selector
from utils.vector import (
    VECTOR_FORMATS, raster_to_gdf, raster_to_vector,
    compute_class_areas, summarize_vector,
)
from utils.landcover import ARID6_CLASSES, get_landcover_for_study_area
from utils.desertification import DESERTIFICATION_LEVELS
from utils.salinity import SALINITY_LEVELS
from utils.indices import calc_mndwi
from utils.export import make_download_label

st.set_page_config(page_title="矢量导出", page_icon="🗺️", layout="wide")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🗺️ 矢量导出设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="塔里木盆地",
        key_prefix="vec",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("📦 结果来源")
    result_source = st.radio(
        "选择要矢量化的结果",
        ["ESA 土地覆盖", "水体掩膜 (MNDWI)", "沙漠化分级", "盐渍化分级", "上传 GeoTIFF"],
        help="ESA 土地覆盖: 直接在线获取 (最快)\n"
             "水体/沙漠化/盐渍化: 需下载卫星影像后计算\n"
             "上传 GeoTIFF: 矢量化自己的分类栅格",
    )

    # 影像类来源需要卫星 + 日期
    need_image = result_source in ["水体掩膜 (MNDWI)", "沙漠化分级", "盐渍化分级"]

    if need_image:
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

    if result_source == "上传 GeoTIFF":
        tif_file = st.file_uploader(
            "上传分类 GeoTIFF (单波段整数分类)",
            type=["tif", "tiff"],
            key="vec_tif",
        )

    st.divider()

    st.subheader("✂️ 矢量化参数")
    min_area = st.slider(
        "最小面积 (像元)", 0, 200, 10,
        help="面积小于该阈值(像元数)的碎小多边形将被过滤，用于消除椒盐噪声",
    )
    simplify = st.slider(
        "简化容差 (像元)", 0.0, 10.0, 0.5, 0.1,
        help="Douglas-Peucker 简化容差，0 表示不简化，值越大边越光滑",
    )
    connectivity = st.radio(
        "连通性", [4, 8], horizontal=True,
        help="4: 四邻域 (默认) | 8: 八邻域 (对角相接合并)",
    )

    st.divider()
    run_clicked = st.button(
        "🗺️ 开始矢量化", type="primary", width="stretch"
    )

# ============================================================
# 主页面
# ============================================================
st.title("🗺️ 矢量导出")
st.markdown(
    f"**研究区: {area_name}** | 结果来源: **{result_source}**\n\n"
    "将分类/分级/掩膜栅格结果矢量化，导出 **GeoJSON / Shapefile / KML**，"
    "可直接在 ArcGIS、QGIS、Google Earth 中打开与编辑。"
)

# ============================================================
# 页面内辅助: 可视化
# ============================================================

def _plot_classification(category, class_names, class_colors, title):
    """绘制分类栅格图 (离散配色 + 图例)"""
    n = len(class_names)
    cmap = ListedColormap(class_colors[:n])
    norm = BoundaryNorm(np.arange(-0.5, n, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(category, cmap=cmap, norm=norm)
    # 图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=class_colors[i], label=class_names[i])
        for i in range(n)
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.0),
              fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    return fig


def _plot_vector_overlay(gdf, class_colors, title):
    """绘制矢量多边形图 (按类别着色)"""
    import geopandas as gpd

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(gdf) == 0:
        ax.text(0.5, 0.5, "无矢量要素", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        return fig

    gdf.plot(
        ax=ax,
        column="class_id",
        categorical=True,
        cmap="tab10",
        legend=True,
        legend_kwds={"title": "类别", "fontsize": 8},
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ============================================================
# 主流程
# ============================================================
if not run_clicked:
    st.info("👈 在侧边栏选择结果来源与矢量化参数后，点击 **开始矢量化**。")
    st.stop()


# ---- Step 1: 生成分类栅格 (class_array + transform + crs + class_values) ----
category = None
transform = None
crs = None
class_values = None
nodata = 0
src_label = ""

with st.spinner(f"⏳ 准备 {result_source} 数据..."):
    with StreamlitErrorBoundary("数据准备", st=st, show_traceback=False):
        if result_source == "ESA 土地覆盖":
            lc = get_landcover_for_study_area(bbox, source="esa", map_to_arid6=True)
            if not lc["success"]:
                st.error(f"❌ 获取 ESA 土地覆盖失败: {lc.get('error')}")
                st.stop()
            category = lc["class_array"].astype(np.int16)
            transform = lc["meta"]["transform"]
            crs = lc["meta"].get("crs", "EPSG:4326")
            class_values = {k: v["name"] for k, v in ARID6_CLASSES.items() if k != 0}
            nodata = 0
            src_label = lc["source_label"]

        elif result_source == "上传 GeoTIFF":
            if tif_file is None:
                st.error("⚠️ 请先上传 GeoTIFF 文件")
                st.stop()
            import rasterio

            tmp_tif = os.path.join(tempfile.gettempdir(), f"vec_upload_{np.random.randint(1e6)}.tif")
            with open(tmp_tif, "wb") as f:
                f.write(tif_file.getvalue())
            with rasterio.open(tmp_tif) as src:
                category = src.read(1).astype(np.int16)
                transform = src.transform
                crs = str(src.crs)
                nodata = src.nodata if src.nodata is not None else 0
            os.remove(tmp_tif)
            # 自动构建类别映射 (按唯一值)
            unique_vals = np.unique(category)
            class_values = {int(v): f"类别{int(v)}" for v in unique_vals}
            src_label = "本地 GeoTIFF"

        else:
            # 影像类来源: 搜索 → 下载 → 计算
            from utils.pc_data import search_images, download_multiband

            results = search_images(
                bbox=bbox,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                collection=satellite,
                cloud_cover_max=cloud_cover,
                max_items=max_items,
            )
            if not results:
                st.warning("⚠️ 未找到符合条件的影像，降低云量阈值试试？")
                st.stop()
            st.success(f"✅ 找到 **{len(results)}** 景影像，使用第一景 **{results[0]['datetime']}**")

            item = results[0]
            tmp_path = os.path.join(tempfile.gettempdir(), f"vec_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)

            import rasterio

            with rasterio.open(tif_path) as src:
                bands_data = src.read().astype(np.float32)
                transform = src.transform
                crs = str(src.crs)
            if os.path.exists(tif_path):
                os.remove(tif_path)

            if result_source == "水体掩膜 (MNDWI)":
                green = bands_data[1]
                swir1 = bands_data[4]
                mndwi = calc_mndwi(green, swir1)
                category = (mndwi > 0).astype(np.int16)
                class_values = {1: "水体"}
                nodata = 0
                src_label = f"水体掩膜 MNDWI ({satellite})"

            elif result_source == "沙漠化分级":
                from utils.desertification import assess_desertification

                res = assess_desertification(bands_data, satellite=satellite)
                category = res.category.astype(np.int16)
                class_values = {lv["code"]: lv["name"] for lv in DESERTIFICATION_LEVELS}
                nodata = None  # 0 = 非沙漠化 (有效类别)
                src_label = f"沙漠化分级 ({satellite})"

            else:  # 盐渍化分级
                from utils.salinity import assess_salinity

                res = assess_salinity(bands_data, satellite=satellite)
                category = res.category.astype(np.int16)
                class_values = {lv["code"]: lv["name"] for lv in SALINITY_LEVELS}
                nodata = None  # 0 = 非盐渍化 (有效类别)
                src_label = f"盐渍化分级 ({satellite})"

st.success(f"✅ 数据准备完成: **{src_label}** (shape={category.shape})")

# ---- Step 2: 矢量化 ----
with st.spinner("✂️ 矢量化中..."):
    with StreamlitErrorBoundary("矢量化", st=st, show_traceback=True):
        gdf = raster_to_gdf(
            category, transform, crs,
            class_values=class_values,
            nodata=nodata,
            min_area_pixels=min_area,
            simplify_tolerance=simplify,
            connectivity=connectivity,
        )

if len(gdf) == 0:
    st.warning("⚠️ 矢量化结果为空。尝试减小最小面积阈值，或检查输入栅格。")
    st.stop()

st.success(f"✅ 矢量化完成: **{len(gdf)}** 个多边形")

# ---- Step 3: 概要 + 面积统计 ----
summary = summarize_vector(gdf)
areas = compute_class_areas(gdf, class_values=class_values)

st.subheader("📊 面积统计")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("多边形总数", summary["polygon_count"])
with col2:
    st.metric("类别数", summary["class_count"])
with col3:
    st.metric("最大类别", areas[0]["class_name"] if areas else "—")

area_df = pd.DataFrame(areas)
area_df.columns = ["类别ID", "类别名称", "多边形数", "面积(公顷)", "面积(km²)"]
st.dataframe(area_df, width="stretch")

st.divider()

# ---- Step 4: 预览 ----
st.subheader("🖼️ 预览")
tab_raster, tab_vector = st.tabs(["分类栅格", "矢量多边形"])

# 分类栅格颜色 (按 class_id)
color_cycle = plt.get_cmap("tab10").colors
class_ids_sorted = sorted(class_values.keys()) if class_values else sorted(gdf["class_id"].unique())
class_colors = {}
for i, cid in enumerate(class_ids_sorted):
    class_colors[cid] = color_cycle[i % len(color_cycle)]
names = [class_values.get(c, str(c)) for c in class_ids_sorted]
colors = [class_colors[c] for c in class_ids_sorted]

with tab_raster:
    fig = _plot_classification(
        category, names, colors,
        title=f"{src_label} — 分类栅格",
    )
    st.pyplot(fig)
    plt.close(fig)

with tab_vector:
    fig = _plot_vector_overlay(gdf, class_colors, title=f"{src_label} — 矢量面 ({len(gdf)} 个)")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ---- Step 5: 导出下载 ----
st.subheader("💾 导出下载")
st.caption("三种格式任选，ArcGIS/QGIS 推荐 Shapefile，Web 展示推荐 GeoJSON，Google Earth 用 KML。")

with StreamlitErrorBoundary("导出", st=st, show_traceback=False):
    col_geojson, col_shp, col_kml = st.columns(3)

    with col_geojson:
        try:
            _, gj_data = raster_to_vector(
                category, transform, crs, fmt="geojson",
                class_values=class_values, nodata=nodata,
                min_area_pixels=min_area, simplify_tolerance=simplify,
                connectivity=connectivity,
            )
            st.download_button(
                "⬇️ 下载 GeoJSON",
                data=gj_data,
                file_name=make_download_label("vector", "geojson"),
                mime=VECTOR_FORMATS["geojson"]["mime"],
                width="stretch",
            )
        except Exception as e:
            st.error(f"GeoJSON 导出失败: {e}")

    with col_shp:
        try:
            _, shp_data = raster_to_vector(
                category, transform, crs, fmt="shapefile",
                class_values=class_values, nodata=nodata,
                min_area_pixels=min_area, simplify_tolerance=simplify,
                connectivity=connectivity,
                layer_name="vector_export",
            )
            st.download_button(
                "⬇️ 下载 Shapefile (zip)",
                data=shp_data,
                file_name=make_download_label("vector", "zip"),
                mime=VECTOR_FORMATS["shapefile"]["mime"],
                width="stretch",
            )
        except Exception as e:
            st.error(f"Shapefile 导出失败: {e}")

    with col_kml:
        try:
            _, kml_data = raster_to_vector(
                category, transform, crs, fmt="kml",
                class_values=class_values, nodata=nodata,
                min_area_pixels=min_area, simplify_tolerance=simplify,
                connectivity=connectivity,
            )
            st.download_button(
                "⬇️ 下载 KML",
                data=kml_data,
                file_name=make_download_label("vector", "kml"),
                mime=VECTOR_FORMATS["kml"]["mime"],
                width="stretch",
            )
        except Exception as e:
            st.error(f"KML 导出失败: {e}")

st.divider()
st.info(
    "💡 **提示**: Shapefile 为 zip 包 (.shp/.shx/.dbf/.prj/.cpg)，"
    "解压后直接用 ArcGIS/QGIS 打开；中文属性已通过 .cpg (UTF-8) 保证正确显示。"
    "若多边形过多导致导出缓慢，可增大「最小面积」阈值过滤碎斑。"
)
