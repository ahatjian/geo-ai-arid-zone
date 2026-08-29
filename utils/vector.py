"""
矢量导出模块 — 栅格分类/分级结果 → 矢量面 (GeoJSON / Shapefile / KML)
=====================================================================
将平台分析产出的分类图、分级图、掩膜等栅格结果矢量化，
导出为 ArcGIS / QGIS / Google Earth 可直接打开的标准矢量格式。

核心能力:
  - raster_to_gdf:        栅格 → GeoDataFrame (矢量化 + 面积过滤 + 简化)
  - gdf_to_geojson:       → GeoJSON 文件 (Web / QGIS 通用)
  - gdf_to_shapefile:     → Shapefile (zip 打包, ArcGIS/QGIS 标准)
  - gdf_to_kml:           → KML 文件 (Google Earth)
  - raster_to_vector:     一站式栅格 → 矢量文件
  - compute_class_areas:  各类别面积统计 (ha / km²)

设计约定:
  - 所有导出函数返回 (文件路径, 二进制内容), 供 Streamlit download_button 使用
  - 面积过滤阈值以"像元数"为单位, 内部换算为 CRS 面积单位
  - 简化容差以"像元"为单位, 内部换算为 CRS 长度单位
  - Shapefile 字段名 ≤10 字符 (class_id/class_name/area_m2 均合规)
  - 中文属性通过 .cpg (UTF-8) 保证 ArcGIS 正确读取

依赖: rasterio, shapely, geopandas, fiona, pyproj, pandas
"""

import os
import tempfile
import zipfile
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ============================================
# 格式定义
# ============================================

VECTOR_FORMATS = {
    "geojson": {
        "name": "GeoJSON",
        "ext": "geojson",
        "mime": "application/geo+json",
        "desc": "Web/GIS 通用格式，QGIS/ArcGIS 直接打开",
    },
    "shapefile": {
        "name": "Shapefile (zip)",
        "ext": "zip",
        "mime": "application/zip",
        "desc": "ArcGIS/QGIS 标准矢量格式 (.shp/.dbf/.prj 打包)",
    },
    "kml": {
        "name": "KML",
        "ext": "kml",
        "mime": "application/vnd.google-earth.kml+xml",
        "desc": "Google Earth 展示格式",
    },
}

# 面积统计默认投影 (Equal Earth, 等面积投影, 避免中纬度面积失真)
DEFAULT_AREA_CRS = "EPSG:6933"


# ============================================
# 内部辅助
# ============================================

def _temp_path(suffix: str) -> str:
    """生成临时文件路径 (不落地内容, 仅占位)"""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = tmp.name
    tmp.close()
    return path


def _read_binary(path: str) -> bytes:
    """读取文件二进制内容"""
    with open(path, "rb") as f:
        return f.read()


def _sanitize_shapefile_fields(gdf) -> "gpd.GeoDataFrame":
    """
    规范化字段以兼容 Shapefile 限制。

    ⚠️ 警告: Shapefile 的 DBF 对字段名/类型有严格限制:
       - 字段名 ≤10 字节 (ASCII)
       - 不支持 int64 (需转 int32)
       - 不支持混合几何类型 (需 explode 成单一 Polygon)
    """
    import geopandas as gpd

    out = gdf.copy()
    # 拆解 MultiPolygon → Polygon (Shapefile 不接受混合几何类型)
    out = out.explode(index_parts=False).reset_index(drop=True)

    # int64 → int32 (DBF 数值字段上限)
    if "class_id" in out.columns:
        out["class_id"] = out["class_id"].astype(np.int32)
    if "area_m2" in out.columns:
        out["area_m2"] = out["area_m2"].astype(np.float64)

    # 字段名规范化 (截断到 10 字符, 去非法字符)
    rename = {}
    for col in out.columns:
        if col == "geometry":
            continue
        new_col = col[:10]
        if new_col != col:
            rename[col] = new_col
    if rename:
        out = out.rename(columns=rename)

    return out


# ============================================
# 栅格 → GeoDataFrame (矢量化)
# ============================================

def raster_to_gdf(
    mask: np.ndarray,
    transform,
    crs,
    class_values: Optional[Dict[int, str]] = None,
    nodata: Optional[int] = 0,
    min_area_pixels: float = 0.0,
    simplify_tolerance: float = 0.0,
    connectivity: int = 4,
):
    """
    将整数分类/分级/掩膜栅格矢量化面要素。

    参数:
        mask: 2D 整数标签数组 (分类/分级/掩膜)
        transform: 仿射变换 (rasterio transform)
        crs: 坐标系 (EPSG 字符串或 CRS 对象)
        class_values: {class_id: "类别名"} 类别映射 (可选)
        nodata: 需要跳过的像元值; 传 None 表示不跳过任何值
                (注意: 沙漠化/盐渍化分级中 0 是有效类别, 应传 None)
        min_area_pixels: 最小面积阈值 (像元数), 过滤碎小多边形
        simplify_tolerance: Douglas-Peucker 简化容差 (像元单位), 0 表示不简化
        connectivity: 4 邻域或 8 邻域连通

    返回:
        geopandas.GeoDataFrame, 含列:
          - class_id   (int)     类别值
          - class_name (str)     类别名
          - area_m2    (float)   几何面积 (CRS 单位²; 若 CRS 为地理坐标则为度², 仅作参考)
          - geometry   (Polygon/MultiPolygon)
    """
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape

    # 统一为 int32 (浮点分级先取整)
    mask = np.asarray(mask)
    if mask.dtype.kind == "f":
        mask = mask.astype(np.int32)
    else:
        mask = mask.astype(np.int32)

    if mask.ndim != 2:
        raise ValueError(f"mask 必须是 2D 数组, 实际 shape={mask.shape}")

    # 有效像元掩膜
    if nodata is None:
        valid = np.ones(mask.shape, dtype=bool)
    else:
        valid = mask != nodata

    # 像元面积与线性尺寸 (CRS 单位)
    pixel_area = abs(transform.a * transform.e)
    pixel_size = (abs(transform.a) + abs(transform.e)) / 2.0
    min_area = min_area_pixels * pixel_area
    tol = simplify_tolerance * pixel_size

    records = []
    geometries = []

    for geom, value in features.shapes(
        mask, mask=valid, connectivity=connectivity, transform=transform
    ):
        v = int(value)
        poly = shape(geom)

        if poly.is_empty:
            continue
        # 修复无效几何 (自交等)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                continue

        # 面积过滤
        if poly.area < min_area:
            continue

        # 简化
        if tol > 0:
            poly = poly.simplify(tol, preserve_topology=True)
            if poly.is_empty:
                continue

        geometries.append(poly)
        records.append({
            "class_id": v,
            "class_name": class_values.get(v) if class_values else str(v),
        })

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)

    # 几何面积: 仅投影坐标下有意义 (米²)。
    # 地理坐标 (EPSG:4326 等) 下 .area 返回度², 既无意义又会触发 geopandas 警告,
    # 故不计算, 真实面积交由 compute_class_areas (内部自动投影到等面积坐标系)。
    if len(gdf) and gdf.crs is not None and not getattr(gdf.crs, "is_geographic", False):
        gdf["area_m2"] = gdf.geometry.area.astype(float)

    return gdf


# ============================================
# GeoDataFrame → 文件 (三格式)
# ============================================

def gdf_to_geojson(gdf, output_path: Optional[str] = None) -> Tuple[str, bytes]:
    """导出 GeoDataFrame 为 GeoJSON 文件"""
    if output_path is None:
        output_path = _temp_path(".geojson")
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path, _read_binary(output_path)


def gdf_to_shapefile(
    gdf,
    output_path: Optional[str] = None,
    layer_name: str = "vector_export",
) -> Tuple[str, bytes]:
    """
    导出 GeoDataFrame 为 Shapefile (zip 打包)。

    ⚠️ 警告: Shapefile 字段名 ≤10 字节、int64 需转 int32、
    混合几何需 explode, 已由 _sanitize_shapefile_fields 统一处理。

    返回:
        (zip 文件路径, zip 二进制内容) — zip 内含 .shp/.shx/.dbf/.prj/.cpg
    """
    import geopandas as gpd

    out = _sanitize_shapefile_fields(gdf)

    # 写入临时目录
    tmpdir = tempfile.mkdtemp(prefix="vector_shp_")
    shp_path = os.path.join(tmpdir, f"{layer_name}.shp")

    try:
        out.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

        # 写 .cpg 文件, 声明 UTF-8 编码, 保证 ArcGIS 正确读取中文属性
        cpg_path = os.path.join(tmpdir, f"{layer_name}.cpg")
        with open(cpg_path, "w", encoding="ascii") as f:
            f.write("UTF-8")

        # zip 打包全部伴生文件
        if output_path is None:
            output_path = _temp_path(".zip")
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(tmpdir)):
                zf.write(os.path.join(tmpdir, fn), arcname=fn)
    finally:
        # 清理临时目录
        for fn in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    return output_path, _read_binary(output_path)


def gdf_to_kml(gdf, output_path: Optional[str] = None) -> Tuple[str, bytes]:
    """
    导出 GeoDataFrame 为 KML 文件 (Google Earth)。

    ⚠️ 警告: KML 要求 WGS84 (EPSG:4326) 坐标, 非 4326 的会自动重投影。
    """
    if output_path is None:
        output_path = _temp_path(".kml")

    out = gdf.copy()
    # KML 必须 WGS84
    if out.crs is not None:
        epsg = out.crs.to_epsg() if hasattr(out.crs, "to_epsg") else None
        if epsg != 4326:
            out = out.to_crs("EPSG:4326")

    # 剔除面积列: 地理坐标下无意义, 且会触发 pyogrio lossy-conversion 警告
    if "area_m2" in out.columns:
        out = out.drop(columns=["area_m2"])

    out.to_file(output_path, driver="KML")
    return output_path, _read_binary(output_path)


# ============================================
# 一站式: 栅格 → 矢量文件
# ============================================

def raster_to_vector(
    mask: np.ndarray,
    transform,
    crs,
    fmt: str = "geojson",
    class_values: Optional[Dict[int, str]] = None,
    nodata: Optional[int] = 0,
    min_area_pixels: float = 0.0,
    simplify_tolerance: float = 0.0,
    connectivity: int = 4,
    output_path: Optional[str] = None,
    layer_name: str = "vector_export",
) -> Tuple[str, bytes]:
    """
    一站式: 栅格分类结果 → 矢量文件。

    参数:
        mask/transform/crs: 同 raster_to_gdf
        fmt: "geojson" | "shapefile" | "kml"
        其余参数: 同 raster_to_gdf

    返回:
        (文件路径, 二进制内容)
    """
    gdf = raster_to_gdf(
        mask, transform, crs,
        class_values=class_values,
        nodata=nodata,
        min_area_pixels=min_area_pixels,
        simplify_tolerance=simplify_tolerance,
        connectivity=connectivity,
    )

    if fmt == "geojson":
        return gdf_to_geojson(gdf, output_path)
    elif fmt == "shapefile":
        return gdf_to_shapefile(gdf, output_path, layer_name=layer_name)
    elif fmt == "kml":
        return gdf_to_kml(gdf, output_path)
    else:
        raise ValueError(
            f"不支持的格式 '{fmt}', 可选: {list(VECTOR_FORMATS.keys())}"
        )


def raster_to_vector_from_file(
    mask: np.ndarray,
    reference_path: str,
    fmt: str = "geojson",
    class_values: Optional[Dict[int, str]] = None,
    nodata: Optional[int] = 0,
    min_area_pixels: float = 0.0,
    simplify_tolerance: float = 0.0,
    connectivity: int = 4,
    output_path: Optional[str] = None,
    layer_name: str = "vector_export",
) -> Tuple[str, bytes]:
    """
    从参考 GeoTIFF 读取 CRS + transform, 将 mask 矢量化导出。

    参数:
        mask: 2D 整数分类/掩膜数组
        reference_path: 参考 GeoTIFF (提供 CRS + transform)
        其余: 同 raster_to_vector
    """
    import rasterio

    with rasterio.open(reference_path) as src:
        transform = src.transform
        crs = src.crs

    return raster_to_vector(
        mask, transform, crs,
        fmt=fmt,
        class_values=class_values,
        nodata=nodata,
        min_area_pixels=min_area_pixels,
        simplify_tolerance=simplify_tolerance,
        connectivity=connectivity,
        output_path=output_path,
        layer_name=layer_name,
    )


# ============================================
# 面积统计
# ============================================

def compute_class_areas(
    gdf,
    class_values: Optional[Dict[int, str]] = None,
    area_crs: str = DEFAULT_AREA_CRS,
) -> List[dict]:
    """
    统计各类别的矢量面积 (公顷 + km²) 与多边形数量。

    参数:
        gdf: raster_to_gdf 返回的 GeoDataFrame
        class_values: {class_id: "类别名"} (可选, 覆盖 gdf 内的 class_name)
        area_crs: 面积计算投影 (默认 Equal Earth 等面积投影);
                  地理坐标 CRS 会自动投影到该坐标系再算面积

    返回:
        list[dict]: 按面积降序, 每项含
          class_id / class_name / polygon_count / area_ha / area_km2
    """
    out = gdf.copy()

    # 地理坐标 → 投影到等面积坐标系, 避免面积失真
    if out.crs is not None and getattr(out.crs, "is_geographic", False):
        try:
            out = out.to_crs(area_crs)
        except Exception:
            # fallback: Web Mercator (中纬度面积会偏大, 仅兜底)
            out = out.to_crs("EPSG:3857")

    areas = []
    for class_id, sub in out.groupby("class_id"):
        area_m2 = float(sub.geometry.area.sum())
        name = (
            class_values.get(int(class_id))
            if class_values
            else (sub["class_name"].iloc[0] if "class_name" in sub.columns else str(class_id))
        )
        areas.append({
            "class_id": int(class_id),
            "class_name": name,
            "polygon_count": int(len(sub)),
            "area_ha": round(area_m2 / 1e4, 4),
            "area_km2": round(area_m2 / 1e6, 4),
        })

    areas.sort(key=lambda x: -x["area_km2"])
    return areas


# ============================================
# 概要信息
# ============================================

def summarize_vector(gdf) -> Dict:
    """
    返回矢量结果的概要信息 (多边形总数 / 类别数 / 各类别面数)。

    返回:
        dict: {polygon_count, class_count, classes: {class_id: polygon_count}}
    """
    if len(gdf) == 0:
        return {"polygon_count": 0, "class_count": 0, "classes": {}}

    counts = gdf.groupby("class_id").size().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}

    return {
        "polygon_count": int(len(gdf)),
        "class_count": int(len(counts)),
        "classes": counts,
    }
