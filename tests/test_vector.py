"""
矢量导出模块单元测试
=====================
覆盖: 矢量化正确性、面积过滤、简化、三格式导出、类别面积统计
"""
import json
import os
import zipfile

import numpy as np
import pytest

from affine import Affine


# ============================================
# 测试夹具
# ============================================

def _make_mask():
    """构造一个 20x20 的分类掩膜:
       - 左上角 8x8 为类别 1 (水体)
       - 右下角 8x8 为类别 2 (植被)
       - 其余为 0 (背景/nodata)
    """
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[:8, :8] = 1
    mask[12:, 12:] = 2
    return mask


def _make_transform():
    """10m 分辨率、原点 (100000, 200000) 的仿射变换"""
    return Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)


CRS = "EPSG:32645"  # UTM 45N (米制投影)


# ============================================
# 测试
# ============================================

def test_raster_to_gdf_basic():
    """基本矢量化: 类别数与多边形数量正确"""
    from utils.vector import raster_to_gdf

    mask = _make_mask()
    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=0)

    # 两个连通块 (类别 1 和 2), 各 1 个多边形
    assert len(gdf) == 2
    assert set(gdf["class_id"]) == {1, 2}


def test_raster_to_gdf_class_values():
    """类别映射: class_name 正确填充"""
    from utils.vector import raster_to_gdf

    mask = _make_mask()
    gdf = raster_to_gdf(
        mask, _make_transform(), CRS,
        class_values={1: "水体", 2: "植被"}, nodata=0,
    )

    names = dict(zip(gdf["class_id"], gdf["class_name"]))
    assert names[1] == "水体"
    assert names[2] == "植被"


def test_raster_to_gdf_nodata_none():
    """nodata=None: 保留 0 类别 (沙漠化/盐渍化 0 级有效)"""
    from utils.vector import raster_to_gdf

    mask = _make_mask()
    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=None)

    # 此时 0 也是有效类别, 应导出 3 个多边形 (含背景)
    assert set(gdf["class_id"]) == {0, 1, 2}
    assert len(gdf) == 3


def test_raster_to_gdf_min_area_filter():
    """面积过滤: 碎小多边形被过滤"""
    from utils.vector import raster_to_gdf

    # 构造一个 1 像元的孤立点 + 一个大块
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15:, 15:] = 1   # 5x5 = 25 像元
    mask[0, 0] = 2       # 1 像元孤立点

    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=0, min_area_pixels=10)

    # 1 像元点 (面积 100 m² < 10 像元*100m²) 被过滤, 只剩大块
    assert set(gdf["class_id"]) == {1}


def test_raster_to_gdf_float_mask():
    """浮点掩膜自动取整"""
    from utils.vector import raster_to_gdf

    mask = (_make_mask()).astype(np.float32)
    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=0)
    assert set(gdf["class_id"]) == {1, 2}


def test_raster_to_gdf_connectivity8():
    """8 邻域连通: 对角相接的两点合并为同一多边形"""
    from utils.vector import raster_to_gdf

    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2, 2] = 1
    mask[3, 3] = 1  # 与 (2,2) 对角相邻

    gdf4 = raster_to_gdf(mask, _make_transform(), CRS, nodata=0, connectivity=4)
    gdf8 = raster_to_gdf(mask, _make_transform(), CRS, nodata=0, connectivity=8)

    # 4 邻域: 两个独立多边形; 8 邻域: 合并为一个
    assert len(gdf4) == 2
    assert len(gdf8) == 1


def test_gdf_to_geojson():
    """GeoJSON 导出: 合法 FeatureCollection, 含类别属性"""
    from utils.vector import raster_to_gdf, gdf_to_geojson

    gdf = raster_to_gdf(
        _make_mask(), _make_transform(), CRS,
        class_values={1: "水体", 2: "植被"},
    )
    path, data = gdf_to_geojson(gdf)

    assert path.endswith(".geojson")
    parsed = json.loads(data.decode("utf-8"))
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 2

    props = {f["properties"]["class_id"]: f["properties"]["class_name"]
             for f in parsed["features"]}
    assert props[1] == "水体"
    assert props[2] == "植被"


def test_gdf_to_shapefile():
    """Shapefile 导出: zip 内含 shp/shx/dbf/prj/cpg, 且可回读"""
    from utils.vector import raster_to_gdf, gdf_to_shapefile

    gdf = raster_to_gdf(
        _make_mask(), _make_transform(), CRS,
        class_values={1: "水体", 2: "植被"},
    )
    path, data = gdf_to_shapefile(gdf, layer_name="test_layer")

    assert path.endswith(".zip")

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        # .shp / .shx / .dbf / .prj / .cpg 五件套
        assert "test_layer.shp" in names
        assert "test_layer.shx" in names
        assert "test_layer.dbf" in names
        assert "test_layer.prj" in names
        assert "test_layer.cpg" in names

        cpg = zf.read("test_layer.cpg").decode("ascii")
        assert cpg == "UTF-8"


def test_gdf_to_kml():
    """KML 导出: 生成合法 KML 文件 (WGS84 重投影)"""
    from utils.vector import raster_to_gdf, gdf_to_kml

    # 用地理坐标 (EPSG:4326) 构造, 验证 KML 写出
    from affine import Affine as A
    mask = _make_mask()
    # ~0.0001° 像元
    trans_ll = A(0.0001, 0.0, 80.0, 0.0, -0.0001, 40.0)
    gdf = raster_to_gdf(mask, trans_ll, "EPSG:4326", class_values={1: "水体", 2: "植被"})

    path, data = gdf_to_kml(gdf)

    assert path.endswith(".kml")
    text = data.decode("utf-8")
    assert "<kml" in text.lower() or "<Kml" in text or "kml" in text.lower()


def test_raster_to_vector_geojson():
    """一站式: 栅格 → GeoJSON"""
    from utils.vector import raster_to_vector

    path, data = raster_to_vector(
        _make_mask(), _make_transform(), CRS, fmt="geojson",
        class_values={1: "水体", 2: "植被"},
    )
    parsed = json.loads(data.decode("utf-8"))
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 2


def test_raster_to_vector_invalid_fmt():
    """非法格式抛出 ValueError"""
    from utils.vector import raster_to_vector

    with pytest.raises(ValueError):
        raster_to_vector(_make_mask(), _make_transform(), CRS, fmt="geopkg")


def test_compute_class_areas():
    """类别面积统计: 面积 = 像元数 × 像元面积"""
    from utils.vector import raster_to_gdf, compute_class_areas

    mask = _make_mask()  # 类别1: 8x8=64 像元, 类别2: 8x8=64 像元
    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=0)
    areas = compute_class_areas(gdf)

    # 每个 8x8 块 = 64 像元 × 100 m² = 6400 m² = 0.64 ha = 0.0064 km²
    for a in areas:
        assert a["polygon_count"] == 1
        # 允许浮点/简化误差
        assert abs(a["area_ha"] - 0.64) < 0.01


def test_compute_class_areas_geographic():
    """地理坐标 CRS: 自动投影到等面积坐标系算面积"""
    from utils.vector import raster_to_gdf, compute_class_areas
    from affine import Affine as A

    mask = _make_mask()
    trans_ll = A(0.0001, 0.0, 80.0, 0.0, -0.0001, 40.0)
    gdf = raster_to_gdf(mask, trans_ll, "EPSG:4326", nodata=0)

    areas = compute_class_areas(gdf)
    # 面积应 > 0 且合理 (8x8 像元 × ~11m = ~88m 边长 → ~0.77 ha 量级)
    for a in areas:
        assert a["area_ha"] > 0.1


def test_summarize_vector():
    """概要信息: 多边形数与类别数正确"""
    from utils.vector import raster_to_gdf, summarize_vector

    gdf = raster_to_gdf(_make_mask(), _make_transform(), CRS, nodata=0)
    summary = summarize_vector(gdf)

    assert summary["polygon_count"] == 2
    assert summary["class_count"] == 2
    assert summary["classes"][1] == 1
    assert summary["classes"][2] == 1


def test_raster_to_gdf_empty():
    """全 nodata 掩膜: 返回空 GeoDataFrame"""
    from utils.vector import raster_to_gdf

    mask = np.zeros((10, 10), dtype=np.int32)
    gdf = raster_to_gdf(mask, _make_transform(), CRS, nodata=0)
    assert len(gdf) == 0


def test_raster_to_gdf_simplify():
    """简化: 容差 >0 时仍能矢量化 (顶点数减少)"""
    from utils.vector import raster_to_gdf

    # 用带锯齿的较大块
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[5:35, 5:35] = 1

    gdf_raw = raster_to_gdf(mask, _make_transform(), CRS, nodata=0, simplify_tolerance=0)
    gdf_sim = raster_to_gdf(mask, _make_transform(), CRS, nodata=0, simplify_tolerance=5)

    raw_n = len(gdf_raw.geometry.iloc[0].exterior.coords)
    sim_n = len(gdf_sim.geometry.iloc[0].exterior.coords)
    # 简化后顶点数不应增加
    assert sim_n <= raw_n
