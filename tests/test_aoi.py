"""
自定义研究区 (AOI) 组件单元测试
测试: GeoJSON 解析 + bbox 验证
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pytest


class TestParseGeoJSON:
    def test_parse_featurecollection(self):
        from utils.aoi import parse_geojson_bbox
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[[80, 38], [90, 38], [90, 44], [80, 44], [80, 38]]]}}
            ]
        }).encode()
        bbox, name = parse_geojson_bbox(geojson)
        assert bbox == [80.0, 38.0, 90.0, 44.0]
        assert "GeoJSON" in name

    def test_parse_single_geometry(self):
        from utils.aoi import parse_geojson_bbox
        geojson = json.dumps({
            "type": "Polygon",
            "coordinates": [[[100, 30], [110, 30], [110, 40], [100, 40], [100, 30]]]
        }).encode()
        bbox, name = parse_geojson_bbox(geojson)
        assert bbox == [100.0, 30.0, 110.0, 40.0]

    def test_parse_multi_geometry_union(self):
        from utils.aoi import parse_geojson_bbox
        # 两个不相连的多边形，取并集外接矩形
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[[80, 38], [85, 38], [85, 40], [80, 40], [80, 38]]]}},
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[[88, 42], [92, 42], [92, 45], [88, 45], [88, 42]]]}},
            ]
        }).encode()
        bbox, _ = parse_geojson_bbox(geojson)
        assert bbox == [80.0, 38.0, 92.0, 45.0]

    def test_parse_invalid(self):
        from utils.aoi import parse_geojson_bbox
        with pytest.raises(Exception):
            parse_geojson_bbox(b"not a geojson")


class TestValidateBBox:
    def test_valid(self):
        from utils.aoi import validate_bbox
        assert validate_bbox([80, 38, 90, 44])

    def test_invalid_order(self):
        from utils.aoi import validate_bbox
        assert not validate_bbox([90, 38, 80, 44])  # lon_min > lon_max
        assert not validate_bbox([80, 44, 90, 38])  # lat_min > lat_max

    def test_out_of_range(self):
        from utils.aoi import validate_bbox
        assert not validate_bbox([-200, 38, 90, 44])  # lon 超界
        assert not validate_bbox([80, -100, 90, 44])  # lat 超界

    def test_wrong_length(self):
        from utils.aoi import validate_bbox
        assert not validate_bbox([80, 38, 90])
