"""导出与可视化模块测试 — export / visualization"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


class TestExportCSV:
    def test_export_dataframe(self, tmp_path):
        from utils.export import export_csv
        df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.6]})
        path, content = export_csv(df, str(tmp_path / "t.csv"))
        assert os.path.exists(path)
        assert b"a,b" in content
        # utf-8-sig BOM (Excel 兼容)
        assert content[:3] == b"\xef\xbb\xbf"

    def test_export_dict(self, tmp_path):
        from utils.export import export_csv
        path, content = export_csv({"x": 1, "y": 2}, str(tmp_path / "d.csv"))
        assert os.path.exists(path)
        assert b"x" in content

    def test_export_list_of_dict(self, tmp_path):
        from utils.export import export_csv
        path, content = export_csv([{"name": "A", "v": 1}], str(tmp_path / "l.csv"))
        assert b"A" in content

    def test_no_path_uses_temp(self):
        """无输出路径时用临时文件"""
        from utils.export import export_csv
        path, content = export_csv(pd.DataFrame({"a": [1]}))
        assert os.path.exists(path)
        os.remove(path)


class TestExportStats:
    def test_stats_csv(self, tmp_path):
        from utils.export import export_stats_csv
        stats = {"植被占比": 0.4, "水体占比": 0.1, "总面积km2": 10.5}
        path, content = export_stats_csv(stats, str(tmp_path / "s.csv"))
        assert os.path.exists(path)
        assert "植被占比".encode("utf-8") in content

    def test_timeseries_csv(self, tmp_path):
        from utils.export import export_timeseries_csv
        path, content = export_timeseries_csv(
            ["2024-01", "2024-02"], [0.3, 0.35],
            str(tmp_path / "ts.csv"),
        )
        assert b"0.35" in content


class TestExportGeoTIFF:
    def test_export_index_geotiff(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin
        from utils.export import export_index_geotiff

        # 造源文件 (提供 CRS/transform 参考)
        src = str(tmp_path / "src.tif")
        arr = np.random.default_rng(0).random((30, 30)).astype(np.float32)
        with rasterio.open(src, "w", driver="GTiff", height=30, width=30, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=from_origin(80, 40, 10, 10)) as dst:
            dst.write(arr, 1)

        out = str(tmp_path / "ndvi.tif")
        result = export_index_geotiff(arr, src, out)
        # 返回 (path, content) 或 path
        path = result[0] if isinstance(result, tuple) else result
        assert os.path.exists(path)
        with rasterio.open(path) as src_ds:
            assert src_ds.count == 1
            assert src_ds.crs is not None

    def test_make_download_label(self):
        from utils.export import make_download_label
        label = make_download_label("ndvi", "tif")
        assert label.startswith("ndvi_") and label.endswith(".tif")
        label2 = make_download_label("ndvi", "csv", timestamp=False)
        assert label2 == "ndvi.csv"


class TestVisualization:
    def test_render_ndvi(self):
        from utils.visualization import render_ndvi
        from PIL import Image
        ndvi = np.random.default_rng(0).uniform(-1, 1, (40, 40))
        img = render_ndvi(ndvi)
        assert isinstance(img, Image.Image)

    def test_render_mndwi(self):
        from utils.visualization import render_mndwi
        from PIL import Image
        mndwi = np.random.default_rng(0).uniform(-1, 1, (40, 40))
        img = render_mndwi(mndwi)
        assert isinstance(img, Image.Image)

    def test_render_classification(self):
        from utils.visualization import render_classification
        from PIL import Image
        cls = np.random.default_rng(0).integers(0, 6, (40, 40)).astype(np.uint8)
        img = render_classification(cls)
        assert isinstance(img, Image.Image)

    def test_plot_histogram(self):
        import plotly.graph_objects as go
        from utils.visualization import plot_histogram
        arr = np.random.default_rng(0).uniform(-1, 1, (40, 40))
        fig = plot_histogram(arr, bins=30)
        assert isinstance(fig, go.Figure)

    def test_plot_time_series(self):
        import plotly.graph_objects as go
        from utils.visualization import plot_time_series
        fig = plot_time_series(
            ["2024-01", "2024-02", "2024-03"], [0.3, 0.35, 0.4],
        )
        assert isinstance(fig, go.Figure)

    def test_nan_handling(self):
        """含 NaN 的数组应能渲染 (NaN 显示为透明/黑)"""
        from utils.visualization import render_ndvi
        arr = np.full((20, 20), 0.5)
        arr[5:10, 5:10] = np.nan
        img = render_ndvi(arr)
        assert img is not None
