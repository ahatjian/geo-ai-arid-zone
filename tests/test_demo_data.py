"""Tests for the offline demonstration data layer."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.demo_data import build_demo_search_results, is_demo_item  # noqa: E402
from utils.pc_data import download_multiband, download_asset  # noqa: E402


def test_demo_search_sentinel_has_enough_scenes_and_assets():
    results = build_demo_search_results(
        [76, 37, 88, 42],
        "2025-01-01",
        "2025-12-31",
        collection="Sentinel-2 L2A",
        max_items=6,
    )
    assert len(results) >= 3
    item = results[0]["item"]
    assert is_demo_item(item)
    assert all(name in item.assets for name in ["B02", "B03", "B04", "B08", "B11", "B12"])


def test_demo_multiband_download_has_six_bands():
    results = build_demo_search_results(
        [94, 36, 98, 39],
        "2025-01-01",
        "2025-12-31",
        collection="Landsat-8",
        max_items=3,
    )
    path = os.path.join(tempfile.gettempdir(), "demo_test_multiband.tif")
    out = download_multiband(results[0]["item"], path, collection="Landsat-8")
    assert out and os.path.exists(path)

    import rasterio

    with rasterio.open(path) as src:
        assert src.count == 6
        arr = src.read()
        assert arr.shape == (6, 192, 192)
        assert np.isfinite(arr).all()


def test_demo_landsat_thermal_download():
    results = build_demo_search_results(
        [82, 43, 90, 45],
        "2025-01-01",
        "2025-12-31",
        collection="Landsat-9",
        max_items=3,
    )
    item = results[0]["item"]
    assert "ST_B10" in item.assets

    path = os.path.join(tempfile.gettempdir(), "demo_test_thermal.tif")
    out = download_asset(item, "ST_B10", path)
    assert out and os.path.exists(path)

    import rasterio

    with rasterio.open(path) as src:
        assert src.count == 1
