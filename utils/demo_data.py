"""Local demonstration data layer for offline/weak-network reliability.

The platform normally depends on Microsoft Planetary Computer. This module
generates deterministic, georeferenced synthetic scenes with realistic
spectral signatures so that every downstream module can still complete an
end-to-end analysis when the network is unavailable.

The generated data is intentionally simple but valid:
- Sentinel-2 style six-band surface reflectance
- Landsat style six-band surface reflectance plus a thermal asset
- spatial patterns for vegetation, water, bare land, and moisture gradients
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


def _cache_root() -> str:
    from config import CACHE_DIR
    root = os.path.join(CACHE_DIR, "demo_data")
    os.makedirs(root, exist_ok=True)
    return root


def _scene_key(bbox: List[float], collection: str, date_str: str) -> str:
    payload = f"{bbox[0]:.5f}_{bbox[1]:.5f}_{bbox[2]:.5f}_{bbox[3]:.5f}_{collection}_{date_str}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


class DemoAsset:
    def __init__(self, href: str, media_type: str = "image/tiff"):
        self.href = href
        self.media_type = media_type
        self.roles: List[str] = []


class DemoItem:
    """Minimal STAC-like item accepted by pc_data download helpers."""

    def __init__(self, item_id: str, date_str: str, bbox: List[float], assets: Dict[str, DemoAsset]):
        self.id = item_id
        self.datetime = datetime.strptime(date_str, "%Y-%m-%d")
        self.bbox = bbox
        self.assets = assets
        self.properties = {"eo:cloud_cover": 2.0, "platform": "demo"}
        self.collection = "demo"


def _make_scene(
    bbox: List[float],
    date_str: str,
    collection: str,
    scene_index: int,
    n_scenes: int,
) -> Dict[str, str]:
    """Write synthetic multiband and optional thermal GeoTIFFs to disk."""
    width, height = 192, 192
    lat = np.linspace(bbox[1], bbox[3], height)
    lon = np.linspace(bbox[0], bbox[2], width)
    yy, xx = np.meshgrid(lat, lon)

    rng = np.random.RandomState(101 + scene_index + int(bbox[0]) % 37)

    # Build a deterministic landscape with broad spatial structure.
    veg = (
        0.18
        + 0.30 * np.exp(-((yy - np.mean(lat)) ** 2) / (0.55 * (bbox[3] - bbox[1]) + 1e-6) ** 2)
        + 0.08 * np.sin(xx * 1.3 + scene_index * 0.7)
        + 0.05 * rng.randn(height, width)
    )
    water = (
        0.16
        * np.exp(-((xx - bbox[0] - 0.28 * (bbox[2] - bbox[0])) ** 2) / (0.06 * (bbox[2] - bbox[0]) + 1e-6) ** 2)
        + 0.12
        * np.exp(-((yy - bbox[1] - 0.68 * (bbox[3] - bbox[1])) ** 2) / (0.05 * (bbox[3] - bbox[1]) + 1e-6) ** 2)
    )
    moisture = np.clip(0.55 - np.abs(yy - np.median(lat)) / (bbox[3] - bbox[1] + 1e-6), 0, 1)

    # Temporal trend gives trend/forecast modules a meaningful signal.
    phase = (scene_index / max(n_scenes - 1, 1)) - 0.5
    veg = np.clip(veg + 0.08 * phase, 0.0, 1.0)
    water = np.clip(water + 0.05 * phase, 0.0, 1.0)

    # Soil/dry background.

    veg_mask = np.clip((veg - water) * 1.6, 0.0, 1.0)
    water_mask = np.clip((water - veg) * 1.8, 0.0, 1.0)
    soil_mask = 1.0 - veg_mask - water_mask

    blue = soil_mask * (0.16 - 0.05 * moisture) + veg_mask * 0.05 + water_mask * 0.035 + 0.01 * rng.randn(height, width)
    green = soil_mask * 0.23 + veg_mask * 0.10 + water_mask * 0.075 + 0.01 * rng.randn(height, width)
    red = soil_mask * 0.28 + veg_mask * 0.06 + water_mask * 0.045 + 0.01 * rng.randn(height, width)
    nir = soil_mask * 0.32 + veg_mask * 0.48 + water_mask * 0.015 + 0.01 * rng.randn(height, width)
    swir1 = soil_mask * 0.40 + veg_mask * 0.21 + water_mask * 0.010 + 0.01 * rng.randn(height, width)
    swir2 = soil_mask * 0.38 + veg_mask * 0.14 + water_mask * 0.008 + 0.01 * rng.randn(height, width)

    stack = np.stack([blue, green, red, nir, swir1, swir2]).astype(np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    import rasterio
    from rasterio.transform import from_bounds

    multiband_path = os.path.join(
        _cache_root(),
        f"{_scene_key(bbox, collection, date_str)}_6band.tif",
    )
    thermal_path = os.path.join(
        _cache_root(),
        f"{_scene_key(bbox, collection, date_str)}_thermal.tif",
    )

    if not os.path.exists(multiband_path):
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
        with rasterio.open(
            multiband_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=6,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(stack)
            dst.set_band_description(1, "B")
            dst.set_band_description(2, "G")
            dst.set_band_description(3, "R")
            dst.set_band_description(4, "NIR")
            dst.set_band_description(5, "SWIR1")
            dst.set_band_description(6, "SWIR2")

    if "Landsat" in collection and not os.path.exists(thermal_path):
        # LST in Kelvin, then encoded into Collection-2 DN using the platform constants.
        lst_kelvin = 292.0 + 8.0 * veg + 5.0 * phase + 2.5 * rng.randn(height, width)
        from config import LANDSAT_ST_SCALE, LANDSAT_ST_OFFSET

        st_dn = ((lst_kelvin - LANDSAT_ST_OFFSET) / LANDSAT_ST_SCALE).astype(np.float32)
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
        with rasterio.open(
            thermal_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(st_dn, 1)

    return {"multiband": multiband_path, "thermal": thermal_path}


def _assets_for(collection: str, scene_files: Dict[str, str]) -> Dict[str, DemoAsset]:
    if "Sentinel" in collection:
        names = ["B02", "B03", "B04", "B08", "B11", "B12"]
    else:
        names = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

    assets = {name: DemoAsset(scene_files["multiband"]) for name in names}
    if "Landsat" in collection and os.path.exists(scene_files["thermal"]):
        asset_name = "ST_B10" if collection in ("Landsat-8", "Landsat-9") else "ST_B6"
        assets[asset_name] = DemoAsset(scene_files["thermal"])
    return assets


def build_demo_search_results(
    bbox: List[float],
    start_date: str,
    end_date: str,
    collection: str = "Sentinel-2 L2A",
    cloud_cover_max: float = 20,
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    """Create enough time steps for trend, forecast, and time-series pages."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        start = datetime.now() - timedelta(days=365)
        end = datetime.now()

    n_scenes = max(3, min(int(max_items), 8))
    if (end - start).days < n_scenes:
        start = end - timedelta(days=n_scenes * 30)

    results: List[Dict[str, Any]] = []
    for i in range(n_scenes):
        date = start + timedelta(days=round((end - start).days * i / max(n_scenes - 1, 1)))
        date_str = date.strftime("%Y-%m-%d")
        scene_files = _make_scene(bbox, date_str, collection, i, n_scenes)
        item_id = f"demo_{_scene_key(bbox, collection, date_str)}"
        item = DemoItem(item_id, date_str, bbox, _assets_for(collection, scene_files))
        results.append(
            {
                "id": item_id,
                "datetime": date_str,
                "cloud_cover": 2.0 + (i % 3) * 0.5,
                "bbox": bbox,
                "thumbnail_url": None,
                "item": item,
                "demo": True,
            }
        )
    return results


def is_demo_item(item: Any) -> bool:
    return bool(getattr(item, "collection", None) == "demo" or getattr(item, "id", "").startswith("demo_"))


def get_demo_rgb(item_id: str, collection: str, width: int = 512) -> Optional[Any]:
    """Render an RGB preview directly from cached demo data."""
    import io

    try:
        if not item_id.startswith("demo_"):
            return None
        # The id encodes bbox/date, but the cached file name is more reliable to locate.
        root = _cache_root()
        candidates = [f for f in os.listdir(root) if f.startswith(item_id.replace("demo_", "")) and f.endswith("_6band.tif")]
        if not candidates:
            return None
        path = os.path.join(root, candidates[0])
        import rasterio
        from PIL import Image

        with rasterio.open(path) as src:
            arr = src.read([3, 2, 1]).astype(np.float32)
        arr = np.transpose(arr, (1, 2, 0))
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        rgb = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        if width and width != img.width:
            height = int(img.height * width / img.width)
            img = img.resize((width, height))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None
