"""Create synthetic 6-band Sentinel-2 style GeoTIFF for testing water segmentation."""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

h, w = 256, 256
y, x = np.ogrid[:h, :w]
cy, cx = h // 2, w // 2

# Water body mask (ellipse lake + river)
water_mask = np.zeros((h, w), dtype=bool)
water_mask |= ((x - cx) ** 2 / (w * 0.25) ** 2 + (y - cy) ** 2 / (h * 0.3) ** 2) < 1
river_start = cy - int(h * 0.2)
water_mask |= (x > cx - 10) & (x < cx + 10) & (y > river_start) & (y < h)

# Generate realistic spectral values for 6 bands (B2,B3,B4,B8,B11,B12)
bands = np.zeros((6, h, w), dtype=np.uint16)
np.random.seed(42)

specs = [
    (1000, 1200),   # B2 Blue: land lower, water slightly higher
    (1400, 1500),   # B3 Green
    (1800, 800),    # B4 Red: water much lower
    (3500, 200),    # B8 NIR: water absorbs NIR heavily
    (2500, 100),    # B11 SWIR1
    (2000, 80),     # B12 SWIR2
]

for b, (land_val, water_val) in enumerate(specs):
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    base = np.where(water_mask, water_val, land_val)
    bands[b] = (base + noise).clip(0, 65535).astype(np.uint16)

# Write GeoTIFF
transform = from_origin(94.0, 39.0, 0.0001, 0.0001)
meta = {
    'driver': 'GTiff', 'height': h, 'width': w, 'count': 6,
    'dtype': 'uint16', 'crs': CRS.from_epsg(4326), 'transform': transform,
    'compress': 'lzw'
}
out_path = 'test_water_image.tif'
with rasterio.open(out_path, 'w', **meta) as dst:
    for i in range(6):
        dst.write(bands[i], i + 1)
    dst.descriptions = ('B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'B11_SWIR1', 'B12_SWIR2')

print(f'Created {out_path}: {h}x{w} x 6 bands (uint16)')
print(f'Water pixels: {water_mask.sum()} / {h*w} ({100*water_mask.sum()/h/w:.1f}%)')
ndwi = (1500.0 - 200.0) / (1500.0 + 200.0)
print(f'NDWI (Green-NIR)/(Green+NIR) for water = {ndwi:.3f} (should be >0 for water)')
