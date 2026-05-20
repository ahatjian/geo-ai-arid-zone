"""Test geoai.segment_water() on synthetic test image - v2."""
import sys
import time

sys.path.insert(0, '.')
import geoai

print("=" * 60)
print("geoai.segment_water() Integration Test v2")
print("=" * 60)
print("band_order: sentinel2 (R=band3, G=band2, B=band1, NIR=band4)")
print("device: cpu")
print("=" * 60)

t_start = time.time()

try:
    result = geoai.segment_water(
        'test_water_image.tif',
        band_order='sentinel2',       # [3,2,1,4] for Sentinel-2 style
        device='cpu',
        patch_size=256,               # match our image size
        overlap_size=32,              # small overlap
        use_osm_water=False,          # skip OSM filtering for test
        use_osm_building=False,
        use_osm_roads=False,
        batch_size=1,
        verbose=True,
    )
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUCCESS! Completed in {elapsed:.1f}s")
    print(f"Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"Keys: {list(result.keys())}")
        for k, v in result.items():
            print(f"  {k}: {type(v).__name__}", end="")
            if hasattr(v, 'shape'):
                print(f" shape={v.shape} dtype={v.dtype}")
            elif hasattr(v, '__len__') and not isinstance(v, str):
                print(f" len={len(v)}")
            else:
                print(f" = {v}")
    elif isinstance(result, (str, list)):
        print(f"Result: {result}")
    else:
        print(f"Result: {result}")

except Exception as e:
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"FAILED after {elapsed:.1f}s")
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
