"""Inspect segment_water and related water.py details."""
import inspect
import geoai.water as gw

# Get full source
src = inspect.getsource(gw.segment_water)
lines = src.split('\n')

# Find band_order related code
print("=== band_order reference lines ===")
for i, line in enumerate(lines):
    lower = line.lower()
    if 'band_order' in lower or 'band' in lower or 'naip' in lower or 'sentinel' in lower:
        print(f"  L{i+1}: {line.strip()}")

print("\n=== make_water_mask call context ===")
for i, line in enumerate(lines):
    if 'make_water_mask' in line or 'owm_kwargs' in line or 'osm_filter' in line:
        print(f"  L{i+1}: {line.strip()}")

# Find make_water_mask import
print("\n=== omniwatermask import ===")
for i, line in enumerate(lines):
    if 'omniwatermask' in line.lower() or 'from o' in line.lower():
        print(f"  L{i+1}: {line.strip()}")

# Find omniwatermask module structure
print("\n=== omniwatermask modules ===")
import omniwatermask
for attr in dir(omniwatermask):
    if not attr.startswith('_'):
        obj = getattr(omniwatermask, attr)
        if inspect.ismodule(obj):
            print(f"  Module: omniwatermask.{attr}")
            # Check if make_water_mask is in submodule
            if hasattr(obj, 'make_water_mask'):
                print(f"    -> has make_water_mask")
                sig = inspect.signature(obj.make_water_mask)
                for n, p in sig.parameters.items():
                    d = p.default
                    if d is inspect.Parameter.empty:
                        print(f"      {n}: required")
                    else:
                        print(f"      {n} = {d!r}")
