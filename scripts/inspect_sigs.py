"""Inspect segment_water and make_water_mask function signatures."""
import inspect

print('=== segment_water ===')
from geoai.water import segment_water
sig = inspect.signature(segment_water)
for name, param in sig.parameters.items():
    if param.default is inspect.Parameter.empty:
        print(f'  {name}: required')
    else:
        print(f'  {name} = {param.default!r}')

print()
print('=== make_water_mask ===')
from omniwatermask.inference import make_water_mask
sig2 = inspect.signature(make_water_mask)
for name, param in sig2.parameters.items():
    if param.default is inspect.Parameter.empty:
        print(f'  {name}: required')
    else:
        print(f'  {name} = {param.default!r}')

print()
print('=== segment_water source (first 30 lines) ===')
import inspect as ins
src = ins.getsource(segment_water)
lines = src.split('\n')[:60]
for line in lines:
    print(line)
