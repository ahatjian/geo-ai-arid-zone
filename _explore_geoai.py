"""探索 geoai-py API — 安全版本"""
import geoai
import inspect

print("=" * 60)
print("geoai-py API Explorer (safe)")
print("=" * 60)

# 获取属性列表（仅名称，不导入）
all_names = sorted(dir(geoai))
print(f"Total names: {len(all_names)}")

funcs = []
errors = []

for name in all_names:
    if name.startswith('_'):
        continue
    try:
        obj = getattr(geoai, name)
        if callable(obj):
            try:
                sig = str(inspect.signature(obj))
                funcs.append((name, sig))
            except Exception:
                funcs.append((name, "(signature unavailable)"))
    except Exception as e:
        errors.append((name, str(e)[:80]))

for name, sig in funcs:
    print(f"  FUNCTION: {name}{sig}")

if errors:
    print()
    print("--- Unavailable (missing deps) ---")
    for name, err in errors:
        print(f"  SKIP: {name} -> {err}")

print()
print(f"Callable: {len(funcs)}, Unavailable: {len(errors)}")
