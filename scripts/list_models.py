"""List all models that need to be downloaded for OmniWaterMask."""
import inspect
import omniwatermask.download_models as dm

print("=== get_models source ===")
src = inspect.getsource(dm.get_models)
print(src)

print("\n=== download_file source ===")
src2 = inspect.getsource(dm.download_file)
print(src2)

print("\n=== download_file_from_hugging_face source ===")
src3 = inspect.getsource(dm.download_file_from_hugging_face)
print(src3)

# Also find model details
print("\n=== Model info ===")
if hasattr(dm, 'MODEL_REPO'):
    print(f"MODEL_REPO: {dm.MODEL_REPO}")
if hasattr(dm, 'MODEL_FILES'):
    print(f"MODEL_FILES: {dm.MODEL_FILES}")
if hasattr(dm, 'REPO_ID'):
    print(f"REPO_ID: {dm.REPO_ID}")

# Try to find all model-related constants
print("\n=== All top-level attributes ===")
for attr in dir(dm):
    if not attr.startswith('_'):
        val = getattr(dm, attr)
        if not inspect.isfunction(val) and not inspect.ismodule(val):
            print(f"  {attr} = {val!r}")
