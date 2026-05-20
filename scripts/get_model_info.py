"""Get all model download info for manual download guide."""
import omniwatermask
from importlib import resources
import pandas as pd
from pathlib import Path
from omniwatermask.download_models import get_model_data_dir

print("Default model dir:", get_model_data_dir())
print()

# Read model CSV
with resources.files('omniwatermask').joinpath('model_download_links.csv').open() as f:
    df = pd.read_csv(f)

print("=== Model file to download ===")
for _, row in df.iterrows():
    fn = row['file_name']
    stem = fn.replace('.pth', '') if fn.endswith('.pth') else fn
    print(f"  Local filename: {fn}")
    print(f"  HF safetensors: {stem}.safetensors")
    print(f"  HF Repo: NickWright/OmniWaterMask")
    print(f"  HF full URL: https://huggingface.co/NickWright/OmniWaterMask/resolve/main/{stem}.safetensors")
    print(f"  HF Mirror URL: https://hf-mirror.com/NickWright/OmniWaterMask/resolve/main/{stem}.safetensors")
    print(f"  Google Drive ID: {row['google_drive_id']}")
    print(f"  Google Drive URL: https://drive.google.com/uc?id={row['google_drive_id']}&export=download")
    print(f"  Model library: {row['model_library']}")
    print(f"  timm model: {row['timm_model_name']}")
    print(f"  Version: {row['version']}")

# Also check what data files exist
print("\n=== omniwatermask data files ===")
omni_dir = Path(omniwatermask.__path__[0])
for f in sorted(omni_dir.rglob("*")):
    if f.is_file() and not f.name.endswith('.py') and not f.name == '__init__.py':
        print(f"  {f.relative_to(omni_dir)}")

# Check geoai models
print("\n=== geoai model info ===")
import geoai
geoai_dir = Path(geoai.__path__[0])
print(f"geoai path: {geoai_dir}")
# Check for models directory
models_dir = geoai_dir / "models"
if models_dir.exists():
    print(f"models dir: {models_dir}")
    for f in sorted(models_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size/1024/1024:.1f} MB)" if f.is_file() else f"  {f.name}/")

# Print HuggingFace cache dir
print(f"\nHF cache dir: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
print(f"  (check if exists: {(Path.home() / '.cache' / 'huggingface' / 'hub').exists()})")
