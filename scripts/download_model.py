"""Download OmniWaterMask model via HF mirror and convert to .pth format.

Uses: https://hf-mirror.com as the HF endpoint (accessible in China).
"""
import os
import sys
from pathlib import Path

# Set mirror BEFORE importing huggingface_hub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from safetensors.torch import load_file
import torch
from huggingface_hub import hf_hub_download

MODEL_DIR = Path.home() / "AppData" / "Local" / "omniwatermask" / "omniwatermask" / "0.4.3"
SAFETENSORS_FN = "PM_model_1.5.38_s1s2_water_flair_convnextv2_base_PT.pth_weights.safetensors"
PTH_FN = "PM_model_1.5.38_s1s2_water_flair_convnextv2_base_PT.pth_weights.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
pth_path = MODEL_DIR / PTH_FN

# Check if already downloaded
if pth_path.exists() and pth_path.stat().st_size > 10 * 1024 * 1024:
    print(f"Model already exists: {pth_path} ({pth_path.stat().st_size/1024/1024:.0f} MB)")
    print("Skipping download.")
    sys.exit(0)

print(f"Downloading OmniWaterMask model via {os.environ['HF_ENDPOINT']}...")
print(f"Target dir: {MODEL_DIR}")

try:
    safetensors_path = hf_hub_download(
        repo_id="NickWright/OmniWaterMask",
        filename=SAFETENSORS_FN,
        cache_dir=str(MODEL_DIR),
        force_download=False,
        resume_download=True,
    )
    print(f"Downloaded safetensors: {safetensors_path}")
    print(f"Size: {Path(safetensors_path).stat().st_size/1024/1024:.0f} MB")

    # Convert to .pth format
    print("Converting safetensors -> pytorch .pth...")
    state = load_file(safetensors_path)
    torch.save(state, str(pth_path))
    print(f"Saved: {pth_path} ({pth_path.stat().st_size/1024/1024:.0f} MB)")
    print("\n=== SUCCESS! Model ready for inference ===")

except Exception as e:
    print(f"\nAuto-download failed: {e}")
    print("\n=== MANUAL DOWNLOAD INSTRUCTIONS ===")
    print(f"1. Download from hf-mirror.com:")
    print(f"   https://hf-mirror.com/NickWright/OmniWaterMask/resolve/main/{SAFETENSORS_FN}")
    print(f"2. Place file in: {MODEL_DIR}")
    print(f"3. Run: python scripts/convert_safetensors_to_pth.py")
