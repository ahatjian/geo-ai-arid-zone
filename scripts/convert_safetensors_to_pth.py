r"""Convert OmniWaterMask .safetensors to .pth format after manual download.

Usage:
    1. Download the .safetensors file manually (see DOWNLOAD_GUIDE.md)
    2. Place it in the model directory (default: C:\Users\AHATJIAN\AppData\Local\omniwatermask\omniwatermask\0.4.3)
    3. Run this script: python scripts/convert_safetensors_to_pth.py
"""
import sys
from pathlib import Path
from safetensors.torch import load_file
import torch

# Model directory
MODEL_DIR = Path.home() / "AppData" / "Local" / "omniwatermask" / "omniwatermask" / "0.4.3"
SAFETENSORS_FILE = "PM_model_1.5.38_s1s2_water_flair_convnextv2_base_PT_weights.safetensors"
PTH_FILE = "PM_model_1.5.38_s1s2_water_flair_convnextv2_base_PT.pth_weights.pth"

# Allow custom model dir
if len(sys.argv) > 1:
    MODEL_DIR = Path(sys.argv[1])

MODEL_DIR.mkdir(parents=True, exist_ok=True)

safetensors_path = MODEL_DIR / SAFETENSORS_FILE
pth_path = MODEL_DIR / PTH_FILE

if not safetensors_path.exists():
    print(f"ERROR: .safetensors file not found at:")
    print(f"  {safetensors_path}")
    print()
    print("Please download the file first. Options:")
    print(f"  1. HF Mirror: https://hf-mirror.com/NickWright/OmniWaterMask/resolve/main/{SAFETENSORS_FILE}")
    print(f"  2. HuggingFace: https://huggingface.co/NickWright/OmniWaterMask/resolve/main/{SAFETENSORS_FILE}")
    print(f"  3. Google Drive: https://drive.google.com/uc?id=15gug9tQZWDCDz8cRF_sgLegYjDmGMiaQ&export=download")
    print()
    print(f"Then place it in: {MODEL_DIR}")
    sys.exit(1)

print(f"Loading {safetensors_path}...")
print(f"File size: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")
state = load_file(str(safetensors_path))

print(f"Saving as {pth_path}...")
torch.save(state, str(pth_path))

print(f"Done! Model saved: {pth_path} ({pth_path.stat().st_size / 1024 / 1024:.1f} MB)")
print()
print("Now you can run: python scripts/test_segment_water.py")
