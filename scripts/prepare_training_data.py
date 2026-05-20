"""
训练数据准备脚本 — 下载 Zenodo dset-s2 水体分割数据集

数据来源: Zenodo record 8310745 (dset-s2)
描述: Sentinel-2 水体分割标注数据集，800+ 景，覆盖全球多种水体类型

运行方式:
    python scripts/prepare_training_data.py

前置条件:
    pip install requests tqdm
"""

import os
import sys
import zipfile
import json
import argparse
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))


ZENODO_RECORD = "8310745"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/dset-s2.zip"
EXPECTED_SIZE_GB = 45  # 约 45GB 解压后


def download_with_progress(url: str, dest_path: str, desc: str = "下载中"):
    """带进度条的下载"""
    import requests
    from tqdm import tqdm

    resp = requests.get(url, stream=True, timeout=300)
    total = int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=desc,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    return dest_path


def extract_zip(zip_path: str, extract_dir: str):
    """解压 ZIP 文件"""
    print(f"解压 {zip_path} → {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"解压完成")


def organize_dataset(raw_dir: str, output_dir: str):
    """
    整理数据集为训练格式:

    期望目录结构:
        data/training/
        ├── images/          # S2 影像 tiles (*.tif)
        ├── masks/           # 水体标注 masks (*.tif)
        └── splits.json      # train/val split 信息
    """
    import shutil

    images_dir = Path(output_dir) / "images"
    masks_dir = Path(output_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(raw_dir)
    tif_files = list(raw.rglob("*.tif")) + list(raw.rglob("*.tiff"))

    image_count = 0
    mask_count = 0

    for f in tif_files:
        fname = f.name.lower()
        if "mask" in fname or "label" in fname or "gt" in fname:
            shutil.copy2(f, masks_dir / f.name)
            mask_count += 1
        else:
            shutil.copy2(f, images_dir / f.name)
            image_count += 1

    print(f"整理完成: {image_count} 张影像, {mask_count} 张标注")

    # 生成 split 文件
    import random
    random.seed(42)

    image_names = sorted([f.name for f in images_dir.glob("*.tif*")])
    random.shuffle(image_names)

    n = len(image_names)
    n_train = int(n * 0.80)
    n_val = int(n * 0.15)
    # n_test = n - n_train - n_val (自动)

    splits = {
        "train": image_names[:n_train],
        "val": image_names[n_train:n_train + n_val],
        "test": image_names[n_train + n_val:],
    }

    splits_path = Path(output_dir) / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"Split 文件: {splits_path}")
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")


def create_tiles_from_images(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 256,
    min_water_ratio: float = 0.01,
):
    """
    将大图切割为训练 tiles (需要 rasterio)。

    参数:
        images_dir: 原始影像目录
        masks_dir: 原始标注目录
        output_dir: 输出 tiles 目录
        tile_size: tile 尺寸 (像素)
        overlap: 重叠像素
        min_water_ratio: 最小水体占比 (过滤含水体过少的 tile)
    """
    try:
        import rasterio
        import numpy as np
    except ImportError:
        print("⚠️  rasterio 未安装，跳过 tile 切割")
        print("    安装: pip install rasterio")
        return

    import rasterio.windows

    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    out_images = Path(output_dir) / "images"
    out_masks = Path(output_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_path.glob("*.tif*"))

    if not image_files:
        print("没有找到影像文件")
        return

    print(f"切割 {len(image_files)} 张影像为 {tile_size}×{tile_size} tiles ...")

    total_tiles = 0
    filtered_tiles = 0

    for img_file in image_files:
        # 查找对应的 mask
        mask_candidates = [
            masks_path / img_file.name,
            masks_path / img_file.name.replace(".tif", "_mask.tif"),
            masks_path / img_file.stem.replace("_img", "_mask") + img_file.suffix,
        ]
        mask_file = None
        for mc in mask_candidates:
            if mc.exists():
                mask_file = mc
                break

        if mask_file is None:
            continue

        with rasterio.open(img_file) as src_img, rasterio.open(mask_file) as src_mask:
            h, w = src_img.height, src_img.width
            stride = tile_size - overlap

            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    window = rasterio.windows.Window(x, y, tile_size, tile_size)

                    # 读取 tile
                    img_tile = src_img.read(window=window)
                    mask_tile = src_mask.read(1, window=window)

                    # 过滤：水体占比太低
                    water_ratio = np.mean(mask_tile > 0)
                    if water_ratio < min_water_ratio:
                        filtered_tiles += 1
                        continue

                    # 保存 tile
                    tile_name = f"{img_file.stem}_y{y}_x{x}.tif"

                    tile_profile = src_img.profile.copy()
                    tile_profile.update(
                        height=tile_size,
                        width=tile_size,
                        transform=rasterio.windows.transform(window, src_img.transform),
                    )

                    with rasterio.open(out_images / tile_name, "w", **tile_profile) as dst:
                        dst.write(img_tile)

                    mask_profile = src_mask.profile.copy()
                    mask_profile.update(
                        height=tile_size,
                        width=tile_size,
                        count=1,
                        transform=rasterio.windows.transform(window, src_mask.transform),
                    )

                    with rasterio.open(out_masks / tile_name, "w", **mask_profile) as dst:
                        dst.write(mask_tile, 1)

                    total_tiles += 1

    print(f"切割完成: {total_tiles} tiles (过滤 {filtered_tiles} 个低水体 tile)")


def main():
    parser = argparse.ArgumentParser(description="准备 AI 水体分割训练数据")
    parser.add_argument("--download", action="store_true", help="从 Zenodo 下载数据集")
    parser.add_argument("--data-dir", default="data/training", help="训练数据目录")
    parser.add_argument("--zenodo-dir", default="data/zenodo_raw", help="Zenodo 原始数据目录")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile 大小")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载")
    parser.add_argument("--skip-organize", action="store_true", help="跳过数据整理")
    parser.add_argument("--skip-tiling", action="store_true", help="跳过 tile 切割")
    args = parser.parse_args()

    print("=" * 60)
    print("Geo AI — 水体分割训练数据准备")
    print("=" * 60)
    print()
    print(f"数据来源: Zenodo record {ZENODO_RECORD}")
    print(f"预期大小: ~{EXPECTED_SIZE_GB} GB (解压后)")
    print(f"输出目录: {args.data_dir}")
    print()

    # Step 1: 下载
    if args.download and not args.skip_download:
        zenodo_path = Path(args.zenodo_dir)
        zenodo_path.mkdir(parents=True, exist_ok=True)

        zip_path = zenodo_path / "dset-s2.zip"
        if zip_path.exists():
            print(f"已存在: {zip_path}, 跳过下载")
        else:
            print(f"开始下载: {DOWNLOAD_URL}")
            print(f"⚠️  文件较大 (~15GB 压缩包), 请耐心等待...")
            download_with_progress(DOWNLOAD_URL, str(zip_path), "dset-s2.zip")
            print("下载完成!")

        # Step 2: 解压
        if not args.skip_organize:
            extract_dir = zenodo_path / "extracted"
            if not extract_dir.exists() or not list(extract_dir.iterdir()):
                extract_zip(str(zip_path), str(extract_dir))
            else:
                print(f"已解压: {extract_dir}")

            # Step 3: 整理
            organize_dataset(str(extract_dir), args.data_dir)

    # Step 4: Tile 切割
    if not args.skip_tiling:
        images_dir = Path(args.data_dir) / "images"
        masks_dir = Path(args.data_dir) / "masks"
        tiles_dir = Path(args.data_dir) / "tiles"

        if images_dir.exists() and masks_dir.exists():
            create_tiles_from_images(
                str(images_dir),
                str(masks_dir),
                str(tiles_dir),
                tile_size=args.tile_size,
                overlap=args.tile_size // 2,
                min_water_ratio=0.01,
            )

    print()
    print("=" * 60)
    print("数据准备完成!")
    print()
    print("下一步:")
    print("  1. 安装 geoai-py: pip install geoai-py")
    print("  2. 确认 GPU 可用: python -c 'import torch; print(torch.cuda.is_available())'")
    print("  3. 运行训练: python scripts/train_water_seg.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
