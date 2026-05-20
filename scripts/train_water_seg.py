"""
UNet 水体分割模型训练脚本 (geoai-py 适配版)

基于 geoai.train_segmentation_model 高层 API 训练 Sentinel-2 影像的水体语义分割模型。
geoai-py 内置完整的训练管线，无需手写 DataLoader / 损失函数 / 训练循环。

运行前请确认:
    1. geoai-py 已安装: pip install geoai-py
    2. GPU 可用: python -c "import torch; print(torch.cuda.is_available())"
    3. 训练数据已准备: python scripts/prepare_training_data.py --download

运行方式:
    # 使用默认配置训练
    python scripts/train_water_seg.py

    # 使用自定义配置
    python scripts/train_water_seg.py --config scripts/train_config.local.json

    # 仅测试推理
    python scripts/train_water_seg.py --inference-only --input path/to/image.tif
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 环境检查
# ---------------------------------------------------------------------------

def check_environment() -> dict:
    """检查运行环境，返回诊断信息的字典。"""
    info = {}

    # Python
    info["python"] = sys.version

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            info["cuda_warning"] = "⚠️ 未检测到 GPU，训练将非常缓慢。建议使用 GPU 环境。"
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["torch_error"] = "❌ PyTorch 未安装"
        info["device"] = "cpu"

    # geoai-py
    try:
        import geoai
        info["geoai_version"] = (
            geoai.__version__ if hasattr(geoai, "__version__") else "installed"
        )
    except ImportError:
        info["geoai_error"] = "❌ geoai-py 未安装。运行: pip install geoai-py"

    return info


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件，过滤掉下划线开头的注释字段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(config: dict, data_dir: str):
    """
    使用 geoai.train_segmentation_model 进行训练。

    geoai-py 内置了:
      - 数据加载 (支持 directory / coco / yolo 三种格式)
      - 模型创建 (UNet / DeepLabV3+ / FPN 等)
      - 训练循环 + 验证 + Early Stopping
      - 模型保存 (best_model.pth / final_model.pth)
      - 训练曲线绘制
    """
    import geoai

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    images_dir = str(Path(data_dir) / "images")
    labels_dir = str(Path(data_dir) / "masks")
    output_dir = output_cfg["model_dir"]

    # 验证数据目录
    for d, label in [(images_dir, "images"), (labels_dir, "masks")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{label} 目录不存在: {d}\n"
                f"请先运行: python scripts/prepare_training_data.py --download"
            )

    print(f"\n{'='*60}")
    print(f"模型架构:  {model_cfg['architecture']} + {model_cfg['encoder_name']}")
    print(f"输入通道:  {model_cfg['in_channels']} 波段 (S2: B2,B3,B4,B8,B11,B12)")
    print(f"输出类别:  {model_cfg['num_classes']} (背景 + 水体)")
    print(f"图像目录:  {images_dir}")
    print(f"标签目录:  {labels_dir}")
    print(f"输出目录:  {output_dir}")
    print(f"Epochs:    {train_cfg['epochs']}")
    print(f"Batch:     {train_cfg['batch_size']}")
    print(f"{'='*60}\n")

    geoai.train_segmentation_model(
        # ---- 数据 ----
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        input_format="directory",

        # ---- 模型 ----
        architecture=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        num_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],

        # ---- 训练超参 ----
        batch_size=train_cfg["batch_size"],
        num_epochs=train_cfg["epochs"],
        learning_rate=train_cfg.get("learning_rate", 0.0001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        val_split=train_cfg.get("validation_split", 0.15),

        # ---- 输出控制 ----
        save_best_only=True,
        plot_curves=True,
        verbose=True,
    )

    # 训练完成后定位模型文件
    best_model = Path(output_dir) / "best_model.pth"
    final_model = Path(output_dir) / "final_model.pth"
    curves_png = Path(output_dir) / "training_curves.png"

    print(f"\n{'='*60}")
    print("训练完成！")
    if best_model.exists():
        print(f"最佳模型: {best_model}")
    if final_model.exists():
        print(f"最终模型: {final_model}")
    if curves_png.exists():
        print(f"训练曲线: {curves_png}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def run_inference(config: dict, input_path: str, output_path: str = None):
    """
    使用训练好的模型进行水体分割推理。

    自动查找 output_dir 下的 best_model.pth 或 final_model.pth，
    调用 geoai.semantic_segmentation 进行滑动窗口推理。
    """
    import numpy as np
    import geoai

    try:
        import rasterio
    except ImportError:
        print("❌ rasterio 未安装。运行: pip install rasterio")
        return

    model_cfg = config["model"]
    output_cfg = config["output"]
    infer_cfg = config["inference"]

    # ---- 1. 定位模型文件 ----
    model_dir = Path(output_cfg["model_dir"])
    candidates = [
        model_dir / "best_model.pth",
        model_dir / "final_model.pth",
    ]
    model_path = None
    for p in candidates:
        if p.exists():
            model_path = str(p)
            break
    if model_path is None:
        # 回退：递归搜索 .pth 文件
        models = list(model_dir.rglob("*.pth"))
        if not models:
            print(f"❌ 在 {model_dir} 下未找到任何 .pth 模型文件")
            print("   请先训练模型: python scripts/train_water_seg.py")
            return
        model_path = str(models[0])

    # ---- 2. 输出路径 ----
    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_water_mask.tif")

    print(f"\n{'='*60}")
    print(f"模型:  {model_path}")
    print(f"输入:  {input_path}")
    print(f"输出:  {output_path}")
    print(f"{'='*60}\n")

    # ---- 3. 推理 ----
    geoai.semantic_segmentation(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        architecture=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        num_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        window_size=infer_cfg["window_size"],
        overlap=infer_cfg["overlap"],
        batch_size=infer_cfg["batch_size"],
    )

    # ---- 4. 统计 (可选: 用 rasterio 读取结果做面积统计) ----
    try:
        with rasterio.open(output_path) as dst:
            mask = dst.read(1)
            water_px = int(np.sum(mask > 0))
            total_px = mask.size
            water_pct = 100.0 * water_px / total_px if total_px else 0

            pixel_area = abs(dst.transform.a * dst.transform.e)
            water_area_km2 = water_px * pixel_area / 1e6

        print(f"✅ 结果已保存: {output_path}")
        print(f"   水体像素: {water_px:,} / {total_px:,} ({water_pct:.2f}%)")
        print(f"   水体面积: ~{water_area_km2:.2f} km²")
    except Exception as e:
        print(f"✅ 结果已保存: {output_path}")
        print(f"   (统计失败: {e})")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="训练 UNet 水体分割模型 (geoai-py)"
    )
    parser.add_argument(
        "--config",
        default="scripts/train_config.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data-dir",
        default="data/training",
        help="训练数据根目录 (含 images/ 和 masks/ 子目录)",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="跳过训练，仅运行推理",
    )
    parser.add_argument(
        "--input",
        help="推理输入 GeoTIFF 路径",
    )
    parser.add_argument(
        "--output",
        help="推理输出 GeoTIFF 路径 (默认: <input>_water_mask.tif)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Geo AI — UNet 水体分割模型训练 (geoai-py)")
    print(f"配置: {args.config}")
    print("=" * 60)
    print()

    # ---- 环境检查 ----
    print("=== 环境检查 ===")
    env = check_environment()
    for key, val in env.items():
        print(f"  {key}: {val}")
    print()

    # ---- 加载配置 ----
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    config = load_config(args.config)

    # ---- 推理模式 ----
    if args.inference_only:
        if not args.input:
            print("❌ 推理模式需要 --input 参数")
            sys.exit(1)
        if "geoai_error" in env:
            print(env["geoai_error"])
            sys.exit(1)
        run_inference(config, args.input, args.output)
        return

    # ---- 训练模式 ----
    if "geoai_error" in env:
        print(env["geoai_error"])
        print("\n请在 GPU 环境中安装 geoai-py 后重新运行:")
        print("  pip install geoai-py")
        print("  python scripts/train_water_seg.py")
        sys.exit(0)

    if not env.get("cuda_available"):
        print("⚠️  未检测到 GPU。CPU 训练非常缓慢。")
        # 非交互式环境下跳过询问
        if sys.stdin.isatty():
            response = input("是否继续? (y/N): ")
            if response.lower() != "y":
                print("已取消。请在 GPU 环境中运行。")
                sys.exit(0)
        else:
            print("   将在 CPU 上运行 (可能非常慢)")

    train(config, args.data_dir)

    print("\n✅ 训练完成!")
    print(f"   推理命令: python scripts/train_water_seg.py --inference-only --input your_image.tif")


if __name__ == "__main__":
    main()
