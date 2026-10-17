"""YOLOv8 水果检测：训练与评估脚本（离线使用，不参与 Web 服务）。

用法：
    训练：python ml/train.py --mode train --epochs 100 --name baseline
    评估：python ml/train.py --mode eval
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "yolo_fruit_data" / "data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="训练 / 评估 YOLOv8 水果检测模型")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="data.yaml 路径")
    p.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "best.pt"))
    p.add_argument("--model", default="yolov8n.pt", help="初始权重（训练用）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--box", type=float, default=7.0, help="box 损失权重")
    p.add_argument("--cls", type=float, default=0.5, help="分类损失权重")
    p.add_argument("--dfl", type=float, default=1.5, help="DFL 损失权重")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="baseline", help="本次 run 的名字")
    p.add_argument("--device", default=None, help="0 表示 GPU，cpu 表示 CPU")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--promote", action="store_true",
                   help="训练结束后把 best.pt 复制到 models/best.pt")
    return p.parse_args()


def pick_device(arg: str | None):
    if arg:
        return arg
    return 0 if torch.cuda.is_available() else "cpu"


def run_train(args) -> Path:
    device = pick_device(args.device)
    print(f"[train] device={device}")
    print(f"[train] data={args.data}")
    print(f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"lr0={args.lr0} box={args.box} cls={args.cls} dfl={args.dfl}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=device,
        amp=False,
        save_period=10,
        patience=args.patience,
        workers=args.workers,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"[train] 本次 run 目录：{save_dir}")
    print(f"[train] 最佳权重：{best}")

    if args.promote and best.is_file():
        target = PROJECT_ROOT / "models" / "best.pt"
        shutil.copy2(best, target)
        print(f"[train] 已复制到 {target}")

    return best


def run_eval(args):
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"权重不存在：{weights}")

    device = pick_device(args.device)
    print(f"[eval] weights={weights} device={device}")
    model = YOLO(str(weights))
    metrics = model.val(data=args.data, device=device)

    print("=" * 40)
    print(f"mAP50     = {metrics.box.map50:.4f}")
    print(f"mAP50-95  = {metrics.box.map:.4f}")
    print("=" * 40)
    return metrics


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()