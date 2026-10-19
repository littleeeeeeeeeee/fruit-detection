"""YOLOv8 水果检测：训练 / 评估脚本（离线使用，不参与 Web 服务）。

训练（带 MLflow 记录）：
    python ml/train.py --mode train --epochs 100 --name baseline \
        --dataset-version v1 --optimizer SGD

只看训练、不记录：
    python ml/train.py --mode train --epochs 2 --name smoke --no-mlflow

评估现有权重：
    python ml/train.py --mode eval
"""
import argparse
import contextlib
import csv
import re
import shutil
import subprocess
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "yolo_fruit_data" / "data.yaml"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

CURVE_KEYS = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
FINAL_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


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
    p.add_argument("--optimizer", default="auto",
                   help="auto / SGD / Adam / AdamW；用 auto 时 lr0 会被忽略")
    p.add_argument("--box", type=float, default=7.5, help="box 损失权重")
    p.add_argument("--cls", type=float, default=0.5, help="分类损失权重")
    p.add_argument("--dfl", type=float, default=1.5, help="DFL 损失权重")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="baseline", help="本次 run 的目录名")
    p.add_argument("--device", default=None, help="0 表示 GPU，cpu 表示 CPU")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--promote", action="store_true",
                   help="训练后把 best.pt 复制到 models/best.pt")
    # MLflow
    p.add_argument("--no-mlflow", action="store_true", help="不记录到 MLflow")
    p.add_argument("--experiment", default="fruit-detection", help="MLflow 实验名")
    p.add_argument("--run-name", default=None, help="MLflow run 名，默认同 --name")
    p.add_argument("--dataset-version", default="v1", help="数据集版本标签，如 v1/v2")
    return p.parse_args()


def pick_device(arg):
    if arg:
        return arg
    return 0 if torch.cuda.is_available() else "cpu"


def git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"

def clean_metric_name(key):
    """MLflow 指标名不允许括号，去掉它们。"""
    return re.sub(r"[()]", "", key).strip()
def read_results_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{(k or "").strip(): v for k, v in row.items()} for row in rows]


def log_curves_and_artifacts(save_dir):
    import mlflow

    csv_path = save_dir / "results.csv"
    if csv_path.is_file():
        rows = read_results_csv(csv_path)
        for row in rows:
            try:
                step = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue
            for key in CURVE_KEYS:
                raw = row.get(key)
                if raw not in (None, ""):
                    try:
                        mlflow.log_metric(clean_metric_name(key), float(raw), step=step)
                    except ValueError:
                        pass
        if rows:
            last = rows[-1]
            for short, key in FINAL_KEYS.items():
                raw = last.get(key)
                if raw not in (None, ""):
                    try:
                        mlflow.log_metric(short, float(raw))
                    except ValueError:
                        pass

    for name in ("results.csv", "args.yaml", "results.png",
                 "confusion_matrix.png", "confusion_matrix_normalized.png"):
        f = save_dir / name
        if f.is_file():
            mlflow.log_artifact(str(f))

    for name in ("best.pt", "last.pt"):
        f = save_dir / "weights" / name
        if f.is_file():
            mlflow.log_artifact(str(f))


def run_train(args):
    device = pick_device(args.device)
    print(f"[train] device={device}")
    print(f"[train] data={args.data}")
    print(f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"optimizer={args.optimizer} lr0={args.lr0} "
          f"box={args.box} cls={args.cls} dfl={args.dfl}")

    with contextlib.ExitStack() as stack:
        if not args.no_mlflow:
            import mlflow
            from ultralytics import settings as ul_settings

            ul_settings.update({"mlflow": False})   # 关掉 ultralytics 自带的记录，避免重复
            mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
            mlflow.set_experiment(args.experiment)
            stack.enter_context(mlflow.start_run(run_name=args.run_name or args.name))

            mlflow.log_params({
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "box": args.box,
                "cls": args.cls,
                "dfl": args.dfl,
                "patience": args.patience,
                "dataset_version": args.dataset_version,
            })
            mlflow.set_tags({
                "dataset_version": args.dataset_version,
                "git_commit": git_sha(),
                "torch": torch.__version__,
                "device": str(device),
            })
            print(f"[train] mlflow experiment={args.experiment} "
                  f"run={args.run_name or args.name} dataset={args.dataset_version}")

        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            optimizer=args.optimizer,
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

        if not args.no_mlflow:
            log_curves_and_artifacts(save_dir)
            print("[train] 已写入 MLflow")

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