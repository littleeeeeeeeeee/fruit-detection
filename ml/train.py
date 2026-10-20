"""YOLOv8 水果检测：训练 / 评估脚本（离线使用，不参与 Web 服务）。

这个脚本把“训练”和“实验记录”放在一起：
1. 解析命令行参数；
2. 可选地打开 MLflow run，记录 params/tags；
3. 调 Ultralytics 训练；
4. 从 ``results.csv`` 回放逐 epoch 指标并上传 artifacts；
5. 可选地把 best.pt 提升到 ``models/best.pt``。

用法示例：
    训练（带 MLflow 记录）：
        python ml/train.py --mode train --epochs 100 --name baseline \
            --dataset-version v1 --optimizer SGD

    只看训练、不记录：
        python ml/train.py --mode train --epochs 2 --name smoke --no-mlflow

    评估现有权重：
        python ml/train.py --mode eval

设计说明：
- 训练数据由 Git tag + DVC 决定，``--dataset-version`` 只写 MLflow tag，
  不会自动切换数据；这是最容易误解的地方。
- MLflow backend 固定为项目根目录的 ``mlflow.db``（SQLite），artifact 默认
  落在 ``mlruns/``；两者都被 .gitignore 排除，提交时要单独附带。
- 关闭 Ultralytics 自带的 MLflow 集成（``ul_settings.update({"mlflow": False})``），
  避免和本脚本的记录重复。
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

# 路径都以脚本位置为基准，保证从任意工作目录执行都能找到项目根。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "yolo_fruit_data" / "data.yaml"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# 逐 epoch 记录的曲线指标（MLflow 里会去掉括号，见 clean_metric_name）。
CURVE_KEYS = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)

# 最后一个 epoch 的汇总指标；MLflow 里用短名，便于 UI 对比。
FINAL_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def parse_args():
    """定义并解析命令行参数。

    返回：
        argparse.Namespace，字段与下面的 add_argument 一一对应。

    关键参数：
        --mode train/eval：训练或只评估。
        --data：data.yaml 路径；默认仓库内 yolo_fruit_data/data.yaml。
        --weights：eval 使用的权重；默认 models/best.pt。
        --model：train 的初始权重；默认 yolov8n.pt（Ultralytics 会自动下载）。
        --optimizer=auto 时 Ultralytics 会忽略 lr0，这是上游行为。
        --promote：训练结束后把本次 best.pt 复制到 models/best.pt。
        --no-mlflow：完全跳过 MLflow（用于 smoke test/离线）。
        --dataset-version：只写 MLflow tag，不切换数据。
    """
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
    # MLflow 相关参数
    p.add_argument("--no-mlflow", action="store_true", help="不记录到 MLflow")
    p.add_argument("--experiment", default="fruit-detection", help="MLflow 实验名")
    p.add_argument("--run-name", default=None, help="MLflow run 名，默认同 --name")
    p.add_argument("--dataset-version", default="v1", help="数据集版本标签，如 v1/v2")
    return p.parse_args()


def pick_device(arg):
    """选择训练/评估设备。

    参数：
        arg: 命令行传入的 --device；None 表示自动选择。

    返回：
        arg 原样返回（例如 "cpu" 或 "0"）；否则有 CUDA 用 0，没有就用 "cpu"。
    """
    if arg:
        return arg
    return 0 if torch.cuda.is_available() else "cpu"


def git_sha():
    """返回当前 Git commit 的短 SHA；不在 Git 仓库/命令失败时返回 "unknown"。

    这个值会写进 MLflow tag ``git_commit``，用于从实验反查代码版本。
    ``stderr=DEVNULL`` 避免在非 Git 目录刷错误信息。
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def clean_metric_name(key):
    """把 Ultralytics 指标名转成 MLflow 可接受的名称。

    MLflow 的 metric key 不允许括号，所以 ``metrics/mAP50(B)`` 会变成
    ``metrics/mAP50B``；斜杠是允许的，保留它便于分组。
    """
    return re.sub(r"[()]", "", key).strip()


def read_results_csv(path):
    """读取 Ultralytics 的 results.csv。

    参数：
        path: results.csv 的 Path。

    返回：
        list[dict]；每个 dict 的 key 去掉首尾空白（Ultralytics 某些版本
        会在列名前后带空格），方便后面按固定 key 取值。

    注意：
        这里不校验列是否存在；调用方用 ``row.get(key)`` 做容错。
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{(k or "").strip(): v for k, v in row.items()} for row in rows]


def log_curves_and_artifacts(save_dir):
    """把一个训练目录里的指标/文件写入当前 MLflow run。

    参数：
        save_dir: Ultralytics 本次 run 的输出目录（含 results.csv、weights/ 等）。

    记录内容：
    - 逐 epoch 曲线：box/cls/dfl loss、precision、recall、mAP50、mAP50-95；
    - 最后一行汇总指标：precision/recall/mAP50/mAP50-95；
    - artifacts：results.csv、args.yaml、results.png、混淆矩阵、best.pt/last.pt。

    容错：
    - CSV 不存在时只上传 artifacts（不报错）；
    - 某行缺少 epoch/指标时跳过该值；
    - 非数值字符串（空、nan 文本）用 ValueError 捕获后忽略。
    """
    import mlflow

    csv_path = save_dir / "results.csv"
    if csv_path.is_file():
        rows = read_results_csv(csv_path)
        for row in rows:
            try:
                step = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue

            # 逐 epoch 曲线：step 用 epoch，方便 MLflow UI 画横轴。
            for key in CURVE_KEYS:
                raw = row.get(key)
                if raw not in (None, ""):
                    try:
                        mlflow.log_metric(clean_metric_name(key), float(raw), step=step)
                    except ValueError:
                        pass

        # 最后一行是最后一个 epoch 的汇总；与 best checkpoint 不同，
        # 这里记录的是 final metrics，用于看收敛终点。
        if rows:
            last = rows[-1]
            for short, key in FINAL_KEYS.items():
                raw = last.get(key)
                if raw not in (None, ""):
                    try:
                        mlflow.log_metric(short, float(raw))
                    except ValueError:
                        pass

    # 训练曲线/配置/混淆矩阵：体积小，直接作为 artifact 上传。
    for name in ("results.csv", "args.yaml", "results.png",
                 "confusion_matrix.png", "confusion_matrix_normalized.png"):
        f = save_dir / name
        if f.is_file():
            mlflow.log_artifact(str(f))

    # 权重文件较大（6MB 级别），但这是复现实验最关键的文件；
    # best.pt 也是 models/best.pt 的来源。
    for name in ("best.pt", "last.pt"):
        f = save_dir / "weights" / name
        if f.is_file():
            mlflow.log_artifact(str(f))


def run_train(args):
    """执行训练，并按需写 MLflow/提升权重。

    流程：
    1. 选设备并打印关键参数；
    2. 如果启用 MLflow：设置 tracking URI/experiment，开 run，记录 params/tags；
    3. 调 ``YOLO.train``；
    4. 训练后回放 results.csv、上传 artifacts；
    5. 如果 ``--promote``，把 best.pt 复制到 models/best.pt。

    返回：
        本次 run 的 best.pt 路径（Path）。

    注意：
    - ``contextlib.ExitStack`` 保证异常时 MLflow run 也会结束；
    - ``amp=False`` 是为了 CPU/兼容性稳定；GPU 上可另开；
    - ``save_period=10`` 每 10 epoch 存一个 checkpoint，便于中断恢复。
    """
    device = pick_device(args.device)
    print(f"[train] device={device}")
    print(f"[train] data={args.data}")
    print(f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"optimizer={args.optimizer} lr0={args.lr0} "
          f"box={args.box} cls={args.cls} dfl={args.dfl}")

    # ExitStack 用来统一管理可选的 MLflow run 上下文：
    # 不开 MLflow 时什么都不压栈，代码路径保持一致。
    with contextlib.ExitStack() as stack:
        if not args.no_mlflow:
            import mlflow
            from ultralytics import settings as ul_settings

            # 关闭 Ultralytics 自带的 MLflow 记录，避免同一次训练出现两套 run。
            ul_settings.update({"mlflow": False})

            # SQLite backend；as_posix() 避免 Windows 反斜杠在 URI 里转义。
            mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
            mlflow.set_experiment(args.experiment)
            stack.enter_context(mlflow.start_run(run_name=args.run_name or args.name))

            # 超参数：全部用标量，方便 MLflow 做对比/排序。
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

            # tags 用于反查代码/数据/环境；dataset_version 只是标签，
            # 真正数据版本由 Git tag + DVC 决定。
            mlflow.set_tags({
                "dataset_version": args.dataset_version,
                "git_commit": git_sha(),
                "torch": torch.__version__,
                "device": str(device),
            })
            print(f"[train] mlflow experiment={args.experiment} "
                  f"run={args.run_name or args.name} dataset={args.dataset_version}")

        # YOLO(args.model) 会加载初始权重；默认 yolov8n.pt 可能触发自动下载。
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
    """用指定权重在 data.yaml 的 val 划分上评估。

    参数：
        args: 解析后的参数；使用 ``--weights`` 和 ``--data``。

    返回：
        Ultralytics 的 metrics 对象（含 box.map50/box.map 等）。

    注意：
    - 这里评估的是 data.yaml 的 val，不是 test；test 没有参与选型；
    - ``model.val`` 会重新扫描 labels 并可能生成 cache；
    - CPU 上 32 张 val 约几秒。
    """
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
    """脚本入口：按 --mode 分派到训练或评估。"""
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
