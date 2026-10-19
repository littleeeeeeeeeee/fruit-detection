# 实验记录（MLflow）

实验跟踪使用 MLflow，后端为 SQLite（`mlflow.db`），实验名 `fruit-detection`。

启动查看界面：

    mlflow ui --backend-store-uri sqlite:///mlflow.db

## 实验对比

| Run | 数据版本 | 模型 | epochs | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|---|---|---|
| baseline-v1 | v1（208 张，未清洗） | yolov8n | 50 | 0.913 | 0.657 | 基线，默认损失权重，SGD lr0=0.01 |
| improved-v2 | v2（166 张，已清洗） | yolov8n | 100（早停于 93） | **0.930** | **0.700** | 剔除 42 张低分辨率样本，训练更充分 |

> 表中指标是 **best checkpoint** 的值（baseline 为 epoch 36，improved-v2 为 epoch 73），
> 与 Ultralytics 按 fitness（主要看 mAP50-95）选出的 best.pt 对应。
> MLflow 的最终 `mAP50` 指标记录最后一个 epoch（improved-v2 为 0.923）：
> 单看 mAP50 的最大值会出现在 epoch 88（0.932），但该 epoch 的 mAP50-95 只有 0.674，
> 不是选中的 best checkpoint。

每个 run 记录的内容：

- **参数**：model / epochs / imgsz / batch / optimizer / lr0 / box / cls / dfl / patience / dataset_version
- **指标**：`train/box_loss`、`train/cls_loss`、`train/dfl_loss`、`metrics/precision`、`metrics/recall`、`metrics/mAP50`、`metrics/mAP50-95`（逐 epoch）
- **产物**：best.pt、last.pt、results.csv、args.yaml、results.png、confusion_matrix.png
- **标签**：dataset_version、git_commit、torch 版本、device

## 分类别指标

| 类别 | baseline-v1 mAP50 | improved-v2 mAP50 |
|---|---|---|
| apple | 0.974 | 0.967 |
| banana | 0.770 | 0.837 |
| orange | 0.995 | 0.988 |

## 优化指标及理由

主要优化指标：**mAP50** 和 **mAP50-95**。

- mAP50 是 IoU=0.5 时各类别 AP 的平均值，衡量“是否检测到正确类别和大致位置”，与页面展示的检测框场景最直接相关；
- mAP50-95 对 IoU 0.5–0.95 取平均，更严格地衡量定位精度，避免只追求“框住大概位置”；
- 本项目只有 3 类且验证集仅 32 张，单类指标容易抖动，因此用整体 mAP 作为主选型指标，同时单独看 banana（基线弱项）的类别指标；
- 不单独用 accuracy：目标检测是密集预测任务，没有简单的分类 accuracy，mAP 更合适。

## 结论：生产环境选用 improved-v2

理由：

1. **整体指标更高**：mAP50 从 0.913 提升到 0.930，mAP50-95 从 0.657 提升到 0.700，后者对定位精度的要求更严格，提升更能说明问题。
2. **补齐了短板类别**：banana 的 mAP50 从 0.770 提升到 0.837（+0.067）。基线中 banana 是明显弱项，苹果和橙子已接近饱和，因此 banana 的提升对整体可用性价值最大。
3. **代价可接受**：apple 与 orange 各下降约 0.007，属于验证集抖动范围（验证集仅 32 张，单张样本影响明显），不足以否定改进。
4. **训练过程可靠**：早停机制在第 73 轮保存最佳权重、第 93 轮停止，说明模型已收敛，不是欠训练或过训练的结果。

## 复现命令

    # baseline（v1 数据）
    python ml/train.py --mode train --epochs 50 --name baseline-v1 \
        --dataset-version v1 --optimizer SGD

    # improved（v2 数据）
    python ml/train.py --mode train --epochs 100 --name improved-v2 \
        --dataset-version v2