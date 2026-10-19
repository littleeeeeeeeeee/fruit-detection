# 当前生产模型登记

- 更新时间：2026-10-19
- 模型文件：`models/best.pt`
- 文件大小：6,256,490 bytes
- SHA256：`18C8507777347AFAEF8592159EBA620A0CBFA3F97CEA420AD3449377A2E15768`
- 架构：YOLOv8n（Ultralytics）
- 任务：三分类目标检测（apple / banana / orange）

## 对应实验

| 项 | 值 |
| --- | --- |
| MLflow run | `improved-v2` |
| run_id | `cc9c5339934f4e4c8b83b6927d49fbfd` |
| 数据版本 | `v2` |
| DVC tag | `data-v2` |
| 数据 commit | `bad7453` |
| 训练代码 commit | `bad7453` |
| 最佳 epoch | 73（共 93 轮早停，按 fitness 选 best.pt） |
| mAP50 | 0.93042（文档写作 0.930） |
| mAP50-95 | 0.69957（文档写作 0.700） |
| precision | 0.88474 |
| recall | 0.89641 |

训练参数：

```text
model=yolov8n.pt
epochs=100
imgsz=640
batch=16
optimizer=auto
lr0=0.01
box=7.5
cls=0.5
dfl=1.5
patience=20
dataset_version=v2
```

## 权重来源

- MLflow artifact：`mlruns/1/cc9c5339934f4e4c8b83b6927d49fbfd/artifacts/best.pt`
- 该 artifact 与 `models/best.pt` 的 SHA256 完全一致；
- 权重在 commit `0177703`（"Add experiment records and dataset documentation; promote improved-v2 model"）中正式提升到 `models/best.pt`。

## 与 baseline 对比

| Run | 数据 | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| `baseline-v1` | v1（208 训练图） | 0.91321 | 0.65717 |
| `improved-v2` | v2（166 训练图） | 0.93042 | 0.69957 |

`improved-v2` 被选为生产模型：整体 mAP 更高，且 banana 类别提升明显；详见
`ml/README_experiments.md`。

## 容器镜像

Docker Hub：`snakeskinbag/fruit-detection`

| Tag | 用途 |
| --- | --- |
| `latest` / `<sha>` | main 生产发布 |
| `production` / `production-<sha>` | production 阶段构建 |
| `staging` / `staging-<sha>` | staging 阶段构建 |

`models/best.pt` 会通过 Dockerfile 的 `COPY models/best.pt ./models/best.pt` 进入镜像，
运行时 `MODEL_PATH` 默认 `/app/models/best.pt`。

## 验证命令

```bash
# 校验权重
sha256sum models/best.pt
# 期望：18c8507777347afaef8592159eba620a0cbfa3f97cea420ad3449377a2e15768

# 看 MLflow run
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 打开 run cc9c5339934f4e4c8b83b6927d49fbfd

# 重新评估
python ml/train.py --mode eval
```

## 回滚

如果新模型回退，使用上一个通过评估的 `models/best.pt` 版本（Git 历史）并重建镜像；
Docker 镜像可使用对应 SHA tag 回滚。
