# 水果检测（YOLOv8 + FastAPI + MLOps）

基于 YOLOv8 的水果目标检测应用：上传图片，返回 apple / banana / orange 的检测框、类别和置信度。
项目同时包含数据版本管理（DVC）、实验跟踪（MLflow）、Docker 镜像和 GitHub Actions CI/CD。

## 功能

- 单张图片检测：`POST /api/detect`
- 批量图片检测：`POST /api/detect/batch`
- 健康检查：`GET /api/health`
- 浏览器上传页面：`GET /`
- three-class YOLOv8n 权重：`models/best.pt`

## 技术栈

| 层 | 技术 |
| --- | --- |
| Web | FastAPI + Uvicorn |
| 模型 | Ultralytics YOLOv8n |
| 推理 | PyTorch（CPU 版镜像） |
| 测试 | pytest + httpx + Playwright |
| 数据 | DVC + Azure Blob Storage |
| 实验 | MLflow（SQLite backend） |
| 容器 | Docker / Docker Hub |
| CI/CD | GitHub Actions |

## 快速开始

要求 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000> 上传图片。

Docker：

```bash
docker build -t fruit-detection:local .
docker run --rm -p 127.0.0.1:8000:8000 fruit-detection:local
```

Docker Hub：`snakeskinbag/fruit-detection:latest`

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/unit tests/integration tests/api tests/e2e -v
```

当前测试：**23 passed**

| 层 | 文件 | 内容 |
| --- | --- | --- |
| unit | `tests/unit/test_image_loading.py` | 空内容、非图片、伪装图片、RGBA/P 模式转 RGB |
| integration | `tests/integration/test_detector_model.py` | 真实权重推理、结构、置信度、坐标边界 |
| API | `tests/api/test_detect_api.py` | 健康检查、首页、单图、415、400、422、批量 |
| e2e | `tests/e2e/test_upload.py` | Playwright 真实浏览器上传，断言检测结果和检测框 |

## 模型与数据

- 生产模型：`models/best.pt`（YOLOv8n，来自 `improved-v2`）
- 最佳 checkpoint 指标：mAP50 = **0.930**，mAP50-95 = **0.700**（improved-v2，epoch 73）
- 数据版本：`data-v1`（208 张训练图）→ `data-v2`（166 张，剔除 42 张低分辨率样本）
- 数据由 DVC 管理，Git 只保存指针文件
- 训练/评估脚本：`ml/train.py`
- 数据说明：`ml/README_data.md`
- 实验说明：`ml/README_experiments.md`
- 数据来源与格式转换：`ml/PROVENANCE.md`
- 当前生产模型登记：`ml/registry/current_model.md`

## 分支策略

```text
feature/* → dev → staging → main
```

- 功能开发在 `feature/*`
- 通过 PR 合并到 `dev`
- `dev` 通过 PR 合并到 `staging`
- `staging` 通过 PR 合并到 `main`
- 合并使用 merge commit，保留 PR 记录

已完成的 4.5 合并演练：

| 步骤 | PR |
| --- | --- |
| feature → dev | <https://github.com/littleeeeeeeeeee/fruit-detection/pull/1> |
| dev → staging | <https://github.com/littleeeeeeeeeee/fruit-detection/pull/2> |
| staging → main | <https://github.com/littleeeeeeeeeee/fruit-detection/pull/3> |

## CI/CD：commit → build → test → staging → prod

工作流：`.github/workflows/tests.yml`

| 触发 | 运行内容 |
| --- | --- |
| PR 到 `main` / `dev` / `staging` | lint + unit + integration + API + e2e + Docker build（no push） |
| push 到 `dev` / `feature/**` | 同上（用于提前发现集成问题） |
| push 到 `staging` | tests + Docker build + 生成 `staging.env` + 构建/推送 `:staging` + smoke test |
| push 到 `main` | tests + Docker build + 推送 `:latest` / `:<sha>` + 生成 `production.env` + 构建/推送 `:production` + smoke test |

流程：

1. 开发者在 `feature/*` 提交代码并开 PR。
2. CI 跑 lint、四组测试，并构建 Docker 镜像。
3. 合并到 `dev` 后再次验证。
4. 合并到 `staging`：用 staging environment secrets 生成 `staging.env`，构建并 smoke test staging 镜像。
5. 合并到 `main`：推送 `:latest` 与 SHA tag，并用 production secrets 构建和 smoke test 生产镜像。

## 仓库配置

Repository secrets：

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Environments：

- `staging`：`STAGING_MODEL_PATH`、`STAGING_CONF_THRESHOLD`、`STAGING_IOU_THRESHOLD`、`STAGING_MAX_IMAGE_MB`
- `production`：`PRODUCTION_MODEL_PATH`、`PRODUCTION_CONF_THRESHOLD`、`PRODUCTION_IOU_THRESHOLD`、`PRODUCTION_MAX_IMAGE_MB`

这些部署配置以 **environment variables** 形式保存（属于非敏感配置），workflow 用 `vars.*` 读取；
Docker Hub 凭据则放在 repository secrets 中，workflow 用 `secrets.*` 读取。

## 仓库结构

```text
.
├── app/                         # FastAPI 服务
├── ml/                          # 训练、清洗、数据/实验文档、模型登记
├── models/best.pt               # 生产权重
├── tests/                       # unit / integration / API / e2e
├── yolo_fruit_data/             # DVC 指针 + data.yaml（原始数据不提交）
├── .github/workflows/tests.yml  # CI/CD
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── DEPLOYMENT.md
```

## 部署

见 [DEPLOYMENT.md](DEPLOYMENT.md)。当前 pipeline 构建并验证 staging/production 镜像；项目要求不包含真实服务器发布流程。

## 已知限制

- 验证集仅 32 张，单张样本对指标影响较大；生产选型解释见 `ml/README_experiments.md`。
- 原始 `fruit_data/` 不进入 Git；DVC 需要 `.dvc/config.local` 中的 Azure SAS token 才能 `dvc pull`。
