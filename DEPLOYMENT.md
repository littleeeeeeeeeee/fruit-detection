# 部署说明（staging / production）

本项目使用 GitHub Actions 完成 CI/CD。部署阶段只负责：

1. 从 GitHub Environment secrets 生成对应环境的 `.env`；
2. 构建并推送环境镜像；
3. 启动容器做 `/api/health` 和 `/api/detect` smoke test。

项目要求明确说明 **不需要真实 release/deployment 流程**，因此没有服务器地址、SSH 或云平台部署步骤。

## 1. 触发路径

| 事件 | 分支 | 作业 |
| --- | --- | --- |
| Pull request | `main` / `dev` / `staging` | lint + 四组测试 + Docker build（不推送） |
| Push | `dev` / `feature/**` | lint + 四组测试 + Docker build |
| Push | `staging` | tests + Docker build + deploy-staging |
| Push | `main` | tests + Docker build + publish + deploy-production |

合并路径：

```text
feature/* → dev → staging → main
```

## 2. GitHub 配置

### Repository secrets

| Secret | 用途 |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub 用户名 |
| `DOCKERHUB_TOKEN` | Docker Hub PAT |

### Environments

`staging`：

- `STAGING_MODEL_PATH`（默认 `/app/models/best.pt`）
- `STAGING_CONF_THRESHOLD`
- `STAGING_IOU_THRESHOLD`
- `STAGING_MAX_IMAGE_MB`

`production`：

- `PRODUCTION_MODEL_PATH`
- `PRODUCTION_CONF_THRESHOLD`
- `PRODUCTION_IOU_THRESHOLD`
- `PRODUCTION_MAX_IMAGE_MB`

这些都是 **environment secrets**，workflow 用 `secrets.STAGING_*` / `secrets.PRODUCTION_*` 读取。
`staging.env` / `production.env` 只在 runner 中生成，不提交仓库（`.gitignore` 已忽略）。

建议给 `production` environment 配置 required reviewers；是否要求人工审批由仓库 owner 在 GitHub Settings 中开启。

## 3. staging 流程

```text
push staging
  └─ tests
      └─ deploy-staging
          ├─ 生成 staging.env
          ├─ 构建镜像：<IMAGE>:staging / :staging-<sha>
          ├─ 推送 Docker Hub
          └─ docker run + /api/health + /api/detect smoke test
```

生成的 `staging.env` 结构：

```dotenv
APP_ENV=staging
MODEL_PATH=/app/models/best.pt
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.7
MAX_IMAGE_MB=10
```

实际值来自 `staging` environment secrets。

## 4. production 流程

```text
push main
  ├─ tests
  ├─ publish
  │   ├─ 登录 Docker Hub
  │   └─ 推送 <IMAGE>:latest / :<sha>
  └─ deploy-production（needs: tests + publish）
      ├─ 生成 production.env
      ├─ 构建镜像：<IMAGE>:production / :production-<sha>
      ├─ 推送 Docker Hub
      └─ docker run + smoke test
```

镜像仓库：`snakeskinbag/fruit-detection`

## 5. 本地复现部署步骤

```bash
docker build -t fruit-detection:local .

cat > staging.env <<'EOF'
APP_ENV=staging
MODEL_PATH=/app/models/best.pt
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.7
MAX_IMAGE_MB=10
EOF

docker run -d --name fruit-staging --env-file staging.env -p 127.0.0.1:8000:8000 fruit-detection:local
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS -F "file=@tests/fixtures/sample.jpg" http://127.0.0.1:8000/api/detect
docker rm -f fruit-staging
```

## 6. 回滚

镜像同时推送 SHA tag：

- `<IMAGE>:<git-sha>`
- `<IMAGE>:staging-<git-sha>`
- `<IMAGE>:production-<git-sha>`

回滚时把部署使用的 tag 指回上一个已验证的 SHA tag 即可。由于当前没有真实主机部署，这属于镜像层回滚。

## 7. 验证记录

- 4.5 的 staging 部署：GitHub Actions run `39186600362`（attempt 2）
- 4.5 的 production / publish：GitHub Actions run `39187082587`
- Docker Hub tags：`latest`、`staging`、`production` 及对应 SHA tag
- 本地工作记录：`work_of_zkq/4.5_COMPLETION.md`
