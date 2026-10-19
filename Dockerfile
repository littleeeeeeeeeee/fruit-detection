# 基础镜像可通过构建参数覆盖，例如：
#   docker build --build-arg BASE_IMAGE=python:3.12-slim -t fruit-detection:local .
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="fruit-detection" \
      org.opencontainers.image.description="YOLOv8 fruit detection API (apple / banana / orange)" \
      org.opencontainers.image.source="https://github.com/littleeeeeeeeeee/fruit-detection"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/best.pt \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# OpenCV (ultralytics 依赖) 在 slim 镜像里需要 libGL/libglib；libgomp 供 torch 使用
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖再拷代码，充分利用 Docker 层缓存
COPY requirements.txt ./
# 推理只需要 CPU 版 torch，避免把体积巨大的 CUDA 依赖打进镜像
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/best.pt ./models/best.pt

# 非 root 运行：固定 UID/GID 10001
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
