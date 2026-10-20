"""FastAPI 应用入口。

路由分层：
- ``/`` 和 ``/static``：返回前端页面/静态资源；
- ``/api/health``：给 Docker healthcheck 和 e2e 启动探针用；
- ``/api/detect`` / ``/api/detect/batch``：图片检测 API。

错误码约定：
- 415 Unsupported Media Type：扩展名不在白名单；
- 413 Request Entity Too Large：文件超过 MAX_IMAGE_MB；
- 400 Bad Request：扩展名合法但字节不是有效图片；
- 422 Unprocessable Entity：FastAPI 自动校验失败（缺 file/files 字段）。
"""
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_EXTENSIONS, MAX_IMAGE_MB
from app.detector import InvalidImageError, get_detector
from app.schemas import BatchResult, ImageResult

# FastAPI 实例：title/version 会出现在 /docs 和 OpenAPI JSON 中。
app = FastAPI(title="Fruit Detection API", version="1.0.0")

# 静态目录：app/static；index.html 与 /static 下的资源都从这里取。
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """返回前端上传页面。

    ``include_in_schema=False`` 让这个页面不出现在 OpenAPI 路由列表里，
    因为它是 HTML 页面，不是 JSON API。
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """健康检查。

    Docker HEALTHCHECK、GitHub Actions smoke test、e2e 启动探针都依赖
    这个固定返回：``{"status": "ok"}``。不要在这里做模型推理或慢操作。
    """
    return {"status": "ok"}


async def _detect_one(file: UploadFile) -> ImageResult:
    """处理单张 UploadFile 的公共逻辑。

    被单图接口和批量接口复用，顺序是：
    1. 扩展名过滤（便宜，先做）；
    2. 读取全部 bytes 并检查大小；
    3. 调用 detector 做图片校验 + 推理；
    4. 把异常映射成 HTTP 状态码；
    5. 组装 ImageResult，并记录耗时。

    参数：
        file: FastAPI 解析出的上传文件。

    异常：
        HTTPException: 415（扩展名）、413（过大）、400（坏图）。
    """
    # 1) 扩展名过滤：Path("...").suffix 对小写/大写都统一成小写再比较。
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"不支持的格式：{suffix or '(无扩展名)'}")

    # 2) 读取完整内容。注意：这一步在大小检查之前，超大文件仍会进内存；
    #    当前限制 10MB 可接受，生产可改成流式读取/提前检查 Content-Length。
    data = await file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {MAX_IMAGE_MB:g} MB")

    # 3) 计时只包住推理/解码，不包含上传传输时间；
    #    perf_counter 比 time.time() 更适合测短间隔。
    started = time.perf_counter()
    try:
        detections = get_detector().detect_bytes(data)
    except InvalidImageError as exc:
        # 领域异常 -> HTTP 400；用 from exc 保留原始 traceback 链。
        raise HTTPException(400, str(exc)) from exc

    # 4) filename 可能为空（某些客户端），统一回退到 "unknown"。
    return ImageResult(
        filename=file.filename or "unknown",
        count=len(detections),
        detections=detections,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/api/detect", response_model=ImageResult)
async def detect(file: UploadFile = File(...)):
    """单图检测接口。

    请求：multipart/form-data，字段名必须为 ``file``。
    响应：``ImageResult``（filename/count/detections/elapsed_ms）。
    """
    return await _detect_one(file)


@app.post("/api/detect/batch", response_model=BatchResult)
async def detect_batch(files: list[UploadFile] = File(...)):
    """批量检测接口。

    请求：multipart/form-data，字段名必须为 ``files``，可重复。
    响应：``BatchResult``（total_images/total_detections/results）。

    实现说明：
    - 当前用列表推导逐张 ``await``，是串行实现；好处是简单、内存可控，
      CPU 推理下不会因为并发把机器打满；代价是总延迟随图片数线性增长。
    - 任何一张图失败都会让整个请求返回对应错误码（当前没有“部分成功”语义）。
    - results 顺序与上传顺序一致，前端依赖这一点展示文件名。
    """
    results = [await _detect_one(f) for f in files]
    return BatchResult(
        total_images=len(results),
        total_detections=sum(r.count for r in results),
        results=results,
    )
