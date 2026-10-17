import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_EXTENSIONS, MAX_IMAGE_MB
from app.detector import InvalidImageError, get_detector
from app.schemas import BatchResult, ImageResult

app = FastAPI(title="Fruit Detection API", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


async def _detect_one(file: UploadFile) -> ImageResult:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"不支持的格式：{suffix or '(无扩展名)'}")

    data = await file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {MAX_IMAGE_MB:g} MB")

    started = time.perf_counter()
    try:
        detections = get_detector().detect_bytes(data)
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc

    return ImageResult(
        filename=file.filename or "unknown",
        count=len(detections),
        detections=detections,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/api/detect", response_model=ImageResult)
async def detect(file: UploadFile = File(...)):
    return await _detect_one(file)


@app.post("/api/detect/batch", response_model=BatchResult)
async def detect_batch(files: list[UploadFile] = File(...)):
    results = [await _detect_one(f) for f in files]
    return BatchResult(
        total_images=len(results),
        total_detections=sum(r.count for r in results),
        results=results,
    )