import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from app.config import CONF_THRESHOLD, IOU_THRESHOLD, MODEL_PATH


class InvalidImageError(ValueError):
    """上传的内容无法解析为图片。"""


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImageError("文件为空")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("不是有效的图片文件") from exc


class FruitDetector:
    def __init__(self, model_path: str, conf: float, iou: float):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect_image(self, img: Image.Image) -> list[dict]:
        results = self.model.predict(
            source=img, conf=self.conf, iou=self.iou, verbose=False
        )
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                })
        return detections

    def detect_bytes(self, data: bytes) -> list[dict]:
        return self.detect_image(load_image(data))


@lru_cache(maxsize=1)
def get_detector() -> FruitDetector:
    return FruitDetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)