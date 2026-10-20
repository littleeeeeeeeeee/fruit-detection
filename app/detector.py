"""YOLO 推理封装。

职责边界：
- 本模块不处理 HTTP、不关心上传字段名，只接收 bytes/PIL.Image；
- 负责图片内容校验、RGB 归一化、YOLO 推理和结果 JSON 化；
- FastAPI 层通过 ``get_detector()`` 拿到进程内单例，避免每个请求都加载权重。

典型调用链：
    UploadFile -> app.main._detect_one -> get_detector().detect_bytes
               -> load_image -> YOLO.predict -> list[dict]
"""
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from app.config import CONF_THRESHOLD, IOU_THRESHOLD, MODEL_PATH


class InvalidImageError(ValueError):
    """上传内容不是可解析图片时抛出的领域异常。

    继承 ValueError 而不是自定义 Exception，是为了让调用方可以用
    “值不合法”这一语义捕获；API 层会把它映射为 HTTP 400。
    """


def load_image(data: bytes) -> Image.Image:
    """把上传的原始字节解析成 RGB 的 PIL.Image。

    参数：
        data: 上传文件的完整字节内容。

    返回：
        已经转换为 RGB 的 PIL.Image；尺寸保持原图。

    异常：
        InvalidImageError: 空字节、不是图片、图片损坏/截断时抛出。

    实现细节：
    1. PIL 的 ``verify()`` 会检查文件头和完整性，但调用后图像对象不能继续用，
       所以这里故意打开两次：第一次 verify，第二次真正加载；
    2. 统一 ``convert("RGB")``，避免 RGBA/P/灰度图进入 YOLO 时通道数不一致；
    3. ``UnidentifiedImageError`` 是 PIL 对“不认识这个格式”的专门异常，
       ``OSError`` 覆盖截断文件等底层读取错误。
    """
    if not data:
        raise InvalidImageError("文件为空")

    try:
        # 第一次打开只做完整性校验；verify() 之后对象不可再用于 load。
        img = Image.open(io.BytesIO(data))
        img.verify()

        # 第二次重新打开，真正解码像素并统一成 RGB。
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("不是有效的图片文件") from exc


class FruitDetector:
    """加载一次 YOLO 权重，重复执行图片检测。

    这个类不做缓存/并发控制；单例和线程安全由 ``get_detector`` 的
    ``lru_cache`` 以及 YOLO 自身的推理实现负责。每个进程只应创建一个实例，
    否则会重复占用模型内存。
    """

    def __init__(self, model_path: str, conf: float, iou: float):
        """加载权重并保存推理阈值。

        参数：
            model_path: best.pt 的路径；不存在时立即失败，避免请求时才报错。
            conf: 置信度阈值，传给 ``model.predict``。
            iou: NMS 的 IoU 阈值，传给 ``model.predict``。

        异常：
            FileNotFoundError: 权重路径不存在。
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        # Ultralytics 的 YOLO 对象同时包含模型结构和权重；
        # 这里加载一次，后续 predict 复用。
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect_image(self, img: Image.Image) -> list[dict]:
        """对一张 PIL 图片执行推理，返回 JSON 友好的检测列表。

        返回的每个元素：
            {
                "class_name": "apple",
                "confidence": 0.9866,       # 保留 4 位小数
                "box": [x1, y1, x2, y2],    # 绝对像素坐标，保留 2 位小数
            }

        说明：
        - ``verbose=False`` 关闭 Ultralytics 的逐图日志，避免 API 日志被刷屏；
        - ``results`` 是 batch 维度列表，单图时通常只有一个元素；
        - ``box.cls``/``box.conf``/``box.xyxy`` 都是 tensor，需要取 ``[0]``
          再转 Python 标量；
        - 坐标保持绝对像素，由前端根据图片 naturalWidth/naturalHeight 转百分比。
        """
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
        """字节入口：先做图片校验/解码，再调用 ``detect_image``。

        参数：
            data: 上传文件的原始字节。

        异常：
            InvalidImageError: 内容不是有效图片。
        """
        return self.detect_image(load_image(data))


@lru_cache(maxsize=1)
def get_detector() -> FruitDetector:
    """返回进程内唯一的 ``FruitDetector``。

    ``lru_cache(maxsize=1)`` 让模型只加载一次：
    - 第一次请求会付模型加载成本（CPU 上约 0.8–1.2 秒）；
    - 之后的请求直接复用，单图推理约十几毫秒；
    - 代价是更新 models/best.pt 后必须重启进程/容器才能生效。

    如果模型文件缺失，第一次调用会抛 FileNotFoundError；FastAPI 启动时
    不会主动调用这个函数，所以错误会在首个检测请求时暴露。
    """
    return FruitDetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
