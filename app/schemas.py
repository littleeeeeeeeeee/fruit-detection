from pydantic import BaseModel


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: list[float]


class ImageResult(BaseModel):
    filename: str
    count: int
    detections: list[Detection]
    elapsed_ms: int


class BatchResult(BaseModel):
    total_images: int
    total_detections: int
    results: list[ImageResult]