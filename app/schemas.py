"""API 响应模型（Pydantic v2）。

这些模型负责：
- 生成 OpenAPI schema（/docs 里能看到字段说明）；
- 在返回前校验类型/结构，防止推理层返回意外字段；
- 统一单图与批量接口的 JSON 形状。

模型本身不包含业务逻辑；字段名一旦改动，前端 ``app/static/index.html``
和 API 测试都需要同步修改。
"""
from pydantic import BaseModel


class Detection(BaseModel):
    """单个检测框。

    Attributes:
        class_name: 类别名，来自模型 names，当前为 apple/banana/orange。
        confidence: 置信度，范围 [0, 1]，detector 已四舍五入到 4 位。
        box: [x1, y1, x2, y2] 绝对像素坐标，左上/右下。
    """

    class_name: str
    confidence: float
    box: list[float]


class ImageResult(BaseModel):
    """单张图片的检测结果。

    Attributes:
        filename: 上传时的原始文件名（可能包含用户输入，前端只做文本展示）。
        count: 检测框数量，等于 ``len(detections)``。
        detections: 检测框列表；无目标时为空列表。
        elapsed_ms: 服务端从开始推理到返回的耗时（毫秒，整数）。
    """

    filename: str
    count: int
    detections: list[Detection]
    elapsed_ms: int


class BatchResult(BaseModel):
    """批量检测结果。

    Attributes:
        total_images: 本次处理的图片数，等于 results 长度。
        total_detections: 所有图片的检测框总数。
        results: 每张图片的 ``ImageResult``，顺序与上传顺序一致。

    说明：当前 batch 是串行逐张推理，因此不会出现部分结果先返回的情况；
    如果未来改成并发，需要保持 results 与输入顺序的对应关系。
    """

    total_images: int
    total_detections: int
    results: list[ImageResult]
