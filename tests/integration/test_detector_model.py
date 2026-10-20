"""集成测试：用真实生产权重跑一次推理，校验结果结构与坐标范围。

和 unit 的区别：
- 这里会真的 ``YOLO(models/best.pt)`` 加载权重并推理，覆盖 ultralytics 集成；
- 因此比 unit 慢，但比 e2e 轻（不启动 uvicorn/浏览器）。

测试数据复用 ``sample_image_bytes``，它来自 tests/fixtures/sample.jpg。
"""
from pathlib import Path

import pytest

from app.config import CONF_THRESHOLD, MODEL_PATH
from app.detector import get_detector, load_image


@pytest.fixture(scope="module")
def detector():
    """模块级 detector：整个文件只加载一次模型。"""
    return get_detector()


@pytest.fixture(scope="module")
def detections(detector, sample_image_bytes):
    """对固定测试图跑一次推理，后续多个断言复用结果，避免重复推理。"""
    return detector.detect_bytes(sample_image_bytes)


def test_model_file_is_present():
    """生产权重必须存在；不存在时整个集成测试没有意义。"""
    assert Path(MODEL_PATH).is_file(), f"生产权重不存在：{MODEL_PATH}"


def test_model_is_loaded_once():
    """验证 lru_cache(maxsize=1)：重复调用返回同一个实例。

    这是性能约束：每个请求都重新 YOLO(...) 会慢几个数量级。
    """
    assert get_detector() is get_detector()


def test_detects_at_least_one_target(detections):
    """固定测试图至少有一个水果目标，且类别必须在三分类范围内。"""
    assert len(detections) >= 1
    assert all(d["class_name"] in {"apple", "banana", "orange"} for d in detections)


def test_detection_structure_is_valid(detections):
    """逐字段校验返回结构、类型、置信度范围和 conf 阈值。"""
    for detection in detections:
        assert set(detection) == {"class_name", "confidence", "box"}
        assert isinstance(detection["class_name"], str)
        assert isinstance(detection["confidence"], float)
        assert 0.0 <= detection["confidence"] <= 1.0

        # detector 内部按 conf 过滤；留 1e-6 容差防止浮点边界。
        assert detection["confidence"] >= CONF_THRESHOLD - 1e-6

        assert isinstance(detection["box"], list)
        assert len(detection["box"]) == 4
        assert all(isinstance(value, float) for value in detection["box"])


def test_boxes_are_inside_the_image(detections, sample_image_bytes):
    """框必须满足左上 < 右下，且大致落在图片范围内。

    YOLO 可能因 letterbox/坐标换算产生 1px 级越界，因此留 1.0 容差；
    这不是放松业务要求，而是承认推理后处理的数值误差。
    """
    width, height = load_image(sample_image_bytes).size

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        assert x1 < x2 and y1 < y2
        assert -1.0 <= x1 and -1.0 <= y1
        assert x2 <= width + 1.0 and y2 <= height + 1.0


def test_detect_image_and_detect_bytes_agree(detector, sample_image_bytes, detections):
    """两条入口（PIL 图 / bytes）对同一张图应给出一致结果。

    这保证 load_image 的转换不会悄悄改变像素/尺寸，导致推理结果不同。
    """
    image = load_image(sample_image_bytes)

    assert detector.detect_image(image) == detections
