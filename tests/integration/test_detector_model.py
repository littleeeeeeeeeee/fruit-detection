"""集成测试：用真实生产权重跑一次推理，校验结果结构与坐标范围。"""
from pathlib import Path

import pytest

from app.config import CONF_THRESHOLD, MODEL_PATH
from app.detector import get_detector, load_image


@pytest.fixture(scope="module")
def detector():
    return get_detector()


@pytest.fixture(scope="module")
def detections(detector, sample_image_bytes):
    return detector.detect_bytes(sample_image_bytes)


def test_model_file_is_present():
    assert Path(MODEL_PATH).is_file(), f"生产权重不存在：{MODEL_PATH}"


def test_model_is_loaded_once():
    # lru_cache(maxsize=1)：重复获取应返回同一个实例
    assert get_detector() is get_detector()


def test_detects_at_least_one_target(detections):
    assert len(detections) >= 1
    assert all(d["class_name"] in {"apple", "banana", "orange"} for d in detections)


def test_detection_structure_is_valid(detections):
    for detection in detections:
        assert set(detection) == {"class_name", "confidence", "box"}
        assert isinstance(detection["class_name"], str)
        assert isinstance(detection["confidence"], float)
        assert 0.0 <= detection["confidence"] <= 1.0
        assert detection["confidence"] >= CONF_THRESHOLD - 1e-6
        assert isinstance(detection["box"], list)
        assert len(detection["box"]) == 4
        assert all(isinstance(value, float) for value in detection["box"])


def test_boxes_are_inside_the_image(detections, sample_image_bytes):
    width, height = load_image(sample_image_bytes).size

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        assert x1 < x2 and y1 < y2
        assert -1.0 <= x1 and -1.0 <= y1
        assert x2 <= width + 1.0 and y2 <= height + 1.0


def test_detect_image_and_detect_bytes_agree(detector, sample_image_bytes, detections):
    image = load_image(sample_image_bytes)

    assert detector.detect_image(image) == detections
