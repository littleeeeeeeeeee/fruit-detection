"""公共测试夹具：项目路径、固定测试图片、FastAPI TestClient。"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_IMAGE_PATH = FIXTURES_DIR / "sample.jpg"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """固定测试图（来自 yolo_fruit_data/images/val，含至少一个水果目标）。"""
    assert SAMPLE_IMAGE_PATH.is_file(), f"缺少测试图片：{SAMPLE_IMAGE_PATH}"
    return SAMPLE_IMAGE_PATH


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path: Path) -> bytes:
    return sample_image_path.read_bytes()


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
