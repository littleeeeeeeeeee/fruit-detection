"""公共测试夹具：项目路径、固定测试图片、FastAPI TestClient。

夹具作用域：
- session：整个 pytest 进程只创建一次，避免重复读文件/重复启动 TestClient；
- TestClient 用 with 上下文，确保 FastAPI lifespan 正常进入/退出。
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

# tests/conftest.py -> tests -> 项目根。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 固定测试图放在 tests/fixtures/，不依赖 yolo_fruit_data 是否已 dvc pull。
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_IMAGE_PATH = FIXTURES_DIR / "sample.jpg"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回项目根目录，供需要拼接路径的测试使用。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """固定测试图（来自 yolo_fruit_data/images/val，含至少一个水果目标）。

    注意：文件名是 .jpg，但文件内容实际是 PNG 字节（历史 fixture）。
    PIL 按内容识别，所以仍能正常解码；API 只检查扩展名，因此也能通过。
    """
    assert SAMPLE_IMAGE_PATH.is_file(), f"缺少测试图片：{SAMPLE_IMAGE_PATH}"
    return SAMPLE_IMAGE_PATH


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path: Path) -> bytes:
    """一次性读取测试图字节，避免每个测试重复 IO。"""
    return sample_image_path.read_bytes()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI 测试客户端。

    使用 with 形式触发 startup/shutdown；当前 app 没有 lifespan 逻辑，
    但这样写对以后新增资源初始化更安全。
    """
    with TestClient(app) as test_client:
        yield test_client
