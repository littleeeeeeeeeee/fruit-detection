"""E2E 公共夹具：启动真实 uvicorn 服务，供 Playwright 使用。

为什么不直接用 TestClient：
- e2e 要验证浏览器 -> HTTP -> FastAPI -> 静态页/JS 的完整链路；
- TestClient 走 ASGI 内存调用，不会启动真实端口，也无法给 Playwright 一个 URL。

生命周期：
1. 找空闲端口；
2. subprocess 启动 ``python -m uvicorn app.main:app``；
3. 轮询 /api/health，最多 60 秒；
4. yield base_url；
5. 测试结束 terminate，超时再 kill。
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# tests/e2e/conftest.py -> tests/e2e -> tests -> 项目根。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    """让操作系统分配一个空闲 TCP 端口。

    注意：bind 后立即关闭，存在“关闭到 uvicorn 真正 bind”之间的竞争窗口；
    在单机测试中概率很低。若未来并行跑多个 e2e worker，建议加重试。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def live_server() -> str:
    """在随机端口启动 uvicorn，返回 base URL；测试结束后关闭。

    返回：
        ``http://127.0.0.1:<port>``，供 Playwright 的 page.goto 使用。

    异常：
        RuntimeError: 进程提前退出，或 60 秒内 /api/health 未就绪。
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    # 继承当前环境；PYTHONUNBUFFERED 让 uvicorn 日志即时输出（虽然这里重定向了）。
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        # 不把 uvicorn 日志混进 pytest 输出；失败时看 RuntimeError 即可。
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            # 进程已经退出就没必要继续等。
            if process.poll() is not None:
                raise RuntimeError(f"uvicorn 提前退出，exit code={process.returncode}")

            try:
                response = httpx.get(f"{base_url}/api/health", timeout=1.0)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    break
            except httpx.HTTPError:
                # 端口还没起来时连接会被拒绝，属于预期，继续轮询。
                pass

            time.sleep(0.5)
        else:
            raise RuntimeError(f"uvicorn 在 60 秒内没有就绪：{base_url}")

        yield base_url
    finally:
        # 先温和 terminate，给 uvicorn 清理机会；10 秒不退再 kill。
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
