"""E2E 测试：真实浏览器上传图片，断言检测结果和检测框出现。

依赖：
- pytest-playwright 不需要；这里直接用 playwright.sync_api；
- 未安装 playwright 时 importorskip 跳过整个模块；
- 安装但 Chromium 不可用时在 fixture 里 skip。

流程：
1. ``live_server`` 启动 uvicorn；
2. Playwright 启动 Chromium；
3. 打开页面、set_input_files、点击按钮；
4. 用 ``wait_for_function`` 等 DOM 更新（不用 sleep，减少 flaky）；
5. 断言消息、列表、框数量。
"""
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="未安装 playwright，跳过 e2e")

from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.jpg"


@pytest.fixture(scope="module")
def page(live_server):
    """模块级浏览器 page：两个 e2e 用例复用同一个浏览器/页面。

    注意：两个用例都会 page.goto，页面状态互不影响；如果未来用例依赖
    上一条的状态，需要改成 function scope。
    """
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except PlaywrightError as exc:
            pytest.skip(f"Chromium 不可用，跳过 e2e：{exc}")

        page = browser.new_page()
        try:
            yield page
        finally:
            browser.close()


@pytest.mark.e2e
def test_single_upload_shows_detection(live_server, page):
    """单图上传：等“个目标”消息 + 至少一个 .box + 列表里出现 apple。"""
    page.goto(live_server, wait_until="domcontentloaded")
    page.set_input_files("#single", str(SAMPLE_IMAGE))
    page.click("#singleBtn")

    # 先等文本出现，再等框渲染；框依赖 img.onload，可能比文本晚一点。
    page.wait_for_function(
        "document.querySelector('#singleMsg').textContent.includes('个目标')",
        timeout=60_000,
    )
    page.wait_for_function(
        "document.querySelectorAll('#overlay .box').length >= 1",
        timeout=30_000,
    )

    message = page.locator("#singleMsg").inner_text()
    items = page.locator("#singleList li").all_inner_texts()
    boxes = page.locator("#overlay .box")

    assert "个目标" in message
    assert items
    assert any("apple" in item for item in items)
    assert boxes.count() >= 1


@pytest.mark.e2e
def test_batch_upload_shows_every_image(live_server, page):
    """批量上传两张图：消息显示 2 张，列表恰好两个外层 li。"""
    page.goto(live_server, wait_until="domcontentloaded")
    page.set_input_files("#batch", [str(SAMPLE_IMAGE), str(SAMPLE_IMAGE)])
    page.click("#batchBtn")

    page.wait_for_function(
        "document.querySelector('#batchMsg').textContent.includes('2 张图片')",
        timeout=60_000,
    )
    page.wait_for_function(
        "document.querySelectorAll('#batchList > li').length === 2",
        timeout=30_000,
    )

    message = page.locator("#batchMsg").inner_text()
    results = page.locator("#batchList > li")

    assert "2 张图片" in message
    assert results.count() == 2
