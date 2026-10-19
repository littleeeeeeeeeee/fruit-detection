"""E2E 测试：真实浏览器上传图片，断言检测结果和检测框出现。"""
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="未安装 playwright，跳过 e2e")

from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.jpg"


@pytest.fixture(scope="module")
def page(live_server):
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
    page.goto(live_server, wait_until="domcontentloaded")
    page.set_input_files("#single", str(SAMPLE_IMAGE))
    page.click("#singleBtn")

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
