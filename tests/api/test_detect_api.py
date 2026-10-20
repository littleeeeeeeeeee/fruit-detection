"""API 测试：健康检查、单图检测、批量检测与错误码。

使用 FastAPI TestClient，不启动真实网络端口；e2e 才启动 uvicorn + 浏览器。

覆盖点：
- /api/health 固定返回；
- / 返回 HTML 上传页；
- 单图成功路径、文件名/计数/耗时；
- 415：扩展名不支持 / 无扩展名；
- 400：扩展名合法但字节不是图片；
- 422：缺 file/files 字段；
- 批量：两张图、文件名顺序、总数一致。
"""
SAMPLE_NAME = "sample.jpg"


def post_file(client, name=SAMPLE_NAME, content=b"", content_type="image/jpeg"):
    """向 /api/detect POST 一个 multipart 文件，返回 Response。

    参数：
        client: TestClient fixture。
        name: 上传文件名（决定扩展名校验结果）。
        content: 文件字节。
        content_type: multipart 中声明的 MIME；服务端实际按 PIL 解码校验。
    """
    return client.post("/api/detect", files={"file": (name, content, content_type)})


def test_health(client):
    """健康检查必须返回 200 + 固定 JSON，Docker/CI 依赖这个契约。"""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_upload_page(client):
    """根路径返回 HTML 页面，而不是 JSON。"""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_detect_returns_detections(client, sample_image_bytes):
    """单图成功路径：文件名回显、count 与 detections 长度一致、耗时为非负整数。"""
    response = post_file(client, content=sample_image_bytes)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == SAMPLE_NAME
    assert isinstance(body["count"], int)
    assert body["count"] >= 1
    assert len(body["detections"]) == body["count"]
    assert isinstance(body["elapsed_ms"], int)
    assert body["elapsed_ms"] >= 0


def test_detect_rejects_unsupported_extension(client, sample_image_bytes):
    """扩展名 .txt 不在白名单，即使内容是有效图片也应 415。"""
    response = post_file(client, name="sample.txt", content=sample_image_bytes, content_type="text/plain")

    assert response.status_code == 415


def test_detect_rejects_filename_without_extension(client, sample_image_bytes):
    """没有扩展名的文件名同样 415。"""
    response = post_file(client, name="sample", content=sample_image_bytes)

    assert response.status_code == 415


def test_detect_rejects_corrupt_image(client):
    """扩展名 .jpg 合法，但字节不是图片 -> 400（PIL 解码失败）。"""
    response = post_file(client, content=b"this is not a real jpeg")

    assert response.status_code == 400


def test_detect_requires_a_file(client):
    """完全不带 file 字段时，FastAPI 参数校验返回 422。"""
    response = client.post("/api/detect")

    assert response.status_code == 422


def test_batch_detects_every_image(client, sample_image_bytes):
    """批量两张相同图片：应返回两条结果，顺序与上传顺序一致。"""
    files = [
        ("files", ("first.jpg", sample_image_bytes, "image/jpeg")),
        ("files", ("second.jpg", sample_image_bytes, "image/jpeg")),
    ]

    response = client.post("/api/detect/batch", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["total_images"] == 2
    assert len(body["results"]) == 2
    assert [item["filename"] for item in body["results"]] == ["first.jpg", "second.jpg"]
    assert body["total_detections"] == sum(item["count"] for item in body["results"])


def test_batch_requires_files(client):
    """批量接口不带 files 字段 -> 422。"""
    response = client.post("/api/detect/batch")

    assert response.status_code == 422
