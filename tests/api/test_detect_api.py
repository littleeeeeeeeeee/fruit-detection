"""API 测试：健康检查、单图检测、批量检测与错误码。"""
SAMPLE_NAME = "sample.jpg"


def post_file(client, name=SAMPLE_NAME, content=b"", content_type="image/jpeg"):
    return client.post("/api/detect", files={"file": (name, content, content_type)})


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_upload_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_detect_returns_detections(client, sample_image_bytes):
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
    response = post_file(client, name="sample.txt", content=sample_image_bytes, content_type="text/plain")

    assert response.status_code == 415


def test_detect_rejects_filename_without_extension(client, sample_image_bytes):
    response = post_file(client, name="sample", content=sample_image_bytes)

    assert response.status_code == 415


def test_detect_rejects_corrupt_image(client):
    response = post_file(client, content=b"this is not a real jpeg")

    assert response.status_code == 400


def test_detect_requires_a_file(client):
    response = client.post("/api/detect")

    assert response.status_code == 422


def test_batch_detects_every_image(client, sample_image_bytes):
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
    response = client.post("/api/detect/batch")

    assert response.status_code == 422
