"""单元测试：图片字节加载与格式归一化。"""
import io

import pytest
from PIL import Image

from app.detector import InvalidImageError, load_image


def make_image_bytes(size=(64, 48), mode="RGB", color=(255, 0, 0), fmt="PNG") -> bytes:
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_empty_bytes_raises_invalid_image():
    with pytest.raises(InvalidImageError, match="文件为空"):
        load_image(b"")


def test_non_image_bytes_raises_invalid_image():
    with pytest.raises(InvalidImageError):
        load_image(b"this is definitely not an image")


def test_text_disguised_as_jpg_raises_invalid_image():
    # 扩展名由调用方决定，load_image 只看字节内容，伪装成 .jpg 的文本同样应被拒绝
    with pytest.raises(InvalidImageError):
        load_image(b"<?php echo 'not an image'; ?>")


def test_rgba_png_is_converted_to_rgb():
    data = make_image_bytes(size=(64, 48), mode="RGBA", color=(10, 20, 30, 128))

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (64, 48)


def test_rgb_png_keeps_dimensions():
    data = make_image_bytes(size=(37, 91), mode="RGB", fmt="PNG")

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (37, 91)


def test_palette_png_is_converted_to_rgb():
    data = make_image_bytes(size=(16, 16), mode="P", color=3, fmt="PNG")

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (16, 16)
