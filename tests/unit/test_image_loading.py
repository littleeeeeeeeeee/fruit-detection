"""单元测试：图片字节加载与格式归一化。

这一层不加载 YOLO，也不经过 FastAPI；只测 ``app.detector.load_image`` 的
边界行为，因此运行很快，适合在 CI 里先跑。

覆盖点：
- 空字节；
- 完全不是图片的文本；
- 伪装成图片的文本（扩展名由调用方决定，load_image 只看字节）；
- RGBA / P 模式统一转 RGB；
- 正常 RGB 尺寸不变。
"""
import io

import pytest
from PIL import Image

from app.detector import InvalidImageError, load_image


def make_image_bytes(size=(64, 48), mode="RGB", color=(255, 0, 0), fmt="PNG") -> bytes:
    """用 PIL 现场生成一张指定模式/尺寸的图片字节。

    参数：
        size: (宽, 高)。
        mode: RGB/RGBA/P 等 PIL 模式。
        color: RGB 元组或调色板索引（P 模式）。
        fmt: 保存格式，默认 PNG（无损，便于断言尺寸/模式）。

    返回：
        bytes，可直接传给 load_image。
    """
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_empty_bytes_raises_invalid_image():
    """空字节应该在进入 PIL 之前就被拒绝，错误信息应为“文件为空”。"""
    with pytest.raises(InvalidImageError, match="文件为空"):
        load_image(b"")


def test_non_image_bytes_raises_invalid_image():
    """随机文本无法被 PIL 识别，应抛 InvalidImageError。"""
    with pytest.raises(InvalidImageError):
        load_image(b"this is definitely not an image")


def test_text_disguised_as_jpg_raises_invalid_image():
    """伪装成 .jpg 的文本同样应被拒绝。

    这条测试强调：扩展名不是安全边界，真正的格式校验在字节内容。
    """
    with pytest.raises(InvalidImageError):
        load_image(b"<?php echo 'not an image'; ?>")


def test_rgba_png_is_converted_to_rgb():
    """RGBA 图（带 alpha）必须转成 RGB，否则 YOLO 输入通道数不匹配。"""
    data = make_image_bytes(size=(64, 48), mode="RGBA", color=(10, 20, 30, 128))

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (64, 48)


def test_rgb_png_keeps_dimensions():
    """正常 RGB 图不应该被缩放/裁剪，尺寸原样保留。"""
    data = make_image_bytes(size=(37, 91), mode="RGB", fmt="PNG")

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (37, 91)


def test_palette_png_is_converted_to_rgb():
    """P（调色板）模式也要转 RGB；这是旧数据/截图常见格式。"""
    data = make_image_bytes(size=(16, 16), mode="P", color=3, fmt="PNG")

    image = load_image(data)

    assert image.mode == "RGB"
    assert image.size == (16, 16)
