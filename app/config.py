"""应用运行时配置。

这个模块是 Web 服务和测试共用的“唯一配置入口”。所有值都可以通过环境变量覆盖，
因此本地开发、Docker、GitHub Actions 的 staging/production 可以共用同一份代码，
只靠环境变量区分行为。

环境变量：
    MODEL_PATH        YOLO 权重路径，默认 <项目根>/models/best.pt
    CONF_THRESHOLD    置信度阈值，默认 0.25
    IOU_THRESHOLD     NMS 的 IoU 阈值，默认 0.7
    MAX_IMAGE_MB      单张图片大小上限（MB），默认 10

设计说明：
- 这里只做“读配置”，不负责创建目录或校验权重；权重是否存在由
  ``FruitDetector.__init__`` 检查，避免 import config 时就抛异常。
- ``float(...)`` 在环境变量非法时会直接抛 ValueError，这是有意的：
  配置写错应该在启动阶段暴露，而不是悄悄回退到默认值。
"""
import os
from pathlib import Path

# 项目根目录：app/config.py -> app -> 项目根。
# 用 resolve() 是为了让相对路径/符号链接在 Docker 和本地都得到绝对路径。
BASE_DIR = Path(__file__).resolve().parent.parent

# 生产权重默认放在仓库的 models/ 下；Docker 镜像里同样复制到 /app/models/best.pt。
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))

# 检测阈值：
# - conf 越低，召回越高，但误检/低置信框更多；
# - iou 越高，NMS 保留的相邻框越多，密集水果场景更明显。
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.7"))

# 上传大小限制，单位 MB；API 层会在读取 bytes 后检查。
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "10"))

# 只允许常见图片扩展名。真正的图片内容仍由 PIL 校验，
# 扩展名只是第一道便宜的过滤，避免把明显不支持的文件读进内存。
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
