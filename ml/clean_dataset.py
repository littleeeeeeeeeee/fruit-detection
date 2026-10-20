"""生成数据集 v2：清洗训练集。

这个脚本只处理 ``images/train`` 与 ``labels/train``，不会碰 val/test，
因此 v1 和 v2 的验证/测试指标仍然可比，也不会把 test 泄漏进训练。

清洗规则：
1. 没有对应 label 文件的图片；
2. label 文件为空；
3. label 行不是 5 列、类别/坐标非法或越界；
4. 图片无法打开/损坏；
5. bbox 面积占全图 < 0.1%；
6. 图片最短边 < 320px；
7. 图片内容重复（MD5 相同）。

删除方式：
- 图片和对应 label 一起移动到 ``yolo_fruit_data/_removed/``；
- 不做物理删除，方便回滚；
- ``_removed`` 不进入 Git/DVC。

注意：
- 脚本是“移动”而不是“复制”，重复运行会把当前 train 继续移走；
  只在 v1→v2 时运行一次，或先备份/恢复；
- 统计按“图片”计数，不是按 bbox 计数。
"""
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# 所有路径都相对脚本位置推导，clone 到任意目录都能运行。
ROOT = Path(__file__).resolve().parent.parent / "yolo_fruit_data"
IMG_TRAIN = ROOT / "images" / "train"
LBL_TRAIN = ROOT / "labels" / "train"
BACKUP = ROOT / "_removed"

# 一个 bbox 至少占全图 0.1% 面积，过滤标注错误/极小目标。
MIN_AREA_RATIO = 0.001

# 最短边小于 320px 的图在 640 输入下会被明显上采样，标注质量也差。
MIN_SIDE = 320


def move_pair(img_path, reason, buckets):
    """把一张图和它的 label 移到备份目录，并累计原因计数。

    参数：
        img_path: 待移除图片的 Path。
        reason: 统计原因（用于最后打印）。
        buckets: collections.Counter，按原因累加。

    副作用：
        - 创建 BACKUP（如不存在）；
        - 移动图片；
        - 如果同名 .txt 存在，一起移动；
        - 修改 buckets。
    """
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), BACKUP / img_path.name)

    lbl = LBL_TRAIN / (img_path.stem + ".txt")
    if lbl.is_file():
        shutil.move(str(lbl), BACKUP / lbl.name)

    buckets[reason] += 1


def parse_labels(lbl_path):
    """解析并校验一个 YOLO label 文件。

    参数：
        lbl_path: .txt label 的 Path。

    返回：
        - list[tuple]：合法行 (class_id, cx, cy, w, h)；
        - None：文件中有非法行（列数、数值、坐标范围、面积不合法）；
        - []：文件为空或只有空行。

    注意：
        YOLO 的坐标是归一化的：cx/cy 在 [0,1]，w/h 在 (0,1]。
        这里不做“class_id 是否 < nc”的检查，类别范围由 data.yaml 约束；
        当前数据只有 0/1/2。
    """
    rows = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            # 空行不是错误，直接忽略。
            continue

        if len(parts) != 5:
            return None

        try:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            return None

        # 坐标必须归一化且在合理范围；宽高必须为正。
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None

        # 面积太小通常是误标/极端小目标。
        if w * h < MIN_AREA_RATIO:
            return None

        rows.append((cid, cx, cy, w, h))

    return rows


def main():
    """遍历 train，按规则移走坏样本并打印统计。

    执行顺序很重要：
    - 先查 label 是否存在/合法，再打开图片（避免对无效样本做解码）；
    - 再做分辨率检查；
    - 最后做 MD5 去重（需要读全文件，成本最高）。
    """
    # 只处理常见图片扩展名；排序保证每次运行顺序稳定，便于复现统计。
    images = sorted(
        p for p in IMG_TRAIN.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    buckets = Counter()
    seen = {}
    total = len(images)

    for img_path in images:
        lbl_path = LBL_TRAIN / (img_path.stem + ".txt")

        # 规则 1：图片没有对应 label。
        if not lbl_path.is_file():
            move_pair(img_path, "无标签文件", buckets)
            continue

        # 规则 2/3/5：label 为空、格式/坐标非法、bbox 过小。
        rows = parse_labels(lbl_path)
        if rows is None:
            move_pair(img_path, "标签非法/越界/过小", buckets)
            continue
        if not rows:
            move_pair(img_path, "标签为空", buckets)
            continue

        # 规则 4：图片无法打开/损坏。
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            move_pair(img_path, "图片损坏", buckets)
            continue

        # 规则 6：分辨率过低。
        if min(w, h) < MIN_SIDE:
            move_pair(img_path, "分辨率过低", buckets)
            continue

        # 规则 7：内容重复（MD5 只看内容，不看文件名）。
        digest = hashlib.md5(img_path.read_bytes()).hexdigest()
        if digest in seen:
            move_pair(img_path, "重复图片", buckets)
            continue
        seen[digest] = img_path.name

    # 重新数一遍剩余图片，避免用 total - sum(buckets) 时把未知原因算错。
    left = len([p for p in IMG_TRAIN.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"[v2] 清洗前 {total} 张，清洗后 {left} 张，移除 {total - left} 张")
    for reason, n in buckets.items():
        print(f"  - {reason}: {n}")

    if total == left:
        print("[v2] 没有发现脏数据，数据集与 v1 相同")


if __name__ == "__main__":
    main()
