"""生成数据集 v2：清洗训练集。

规则（只处理 images/train 与 labels/train）：
1. 没有对应 label 文件的图片
2. label 文件为空
3. label 行格式非法 / 坐标越界
4. 图片无法打开
5. bbox 面积过小（< 0.1% 图像面积）或宽高 < 320px 的图

被删除的文件移动到 yolo_fruit_data/_removed/ 下备份，可回滚。
"""
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "yolo_fruit_data"
IMG_TRAIN = ROOT / "images" / "train"
LBL_TRAIN = ROOT / "labels" / "train"
BACKUP = ROOT / "_removed"
MIN_AREA_RATIO = 0.001
MIN_SIDE = 320


def move_pair(img_path, reason, buckets):
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), BACKUP / img_path.name)
    lbl = LBL_TRAIN / (img_path.stem + ".txt")
    if lbl.is_file():
        shutil.move(str(lbl), BACKUP / lbl.name)
    buckets[reason] += 1


def parse_labels(lbl_path):
    rows = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return None
        try:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            return None
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        if w * h < MIN_AREA_RATIO:
            return None
        rows.append((cid, cx, cy, w, h))
    return rows


def main():
    images = sorted(
        p for p in IMG_TRAIN.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    buckets = Counter()
    seen = {}
    total = len(images)

    for img_path in images:
        lbl_path = LBL_TRAIN / (img_path.stem + ".txt")

        if not lbl_path.is_file():
            move_pair(img_path, "无标签文件", buckets)
            continue

        rows = parse_labels(lbl_path)
        if rows is None:
            move_pair(img_path, "标签非法/越界/过小", buckets)
            continue
        if not rows:
            move_pair(img_path, "标签为空", buckets)
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            move_pair(img_path, "图片损坏", buckets)
            continue

        if min(w, h) < MIN_SIDE:
            move_pair(img_path, "分辨率过低", buckets)
            continue

        digest = hashlib.md5(img_path.read_bytes()).hexdigest()
        if digest in seen:
            move_pair(img_path, "重复图片", buckets)
            continue
        seen[digest] = img_path.name

    left = len([p for p in IMG_TRAIN.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"[v2] 清洗前 {total} 张，清洗后 {left} 张，移除 {total - left} 张")
    for reason, n in buckets.items():
        print(f"  - {reason}: {n}")
    if total == left:
        print("[v2] 没有发现脏数据，数据集与 v1 相同")


if __name__ == "__main__":
    main()
