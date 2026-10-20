"""把 Pascal VOC XML 标注转换为 YOLO 格式的数据集。

输入：
    fruit_data/
        train/  val/  test/       # 原数据按 split 分目录
        *.jpg / *.xml             # 图片与同名 XML 标注

输出：
    yolo_fruit_data/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml                 # 相对路径 + nc/names

核心转换：
    VOC: xmin, ymin, xmax, ymax（绝对像素，左上/右下）
    YOLO: class_id cx cy w h（归一化到 0..1，中心点 + 宽高）

边界处理：
- XML 的 <size> 宽高为 0 时，用 PIL 读取真实图片尺寸；
- 未知类别跳过并打印警告；
- XML 解析失败返回 None，主流程跳过该样本；
- 如果原目录没有 train/val/test，则按 80/10/10 随机划分。

注意：
- 脚本是“覆盖式”生成：同名输出文件会被覆盖；
- 原始数据不在 Git 中，需要本地放置 fruit_data/ 或使用 DVC 已版本化结果；
- 路径以脚本位置为基准，clone 到任何目录都能运行。
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# 项目根/输入/输出都用脚本位置推导，避免旧机器的绝对路径。
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "fruit_data"
CLASSES = ["apple", "banana", "orange"]
OUTPUT_DIR = BASE_DIR / "yolo_fruit_data"


def convert_xml_to_yolo(xml_path, img_path, classes):
    """把单个 XML 转成若干行 YOLO 文本（不写文件）。

    参数：
        xml_path: Pascal VOC XML 路径。
        img_path: 对应图片路径，仅在 XML 尺寸为 0 时用来读取真实尺寸。
        classes: 类别名列表，顺序决定 class_id。

    返回：
        list[str]：每行 ``class_id cx cy w h``，保留 6 位小数；
        None：XML 解析失败（文件损坏/字段缺失等）。

    说明：
        这里不校验坐标是否越界；如果 XML 本身有错，生成的框可能落在图外。
        当前原始数据已经人工检查过，清洗阶段另有 bbox 面积/坐标规则。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 读取 XML 中声明的图片尺寸。
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        # 如果 XML 里的尺寸为 0，从实际图片读取。
        if img_w == 0 or img_h == 0:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            print(f"  ⚠️  XML 尺寸为 0，从图片读取: {img_path.name} -> {img_w}x{img_h}")

        yolo_lines = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip()

            # 只保留 apples/banana/orange；未知类别跳过而不是静默丢图。
            if class_name not in classes:
                print(f"  ⚠️  未知类别: {class_name}，跳过")
                continue

            class_id = classes.index(class_name)

            # VOC 的 bndbox 是绝对像素坐标。
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换到 YOLO 格式：中心点 + 宽高，全部除以图片宽高归一化。
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_lines

    except Exception as e:
        # 宽泛捕获是有意的：单个坏 XML 不应该让整个转换中断。
        print(f"  ❌ 解析失败 {xml_path}: {e}")
        return None


def find_all_files(directory, extensions):
    """递归查找目录下所有指定扩展名的文件。

    参数：
        directory: 根目录。
        extensions: 扩展名列表，如 ['.jpg', '.xml']。

    返回：
        list[Path]；顺序不保证，调用方通常按 stem 建索引。
    """
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return files


def main():
    """执行完整的转换流程：扫描 -> 配对 -> 分 split -> 复制/写标签 -> data.yaml。"""
    raw_dir = Path(RAW_DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not raw_dir.exists():
        print(f"❌ 错误：输入目录不存在: {raw_dir}")
        return

    print("=" * 60)
    print("开始处理数据集...")
    print("=" * 60)

    # 1. 查找所有图片（支持常见格式）。
    print("\n📂 查找图片...")
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_images = find_all_files(raw_dir, image_exts)
    print(f"  找到 {len(all_images)} 张图片")

    # 2. 查找所有 XML。
    print("\n📂 查找 XML 标注...")
    all_xmls = find_all_files(raw_dir, ['.xml'])
    print(f"  找到 {len(all_xmls)} 个 XML 文件")

    if len(all_images) == 0:
        print("❌ 错误：没有找到任何图片")
        return

    # 3. 建索引：stem -> xml 路径。假设文件名唯一；若重复，后者覆盖前者。
    xml_dict = {}
    for xml_path in all_xmls:
        stem = xml_path.stem
        xml_dict[stem] = xml_path

    # 4. 按 stem 配对图片与 XML；没有 XML 的图片记录到 unpaired。
    print("\n🔗 配对图片和标注...")
    paired = []
    unpaired_images = []

    for img_path in all_images:
        stem = img_path.stem
        if stem in xml_dict:
            paired.append((img_path, xml_dict[stem]))
        else:
            unpaired_images.append(img_path)

    print(f"  成功配对: {len(paired)} 对")
    print(f"  无标注的图片: {len(unpaired_images)} 张")

    # 5. 判断原目录是否已经有 train/val/test 结构。
    has_split = False
    for folder_name in ['train', 'val', 'test']:
        if (raw_dir / folder_name).exists():
            has_split = True
            break

    # 6. 创建输出目录（已存在则复用）。
    print("\n📁 创建输出目录...")
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    # 7. 分配 split：
    #    - 有原目录结构：按路径中的 test/val/train 判断，未知则归 train；
    #    - 没有：随机 shuffle 后 80/10/10。
    if has_split:
        print("\n📋 检测到 train/val/test 结构，按原文件夹分配...")
        split_assignment = {}
        for img_path, xml_path in paired:
            relative = img_path.relative_to(raw_dir)
            parts = relative.parts
            if 'test' in parts:
                split_assignment[img_path] = 'test'
            elif 'val' in parts:
                split_assignment[img_path] = 'val'
            elif 'train' in parts:
                split_assignment[img_path] = 'train'
            else:
                split_assignment[img_path] = 'train'
    else:
        print("\n📋 未检测到分文件夹，自动划分数据集...")
        random.shuffle(paired)
        n = len(paired)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        split_assignment = {}
        for i, (img_path, _) in enumerate(paired):
            if i < n_train:
                split_assignment[img_path] = 'train'
            elif i < n_train + n_val:
                split_assignment[img_path] = 'val'
            else:
                split_assignment[img_path] = 'test'

    # 8. 转换并复制：图片复制到 images/<split>，标签写到 labels/<split>。
    print("\n🔄 转换标注并复制文件...")
    stats = {'train': 0, 'val': 0, 'test': 0}

    for img_path, xml_path in paired:
        split = split_assignment[img_path]

        # 转换 XML（传入图片路径，用于读取真实尺寸）。
        yolo_lines = convert_xml_to_yolo(xml_path, img_path, CLASSES)

        if yolo_lines is None:
            continue

        # 复制图片（保留元数据）。
        img_output = output_dir / 'images' / split / img_path.name
        shutil.copy2(img_path, img_output)

        # 写 YOLO txt；一张图一个文件。
        txt_output = output_dir / 'labels' / split / (img_path.stem + '.txt')
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        stats[split] += 1

    # 9. 生成 data.yaml：不写 path，ultralytics 用 yaml 所在目录作为根。
    print("\n📝 生成 data.yaml...")
    yaml_content = f"""# YOLO 数据集配置文件
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 10. 打印统计和下一步命令。
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    print(f"\n📊 数据统计:")
    for split, count in stats.items():
        print(f"  {split}: {count} 张图片")

    print(f"\n📁 输出目录: {output_dir}")
    print(f"📄 配置文件: {yaml_path}")

    print("\n🎯 下一步：")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100")

    if len(unpaired_images) > 0:
        print(f"\n⚠️  注意: {len(unpaired_images)} 张图片没有对应的 XML，已跳过")


if __name__ == "__main__":
    main()
