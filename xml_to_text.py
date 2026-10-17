import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

RAW_DATA_DIR = r"C:\Users\21555\PycharmProjects\pythonProject\fruit data"
CLASSES = ["apple", "banana", "orange"]
OUTPUT_DIR = r"C:\Users\21555\PycharmProjects\pythonProject\yolo_fruit data"

def convert_xml_to_yolo(xml_path, img_path, classes):
    """单个 XML 转 YOLO TXT 内容，如果 XML 尺寸为 0，从图片读取"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 读取图片尺寸
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        # 如果 XML 里的尺寸为 0，从实际图片读取
        if img_w == 0 or img_h == 0:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            print(f"  ⚠️  XML 尺寸为 0，从图片读取: {img_path.name} -> {img_w}x{img_h}")

        yolo_lines = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip()

            if class_name not in classes:
                print(f"  ⚠️  未知类别: {class_name}，跳过")
                continue

            class_id = classes.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换到 YOLO 格式（归一化）
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_lines

    except Exception as e:
        print(f"  ❌ 解析失败 {xml_path}: {e}")
        return None


def find_all_files(directory, extensions):
    """递归查找所有指定扩展名的文件"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return files


def main():
    raw_dir = Path(RAW_DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not raw_dir.exists():
        print(f"❌ 错误：输入目录不存在: {raw_dir}")
        return

    print("=" * 60)
    print("开始处理数据集...")
    print("=" * 60)

    # 1. 查找所有图片
    print("\n📂 查找图片...")
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_images = find_all_files(raw_dir, image_exts)
    print(f"  找到 {len(all_images)} 张图片")

    # 2. 查找所有 XML
    print("\n📂 查找 XML 标注...")
    all_xmls = find_all_files(raw_dir, ['.xml'])
    print(f"  找到 {len(all_xmls)} 个 XML 文件")

    if len(all_images) == 0:
        print("❌ 错误：没有找到任何图片")
        return

    # 3. 创建 XML 查找字典（按文件名）
    xml_dict = {}
    for xml_path in all_xmls:
        stem = xml_path.stem
        xml_dict[stem] = xml_path

    # 4. 配对图片和 XML
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

    # 5. 判断是否需要自动划分
    has_split = False
    for folder_name in ['train', 'val', 'test']:
        if (raw_dir / folder_name).exists():
            has_split = True
            break

    # 6. 创建输出目录结构
    print("\n📁 创建输出目录...")
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    # 7. 分配数据
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

    # 8. 转换并复制文件
    print("\n🔄 转换标注并复制文件...")
    stats = {'train': 0, 'val': 0, 'test': 0}

    for img_path, xml_path in paired:
        split = split_assignment[img_path]

        # 转换 XML（传入图片路径，用于读取真实尺寸）
        yolo_lines = convert_xml_to_yolo(xml_path, img_path, CLASSES)

        if yolo_lines is None:
            continue

        # 复制图片
        img_output = output_dir / 'images' / split / img_path.name
        shutil.copy2(img_path, img_output)

        # 写入 TXT
        txt_output = output_dir / 'labels' / split / (img_path.stem + '.txt')
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        stats[split] += 1

    # 9. 生成 data.yaml
    print("\n📝 生成 data.yaml...")
    yaml_content = f"""# YOLO 数据集配置文件
train: {output_dir / 'images' / 'train'}
val: {output_dir / 'images' / 'val'}
test: {output_dir / 'images' / 'test'}

nc: {len(CLASSES)}
names: {CLASSES}
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 10. 打印统计
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