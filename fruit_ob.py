import torch
import os
from ultralytics import YOLO


def train():
    """训练模型"""
    print("=" * 60)
    print("开始训练")
    print("PyTorch 版本：", torch.__version__)
    print("CUDA 是否可用：", torch.cuda.is_available())

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO("yolov8n.pt")

    model.train(
        data="yolo_fruit_data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        device=device,
        amp=False,
        save_period=10,
        patience=20,
        workers=4,
        name="my_detection",
        exist_ok=False,
        verbose=True,
        box=7,
        cls = 0.5,
        dfl = 1.5,
    )

    print("训练完成！")


def predict_image():
    """预测单张图片"""
    model_path = "runs/detect/my_detection/weights/best.pt"

    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练！")
        return

    image_path = input("请输入图片路径：").strip()
    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        save=True,
        show=True,
        conf=0.25,
        iou=0.7,
        line_width=2,
    )

    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("未检测到目标")
            continue

        print(f"检测到 {len(boxes)} 个目标：")
        print("-" * 40)

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = model.names[cls_id]

            print(f"类别：{class_name}")
            print(f"置信度：{conf:.3f}")
            print(f"坐标：x1={xyxy[0]:.1f}, y1={xyxy[1]:.1f}, x2={xyxy[2]:.1f}, y2={xyxy[3]:.1f}")
            print("-" * 40)


def predict_folder():
    """预测文件夹里所有图片"""
    model_path = "runs/detect/my_detection/weights/best.pt"

    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练！")
        return

    folder_path = input("请输入文件夹路径：").strip()
    model = YOLO(model_path)

    model.predict(
        source=folder_path,
        save=True,
        show=True,
        conf=0.25,
        iou=0.7,
        line_width=2,
    )

    print("检测完成！结果保存在 runs/detect/predict 文件夹中")


def evaluate():
    """评估模型"""
    model_path = "runs/detect/my_detection/weights/best.pt"

    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练！")
        return

    model = YOLO(model_path)
    metrics = model.val(
        data="yolo_fruit_data/data.yaml",
        device=0,
    )

    print("=" * 40)
    print("模型评估结果：")
    print(f"mAP@0.5：{metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95：{metrics.box.map:.4f}")


def main_menu():
    """主菜单"""
    while True:
        print("\n" + "=" * 40)
        print("YOLO 目标检测程序")
        print("=" * 40)
        print("1. 训练模型")
        print("2. 预测单张图片")
        print("3. 预测文件夹里所有图片")
        print("4. 评估模型")
        print("5. 退出程序")
        print("=" * 40)

        choice = input("请输入数字选择功能：").strip()

        if choice == "1":
            train()
        elif choice == "2":
            predict_image()
        elif choice == "3":
            predict_folder()
        elif choice == "4":
            evaluate()
        elif choice == "5":
            print("程序退出")
            break
        else:
            print("输入错误，请重新输入！")


if __name__ == "__main__":
    main_menu()


# box=7.5,cls = 0.5,dfl = 1.5:
# Model summary (fused): 72 layers, 3,006,233 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0s/it 1.0s
#                    all         32         76      0.877      0.883      0.943      0.689
#                  apple         12         25      0.765       0.96      0.973      0.741
#                 banana         13         26      0.895      0.769      0.863      0.534
#                 orange         12         25      0.972       0.92      0.992      0.792
# Speed: 0.4ms preprocess, 7.8ms inference, 0.0ms loss, 2.7ms postprocess per image
# Results saved to C:\Users\21555\PycharmProjects\pythonProject\runs\detect\my_detection


# box=7.5,cls = 0.7,dfl = 1.5:
# Model summary (fused): 72 layers, 3,006,233 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
#                    all         32         76      0.871      0.932      0.943      0.678
#                  apple         12         25      0.747          1      0.967      0.759
#                 banana         13         26      0.883      0.875      0.872      0.517
#                 orange         12         25      0.981       0.92      0.989      0.759
# Speed: 0.4ms preprocess, 7.4ms inference, 0.0ms loss, 1.4ms postprocess per image


# box=8,cls = 0.5,dfl = 1.5:
# Model summary (fused): 72 layers, 3,006,233 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
#                    all         32         76      0.861       0.88       0.95      0.683
#                  apple         12         25      0.747       0.96      0.972      0.751
#                 banana         13         26      0.876      0.731      0.886      0.534
#                 orange         12         25       0.96      0.949       0.99      0.764
# Speed: 0.4ms preprocess, 7.2ms inference, 0.0ms loss, 2.6ms postprocess per image
#
# 用脚本画框展示数据集
# f1的所有构成的概念
# 超参数优化