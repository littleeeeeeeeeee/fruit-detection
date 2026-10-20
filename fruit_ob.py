"""早期命令行版水果检测脚本（历史保留）。

这个脚本是 Web 应用/``ml/train.py`` 之前的手工实验入口：
- 菜单交互；
- 训练、单图预测、文件夹预测、评估；
- 权重路径固定为 ``runs/detect/my_detection/weights/best.pt``。

当前状态：
- 正式训练请用 ``ml/train.py``（支持 MLflow、参数化、--promote）；
- 正式推理请用 FastAPI 服务或 ``app.detector``；
- 本文件保留是为了说明项目演进历史，路径/设备假设仍偏原开发机。

注意：
- ``predict_image`` / ``predict_folder`` 用 ``input()`` 交互，不能用于自动化；
- ``evaluate`` 的 device 固定为 0，CPU 机器上会失败；
- 如果只是演示，请不要用这个脚本，直接用 Web/Docker。
"""
import torch
import os
from ultralytics import YOLO


def train():
    """按硬编码参数训练 YOLOv8n（历史实验配置）。

    关键参数：
        data=yolo_fruit_data/data.yaml
        epochs=100, imgsz=640, batch=16, lr0=0.01
        device=0（有 CUDA）否则 cpu
        amp=False, save_period=10, patience=20
        box=7.0, cls=0.5, dfl=1.5
    输出目录：runs/detect/my_detection
    """
    print("=" * 60)
    print("开始训练")
    print("PyTorch 版本：", torch.__version__)
    print("CUDA 是否可用：", torch.cuda.is_available())

    # 有 GPU 用 0，否则 cpu；没有让用户选择。
    device = 0 if torch.cuda.is_available() else 'cpu'

    # 初始权重 yolov8n.pt；首次运行会下载。
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
        cls=0.5,
        dfl=1.5,
    )

    print("训练完成！")


def predict_image():
    """交互式输入一张图片路径，预测并显示/保存结果。"""
    # 历史固定路径：新训练脚本的 run 名如果是别的，需要手动改。
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
    """交互式输入文件夹，批量预测并保存结果。"""
    model_path = "runs/detect/my_detection/weights/best.pt"

    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练！")
        return

    folder_path = input("请输入文件夹路径：").strip()
    model = YOLO(model_path)

    # save=True：把带框图片写到 runs/detect/predict*；show=True：弹窗显示。
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
    """在 data.yaml 的 val 上评估固定权重（设备固定 0）。"""
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
    """命令行菜单循环；输入 5 退出。"""
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


# ---------------------------------------------------------------------------
# 以下是原开发机上的历史实验记录，保留作超参数对比参考：
#
# box=7.5, cls=0.5, dfl=1.5:
# Model summary (fused): 72 layers, 3,006,233 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
#                    all         32         76      0.877      0.883      0.943      0.689
#                  apple         12         25      0.765       0.96      0.973      0.741
#                 banana         13         26      0.895      0.769      0.863      0.534
#                 orange         12         25      0.972       0.92      0.992      0.792
#
# box=7.5, cls=0.7, dfl=1.5:
#                    all         32         76      0.871      0.932      0.943      0.678
#
# box=8.0, cls=0.5, dfl=1.5:
#                    all         32         76      0.861       0.88       0.95      0.683
#
# 备注：这些是早期 50 epoch/未清洗数据上的实验值，不能和当前
# improved-v2 的 0.930/0.700 直接比较。
# ---------------------------------------------------------------------------
