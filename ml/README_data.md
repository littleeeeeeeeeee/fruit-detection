# 数据集说明

## 数据来源

水果目标检测数据集，共 3 个类别：apple、banana、orange。
图像为真实拍摄的水果照片，标注格式为 YOLO（每张图片对应一个 `.txt`，每行 `class cx cy w h`，坐标已归一化）。
原始数据与 XML 标注保存在 `fruit_data/`，转换脚本为 `ml/data_pipeline.py`，转换后生成 YOLO 格式的 `yolo_fruit_data/`。

## 版本管理方式

数据使用 DVC 管理，远程存储为 Azure Blob Storage：

| 配置项 | 值 |
|---|---|
| 远程名 | `myazure` |
| 存储账户 | `wut2026` |
| 容器 | `dvc` |
| 远程地址 | `azure://dvc` |

Git 中只保留轻量指针文件：

- `yolo_fruit_data/images.dvc`
- `yolo_fruit_data/labels.dvc`

认证使用 SAS token，保存在 `.dvc/config.local`（已被 Git 忽略），**不进入版本库**。

## 版本对照表

| 数据版本 | Git tag | 对应 commit | 训练集 | 验证集 | 变化说明 |
|---|---|---|---|---|---|
| v1 | `data-v1` | `b13a344` | 208 | 32 | 初始可用数据集 |
| v2 | `data-v2` | `bad7453` | 166 | 32 | 剔除 42 张最短边 < 320px 的低分辨率样本 |

验证集与测试集在两个版本之间**保持不变**，因此 v1 与 v2 训练出的模型指标可直接比较。

## 变更明细（v1 → v2）

清洗规则实现在 `ml/clean_dataset.py`，逐条检查：

1. 缺少对应 label 文件的图片
2. label 文件为空
3. label 行数不为 5 列、或坐标越界
4. 图片无法正常打开
5. bbox 面积占整图比例 < 0.1%
6. 图片最短边 < 320px
7. 内容完全重复（md5 相同）

v2 实际移除：**42 张，原因均为"分辨率过低"**，占原训练集的 20%。
被移除的文件备份在 `yolo_fruit_data/_removed/`（不纳入 DVC 与 Git），可随时回滚。

## 代码与数据的对应关系

每个训练 run 在 MLflow 中同时记录了：

- `dataset_version`（v1 / v2）
- `git_commit`（训练时的代码版本）

因此可以回答："这个模型是由哪份代码、哪个数据版本训练出来的"。

## 复现数据

    dvc pull          # 拉取当前 tag 对应的数据
    git checkout data-v1 && dvc pull    # 切到 v1 数据
    git checkout data-v2 && dvc pull    # 切到 v2 数据