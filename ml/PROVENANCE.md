# 数据来源与格式转换（Provenance）

## 1. 原始数据

- 位置：`fruit_data/`（本地工作区，**不提交 Git**，`.gitignore` 已忽略）
- 规模：300 张 JPEG + 300 个 Pascal VOC XML 标注
- 类别：`apple`、`banana`、`orange`
- 目录：`fruit_data/train`、`fruit_data/val`、`fruit_data/test`
- 原始来源：课程/项目交接提供的 fruit detection 数据集；交接材料没有记录公开下载 URL 或许可证，这是一个已知的 provenance 缺口。

## 2. XML → YOLO 转换

转换脚本：项目根目录 `xml_to_text.py`（历史脚本，保留作来源说明）。

流程：

1. 递归收集 `fruit_data/` 下的图片和 XML；
2. 按文件名 stem 配对图片与 XML；
3. 读取 XML 的 `<size>`；若宽/高为 0，则从图片读取真实尺寸；
4. 对每个 `<object>` 读取类别和 `bndbox`，转换为 YOLO 归一化格式：
   ```text
   class_id x_center y_center width height
   ```
5. 按原始 `train/val/test` 目录分配 split，复制图片并写 `.txt`；
6. 生成 `yolo_fruit_data/data.yaml`（`nc=3`，类别名 apple/banana/orange）。

转换结果：

| split | 图片 | 标签 |
| --- | --- | --- |
| train | 208（v1）→ 166（v2） | 同图片数 |
| val | 32 | 32 |
| test | 60 | 60 |

> 脚本中的 `RAW_DATA_DIR` / `OUTPUT_DIR` 以脚本自身位置为基准，分别指向仓库根目录下的
> `fruit_data/` 和 `yolo_fruit_data/`，clone 到任何路径都不用改；生成的 `data.yaml` 只写
> 相对的 split 路径。原始数据不在 Git 中，需要本地放置 `fruit_data/`，或直接使用 DVC 中已版本化的结果。

## 3. 数据版本（DVC）

DVC 远程：

```text
remote: myazure
url:    azure://dvc
account: wut2026 / container dvc
auth:   .dvc/config.local（本地文件，含 SAS token，不提交）
```

Git 中只保存指针：

- `yolo_fruit_data/images.dvc` — md5 `66b815f2be91282b418d4ecb66fd161e.dir`，258 个文件
- `yolo_fruit_data/labels.dvc` — md5 `1bbab709fc3581b17d8349a1ffe5d1fd.dir`，260 个文件

版本对照：

| 版本 | Git tag | 对应 commit | 训练集 | 验证集 | 变化 |
| --- | --- | --- | --- | --- | --- |
| v1 | `data-v1` | `b13a344` | 208 | 32 | 初始可用数据集 |
| v2 | `data-v2` | `bad7453` | 166 | 32 | 剔除 42 张最短边 < 320px 的低分辨率训练图 |

验证集和测试集在两个版本之间保持不变，因此 v1/v2 指标可以比较。

## 4. v1 → v2 清洗

清洗脚本：`ml/clean_dataset.py`。

规则：

1. 没有对应 label 的图片；
2. label 为空；
3. label 行格式非法或坐标越界；
4. 图片无法打开；
5. bbox 面积占全图 < 0.1%；
6. 图片最短边 < 320px；
7. 图片内容重复（md5）。

v2 实际移除：42 张，原因均为“分辨率过低”。移除文件备份在
`yolo_fruit_data/_removed/`（不进入 Git/DVC）。

## 5. 代码与数据的对应关系

生产模型 `models/best.pt` 来自 `improved-v2` run：

- 数据版本：`v2`
- DVC tag：`data-v2`
- 训练代码 commit：`bad7453`
- MLflow run：`cc9c5339934f4e4c8b83b6927d49fbfd`
- 权重提升到 `models/best.pt` 的 commit：`0177703`

训练脚本会把 `dataset_version`（v1/v2）和 `git_commit` 写入 MLflow tags，因此可以从 run 反查代码和数据版本。

## 6. 复现

```bash
# 当前 tag 对应数据
dvc pull

# 切到 v1
git checkout data-v1
dvc pull

# 切到 v2
git checkout data-v2
dvc pull
```

训练/评估：

```bash
python ml/train.py --mode train --epochs 100 --dataset-version v2
python ml/train.py --mode eval
```

> `yolo_fruit_data/data.yaml` 不再写 `path`，YOLO 会以该文件所在目录作为数据集根，
> 因此仓库 clone 到任意路径都能直接训练；只有早期脚本 `xml_to_text.py` 仍保留旧机器绝对路径。

## 7. 已知问题

- 原始数据没有公开来源 URL / 许可证记录；
- labels 指针包含 260 个文件，比 images 的 258 个多 2 个，可能是历史遗留标签文件；训练时以图片配对为准。
