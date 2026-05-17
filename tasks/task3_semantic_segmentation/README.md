# HW2 Task 3: U-Net Segmentation and Loss Engineering

从零搭建图像分割训练工程，在 `iccv09Data` 上完成像素级语义分割训练，并比较三种损失函数配置在验证集上的 `mIoU` 表现：

- `Cross-Entropy Loss`
- 手写 `Dice Loss`
- `Cross-Entropy Loss + Dice Loss`

## 1. HW2 任务要求对齐

| HW2 Task 3 要求 | 本目录中的实现 |
|---|---|
| 从零搭建图像分割训练工程 / API | `train.py`、`compare_losses.py`、`predict.py` 与 `src/segmentation/` 组成完整训练、对比和推理入口。 |
| 实现带 Skip Connection 的 `U-Net` | `src/segmentation/model.py` 中手写轻量 U-Net 编码器、解码器与跳跃连接。 |
| 使用 Stanford Background Dataset | 数据读取逻辑面向 `iccv09Data` / Stanford Background 风格目录。 |
| 手写 `Dice Loss` | `src/segmentation/losses.py` 中实现多类 `SoftDiceLoss`，并支持 `ce`、`dice`、`combo` 三种模式。 |
| 使用 `mIoU` 比较损失函数 | `runs/loss_comparison/comparison_summary.json` 保存三组验证集 `mIoU`、pixel accuracy 和 per-class IoU。 |

## 2. 项目结构

- `iccv09Data/`
  - Stanford Background / ICCV09 风格数据
- `src/segmentation/`
  - `dataset.py`：数据读取、训练/验证划分、类别权重
  - `model.py`：手写轻量 `U-Net`
  - `losses.py`：手写 `SoftDiceLoss` 与组合损失
  - `metrics.py`：`pixel accuracy`、`mIoU`、`per-class IoU`
- `train.py`
  - 单组训练入口
- `compare_losses.py`
  - 三种损失配置的一键对比实验入口
- `predict.py`
  - 单张图像预测与可视化
- `runs/`
  - 所有训练结果、曲线图、混淆矩阵和预测对比图
- `report_figures/`
  - 报告中使用的数据集与标签样例图
- `report.md`
  - 实验报告

## 3. 数据与任务定义

本次实验使用 `labels/*.regions.txt` 作为语义分割标签，共 `8` 类：

1. `sky`
2. `tree`
3. `road`
4. `grass`
5. `water`
6. `building`
7. `mountain`
8. `foreground_object`

说明：

- 标签中的负数像素表示未知区域
- 训练时统一映射为 `ignore_index = 255`
- 模型不会对未知像素计算损失

## 4. 模型设计

模型采用手写轻量 `U-Net`，包含：

- 编码器下采样路径
- 解码器上采样路径
- `Skip Connection`
- 最终 `1x1 Conv` 输出像素级类别预测

## 5. 损失函数工程

### 5.1 Cross-Entropy Loss

标准像素级分类损失，用于逐像素监督学习。

### 5.2 手写 Dice Loss

已在 `src/segmentation/losses.py` 中手动实现 `SoftDiceLoss`，核心思想是直接优化预测区域与真实区域的重叠程度，缓解前景/背景不平衡问题。

### 5.3 组合损失

组合损失形式为：

```text
L = L_ce + L_dice
```

其中：

- `L_ce` 负责稳定像素分类
- `L_dice` 负责提高区域重叠质量

## 6. 训练配置

正式对比实验保存在：

- [runs/loss_comparison](runs/loss_comparison)

实验设置：

- Model: `U-Net`
- Epochs: `100`
- Batch size: `8`
- Learning rate: `3e-4`
- Optimizer: `AdamW`
- Input size: `256 x 256`
- Task: `regions`
- Seed: `42`

## 7. 运行方式

### 单组训练

只用交叉熵：

```bash
python train.py --task regions --loss-mode ce --epochs 100 --batch-size 8 --output-dir runs/regions_ce
```

只用 Dice：

```bash
python train.py --task regions --loss-mode dice --epochs 100 --batch-size 8 --output-dir runs/regions_dice
```

组合损失：

```bash
python train.py --task regions --loss-mode combo --epochs 100 --batch-size 8 --output-dir runs/regions_combo
```

### 一键完成三组对比

```bash
python compare_losses.py --task regions --epochs 100 --batch-size 8 --output-root runs/loss_comparison
```

## 8. 输出结果位置

正式实验的主要结果位于：

- [runs/loss_comparison/comparison_summary.json](runs/loss_comparison/comparison_summary.json)
- [report_figures/segmentation_dataset_label_example.png](report_figures/segmentation_dataset_label_example.png)
- [runs/loss_comparison/miou_loss_comparison.png](runs/loss_comparison/miou_loss_comparison.png)
- [runs/loss_comparison/confusion_matrix_comparison.png](runs/loss_comparison/confusion_matrix_comparison.png)
- [runs/loss_comparison/0000382_prediction_comparison.png](runs/loss_comparison/0000382_prediction_comparison.png)

每组实验各自还包含：

- `best.pt`
- `last.pt`
- `history.json`
- `training_curves.png`
- `best_confusion_matrix.png`

## 9. 实验结论

根据当前 `runs/loss_comparison` 中的正式结果：

| Loss | Best Epoch | Best Val mIoU | Best Val Pixel Acc |
|---|---:|---:|---:|
| Cross-Entropy | 90 | 0.2587 | 0.7667 |
| Dice | 72 | 0.3055 | 0.7671 |
| Cross-Entropy + Dice | 90 | 0.3027 | 0.7748 |

结论：

- `Dice Loss` 在本次实验中取得了最高的验证集 `mIoU`
- `CE + Dice` 的整体表现也明显优于仅使用 `Cross-Entropy`
- 这说明在前景/背景像素不平衡的分割任务中，`Dice Loss` 对区域重叠质量更敏感，更有利于提升分割表现
