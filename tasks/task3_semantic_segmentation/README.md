# HW2 Task 3: U-Net Segmentation and Loss Engineering

本任务基于 `iccv09Data` 数据集，从零搭建 `U-Net` 图像分割训练工程，并比较三种损失函数在验证集上的表现：

- `Cross-Entropy Loss`
- 手写 `Dice Loss`
- `Cross-Entropy Loss + Dice Loss`

## 1. 任务目标

本实验对应 HW2 的 Task 3，核心要求包括：

- 从零实现 `U-Net` 分割模型
- 使用 Stanford Background / ICCV09 风格数据集完成像素级训练
- 手写 `Dice Loss`
- 对比 `Cross-Entropy`、`Dice` 和 `Cross-Entropy + Dice`
- 使用验证集 `mIoU` 作为主要评价指标

## 2. 项目结构

- `src/segmentation/`
  - `dataset.py`：数据集读取、训练/验证划分与预处理
  - `model.py`：手写 `U-Net`
  - `losses.py`：`Cross-Entropy`、`Dice` 与组合损失
  - `metrics.py`：`pixel accuracy`、`mean accuracy`、`mIoU`
- `train.py`
  - 单组训练入口
- `compare_losses.py`
  - 三组损失函数对比实验入口
- `predict.py`
  - 单张图像预测与可视化
- `runs/`
  - 保存训练日志、权重、曲线图和混淆矩阵
- `outputs/`
  - 保存样例预测图

## 3. 数据与标签说明

本实验使用 `labels/*.regions.txt` 作为语义分割标签，共 `8` 类：

1. `sky`
2. `tree`
3. `road`
4. `grass`
5. `water`
6. `building`
7. `mountain`
8. `foreground_object`

说明：

- 标签中相同数字表示同一语义类别
- 负数像素表示未知区域
- 训练时未知区域统一映射到 `ignore_index = 255`

## 4. 模型结构

模型采用手写 `U-Net`，包含：

- 编码器下采样路径
- 解码器上采样路径
- `Skip Connection`
- 最后 `1x1 Conv` 输出像素级分类结果

## 5. 损失函数设计

### 5.1 Cross-Entropy Loss

标准像素级分类损失，用于逐像素监督学习。

### 5.2 手写 Dice Loss

在 `src/segmentation/losses.py` 中手动实现多类 `Dice Loss`，通过优化预测区域与真实区域的重叠程度，缓解类别不平衡问题。

### 5.3 组合损失

组合损失形式为：

```text
L = L_ce + L_dice
```

其中：

- `L_ce` 保证像素分类稳定
- `L_dice` 强化区域重叠质量

## 6. 训练配置

当前正式实验使用：

- Model: `U-Net`
- Epochs: `100`
- Batch size: `8`
- Learning rate: `3e-4`
- Optimizer: `AdamW`
- Input size: `320 x 240`
- Task: `regions`
- Validation split: 随机抽取 `15%`
- Seed: `42`

## 7. 运行方式

### 单独训练三种损失函数

仅使用交叉熵：

```bash
python train.py --data-root iccv09Data --task regions --loss-mode ce --epochs 100 --batch-size 8 --val-ratio 0.15 --seed 42 --output-dir runs/ce --amp
```

仅使用 Dice：

```bash
python train.py --data-root iccv09Data --task regions --loss-mode dice --epochs 100 --batch-size 8 --val-ratio 0.15 --seed 42 --output-dir runs/dice --amp
```

组合损失：

```bash
python train.py --data-root iccv09Data --task regions --loss-mode combo --ce-weight 1.0 --dice-weight 1.0 --epochs 100 --batch-size 8 --val-ratio 0.15 --seed 42 --output-dir runs/combo --amp
```

### 绘制三组验证集 mIoU 对比图

结果保存在：

```text
runs/val_miou_comparison.png
```

### 单张图像预测

以样例图 `0000382.jpg` 为例：

```bash
python predict.py --checkpoint runs/combo/best.pt --image iccv09Data/images/0000382.jpg --output outputs/0000382_combo.png
```

## 8. 主要输出文件

- `runs/ce/`：Cross-Entropy 训练结果
- `runs/dice/`：Dice 训练结果
- `runs/combo/`：Cross-Entropy + Dice 训练结果
- `runs/val_miou_comparison.png`：三种损失函数验证集 `mIoU` 对比曲线
- `outputs/0000382_ce.png`：CE 模型在样例图上的预测结果
- `outputs/0000382_dice.png`：Dice 模型在样例图上的预测结果
- `outputs/0000382_combo.png`：组合损失模型在样例图上的预测结果

每个实验目录通常包含：

- `best.pt`
- `last.pt`
- `history.json`
- `history.csv`
- `training_curves.png`
- `best_confusion_matrix.png`

## 9. 实验结果

当前三组实验的最佳验证集结果如下：

| Loss | Best Epoch | Best Val mIoU | Best Val Pixel Acc |
|---|---:|---:|---:|
| Cross-Entropy | 58 | 0.6436 | 0.8432 |
| Dice | 83 | 0.6477 | 0.8446 |
| Cross-Entropy + Dice | 66 | 0.6531 | 0.8489 |



三种方法在各自最佳 epoch 下的各类别 IoU 与验证集 mIoU 如下：

| Method | sky | tree | road | grass | water | building | mountain | foreground_object | Val mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-Entropy | 0.8623 | 0.6661 | 0.8637 | 0.5477 | 0.6559 | 0.7384 | 0.2066 | 0.6077 | 0.6436 |
| Dice | 0.8655 | 0.6631 | 0.8602 | 0.5352 | 0.6490 | 0.7327 | 0.2670 | 0.6093 | 0.6477 |
| Cross-Entropy + Dice | 0.8745 | 0.6686 | 0.8654 | 0.5641 | 0.6605 | 0.7458 | 0.2333 | 0.6124 | 0.6531 |

## 10. 结论

- 三种损失函数都能够在该数据集上获得较好的分割效果
- 单独使用 `Dice Loss` 已明显优于单独使用 `Cross-Entropy`
- `Cross-Entropy + Dice` 取得了最高的验证集 `mIoU = 0.6531`
- 说明在该分割任务中，组合损失兼顾了像素级分类稳定性与区域重叠优化能力，整体表现最佳
