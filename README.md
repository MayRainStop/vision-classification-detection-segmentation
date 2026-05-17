# Homework 2: Vision Classification, Detection and Segmentation

本仓库是《深度学习与空间智能》`Homework 2` 的本地实现与实验结果整理版本。项目围绕三个视觉任务展开：基于 `Oxford 102 Flowers` 的细粒度花卉图像分类、基于 `Road Vehicle Images Dataset` 的道路车辆检测与视频跟踪计数，以及基于 `Stanford Background / ICCV09` 风格数据的语义分割与损失函数对比。

## 项目内容

- 完成花卉图像分类实验，比较 `ResNet` baseline、`ImageNet` 预训练、`SE/CBAM` 注意力模块、`ViT/Swin` 和 `ConvNeXt-Tiny` 等模型。
- 完成道路车辆检测实验，使用 `YOLOv8s` 微调多类别车辆检测器，并保存训练曲线、PR/F1 曲线、混淆矩阵和验证集预测样例。
- 完成视频跟踪与越线计数实验，基于 `YOLOv8s + ByteTrack` 输出 tracking ID、bounding box 和累计越线数量，并分析遮挡造成的 ID switch。
- 完成语义分割实验，使用轻量 `U-Net` 比较 `Cross-Entropy Loss`、`Dice Loss` 和 `Cross-Entropy + Dice`，并整理 mIoU、pixel accuracy、per-class IoU、混淆矩阵和样例预测图。
- 整理三项任务的源代码、实验结果、可视化图像和课程报告，便于复现实验与检查提交材料。

## 报告与提交链接

实验报告：
- `docs/report.pdf`
- GitHub Repo：`https://github.com/MayRainStop/vision-classification-detection-segmentation`
- 模型权重与大文件下载地址：`TODO`

## 仓库目录结构

PS: 目录中不包含放在网盘的大模型权重、原始大数据集和视频文件。

```text
.
|- docs/
|  |- assignment/
|  |- report.tex
|  `- report.pdf
|- tasks/
|  |- task1_flower_classification/
|  |  |- src/flower_classification/
|  |  |- tests/
|  |  |- scripts/
|  |  `- experiment_results/
|  |- task2_road_vehicle_detection/
|  |  |- YOLOtrain.py
|  |  |- YOLOcount.py
|  |  |- data.yaml
|  |  |- train_results/
|  |  |- report_figures/
|  |  `- screenshots/
|  `- task3_semantic_segmentation/
|     |- src/segmentation/
|     |- train.py
|     |- compare_losses.py
|     |- predict.py
|     |- report_figures/
|     `- runs/loss_comparison/
|- pyproject.toml
|- requirements.txt
|- README.md
`- .gitignore
```

各目录和文件说明如下：

- `docs/report.tex`、`docs/report.pdf`：实验报告源码与编译后的 PDF。
- `docs/assignment/`：作业原始要求文件。
- `tasks/task1_flower_classification/`：花卉分类任务代码、测试、服务器脚本和实验结果。
- `tasks/task1_flower_classification/src/flower_classification/`：数据读取、模型定义、训练逻辑和网格搜索代码。
- `tasks/task1_flower_classification/experiment_results/final_report/`：任务一最终报告使用的 CSV/JSON 汇总结果、分析图和错分分析文件。
- `tasks/task2_road_vehicle_detection/`：车辆检测、跟踪和越线计数任务代码与可视化结果。
- `tasks/task2_road_vehicle_detection/train_results/`：`YOLOv8s` 训练曲线、验证结果、PR/F1 曲线和混淆矩阵；其中 `weights/` 权重目录不放入 GitHub。
- `tasks/task2_road_vehicle_detection/report_figures/`：车辆检测数据集样例和类别分布图。
- `tasks/task2_road_vehicle_detection/screenshots/`：遮挡和 ID switch 分析所用关键帧。
- `tasks/task3_semantic_segmentation/`：语义分割任务代码、损失函数对比实验和结果整理。
- `tasks/task3_semantic_segmentation/src/segmentation/`：U-Net、数据集读取、损失函数和指标实现。
- `tasks/task3_semantic_segmentation/runs/loss_comparison/`：三种损失函数的训练曲线、mIoU 对比、混淆矩阵和样例预测图；其中 `best.pt` / `last.pt` 不放入 GitHub。

## 环境依赖

建议使用 `Python 3.10+`。主要依赖如下：

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `ultralytics`
- `opencv-python`

如果需要重新编译实验报告，还需要安装：

- `TeX Live`
- `XeLaTeX`
- `latexmk`

可以使用如下命令安装 Python 依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

## 数据集和大文件放置方式

本仓库没有上传原始大数据、视频和模型权重。需要复现实验时，建议按如下方式放置：

```text
tasks/
|- task1_flower_classification/
|  `- data/
|     |- jpg/
|     |- imagelabels.mat
|     `- setid.mat
|- task2_road_vehicle_detection/
|  |- train/
|  |- valid/
|  |- test2.mp4
|  |- tracking_analysis.mp4
|  `- train_results/weights/
|     |- best.pt
|     `- last.pt
`- task3_semantic_segmentation/
   `- runs/loss_comparison/
      |- ce/
      |  |- best.pt
      |  `- last.pt
      |- dice/
      |  |- best.pt
      |  `- last.pt
      `- combo/
         |- best.pt
         `- last.pt
```

其中，`Oxford 102 Flowers` 和 `Stanford Background / ICCV09` 风格数据均可根据代码重新准备；任务二的道路车辆数据、输入视频、输出视频和训练权重建议从网盘下载后放回对应目录。

## 如何运行花卉分类实验

运行以下命令可训练最终 `ConvNeXt-Tiny 384px` 配置：

```bash
set PYTHONPATH=tasks\task1_flower_classification\src
python -m flower_classification.train ^
  --data-root tasks\task1_flower_classification\data ^
  --model convnext_tiny ^
  --epochs 200 ^
  --batch-size 16 ^
  --lr 0.0001 ^
  --weight-decay 0.00001 ^
  --label-smoothing 0.1 ^
  --early-stopping-patience 15 ^
  --image-size 384 ^
  --num-workers 4
```

运行单元测试：

```bash
set PYTHONPATH=tasks\task1_flower_classification\src
python -m unittest discover -s tasks\task1_flower_classification\tests
```

## 如何运行车辆检测与跟踪实验

训练 `YOLOv8s` 检测器：

```bash
cd tasks\task2_road_vehicle_detection
python YOLOtrain.py
```

使用训练好的权重对视频进行跟踪和越线计数：

```bash
python YOLOcount.py ^
  --video test2.mp4 ^
  --model train_results\weights\best.pt ^
  --output tracking_v8s_demo.mp4
```

如果没有配置 `WANDB_API_KEY`，训练脚本会关闭在线 W&B 日志，仅保留本地结果。

## 如何运行语义分割实验

运行以下命令可对比三种损失函数：

```bash
cd tasks\task3_semantic_segmentation
python compare_losses.py ^
  --task regions ^
  --epochs 100 ^
  --batch-size 8 ^
  --output-root runs/loss_comparison
```

也可以单独训练某一种损失函数：

```bash
python train.py --task regions --loss-mode ce --epochs 100 --batch-size 8 --output-dir runs/regions_ce
python train.py --task regions --loss-mode dice --epochs 100 --batch-size 8 --output-dir runs/regions_dice
python train.py --task regions --loss-mode combo --epochs 100 --batch-size 8 --output-dir runs/regions_combo
```

## 主要实验结果

三项任务的代表性结果如下：

```text
Task 1: ConvNeXt-Tiny 384px, 10 seeds
mean test accuracy = 95.60% ± 0.61pp
best single-seed test accuracy = 96.50%

Task 2: YOLOv8s road vehicle detection
precision = 0.667
recall = 0.412
mAP50 = 0.520
mAP50-95 = 0.304

Task 3: U-Net semantic segmentation
best loss = Dice Loss
best validation mIoU = 0.3055
```

主要结果文件说明如下：

- `tasks/task1_flower_classification/experiment_results/final_report/final_summary.json`：任务一最终结果汇总。
- `tasks/task1_flower_classification/experiment_results/final_report/convnext_tiny_384_best10_seed_results.csv`：最终 `ConvNeXt-Tiny 384px` 的 10 seed 稳定性结果。
- `tasks/task1_flower_classification/experiment_results/final_report/figures/`：任务一报告图像，包括模型对比、训练曲线、超参数热力图和错分样例。
- `tasks/task2_road_vehicle_detection/train_results/results.csv`：任务二检测训练和验证指标。
- `tasks/task2_road_vehicle_detection/train_results/results.png`：`YOLOv8s` 训练曲线。
- `tasks/task2_road_vehicle_detection/train_results/BoxPR_curve.png`、`BoxF1_curve.png`：检测阈值相关曲线。
- `tasks/task2_road_vehicle_detection/train_results/confusion_matrix.png`：车辆检测混淆矩阵。
- `tasks/task2_road_vehicle_detection/screenshots/`：视频跟踪遮挡分析关键帧。
- `tasks/task3_semantic_segmentation/runs/loss_comparison/comparison_summary.json`：三种损失函数结果汇总。
- `tasks/task3_semantic_segmentation/runs/loss_comparison/miou_loss_comparison.png`：三种损失函数的 mIoU 与 loss 对比。
- `tasks/task3_semantic_segmentation/runs/loss_comparison/confusion_matrix_comparison.png`：三种损失函数的混淆矩阵对比。
- `tasks/task3_semantic_segmentation/runs/loss_comparison/0000382_prediction_comparison.png`：样例预测对比图。

