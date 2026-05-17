# 深度学习与空间智能 HW2

这是一个可直接作为 GitHub 仓库根目录使用的 HW2 总项目。根目录只保留总说明、依赖、项目配置和总报告；三个任务全部并列放在 `tasks/` 下，避免 Task 1 散落在最外层。

## Repository Layout

```text
.
|- README.md
|- requirements.txt
|- pyproject.toml
|- .gitignore
|- docs/
|  |- report.tex
|  |- report.pdf
|  `- assignment/
|     `- HW2_深度学习与空间智能.pdf
`- tasks/
   |- task1_flower_classification/
   |  |- README.md
   |  |- src/flower_classification/
   |  |- tests/
   |  |- scripts/
   |  `- experiment_results/final_report/
   |- task2_road_vehicle_detection/
   |  |- README.md
   |  |- HW2_Task2_Road_Vehicle_Detection.ipynb
   |  |- YOLOtrain.py
   |  |- YOLOcount.py
   |  |- data.yaml
   |  |- train_results/
   |  |- report_figures/
   |  `- screenshots/
   `- task3_semantic_segmentation/
      |- README.md
      |- report.md
      |- train.py
      |- compare_losses.py
      |- predict.py
      |- src/segmentation/
      |- report_figures/
      `- runs/loss_comparison/
```

## HW2 Tasks

| Task | HW2 requirement keywords | Folder | Headline result |
| --- | --- | --- | --- |
| Task 1 | 102 Category Flower Dataset, CNN baseline, ImageNet pretraining, SE/CBAM, ViT/Swin, Accuracy | `tasks/task1_flower_classification/` | ConvNeXt-Tiny 384px, 10 seeds: Test Acc. `95.60% +/- 0.61pp` |
| Task 2 | Road Vehicle Images Dataset, YOLOv8, Bounding Box, ID / Tracking ID, ID-switch analysis | `tasks/task2_road_vehicle_detection/` | YOLOv8s: P `0.667`, R `0.412`, mAP50 `0.520`, mAP50-95 `0.304` |
| Task 3 | Stanford Background Dataset, U-Net, Skip Connection, Dice Loss, mIoU | `tasks/task3_semantic_segmentation/` | Dice Loss best Val mIoU `0.3055` |

## Submission Links

- GitHub repository: https://github.com/MayRainStop/vision-classification-detection-segmentation
- Model weights / large artifacts cloud drive: TODO

## GitHub Submission Contents

This folder is organized as a GitHub-ready repository. The GitHub upload should include the reproducible code, report, small result summaries, and visual evidence:

- Root project files: `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`
- Final report: `docs/report.tex`, `docs/report.pdf`
- HW2 requirement PDF, if needed for review: `docs/assignment/HW2_深度学习与空间智能.pdf`
- Task 1 code and analysis assets: `tasks/task1_flower_classification/src/`, `tests/`, `scripts/`, `README.md`, and the small CSV/JSON/PNG files under `experiment_results/final_report/`
- Task 2 code and analysis assets: `YOLOtrain.py`, `YOLOcount.py`, `data.yaml`, `HW2_Task2_Road_Vehicle_Detection.ipynb`, `README.md`, `report_figures/`, `screenshots/`, and non-weight files in `train_results/`
- Task 3 code and analysis assets: `train.py`, `compare_losses.py`, `predict.py`, `src/segmentation/`, `report.md`, `README.md`, `report_figures/`, and non-checkpoint CSV/JSON/PNG files in `runs/loss_comparison/`

## Cloud Drive Contents

Do not commit large datasets, model checkpoints, or videos to GitHub. Put them in the cloud drive linked above. A recommended cloud-drive structure outside the GitHub repository is:

```text
HW2_large_artifacts/
|- task1_flower_classification/
|  `- best_checkpoints/
|     |- convnext_tiny_384_best_seed314_lr0p0001_wd1e-05_bs16_ls0p1_best.pt
|     |- best_overall_swin_t_pretrained_lr0p0001_wd0p0001_bs16_ls0p1.pt
|     |- best_scratch_resnet34_scratch_lr0p0003_wd0p0001_bs32_ls0p1.pt
|     `- best_cnn_resnet18_cbam_pretrained_lr0p0003_wd0p0001_bs64_ls0p1.pt
|- task2_road_vehicle_detection/
|  |- weights/
|  |  |- best.pt
|  |  `- last.pt
|  |- videos/
|  |  |- test2.mp4
|  |  `- tracking_analysis.mp4
|  `- dataset/
|     |- train/
|     `- valid/
`- task3_semantic_segmentation/
   `- checkpoints/
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

The raw Oxford 102 Flowers and Stanford Background datasets are public and can be regenerated/downloaded by the code, so they do not need to be uploaded unless the course platform explicitly asks for full local data. The Task 2 road-vehicle training/validation split is less standard, so it is safer to keep it in the cloud drive together with the trained YOLOv8s weights and input/output videos.

## Not Submitted

Local cache and build byproducts should stay out of both GitHub and the cloud-drive package unless specifically requested: `__pycache__/`, `.pytest_cache/`, LaTeX auxiliary files, W&B local cache folders, temporary folders, and interrupted training scratch outputs. These patterns are covered by `.gitignore` where appropriate.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Task 1: Flower Classification

```powershell
$env:PYTHONPATH = "tasks\task1_flower_classification\src"
python -m flower_classification.train --data-root tasks\task1_flower_classification\data --model convnext_tiny --epochs 200 --batch-size 16 --lr 0.0001 --weight-decay 0.00001 --label-smoothing 0.1 --early-stopping-patience 15 --image-size 384 --num-workers 4
```

Tests:

```powershell
$env:PYTHONPATH = "tasks\task1_flower_classification\src"
python -m unittest discover -s tasks\task1_flower_classification\tests
```

## Task 2: Vehicle Detection and Tracking

```powershell
cd tasks\task2_road_vehicle_detection
python YOLOtrain.py
python YOLOcount.py --video test2.mp4 --model train_results\weights\best.pt --output tracking_v8s_demo.mp4
```

Online W&B logging is optional. Set `WANDB_API_KEY` before training if needed; otherwise Task 2 training disables W&B network logging.

## Task 3: Semantic Segmentation

```powershell
cd tasks\task3_semantic_segmentation
python compare_losses.py --task regions --epochs 100 --batch-size 8 --output-root runs/loss_comparison
```

Single-loss runs:

```powershell
python train.py --task regions --loss-mode ce --epochs 100 --batch-size 8 --output-dir runs/regions_ce
python train.py --task regions --loss-mode dice --epochs 100 --batch-size 8 --output-dir runs/regions_dice
python train.py --task regions --loss-mode combo --epochs 100 --batch-size 8 --output-dir runs/regions_combo
```

## Report

- Source: `docs/report.tex`
- PDF: `docs/report.pdf`

Compile:

```powershell
cd docs
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
latexmk -c report.tex
```

## Key Results

Task 1:

- `tasks/task1_flower_classification/experiment_results/final_report/final_summary.json`
- `tasks/task1_flower_classification/experiment_results/final_report/final_key_results.csv`
- `tasks/task1_flower_classification/experiment_results/final_report/convnext_tiny_384_best10_seed_results.csv`

Task 2:

- `tasks/task2_road_vehicle_detection/train_results/results.csv`
- `tasks/task2_road_vehicle_detection/train_results/results.png`
- `tasks/task2_road_vehicle_detection/report_figures/detection_dataset_samples.png`
- `tasks/task2_road_vehicle_detection/report_figures/detection_class_distribution.png`
- `tasks/task2_road_vehicle_detection/train_results/confusion_matrix.png`
- `tasks/task2_road_vehicle_detection/screenshots/frame_378.jpg`
- `tasks/task2_road_vehicle_detection/screenshots/frame_424.jpg`
- `tasks/task2_road_vehicle_detection/screenshots/frame_440.jpg`

Task 3:

- `tasks/task3_semantic_segmentation/runs/loss_comparison/comparison_summary.json`
- `tasks/task3_semantic_segmentation/report_figures/segmentation_dataset_label_example.png`
- `tasks/task3_semantic_segmentation/runs/loss_comparison/miou_loss_comparison.png`
- `tasks/task3_semantic_segmentation/runs/loss_comparison/confusion_matrix_comparison.png`
- `tasks/task3_semantic_segmentation/runs/loss_comparison/0000382_prediction_comparison.png`
