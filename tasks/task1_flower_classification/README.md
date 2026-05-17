# HW2 Task 1: Oxford 102 Flowers Classification

This folder contains the full Task 1 flower-classification project: source code, tests, AutoDL scripts, local dataset files, and downloaded experiment results.

## HW2 Requirement Alignment

| HW2 Task 1 requirement | Implementation in this folder |
| --- | --- |
| Use the 102 Category Flower Dataset | Uses Oxford 102 Flowers with the official train/validation/test split from `data/setid.mat` and labels from `data/imagelabels.mat`. |
| Train CNN baselines such as ResNet-18 / ResNet-34 | ResNet-18 and ResNet-34 baselines are implemented in `src/flower_classification/models.py` and covered in the original grid. |
| Compare ImageNet pretraining and training from scratch | Both pretrained and scratch ResNet-18/34 runs are included in `experiment_results/final_report/`. |
| Add SE-block and CBAM attention modules | ResNet-18/34-SE and ResNet-18/34-CBAM variants are implemented and evaluated. |
| Try ViT/Swin style Transformer models | ViT-B/16 and Swin-T are included as comparison models. |
| Report accuracy and hyperparameter analysis | Validation/test accuracy, learning rate, weight decay, batch size, label smoothing, input size, and seed-repeat stability are summarized in the root report. |

## Layout

```text
task1_flower_classification/
|- README.md
|- data/
|  |- 102flowers.tgz
|  |- jpg/
|  |- imagelabels.mat
|  `- setid.mat
|- src/flower_classification/
|  |- data.py
|  |- models.py
|  |- train.py
|  `- grid_search.py
|- tests/
|- scripts/
`- experiment_results/
   `- final_report/
```

## Final Result

| Model | Image Size | LR | Weight Decay | Batch | Label Smoothing | Seeds | Val Acc | Test Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 384 | `1e-4` | `1e-5` | 16 | 0.1 | 10 | 97.28% +/- 0.48pp | 95.60% +/- 0.61pp |

The best single seed in the final group is seed `314`, with 96.50% test accuracy. The report uses the 10-seed mean as the headline result.

## Run

From the repository root:

```powershell
$env:PYTHONPATH = "tasks\task1_flower_classification\src"
python -m flower_classification.train --data-root tasks\task1_flower_classification\data --model convnext_tiny --epochs 200 --batch-size 16 --lr 0.0001 --weight-decay 0.00001 --label-smoothing 0.1 --early-stopping-patience 15 --image-size 384 --num-workers 4
```

Run unit tests:

```powershell
$env:PYTHONPATH = "tasks\task1_flower_classification\src"
python -m unittest discover -s tasks\task1_flower_classification\tests
```

If `data/jpg/` is missing, extract the archive from this folder:

```powershell
tar -xzf tasks\task1_flower_classification\data\102flowers.tgz -C tasks\task1_flower_classification\data
```

## Key Result Files

- `experiment_results/final_report/final_summary.json`
- `experiment_results/final_report/final_key_results.csv`
- `experiment_results/final_report/convnext_tiny_384_best10_seed_results.csv`
- `experiment_results/final_report/stability_and_resolution_checks.csv`
- `experiment_results/final_report/convnext384_error_analysis/`
- `experiment_results/final_report/figures/task1_convnext384_top_confusions.png`
- `experiment_results/final_report/figures/task1_convnext384_misclassified_examples.png`
