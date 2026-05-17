# Oxford 102 Flowers Final Experiment Results

This folder consolidates the result files used by the final report. It keeps the original 144-run grid-search summary and adds the later CNN-focused ConvNeXt experiments, seed repeats, and resolution checks.

## Final Selected Result

| Model | Image Size | LR | WD | Batch | Label Smoothing | Seeds | Val Acc | Test Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 384 | 1e-4 | 1e-5 | 16 | 0.1 | 10 | 97.28% +/- 0.48pp | 95.60% +/- 0.61pp |

The strongest and most stable configuration is `convnext_tiny`, ImageNet pretrained, 384px input, `lr=1e-4`, `weight_decay=1e-5`, `batch_size=16`, and `label_smoothing=0.1`.

## Experiment Coverage

| Group | Completed Runs |
| --- | ---: |
| Original CNN grid | 128 |
| Original Transformer grid | 16 |
| Extended pretrained CNN grid | 120 |
| Extended CNN seed repeats | 3 |
| Supplementary repeats | 12 |
| ConvNeXt final checks | 10 |
| ConvNeXt 384px expansion | 14 |
| Total completed training runs | 303 |

## Key Comparisons

| Comparison | Runs | Test Acc |
| --- | ---: | ---: |
| ConvNeXt-Tiny 384px final config | 10 | 95.60% +/- 0.61pp |
| ConvNeXt-Tiny 224px stable config | 10 | 94.71% +/- 0.31pp |
| ConvNeXt-Tiny 384px tuning challenger (`lr=3e-4`, `wd=1e-5`) | 3 | 94.29% +/- 0.44pp |
| Swin-T previous best config repeat | 3 | 93.98% +/- 0.26pp |

## Files

- `final_summary.json`: machine-readable final summary used by README/report updates.
- `final_key_results.csv`: compact headline comparison table.
- `convnext_tiny_384_best10_seed_results.csv`: per-seed rows for the final selected 384px configuration.
- `stability_and_resolution_checks.csv`: seed-repeat and resolution-comparison statistics.
- `convnext384_error_analysis/`: seed-314 test-set predictions, confusion counts, top confused class pairs, and error-analysis summary.
- `figures/task1_convnext384_top_confusions.png`: most frequent test-set class-confusion pairs.
- `figures/task1_convnext384_misclassified_examples.png`: high-confidence misclassified test examples.
- `all_results_sorted.csv`, `top20_results.csv`, `best_by_model.csv`, `summary.json`: original 144-run grid-search summary retained for reproducibility.
- `best_checkpoints/`: selected checkpoints, including the final ConvNeXt-384 seed-314 checkpoint used for error analysis. Full per-trial checkpoints are intentionally not mirrored because they are large and not needed for the report tables.
