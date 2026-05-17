# Best Checkpoints

This local folder contains the selected best checkpoints downloaded from AutoDL:

- Best overall: `best_overall_swin_t_pretrained_lr0p0001_wd0p0001_bs16_ls0p1.pt`
- Best CNN: `best_cnn_resnet18_cbam_pretrained_lr0p0003_wd0p0001_bs64_ls0p1.pt`
- Best from scratch: `best_scratch_resnet34_scratch_lr0p0003_wd0p0001_bs32_ls0p1.pt`
- Final ConvNeXt-384 analysis checkpoint: `convnext_tiny_384_best_seed314_lr0p0001_wd1e-05_bs16_ls0p1_best.pt`

The `.pt` files are intentionally ignored by git because standard GitHub blocks files larger than 100 MB. Use Git LFS or an external release artifact if the checkpoints need to be published.
