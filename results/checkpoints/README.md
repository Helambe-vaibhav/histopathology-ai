# Model Checkpoints

PyTorch `.pth` checkpoint weights (>120 MB each) are excluded from Git to prevent large binary blobs in the repository.

To generate the trained checkpoints locally:

```bash
# Retrain and evaluate full pipeline
bash scripts/retrain_and_evaluate_full_pipeline.sh

# Or launch 100-epoch production run
bash launch_retrain_100ep.sh
```

Trained checkpoints will be saved here:
- `results/checkpoints/unet_best.pth`
- `results/checkpoints/m3_aug_best.pth`
- `results/checkpoints/m4_combined_best.pth`
