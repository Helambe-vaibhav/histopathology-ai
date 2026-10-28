# MoNuSeg Histopathology Dataset

The raw dataset images, XML annotations, binary masks, and zero-leakage pre-extracted tiles are excluded from Git version control to keep the repository lightweight.

## Setting Up Dataset

You can easily set up and structure the MoNuSeg dataset using the automated script:

```bash
# Option 1: Using downloaded archive from Kaggle or Grand Challenge
python scripts/setup_monuseg_data.py --archive /path/to/monuseg.zip

# Option 2: Pointing to an extracted directory
python scripts/setup_monuseg_data.py --source /path/to/extracted/folder
```

This will automatically create the required directory structure:
```
data/
└── MoNuSeg/
    ├── images/        # .tif slide crops
    ├── annotations/   # .xml boundary annotations
    ├── masks/         # Binary ground-truth masks
    └── tiles/         # 256x256 normalized tiles (50% overlap, stride 128)
```
