"""Debug tool: inspect unique values in an exported instance-mask PNG.

Usage:
    python -m src.tools.check_mask [mask_dir]
Default mask_dir: output_dataset/multiview/mask
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

mask_dir = sys.argv[1] if len(sys.argv) > 1 else "output_dataset/multiview/mask"
mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
if mask_files:
    # Check the first mask file
    img = Image.open(mask_files[0])
    arr = np.array(img)
    print(f"Dir: {mask_dir}")
    print(f"File: {os.path.basename(mask_files[0])}")
    print(f"Unique values in Red channel (Semantic Class IDs): {np.unique(arr[:, :, 0])}")
    print(f"Unique values in Green channel (Instance IDs): {np.unique(arr[:, :, 1])}")
else:
    print(f"No mask files found in: {mask_dir}")
