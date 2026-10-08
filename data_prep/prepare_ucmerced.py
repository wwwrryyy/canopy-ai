"""
prepare_ucmerced.py
===================
Prepares the UC Merced Land Use dataset for SNN training:
1. Extracts 128x128 patches from 256x256 aerial images (4 non-overlapping quadrants per image).
2. Converts to grayscale [0, 1] for Stage 1 (Despeckler) & Stage 2 (Autoencoder).
3. Preserves class folder structure for Stage 3 (Spiking Classifier).
"""

import os
import glob
import argparse
import cv2
import numpy as np


def prepare_ucmerced(raw_dir: str, out_dir: str, patch_size: int = 128):
    os.makedirs(out_dir, exist_ok=True)
    all_patches_dir = os.path.join(out_dir, "all_patches")
    agri_patches_dir = os.path.join(out_dir, "agricultural_patches")
    os.makedirs(all_patches_dir, exist_ok=True)
    os.makedirs(agri_patches_dir, exist_ok=True)

    class_dirs = sorted([d for d in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(d)])
    print(f"[*] Found {len(class_dirs)} classes in {raw_dir}")

    total_extracted = 0
    agri_extracted = 0

    for cdir in class_dirs:
        cname = os.path.basename(cdir)
        c_out_dir = os.path.join(out_dir, "by_class", cname)
        os.makedirs(c_out_dir, exist_ok=True)

        tif_files = sorted(glob.glob(os.path.join(cdir, "*.tif")))
        for fpath in tif_files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            base_name = os.path.splitext(os.path.basename(fpath))[0]

            # Extract 4 non-overlapping 128x128 quadrants
            q_idx = 0
            for r in range(0, h - patch_size + 1, patch_size):
                for c in range(0, w - patch_size + 1, patch_size):
                    patch = img[r : r + patch_size, c : c + patch_size]
                    fname = f"{base_name}_q{q_idx}.png"

                    # 1. Save to class folder
                    cv2.imwrite(os.path.join(c_out_dir, fname), patch)

                    # 2. Save to unified patches pool
                    cv2.imwrite(os.path.join(all_patches_dir, fname), patch)

                    # 3. If agricultural/forest/chaparral, save to agri pool
                    if cname in ["agricultural", "forest", "chaparral", "river"]:
                        cv2.imwrite(os.path.join(agri_patches_dir, fname), patch)
                        agri_extracted += 1

                    total_extracted += 1
                    q_idx += 1

    print(f"[+] Extraction complete!")
    print(f"    Total patches: {total_extracted} -> {all_patches_dir}")
    print(f"    Agricultural/natural patches: {agri_extracted} -> {agri_patches_dir}")
    print(f"    Class-stratified patches -> {os.path.join(out_dir, 'by_class')}")
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Prepare UC Merced Land Use dataset for SNN training")
    parser.add_argument("--raw-dir", type=str, default="../UCMerced_LandUse/Images",
                        help="Path to UCMerced LandUse Images folder")
    parser.add_argument("--out-dir", type=str, default="../data/ucmerced_prepared",
                        help="Output directory for processed patches")
    parser.add_argument("--patch-size", type=int, default=128)
    args = parser.parse_args()

    prepare_ucmerced(
        raw_dir=os.path.abspath(args.raw_dir),
        out_dir=os.path.abspath(args.out_dir),
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
