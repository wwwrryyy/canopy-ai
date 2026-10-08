"""
extract_patches.py
==================
Extracts fixed-dimension (e.g. 128x128) texture patches from high-resolution
raw dataset images (such as Kylberg Texture Dataset) for training SNNs.
"""

import os
import glob
import argparse
import cv2
import numpy as np


def extract_patches(raw_dir: str, out_dir: str, patch_size: int = 128, patches_per_img: int = 20, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    print(f"[*] Found {len(raw_files)} raw images in {raw_dir}")

    total_extracted = 0
    skipped = 0

    for idx, fpath in enumerate(raw_files):
        # Skip truncated / empty files
        if os.path.getsize(fpath) < 1_000_000:
            skipped += 1
            continue

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape
        if h < patch_size or w < patch_size:
            skipped += 1
            continue

        base_name = os.path.splitext(os.path.basename(fpath))[0]

        for p_idx in range(patches_per_img):
            top = rng.integers(0, h - patch_size + 1)
            left = rng.integers(0, w - patch_size + 1)
            patch = img[top : top + patch_size, left : left + patch_size]

            patch_fname = f"{base_name}_p{p_idx:03d}.png"
            patch_path = os.path.join(out_dir, patch_fname)
            cv2.imwrite(patch_path, patch)
            total_extracted += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(raw_files):
            print(f"    Processed [{idx + 1}/{len(raw_files)}] images -> {total_extracted} patches saved")

    print(f"[+] Patch extraction complete! Saved {total_extracted} patches to {out_dir} (skipped {skipped} corrupt/empty images).")
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract patches from raw texture images")
    parser.add_argument("--raw-dir", type=str, default="../KylbergTextureDataset-1.0-png-originals",
                        help="Path to raw high-resolution images")
    parser.add_argument("--out-dir", type=str, default="../data/patches",
                        help="Output directory for extracted patches")
    parser.add_argument("--patch-size", type=int, default=128, help="Square patch dimensions (HxW)")
    parser.add_argument("--patches-per-image", type=int, default=20, help="Number of random crops per image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_patches(
        raw_dir=os.path.abspath(args.raw_dir),
        out_dir=os.path.abspath(args.out_dir),
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_image,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
