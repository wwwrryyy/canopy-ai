"""
prepare_external_pathologies.py
================================
Prepares standardized 64x64 agricultural pathology dataset in data/external_pathologies/:
- Class 0: Healthy Canopy (real potato plants from DatasetNinja + UCMerced agriculture)
- Class 1: Fungal/Bacterial Disease (synthetic necrotic lesions on healthy canopy)
- Class 2: Pest Infestation (synthetic chewed defoliation holes & stippling)
- Class 3: Water Stress (real stressed potato plants from DatasetNinja)
- Class 4: Weed Invasion (real weed crops from WeedsGalore UAV dataset)
"""

import os
import sys
import glob
import json
import argparse
from typing import Optional, Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "models", "classifier")
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

from models.classifier.classifier import CLASS_NAMES
from models.classifier.dataset import AgriculturalPathologyDataset


def ensure_class_dirs(output_dir: str):
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def extract_potato_plants(
    potato_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (64, 64),
    max_per_class: int = 1200,
) -> Tuple[int, int]:
    """
    Extracts annotated plant bounding boxes from multispectral-potato-plants-images-DatasetNinja.
    - 'healthy' -> Class 0: Healthy Canopy
    - 'stressed' -> Class 3: Water Stress
    """
    healthy_dir = os.path.join(output_dir, "Healthy Canopy")
    stressed_dir = os.path.join(output_dir, "Water Stress")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(stressed_dir, exist_ok=True)

    healthy_count = 0
    stressed_count = 0

    for split in ("train", "test"):
        split_dir = os.path.join(potato_dir, split)
        ann_dir = os.path.join(split_dir, "ann")
        img_dir = os.path.join(split_dir, "img")
        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            continue

        ann_files = sorted(glob.glob(os.path.join(ann_dir, "Image_*.json")))
        for af in ann_files:
            base_name = os.path.basename(af)
            img_name = base_name[:-5]  # strip '.json'
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            try:
                with open(af, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
            except Exception:
                continue

            for idx, obj in enumerate(ann_data.get("objects", [])):
                cls_title = obj.get("classTitle", "").lower()
                if cls_title not in ("healthy", "stressed"):
                    continue

                if cls_title == "healthy" and healthy_count >= max_per_class:
                    continue
                if cls_title == "stressed" and stressed_count >= max_per_class:
                    continue

                exterior = obj.get("points", {}).get("exterior", [])
                if len(exterior) < 2:
                    continue

                x1, y1 = int(round(exterior[0][0])), int(round(exterior[0][1]))
                x2, y2 = int(round(exterior[1][0])), int(round(exterior[1][1]))

                # Clamp bounding box
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                box_w = x2 - x1
                box_h = y2 - y1
                # Filter degenerate or minuscule crops
                if box_w < 12 or box_h < 12:
                    continue

                crop = img[y1:y2, x1:x2]
                interp = cv2.INTER_AREA if (box_w > target_size[1] or box_h > target_size[0]) else cv2.INTER_LINEAR
                resized = cv2.resize(crop, target_size, interpolation=interp)

                img_id = os.path.splitext(img_name)[0]
                if cls_title == "healthy":
                    out_path = os.path.join(healthy_dir, f"potato_{split}_{img_id}_{idx:02d}.png")
                    cv2.imwrite(out_path, resized)
                    healthy_count += 1
                else:
                    out_path = os.path.join(stressed_dir, f"potato_{split}_{img_id}_{idx:02d}.png")
                    cv2.imwrite(out_path, resized)
                    stressed_count += 1

                if healthy_count >= max_per_class and stressed_count >= max_per_class:
                    break

    print(f"[+] Extracted potato plants: {healthy_count} Healthy Canopy, {stressed_count} Water Stress")
    return healthy_count, stressed_count


def extract_weedsgalore(
    weedsgalore_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (64, 64),
    max_samples: int = 1200,
) -> int:
    """
    Extracts real UAV weed patches from WeedsGalore dataset where segmentation label == 2 (weed).
    Constructs RGB composite from _R, _G, _B channels and crops 64x64 centered on weed clusters.
    """
    weed_out_dir = os.path.join(output_dir, "Weed Invasion")
    os.makedirs(weed_out_dir, exist_ok=True)

    # Locate dataset root (could be weedsgalore/weedsgalore-dataset or subdirs)
    possible_roots = [
        weedsgalore_dir,
        os.path.join(weedsgalore_dir, "weedsgalore-dataset"),
    ]
    ds_root = None
    for r in possible_roots:
        if os.path.isdir(r):
            # Check for date subfolders
            for item in os.listdir(r):
                if os.path.isdir(os.path.join(r, item, "semantics")):
                    ds_root = r
                    break
        if ds_root:
            break

    if not ds_root:
        print("[-] WeedsGalore image folders with 'semantics/' not found in:", weedsgalore_dir)
        return 0

    count = 0
    date_folders = sorted([d for d in os.listdir(ds_root) if os.path.isdir(os.path.join(ds_root, d, "semantics"))])
    print(f"[*] Found {len(date_folders)} date folders in WeedsGalore: {date_folders}")

    for df in date_folders:
        sem_dir = os.path.join(ds_root, df, "semantics")
        img_dir = os.path.join(ds_root, df, "images")
        sem_files = sorted(glob.glob(os.path.join(sem_dir, "*.png")))

        for sf in sem_files:
            if count >= max_samples:
                break

            base_name = os.path.splitext(os.path.basename(sf))[0]
            r_path = os.path.join(img_dir, f"{base_name}_R.png")
            g_path = os.path.join(img_dir, f"{base_name}_G.png")
            b_path = os.path.join(img_dir, f"{base_name}_B.png")

            if not (os.path.exists(r_path) and os.path.exists(g_path) and os.path.exists(b_path)):
                continue

            sem = cv2.imread(sf, cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue

            # Weed mask (bg:0, crop:1, weed:2 or >1)
            weed_mask = (sem >= 2).astype(np.uint8)
            num_weed_pixels = int(np.sum(weed_mask))
            if num_weed_pixels < 25:
                continue

            r_img = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
            g_img = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
            b_img = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
            if r_img is None or g_img is None or b_img is None:
                continue

            # Build BGR composite for OpenCV
            bgr = cv2.merge([b_img, g_img, r_img])
            h, w = bgr.shape[:2]

            # Find connected components of weed regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(weed_mask, connectivity=8)

            for comp_idx in range(1, num_labels):
                if count >= max_samples:
                    break
                area = stats[comp_idx, cv2.CC_STAT_AREA]
                if area < 30:
                    continue

                cx = int(round(centroids[comp_idx][0]))
                cy = int(round(centroids[comp_idx][1]))

                th, tw = target_size
                half_h, half_w = th // 2, tw // 2
                y1 = max(0, cy - half_h)
                y2 = min(h, y1 + th)
                if y2 - y1 < th:
                    y1 = max(0, y2 - th)

                x1 = max(0, cx - half_w)
                x2 = min(w, x1 + tw)
                if x2 - x1 < tw:
                    x1 = max(0, x2 - tw)

                crop = bgr[y1:y2, x1:x2]
                if crop.shape[0] != th or crop.shape[1] != tw:
                    crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

                out_path = os.path.join(weed_out_dir, f"weed_{df}_{base_name}_{comp_idx:03d}.png")
                cv2.imwrite(out_path, crop)
                count += 1

    print(f"[+] Extracted {count} real weed patches from WeedsGalore")
    return count


def synthesize_missing_classes(
    output_dir: str,
    target_count: int = 1000,
    target_size: Tuple[int, int] = (64, 64),
    seed: int = 42,
):
    """
    Synthesizes classes that lack external datasets (Fungal/Bacterial Disease and Pest Infestation)
    as well as fallback for Weed Invasion if needed, ensuring class balance across all 5 classes.
    """
    helper_ds = AgriculturalPathologyDataset(
        target_size=target_size,
        is_train=True,
        seed=seed,
    )

    classes_to_check = {
        "Fungal/Bacterial Disease": helper_ds._synthesize_fungal_disease,
        "Pest Infestation": helper_ds._synthesize_pest_infestation,
        "Weed Invasion": helper_ds._synthesize_weed_invasion,
    }

    # Gather host base crops (from real Healthy Canopy folder if populated)
    healthy_files = glob.glob(os.path.join(output_dir, "Healthy Canopy", "*.png"))

    for class_name, synth_fn in classes_to_check.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        existing = glob.glob(os.path.join(class_dir, "*.png"))
        needed = target_count - len(existing)

        if needed <= 0:
            print(f"[*] {class_name} already has {len(existing)} samples.")
            continue

        print(f"[*] Synthesizing {needed} patches for '{class_name}'...")
        for i in range(needed):
            if healthy_files:
                base_bgr = cv2.imread(healthy_files[(i + len(existing)) % len(healthy_files)])
                base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            else:
                base_rgb = helper_ds._load_base_patch(i)

            synth_rgb = synth_fn(base_rgb)
            synth_bgr = cv2.cvtColor((synth_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

            out_path = os.path.join(class_dir, f"synth_{i:04d}.png")
            cv2.imwrite(out_path, synth_bgr)

        print(f"[+] '{class_name}' now has {len(glob.glob(os.path.join(class_dir, '*.png')))} samples.")


def main():
    parser = argparse.ArgumentParser(description="Prepare external pathology dataset for Stage 3 Classifier")
    parser.add_argument("--potato-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "multispectral-potato-plants-images-DatasetNinja"))
    parser.add_argument("--weedsgalore-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "weedsgalore"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "external_pathologies"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    args = parser.parse_args()

    ensure_class_dirs(args.output_dir)

    # 1. Extract potato plants
    if os.path.isdir(args.potato_dir):
        extract_potato_plants(args.potato_dir, args.output_dir, max_per_class=args.samples_per_class)
    else:
        print(f"[-] Potato directory not found at: {args.potato_dir}")

    # 2. Extract WeedsGalore (if unzipped/available)
    if os.path.isdir(args.weedsgalore_dir):
        extract_weedsgalore(args.weedsgalore_dir, args.output_dir, max_samples=args.samples_per_class)

    # 3. Fill missing classes with calibrated synthetic generators
    synthesize_missing_classes(args.output_dir, target_count=args.samples_per_class)

    print("\n=== Dataset Preparation Summary ===")
    for cls in CLASS_NAMES:
        c_dir = os.path.join(args.output_dir, cls)
        count = len(glob.glob(os.path.join(c_dir, "*.png")))
        print(f"  - {cls}: {count} patches")


if __name__ == "__main__":
    main()
