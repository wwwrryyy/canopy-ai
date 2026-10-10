"""
generate_trt_calibration.py
===========================
Generates a representative INT8 calibration dataset for NVIDIA TensorRT.

To maintain precision when converting FP32 ONNX graphs to INT8 TensorRT engines,
TensorRT requires a representative subset of data to calibrate activation histograms
(typically 500-1000 frames).

This script extracts fixed-size patches from healthy canopies and saves them
as a raw numpy array.
"""

import os
import glob
import numpy as np
import cv2
import argparse

def generate_calibration_dataset(
    data_dir: str, 
    output_path: str, 
    num_samples: int = 500,
    crop_size: int = 128
) -> None:
    """
    Samples healthy crop patches and normalizes them into a (N, C, H, W) numpy tensor.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
    if not image_paths:
        raise FileNotFoundError(f"No imagery found in {data_dir}")
        
    np.random.shuffle(image_paths)
    
    calibration_data = []
    count = 0
    
    for path in image_paths:
        if count >= num_samples:
            break
            
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        # Convert BGR to RGB (assuming multispectral synthesis or standard RGB inference uses standard channel order)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Center crop or resize
        h, w, c = img.shape
        if h < crop_size or w < crop_size:
            img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        else:
            y = (h - crop_size) // 2
            x = (w - crop_size) // 2
            img = img[y:y+crop_size, x:x+crop_size]
            
        # Normalize to [0, 1] float32
        img_normalized = img.astype(np.float32) / 255.0
        
        # Transpose to [C, H, W]
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        calibration_data.append(img_chw)
        count += 1
        
    if not calibration_data:
        raise RuntimeError("Failed to load any valid images for calibration.")
        
    # Stack into [N, C, H, W]
    dataset = np.stack(calibration_data, axis=0)
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, dataset)
    print(f"[+] Successfully generated TensorRT INT8 calibration dataset.")
    print(f"[+] Saved to: {output_path} | Shape: {dataset.shape} | Range: [{dataset.min():.2f}, {dataset.max():.2f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TensorRT Calibration Dataset")
    parser.add_argument("--data-dir", type=str, default="data/ucmerced_prepared", help="Path to representative imagery")
    parser.add_argument("--output", type=str, default="exports/tensorrt_calibration_dataset.npy", help="Output .npy path")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to extract")
    parser.add_argument("--size", type=int, default=128, help="Crop/Image size for calibration")
    args = parser.parse_args()
    
    generate_calibration_dataset(args.data_dir, args.output, args.samples, args.size)
