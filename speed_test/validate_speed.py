import cv2
import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt

def load_and_preprocess():
    path_0p5 = r'E:\multi_source_info\data_dir\20260407_Sample_test\test4_0p5mps.tif'
    path_3 = r'E:\multi_source_info\data_dir\20260407_Sample_test\test4_3mps.tif'

    print(f"Loading images...")
    img05_raw = tifffile.imread(path_0p5)
    img30_raw = tifffile.imread(path_3)

    img05_p1 = img05_raw[:img05_raw.shape[0]//2, 200:1536-200]
    img30_le = img30_raw[:, 200:1536-200]

    if img05_p1.dtype == np.uint16: img05_p1 = (img05_p1 >> 8).astype(np.uint8)
    if img30_le.dtype == np.uint16: img30_le = (img30_le >> 8).astype(np.uint8)

    return img05_p1, img30_le

def get_ores_at_threshold(img, thresh):
    """Attempt to find 4 ores using the given threshold."""
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 1500]
    
    if len(valid) == 4:
        valid = sorted(valid, key=lambda c: cv2.boundingRect(c)[1])
        ores = []
        for cnt in valid:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 15
            y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
            crop_img = img[y1:y2, x1:x2]
            mask = np.zeros_like(img, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            crop_mask = mask[y1:y2, x1:x2]
            ores.append({'img': crop_img, 'mask': crop_mask, 'area': cv2.contourArea(cnt)})
        return ores
    return None

def find_unified_threshold(img1, img2):
    """Find a single threshold that yields exactly 4 ores in both images."""
    for thresh in range(250, 180, -1):
        ores1 = get_ores_at_threshold(img1, thresh)
        if ores1:
            ores2 = get_ores_at_threshold(img2, thresh)
            if ores2:
                print(f"Unified threshold found: {thresh}")
                return thresh, ores1, ores2
    return None, None, None

def run():
    img05, img30 = load_and_preprocess()
    
    thresh, ores05, ores30 = find_unified_threshold(img05, img30)
    
    if not ores05 or not ores30:
        print("Error: Could not find a unified threshold that yields exactly 4 ores in both images.")
        return
    
    num = 4
    plt.figure(figsize=(15, 4 * num))
    print(f"\nSummary Table (Masked Pixels Only, Thresh={thresh}):")
    print(f"{'Ore':<5} | {'0.5 m/s Mean':<12} | {'3.0 m/s Mean':<12} | {'Rel Diff':<10} | {'0.5 m/s Std':<10} | {'3.0 m/s Std':<10}")
    print("-" * 85)
    
    for i in range(num):
        o1_data = ores05[i]
        o2_data = ores30[i]
        
        pixels1 = o1_data['img'][o1_data['mask'] > 127]
        pixels2 = o2_data['img'][o2_data['mask'] > 127]
        
        m1, s1 = (np.mean(pixels1), np.std(pixels1)) if len(pixels1) > 0 else (0,0)
        m2, s2 = (np.mean(pixels2), np.std(pixels2)) if len(pixels2) > 0 else (0,0)
        
        rel_diff = ((m2 - m1) / m1 * 100) if m1 != 0 else 0
        
        print(f"{i+1:<5} | {m1:<12.2f} | {m2:<12.2f} | {rel_diff:<+9.2f}% | {s1:<10.2f} | {s2:<10.2f}")
        
        # Plotting
        plt.subplot(num, 3, 3*i + 1)
        plt.imshow(o1_data['img'], cmap='gray', vmin=0, vmax=255)
        plt.title(f"Ore {i+1} (0.5m/s)\nMean:{m1:.2f}")
        plt.axis('off')
        
        plt.subplot(num, 3, 3*i + 2)
        plt.imshow(o2_data['img'], cmap='gray', vmin=0, vmax=255)
        plt.title(f"Ore {i+1} (3.0m/s)\nMean:{m2:.2f}")
        plt.axis('off')

        plt.subplot(num, 3, 3*i + 3)
        plt.hist(pixels1, bins=50, range=(0, 255), color='blue', alpha=0.5, label='0.5 m/s', density=True)
        plt.hist(pixels2, bins=50, range=(0, 255), color='red', alpha=0.5, label='3.0 m/s', density=True)
        plt.title(f"Histogram (Rel Diff: {rel_diff:+.2f}%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('ores_detailed_mask_comparison.png')
    print("\nDetailed result saved to ores_detailed_mask_comparison.png")

if __name__ == "__main__":
    run()
