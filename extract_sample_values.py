import os
import sys
import matplotlib.pyplot as plt

# Ensure the script's directory is in the path so local 'utils_II' can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import cv2
import pickle
import numpy as np
import pandas as pd
from utils_II import get_bricks, get_bricks_watershed, classify_contour, save_contour_data, warp_straighten

def main():
    # data_dir = r'E:\multi_source_info\data_dir\20260402'
    # data_dir = r'E:\multi_source_info\data_dir\20260331'
    # data_dir = r'E:\multi_source_info\data_dir\20260407_Sample_test'
    roi = [0, 1200, 200, 1336]; all_type = None; align_direct = 'y'

    # data_dir = r'E:\multi_source_info\data_dir\20260401'; 
    # roi = [0, 1400, 100, 1586]; all_type = 'ore'; align_direct = 'x'
    
    th_val = 190; fy = 0.9909 #fy单独控制高能图像的校准比例
    
    # data_dir = r'E:\multi_source_info\data_dir\20260512_180kV1ma_all_dual\dual'
    # roi = [0, -1, 0, -1]; all_type = None; align_direct = 'y'; th_val = 200
    # # Set to a specific string (e.g., 'ore', 'disk', 'block', 'step_sample') to force all contours 
    # # to be classified as that type. Set to None for automatic classification based on geometry.
    
    reverse_sort = False

    # data_dir = r'E:\multi_source_info\data_dir\20260325_yinshan'; roi = [0, 1625, 200, 1336]
    # all_type = 'ore'; th_val = 140; fy = 0.9909; align_direct = 'x'; reverse_sort = True

    data_dir = r'E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test'
    # data_dir = r'E:\multi_source_info\data_dir\20260409_TYM-data\TYM_converted_results'
    roi_125 = [250, -1, 0, -1]; th_val_125 = 160; 
    roi_270 = [0, -1, 0, -1]; th_val_270 = 151; 

    # data_dir = r'E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test_2'
    # roi_125 = [250, 2100, 0, -1]; th_val_125 = 140; 

    # Path-specific threshold method: Use BINARY_INV for 0409 TYM-data, otherwise BINARY
    th_type = cv2.THRESH_BINARY_INV if "0409" in data_dir else cv2.THRESH_BINARY
    # Dedicated function for 0409 dataset using Watershed segmentation
    fn_get_bricks = get_bricks_watershed if "0409" in data_dir else get_bricks
    
    # Extract folder name from data_dir to create a subfolder in results
    folder_name = os.path.basename(data_dir.rstrip('\\'))
    output_dir = os.path.join('results', folder_name + '_16bit')
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Adaptive file discovery: support both .tif (with 'kv') and .png (with 'dual')
    # For 0409 datasets, exclusively read files with the suffix '_cropped.tif'
    if "0409" in data_dir:
        tif_files = [f for f in os.listdir(data_dir) if f.lower().endswith('_cropped.tif')]
    else:
        tif_files = [f for f in os.listdir(data_dir) 
                     if (f.lower().endswith('.tif') and 'kv' in f.lower()) or 
                        (f.lower().endswith('.png') and 'dual' in f.lower())]
    
    if not tif_files:
        print(f"No valid image files found in {data_dir} (0409 expects '_cropped.tif', others expect '.tif' containing 'kv' or '.png' containing 'dual')")
        return

    all_summaries = []

    for filename in tif_files:
            # DYNAMIC PARAMETERS: Handle specific 0409 270us compression
            
            # 默认值
            vscale = 1.0
            vinterp = cv2.INTER_LINEAR
            ellipse_limit = 0.98
            if "0409" in data_dir:
                fy = 1

                if "270us" in filename.lower():
                    roi = roi_270
                    vscale = 1/1.5
                    vinterp = cv2.INTER_AREA
                    th_val = th_val_270
                elif "125us" in filename.lower():
                    roi = roi_125
                    th_val = th_val_125
            else:
                # 20260331 等其他数据集走默认参数
                roi = roi  # 使用 main() 开头定义的 roi
                th_val = th_val

            image_path = os.path.join(data_dir, filename)
            
            print(f"Processing {filename} (vscale={vscale:.2f})...")
            
            # Using get_bricks with synchronized params from standard_sample.py
            try:
                pixels, contoured, ori_low_high, r_pixels, contoured_r, box_images, cnt_filtered = fn_get_bricks(image_path,
                                                                                                             roi = roi, 
                                                                                                             th_val = th_val, 
                                                                                                             th_type = th_type, 
                                                                                                             fx=1.0, fy=fy, 
                                                                                                             sort_direction=align_direct,
                                                                                                             max_colwidth= 35,
                                                                                                             vscale=vscale,
                                                                                                             vscale_interp=vinterp,
                                                                                                             reverse_sort=reverse_sort)
                base_name = os.path.splitext(filename)[0]
                
                # Save Standard contoured image
                img_std_output = os.path.join(output_dir, f"{base_name}_contoured.png")
                cv2.imwrite(img_std_output, contoured)

                low_roi, high_roi = ori_low_high[0], ori_low_high[1]
                
                # Classify each contour and save respectively
                mean_summaries = []
                for i, cnt in enumerate(cnt_filtered):
                    # Warp the object for precise analysis using ALIGNED ROI images
                    warped_low, M_inv = warp_straighten(low_roi, cnt)
                    warped_high, _ = warp_straighten(high_roi, cnt)
                    
                    # Create a mask to filter out background pixels when saving high_low_images
                    mask = np.zeros(low_roi.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    warped_mask, _ = warp_straighten(mask, cnt)
                    
                    bg_val_low = 65535 if warped_low.dtype == np.uint16 else 255
                    bg_val_high = 65535 if warped_high.dtype == np.uint16 else 255
                    
                    warped_low_save = warped_low.copy()
                    warped_low_save[warped_mask < 128] = bg_val_low
                    
                    warped_high_save = warped_high.copy()
                    warped_high_save[warped_mask < 128] = bg_val_high

                    warped_bundle = [warped_low_save, warped_high_save, None]

                    # Use STRAIGHTENED image to classify
                    cur_pixels_low = pixels[i][0]
                    label, meta = classify_contour(cnt, ellipse_limit = ellipse_limit, box_image_low=warped_low, pixels_low=cur_pixels_low, all_type=all_type)
                    
                    # REFINEMENT: Handle Disk Core Sampling and Step Sampling
                    save_pixels_low = meta["refined_pixels_low"]
                    save_pixels_high = pixels[i][1] # Fallback
                    
                    if label == 'step_sample':
                        from utils_II import get_step_pixels_list
                        save_pixels_high = get_step_pixels_list(warped_high, meta["sampling_boxes"])

                    if label == 'disk':
                        from utils_II import get_disk_core_info
                        core_pixels, center, scaled_cnt = get_disk_core_info(low_roi, cnt)
                        save_pixels_low = core_pixels
                        core_pixels_high, _, _ = get_disk_core_info(high_roi, cnt)
                        save_pixels_high = core_pixels_high
                        # Update annotation for disk core on the image
                        m_core = float(np.mean(core_pixels)) if core_pixels.size > 0 else 0
                        s_core = float(np.std(core_pixels)) if core_pixels.size > 0 else 0
                        
                        # Draw blue scaled contour
                        cv2.drawContours(contoured, [scaled_cnt], -1, (255, 0, 0), 1)
                        # Overwrite/Add core statistics in Yellow
                        cv2.putText(contoured, f"m:{m_core:.1f} s:{s_core:.1f}", (center[0] - 30, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                        
                    if label == 'block':
                        from utils_II import get_inner_95_pixels
                        save_pixels_low = get_inner_95_pixels(low_roi, cnt)
                        save_pixels_high = get_inner_95_pixels(high_roi, cnt)
                        
                        m_final = float(np.mean(save_pixels_low)) if save_pixels_low.size > 0 else 0
                        s_final = float(np.std(save_pixels_low)) if save_pixels_low.size > 0 else 0
                    
                    # Visual Feedback: Draw the bounding rectangle (minAreaRect) used for warping
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    cv2.polylines(contoured, [np.int32(box)], True, (255, 0, 255), 1)

                    # Visual Feedback: Annotate stats and classification label
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        # 1. Label
                        cv2.putText(contoured, label, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        # 2. Refined Stats (Yellow for disk, default for others)
                        stats_color = (0, 255, 255) if label == 'disk' else (0, 255, 255) # Keep yellow for visibility
                        
                        m_final = m_core if label == 'disk' else float(np.mean(save_pixels_low)) if isinstance(save_pixels_low, np.ndarray) else float(np.mean(pixels[i][0]))
                        s_final = s_core if label == 'disk' else float(np.std(save_pixels_low)) if isinstance(save_pixels_low, np.ndarray) else float(np.std(pixels[i][0]))
                        
                        cv2.putText(contoured, f"m:{m_final:.1f} s:{s_final:.1f}", (cx - 30, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, stats_color, 1)

                    # Record means for console output
                    m_val_log = float(np.mean(save_pixels_low)) if isinstance(save_pixels_low, np.ndarray) else float(np.mean(pixels[i][0]))
                    if meta["step_means"]:
                        mean_summaries.append([round(m, 1) for m in meta["step_means"]] + [round(m_val_log, 1)])

                    # Visual Feedback: Draw the 10 sampling boxes
                    if meta["sampling_boxes"]:
                        for box_local in meta["sampling_boxes"]:
                            box_global = cv2.perspectiveTransform(box_local.reshape(-1, 1, 2), M_inv).reshape(4, 2)
                            cv2.polylines(contoured, [box_global.astype(np.int32)], True, (0, 255, 0), 1)

                    # Save the WARPED ROI images with REDINED pixels
                    save_contour_data(output_dir, base_name, label, i, 
                                      save_pixels_low, save_pixels_high, warped_bundle)
                    
                    # EXTRACT STEP TRANSITION ZONE (transition.pkl)
                    if label == 'step_sample':
                        step_means = meta["step_means"]
                        if step_means and len(step_means) == 10:
                            # Higher mean value means thinner step (higher transmission)
                            if step_means[0] > step_means[9]:
                                # index 0 is thin, index 9 is thick.
                                # 3rd mutation boundary is between index 2 and index 3 (0-indexed).
                                y_boundary = int(3 * (warped_low.shape[0] / 10.0))
                            else:
                                # index 9 is thin, index 0 is thick.
                                # 3rd mutation boundary is between index 7 and index 6 (0-indexed).
                                y_boundary = int(7 * (warped_low.shape[0] / 10.0))
                            
                            y_start = max(0, y_boundary - 5)
                            y_end = min(warped_low.shape[0], y_boundary + 5)
                            
                            # Use 10% horizontal margins to avoid edge artifacts
                            roi_x1 = int(warped_low.shape[1] * 0.1)
                            roi_x2 = warped_low.shape[1] - roi_x1
                            
                            pixels_low_steep = warped_low[y_start:y_end, roi_x1:roi_x2].flatten()
                            pixels_high_steep = warped_high[y_start:y_end, roi_x1:roi_x2].flatten()
                            
                            # Save to _transition.pkl
                            steep_output = os.path.join(output_dir, "pixel_values", f"{base_name}_{label}_{i}_transition.pkl")
                            with open(steep_output, 'wb') as sf:
                                pickle.dump({
                                    'pixels_low': pixels_low_steep,
                                    'pixels_high': pixels_high_steep
                                }, sf)
                            print(f"--> Saved transition pixels (+/-5 rows around row {y_boundary}) to {steep_output}")

                            # Draw Transition Box on contoured image (BGR: Red, thickness=2)
                            box_local_transition = np.array([
                                [roi_x1, y_start], [roi_x2, y_start], [roi_x2, y_end], [roi_x1, y_end]
                            ], dtype="float32")
                            box_global_transition = cv2.perspectiveTransform(box_local_transition.reshape(-1, 1, 2), M_inv).reshape(4, 2)
                            cv2.polylines(contoured, [box_global_transition.astype(np.int32)], True, (0, 0, 255), 2)
                            
                            # Annotate "T3" near the transition boundary
                            tx, ty = int(box_global_transition[0][0]), int(box_global_transition[0][1])
                            cv2.putText(contoured, "T3", (tx - 15, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


                    
                    # 4. Collect data for final summary table
                    entry_base = {
                        "File": filename,
                        "Obj": f"#{i}",
                        "Type": label
                    }
                    
                    if label == 'step_sample' and isinstance(save_pixels_low, list):
                        m_list = [round(np.mean(p), 1) for p in save_pixels_low]
                        s_list = [round(np.std(p), 1) for p in save_pixels_low]
                        all_summaries.append({**entry_base, "Metric": "Mean", "Value": m_list})
                        all_summaries.append({**entry_base, "Metric": "Std ", "Value": s_list})
                    else:
                        m_val = round(np.mean(save_pixels_low), 1) if save_pixels_low.size > 0 else 0
                        s_val = round(np.std(save_pixels_low), 1) if save_pixels_low.size > 0 else 0
                        all_summaries.append({**entry_base, "Metric": "Mean", "Value": m_val})
                        all_summaries.append({**entry_base, "Metric": "Std ", "Value": s_val})
                
                if mean_summaries:
                    print(f"--> Means [S1-S10, TOTAL] for {filename}: {mean_summaries}")
                
                # Save summary image in specialized subfolder
                summary_dir = os.path.join(output_dir, "contoured_images")
                if not os.path.exists(summary_dir): os.makedirs(summary_dir)
                
                img_std_output = os.path.join(summary_dir, f"{base_name}_contoured.png")
                cv2.imwrite(img_std_output, contoured)
                
                print(f"Successfully processed {filename}. Found {len(pixels)} contours.")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Output Final Summary Table
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        print("\n" + "="*80)
        print(f"ANALYSIS SUMMARY REPORT: {folder_name}")
        print("="*80)
        # Using a wide display to accommodate step lists
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    main()
