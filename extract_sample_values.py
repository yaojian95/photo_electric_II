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
    # roi = [0, 1200, 200, 1336]; th_val = 190; fy = 0.9909
    data_dir = r'E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test'
    # data_dir = r'E:\multi_source_info\data_dir\20260409_TYM-data\TYM_converted_results'
    roi_125 = [960, 1900, 0, -1]; th_val_125 = 160; fy = 1 #fy单独控制高能图像的校准比例
    roi_270 = [687, 3000, 0, -1]; th_val_270 = 151; fy = 1

    # Path-specific threshold method: Use BINARY_INV for 0409 TYM-data, otherwise BINARY
    th_type = cv2.THRESH_BINARY_INV if "0409" in data_dir else cv2.THRESH_BINARY
    # Dedicated function for 0409 dataset using Watershed segmentation
    fn_get_bricks = get_bricks_watershed if "0409" in data_dir else get_bricks
    
    # Extract folder name from data_dir to create a subfolder in results
    folder_name = os.path.basename(data_dir.rstrip('\\'))
    output_dir = os.path.join('results', folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Adaptive file discovery: find all .tif files with 'kV' in the name
    tif_files = [f for f in os.listdir(data_dir) 
                 if f.lower().endswith('.tif') and 'kv' in f.lower()]
    
    if not tif_files:
        print(f"No matching kV .tif files found in {data_dir}")
        return

    all_summaries = []

    for filename in tif_files:
            # DYNAMIC PARAMETERS: Handle specific 0409 270us compression
            roi = roi_125
            th_val = th_val_125
            vscale = 1.0
            vinterp = cv2.INTER_LINEAR
            
            if "0409" in data_dir and "270us" in filename.lower():
                roi = roi_270
                vscale = 1.0 / 2.7
                vinterp = cv2.INTER_AREA
                th_val = th_val_270
            elif "0409" in data_dir:
                roi = roi_125 # Default for other 0409 data
                th_val = th_val_125

            image_path = os.path.join(data_dir, filename)
            
            print(f"Processing {filename} (vscale={vscale:.2f})...")
            
            # Using get_bricks with synchronized params from standard_sample.py
            try:
                pixels, contoured, ori_low_high, r_pixels, contoured_r, box_images, cnt_filtered = fn_get_bricks(image_path,
                                                                                                             roi = roi, 
                                                                                                             th_val = th_val, 
                                                                                                             th_type = th_type, 
                                                                                                             fx=1.0, fy=fy, 
                                                                                                             sort_direction='y',
                                                                                                             vscale=vscale,
                                                                                                             vscale_interp=vinterp)
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
                    warped_bundle = [warped_low, warped_high, None]

                    # Use STRAIGHTENED image to classify
                    cur_pixels_low = pixels[i][0]
                    label, meta = classify_contour(cnt, box_image_low=warped_low, pixels_low=cur_pixels_low)
                    
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
