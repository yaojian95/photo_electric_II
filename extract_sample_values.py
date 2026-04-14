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
from utils_II import get_bricks, classify_contour, save_contour_data, warp_straighten

def main():
    # data_dir = r'E:\multi_source_info\data_dir\20260402'
    data_dir = r'E:\multi_source_info\data_dir\20260331'
    # data_dir = r'E:\multi_source_info\data_dir\20260407_Sample_test'
    
    # Extract folder name from data_dir to create a subfolder in results
    folder_name = os.path.basename(data_dir)
    output_dir = os.path.join('results', folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Adaptive file discovery: find all .tif files with 'kV' in the name
    tif_files = [f for f in os.listdir(data_dir) 
                 if f.lower().endswith('.tif') and 'kv' in f.lower()]
    
    if not tif_files:
        print(f"No matching kV .tif files found in {data_dir}")
        return

    for filename in tif_files:
            image_path = os.path.join(data_dir, filename)
            
            print(f"Processing {filename}...")
            
            # Using get_bricks with synchronized params from standard_sample.py
            try:
                pixels, contoured, ori_low_high, r_pixels, contoured_r, box_images, cnt_filtered = get_bricks(image_path,
                                                                                                            roi = [0, 1200, 200, 1336], 
                                                                                                            th_val = 190, fx=1.0, fy = 0.9909, sort_direction='y')
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
                        
                    if label == 'block':
                        from utils_II import get_inner_95_pixels
                        save_pixels_low = get_inner_95_pixels(low_roi, cnt)
                        save_pixels_high = get_inner_95_pixels(high_roi, cnt)
                        
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                            # Draw blue scaled contour (instead of a static circle)
                            cv2.drawContours(contoured, [scaled_cnt], -1, (255, 0, 0), 1)
                            # Overwrite/Add core statistics in Yellow
                            cv2.putText(contoured, f"m:{m_core:.1f} s:{s_core:.1f}", (cx - 30, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    
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

if __name__ == "__main__":
    main()
