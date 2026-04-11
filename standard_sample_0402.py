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
from utils_II import get_bricks, classify_contour, save_contour_data

def main():
    # data_dir = r'E:\multi_source_info\data_dir\20260402'
    data_dir = r'E:\multi_source_info\data_dir\20260407_Sample_test'
    
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
                                                                                                            roi = [0, 1000, 200, 1336], 
                                                                                                            th_val = 175)
                base_name = os.path.splitext(filename)[0]
                
                # Save Standard contoured image
                img_std_output = os.path.join(output_dir, f"{base_name}_contoured.png")
                cv2.imwrite(img_std_output, contoured)

                # Classify each contour and save respectively
                for i, cnt in enumerate(cnt_filtered):
                    label = classify_contour(cnt)
                    save_contour_data(output_dir, base_name, label, i, 
                                      pixels[i][0], pixels[i][1], box_images[i])
                
                print(f"Successfully processed {filename}. Found {len(pixels)} contours.")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
