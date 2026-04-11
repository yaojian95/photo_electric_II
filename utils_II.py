import sys
import os

import numpy as np
import cv2
import pandas as pd

folder_path = r'E:\photo_electric\jt_ore_sorting-main\jt_ore_sorting-main'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from dataloader import split_dual_xray_image
from preprocessing import get_contour_pixels, get_contour_box_image
import pickle

def classify_contour(cnt):
    """
    Classifies a contour as 'block', 'disk', or 'ore' based on geometric properties.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # 1. Circularity for disks (circular or elliptical)
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # 2. Rectangularity for standard samples (blocks/bricks)
    # Using minAreaRect to handle tilted rectangles
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    rect_area = w * h
    rectangularity = 0
    if rect_area > 0:
        rectangularity = area / rect_area
        
    # Thresholding
    if circularity > 0.82: # High circularity -> disk
        return "disk"
    elif rectangularity > 0.85: # High rectangularity -> block
        return "block"
    else: # Otherwise -> ore
        return "ore"

def save_contour_data(output_dir, base_name, label, index, pixels_low, pixels_high, box_images):
    """
    Saves pixel data and ROI box images for a classified contour (Low/High separate, No R).
    """
    # 1. Save .pkl data (No R)
    data_output = os.path.join(output_dir, f"{base_name}_{label}_{index}_data.pkl")
    data_to_save = {
        'pixels_low': pixels_low,
        'pixels_high': pixels_high
    }
    with open(data_output, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    # 2. Save ROI box images (low, high) separately
    box_low, box_high, _ = box_images
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_{label}_{index}_low.png"), box_low)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_{label}_{index}_high.png"), box_high)

def compute_R(low, high, I0_low = 195, I0_high = 196, input = 'images', method = 'a', const = [5, 20]):

    '''
    input: 'images' or 'pixels', whole images of low and high energy or pixels of rocks
    
    '''

    # if isinstance(low, pd.Series):
    #     low = low.to_list()
    #     high = high.to_list()

    if input == 'images':
        if method == 'a':
            return np.log(I0_low/(low+1e-6) + const[0] )/np.log(I0_high/(high+1e-6) + const[1])

        elif method == 'b':
            return np.log((low + 1e-6))/(np.log(high+1e-6 + 200.0))
        
    elif input == 'pixels':
        R_values = []
        for i in low.index:

            if method == 'a':
                R_i =np.log(I0_low/(low[i]+1e-6) + const[0] )/np.log(I0_high/(high[i]+1e-6) + const[1])

            elif method == 'b':
                R_i = np.log((low[i] + 1e-6))/(np.log(high[i]+1e-6 + 200.0))   
            R_values.append(R_i)

        return pd.Series(R_values, index=low.index)

def get_bricks(path = 'all_unnorm.png', roi = [200, -1, 600, 800], th_val = 175):
    data_int8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(data_int8[0:300, 200:1000].mean(), data_int8[0:300, 2000:2500].mean())
    # all_unnorm = all_unnorm.astype(np.int8)
    low_ori, high_ori = split_dual_xray_image(data_int8.T)
    low, high = low_ori.T[roi[0]:roi[1], roi[2]:roi[3]], high_ori.T[roi[0]:roi[1], roi[2]:roi[3]]
    extra_bottom = low[0:10, :]
    low, high = np.vstack((low, extra_bottom)), np.vstack((high, extra_bottom))

    r_image = compute_R(low, high, I0_low = 195, I0_high = 196, 
                     input = 'images', method = 'a', const = [5, 20])
    print(low.shape, r_image.shape)
    _, thresholded = cv2.threshold(low.copy(), th_val, 255, cv2.THRESH_BINARY)

    # Find contours using cv2.RETR_TREE and cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Preserve float R-image for visualization (don't normalize to 0-255)
    contoured = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR).copy()
    contoured_r = r_image.copy()

    cnt_filtered = []
    pixels = []; r_pixels = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 50000:
            cnt_filtered.append(cnt)
    cnt_filtered = sorted(cnt_filtered, key=lambda c: ((cv2.boundingRect(c)[1])))

    box_images = []

    for cnt in cnt_filtered:
        pixels_i_low, pixels_i_high, r_values_i = get_contour_pixels(low, cnt), get_contour_pixels(high, cnt), get_contour_pixels(r_image, cnt)
        pixels.append([pixels_i_low, pixels_i_high]); r_pixels.append(r_values_i)
        # print(len(pixels_i_low), len(pixels_i_high), len(r_values_i))

        box_low, box_high, box_r= get_contour_box_image(low, cnt, margin = 0), get_contour_box_image(high, cnt, margin = 0), get_contour_box_image(r_image, cnt, margin = 0)
        box_images.append([box_low, box_high, box_r])

    _ = cv2.drawContours(contoured, cnt_filtered, -1, (0, 0, 255), 2)
    # R-image remains raw float for scientific plotting with colorbar
    for contour, text_i in zip(cnt_filtered, np.arange(len(cnt_filtered))):
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Put the index number near the contour
        cv2.putText(contoured, str(text_i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # cv2.putText(contoured_r, str(text_i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2.0, 2)

    # plt.imshow(thresholded, cmap='gray')
    # plt.colorbar()

    return pixels, contoured, [low_ori.T, high_ori.T], r_pixels, contoured_r, box_images, cnt_filtered
