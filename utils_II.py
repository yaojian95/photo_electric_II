import sys
import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import pickle

def get_contour_centroid(contour):
    """Calculates the geometric centroid (cX, cY) of a contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def sort_contours(contours, tolerance=35, max_len=9, direction='y', reverse=False):
    """
    Sorts contours using a tiered approach.
    direction='x': Row-major (Group by Row Y, Sort by X inside row).
    direction='y': Column-major (Group by Column X, Sort by Y inside column).
    """
    if not contours:
        return []
        
    if direction == 'x':
        sort_index = 1   # Row sorting depends first on vertical (Y)
        group_index = 0  # Then horizontal (X) within row
    elif direction == 'y':
        sort_index = 0   # Column sorting depends first on horizontal (X)
        group_index = 1  # Then vertical (Y) within column
    else:
        raise ValueError("Direction must be 'x' (row-major) or 'y' (column-major).")

    # Calculate centroids
    centers = [get_contour_centroid(cnt) for cnt in contours]
    
    # Initial sort to group them
    sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i][sort_index])
    sorted_contours = [contours[i] for i in sorted_indices]
    sorted_centers = [centers[i] for i in sorted_indices]
    
    groups = []
    current_group = []
    previous = None
    
    for cnt, center in zip(sorted_contours, sorted_centers):
        if previous is None:
            current_group.append((cnt, center[group_index]))
            previous = center[sort_index]
        else:
            # Check if within tolerance and limit
            if abs(center[sort_index] - previous) <= tolerance and len(current_group) < max_len:
                current_group.append((cnt, center[group_index]))
                previous = center[sort_index]
            else:
                # Sort the group by the secondary axis
                current_group_sorted = sorted(current_group, key=lambda item: item[1], reverse=reverse)
                groups.extend([item[0] for item in current_group_sorted])
                current_group = [(cnt, center[group_index])]
                previous = center[sort_index]
    
    if current_group:
        current_group_sorted = sorted(current_group, key=lambda item: item[1], reverse=reverse)
        groups.extend([item[0] for item in current_group_sorted])
    
    return groups

def get_contour_pixels(image, contour):
    """Returns pixel values inside the given contour."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return image[mask == 255]

def get_contour_box_image(image, contour, margin=10):
    """Returns a cropped image of the bounding box of the contour with an optional margin."""
    x, y, w, h = cv2.boundingRect(contour)
    y1, y2 = max(y - margin, 0), min(y + h + margin, image.shape[0])
    x1, x2 = max(x - margin, 0), min(x + w + margin, image.shape[1])
    return image[y1:y2, x1:x2]

def get_contour_box_image_with_background(image, contour, margin=0, background_value=255):
    """Extracts the smallest rectangle containing the contour with a specified background value."""
    x, y, w, h = cv2.boundingRect(contour)
    y1, y2 = max(y - margin, 0), min(y + h + margin, image.shape[0])
    x1, x2 = max(x - margin, 0), min(x + w + margin, image.shape[1])
    
    box_image = image[y1:y2, x1:x2]
    mask = np.zeros_like(box_image, dtype=np.uint8)
    # Adjust contour coordinates to the cropped image
    contour_offset = contour - [x1, y1]
    cv2.drawContours(mask, [contour_offset], -1, 255, thickness=cv2.FILLED)
    
    result = np.full_like(box_image, background_value, dtype=np.uint8)
    result[mask == 255] = box_image[mask == 255]
    return result

def split_dual_xray_image(image, offset_up=0, offset_down=0, fx=0.9909, fy=1.0):
    """
    Splits a dual-energy X-ray image (stacked horizontally after T) into low and high energy parts.
    Integrates geometric distortion correction for the high-energy channel.
    
    Parameters:
        image: Transposed raw image (channel-stacked along rows).
        offset_up, offset_down: Vertical cropping offsets.
        fx: Horizontal correction factor (default 0.9909).
        fy: Vertical correction factor (default 1.0).
    """
    height = image.shape[0]
    # Split into two channels
    low_power_image = image[offset_up:int(height / 2) - offset_down, :]
    high_power_image = image[int(height / 2) + offset_up:height - offset_down, :]

    # Apply distortion correction to high-energy part before returning
    high_power_image = correct_high_energy_distortion(high_power_image, fx, fy)

    return low_power_image, high_power_image

def warp_straighten(image, cnt):
    """
    Warps a tilted rectangular object to be axis-aligned with robust point ordering.
    Returns: (warped_img, M_inv)
    """
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    
    # Standardize orientation: ensure h > w
    if w > h:
        w, h = h, w
        angle += 90
        
    src_pts = cv2.boxPoints(((cx, cy), (w, h), angle))
    
    # Sort src_pts: top-left, top-right, bottom-right, bottom-left
    # 1. Sort by y: top two are 0 and 1
    # 2. Between those two, sort by x: top-left is min-x
    src_pts = src_pts[np.argsort(src_pts[:, 1])] # Sort by y
    top_two = src_pts[:2][np.argsort(src_pts[:2, 0])] # top_left, top_right
    bot_two = src_pts[2:][np.argsort(src_pts[2:, 0])][::-1] # bottom_right, bottom_left
    src_pts = np.vstack([top_two, bot_two]).astype("float32")
    
    dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (int(w), int(h)))
    
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    return warped, M_inv

def get_10_step_means(straightened_image, margin_x=0.1, margin_y=0.25):
    """
    Divides the image into 10 vertical segments and calculates means of core areas.
    Coverage: Horizontal (1 - 2*margin_x), Vertical per step (1 - 2*margin_y).
    """
    h, w = straightened_image.shape[:2]
    step_h = h / 10.0
    means = []
    sampling_boxes = []
    
    for i in range(10):
        y_start = int(i * step_h)
        y_end = int((i + 1) * step_h)
        
        # Calculate sub-segment core
        seg_h = y_end - y_start
        m_y = int(seg_h * margin_y)
        m_x = int(w * margin_x)
        
        roi_y1, roi_y2 = y_start + m_y, y_end - m_y
        roi_x1, roi_x2 = m_x, w - m_x
        
        # Ensure valid ROI
        if roi_y2 <= roi_y1: roi_y1, roi_y2 = y_start, y_end
        if roi_x2 <= roi_x1: roi_x1, roi_x2 = 0, w
        
        sample_area = straightened_image[roi_y1:roi_y2, roi_x1:roi_x2]
        means.append(float(np.mean(sample_area)) if sample_area.size > 0 else 0.0)
        
        sampling_boxes.append(np.array([
            [roi_x1, roi_y1], [roi_x2, roi_y1], [roi_x2, roi_y2], [roi_x1, roi_y2]
        ], dtype="float32"))
        
    return means, sampling_boxes

def check_step_gradient(straightened_image, threshold=10.0):
    """
    Legacy/Simplified check. Refactored to use 10-step logic for classification.
    """
    means, _ = get_10_step_means(straightened_image)
    if not any(means): return False, 0.0, 0.0, 0.0
    
    # Simple check: diff between extreme means
    top_mean = means[0]
    bot_mean = means[-1]
    mid_mean = np.mean(means[1:9])
    
    diff_top = abs(top_mean - mid_mean)
    diff_bot = abs(bot_mean - mid_mean)
    
    # 1. Pearson Correlation for global trend linearity
    x_axis = np.arange(len(means))
    r_matrix = np.corrcoef(x_axis, means)
    r_val = r_matrix[0, 1] if r_matrix.shape == (2, 2) else 0
    
    # 2. Dynamic Span Threshold: scales with intensity level
    # Formula: max(min_absolute_span, coefficient * mean_intensity)
    avg_intensity = np.mean(means)
    span = np.max(means) - np.min(means)
    dynamic_threshold = max(2.0, 0.05 * avg_intensity)
    
    is_monotonic_trend = abs(r_val) > 0.7 and span > dynamic_threshold
    
    # 3. Final Decision: Sudden Jump OR Reliable Monotonic Trend
    is_step = (diff_top > threshold or diff_bot > threshold) or is_monotonic_trend
    
    return is_step, top_mean, mid_mean, bot_mean

def classify_contour(cnt, box_image_low=None, pixels_low=None):
    """
    Classifies the contour into ['block', 'step_sample', 'disk', 'ore'].
    Returns: (label, meta)
    """
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    rectangularity = area / rect_area if rect_area > 0 else 0
    circularity = (4 * np.pi * area) / (peri**2) if peri > 0 else 0
    
    # Ellipse ratio for disk detection
    ellipse_ratio = 0
    if len(cnt) >= 5:
        (ex, ey), (ema, emi), eang = cv2.fitEllipse(cnt)
        ellipse_area = (np.pi * ema * emi) / 4
        ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0

    meta = {
        "rectangularity": round(rectangularity, 2),
        "circularity": round(circularity, 2),
        "step_means": None,
        "sampling_boxes": [],
        "refined_pixels_low": pixels_low,
        "disk_core": None, # (center, radius)
        "top_m": 0, "mid_m": 0, "bot_m": 0
    }

    if circularity > 0.82 or ellipse_ratio > 0.92:
        label = 'disk'
        # DISK REFINEMENT: Calculate stats on 2/3 core
        if box_image_low is not None:
            core_pixels, center, core_radius = get_disk_core_info(box_image_low, cnt)
            meta["refined_pixels_low"] = core_pixels
            meta["disk_core"] = (center, core_radius)
    elif rectangularity > 0.75:
        label = 'block'
        if box_image_low is not None:
            means, s_boxes = get_10_step_means(box_image_low)
            meta["step_means"] = means
            meta["sampling_boxes"] = s_boxes
            
            is_step, m_top, m_mid, m_bot = check_step_gradient(box_image_low)
            meta["top_m"], meta["mid_m"], meta["bot_m"] = m_top, m_mid, m_bot
            if is_step:
                label = "step_sample"
                meta["refined_pixels_low"] = get_step_pixels_list(box_image_low, s_boxes)
            else:
                label = "block"
                # (Optional) block erosion is now fully handled via global ROIs in the main runner to avoid coordinate transformation bugs

        # 2. Fallback to std if only pixels are available
        elif pixels_low is not None:
            if meta["std"] > 5.0: 
                label = "step_sample"
            else:
                label = "block"
                # (Optional) could erode here but box_image is preferred
        else:
            label = "block"
    else: 
        label = "ore"
        # ORE keeps ALL pixels as requested by user
        meta["refined_pixels_low"] = pixels_low
        
    return label, meta

def save_contour_data(output_dir, base_name, label, index, pixels_low, pixels_high, warped_images):
    """
    Saves results into organized subfolders: pixel_values and high_low_images.
    """
    # Create subfolders
    paths = {
        'pixels': os.path.join(output_dir, "pixel_values"),
        'images': os.path.join(output_dir, "high_low_images")
    }
    for p in paths.values():
        if not os.path.exists(p): os.makedirs(p)

    # 1. Save .pkl data
    data_output = os.path.join(paths['pixels'], f"{base_name}_{label}_{index}_data.pkl")
    
    # pixels_low can be a list of arrays (for step_sample) or a single array
    data_to_save = {
        'pixels_low': pixels_low,
        'pixels_high': pixels_high
    }
    with open(data_output, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    # 2. Save ROI box images (low, high) - expected to be already warped
    box_low, box_high, _ = warped_images
    cv2.imwrite(os.path.join(paths['images'], f"{base_name}_{label}_{index}_low.png"), box_low)
    cv2.imwrite(os.path.join(paths['images'], f"{base_name}_{label}_{index}_high.png"), box_high)

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

def get_step_pixels_list(warped_img, sampling_boxes_local):
    """
    Extracts a list of 10 pixel arrays, one for each step's sampling core.
    """
    step_pixels_list = []
    for box_local in sampling_boxes_local:
        # box_local is (4, 2): [tl, tr, br, bl]
        x1, y1 = int(box_local[0][0]), int(box_local[0][1])
        x2, y2 = int(box_local[2][0]), int(box_local[2][1])
        step_pixels_list.append(warped_img[y1:y2, x1:x2].flatten())
    return step_pixels_list

def get_disk_core_info(image, cnt, scale=2/3):
    """
    Calculates 2/3 area/radius core pixels by scaling the contour toward its centroid.
    This handles elliptical or irregular disks better than circular approximation.
    """
    M = cv2.moments(cnt)
    if M["m00"] == 0: return np.array([]), (0,0), cnt
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    
    # Scale contour points toward centroid
    # P_new = Centroid + scale * (P_old - Centroid)
    scaled_cnt = []
    for pt in cnt:
        p = pt[0]
        new_p = centroid + scale * (p - centroid)
        scaled_cnt.append([new_p])
    
    scaled_cnt = np.array(scaled_cnt).astype(np.int32)
    
    # Create mask for scaled contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [scaled_cnt], -1, 255, -1)
    
    core_pixels = image[mask == 255]
    return core_pixels, (cx, cy), scaled_cnt

def get_inner_95_pixels(image, cnt):
    """
    Extracts the inner ~95% pixels of a contour to avoid edge effects.
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    orig_area = np.count_nonzero(mask)
    if orig_area == 0: return np.array([])
    
    target_area = 0.95 * orig_area
    eroded_mask = mask.copy()
    kernel = np.ones((3,3), np.uint8)
    
    # Iteratively erode until we hit ~95% or can't erode more
    for _ in range(5): # Max 5 iterations to avoid excessive shrinkage
        temp = cv2.erode(eroded_mask, kernel, iterations=1)
        if np.count_nonzero(temp) < target_area:
            break
        eroded_mask = temp
        
    return image[eroded_mask == 255]

def get_bricks(path = 'all_unnorm.png', roi = [200, -1, 600, 800], th_val = 175, fx=0.99, fy=1.0, sort_direction='y'):
    data_int8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # all_unnorm = all_unnorm.astype(np.int8)
    low_ori, high_ori = split_dual_xray_image(data_int8.T, fx=fx, fy=fy)
    low, high = low_ori.T[roi[0]:roi[1], roi[2]:roi[3]], high_ori.T[roi[0]:roi[1], roi[2]:roi[3]]
    extra_bottom = low[0:10, :]
    low, high = np.vstack((low, extra_bottom)), np.vstack((high, extra_bottom))

    r_image = compute_R(low, high, I0_low = 195, I0_high = 196, 
                     input = 'images', method = 'a', const = [5, 20])
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
    
    # Use tiered sorting algorithm for robust indexing
    cnt_filtered = sort_contours(cnt_filtered, direction=sort_direction)

    box_images = []

    for i, cnt in enumerate(cnt_filtered):
        # 1. Extract pure pixels BEFORE drawing anything
        pixels_i_low, pixels_i_high, r_values_i = get_contour_pixels(low, cnt), get_contour_pixels(high, cnt), get_contour_pixels(r_image, cnt)
        pixels.append([pixels_i_low, pixels_i_high]); r_pixels.append(r_values_i)
        
        box_low, box_high, box_r= get_contour_box_image(low, cnt, margin = 0), get_contour_box_image(high, cnt, margin = 0), get_contour_box_image(r_image, cnt, margin = 0)
        box_images.append([box_low, box_high, box_r])

        # 2. Draw results on contoured image
        cv2.drawContours(contoured, [cnt], -1, (0, 0, 255), 1)
        
        # Calculate stats on INNER 95% pixels to avoid edge noise
        inner_pixels = get_inner_95_pixels(low, cnt)
        m_val_inner = float(np.mean(inner_pixels)) if inner_pixels.size > 0 else float(np.mean(pixels_i_low))
        s_val_inner = float(np.std(inner_pixels)) if inner_pixels.size > 0 else float(np.std(pixels_i_low))
        
        # Calculate Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw Compact Info Block: Just the ID
        cv2.putText(contoured, f"#{i}", (cX - 15, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # plt.imshow(thresholded, cmap='gray')
    # plt.colorbar()

    return pixels, contoured, [low, high], r_pixels, contoured_r, box_images, cnt_filtered

def correct_high_energy_distortion(image: np.ndarray, fx: float, fy: float = 1.0) -> np.ndarray:
    """
    针对高能和低能闪烁体探测器高度不同导致的扇形投影畸变进行校正。
    该函数通过双向缩放及对称补齐/裁剪，使校正后的图像保持原始尺寸。
    
    Parameters:
        image (np.ndarray): 输入的高能图像 (grayscale or color)。
        fx (float): 横向修正系数。
        fy (float): 纵向修正系数 (默认 1.0)。
        fx和fy只是为了防止图像转置，实际只有一个方向畸变。0.9909 for both 银山设备和厂房设备。

    Returns:
        np.ndarray: 校正后的图像，尺寸与输入一致。
    """
    if (fx == 1.0 and fy == 1.0) or image is None:
        return image
    
    h, w = image.shape[:2]
    
    # 1. 按照系数进行双向缩放
    resized = cv2.resize(image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    
    new_h, new_w = resized.shape[:2]
    
    # 2. 对图像进行横向补齐或裁剪，使其保持原始宽度 w
    if new_w < w:
        total_pad = w - new_w
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    elif new_w > w:
        total_crop = new_w - w
        crop_left = total_crop // 2
        resized = resized[:, crop_left : crop_left + w].copy()

    # 3. 对图像进行纵向补齐或裁剪，使高度保持原始尺寸 h
    if new_h < h:
        total_pad = h - new_h
        pad_top = total_pad // 2
        pad_bot = total_pad - pad_top
        resized = cv2.copyMakeBorder(resized, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif new_h > h:
        total_crop = new_h - h
        crop_top = total_crop // 2
        resized = resized[crop_top : crop_top + h, :].copy()
        
    return resized
