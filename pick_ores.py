import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ----------------- 从 utils_II.py 直接搬迁的核心功能函数 -----------------

def get_contour_centroid(contour):
    """Calculates the geometric centroid (cX, cY) of a contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def sort_contours(contours, tolerance=35, max_len=20, direction='y', reverse=False):
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

def correct_high_energy_distortion(image: np.ndarray, fx: float, fy: float = 1.0) -> np.ndarray:
    """
    针对高能和低能闪烁体探测器高度不同导致的扇形投影畸变进行校正。
    该函数通过双向缩放及对称补齐/裁剪，使校正后的图像保持原始尺寸。
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

def split_dual_xray_image(image, offset_up=0, offset_down=0, fx=0.9909, fy=1.0):
    """
    Splits a dual-energy X-ray image (stacked horizontally after T) into low and high energy parts.
    """
    height = image.shape[0]
    # Split into two channels
    low_power_image = image[offset_up:int(height / 2) - offset_down, :]
    high_power_image = image[int(height / 2) + offset_up:height - offset_down, :]

    # Apply distortion correction to high-energy part before returning
    high_power_image = correct_high_energy_distortion(high_power_image, fx, fy)

    return low_power_image, high_power_image

def compute_R(low, high, I0_low = 195, I0_high = 196, input = 'images', method = 'a', const = [5, 20]):
    """
    Computes R value image from low and high energy images.
    """
    if input == 'images':
        if method == 'a':
            return np.log(I0_low/(low+1e-6) + const[0] )/np.log(I0_high/(high+1e-6) + const[1])
        elif method == 'b':
            return np.log((low + 1e-6))/(np.log(high+1e-6 + 200.0))
    return np.array([])

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

def get_bricks(path = 'all_unnorm.png', roi = [200, -1, 600, 800], th_val = 175, 
    th_type = cv2.THRESH_BINARY, fx=0.99, fy=1.0, sort_direction='y', max_colwidth = 35,
    vscale=1.0, vscale_interp=cv2.INTER_LINEAR, reverse_sort=False):
    """
    Main pipeline for batch feature extraction.
    """
    data_int8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if data_int8 is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    low_ori, high_ori = split_dual_xray_image(data_int8.T, fx=fx, fy=fy)
    low, high = low_ori.T[roi[0]:roi[1], roi[2]:roi[3]], high_ori.T[roi[0]:roi[1], roi[2]:roi[3]]

    # Apply vertical scaling if requested (AFTER ROI selection to keep ROI coords valid)
    if vscale != 1.0:
        low = cv2.resize(low, (low.shape[1], int(low.shape[0] * vscale)), interpolation=vscale_interp)
        high = cv2.resize(high, (high.shape[1], int(high.shape[0] * vscale)), interpolation=vscale_interp)

    r_image = compute_R(low, high, I0_low = 195, I0_high = 196, 
                     input = 'images', method = 'a', const = [5, 20])
    _, thresholded = cv2.threshold(low.copy(), th_val, 255, th_type)

    # Find contours using cv2.RETR_TREE and cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoured = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR).copy()
    contoured_r = r_image.copy()

    cnt_filtered = []
    pixels = []; r_pixels = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 50000:
            cnt_filtered.append(cnt)
    
    # Use tiered sorting algorithm for robust indexing
    cnt_filtered = sort_contours(cnt_filtered, direction=sort_direction, tolerance=max_colwidth, reverse=reverse_sort)

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
        
        # Calculate Centroid
        cX, cY = get_contour_centroid(cnt)

        # Draw Compact Info Block: Just the ID
        cv2.putText(contoured, f"#{i}", (cX - 15, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return pixels, contoured, [low, high], r_pixels, contoured_r, box_images, cnt_filtered

def get_bricks_watershed(path = 'all_unnorm.png', roi = [200, -1, 600, 800], th_val = 175, 
    th_type = cv2.THRESH_BINARY, fx=0.99, fy=1.0, sort_direction='y', max_colwidth = 35,
    vscale=1.0, vscale_interp=cv2.INTER_LINEAR, reverse_sort=False):
    """
    Watershed-based pipeline for separating touching/overlapping samples.
    """
    data_int8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if data_int8 is None:
        raise FileNotFoundError(f"Could not read image at {path}")
        
    low_ori, high_ori = split_dual_xray_image(data_int8.T, fx=fx, fy=fy)
    low, high = low_ori.T[roi[0]:roi[1], roi[2]:roi[3]], high_ori.T[roi[0]:roi[1], roi[2]:roi[3]]

    # Apply vertical scaling if requested
    if vscale != 1.0:
        low = cv2.resize(low, (low.shape[1], int(low.shape[0] * vscale)), interpolation=vscale_interp)
        high = cv2.resize(high, (high.shape[1], int(high.shape[0] * vscale)), interpolation=vscale_interp)

    r_image = compute_R(low, high, I0_low = 195, I0_high = 196, 
                     input = 'images', method = 'a', const = [5, 20])
    
    # 1. Initial Thresholding
    _, thresholded = cv2.threshold(low.copy(), th_val, 255, th_type)
    
    # 2. Morphological Opening to remove noise/tiny bridges
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Distance Transform to find center of blobs
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 4. Threshold distance transform to get seeds (Sure Foreground)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 5. Find unknown region
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6. Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 7. Watershed on 3-channel image
    img_bgr = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)
    
    # 8. Reconstruct contours from markers
    cnt_filtered = []
    num_objects = np.max(markers)
    for label in range(2, num_objects + 1):
        obj_mask = np.zeros(low.shape, dtype=np.uint8)
        obj_mask[markers == label] = 255
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                
                if h > 800:
                    y_split = y + h // 2
                    mask1 = np.zeros(low.shape, dtype=np.uint8)
                    mask1[y:y_split, x:x+w] = obj_mask[y:y_split, x:x+w]
                    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_filtered.extend([c1 for c1 in cnts1 if cv2.contourArea(c1) > 100])
                    
                    mask2 = np.zeros(low.shape, dtype=np.uint8)
                    mask2[y_split:y+h, x:x+w] = obj_mask[y_split:y+h, x:x+w]
                    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_filtered.extend([c2 for c2 in cnts2 if cv2.contourArea(c2) > 100])
                    
                elif 600 < h <= 800:
                    y_split = y + 429
                    mask1 = np.zeros(low.shape, dtype=np.uint8)
                    mask1[y:y_split, x:x+w] = obj_mask[y:y_split, x:x+w]
                    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_filtered.extend([c1 for c1 in cnts1 if cv2.contourArea(c1) > 100])
                    
                    mask2 = np.zeros(low.shape, dtype=np.uint8)
                    mask2[y_split:y+h, x:x+w] = obj_mask[y_split:y+h, x:x+w]
                    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_filtered.extend([c2 for c2 in cnts2 if cv2.contourArea(c2) > 100])
                    
                else:
                    cnt_filtered.append(c)
    
    cnt_filtered = sort_contours(cnt_filtered, direction=sort_direction, tolerance=max_colwidth, reverse=reverse_sort)
    
    contoured = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR).copy()
    contoured_r = r_image.copy()
    pixels = []; r_pixels = []
    box_images = []

    for i, cnt in enumerate(cnt_filtered):
        pixels_i_low, pixels_i_high, r_values_i = get_contour_pixels(low, cnt), get_contour_pixels(high, cnt), get_contour_pixels(r_image, cnt)
        pixels.append([pixels_i_low, pixels_i_high]); r_pixels.append(r_values_i)
        
        box_low, box_high, box_r= get_contour_box_image(low, cnt, margin = 0), get_contour_box_image(high, cnt, margin = 0), get_contour_box_image(r_image, cnt, margin = 0)
        box_images.append([box_low, box_high, box_r])

        cv2.drawContours(contoured, [cnt], -1, (0, 0, 255), 1)
        cX, cY = get_contour_centroid(cnt)
        cv2.putText(contoured, f"#{i}", (cX - 15, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return pixels, contoured, [low, high], r_pixels, contoured_r, box_images, cnt_filtered

# --------------------------------------------------------------------------

def save_image_robust(image: np.ndarray, save_path: str) -> bool:
    """
    在 Windows 下鲁棒地保存包含非 ASCII/中文路径的图片。
    
    参数 (Parameters):
    ------------------
    image : np.ndarray
        要保存的图片数组。
    save_path : str
        图片保存的完整路径。
        
    返回 (Returns):
    ---------------
    bool: 保存成功返回 True，失败返回 False。
    """
    try:
        save_path = str(Path(save_path).resolve())
        # 创建父文件夹（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        if not ext:
            ext = '.png'
            save_path += ext
        
        result, img_encode = cv2.imencode(ext, image)
        if result:
            img_encode.tofile(save_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error saving image robustly: {e}")
        return False

def label_ores(
    image_path: str,
    model1_results: list,
    model2_results: list,
    output_path: str = None,
    roi: list = [200, -1, 600, 800],
    th_val: int = 175,
    use_watershed: bool = False,
    fx: float = 0.99,
    fy: float = 1.0,
    sort_direction: str = 'y',
    max_colwidth: int = 35,
    vscale: float = 1.0,
    alpha: float = 0.4,
    method: int = 2,
    reverse_sort: bool = False
) -> tuple:
    """
    提取矿石轮廓，根据两个分类模型的预测结果进行分类并绘制精美、高对比度的可视化标签。
    支持方法 1 (二分类配对)、方法 2 (Zeff等分四档及加权分档) 和方法 3 (静态硬编码ID判定)。
    
    参数含义 (Parameters):
    --------------------
    image_path : str
        输入摆放矿石的图片文件路径 (可以是 8 位或 16 位 stacked dual-energy 图像)。
        类型：str
    model1_results : list or np.ndarray
        在方法 1 中：模型 1 给出每个矿石的二分类结果 (精矿 1，废矿 0)；
        在方法 2 中：模型 1 预测每个矿石的有效原子序数 Zeff 实数值。
        类型：list 或 np.ndarray
    model2_results : list or np.ndarray
        在方法 1 中：模型 2 给出每个矿石的二分类结果 (精矿 1，废矿 0)；
        在方法 2 中：模型 2 预测每个矿石的有效原子序数 Zeff 实数值。
        类型：list 或 np.ndarray
    output_path : str, optional
        标注后的图像保存路径。如果为 None，则不进行磁盘写入，仅返回处理后的图像。
        类型：str, 默认 None
    roi : list of 4 ints, default [200, -1, 600, 800]
        感兴趣的区域 [y1, y2, x1, x2]，传入给 contour 提取函数。若 y2 或 x2 为 -1，代表到边界。
        类型：list (包含4个整数)
    th_val : int, default 175
        图像二值化时的灰度阈值，用于检测矿石轮廓。
        类型：int
    use_watershed : bool, default False
        是否启用基于分水岭算法的 `get_bricks_watershed` 提取轮廓。如果为 False，则使用传统的 `get_bricks`。
        类型：bool
    fx : float, default 0.99
        高能图像几何校正的横向比例参数。
        类型：float
    fy : float, default 1.0
        高能图像几何校正的纵向比例参数。
        类型：float
    sort_direction : str, default 'y'
        提取出轮廓后的空间排序方向，'y' 表示列优先排序 (从上到下，从左到右)；'x' 表示行优先排序。
        类型：str
    max_colwidth : int, default 35
        排序时的容差距离 (同一行或同一列的判定阈值)。
        类型：int
    vscale : float, default 1.0
        纵向缩放系数。
        类型：float
    alpha : float, default 0.4
        半透明填充遮罩 (mask overlay) 的不透明度比例，范围在 0.0 到 1.0 之间。
        类型：float
    method : int, default 2
        选择分类标注的方法。
        - 1: 方法1，基于两个模型的 0、1 二分类预测结果进行配对组合分类 (如 11, 10, 01, 00)。
        - 2: 方法2，基于两个模型分别输出每块矿石的预测 Zeff，归一化等分为四档进行分类。
        - 3: 方法3，基于静态硬编码ID进行四档划分。
        类型：int
    reverse_sort : bool, default False
        是否对排序的每一档进行逆向排列。
        - False: 在 row-major ('x') 下从左到右，在 column-major ('y') 下从上到下。
        - True: 在 row-major ('x') 下从右到左，在 column-major ('y') 下从下到上。
        类型：bool

    返回值 (Returns):
    -----------------
    tuple: (labeled_img, cnt_filtered)
        labeled_img (np.ndarray): 同样大小的 BGR 彩彩色标注图像，其中每个矿石轮廓都有半透明遮罩、边框和高对比度文本。
        cnt_filtered (list): 提取并排序后的轮廓列表。
        类型：(np.ndarray, list)
    """
    # 1. 进行原图矿石轮廓提取与定位
    if use_watershed:
        print("Using Watershed algorithm for contour extraction...")
        pixels, _, imgs, _, _, _, cnt_filtered = get_bricks_watershed(
            path=image_path,
            roi=roi,
            th_val=th_val,
            fx=fx,
            fy=fy,
            sort_direction=sort_direction,
            max_colwidth=max_colwidth,
            vscale=vscale,
            reverse_sort=reverse_sort
        )
    else:
        print("Using standard threshold algorithm for contour extraction...")
        pixels, _, imgs, _, _, _, cnt_filtered = get_bricks(
            path=image_path,
            roi=roi,
            th_val=th_val,
            fx=fx,
            fy=fy,
            sort_direction=sort_direction,
            max_colwidth=max_colwidth,
            vscale=vscale,
            reverse_sort=reverse_sort
        )
    
    # 提取低能通道图像（轮廓在这个通道上定位）
    low_img = imgs[0]
    
    # 将低能单通道灰度图转换为 3 通道 BGR 彩色图，以便在上面绘制彩色标签
    if len(low_img.shape) == 2:
        labeled_img = cv2.cvtColor(low_img, cv2.COLOR_GRAY2BGR)
    else:
        labeled_img = low_img.copy()
        
    # 水平翻转底图，以响应用户左右反转结果的要求
    labeled_img = cv2.flip(labeled_img, 1)
    
    # 将所有提取出的轮廓坐标进行水平镜像反转，使其与反转后的底图完美对齐
    W = labeled_img.shape[1]
    for cnt in cnt_filtered:
        cnt[:, 0, 0] = W - 1 - cnt[:, 0, 0]
        
    num_contours = len(cnt_filtered)
    print(f"Extracted {num_contours} ore contours.")
    
    # 2. 健壮性匹配：根据选择的分型方法 (Method 1, 2 或 3) 分别处理预测结果与绘制样式
    if method == 3:
        print("Running Method 3: Fixed ID List-based classification...")
        
        # 用户指定的 1-based 矿石编号集合
        top_25 = {3, 10, 11, 13, 14, 16, 22, 26, 27, 28, 31, 42, 43, 44, 45, 54, 56, 66, 69, 78, 80, 81, 88, 89, 90, 108, 109, 112, 113, 120, 121}
        mid_25_50 = {4, 8, 23, 24, 25, 30, 32, 33, 34, 40, 41, 46, 47, 48, 50, 63, 65, 68, 70, 71, 72, 75, 83, 94, 96, 101, 104, 107, 111, 114}
        mid_50_75 = {5, 6, 9, 12, 18, 20, 29, 36, 39, 51, 57, 58, 59, 60, 61, 64, 67, 73, 76, 84, 86, 91, 93, 98, 99, 100, 106, 115, 118, 119}
        bot_25 = {1, 2, 7, 15, 17, 19, 21, 35, 37, 38, 49, 52, 53, 55, 62, 74, 77, 79, 82, 85, 87, 92, 95, 97, 102, 103, 105, 110, 116, 117}
        
        class_keys = []
        label_texts = []
        for i in range(num_contours):
            ore_id = i + 1  # 1-based index
            if ore_id in top_25:
                class_keys.append("11") # Class 1 (Green)
                label_texts.append("11")
            elif ore_id in mid_25_50:
                class_keys.append("10") # Class 2 (Orange)
                label_texts.append("10")
            elif ore_id in mid_50_75:
                class_keys.append("01") # Class 3 (Blue)
                label_texts.append("01")
            elif ore_id in bot_25:
                class_keys.append("00") # Class 4 (Red)
                label_texts.append("00")
            else:
                # 健壮性兜底 fallback (按位置比例划分档位)
                ratio = ore_id / max(num_contours, 1)
                if ratio <= 0.25:
                    class_keys.append("11")
                    label_texts.append("11")
                elif ratio <= 0.50:
                    class_keys.append("10")
                    label_texts.append("10")
                elif ratio <= 0.75:
                    class_keys.append("01")
                    label_texts.append("01")
                else:
                    class_keys.append("00")
                    label_texts.append("00")
                
    elif method == 2:
        print("Running Method 2: Zeff Tier-based classification...")
        preds1 = list(model1_results)
        preds2 = list(model2_results)
        
        # 默认使用预测均值作为填充值，防止极端分档偏差
        mean_v1 = np.mean(preds1) if preds1 else 12.0
        mean_v2 = np.mean(preds2) if preds2 else 12.0
        
        if len(preds1) < num_contours:
            print(f"Model 1 predictions length ({len(preds1)}) is shorter than contours ({num_contours}). Padding with mean: {mean_v1:.2f}")
            preds1.extend([mean_v1] * (num_contours - len(preds1)))
        else:
            preds1 = preds1[:num_contours]
            
        if len(preds2) < num_contours:
            print(f"Model 2 predictions length ({len(preds2)}) is shorter than contours ({num_contours}). Padding with mean: {mean_v2:.2f}")
            preds2.extend([mean_v2] * (num_contours - len(preds2)))
        else:
            preds2 = preds2[:num_contours]
            
        p1_arr = np.array(preds1, dtype=np.float32)
        p2_arr = np.array(preds2, dtype=np.float32)
        
        # 各自最大和最小值归一化到 0-1
        p1_min, p1_max = np.min(p1_arr), np.max(p1_arr)
        p2_min, p2_max = np.min(p2_arr), np.max(p2_arr)
        
        denom1 = (p1_max - p1_min) if (p1_max - p1_min) > 1e-6 else 1.0
        denom2 = (p2_max - p2_min) if (p2_max - p2_min) > 1e-6 else 1.0
        
        p1_norm = (p1_arr - p1_min) / denom1
        p2_norm = (p2_arr - p2_min) / denom2
        
        # 等分为四档函数 [0, 1] -> {1, 2, 3, 4}
        def get_tier(x):
            if x >= 1.0:
                return 4
            elif x <= 0.0:
                return 1
            else:
                return int(x * 4) + 1
                
        tiers1 = [get_tier(val) for val in p1_norm]
        tiers2 = [get_tier(val) for val in p2_norm]
        
        # 计算已归一化 Zeff 的加权值，不用进行二次全局归一化，直接判定所属档位
        p_weighted_norm = 0.6 * p1_norm + 0.4 * p2_norm
        tiers_weighted = [get_tier(val) for val in p_weighted_norm]
        
        class_keys = []   # 颜色映射键值 (绿色11, 橙色10, 蓝色01, 红色00)
        label_texts = []  # 图像上显示的文案 (T1 - T4)
        
        for i in range(num_contours):
            t1 = tiers1[i]
            t2 = tiers2[i]
            
            # 分档一致：归于当前档；分档不一致：归于加权 zeff 对应的档
            if t1 == t2:
                final_tier = t1
            else:
                final_tier = tiers_weighted[i]
                
            # 映射为 4 类样式 (为了高级一致的彩色显示)
            if final_tier == 1:
                class_keys.append("11") # 第一档标记为第一类 (绿色)
                label_texts.append("1")
            elif final_tier == 2:
                class_keys.append("10") # 第二档标记为第二类 (橙色)
                label_texts.append("2")
            elif final_tier == 3:
                class_keys.append("01") # 第三档标记为第三类 (蓝色)
                label_texts.append("3")
            else:
                class_keys.append("00") # 第四档标记为第四类 (红色)
                label_texts.append("4")
                
    else:
        # Method 1: Binary predictions pairing
        print("Running Method 1: Binary predictions pairing...")
        preds1 = list(model1_results)
        preds2 = list(model2_results)
        
        if len(preds1) < num_contours:
            print(f"Model 1 predictions length ({len(preds1)}) is shorter than contours ({num_contours}). Padding with 0 (waste).")
            preds1.extend([0] * (num_contours - len(preds1)))
        else:
            preds1 = preds1[:num_contours]
            
        if len(preds2) < num_contours:
            print(f"Model 2 predictions length ({len(preds2)}) is shorter than contours ({num_contours}). Padding with 0 (waste).")
            preds2.extend([0] * (num_contours - len(preds2)))
        else:
            preds2 = preds2[:num_contours]
            
        class_keys = []
        label_texts = []
        for i in range(num_contours):
            p1 = int(preds1[i])
            p2 = int(preds2[i])
            class_str = f"{p1}{p2}"
            class_keys.append(class_str)
            label_texts.append(class_str)

    # 定义 4 个类别的色彩面板 (BGR 格式) 
    # Class 1 ("11" / "T1"): Emerald Green -> (55, 209, 76)
    # Class 2 ("10" / "T2"): Sun Orange -> (25, 140, 241)
    # Class 3 ("01" / "T3"): Ocean Blue -> (224, 130, 9)
    # Class 4 ("00" / "T4"): Soft Red -> (75, 75, 255)
    class_styles = {
        "11": {"color": (55, 209, 76), "name": "Class 1 (Tier 1)"},
        "10": {"color": (25, 140, 241), "name": "Class 2 (Tier 2)"},
        "01": {"color": (224, 130, 9), "name": "Class 3 (Tier 3)"},
        "00": {"color": (75, 75, 255), "name": "Class 4 (Tier 4)"}
    }
    
    # 创建半透明遮罩层
    overlay = labeled_img.copy()
    
    # 3. 逐个轮廓画上标签半透明蒙层
    for i, cnt in enumerate(cnt_filtered):
        class_str = class_keys[i]
        style = class_styles.get(class_str, class_styles["00"])
        color = style["color"]
        cv2.drawContours(overlay, [cnt], -1, color, thickness=cv2.FILLED)
        
    # 融合半透明遮罩与原图
    cv2.addWeighted(overlay, alpha, labeled_img, 1 - alpha, 0, dst=labeled_img)
    
    # 4. 在融合后的图像上绘制不透明的边框和高对比度文字
    # 建立统一的类别数值文案映射 ("11" -> "1", "10" -> "2", "01" -> "3", "00" -> "4")
    class_label_names = {
        "11": "1",
        "10": "2",
        "01": "3",
        "00": "4"
    }

    for i, cnt in enumerate(cnt_filtered):
        class_str = class_keys[i]
        text = class_label_names.get(class_str, "4")
        style = class_styles.get(class_str, class_styles["00"])
        color = style["color"]
        
        # 绘制轮廓边界 (不透明)
        cv2.drawContours(labeled_img, [cnt], -1, color, thickness=2)
        
        # 计算轮廓的质心 (centroid) 用于放置文字
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = cv2.boundingRect(cnt)[:2]
            
        # 绘制文本高对比度黑底背景盒 (使用 OpenCV 绘制标准数字)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 获取文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 确定文本框的坐标 (以质心为中心放置)
        box_x1 = cX - text_width // 2 - 5
        box_y1 = cY - text_height // 2 - 4
        box_x2 = cX + text_width // 2 + 5
        box_y2 = cY + text_height // 2 + 4
        
        # 确保坐标在图像尺寸范围内
        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(labeled_img.shape[1] - 1, box_x2)
        box_y2 = min(labeled_img.shape[0] - 1, box_y2)
        
        # 绘制深灰色圆角矩形背景 (带边框)
        cv2.rectangle(labeled_img, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(labeled_img, (box_x1, box_y1), (box_x2, box_y2), (200, 200, 200), 1)
        
        # 在背景盒中绘制明亮的文字 (白字数字)
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2
        cv2.putText(labeled_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # 另外在矿石外接矩形的上方（或最顶部时贴紧上沿）附上带有黑色背景框的 1-based 序号，彻底与中心分类结果拉开空间
        index_text = f"#{i + 1}"
        (id_w, id_h), id_baseline = cv2.getTextSize(index_text, font, 0.35, 1)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 确定序号框的坐标（在矿石上方，下边缘与矿石顶边缘有 2 像素间隙）
        id_box_x1 = max(0, x)
        id_box_y1 = max(0, y - id_h - 8)
        id_box_x2 = min(labeled_img.shape[1] - 1, x + id_w + 6)
        id_box_y2 = max(0, y - 2)
        
        # 如果上方越界（矿石靠近图像最顶部），则将序号框置于矿石内部的左上角
        if id_box_y1 <= 0:
            id_box_y1 = max(0, y + 2)
            id_box_y2 = min(labeled_img.shape[0] - 1, y + id_h + 8)
            
        cv2.rectangle(labeled_img, (id_box_x1, id_box_y1), (id_box_x2, id_box_y2), (30, 30, 30), cv2.FILLED)
        cv2.rectangle(labeled_img, (id_box_x1, id_box_y1), (id_box_x2, id_box_y2), (180, 180, 180), 1)
        cv2.putText(labeled_img, index_text, (id_box_x1 + 3, id_box_y1 + id_h + 3), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. 可选写入磁盘
    if output_path:
        success = save_image_robust(labeled_img, output_path)
        if success:
            print(f"Successfully saved labeled ore image to: {output_path}")
        else:
            print(f"Failed to save labeled ore image to: {output_path}")
            
    return labeled_img, cnt_filtered

if __name__ == '__main__':
    # =========================================================================
    # 文件内输入参数设置 (Configuration parameters inside the file)
    # =========================================================================
    
    # 1. 选择分类方法 (Select classification method)
    # method = 1: 方法1 (基于两个模型的 0、1 二分类预测结果进行配对组合分类)
    # method = 2: 方法2 (基于两个模型预测的每块矿石的 zeff，归一化、等分四档及加权融合分类)
    # method = 3: 方法3 (基于用户指定的 1-based 矿石编号静态列表进行分档渲染)
    method = 3
    
    # 2. 待处理图片路径 (Input image path)
    # image_path = r"E:\multi_source_info\data_dir\20260325_yinshan\big_ores_position_2_160kV.tif"
    image_path = r"E:\multi_source_info\data_dir\20260518_CuO.tif"
    # 3. 标注输出图片保存路径 (Output labeled image save path)
    output_path = r"results\labeled_ores_yinshan.png"
    
    # 4. 模型预测数据 (Model prediction data)
    if method == 1:
        # 方法1对应输入：二分类 0 或 1 (精矿为 1，废矿为 0)
        model1_results = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
        model2_results = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    elif method == 2:
        # 方法2对应输入：两个模型分别预测的每块矿石的 Zeff 实数值 (如 11 到 23 之间)
        model1_results = [11.5, 12.0, 18.2, 19.5, 11.2, 14.3, 16.8, 12.1, 11.8, 22.4, 21.0, 13.5, 11.6, 17.5, 19.0, 20.2]
        model2_results = [11.2, 12.5, 17.9, 19.0, 11.5, 13.8, 17.2, 12.3, 11.4, 23.0, 20.5, 13.9, 11.8, 17.0, 19.2, 20.0]
    else:
        # 方法3不需要模型预测，静态硬编码 1-based 矿石分类集合
        model1_results = []
        model2_results = []
        
    # 5. 轮廓提取及算法控制参数 (Contour extraction & algorithm parameters)
    roi = [0, 1625, 200, 1336]      # 感兴趣区域 [y1, y2, x1, x2]
    th_val = 140                    # 二值化灰度阈值
    use_watershed = False           # 是否启用分水岭算法 (True / False)
    fx = 1.0                        # 畸变校正横向比例
    fy = 0.9909                     # 畸变校正纵向比例
    sort_direction = 'x'            # 排序方向：'x' 代表行优先排序，'y' 代表列优先排序
    max_colwidth = 35               # 排序聚类阈值
    vscale = 1.0                    # 纵向缩放比例
    alpha = 0.4                     # 半透明蒙层不透明度系数 (0.0 - 1.0)
    reverse_sort = True            # 是否逆序排序 (Yinshan 银山数据集为 True 以实现右向左每行排序)
    
    # =========================================================================
    
    print(f"Loading image for test: {image_path}")
    print(f"Selected Method: Method {method}")
    print(f"Output path: {output_path}")
    
    # 执行标注
    label_ores(
        image_path=image_path,
        model1_results=model1_results,
        model2_results=model2_results,
        output_path=output_path,
        roi=roi,
        th_val=th_val,
        use_watershed=use_watershed,
        fx=fx,
        fy=fy,
        sort_direction=sort_direction,
        max_colwidth=max_colwidth,
        vscale=vscale,
        alpha=alpha,
        method=method,
        reverse_sort=reverse_sort
    )

