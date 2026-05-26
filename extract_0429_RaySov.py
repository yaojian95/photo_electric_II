import os
import cv2
import numpy as np
import pandas as pd
import pickle

from utils_II import (
    compute_R, sort_contours, get_contour_pixels, get_contour_box_image, 
    get_inner_95_pixels, classify_contour, save_contour_data, warp_straighten, 
    correct_high_energy_distortion, normalize_image,
    get_10_step_means, get_step_pixels_list
)

def find_step_sample_corners(cnt):
    """
    Finds the 4 true corners of the step sample, ignoring the wider bracket at the bottom.
    Returns: [top_left, top_right, bottom_right, bottom_left]
    """
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    
    lines = []
    n = len(approx)
    for i in range(n):
        p1 = approx[i]
        p2 = approx[(i+1)%n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        # Calculate angle of the line segment relative to horizontal
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        # We look for roughly vertical lines (45 to 135 degrees)
        if 45 < angle < 135:
            lines.append({'p1': p1, 'p2': p2, 'len': length, 'x': (p1[0]+p2[0])/2})
            
    lines.sort(key=lambda item: item['len'], reverse=True)
    
    # If we couldn't find at least two vertical lines, fallback to bounding box
    if len(lines) < 2:
        x, y, w, h = cv2.boundingRect(cnt)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

    # The two longest vertical lines should correspond to the left and right edges
    edge1, edge2 = lines[0], lines[1]
    
    if edge1['x'] < edge2['x']:
        left_edge, right_edge = edge1, edge2
    else:
        left_edge, right_edge = edge2, edge1
        
    # For the left edge, smaller Y is top-left, larger Y is bottom-left
    tl = left_edge['p1'] if left_edge['p1'][1] < left_edge['p2'][1] else left_edge['p2']
    bl = left_edge['p1'] if left_edge['p1'][1] > left_edge['p2'][1] else left_edge['p2']
    
    # For the right edge, smaller Y is top-right, larger Y is bottom-right
    tr = right_edge['p1'] if right_edge['p1'][1] < right_edge['p2'][1] else right_edge['p2']
    br = right_edge['p1'] if right_edge['p1'][1] > right_edge['p2'][1] else right_edge['p2']
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_step_sample(image, corners, width_ratio=1.0):
    """
    Warps the image defined by the 4 corners into a straightened rectangular image.
    """
    tl, tr, br, bl = corners
    
    width_top = np.hypot(tl[0] - tr[0], tl[1] - tr[1])
    width_bottom = np.hypot(bl[0] - br[0], bl[1] - br[1])
    width = int(max(width_top, width_bottom) * width_ratio)
    
    height_left = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    height_right = np.hypot(tr[0] - br[0], tr[1] - br[1])
    height = int(max(height_left, height_right))
    
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    M_inv = np.linalg.inv(M)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M_inv, M, width, height


def get_bricks_raysov_global(path, read_mode='8bit', roi=[0, -1, 0, -1], th_val=190, th_type=cv2.THRESH_BINARY):
    """
    全局提取函数：不对图像做预分割，而是提取整体阶梯轮廓后，再分离出0.6mm和1.2mm的部分。
    """
    file_bytes = np.fromfile(path, dtype=np.uint8)
    data_raw = cv2.imdecode(file_bytes, cv2.IMREAD_ANYDEPTH)
    if data_raw is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    if read_mode == '16bit':
        data_processed = normalize_image(data_raw, current_max=50000.0, target_ratio=0.8, target_bit_depth=16)
    elif read_mode == '8bit':
        data_processed = normalize_image(data_raw, current_max=50000.0, target_ratio=0.8, target_bit_depth=8)
    else:
        raise ValueError("read_mode must be '8bit' or '16bit'")

    # 获取全图一半的宽度，通常为512
    half_w = data_processed.shape[1] // 2
    
    # 提取完整低能(左半边)和高能(右半边)
    low_ori = data_processed[:, 0:half_w]
    high_ori = data_processed[:, half_w:]
    
    roi_y1, roi_y2, roi_x1, roi_x2 = roi
    if roi_y2 == -1: roi_y2 = low_ori.shape[0]
    if roi_x2 == -1: roi_x2 = low_ori.shape[1]

    low = low_ori[roi_y1:roi_y2, roi_x1:roi_x2]
    high = high_ori[roi_y1:roi_y2, roi_x1:roi_x2]

    _, thresholded = cv2.threshold(low.copy(), th_val, 255, th_type)
    if thresholded.dtype != np.uint8:
        thresholded = thresholded.astype(np.uint8)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if low.dtype == np.uint16:
        contoured = cv2.cvtColor((low / 256).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        contoured = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR).copy()

    # 过滤微小噪点轮廓
    cnt_filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    if not cnt_filtered:
        return None
    
    # 假设最大的轮廓就是包含阶梯和支架的主轮廓
    cnt_filtered.sort(key=cv2.contourArea, reverse=True)
    best_cnt = cnt_filtered[0]
    
    # 获取剔除支架后的真实角点
    corners = find_step_sample_corners(best_cnt)
    
    # 根据真实角点拉直图像，摆脱支架带来的偏斜干扰
    warped_low, M_inv, M, width, height = warp_step_sample(low, corners)
    warped_high, _, _, _, _ = warp_step_sample(high, corners)
    
    # 中心线在全局坐标里是 half_w (如512)，但因为我们是对low_ori (宽0-511)操作的
    # 低能区本身的分界线在局部也就是 half_w / 2 (如256)
    global_mid_x = half_w // 2 
    local_mid_x = global_mid_x - roi_x1
    
    # 将边界线映射到拉直图的坐标系中，计算切割点位置
    line_pts = np.array([[[local_mid_x, 0]], [[local_mid_x, height-1]]], dtype=np.float32)
    mapped_pts = cv2.perspectiveTransform(line_pts, M)
    
    split_x = int((mapped_pts[0][0][0] + mapped_pts[1][0][0]) / 2)
    split_x = max(1, min(split_x, width - 2))
    
    # 绘制可视化信息：外轮廓和定位的四个角点
    cv2.drawContours(contoured, [best_cnt], -1, (0, 0, 255), 1)
    for pt in corners:
        cv2.circle(contoured, tuple(pt.astype(int)), 4, (255, 0, 0), -1)

    return contoured, warped_low, warped_high, split_x, M_inv, corners


def main():
    data_dir = r'E:\multi_source_info\data_dir\20260429\20260429\阶梯'
    
    # --- 文件名翻译与重命名 ---
    translations = {
        "铁阶梯": "Fe_step",
        "铜阶梯": "Cu_step",
        "铝阶梯": "Al_step",
        "矿石": "Ore",
        "暗场": "DarkField",
        "空场": "AirField",
        "校准前": "uncalib",
        "校准后": "calib"
    }
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            new_filename = filename
            for ch, en in translations.items():
                new_filename = new_filename.replace(ch, en)
            
            if new_filename != filename:
                old_path = os.path.join(data_dir, filename)
                new_path = os.path.join(data_dir, new_filename)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename}  ->  {new_filename}")
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
    # --------------------------

    read_mode = '16bit' # 选项: '8bit' 或 '16bit'
    roi = [0, -1, 0, -1] # 处理所有高度和宽度
    
    if read_mode == '16bit':
        th_val = 39000
    else:
        th_val = 190
        
    th_type = cv2.THRESH_BINARY
    
    folder_name = '20260429_RaySov_16bit'
    output_dir = os.path.join('results', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_files = [f for f in os.listdir(data_dir) 
                 if f.lower().endswith('.tif') and ('calib' in f or '校准后' in f) and 'uncalib' not in f]
    
    if not tif_files:
        print(f"No matching calibrated .tif files found in {data_dir}")
        return

    all_summaries = []

    for filename in tif_files:
        image_path = os.path.join(data_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 只处理阶梯数据，因为根据要求0429每张图片只有一种且阶梯放在边界
        if 'step' not in base_name.lower() and '阶梯' not in base_name:
            continue
            
        print(f"Processing {filename} ...")
        try:
            result = get_bricks_raysov_global(image_path, read_mode=read_mode, roi=roi, th_val=th_val, th_type=th_type)
            if result is None:
                print(f"No contours found in {filename}")
                continue
                
            contoured, warped_low, warped_high, split_x, M_inv, corners = result
            
            # 以中线为界，切分出 0.6mm (左侧) 和 1.2mm (右侧) 的独立图像
            warped_low_06 = warped_low[:, :split_x]
            warped_high_06 = warped_high[:, :split_x]
            
            warped_low_12 = warped_low[:, split_x:]
            warped_high_12 = warped_high[:, split_x:]
            
            # 分别针对 0.6mm 和 1.2mm 进行各自独立的 10 阶采样
            means_06, boxes_06 = get_10_step_means(warped_low_06)
            pixels_low_06 = get_step_pixels_list(warped_low_06, boxes_06)
            pixels_high_06 = get_step_pixels_list(warped_high_06, boxes_06)
            
            means_12, boxes_12 = get_10_step_means(warped_low_12)
            pixels_low_12 = get_step_pixels_list(warped_low_12, boxes_12)
            pixels_high_12 = get_step_pixels_list(warped_high_12, boxes_12)
            
            # 将两种采样框通过逆透视变换重新绘制到原全局图上
            # 0.6mm 局部X坐标不变
            for box in boxes_06:
                box_global = cv2.perspectiveTransform(box.reshape(-1, 1, 2), M_inv).reshape(4, 2)
                cv2.polylines(contoured, [box_global.astype(np.int32)], True, (0, 255, 0), 1)
                
            # 1.2mm 局部X坐标需要先加上 split_x 偏移量
            for box in boxes_12:
                box_shifted = box.copy()
                box_shifted[:, 0] += split_x
                box_global = cv2.perspectiveTransform(box_shifted.reshape(-1, 1, 2), M_inv).reshape(4, 2)
                cv2.polylines(contoured, [box_global.astype(np.int32)], True, (0, 255, 255), 1)
                
            # 保存带绘制采样框的输出图像
            img_std_output = os.path.join(output_dir, f"{base_name}_global_contoured.png")
            cv2.imwrite(img_std_output, contoured)
            
            # 保存两组数据的序列和可视化信息
            for f_type, p_low, p_high in [('0.6mm', pixels_low_06, pixels_high_06), ('1.2mm', pixels_low_12, pixels_high_12)]:
                output_base_name = f"{base_name}_{f_type}"
                
                w_bundle = [warped_low_06 if f_type == '0.6mm' else warped_low_12, 
                            warped_high_06 if f_type == '0.6mm' else warped_high_12, None]
                
                save_contour_data(output_dir, output_base_name, 'step_sample', 0, p_low, p_high, w_bundle)
                
                m_list = [round(np.mean(p), 1) for p in p_low]
                s_list = [round(np.std(p), 1) for p in p_low]
                
                entry_base = {
                    "File": f"{filename} ({f_type})",
                    "Obj": "Step Sample",
                    "Type": "10-step"
                }
                all_summaries.append({**entry_base, "Metric": "Mean", "Value": m_list})
                all_summaries.append({**entry_base, "Metric": "Std ", "Value": s_list})

            print(f"Successfully processed {filename} -> split into 0.6mm and 1.2mm.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        print("\n" + "="*80)
        print(f"ANALYSIS SUMMARY REPORT: {folder_name}")
        print("="*80)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    main()
