import os
import cv2
import numpy as np
import pandas as pd
import pickle
import re
from utils_II import normalize_image, save_contour_data

def process_masks_and_extract():
    base_dir = r'E:\multi_source_info\data_dir\20260429_mask_generated'
    
    # 结果保存路径：仿照 extract_sample_values.py 存放在当前工程的 results 目录下
    folder_name = os.path.basename(base_dir.rstrip('\\'))
    output_dir = os.path.join('results', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_data = []

    for category in ['ore', 'steps']:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        tif_files = [f for f in os.listdir(cat_dir) if f.lower().endswith('.tif')]
        
        # 翻译字典
        translations = {
            "铁阶梯": "Fe_step",
            "铜阶梯": "Cu_step",
            "铝阶梯": "Al_step",
            "矿石": "Ore",
            "校准前": "uncalib",
            "校准后": "calib"
        }

        for tif_name in tif_files:
            tif_path = os.path.join(cat_dir, tif_name)
            png_name = os.path.splitext(tif_name)[0] + '.png'
            png_path = os.path.join(cat_dir, png_name)
            
            # 将原始文件名翻译为英文作为保存的基础名
            base_name = os.path.splitext(tif_name)[0]
            for ch, en in translations.items():
                base_name = base_name.replace(ch, en)
            
            if not os.path.exists(png_path):
                print(f"Warning: Mask {png_name} not found for {tif_name}. Skipping.")
                continue
                
            print(f"Processing {tif_name} ({category})...")
            
            # 使用 numpy 读取避免中文路径报错
            tif_bytes = np.fromfile(tif_path, dtype=np.uint8)
            img = cv2.imdecode(tif_bytes, cv2.IMREAD_ANYDEPTH)
            
            png_bytes = np.fromfile(png_path, dtype=np.uint8)
            mask = cv2.imdecode(png_bytes, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"Error reading {tif_name} or its mask.")
                continue
                
            # 依据后缀判断是否需要归一化
            if 'orig' in tif_name.lower():
                # 映射到 255 的 80% (约 204)
                img = normalize_image(img, current_max=50000.0, target_ratio=0.8, target_bit_depth=8)
                norm_status = "Normalized (8bit 80%)"
            elif 'user' in tif_name.lower():
                # 已经归一化到 65536*0.8 (16位)，现在降为 8 位
                img = normalize_image(img, current_max=65535.0, target_ratio=1.0, target_bit_depth=8)
                norm_status = "Normalized (8bit from 16bit-user)"
            else:
                norm_status = "Unknown (Raw)"
                
            # 严格按照左右对半分：左侧低能，右侧高能
            half_w = img.shape[1] // 2
            low_img = img[:, :half_w]
            high_img = img[:, half_w:]
            
            # 掩膜同样对半分
            low_mask = mask[:, :half_w]
            
            # 二值化掩膜 (假设掩膜区域为白色 255)
            _, low_mask_bin = cv2.threshold(low_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 自动纠正反色掩膜
            if np.mean(low_mask_bin) > 240:
                low_mask_bin = cv2.bitwise_not(low_mask_bin)
            
            if category == 'ore':
                # 解析矿石编号逻辑：
                # 1. 优先匹配 Ore-A-B- 模式 (A, B 可以是数字或字母如 PbZn)
                # 注意：只有当 B 部分不包含 'mm' 时才认为是双编号情况
                match = re.search(r'Ore-([^-]+)-([^-]+)-', base_name, re.IGNORECASE)
                if match and 'mm' not in match.group(2).lower():
                    id1, id2 = match.groups()
                    actual_id = id1 if '0.6mm' in tif_name.lower() else id2
                    # 将文件名中的 Ore-A-B- 替换为 Ore-实际ID-
                    save_name = base_name.replace(match.group(0), f"Ore-{actual_id}-")
                elif '01' in base_name:
                    actual_id = '01'
                    save_name = base_name
                else:
                    save_name = base_name

                # Ore 提取全局掩膜区域的像素
                pixels_low = low_img[low_mask_bin == 255]
                # 对高能区域应用同样的掩膜位置
                pixels_high = high_img[low_mask_bin == 255]
                
                m_low = np.mean(pixels_low) if pixels_low.size > 0 else 0
                s_low = np.std(pixels_low) if pixels_low.size > 0 else 0
                m_high = np.mean(pixels_high) if pixels_high.size > 0 else 0
                s_high = np.std(pixels_high) if pixels_high.size > 0 else 0
                
                # 仿照 extract_sample_values.py 保存 pkl 和图
                x, y, w, h = cv2.boundingRect(low_mask_bin)
                box_low = low_img[y:y+h, x:x+w]
                box_high = high_img[y:y+h, x:x+w]
                save_contour_data(output_dir, save_name, 'ore', 0, pixels_low, pixels_high, [box_low, box_high, None])

                all_data.append({
                    "File": tif_name,
                    "Category": "Ore",
                    "Normalization": norm_status,
                    "Step": "All",
                    "Mean_LE": round(float(m_low), 2),
                    "Std_LE": round(float(s_low), 2),
                    "Mean_HE": round(float(m_high), 2),
                    "Std_HE": round(float(s_high), 2)
                })
                
            elif category == 'steps':
                # 寻找轮廓
                contours, _ = cv2.findContours(low_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in contours if cv2.contourArea(c) > 50]
                
                if not cnts:
                    print(f"Warning: No valid mask found in {png_name}")
                    continue

                # 排序规则：从下到上 (从薄到厚)
                # 用户说：上到下变薄。所以底部最薄，顶部最厚。薄到厚 = 从下往上排。
                cnts.sort(key=lambda c: cv2.boundingRect(c)[1], reverse=True)
                
                pixels_low_list = []
                pixels_high_list = []
                m_low_list = []
                m_high_list = []

                if len(cnts) == 1:
                    # 自动切分逻辑
                    print(f"  -> Only 1 bounding contour. Slicing into 10 steps (Bottom to Top).")
                    x_b, y_b, w_b, h_b = cv2.boundingRect(cnts[0])
                    step_h = h_b / 10.0
                    for idx in range(10):
                        # 从下往上切
                        curr_y_end = int(y_b + h_b - idx * step_h)
                        curr_y_start = int(y_b + h_b - (idx + 1) * step_h)
                        
                        step_mask = np.zeros_like(low_mask_bin)
                        cv2.rectangle(step_mask, (x_b, curr_y_start), (x_b+w_b, curr_y_end), 255, -1)
                        step_mask = cv2.bitwise_and(step_mask, low_mask_bin)
                        
                        p_l = low_img[step_mask == 255]
                        p_h = high_img[step_mask == 255]
                        
                        pixels_low_list.append(p_l)
                        pixels_high_list.append(p_h)
                        m_low_list.append(round(float(np.mean(p_l)), 1) if p_l.size > 0 else 0)
                        m_high_list.append(round(float(np.mean(p_h)), 1) if p_h.size > 0 else 0)
                else:
                    if len(cnts) != 10:
                        print(f"  -> Warning: Found {len(cnts)} step contours in {png_name}")
                        
                    for c in cnts:
                        step_mask = np.zeros_like(low_mask_bin)
                        cv2.drawContours(step_mask, [c], -1, 255, -1)
                        
                        p_l = low_img[step_mask == 255]
                        p_h = high_img[step_mask == 255]
                        
                        pixels_low_list.append(p_l)
                        pixels_high_list.append(p_h)
                        m_low_list.append(round(float(np.mean(p_l)), 1) if p_l.size > 0 else 0)
                        m_high_list.append(round(float(np.mean(p_h)), 1) if p_h.size > 0 else 0)

                # 保存为 step_sample 格式
                all_pts = np.concatenate(cnts)
                x_g, y_g, w_g, h_g = cv2.boundingRect(all_pts)
                box_low = low_img[y_g:y_g+h_g, x_g:x_g+w_g]
                box_high = high_img[y_g:y_g+h_g, x_g:x_g+w_g]
                save_contour_data(output_dir, base_name, 'step_sample', 0, pixels_low_list, pixels_high_list, [box_low, box_high, None])

                all_data.append({
                    "File": tif_name,
                    "Category": "Step",
                    "Normalization": norm_status,
                    "Step": "S1-S10 (Thin->Thick)",
                    "Mean_LE": m_low_list,
                    "Std_LE": "-",
                    "Mean_HE": m_high_list,
                    "Std_HE": "-"
                })

    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(output_dir, "extracted_values_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessing complete. Results saved in: {output_dir}")
        print("\n" + "="*90)
        print("EXTRACTION SUMMARY (Thin to Thick)")
        print("="*90)
        print(df.head(20).to_string(index=False))
        print("="*90)

if __name__ == '__main__':
    process_masks_and_extract()
