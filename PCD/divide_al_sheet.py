import os
import pickle
import numpy as np
import cv2

def divide_and_extract_al_steps(
    # contour_results_dir: str = r"E:\photo_electric_II\results\20260512_dual_180kV_1mA_no_subtracting_noise\contour_results",
    # contour_results_dir: str = r"E:\photo_electric_II\results\20260512_180kV_1mA_subtracting_noise\contour_results",
    contour_results_dir: str = r"E:\photo_electric_II\results\20260512_160kV_1mA_subtracting_noise\contour_results",
    margin_x: float = 0.15,
    margin_y: float = 0.25,
    scale: float = 0.9
) -> None:
    """
    分出铝阶梯(step3)的左、中、右三个区域，每个区域取出10个厚度的像素数据，并保存为与原格式相同的pkl和png文件。
    并在对应能量段的_dual文件夹中输出可视化的区域划分和采样框标注图像文件。
    
    参数类型、含义及用法：
    ------------------
    参数：
    - contour_results_dir (str): 轮廓结果提取输出的根目录。默认为 E:\\photo_electric_II\\results\\20260512_dual\\contour_results。
    - margin_x (float): 在每个分割区域（左、中、右）内提取像素时，横向左右两端剔除的比例。为了避免铁片分界线和侧边边缘的影响，默认值为 0.15。
    - margin_y (float): 在每个台阶（10层）内提取像素时，纵向上下两端剔除的比例。为了避免台阶过渡区的影响，默认值为 0.25。
    - scale (float): 每个阶梯（台阶）单元格内缩放（缩小）的比例，用于过滤台阶四周边缘的本底噪声和过渡段影响。默认值为 0.9。
    
    用法：
    - 直接在脚本中调用 divide_and_extract_al_steps() 即可对根目录下所有能量段的 step3 数据进行单元格级缩放、自动切割、提取、保存以及区域标注的可视化输出。
    """
    if not os.path.exists(contour_results_dir):
        print(f"Error: Directory {contour_results_dir} does not exist.")
        return

    # 获取所有以 _dual 或 _noNorm_R 结尾的子目录（对应各个能量段）
    subdirs = [d for d in os.listdir(contour_results_dir) 
               if os.path.isdir(os.path.join(contour_results_dir, d)) and (d.endswith('_dual') or d.endswith('_noNorm_R'))]
    
    if not subdirs:
        print(f"No energy band directories (ending with '_dual' or '_noNorm_R') found in {contour_results_dir}")
        return

    print(f"Found {len(subdirs)} energy bands to process: {subdirs}")

    for energy_band in subdirs:
        base_dir = os.path.join(contour_results_dir, energy_band)
        images_dir = os.path.join(base_dir, "high_low_images")
        pixels_dir = os.path.join(base_dir, "pixel_values")

        # 查找铝阶梯（step3）的低能和高能图像
        low_img_path = os.path.join(images_dir, f"{energy_band}_step_sample_3_low.png")
        high_img_path = os.path.join(images_dir, f"{energy_band}_step_sample_3_high.png")

        if not os.path.exists(low_img_path) or not os.path.exists(high_img_path):
            print(f"[{energy_band}] Warning: Al step images not found. Skipping.")
            continue

        # 读取图像（16位，使用 cv2.IMREAD_UNCHANGED）
        low_img = cv2.imread(low_img_path, cv2.IMREAD_UNCHANGED)
        high_img = cv2.imread(high_img_path, cv2.IMREAD_UNCHANGED)

        if low_img is None or high_img is None:
            print(f"[{energy_band}] Error reading images. Skipping.")
            continue

        h, w = low_img.shape[:2]
        
        # 整体区域不再缩小，直接使用原始图像边界
        col_min = 0
        col_max = w
        row_min = 0
        row_max = h

        w_act = w
        h_act = h

        # 确定背景像素值
        bg_val_low = 65535 if low_img.dtype == np.uint16 else 255
        bg_val_high = 65535 if high_img.dtype == np.uint16 else 255

        # 将宽度等分为 3 部分
        w_part = w_act / 3.0
        regions = {
            'left': (col_min, col_min + int(w_part)),
            'mid': (col_min + int(w_part), col_min + int(2.0 * w_part)),
            'right': (col_min + int(2.0 * w_part), col_max)
        }

        # 缓存每个区域的 10 个台阶的像素 data
        region_pixels_low = {r: [] for r in regions}
        region_pixels_high = {r: [] for r in regions}

        # 创建可视化图像，在低能图像上标出三块区域和采样框
        if low_img.dtype == np.uint16:
            vis_img = cv2.cvtColor((low_img / 256).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis_img = cv2.cvtColor(low_img, cv2.COLOR_GRAY2BGR)

        # 绘制整体区域的外边界线 (橙色)
        cv2.rectangle(vis_img, (col_min, row_min), (col_max - 1, row_max - 1), (0, 165, 255), 1)

        # 绘制划分左右中三个区域的黄色分界线 (0, 255, 255)
        cv2.line(vis_img, (col_min + int(w_part), row_min), (col_min + int(w_part), row_max), (0, 255, 255), 1)
        cv2.line(vis_img, (col_min + int(2.0 * w_part), row_min), (col_min + int(2.0 * w_part), row_max), (0, 255, 255), 1)

        # 区域英文标注文本
        cv2.putText(vis_img, "Left", (col_min + 2, row_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(vis_img, "Mid", (col_min + int(w_part) + 2, row_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(vis_img, "Right", (col_min + int(2.0 * w_part) + 2, row_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 按照 10 个台阶的高度进行循环提取
        step_h = h_act / 10.0
        for i in range(10):
            y_start = row_min + int(i * step_h)
            y_end = row_min + int((i + 1) * step_h)
            seg_h = y_end - y_start

            for region_name, (col_start, col_end) in regions.items():
                w_part_i = col_end - col_start
                
                # 计算当前单元格（阶梯内区域）的中心点
                cx = (col_start + col_end) / 2.0
                cy = (y_start + y_end) / 2.0

                # 仅缩小每个阶梯内单元格的采样面积
                w_scaled = w_part_i * scale
                h_scaled = seg_h * scale

                # 应用 margins 剔除边缘
                m_x = int(w_scaled * margin_x)
                m_y = int(h_scaled * margin_y)

                roi_x1 = int(cx - w_scaled / 2.0) + m_x
                roi_x2 = int(cx + w_scaled / 2.0) - m_x
                roi_y1 = int(cy - h_scaled / 2.0) + m_y
                roi_y2 = int(cy + h_scaled / 2.0) - m_y

                # 边界保护
                if roi_x2 <= roi_x1:
                    roi_x1, roi_x2 = col_start, col_end
                if roi_y2 <= roi_y1:
                    roi_y1, roi_y2 = y_start, y_end

                # 在可视化图像上绘制当前台阶当前区域的采样框
                color_map = {
                    'left': (255, 255, 0),  # Cyan
                    'mid': (0, 255, 0),     # Green
                    'right': (0, 0, 255)    # Red
                }
                cv2.rectangle(vis_img, (roi_x1, roi_y1), (roi_x2, roi_y2), color_map[region_name], 1)

                # 裁剪得到采样区域
                sub_low = low_img[roi_y1:roi_y2, roi_x1:roi_x2]
                sub_high = high_img[roi_y1:roi_y2, roi_x1:roi_x2]

                # 展平并过滤掉背景像素值
                flat_low = sub_low.flatten()
                flat_high = sub_high.flatten()

                valid_low = flat_low[flat_low != bg_val_low]
                valid_high = flat_high[flat_high != bg_val_high]

                # 如果全部过滤空了，作为容错使用原数据
                if len(valid_low) == 0:
                    valid_low = flat_low
                if len(valid_high) == 0:
                    valid_high = flat_high

                region_pixels_low[region_name].append(valid_low)
                region_pixels_high[region_name].append(valid_high)

        # 保存为 .pkl 数据和裁剪后的可视化 .png 图像
        for region_name, (col_start, col_end) in regions.items():
            data_to_save = {
                'pixels_low': region_pixels_low[region_name],
                'pixels_high': region_pixels_high[region_name]
            }

            # 1. 保存全称形式的 pkl 文件
            pkl_path = os.path.join(pixels_dir, f"{energy_band}_step_sample_3_{region_name}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_to_save, f)

            # 2. 裁剪并保存完整区域的图像 (不再受限于全局缩小)
            cropped_low = low_img[row_min:row_max, col_start:col_end]
            cropped_high = high_img[row_min:row_max, col_start:col_end]
            cv2.imwrite(os.path.join(images_dir, f"{energy_band}_step_sample_3_{region_name}_low.png"), cropped_low)
            cv2.imwrite(os.path.join(images_dir, f"{energy_band}_step_sample_3_{region_name}_high.png"), cropped_high)

        # 清理旧的缩写形式文件（如果存在）
        for name_variant in ['L', 'M', 'R']:
            old_pkl = os.path.join(pixels_dir, f"{energy_band}_step_sample_3_{name_variant}.pkl")
            if os.path.exists(old_pkl):
                os.remove(old_pkl)
            old_low_png = os.path.join(images_dir, f"{energy_band}_step_sample_3_{name_variant}_low.png")
            if os.path.exists(old_low_png):
                os.remove(old_low_png)
            old_high_png = os.path.join(images_dir, f"{energy_band}_step_sample_3_{name_variant}_high.png")
            if os.path.exists(old_high_png):
                os.remove(old_high_png)

        # 保存区域划分与采样核心的可视化标注效果图到_dual子文件夹中
        cv2.imwrite(os.path.join(base_dir, f"{energy_band}_step_sample_3_division.png"), vis_img)

        print(f"[{energy_band}] Successfully processed Left/Mid/Right steps data (scaled each cell to {scale}) and saved visual annotation.")

    # 打印其中一个能量段的提取结果作为验证
    demo_band = "20_dual"
    if demo_band not in subdirs and len(subdirs) > 0:
        demo_band = subdirs[0]
        
    if demo_band in subdirs:
        print("\n" + "=" * 80)
        print(f"VERIFICATION TABLE FOR {demo_band} step_sample_3 (scaled each cell to {scale}):")
        print("=" * 80)
        print(f"{'Step':<6} | {'Left (Al only) Mean':<20} | {'Mid (Al+0.3mm Fe) Mean':<22} | {'Right (Al+0.6mm Fe) Mean':<24}")
        print("-" * 80)
        
        base_dir = os.path.join(contour_results_dir, demo_band)
        pixels_dir = os.path.join(base_dir, "pixel_values")
        
        # 加载分出来的三个区域
        left_pkl = pickle.load(open(os.path.join(pixels_dir, f"{demo_band}_step_sample_3_left.pkl"), "rb"))
        mid_pkl = pickle.load(open(os.path.join(pixels_dir, f"{demo_band}_step_sample_3_mid.pkl"), "rb"))
        right_pkl = pickle.load(open(os.path.join(pixels_dir, f"{demo_band}_step_sample_3_right.pkl"), "rb"))
        
        for i in range(10):
            mean_l = np.mean(left_pkl['pixels_low'][i])
            mean_m = np.mean(mid_pkl['pixels_low'][i])
            mean_r = np.mean(right_pkl['pixels_low'][i])
            print(f"{i:<6} | {mean_l:<20.1f} | {mean_m:<22.1f} | {mean_r:<24.1f}")
        print("=" * 80)

if __name__ == "__main__":
    divide_and_extract_al_steps()
