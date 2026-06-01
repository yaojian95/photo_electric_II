import cv2
import numpy as np
import os
import glob

def auto_crop_xrt_16bit(input_dir, output_dir, std_threshold=3000.0, margin=50):
    r"""
    自动识别并裁剪16位XRT图像上下均匀灰度块
    
    参数解释:
    --------------------------------------------------
    input_dir : str
        类型: 字符串 (str)
        含义: 原始tif图片所在文件夹的绝对或相对路径。
        用法: 例如 r"E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test"
        
    output_dir : str
        类型: 字符串 (str)
        含义: 裁剪后图片保存目标文件夹的路径。如果文件夹不存在，函数会自动创建。
        用法: 例如 r"E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test_cropped"
        
    std_threshold : float, 可选
        类型: 浮点数 (float)
        含义: 识别有效数据区的行像素标准差阈值。
        用法: 对于16位高动态范围灰度图像（0-65535），探测器的随机本底噪声产生的行标准差通常在 800 - 1500 之间。
              因此，该阈值应设置在 2000 - 5000 之间。默认值为 3000.0。
              如果设得过低（例如 <= 1000），则会把纯空气背景行也误判为有效数据，导致无法裁剪。
              
    margin : int, 可选
        类型: 整数 (int)
        含义: 裁剪上下边界的外扩安全边距（以像素为单位）。
        用法: 默认值为 50。在检测到的有效样品行区域之上和之下，各自多保留 margin 行像素，
              防止因边缘阈值截断而切除样品的微弱边缘，保证图像完整性。
    --------------------------------------------------
    """
    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有tif文件 (支持 .tif 和 .tiff)
    search_pattern = os.path.join(input_dir, '*.[tT][iI][fF]*')
    img_paths = glob.glob(search_pattern)

    if not img_paths:
        print(f"在 {input_dir} 下没有找到TIFF文件。")
        return

    print(f"找到 {len(img_paths)} 张图片，开始处理...")

    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        
        # 使用 cv2.IMREAD_UNCHANGED 确保以原始16位深度读取数据
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"无法读取图片: {file_name}，已跳过。")
            continue
            
        if img.dtype != np.uint16:
            print(f"警告: {file_name} 不是16位无符号整型 (当前dtype: {img.dtype})")

        # 核心逻辑：计算每一行的像素标准差
        row_stds = np.std(img, axis=1)

        # 找到所有标准差大于阈值的行索引 (即非均匀灰度块的区域)
        valid_rows = np.where(row_stds > std_threshold)[0]

        if len(valid_rows) == 0:
            print(f"警告: {file_name} 中未检测到有效数据区 (可能全是纯灰色)，已跳过。")
            continue

        # 获取矿石数据的上下边界并外扩 margin 行以确保安全
        top_edge = max(0, valid_rows[0] - margin)
        bottom_edge = min(img.shape[0] - 1, valid_rows[-1] + margin)

        # 沿着高度方向进行裁剪
        cropped_img = img[top_edge:bottom_edge+1, :]

        # 构造输出路径并保存为16位TIFF
        output_path = os.path.join(output_dir, file_name.replace(".tif", "_cropped.tif"))
        cv2.imwrite(output_path, cropped_img)
        
        print(f"成功处理: {file_name} -> 原尺寸 {img.shape}, 裁剪后 {cropped_img.shape} (保留安全边距: {margin})")

    print("所有批处理任务已完成！")

# ==========================================
# 使用示例：修改这里的路径为你的实际文件夹路径
# ==========================================
if __name__ == "__main__":
    # 输入文件夹路径 (例如: './raw_xrt_data')
    INPUT_FOLDER = r"E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test" 
    # 输出文件夹路径 (例如: './cropped_xrt_data')
    OUTPUT_FOLDER = r"E:\multi_source_info\data_dir\20260409_TYM-data\TYM_test" 
    
    # 运行函数，采用修正后的16位适配阈值与安全保留边距
    auto_crop_xrt_16bit(INPUT_FOLDER, OUTPUT_FOLDER, std_threshold=3000.0, margin=50)