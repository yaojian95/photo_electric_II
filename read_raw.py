import os
import glob
import re
import numpy as np
import cv2

def convert_txt_and_raw_to_png(src_dir: str, dst_dir: str = None, filter_keyword: str = "校准后", width: int = 1024, height: int = 1024):
    r"""
    Reads 16-bit text images (tab/space separated integers) or 16-bit binary RAW images 
    from the source directory (including subfolders), converts them to PNG, and saves them in the destination directory.
    
    Parameters:
    - src_dir (str): Directory containing the files.
      类型: str
      含义: 包含要转换的 .txt 或 .raw 图像数据文件的源目录路径（将递归遍历其子文件夹）。
      用法: 传入绝对路径或相对路径字符串。
    - dst_dir (str, optional): Directory where the converted .png files will be saved. Default is None.
      类型: str (可选)
      含义: 转换后的 .png 文件保存的目标目录路径。如果为 None，则自动设为 src_dir 后拼接 "converted_pngs" 的路径。
      用法: 传入目录路径字符串，或保持默认 None。
    - filter_keyword (str, optional): Keyword that must be present in the filename to convert it. Default is "校准后".
      类型: str (可选)
      含义: 文件过滤关键字。只有当文件名中含有该参数对应的字符串时，才会进行转换。如果为 None 或空字符串，则转换所有匹配的文件。
      用法: 默认值为 "校准后"。
    - width (int): Default expected width of the RAW image if not specified in filename. Default is 1024.
      类型: int
      含义: RAW 图像的默认期望宽度（若文件名中未包含形如 W_H 或 WxH 的尺寸标注，则使用此默认值）。
      用法: 默认值为 1024。
    - height (int): Default expected height of the RAW image if not specified in filename. Default is 1024.
      类型: int
      含义: RAW 图像的默认期望高度（若文件名中未包含形如 W_H 或 WxH 的尺寸标注，则使用此默认值）。
      用法: 默认值为 1024。
    """
    if dst_dir is None:
        dst_dir = os.path.join(src_dir, 'converted_pngs')

    abs_dst_dir = os.path.abspath(dst_dir)

    for root, dirs, filenames in os.walk(src_dir):
        # Skip the destination folder to prevent infinite loop or scanning already converted files
        if os.path.abspath(root).startswith(abs_dst_dir):
            continue

        for filename in filenames:
            if not (filename.endswith('.raw') or filename.endswith('.txt')):
                continue

            # Check if filename contains the keyword
            if filter_keyword and filter_keyword not in filename:
                continue

            file_path = os.path.join(root, filename)
            print(f"Processing: {filename}")
            
            try:
                if file_path.endswith('.raw'):
                    # Read raw file as 16-bit unsigned integer
                    with open(file_path, 'rb') as f:
                        img_data = np.fromfile(f, dtype=np.uint16)
                    
                    # Try to parse dimensions from filename (e.g. '1024_1024' or '1024x1024')
                    match = re.findall(r'(\d+)[_x](\d+)', filename)
                    if match:
                        h, w = int(match[0][0]), int(match[0][1])
                    else:
                        h, w = height, width
                    
                    # Check if size matches expected
                    if img_data.size != w * h:
                        print(f"Warning: {filename} size ({img_data.size}) does not match expected ({w * h}). Skipping.")
                        continue
                    
                    img_data = img_data.reshape((h, w))
                    out_filename = filename.replace('.raw', '.png')
                else:
                    # Load text file as 16-bit unsigned integer array
                    img_data = np.loadtxt(file_path, dtype=np.uint16)
                    out_filename = filename.replace('.txt', '.png')
                
                # Determine output subfolder based on relative path to preserve hierarchy
                rel_dir = os.path.relpath(root, src_dir)
                if rel_dir == '.':
                    file_dst_dir = dst_dir
                else:
                    file_dst_dir = os.path.join(dst_dir, rel_dir)

                if not os.path.exists(file_dst_dir):
                    os.makedirs(file_dst_dir)

                out_filepath = os.path.join(file_dst_dir, out_filename)
                
                # Use cv2.imencode to handle Chinese paths
                is_success, im_buf_arr = cv2.imencode(".png", img_data)
                if is_success:
                    im_buf_arr.tofile(out_filepath)
                    print(f"Saved: {out_filepath}")
                else:
                    print(f"Error encoding {filename}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # source_dir = r"E:\multi_source_info\data_dir\石块线阵20260426\石块线阵20260426"
    # source_dir = r"E:\multi_source_info\data_dir\20260206_TYM"
    source_dir = r"E:\multi_source_info\data_dir\20260206_TYM\TDI探测器"
    convert_txt_and_raw_to_png(source_dir)

