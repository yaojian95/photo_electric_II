import os
import numpy as np
import cv2
import sys
import pandas as pd
from pathlib import Path

# Translation mapping for filenames
TRANS_MAP = {
    "校准前": "pre_calib",
    "校准后": "post_calib",
    "校准": "calib",
    "圆片": "disc",
    "矿石": "ore",
    "副本": "copy",
    "原始": "original",
    "空气": "air",
    "背景": "bg",
    "数据": "data",
    "标准件阶梯":"step",
    "阶梯":"step"
}

def translate_name(name):
    """Translates Chinese characters in filename to English based on TRANS_MAP."""
    new_name = name
    for cn, en in TRANS_MAP.items():
        new_name = new_name.replace(cn, en)
    # Remove potentially problematic non-ascii characters if any remain
    new_name = "".join([c if ord(c) < 128 else "_" for c in new_name])
    return new_name

def convert_txt_to_img(data_root, output_folder="converted_results", ext=".tif", use_16bit=True):
    """
    Recursively finds .txt files, converts 2D data to specified image format.
    Saves all output images to a centralized output_folder under data_root.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    root_path = Path(data_root)
    # Create centralized output directory
    out_path = root_path / output_folder
    out_path.mkdir(parents=True, exist_ok=True)

    precision_str = "16-bit raw" if use_16bit else "8-bit normalized"
    print(f"Starting conversion in: {root_path}")
    print(f"Output folder: {out_path}")
    print(f"Format: {ext}, Precision: {precision_str}")
    
    count_2d = 0
    count_1d = 0
    count_skipped = 0

    for txt_file in root_path.rglob("*.txt"):
        # Skip files already in output folder to avoid infinite recursion or re-processing
        if out_path in txt_file.parents:
            continue
            
        if txt_file.stat().st_size == 0:
            continue

        # Skip files containing 'offset' or 'air' (case-insensitive)
        fn_lower = txt_file.name.lower()
        if "offset" in fn_lower or "air" in fn_lower:
            continue

        try:
            # Load data
            try:
                df = pd.read_csv(txt_file, sep='\t', header=None)
                if df.shape[1] == 1:
                    df = pd.read_csv(txt_file, sep=r'\s+', header=None, engine='python')
            except Exception:
                df = pd.read_csv(txt_file, sep=None, header=None, engine='python')

            data = df.values
            
            if data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 1:
                # Clean NaNs
                data = data[~np.all(np.isnan(data), axis=1)]
                data = data[:, ~np.all(np.isnan(data), axis=0)]
                
                if data.size == 0:
                    count_skipped += 1
                    continue

                if use_16bit:
                    # Clip to uint16 range casting
                    img = np.clip(data, 0, 65535).astype(np.uint16)
                else:
                    # Normalize 0-255 (8-bit)
                    d_min, d_max = np.nanmin(data), np.nanmax(data)
                    if d_max > d_min:
                        img = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(data, dtype=np.uint8)
                
                # Generate translated image filename
                base_name = txt_file.stem
                eng_base = translate_name(base_name)
                # Save to centralized directory
                final_img_path = out_path / (eng_base + ext)
                
                # Robust save for Unicode/Chinese paths (though out_path is likely ascii, txt_file.name might not be)
                ret, buf = cv2.imencode(ext, img)
                if ret:
                    with open(final_img_path, "wb") as f:
                        f.write(buf)
                    print(f"Converted: {txt_file.name} -> {final_img_path.name}")
                    count_2d += 1
                else:
                    print(f"Error encoding image for {txt_file.name}")
                    count_skipped += 1
            else:
                count_1d += 1

        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
            count_skipped += 1

    print("\nProcessing complete:")
    print(f"  2D files converted: {count_2d}")
    print(f"  Output location: {out_path}")
    print(f"  1D files skipped: {count_1d}")
    print(f"  Files with errors: {count_skipped}")

if __name__ == "__main__":
    # Settings
    target_dir = r"E:\multi_source_info\data_dir\20260409_TYM-data"
    output_folder_name = "converted_results" # 集中存放图片的文件夹名
    output_format = ".tif" # 可选 .png, .tif, .jpg 等
    use_16bit_precision = True # 设置为 True 即保持 16 位原始精度
    
    if os.path.exists(target_dir):
        convert_txt_to_img(target_dir, output_folder=output_folder_name, ext=output_format, use_16bit=use_16bit_precision)
    else:
        print(f"Target directory not found: {target_dir}")
