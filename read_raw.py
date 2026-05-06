import os
import glob
import numpy as np
import cv2

def convert_raw_to_png(src_dir: str, dst_dir: str, width: int = 1024, height: int = 1024):
    """
    Reads 16-bit RAW images from the source directory, converts them to PNG,
    and saves them in the destination directory.
    
    Parameters:
    - src_dir (str): Directory containing the .raw files.
    - dst_dir (str): Directory where the converted .png files will be saved.
    - width (int): Expected width of the image. Default is 1024.
    - height (int): Expected height of the image. Default is 1024.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    raw_files = glob.glob(os.path.join(src_dir, '*.raw'))
    
    for raw_file in raw_files:
        filename = os.path.basename(raw_file)
        print(f"Processing: {filename}")
        
        # Read the raw file as 16-bit unsigned integer
        try:
            with open(raw_file, 'rb') as f:
                img_data = np.fromfile(f, dtype=np.uint16)
                
            # Check if size matches expected
            if img_data.size != width * height:
                print(f"Warning: {filename} size ({img_data.size}) does not match expected ({width * height}). Skipping.")
                continue
                
            img_data = img_data.reshape((height, width))
            
            out_filename = filename.replace('.raw', '.png')
            out_filepath = os.path.join(dst_dir, out_filename)
            
            # Use cv2.imencode to handle Chinese paths
            is_success, im_buf_arr = cv2.imencode(".png", img_data)
            if is_success:
                im_buf_arr.tofile(out_filepath)
                print(f"Saved: {out_filepath}")
            else:
                print(f"Error encoding {filename}")
                
        except Exception as e:
            print(f"Error processing {raw_file}: {e}")

if __name__ == "__main__":
    source_dir = r"E:\multi_source_info\data_dir\石块线阵20260426\石块线阵20260426"
    dest_dir = r"E:\multi_source_info\data_dir\石块线阵20260426\converted_pngs"
    convert_raw_to_png(source_dir, dest_dir)
