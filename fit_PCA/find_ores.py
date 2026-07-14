import pickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils_II

def load_data():
    pkl_path = r'E:\photo_electric_II\data\0325_0519_0520_input_cleaned_dataset_le2.pkl'
    try:
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
    except Exception:
        import numpy.core.numeric
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = numpy.core.numeric
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
    return input_all

def main():
    dataset = load_data()
    df = dataset[1]
    pixels_low = dataset[0][0]
    pixels_high = dataset[0][1]
    
    target_ids = [293, 107, 211, 23]
    for g_id in target_ids:
        row_idx = df[df['global_id'] == g_id].index[0]
        row_info = df.iloc[row_idx]
        p_low = pixels_low.iloc[row_idx]
        p_high = pixels_high.iloc[row_idx]
        print(f"\nOre #{g_id} (thickness={row_info['mean_thickness']:.2f}mm, Fe={row_info['Fe']:.2f}%, Cu={row_info['Cu']:.3f}%)")
        print(f"  - p_low: shape={p_low.shape}, min={p_low.min()}, max={p_low.max()}")
        print(f"  - p_high: shape={p_high.shape}, min={p_high.min()}, max={p_high.max()}")

if __name__ == '__main__':
    main()
