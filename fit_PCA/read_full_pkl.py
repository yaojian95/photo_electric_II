import pickle
import pandas as pd
import numpy as np
import sys
import os

def load_full_pkl(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
    except Exception as e:
        print("Encountered module error, applying numpy._core hack...")
        import numpy.core.numeric
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = numpy.core.numeric
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
            
    return input_all

def analyze_dataset(input_all):
    print(f"\n=============================================")
    print(f"Dataset contains {len(input_all)} main components:")
    print(f"=============================================\n")
    
    # 1. Pixels Data
    pixels = input_all[0]
    print(f"[Component 0]: pixels")
    print(f"  - Type: {type(pixels)}")
    print(f"  - Length: {len(pixels)} (Channels)")
    for i, ch in enumerate(pixels):
        sample_item = ch.iloc[0] if len(ch) > 0 else None
        print(f"    * Channel {i} -> Type: {type(ch)}, Length: {len(ch)}, Sample pixel array shape: {getattr(sample_item, 'shape', 'N/A')}, dtype: {getattr(sample_item, 'dtype', 'N/A')}")
    print()
    
    # 2. DataFrame (Ground Truth & Morphological Features)
    data = input_all[1]
    print(f"[Component 1]: data")
    print(f"  - Type: {type(data)}")
    print(f"  - Shape: {data.shape} (Number of ores: {data.shape[0]}, Features: {data.shape[1]})")
    print(f"  - Columns sample: {list(data.columns)[:8]} ...")
    print()
    
    # 3. Source list mapping (Usually bounding boxes or image references per source)
    if len(input_all) > 2:
        item2 = input_all[2]
        print(f"[Component 2]: (Additional Data 1)")
        print(f"  - Type: {type(item2)}")
        if isinstance(item2, dict):
            print(f"  - Keys: {list(item2.keys())}")
            first_key = list(item2.keys())[0]
            val = item2[first_key]
            print(f"  - Value under '{first_key}': Type: {type(val)}, Length: {len(val) if isinstance(val, list) else 'N/A'}")
            if isinstance(val, list) and len(val) > 0:
                print(f"    * First element type: {type(val[0])}, shape: {getattr(val[0], 'shape', 'N/A')}")
        print()
        
    # 4. Source dictionary mapping (Usually metadata or background properties per source)
    if len(input_all) > 3:
        item3 = input_all[3]
        print(f"[Component 3]: (Additional Data 2)")
        print(f"  - Type: {type(item3)}")
        if isinstance(item3, dict):
            print(f"  - Keys: {list(item3.keys())}")
            first_key = list(item3.keys())[0]
            val = item3[first_key]
            print(f"  - Value under '{first_key}': Type: {type(val)}")
            if isinstance(val, dict):
                print(f"    * Keys inside this sub-dict: {list(val.keys())}")
                first_sub_key = list(val.keys())[0]
                sub_val = val[first_sub_key]
                print(f"    * Type of inner value: {type(sub_val)}, shape/len: {getattr(sub_val, 'shape', len(sub_val) if hasattr(sub_val, '__len__') else 'N/A')}")
        print()
        
    print("Done analysis.")

if __name__ == '__main__':
    pkl_path = r'E:\photo_electric_II\data\0325_0519_0520_input_cleaned_dataset_le2.pkl'
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
    else:
        dataset = load_full_pkl(pkl_path)
        analyze_dataset(dataset)
