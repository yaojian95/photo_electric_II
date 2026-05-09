import os
import re
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils_II import calculate_effective_z

# 1. Parse fitting parameters from txt file
params_file = 'results/thickness_decoupling/z_decouple/20260331/fitting_parameters.txt'
all_params = {} # all_params[case][voltage] = {intercept, cL, cH, cL2, cLH, cH2}

if not os.path.exists(params_file):
    print(f"Error: {params_file} not found.")
    exit()

with open(params_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_case = None
current_voltage = None

for line in lines:
    line = line.strip()
    if line.startswith('=== Voltage:'):
        # Format: === Voltage: 140kV | Case: Al6_CuFe4 ===
        match = re.search(r'Voltage: (\w+) \| Case: (\w+)', line)
        if match:
            current_voltage = match.group(1)
            current_case = match.group(2)
            if current_case not in all_params:
                all_params[current_case] = {}
            all_params[current_case][current_voltage] = {}
            
    elif line.startswith('Model 2 (Poly) Intercept:'):
        if current_case and current_voltage:
            intercept = float(line.split(':')[1].strip())
            all_params[current_case][current_voltage]['intercept'] = intercept
    elif line.startswith('Coef L:'):
        if current_case and current_voltage:
            all_params[current_case][current_voltage]['cL'] = float(line.split(':')[1].strip())
    elif line.startswith('Coef H:'):
        if current_case and current_voltage:
            all_params[current_case][current_voltage]['cH'] = float(line.split(':')[1].strip())
    elif line.startswith('Coef L^2:'):
        if current_case and current_voltage:
            all_params[current_case][current_voltage]['cL2'] = float(line.split(':')[1].strip())
    elif line.startswith('Coef L H:'):
        if current_case and current_voltage:
            all_params[current_case][current_voltage]['cLH'] = float(line.split(':')[1].strip())
    elif line.startswith('Coef H^2:'):
        if current_case and current_voltage:
            all_params[current_case][current_voltage]['cH2'] = float(line.split(':')[1].strip())

# Define the test dataset directory name
# Use "20260325_yinshan" to test the new Yinshan dataset (first 98 ores)
# Use "20260331" to test the original calibration disks
TEST_DATA_DIR = "20260325_yinshan"
MAX_SAMPLES = 98 if TEST_DATA_DIR == "20260325_yinshan" else 20

# Extract the pure date string (e.g. "20260325") to lookup grades
TEST_DATE = re.search(r'(\d{8})', TEST_DATA_DIR).group(1) if re.search(r'(\d{8})', TEST_DATA_DIR) else "20260331"

# Load disk grades from external config file
config_path = r'E:\multi_source_info\data_dir\disk_grades.json'

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    if TEST_DATE in full_config:
        disk_grades = {int(k): tuple(v) for k, v in full_config[TEST_DATE].items()}
        print(f"Successfully loaded disk grades for validation dataset: {TEST_DATE}")
    else:
        print(f"Warning: Date {TEST_DATE} not found in {config_path}. Using empty grades.")
        disk_grades = {}
else:
    print(f"Error: Config file {config_path} not found. Using empty grades.")
    disk_grades = {}

import pandas as pd

def predict_Z(L, H, p):
    return p['intercept'] + p['cL']*L + p['cH']*H + p['cL2']*(L**2) + p['cLH']*(L*H) + p['cH2']*(H**2)

base_output_dir = f'results/thickness_decoupling/disk_Z_predictions_model2_{TEST_DATA_DIR}'
input_img_dir = f'results/{TEST_DATA_DIR}/high_low_images'
input_pkl_dir = f'results/{TEST_DATA_DIR}/pixel_values'
os.makedirs(base_output_dir, exist_ok=True)

print("\nStarting Z prediction and heatmap generation for disks across all scenarios...")

for case_name, voltages_dict in all_params.items():
    print(f"\n================ Processing Scenario: {case_name} ================")
    
    for voltage, params in voltages_dict.items():
        print(f"  -> Voltage: {voltage}")
        
        # Check if all required coefficients were parsed correctly
        required_keys = ['intercept', 'cL', 'cH', 'cL2', 'cLH', 'cH2']
        if not all(k in params for k in required_keys):
            print(f"     Warning: Missing Model 2 coefficients for {case_name} {voltage}. Skipping.")
            continue
        
        out_dir = os.path.join(base_output_dir, case_name, voltage)
        os.makedirs(out_dir, exist_ok=True)
        heatmap_dir = os.path.join(out_dir, 'heatmaps')
        hist_dir = os.path.join(out_dir, 'histograms')
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(hist_dir, exist_ok=True)
        
        disk_ids = []
        disk_Z_means = []
        disk_Z_stds = []
        
        for d_id in range(1, MAX_SAMPLES + 1):
            # --- 1. Process PKL for 1D Statistics, Histogram, and Mean Z ---
            if TEST_DATA_DIR == "20260325_yinshan":
                pkl_name = f"1_98_position_3_{voltage}_ore_{d_id}_data.pkl"
            else:
                pkl_name = f"{voltage}_4mA_disk_{d_id}_data.pkl"
            pkl_path = os.path.join(input_pkl_dir, pkl_name)
            has_pkl_data = False
            
            if os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                    
                max_pkl_L = 65535 if d['pixels_low'].dtype == np.uint16 else 255
                max_pkl_H = 65535 if d['pixels_high'].dtype == np.uint16 else 255
                
                L_pkl = d['pixels_low'].astype(np.float32)
                H_pkl = d['pixels_high'].astype(np.float32)
                
                valid_pkl = (L_pkl > 1) & (H_pkl > 1) & (L_pkl < max_pkl_L - 1) & (H_pkl < max_pkl_H - 1)
                L_v_pkl = L_pkl[valid_pkl]
                H_v_pkl = H_pkl[valid_pkl]
                
                if len(L_v_pkl) > 0:
                    Z_pred_pkl = predict_Z(L_v_pkl, H_v_pkl, params)
                    vmin_val = np.percentile(Z_pred_pkl, 5)
                    vmax_val = np.percentile(Z_pred_pkl, 95)
                    
                    # Remove extreme outliers (1% on both ends) for robust statistics
                    p1 = np.percentile(Z_pred_pkl, 1)
                    p99 = np.percentile(Z_pred_pkl, 99)
                    robust_mask = (Z_pred_pkl >= p1) & (Z_pred_pkl <= p99)
                    Z_pred_robust = Z_pred_pkl[robust_mask]
                    
                    mean_val = np.mean(Z_pred_robust)
                    std_val = np.std(Z_pred_robust)
                    
                    # Plot Histogram from PKL data
                    plt.figure(figsize=(7, 5))
                    Z_pred_clipped = np.clip(Z_pred_pkl, vmin_val - (vmax_val-vmin_val)*0.5, vmax_val + (vmax_val-vmin_val)*0.5)
                    sns.histplot(Z_pred_clipped, bins=50, color='blue', kde=True, label=f"Robust Mean: {mean_val:.2f}\nRobust Std: {std_val:.2f}")
                    plt.title(f"{case_name} | {voltage} | Disk {d_id} Z-Distribution (PKL)")
                    plt.xlabel("Predicted Equivalent Atomic Number (Z)")
                    plt.ylabel("Pixel Count")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(hist_dir, f"disk_{d_id}_hist.png"), dpi=150)
                    plt.close()
                    
                    disk_ids.append(d_id)
                    disk_Z_means.append(mean_val) # Use robust mean for regression
                    disk_Z_stds.append(std_val)   # Use robust std for regression
                    has_pkl_data = True

            # --- 2. Process PNG for 2D Spatial Heatmap ---
            if TEST_DATA_DIR == "20260325_yinshan":
                low_name = f"1_98_position_3_{voltage}_ore_{d_id}_low.png"
                high_name = f"1_98_position_3_{voltage}_ore_{d_id}_high.png"
            else:
                low_name = f"{voltage}_4mA_disk_{d_id}_low.png"
                high_name = f"{voltage}_4mA_disk_{d_id}_high.png"
            
            low_path = os.path.join(input_img_dir, low_name)
            high_path = os.path.join(input_img_dir, high_name)
            
            if os.path.exists(low_path) and os.path.exists(high_path):
                img_L = cv2.imread(low_path, cv2.IMREAD_ANYDEPTH)
                img_H = cv2.imread(high_path, cv2.IMREAD_ANYDEPTH)
                
                if img_L is not None and img_H is not None:
                    max_val_L = 65535 if img_L.dtype == np.uint16 else 255
                    max_val_H = 65535 if img_H.dtype == np.uint16 else 255
                    
                    img_L = img_L.astype(np.float32)
                    img_H = img_H.astype(np.float32)
                    
                    mask = (img_L > 5) & (img_H > 5) & (img_L < max_val_L - 1) & (img_H < max_val_H - 1)
                    L_v = img_L[mask]
                    H_v = img_H[mask]
                    
                    if len(L_v) > 0:
                        Z_pred_img = predict_Z(L_v, H_v, params)
                        
                        # Use PKL percentiles if available, otherwise fallback to IMG percentiles
                        if not has_pkl_data:
                            vmin_val = np.percentile(Z_pred_img, 5)
                            vmax_val = np.percentile(Z_pred_img, 95)
                            
                        Z_map = np.zeros_like(img_L, dtype=np.float32)
                        Z_map[mask] = Z_pred_img
                        Z_map[~mask] = np.nan
                        
                        plt.figure(figsize=(7, 6))
                        plt.imshow(Z_map, cmap='jet', vmin=vmin_val, vmax=vmax_val)
                        plt.colorbar(label='Predicted Equivalent Atomic Number (Z)')
                        plt.title(f"{case_name} | {voltage} | Disk {d_id} Z-Heatmap (PNG)")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(heatmap_dir, f"disk_{d_id}_heatmap.png"), dpi=150)
                        plt.close()
        
        # --- Z vs Equivalent Z_eff Grade (Formula 2) ---
        eq_Z_cu_fe = []
        eq_Z_cu_fe_s = []
        z_means_for_grades = []
        z_stds_for_grades = []
        
        for d_id, z_mean, z_std in zip(disk_ids, disk_Z_means, disk_Z_stds):
            if d_id in disk_grades:
                cu, fe, s = disk_grades[d_id]
                z_eff_cufe, z_eff_cufes = calculate_effective_z(cu, fe, s)
                
                eq_Z_cu_fe.append(z_eff_cufe)
                eq_Z_cu_fe_s.append(z_eff_cufes)
                z_means_for_grades.append(z_mean)
                z_stds_for_grades.append(z_std)
                
        if z_means_for_grades:
            z_means_arr = np.array(z_means_for_grades)
            ss_tot = np.sum((z_means_arr - np.mean(z_means_arr))**2)
            
            # Combined Plot: Mean Predicted Z vs Equivalent Z_eff
            plt.figure(figsize=(10, 6))
            
            # Plot Eq_Z(Cu+Fe+S) only
            plt.scatter(eq_Z_cu_fe_s, z_means_for_grades, color='tab:green', marker='o', s=20, label='Disk Data (EqZ Cu+Fe+S)')
            
            valid_d_ids = [d for d in disk_ids if d in disk_grades]
            # for i, d_id_val in enumerate(valid_d_ids):
            #     plt.annotate(f"D{d_id_val}", (eq_Z_cu_fe_s[i], z_means_for_grades[i]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=10, color='tab:green')

            if len(eq_Z_cu_fe_s) > 1:
                coeffs_cfs = np.polyfit(eq_Z_cu_fe_s, z_means_for_grades, 1)
                fit_line_cfs = np.poly1d(coeffs_cfs)
                r2_cfs = 1 - np.sum((z_means_arr - fit_line_cfs(eq_Z_cu_fe_s))**2) / ss_tot if ss_tot > 0 else 0
                x_vals_cfs = np.array([min(eq_Z_cu_fe_s), max(eq_Z_cu_fe_s)])
                plt.plot(x_vals_cfs, fit_line_cfs(x_vals_cfs), '-.', color='darkgreen', alpha=0.8, label=f'Fit: Pred_Z={coeffs_cfs[0]:.2f}*$Z_{{eff}}$ + {coeffs_cfs[1]:.2f} ($R^2$={r2_cfs:.3f})')

            plt.title(f"{case_name} | {voltage} - Predicted Z vs Equivalent $Z_{{eff}}$ (Formula 2)", fontsize=14)
            plt.xlabel("Equivalent Atomic Number $Z_{eff}$ (Exponent=2.94, Gangue Z=11)", fontsize=12)
            plt.ylabel("Mean Predicted Equivalent Atomic Number (Z)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{voltage}_Z_vs_EqZ.png"), dpi=150)
            plt.close()
            
print(f"\nDone! Prediction plots, histograms, and heatmaps saved to: {base_output_dir}")
