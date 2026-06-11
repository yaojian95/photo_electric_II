import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# Configure paths
data_dir = r"E:\multi_source_info\data_dir\20260611_metal_sheet_yanjiuyuan"
file_sticks = os.path.join(data_dir, "160kV_4mA_rawdata_with_sticks.csv")
file_120 = os.path.join(data_dir, "120kV_4mA_rawdata.csv")
file_140 = os.path.join(data_dir, "140kV_4mA_rawdata.csv")
file_160 = os.path.join(data_dir, "160kV_4mA_rawdata.csv")

output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# 1. Analyze sticks to find belt edges
df_sticks = pd.read_csv(file_sticks)
le_sticks = df_sticks['Average'].values[:1536]
he_sticks = df_sticks['Average'].values[1536:3072]

# Find drops in intensity
# We can use negative of the values or just look for minima
# A simple way: smooth the signal, find where it drops significantly compared to center
center_val = np.median(le_sticks[600:900])
threshold = center_val * 0.8
belt_pixels = np.where(le_sticks < threshold)[0]
# Let's find the left and right stick positions by looking for the first sudden drop from the outer edges
try:
    bg_left = np.median(le_sticks[:50])
    bg_right = np.median(le_sticks[1480:1530])
    
    left_edge = np.where(le_sticks[:500] < bg_left * 0.95)[0][0]
    right_edge = 1535 - np.where(le_sticks[::-1][:500] < bg_right * 0.95)[0][0]
    
    peaks = [left_edge, right_edge]
    print(f"Detected sudden drop locations at pixels: {peaks}")
except Exception as e:
    print(f"Error finding sticks: {e}")

# the sticks should be placed at the edges
# let's just plot it to see
plt.figure(figsize=(12, 6))
plt.plot(le_sticks, label='LE Sticks')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
if 'peaks' in locals() and len(peaks) >= 2:
    for i, p in enumerate(peaks):
        label_str = 'Belt Edge' if i == 0 else "_nolegend_"
        plt.axvline(p, color='g', linestyle=':', label=label_str)
        # Add text annotation
        plt.text(p + 15, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1, f'Pixel {p}', color='g', rotation=90, va='bottom')
plt.title("160kV 4mA with Sticks (LE)")
plt.xlabel("Pixel Index")
plt.ylabel("Grayscale Average")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid()
plt.savefig(os.path.join(output_dir, "sticks_le.png"))
plt.close()

# 2. Evaluate inverse square law
# Theoretical curve:
# center = 767.5
# 45 degrees at edge (0 and 1535)
# tan(theta) = (x - 767.5) / D
# At edge x = 1535, theta = 45 degrees => tan(45) = 1 => D = 1535 - 767.5 = 767.5
pixels = np.arange(1536)
D = 767.5
x = pixels - 767.5
theta = np.arctan(x / D)

# r^2 = D^2 + x^2
inverse_square = 1.0 / (D**2 + x**2)
# Normalize to 1 at center
inverse_square_normalized = inverse_square / (1.0 / D**2)

# alternative: flat detector irradiance cos^3(theta)
cos_theta = np.cos(theta)
cos3_normalized = cos_theta**3
cos2_normalized = cos_theta**2

# Load raw data without sticks
df_120 = pd.read_csv(file_120)
df_140 = pd.read_csv(file_140)
df_160 = pd.read_csv(file_160)

for file, df, name in zip([file_120, file_140, file_160], [df_120, df_140, df_160], ['120kV', '140kV', '160kV']):
    le = df['Average'].values[:1536]
    he = df['Average'].values[1536:3072]
    
    # Normalize data using center pixels (e.g. 700-800)
    le_center = np.median(le[700:800])
    he_center = np.median(he[700:800])
    
    le_norm = le / le_center
    he_norm = he / he_center
    
    # Calculate Goodness of Fit (R^2) in the valid region
    valid_mask = (pixels >= 111) & (pixels <= 1419)
    valid_le = le_norm[valid_mask]
    valid_he = he_norm[valid_mask]
    valid_inv_sq = inverse_square_normalized[valid_mask]
    valid_cos3 = cos3_normalized[valid_mask]
    
    # Function to calculate R^2
    def calc_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
        
    r2_le_invsq = calc_r2(valid_le, valid_inv_sq)
    r2_le_cos3 = calc_r2(valid_le, valid_cos3)
    r2_he_invsq = calc_r2(valid_he, valid_inv_sq)
    r2_he_cos3 = calc_r2(valid_he, valid_cos3)

    # LE Plot
    plt.figure(figsize=(12, 8))
    plt.plot(pixels, le_norm, label='LE Data Normalized', alpha=0.8, color='blue')
    plt.plot(pixels, inverse_square_normalized, 'k--', label=f'1/r^2 (R^2={r2_le_invsq:.4f})')
    plt.plot(pixels, cos3_normalized, 'r--', label=f'cos^3 theta (R^2={r2_le_cos3:.4f})')
    
    if 'peaks' in locals() and len(peaks) >= 2:
        for i, p in enumerate(peaks):
            label_str = 'Belt Edge' if i == 0 else "_nolegend_"
            plt.axvline(p, color='g', linestyle=':', label=label_str)
            plt.text(p + 15, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1, f'Pixel {p}', color='g', rotation=90, va='bottom')
            
    plt.title(f"{name} Raw Data vs Theoretical Inverse Square (Low Energy)")
    plt.xlabel("Pixel Index")
    plt.ylabel("Normalized Intensity")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{name}_inverse_square_fit_LE.png"))
    plt.close()

    # HE Plot
    plt.figure(figsize=(12, 8))
    plt.plot(pixels, he_norm, label='HE Data Normalized', alpha=0.8, color='orange')
    plt.plot(pixels, inverse_square_normalized, 'k--', label=f'1/r^2 (R^2={r2_he_invsq:.4f})')
    plt.plot(pixels, cos3_normalized, 'r--', label=f'cos^3 theta (R^2={r2_he_cos3:.4f})')
    
    if 'peaks' in locals() and len(peaks) >= 2:
        for i, p in enumerate(peaks):
            label_str = 'Belt Edge' if i == 0 else "_nolegend_"
            plt.axvline(p, color='g', linestyle=':', label=label_str)
            plt.text(p + 15, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1, f'Pixel {p}', color='g', rotation=90, va='bottom')
            
    plt.title(f"{name} Raw Data vs Theoretical Inverse Square (High Energy)")
    plt.xlabel("Pixel Index")
    plt.ylabel("Normalized Intensity")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{name}_inverse_square_fit_HE.png"))
    plt.close()

print("Analysis complete. Check results in", output_dir)
