import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

materials = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
voltages = ['140kV', '160kV', '180kV']

output_dir = 'results/thickness_decoupling'
os.makedirs(output_dir, exist_ok=True)

for voltage in voltages:
    plt.figure(figsize=(10, 8))
    for idx, mat_name in materials.items():
        file_path = f'results/20260331/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        L = data['pixels_low']
        H = data['pixels_high']
        
        # Filter valid pixels
        valid = (L > 10) & (H > 10) & (L < 255) & (H < 255)
        L_valid = L[valid]
        H_valid = H[valid]
        
        # Subsample for faster fitting and plotting
        if len(L_valid) > 10000:
            indices = np.random.choice(len(L_valid), 10000, replace=False)
            L_sub = L_valid[indices]
            H_sub = H_valid[indices]
        else:
            L_sub = L_valid
            H_sub = H_valid
            
        # Fit polynomial H = a L^2 + b L + c
        # np.polyfit returns coefficients [a, b, c] for a*x^2 + b*x + c
        coeffs = np.polyfit(L_sub, H_sub, 2)
        fit_fn = np.poly1d(coeffs)
        
        # Calculate R-squared
        H_pred = fit_fn(L_sub)
        ss_res = np.sum((H_sub - H_pred)**2)
        ss_tot = np.sum((H_sub - np.mean(H_sub))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"Voltage: {voltage}, Material: {mat_name}")
        print(f"  Fit: H = {coeffs[0]:.4e}*L^2 + {coeffs[1]:.4e}*L + {coeffs[2]:.4e}")
        print(f"  R^2: {r2:.4f}")
        
        # Plot
        plt.scatter(L_sub, H_sub, alpha=0.1, label=f"{mat_name} scatter")
        
        L_range = np.linspace(min(L_sub), max(L_sub), 100)
        plt.plot(L_range, fit_fn(L_range), label=f"{mat_name} fit: {coeffs[0]:.2e}*L^2 + ...", linewidth=2)
        
    plt.xlabel("Low Energy (L)")
    plt.ylabel("High Energy (H)")
    plt.title(f"{voltage} H vs L for Cu, Fe, Al")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{voltage}_H_vs_L_fit.png")
    plt.close()
    
print(f"Analysis complete. Plots saved to {output_dir}")
