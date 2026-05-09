import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils_II import calculate_effective_z

def simple_weighted_z(cu, fe, s, z_base=11.0):
    """
    计算原子序数简单加权平均：Z_eff = sum(w_i * Z_i)
    """
    w_cu = cu / 100.0
    w_fe = fe / 100.0
    w_s = s / 100.0
    w_gangue = max(0.0, 1.0 - w_cu - w_fe - w_s)
    
    Z_CU = 29.0
    Z_FE = 26.0
    Z_S = 16.0
    
    return w_cu * Z_CU + w_fe * Z_FE + w_s * Z_S + w_gangue * z_base

def main():
    config_path = r'E:\multi_source_info\data_dir\disk_grades.json'
    target_date = "20260325"
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
        
    if target_date not in full_config:
        print(f"Error: {target_date} not found in config.")
        return
        
    disk_grades = {int(k): tuple(v) for k, v in full_config[target_date].items()}
    
    # Sort by Disk ID
    d_ids = sorted(disk_grades.keys())
    
    # Storage for different methods
    z_baseline = []       # Mayneord (Cu+Fe+S), exp=2.94, z_base=11.0
    z_no_s = []           # Mayneord (Cu+Fe), exp=2.94, z_base=11.0
    z_simple = []         # Simple weighted average (exp=1.0)
    z_exp35 = []          # Mayneord, exp=3.5
    z_base14 = []         # Mayneord, z_base=14.0 (Si)
    
    valid_ids = []
    
    for d_id in d_ids:
        # 0325 test set focus: 1-98
        if d_id > 98:
            continue
            
        cu, fe, s = disk_grades[d_id]
        
        # 1. Baseline: Mayneord Cu+Fe+S (exp=2.94, base=11)
        _, z_base_val = calculate_effective_z(cu, fe, s, z_base=11.0, exponent=2.94)
        z_baseline.append(z_base_val)
        
        # 2. No Sulphur: Mayneord Cu+Fe (exp=2.94, base=11)
        z_no_s_val, _ = calculate_effective_z(cu, fe, s, z_base=11.0, exponent=2.94)
        z_no_s.append(z_no_s_val)
        
        # 3. Simple Weighted Average (Mass fraction weighted)
        z_simple_val = simple_weighted_z(cu, fe, s, z_base=11.0)
        z_simple.append(z_simple_val)
        
        # 4. Exponent 3.5
        _, z_exp35_val = calculate_effective_z(cu, fe, s, z_base=11.0, exponent=3.5)
        z_exp35.append(z_exp35_val)
        
        # 5. Base Z = 14 (Si)
        _, z_base14_val = calculate_effective_z(cu, fe, s, z_base=14.0, exponent=2.94)
        z_base14.append(z_base14_val)
        
        valid_ids.append(d_id)
        
    # Convert to numpy arrays for vector operations
    z_baseline = np.array(z_baseline)
    z_no_s = np.array(z_no_s)
    z_simple = np.array(z_simple)
    z_exp35 = np.array(z_exp35)
    z_base14 = np.array(z_base14)
    
    # Create Output Directory
    out_dir = r"E:\photo_electric_II\results\thickness_decoupling\zeff_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------- Plotting ----------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Absolute Values Overview
    ax = axes[0, 0]
    ax.plot(valid_ids, z_baseline, 'o-', color='tab:blue', label='Baseline (Mayneord, Exp=2.94, S included)', markersize=4, alpha=0.8)
    ax.plot(valid_ids, z_simple, 's-', color='tab:orange', label='Simple Mass Weighted (Exp=1.0)', markersize=4, alpha=0.8)
    ax.plot(valid_ids, z_exp35, '^-', color='tab:green', label='High Exponent (Exp=3.5)', markersize=4, alpha=0.8)
    ax.set_title("Absolute $Z_{eff}$ Values Across Ore Samples", fontsize=12)
    ax.set_xlabel("Ore ID")
    ax.set_ylabel("Effective Atomic Number ($Z_{eff}$)")
    ax.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Impact of Including Sulphur
    ax = axes[0, 1]
    diff_s = z_baseline - z_no_s
    ax.scatter(valid_ids, diff_s, color='tab:red', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Impact of Sulphur: $Z_{eff}$(Cu+Fe+S) - $Z_{eff}$(Cu+Fe)", fontsize=12)
    ax.set_xlabel("Ore ID")
    ax.set_ylabel("$\Delta Z_{eff}$")
    # Add text box with mean difference
    ax.text(0.05, 0.95, f"Mean $\Delta$: {np.mean(diff_s):.3f}\nMax $\Delta$: {np.max(diff_s):.3f}", 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    # Plot 3: Simple Weighted vs Mayneord
    ax = axes[1, 0]
    diff_simple = z_baseline - z_simple
    ax.scatter(valid_ids, diff_simple, color='tab:purple', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Impact of Physics Model: Mayneord (2.94) vs Simple Weighted (1.0)", fontsize=12)
    ax.set_xlabel("Ore ID")
    ax.set_ylabel("$\Delta Z_{eff}$ (Mayneord - Simple)")
    ax.text(0.05, 0.95, f"Mean $\Delta$: {np.mean(diff_simple):.3f}\nMin $\Delta$: {np.min(diff_simple):.3f}", 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    # Plot 4: Impact of Gangue Base Z
    ax = axes[1, 1]
    diff_base = z_base14 - z_baseline
    ax.scatter(valid_ids, diff_base, color='tab:brown', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Impact of Gangue: Si Base (Z=14) - Na/Mg/Al Base (Z=11)", fontsize=12)
    ax.set_xlabel("Ore ID")
    ax.set_ylabel("$\Delta Z_{eff}$")
    ax.text(0.05, 0.95, f"Mean $\Delta$: {np.mean(diff_base):.3f}\nMin $\Delta$: {np.min(diff_base):.3f}", 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle("Comparison of Effective Atomic Number ($Z_{eff}$) Calculation Methods (Yinshan 0325 Dataset)", fontsize=16)
    plt.tight_layout()
    
    out_file = os.path.join(out_dir, "Zeff_Methods_Comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {out_file}")
    plt.close()

if __name__ == "__main__":
    main()
