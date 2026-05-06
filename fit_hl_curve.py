import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

materials = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
voltages = ['140kV', '160kV', '180kV']

output_dir = 'results/thickness_decoupling'
os.makedirs(output_dir, exist_ok=True)

def perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, limits=None):
    """
    通用 2x3 综合分析绘图函数
    limits: { 'raw_x': (min, max), 'raw_y': (min, max), 'log_x': (min, max), 'log_y': (min, max) }
    """
    os.makedirs(output_subdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    I0 = 204.0

    # 自动寻找线性区间算法
    def find_linear_pts(x_pts, y_pts, label=""):
        best_n = 3
        if len(x_pts) < 3: return len(x_pts)
        prev_r2 = 1.0
        for n in range(3, len(x_pts) + 1):
            cur_x, cur_y = x_pts[:n], y_pts[:n]
            c = np.polyfit(cur_x, cur_y, 1)
            f = np.poly1d(c)
            ss_res = np.sum((cur_y - f(cur_x))**2)
            ss_tot = np.sum((cur_y - np.mean(cur_y))**2)
            cur_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            if n > 3 and (cur_r2 < 0.99 or cur_r2 < prev_r2 - 0.005):
                print(f"  [Linear Range Alert] {label} stopped at n={n-1} ({prev_r2:.4f} -> {cur_r2:.4f})")
                break
            best_n, prev_r2 = n, cur_r2
        return best_n

    for mat_name, (L_list, H_list) in samples_dict.items():
        step_L_means = [np.mean(l) if l.size > 0 else np.nan for l in L_list]
        step_L_stds = [np.std(l) if l.size > 0 else np.nan for l in L_list]
        step_H_means = [np.mean(h) if h.size > 0 else np.nan for h in H_list]
        step_H_stds = [np.std(h) if h.size > 0 else np.nan for h in H_list]
        
        L_all = np.concatenate(L_list).astype(np.float32)
        H_all = np.concatenate(H_list).astype(np.float32)
        
        valid = (L_all > 1) & (H_all > 1) & (L_all < 255) & (H_all < 255)
        L_v, H_v = L_all[valid], H_all[valid]
        
        cur_x_vals = x_coords_dict[mat_name][:len(L_list)]
        plot_x = cur_x_vals - 10 if 'Al' in mat_name and 'step' in title_prefix.lower() else cur_x_vals
        display_label = f"{mat_name}" + (" (t-10mm)" if 'Al' in mat_name and 'step' in title_prefix.lower() else "")

        # Row 1: Raw Intensity
        if len(L_v) > 0:
            axes[0, 0].scatter(L_v, H_v, alpha=0.05, s=0.5)
            coeffs = np.polyfit(L_v, H_v, 2)
            axes[0, 0].plot(np.sort(L_v), np.poly1d(coeffs)(np.sort(L_v)), label=display_label)
        
        axes[0, 1].errorbar(plot_x, step_L_means, yerr=step_L_stds, fmt='o-', capsize=3, label=display_label, alpha=0.8)
        axes[0, 2].errorbar(plot_x, step_H_means, yerr=step_H_stds, fmt='o-', capsize=3, label=display_label, alpha=0.8)

        # Row 2: Log Transform
        log_L_v, log_H_v = np.log(I0 / np.maximum(L_v, 1e-6)), np.log(I0 / np.maximum(H_v, 1e-6))
        log_L_means = np.log(I0 / np.array(step_L_means))
        log_H_means = np.log(I0 / np.array(step_H_means))

        if len(log_L_v) > 0:
            axes[1, 0].scatter(log_L_v, log_H_v, alpha=0.05, s=0.5)
            l_coeffs = np.polyfit(log_L_v, log_H_v, 2)
            axes[1, 0].plot(np.sort(log_L_v), np.poly1d(l_coeffs)(np.sort(log_L_v)), label=display_label)

        line_color = axes[0, 1].get_lines()[-1].get_color()
        n_l = find_linear_pts(cur_x_vals, log_L_means, f"{mat_name} Low-E")
        l_fit = np.poly1d(np.polyfit(cur_x_vals[:n_l], log_L_means[:n_l], 1))
        axes[1, 1].plot(plot_x[:n_l], l_fit(cur_x_vals[:n_l]), '--', color=line_color, label=f"{display_label} (n={n_l})")
        axes[1, 1].plot(plot_x, log_L_means, 'o', color=line_color, alpha=0.3)

        n_h = find_linear_pts(cur_x_vals, log_H_means, f"{mat_name} High-E")
        h_fit = np.poly1d(np.polyfit(cur_x_vals[:n_h], log_H_means[:n_h], 1))
        axes[1, 2].plot(plot_x[:n_h], h_fit(cur_x_vals[:n_h]), '--', color=line_color, label=f"{display_label} (n={n_h})")
        axes[1, 2].plot(plot_x, log_H_means, 'o', color=line_color, alpha=0.3)

    # Apply Limits
    if limits:
        axes[0, 0].set_xlim(limits['raw_x']); axes[0, 0].set_ylim(limits['raw_y'])
        axes[0, 1].set_ylim(limits['raw_y'])
        axes[0, 2].set_ylim(limits['raw_y'])
        axes[1, 0].set_xlim(limits['log_x']); axes[1, 0].set_ylim(limits['log_y'])
        axes[1, 1].set_ylim(limits['log_y'])
        axes[1, 2].set_ylim(limits['log_y'])

    # Styles
    axes[0, 0].set_title("H vs L Fit")
    axes[0, 1].set_title(f"{x_label} vs Low Energy")
    axes[0, 2].set_title(f"{x_label} vs High Energy")
    axes[1, 0].set_title(r"$\ln(I_0/H)$ vs $\ln(I_0/L)$ Fit")
    axes[1, 1].set_title(f"{x_label} vs " + r"$\ln(I_0/L)$")
    axes[1, 2].set_title(f"{x_label} vs " + r"$\ln(I_0/H)$")
    
    for r in range(2): 
        for c in range(3): 
            axes[r, c].grid(True); axes[r, c].legend(fontsize='x-small')

    plt.suptitle(f"{title_prefix} Analysis for {voltage} (I0={I0})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_subdir}/{voltage}_analysis.png")
    plt.close()

# --- Main Execution ---
voltages = ['140kV', '160kV', '180kV']
base_results_dir = 'results/thickness_decoupling/H_L_fit'
step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
thicknesses = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step_block': np.arange(12, 32, 2) }

# 定义不同组的显示限制
step_limits = {'raw_x': (0, 120),  'raw_y': (0, 130),  'log_x': (0.5, 5.0), 'log_y': (0.5, 5.0)}
disk_limits = {'raw_x': (140, 210), 'raw_y': (140, 210), 'log_x': (0, 0.4),   'log_y': (0, 0.4)}

for voltage in voltages:
    print(f"\n>>> Processing {voltage} ...")
    
    # 1. Step Samples Analysis
    step_data = {}
    for idx, name in step_mats.items():
        p = f'results/20260331/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
        if os.path.exists(p):
            with open(p, 'rb') as f:
                d = pickle.load(f)
                step_data[name] = (d['pixels_low'], d['pixels_high'])
    
    if step_data:
        perform_comprehensive_analysis(voltage, step_data, f"{base_results_dir}/steps", 
                                       "Step Sample", "Thickness (mm)", thicknesses, limits=step_limits)

    # 2. Disk Samples Analysis (IDs 9-17)
    disk_ids = range(9, 18)
    disk_L_list, disk_H_list = [], []
    for d_id in disk_ids:
        p = f'results/20260331/pixel_values/{voltage}_4mA_disk_{d_id}_data.pkl'
        if os.path.exists(p):
            with open(p, 'rb') as f:
                d = pickle.load(f)
                disk_L_list.append(d['pixels_low'])
                disk_H_list.append(d['pixels_high'])
    
    if disk_L_list:
        disk_data = { "Mixed_Disks": (disk_L_list, disk_H_list) }
        disk_x_coords = { "Mixed_Disks": np.array(list(disk_ids)) }
        perform_comprehensive_analysis(voltage, disk_data, f"{base_results_dir}/disks", 
                                       "Disk Sample (IDs 9-17)", "Disk ID (Grade proxy)", disk_x_coords, limits=disk_limits)

print(f"\nAnalysis complete. Results saved in {base_results_dir}")
