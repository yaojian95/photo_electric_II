import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import glob

def perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, limits=None, color_by_step=False, plot_mode='all', I0=204.0):
    """
    通用 2x3 综合分析绘图函数
    """
    os.makedirs(output_subdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 计算自适应坐标限制 (扫描所有数据)
    all_L_pts, all_H_pts = [], []
    all_log_L_pts, all_log_H_pts = [], []
    all_x_vals = []
    
    for mat_name, (L_list, H_list) in samples_dict.items():
        cur_x = x_coords_dict[mat_name][:len(L_list)]
        if 'Al' in mat_name and 'step' in title_prefix.lower(): cur_x = cur_x - 10
        all_x_vals.append(cur_x)
        
        for l, h in zip(L_list, H_list):
            valid = (l >= 0) & (h >= 0) & (l < 255) & (h < 255)
            if np.any(valid):
                lv, hv = l[valid].astype(np.float32), h[valid].astype(np.float32)
                if plot_mode == 'means':
                    # 只采集各样本的均值点，以便坐标轴紧凑
                    m_l, m_h = np.mean(lv), np.mean(hv)
                    all_L_pts.append(m_l); all_H_pts.append(m_h)
                    all_log_L_pts.append(np.log(I0 / max(m_l, 1e-6))); all_log_H_pts.append(np.log(I0 / max(m_h, 1e-6)))
                else:
                    # 采集所有像素
                    all_L_pts.append(lv); all_H_pts.append(hv)
                    all_log_L_pts.append(np.log(I0 / np.maximum(lv, 1e-6))); all_log_H_pts.append(np.log(I0 / np.maximum(hv, 1e-6)))

    if not all_L_pts: 
        plt.close()
        return

    def get_robust_range(data_list):
        if isinstance(data_list[0], np.ndarray):
            combined = np.concatenate(data_list)
            return np.percentile(combined, [2, 98]) # 使用2%-98%分位数，避免极端噪声拉伸坐标轴
        else:
            arr = np.array(data_list)
            return np.min(arr), np.max(arr)

    l_min, l_max = get_robust_range(all_L_pts)
    h_min, h_max = get_robust_range(all_H_pts)
    raw_min, raw_max = min(l_min, h_min), max(l_max, h_max)
    pad_raw = (raw_max - raw_min) * 0.15
    raw_lims = (max(0, raw_min - pad_raw), min(255, raw_max + pad_raw))

    log_l_min, log_l_max = get_robust_range(all_log_L_pts)
    log_h_min, log_h_max = get_robust_range(all_log_H_pts)
    log_min, log_max = min(log_l_min, log_h_min), max(log_l_max, log_h_max)
    pad_log = (log_max - log_min) * 0.15
    log_lims = (max(0, log_min - pad_log), log_max + pad_log)

    X_glob = np.concatenate(all_x_vals)
    x_min, x_max = X_glob.min(), X_glob.max()
    pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    x_lims = (x_min - pad_x, x_max + pad_x)

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
            if n > 3 and (cur_r2 < 0.98 or cur_r2 < prev_r2 - 0.015):
                # print(f"  [Linear Range Alert] {label} stopped at n={n-1} ({prev_r2:.4f} -> {cur_r2:.4f})")
                break
            best_n, prev_r2 = n, cur_r2
        return best_n

    for mat_name, (L_list, H_list) in samples_dict.items():
        # 计算每个样本的均值和标准差 (过滤 1 和 255)
        step_L_means = []
        step_L_stds = []
        step_H_means = []
        step_H_stds = []
        for l, h in zip(L_list, H_list):
            v_idx = (l >= 0) & (h >= 0) & (l < 255) & (h < 255)
            if np.any(v_idx):
                lv, hv = l[v_idx], h[v_idx]
                step_L_means.append(np.mean(lv)); step_L_stds.append(np.std(lv))
                step_H_means.append(np.mean(hv)); step_H_stds.append(np.std(hv))
            else:
                step_L_means.append(np.nan); step_L_stds.append(np.nan)
                step_H_means.append(np.nan); step_H_stds.append(np.nan)

        step_L_means, step_L_stds = np.array(step_L_means), np.array(step_L_stds)
        step_H_means, step_H_stds = np.array(step_H_means), np.array(step_H_stds)

        # 扁平化数据进行散点图绘制
        valid_flat = []
        L_flat_valid = []
        H_flat_valid = []
        for l, h in zip(L_list, H_list):
            v = (l >= 0) & (h >= 0) & (l < 255) & (h < 255)
            if np.any(v):
                L_flat_valid.append(l[v])
                H_flat_valid.append(h[v])
                
        if L_flat_valid:
            L_v = np.concatenate(L_flat_valid).astype(np.float32)
            H_v = np.concatenate(H_flat_valid).astype(np.float32)
        else:
            L_v, H_v = np.array([]), np.array([])
        
        cur_x_vals = x_coords_dict[mat_name][:len(L_list)]
        plot_x = cur_x_vals - 10 if 'Al' in mat_name and 'step' in title_prefix.lower() else cur_x_vals
        display_label = f"{mat_name}" + (" (t-10mm)" if 'Al' in mat_name and 'step' in title_prefix.lower() else "")

        # 先绘制 axes[0, 1] 以获取该 material 的 base_color
        eb_alpha = 0.3 if plot_mode == 'means' else 0.6
        line, = axes[0, 1].plot(plot_x, step_L_means, 'o-', markersize=5, label=display_label, linewidth=1.5)
        base_color = line.get_color()
        axes[0, 1].errorbar(plot_x, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)
        
        axes[0, 2].plot(plot_x, step_H_means, 'o-', markersize=5, label=display_label, linewidth=1.5, color=base_color)
        axes[0, 2].errorbar(plot_x, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)

        # Row 1: Raw Intensity
        if len(L_v) > 0:
            if color_by_step:
                cmap = plt.get_cmap('tab10' if len(L_list) <= 10 else 'tab20')
                for i, (l, h) in enumerate(zip(L_list, H_list)):
                    valid_i = (l >= 0) & (h >= 0) & (l < 255) & (h < 255)
                    if np.any(valid_i):
                        if plot_mode == 'all':
                            axes[0, 0].scatter(l[valid_i], h[valid_i], color=cmap(i), alpha=0.05, s=0.5, label=f"ID:{cur_x_vals[i]:.1f}" if i < len(cur_x_vals) else None)
                        else: # 'means'
                            m_l, m_h = np.mean(l[valid_i]), np.mean(h[valid_i])
                            s_l, s_h = np.std(l[valid_i]), np.std(h[valid_i])
                            axes[0, 0].errorbar(m_l, m_h, xerr=s_l, yerr=s_h, fmt='none', color=cmap(i), capsize=2, alpha=0.3)
                            axes[0, 0].scatter(m_l, m_h, color=cmap(i), s=40, label=f"ID:{cur_x_vals[i]:.1f}" if i < len(cur_x_vals) else None, edgecolors='none')
                fit_color = 'black'
            else:
                if plot_mode == 'all':
                    axes[0, 0].scatter(L_v, H_v, color=base_color, alpha=0.05, s=0.5)
                else:
                    m_L, m_H = np.mean(L_v), np.mean(H_v)
                    s_L, s_H = np.std(L_v), np.std(H_v)
                    axes[0, 0].errorbar(m_L, m_H, xerr=s_L, yerr=s_H, fmt='none', color=base_color, capsize=2, alpha=0.3)
                    axes[0, 0].scatter(m_L, m_H, color=base_color, s=40, edgecolors='none')
                fit_color = base_color
            
            # 使用均值进行多项式拟合
            valid_m = ~np.isnan(step_L_means) & ~np.isnan(step_H_means)
            if np.sum(valid_m) > 2:
                coeffs = np.polyfit(step_L_means[valid_m], step_H_means[valid_m], 2)
                x_fit = np.linspace(raw_lims[0], raw_lims[1], 100)
                axes[0, 0].plot(x_fit, np.poly1d(coeffs)(x_fit), color=fit_color, label=f"{display_label} Fit")

        # Row 2: Log Transform
        log_L_v, log_H_v = np.log(I0 / np.maximum(L_v, 1e-6)), np.log(I0 / np.maximum(H_v, 1e-6))
        log_L_means = np.log(I0 / np.maximum(np.array(step_L_means), 1e-6))
        log_H_means = np.log(I0 / np.maximum(np.array(step_H_means), 1e-6))

        if len(log_L_v) > 0:
            if color_by_step:
                cmap = plt.get_cmap('tab10' if len(L_list) <= 10 else 'tab20')
                for i, (l, h) in enumerate(zip(L_list, H_list)):
                    valid_i = (l >= 0) & (h >= 0) & (l < 255) & (h < 255)
                    if np.any(valid_i):
                        ll = np.log(I0 / np.maximum(l[valid_i], 1e-6))
                        hh = np.log(I0 / np.maximum(h[valid_i], 1e-6))
                        if plot_mode == 'all':
                            axes[1, 0].scatter(ll, hh, color=cmap(i), alpha=0.05, s=0.5)
                        else: # 'means'
                            m_ll, m_hh = np.mean(ll), np.mean(hh)
                            s_ll, s_hh = np.std(ll), np.std(hh)
                            axes[1, 0].errorbar(m_ll, m_hh, xerr=s_ll, yerr=s_hh, fmt='none', color=cmap(i), capsize=2, alpha=0.3)
                            axes[1, 0].scatter(m_ll, m_hh, color=cmap(i), s=40, edgecolors='none')
                fit_color = 'black'
            else:
                if plot_mode == 'all':
                    axes[1, 0].scatter(log_L_v, log_H_v, color=base_color, alpha=0.05, s=0.5)
                else:
                    m_ll, m_hh = np.mean(log_L_v), np.mean(log_H_v)
                    s_ll, s_hh = np.std(log_L_v), np.std(log_H_v)
                    axes[1, 0].errorbar(m_ll, m_hh, xerr=s_ll, yerr=s_hh, fmt='none', color=base_color, capsize=2, alpha=0.3)
                    axes[1, 0].scatter(m_ll, m_hh, color=base_color, s=40, edgecolors='none')
                fit_color = base_color
                
            # 使用均值进行多项式拟合
            valid_m = ~np.isnan(log_L_means) & ~np.isnan(log_H_means) & np.isfinite(log_L_means) & np.isfinite(log_H_means)
            if np.sum(valid_m) > 2:
                l_coeffs = np.polyfit(log_L_means[valid_m], log_H_means[valid_m], 2)
                x_fit_log = np.linspace(log_lims[0], log_lims[1], 100)
                axes[1, 0].plot(x_fit_log, np.poly1d(l_coeffs)(x_fit_log), color=fit_color, label=f"{display_label} Fit")

        line_color = base_color
        
        # 防止拟合时出现 nan 或无穷大
        valid_l_idx = np.isfinite(log_L_means) & ~np.isnan(log_L_means)
        valid_h_idx = np.isfinite(log_H_means) & ~np.isnan(log_H_means)
        
        if np.any(valid_l_idx):
            n_l = find_linear_pts(cur_x_vals[valid_l_idx], log_L_means[valid_l_idx], f"{mat_name} Low-E")
            if n_l > 1:
                l_fit = np.poly1d(np.polyfit(cur_x_vals[valid_l_idx][:n_l], log_L_means[valid_l_idx][:n_l], 1))
                axes[1, 1].plot(plot_x[valid_l_idx][:n_l], l_fit(cur_x_vals[valid_l_idx][:n_l]), '--', color=line_color, label=f"{display_label} (n={n_l})")
            axes[1, 1].plot(plot_x[valid_l_idx], log_L_means[valid_l_idx], 'o', color=line_color, alpha=0.3)

        if np.any(valid_h_idx):
            n_h = find_linear_pts(cur_x_vals[valid_h_idx], log_H_means[valid_h_idx], f"{mat_name} High-E")
            if n_h > 1:
                h_fit = np.poly1d(np.polyfit(cur_x_vals[valid_h_idx][:n_h], log_H_means[valid_h_idx][:n_h], 1))
                axes[1, 2].plot(plot_x[valid_h_idx][:n_h], h_fit(cur_x_vals[valid_h_idx][:n_h]), '--', color=line_color, label=f"{display_label} (n={n_h})")
            axes[1, 2].plot(plot_x[valid_h_idx], log_H_means[valid_h_idx], 'o', color=line_color, alpha=0.3)

    # Apply Adaptive Limits
    axes[0, 0].set_xlim(raw_lims); axes[0, 0].set_ylim(raw_lims)
    axes[0, 1].set_xlim(x_lims);   axes[0, 1].set_ylim(raw_lims)
    axes[0, 2].set_xlim(x_lims);   axes[0, 2].set_ylim(raw_lims)
    
    axes[1, 0].set_xlim(log_lims); axes[1, 0].set_ylim(log_lims)
    axes[1, 1].set_xlim(x_lims);   axes[1, 1].set_ylim(log_lims)
    axes[1, 2].set_xlim(x_lims);   axes[1, 2].set_ylim(log_lims)

    # Styles
    axes[0, 0].set_title("H vs L Fit")
    axes[0, 1].set_title(f"{x_label} vs Low Energy")
    axes[0, 2].set_title(f"{x_label} vs High Energy")
    axes[1, 0].set_title(r"$\ln(I_0/H)$ vs $\ln(I_0/L)$ Fit")
    axes[1, 1].set_title(f"{x_label} vs " + r"$\ln(I_0/L)$")
    axes[1, 2].set_title(f"{x_label} vs " + r"$\ln(I_0/H)$")
    
    for r in range(2): 
        for c in range(3): 
            axes[r, c].grid(True)
            leg = axes[r, c].legend(fontsize='x-small')
            if leg:
                for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
                    lh.set_alpha(1.0)

    plt.suptitle(f"{title_prefix} Analysis for {voltage} (I0={I0})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_subdir}/{voltage}_analysis.png")
    plt.close()

# --- Main Execution ---
voltages = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
filter_types = ['0.6mm', '1.2mm']
input_dir = 'results/20260429_mask_generated'
I0_val = 204.0  # 8-bit normalized

step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
thicknesses = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }

for f_type in filter_types:
    print(f"\n========================================")
    print(f"Processing Filter: {f_type}")
    print(f"========================================")
    
    base_results_dir = f'results/thickness_decoupling/H_L_fit/20260429_mask_generated/{f_type}'
    
    for voltage in voltages:
        print(f"\n>>> Processing {voltage} ...")
        
        # 1. Step Samples Analysis
        step_data = {}
        for idx, name in step_mats.items():
            # 0429 naming format for steps: {mat_name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl
            p = f'{input_dir}/pixel_values/{name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl'
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                    step_data[name] = (d['pixels_low'], d['pixels_high'])
        
        if step_data:
            perform_comprehensive_analysis(voltage, step_data, f"{base_results_dir}/steps", 
                                           f"Step Sample ({f_type})", "Thickness (mm)", thicknesses, plot_mode='all', I0=I0_val)

        # 2. Ore Samples Analysis
        # Ore naming format: ore-*-{f_type}-{voltage}-2mA-orig_ore_0_data.pkl
        ore_files = glob.glob(f'{input_dir}/pixel_values/ore-*-{f_type}-{voltage}-2mA-orig_ore_0_data.pkl')
        
        ore_L_list, ore_H_list = [], []
        ore_names = []
        
        for p in ore_files:
            # Extract ore name prefix like 'ore-01' from the filename
            basename = os.path.basename(p)
            match = re.match(r'(ore-[^-]+(?:-[^-]+)?)-', basename)
            if match:
                ore_name = match.group(1)
            else:
                ore_name = basename.split('-')[1] # fallback
                
            with open(p, 'rb') as f:
                d = pickle.load(f)
                # Ensure the lists are wrapped correctly depending on how extract saved them
                # For ore, it might just be a single list per file, but the plotting expects a list of steps
                # So we make each ore a single "step"
                if isinstance(d['pixels_low'], list):
                    ore_L_list.append(d['pixels_low'][0])
                    ore_H_list.append(d['pixels_high'][0])
                else:
                    ore_L_list.append(d['pixels_low'])
                    ore_H_list.append(d['pixels_high'])
                ore_names.append(ore_name)
        
        if ore_L_list:
            disk_data = { "Mixed_Ores": (ore_L_list, ore_H_list) }
            # Since we don't have a reliable Z_eff config for these specific ores yet, we just use a dummy index
            dummy_coords = np.arange(len(ore_L_list))
            disk_x_coords = { "Mixed_Ores": dummy_coords }
            
            perform_comprehensive_analysis(voltage, disk_data, f"{base_results_dir}/ores", 
                                           f"Ore Sample ({f_type})", "Ore Index", disk_x_coords, color_by_step=True, plot_mode='means', I0=I0_val)

print(f"\nAnalysis complete. Results saved in results/thickness_decoupling/H_L_fit/20260429_mask_generated")
