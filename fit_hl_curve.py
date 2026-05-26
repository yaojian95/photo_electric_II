import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from utils_II import calculate_effective_z

materials = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
voltages = ['140kV', '160kV', '180kV']

output_dir = 'results/thickness_decoupling'
os.makedirs(output_dir, exist_ok=True)

def perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, limits=None, color_by_step=False, plot_mode='all', I0=204.0):
    """
    通用 2x3 综合分析绘图函数
    limits: { 'raw_x': (min, max), 'raw_y': (min, max), 'log_x': (min, max), 'log_y': (min, max) }
    color_by_step (bool): 是否按照步进/圆盘的索引使用不同的颜色绘制散点图。
    plot_mode (str): 'all' 为绘制所有像素点, 'means' 为仅绘制均值及误差棒。
    """
    os.makedirs(output_subdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # I0 passed as argument

    # 1. 计算自适应坐标限制 (扫描所有数据)
    all_L_pts, all_H_pts = [], []
    all_log_L_pts, all_log_H_pts = [], []
    all_x_vals = []
    
    for mat_name, (L_list, H_list) in samples_dict.items():
        cur_x = x_coords_dict[mat_name][:len(L_list)]
        if 'Al' in mat_name and 'step' in title_prefix.lower(): cur_x = cur_x - 10
        all_x_vals.append(cur_x)
        
        for l, h in zip(L_list, H_list):
            v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
            valid = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
            if np.any(valid):
                lv, hv = l[valid].astype(np.float32), h[valid].astype(np.float32)
                if plot_mode == 'means':
                    # 只采集各样本的均值点，以便坐标轴紧凑
                    m_l, m_h = np.mean(lv), np.mean(hv)
                    all_L_pts.append(m_l); all_H_pts.append(m_h)
                    all_log_L_pts.append(np.log(I0 / m_l)); all_log_H_pts.append(np.log(I0 / m_h))
                else:
                    # 采集所有像素
                    all_L_pts.append(lv); all_H_pts.append(hv)
                    all_log_L_pts.append(np.log(I0 / lv)); all_log_H_pts.append(np.log(I0 / hv))

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
    raw_lims = (max(0, raw_min - pad_raw), raw_max + pad_raw)

    log_l_min, log_l_max = get_robust_range(all_log_L_pts)
    log_h_min, log_h_max = get_robust_range(all_log_H_pts)
    log_min, log_max = min(log_l_min, log_h_min), max(log_l_max, log_h_max)
    pad_log = (log_max - log_min) * 0.15
    log_lims = (max(0, log_min - pad_log), log_max + pad_log)

    X_glob = np.concatenate(all_x_vals)
    is_categorical = X_glob.dtype.kind in 'U S O'
    
    if is_categorical:
        unique_labels = []
        for val in X_glob:
            if val not in unique_labels: unique_labels.append(val)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        x_min, x_max = 0, len(unique_labels) - 1
        pad_x = 0.5
        x_lims = (x_min - pad_x, x_max + pad_x)
    else:
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
                print(f"  [Linear Range Alert] {label} stopped at n={n-1} ({prev_r2:.4f} -> {cur_r2:.4f})")
                break
            best_n, prev_r2 = n, cur_r2
        return best_n

    for mat_name, (L_list, H_list) in samples_dict.items():
        # 计算每个样本的均值和标准差 (过滤 1 和 255)
        step_L_means = []
        step_L_stds = []
        step_H_means = []
        step_H_stds = []
        log_L_means = []
        log_H_means = []
        for l, h in zip(L_list, H_list):
            v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
            v_idx = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
            if np.any(v_idx):
                lv, hv = l[v_idx], h[v_idx]
                step_L_means.append(np.mean(lv)); step_L_stds.append(np.std(lv))
                step_H_means.append(np.mean(hv)); step_H_stds.append(np.std(hv))
                
                log_l_vals = np.log(I0 / np.maximum(lv, 1.0))
                log_h_vals = np.log(I0 / np.maximum(hv, 1.0))
                log_L_means.append(np.mean(log_l_vals))
                log_H_means.append(np.mean(log_h_vals))
            else:
                step_L_means.append(np.nan); step_L_stds.append(np.nan)
                step_H_means.append(np.nan); step_H_stds.append(np.nan)
                log_L_means.append(np.nan); log_H_means.append(np.nan)

        step_L_means, step_L_stds = np.array(step_L_means), np.array(step_L_stds)
        step_H_means, step_H_stds = np.array(step_H_means), np.array(step_H_stds)
        log_L_means = np.array(log_L_means)
        log_H_means = np.array(log_H_means)

        L_all = np.concatenate(L_list).astype(np.float32)
        H_all = np.concatenate(H_list).astype(np.float32)
        v_max = 65535 if L_all.dtype == np.uint16 or np.max(L_all) > 255 else 255
        lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
        valid = (L_all >= lower_th) & (H_all >= lower_th) & (L_all < v_max) & (H_all < v_max)
        L_v, H_v = L_all[valid], H_all[valid]
        
        cur_x_raw = x_coords_dict[mat_name][:len(L_list)]
        if is_categorical:
            cur_x_vals = np.array([label_to_idx[str(l)] for l in cur_x_raw])
        else:
            cur_x_vals = np.array(cur_x_raw)

        plot_x = cur_x_vals - 10 if (not is_categorical and 'Al' in mat_name and 'step' in title_prefix.lower()) else cur_x_vals
        display_label = f"{mat_name}" + (" (t-10mm)" if (not is_categorical and 'Al' in mat_name and 'step' in title_prefix.lower()) else "")

        # 先绘制 axes[0, 1] 以获取该 material 的 base_color
        eb_alpha = 0.3 if plot_mode == 'means' else 0.6
        # 绘制带实心的折线和点 (不透明)
        line, = axes[0, 1].plot(plot_x, step_L_means, 'o-', markersize=5, label=display_label, linewidth=1.5)
        base_color = line.get_color()
        # 绘制淡色的误差棒
        axes[0, 1].errorbar(plot_x, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)
        
        axes[0, 2].plot(plot_x, step_H_means, 'o-', markersize=5, label=display_label, linewidth=1.5, color=base_color)
        axes[0, 2].errorbar(plot_x, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)

        # Row 1: Raw Intensity
        if len(L_v) > 0:
            if color_by_step:
                cmap = plt.get_cmap('tab10' if len(L_list) <= 10 else 'tab20')
                for i, (l, h) in enumerate(zip(L_list, H_list)):
                    v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
                    lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
                    valid_i = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
                    if np.any(valid_i):
                        if plot_mode == 'all':
                            axes[0, 0].scatter(l[valid_i], h[valid_i], color=cmap(i), alpha=0.05, s=0.5, label=f"ID:{cur_x_raw[i]}")
                        else: # 'means'
                            m_l, m_h = np.mean(l[valid_i]), np.mean(h[valid_i])
                            s_l, s_h = np.std(l[valid_i]), np.std(h[valid_i])
                            axes[0, 0].errorbar(m_l, m_h, xerr=s_l, yerr=s_h, fmt='none', color=cmap(i), capsize=2, alpha=0.3)
                            axes[0, 0].scatter(m_l, m_h, color=cmap(i), s=40, label=f"ID:{cur_x_raw[i]}", edgecolors='none')
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
            if np.any(valid_m):
                coeffs = np.polyfit(step_L_means[valid_m], step_H_means[valid_m], 2)
                x_fit = np.linspace(raw_lims[0], raw_lims[1], 100)
                axes[0, 0].plot(x_fit, np.poly1d(coeffs)(x_fit), color=fit_color, label=f"{display_label} Fit")

        # Row 2: Log Transform
        log_L_v, log_H_v = np.log(I0 / np.maximum(L_v, 1.0)), np.log(I0 / np.maximum(H_v, 1.0))
        # log_L_means 和 log_H_means 已经在上方采用先取对数再求均值的方式计算完毕

        if len(log_L_v) > 0:
            if color_by_step:
                cmap = plt.get_cmap('tab10' if len(L_list) <= 10 else 'tab20')
                for i, (l, h) in enumerate(zip(L_list, H_list)):
                    v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
                    lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
                    valid_i = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
                    if np.any(valid_i):
                        ll = np.log(I0 / np.maximum(l[valid_i], 1.0))
                        hh = np.log(I0 / np.maximum(h[valid_i], 1.0))
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
            valid_m = ~np.isnan(log_L_means) & ~np.isnan(log_H_means)
            if np.any(valid_m):
                l_coeffs = np.polyfit(log_L_means[valid_m], log_H_means[valid_m], 2)
                x_fit_log = np.linspace(log_lims[0], log_lims[1], 100)
                axes[1, 0].plot(x_fit_log, np.poly1d(l_coeffs)(x_fit_log), color=fit_color, label=f"{display_label} Fit")

        line_color = base_color
        n_l = find_linear_pts(cur_x_vals, log_L_means, f"{mat_name} Low-E")
        l_fit = np.poly1d(np.polyfit(cur_x_vals[:n_l], log_L_means[:n_l], 1))
        axes[1, 1].plot(plot_x[:n_l], l_fit(cur_x_vals[:n_l]), '--', color=line_color, label=f"{display_label} (n={n_l})")
        axes[1, 1].plot(plot_x, log_L_means, 'o', color=line_color, alpha=0.3)

        n_h = find_linear_pts(cur_x_vals, log_H_means, f"{mat_name} High-E")
        h_fit = np.poly1d(np.polyfit(cur_x_vals[:n_h], log_H_means[:n_h], 1))
        axes[1, 2].plot(plot_x[:n_h], h_fit(cur_x_vals[:n_h]), '--', color=line_color, label=f"{display_label} (n={n_h})")
        axes[1, 2].plot(plot_x, log_H_means, 'o', color=line_color, alpha=0.3)

    # Apply Adaptive Limits
    # Row 0: Intensity
    axes[0, 0].set_xlim(raw_lims); axes[0, 0].set_ylim(raw_lims)
    axes[0, 1].set_xlim(x_lims);   axes[0, 1].set_ylim(raw_lims)
    axes[0, 2].set_xlim(x_lims);   axes[0, 2].set_ylim(raw_lims)
    
    # Row 1: Log transform
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
            if is_categorical and (r, c) in [(0, 1), (0, 2), (1, 1), (1, 2)]:
                axes[r, c].set_xticks(range(len(unique_labels)))
                axes[r, c].set_xticklabels(unique_labels, rotation=45, fontsize=8)
            if leg:
                for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
                    lh.set_alpha(1.0)

    full_title = f"{title_prefix} Analysis for {voltage} (I0={I0})"
    if "ore" in title_prefix.lower():
        has_16bit = False
        for mat_name, (L_list, H_list) in samples_dict.items():
            for l in L_list:
                if l.dtype == np.uint16 or np.max(l) > 255:
                    has_16bit = True
                    break
            if has_16bit: break
        ex_val = 2560 if has_16bit else 10
        full_title += f" (Excluding Grayscale < {ex_val})"
    plt.suptitle(full_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_subdir}/{voltage}_analysis.png")
    plt.close()

# --- Main Execution ---
voltages = ['140kV', '150kV', '160kV', '170kV', '180kV']
input_dir = 'results/20260401_16bit/'
base_results_dir = f'results/thickness_decoupling/H_L_fit/{input_dir.strip("/").split("/")[-1]}'
I0_val = 52428.0 if '16bit' in input_dir else 204.0
step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
thicknesses = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step_block': np.arange(12, 32, 2) }

# Load disk grades config for Z_eff proxy
config_path = r'E:\multi_source_info\data_dir\disk_grades.json'
with open(config_path, 'r', encoding='utf-8') as f:
    full_config = json.load(f)
date_match = re.search(r'(\d{8})', input_dir)
data_date = date_match.group(1) if date_match else "20260331"
grades_config = full_config.get(data_date, {})

# 定义不同组的显示限制
step_limits = {'raw_x': (0, 120),  'raw_y': (0, 130),  'log_x': (0.5, 5.0), 'log_y': (0.5, 5.0)}
disk_limits = {'raw_x': (140, 210), 'raw_y': (140, 210), 'log_x': (0, 0.4),   'log_y': (0, 0.4)}

for voltage in voltages:
    print(f"\n>>> Processing {voltage} ...")
    
    # 1. Step Samples Analysis
    step_data = {}
    for idx, name in step_mats.items():
        p = f'{input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
        if os.path.exists(p):
            with open(p, 'rb') as f:
                d = pickle.load(f)
                step_data[name] = (d['pixels_low'], d['pixels_high'])
    
    if step_data:
        perform_comprehensive_analysis(voltage, step_data, f"{base_results_dir}/steps", 
                                       "Step Sample", "Thickness (mm)", thicknesses, plot_mode='all', I0=I0_val)

    # 2. Disk Samples Analysis
    import glob
    disk_files = glob.glob(f'{input_dir}/pixel_values/*{voltage}*disk*_data.pkl')
    
    disk_L_list, disk_H_list = [], []
    active_z_effs = []
    for p in disk_files:
        fname = os.path.basename(p)
        match = re.search(r'disk_(\d+)_data\.pkl', fname, re.IGNORECASE)
        d_id = match.group(1) if match else "???"
        
        with open(p, 'rb') as f:
            d = pickle.load(f)
            disk_L_list.append(d['pixels_low'][0] if isinstance(d['pixels_low'], list) else d['pixels_low'])
            disk_H_list.append(d['pixels_high'][0] if isinstance(d['pixels_high'], list) else d['pixels_high'])
            
            # Calculate Z_eff as grade proxy
            if str(d_id) in grades_config:
                cu, fe, s = grades_config[str(d_id)]
                _, z_eff = calculate_effective_z(cu, fe, s)
                active_z_effs.append(z_eff)
            else:
                active_z_effs.append(float(d_id) if d_id.isdigit() else 0.0)
    
    if disk_L_list:
        disk_data = { "Mixed_Disks": (disk_L_list, disk_H_list) }
        disk_x_coords = { "Mixed_Disks": np.array(active_z_effs) }
        perform_comprehensive_analysis(voltage, disk_data, f"{base_results_dir}/disks", 
                                       "Disk Sample (Z_eff Proxy)", "Equivalent Atomic Number (Z_eff)", disk_x_coords, color_by_step=True, plot_mode='means', I0=I0_val)

    # 3. Ore Samples Analysis
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    raw_ore_files = glob.glob(f'{input_dir}/pixel_values/*{voltage}*ore*_data.pkl')
    ore_files = sorted(raw_ore_files, key=natural_sort_key)
    
    ore_L_list, ore_H_list, ore_ids = [], [], []
    for p in ore_files:
        fname = os.path.basename(p)
        match = re.search(r'(?:(.*?)_)?ore_(\d+)_data\.pkl', fname, re.IGNORECASE)
        if match:
            prefix, oid = match.groups()
            oid_int = int(oid)
            if prefix and "1_20" in prefix:
                label = str(oid_int)
            elif prefix and "21_38" in prefix:
                label = str(oid_int + 20)
            else:
                label = f"{prefix}_{oid}" if prefix and prefix not in [voltage, "ores", f"{voltage}_6"] else oid
        else:
            label = "???"
        
        with open(p, 'rb') as f:
            d = pickle.load(f)
            ore_L_list.append(d['pixels_low'][0] if isinstance(d['pixels_low'], list) else d['pixels_low'])
            ore_H_list.append(d['pixels_high'][0] if isinstance(d['pixels_high'], list) else d['pixels_high'])
            ore_ids.append(label)
            
    if ore_L_list:
        ore_data = { "Mixed_Ores": (ore_L_list, ore_H_list) }
        ore_x_coords = { "Mixed_Ores": np.array(ore_ids) }
        perform_comprehensive_analysis(voltage, ore_data, f"{base_results_dir}/ores", 
                                       "Ore Sample", "Ore ID", ore_x_coords, color_by_step=True, plot_mode='means', I0=I0_val)

print(f"\nAnalysis complete. Results saved in {base_results_dir}")
