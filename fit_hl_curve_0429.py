import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import glob
import utils_II

# 自动寻找线性区间算法 (移至顶层以便通用)
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
            break
        best_n, prev_r2 = n, cur_r2
    return best_n

def perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, color_by_step=False, plot_mode='all', I0=204.0, 
                                   raw_lims_global=None, log_lims_global=None):
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
            v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
            valid = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
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

    if raw_lims_global is not None:
        raw_lims = raw_lims_global
    else:
        l_min, l_max = get_robust_range(all_L_pts)
        h_min, h_max = get_robust_range(all_H_pts)
        raw_min, raw_max = min(l_min, h_min), max(l_max, h_max)
        pad_raw = (raw_max - raw_min) * 0.15
        raw_lims = (max(0, raw_min - pad_raw), raw_max + pad_raw)

    if log_lims_global is not None:
        log_lims = log_lims_global
    else:
        log_l_min, log_l_max = get_robust_range(all_log_L_pts)
        log_h_min, log_h_max = get_robust_range(all_log_H_pts)
        log_min, log_max = min(log_l_min, log_h_min), max(log_l_max, log_h_max)
        pad_log = (log_max - log_min) * 0.15
        log_lims = (max(0, log_min - pad_log), log_max + pad_log)

    X_glob = np.concatenate(all_x_vals)
    is_categorical = X_glob.dtype.kind in 'U S O'
    
    if is_categorical:
        # 如果是类别型（如矿石 ID），映射为 0, 1, 2...
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

    # find_linear_pts 已移至顶层

    for mat_name, (L_list, H_list) in samples_dict.items():
        # 计算每个样本的均值和标准差 (过滤 1 和 255)
        step_L_means = []
        step_L_stds = []
        step_H_means = []
        step_H_stds = []
        for l, h in zip(L_list, H_list):
            v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
            v_idx = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
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
            v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
            v = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
            if np.any(v):
                L_flat_valid.append(l[v])
                H_flat_valid.append(h[v])
                
        if L_flat_valid:
            L_v = np.concatenate(L_flat_valid).astype(np.float32)
            H_v = np.concatenate(H_flat_valid).astype(np.float32)
        else:
            L_v, H_v = np.array([]), np.array([])
        
        cur_x_raw = x_coords_dict[mat_name][:len(L_list)]
        if is_categorical:
            cur_x_vals = np.array([label_to_idx[str(l)] for l in cur_x_raw])
        else:
            cur_x_vals = np.array(cur_x_raw)

        plot_x = cur_x_vals - 10 if (not is_categorical and 'Al' in mat_name and 'step' in title_prefix.lower()) else cur_x_vals
        display_label = f"{mat_name}" + (" (t-10mm)" if (not is_categorical and 'Al' in mat_name and 'step' in title_prefix.lower()) else "")

        # 先绘制 axes[0, 1] 以获取该 material 的 base_color
        eb_alpha = 0.3 if plot_mode == 'means' else 0.6
        line, = axes[0, 1].plot(plot_x, step_L_means, 'o-', markersize=5, label=display_label, linewidth=1.5)
        base_color = line.get_color()
        axes[0, 1].errorbar(plot_x, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)
        
        axes[0, 2].plot(plot_x, step_H_means, 'o-', markersize=5, label=display_label, linewidth=1.5, color=base_color)
        axes[0, 2].errorbar(plot_x, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)

        # 为最后 4 个点添加灰度值标注
        for i in range(max(0, len(plot_x)-7), len(plot_x)):
            if not np.isnan(step_L_means[i]):
                axes[0, 1].text(plot_x[i], step_L_means[i]+2, f"{step_L_means[i]:.1f}", fontsize=8, ha='center', va='bottom', color=base_color)
            if not np.isnan(step_H_means[i]):
                axes[0, 2].text(plot_x[i], step_H_means[i]+2, f"{step_H_means[i]:.1f}", fontsize=8, ha='center', va='bottom', color=base_color)

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
                    v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
                    lower_th = utils_II.get_ore_lower_threshold("ore" in title_prefix.lower(), v_max)
                    valid_i = (l >= lower_th) & (h >= lower_th) & (l < v_max) & (h < v_max)
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
            axes[r, c].grid(True, alpha=0.3)
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
script_dir = os.path.dirname(os.path.abspath(__file__))

include_0331 = True  # 是否将 0331 数据并入 0.6mm 分析
# analysis_target: "step" (仅分析阶梯), "ore" (仅分析矿石), "all" (分析全部)
# analysis_target = "step" 
analysis_target = "all"
# analysis_target = "ore" 

# 衰减系数提取方式：已更新为遍历所有阶梯厚度，分别计算并保存到独立的文件夹。

voltages_0429 = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
voltages_0331 = ['140kV', '160kV', '180kV']
filter_types = ['0.6mm', '1.2mm']
input_dir_0429 = os.path.join(script_dir, 'results/20260429_mask_generated_16bit')
input_dir_0331 = os.path.join(script_dir, 'results/20260331_16bit') # 如果0331也提取了16位，请将这里也加上 _16bit
I0_0429 = 52428.0 if '16bit' in input_dir_0429 else 204.0
I0_0331 = 52428.0 if '16bit' in input_dir_0331 else 204.0

step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
thicknesses_0429 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
thicknesses_0331 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }

# 1. 全局数据采集 (跨越所有 Filter 和 Voltage)
global_raw_data = {ft: {mat: {'v': [], 'cur_x': [], 'log_l': [], 'log_h': [], 'vl': [], 'vh': []} for mat in step_mats.values()} for ft in filter_types}
global_raw_pts = []          # 全局灰度值 (用于统一子图 1, 2, 3 的 Y 轴)
global_log_pts = []          # 全局 ln(I0/I) 值 (用于统一子图 4, 5, 6 的 Y 轴)

def get_dynamic_ylim(vals, default=(0, 1.0), pad_ratio=0.1):
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0: return default
    v_min, v_max = np.percentile(vals, 1), np.percentile(vals, 99)
    pad = (v_max - v_min) * pad_ratio if v_max > v_min else 0.1
    return (v_min - pad, v_max + pad)

print(">>> Scanning data for Y-axis limits...")
if analysis_target in ["step", "all"]:
    print(">>> Scanning step data...")
    for f_type in filter_types:
        cur_voltages = voltages_0331 + voltages_0429 if (f_type == '0.6mm' and include_0331) else voltages_0429
        
        for voltage in cur_voltages:
            is_0331 = voltage in voltages_0331
            cur_input_dir = input_dir_0331 if is_0331 else input_dir_0429
            cur_I0 = I0_0331 if is_0331 else I0_0429
            cur_thick_map = thicknesses_0331 if is_0331 else thicknesses_0429
            
            v_int = int(re.search(r'(\d+)', voltage).group(1))
            
            for idx, name in step_mats.items():
                if is_0331:
                    p = f'{cur_input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
                else:
                    p = f'{cur_input_dir}/pixel_values/{name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl'
                    
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        d = pickle.load(f)
                    l_list, h_list = d['pixels_low'], d['pixels_high']
                    cur_x = cur_thick_map[name]
                    
                    v_max = 65535 if l_list[0].dtype == np.uint16 or np.max(l_list[0]) > 255 else 255
                    m_l = np.array([np.mean(px[(px>=0)&(px<v_max)]) if np.any((px>=0)&(px<v_max)) else np.nan for px in l_list])
                    m_h = np.array([np.mean(px[(px>=0)&(px<v_max)]) if np.any((px>=0)&(px<v_max)) else np.nan for px in h_list])
                    
                    # 修正：采用先算对数再求均值，与矿石计算逻辑保持物理一致性
                    log_l = np.array([np.mean(np.log(cur_I0 / np.maximum(px[(px>=0)&(px<v_max)], 1.0))) if np.any((px>=0)&(px<v_max)) else np.nan for px in l_list])
                    log_h = np.array([np.mean(np.log(cur_I0 / np.maximum(px[(px>=0)&(px<v_max)], 1.0))) if np.any((px>=0)&(px<v_max)) else np.nan for px in h_list])
                    
                    vl, vh = np.isfinite(log_l), np.isfinite(log_h)
                    
                    global_raw_data[f_type][name]['v'].append(v_int)
                    global_raw_data[f_type][name]['cur_x'].append(cur_x)
                    global_raw_data[f_type][name]['log_l'].append(log_l)
                    global_raw_data[f_type][name]['log_h'].append(log_h)
                    global_raw_data[f_type][name]['vl'].append(vl)
                    global_raw_data[f_type][name]['vh'].append(vh)
                    
                    global_raw_pts.extend(m_l[np.isfinite(m_l)])
                    global_raw_pts.extend(m_h[np.isfinite(m_h)])
                    global_log_pts.extend(log_l[np.isfinite(log_l)])
                    global_log_pts.extend(log_h[np.isfinite(log_h)])

    global_raw_ylim = get_dynamic_ylim(global_raw_pts, default=(0, 255), pad_ratio=0.15)
    global_log_ylim = get_dynamic_ylim(global_log_pts, default=(0, 5.0), pad_ratio=0.15)
else:
    global_raw_ylim = (0, 65535) if '16bit' in input_dir_0429 else (0, 255)
    global_log_ylim = (0, 5.0)

# 3. 物质物理常数 (用于计算高能理论比值: mu ~ Z/Ar * rho)
mat_physics = {
    'Cu_step': {'Z': 29, 'Ar': 63.546, 'rho': 8.96},
    'Fe_step': {'Z': 26, 'Ar': 55.845, 'rho': 7.87},
    'Al_step': {'Z': 13, 'Ar': 26.982, 'rho': 2.70}
}

# 2. 正式绘图循环
for f_type in filter_types:
    print(f"\n========================================")
    print(f"Processing Filter: {f_type} (Unified Limits)")
    print(f"========================================")
    out_base = os.path.basename(input_dir_0429)
    base_results_dir = os.path.join(script_dir, f'results/thickness_decoupling/H_L_fit/{out_base}/{f_type}')
    
    cur_voltages = voltages_0429
    if f_type == '0.6mm' and include_0331:
        cur_voltages = voltages_0331 + voltages_0429
        
    for voltage in cur_voltages:
        is_0331 = voltage in voltages_0331
        cur_input_dir = input_dir_0331 if is_0331 else input_dir_0429
        cur_I0 = I0_0331 if is_0331 else I0_0429
        cur_thick_map = thicknesses_0331 if is_0331 else thicknesses_0429
        
        # 1. Step 分析
        if analysis_target in ["step", "all"]:
            step_data = {}
            for idx, name in step_mats.items():
                if is_0331:
                    p = f'{cur_input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
                else:
                    p = f'{cur_input_dir}/pixel_values/{name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl'
                
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        d = pickle.load(f)
                        step_data[name] = (d['pixels_low'], d['pixels_high'])
            
            if step_data:
                perform_comprehensive_analysis(voltage, step_data, f"{base_results_dir}/steps", f"Step Sample ({f_type})", "Thickness (mm)", cur_thick_map, plot_mode='all', I0=cur_I0,
                                               raw_lims_global=global_raw_ylim, log_lims_global=global_log_ylim)

        # 2. Ore 分析 (仅针对 0429 文件夹)
        if analysis_target in ["ore", "all"] and not is_0331:
            data_suffix = 'user' # 可改为 'orig'，原始数据，归一化到50000；user：自己归一化到65536*0.8(16位)，再降为8位
            
            def natural_sort_key(s):
                import re
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            raw_ore_files = glob.glob(f'{input_dir_0429}/pixel_values/ore-*-{f_type}-{voltage}-2mA-{data_suffix}_ore_0_data.pkl')
            ore_files = sorted(raw_ore_files, key=natural_sort_key)
            
            ore_L_list, ore_H_list, ore_ids = [], [], []
            for p in ore_files:
                fname = os.path.basename(p)
                # 提取 Ore- 和 下一个横杠之间的 ID
                match = re.search(r'Ore-([^-]+)-', fname, re.IGNORECASE)
                oid = match.group(1) if match else "???"
                
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                    ore_L_list.append(d['pixels_low'][0] if isinstance(d['pixels_low'], list) else d['pixels_low'])
                    ore_H_list.append(d['pixels_high'][0] if isinstance(d['pixels_high'], list) else d['pixels_high'])
                    ore_ids.append(oid)
            if ore_L_list:
                perform_comprehensive_analysis(voltage, {"Mixed_Ores": (ore_L_list, ore_H_list)}, f"{base_results_dir}/ores", f"Ore Sample ({f_type})", "Ore ID", {"Mixed_Ores": ore_ids}, color_by_step=True, plot_mode='means', I0=cur_I0,
                                               raw_lims_global=global_raw_ylim, log_lims_global=global_log_ylim)

    # === 绘制所有阶梯的汇总图并保存 ===
    if analysis_target in ["step", "all"]:
        max_steps = max(len(t) for t in thicknesses_0429.values())
        for step_idx in range(max_steps):
            cu_th = thicknesses_0429['Cu_step'][step_idx] if step_idx < len(thicknesses_0429['Cu_step']) else "N/A"
            al_th = thicknesses_0429['Al_step'][step_idx] if step_idx < len(thicknesses_0429['Al_step']) else "N/A"
            step_name = f"{cu_th}mm_CuFe_{al_th}mm_Al"
            
            print(f">>> Generating Slope Summary Plot for step: {step_name} [{f_type}] ...")
            
            # 为当前 step_idx 提取数据
            step_storage = {mat: {'v': [], 'mu_l': [], 'mu_h': []} for mat in step_mats.values()}
            step_mu_vals, step_lh_vals, step_inter_l, step_inter_h, step_diff = [], [], [], [], []
            
            for mat in list(step_mats.values()):
                v_list = global_raw_data[f_type][mat]['v']
                for i in range(len(v_list)):
                    v_int = v_list[i]
                    cur_x = global_raw_data[f_type][mat]['cur_x'][i]
                    log_l = global_raw_data[f_type][mat]['log_l'][i]
                    log_h = global_raw_data[f_type][mat]['log_h'][i]
                    vl = global_raw_data[f_type][mat]['vl'][i]
                    vh = global_raw_data[f_type][mat]['vh'][i]
                    
                    if step_idx < len(cur_x):
                        t_mm = cur_x[step_idx]
                        mu_l = log_l[step_idx] / t_mm if vl[step_idx] and t_mm > 0 else np.nan
                        mu_h = log_h[step_idx] / t_mm if vh[step_idx] and t_mm > 0 else np.nan
                    else:
                        mu_l, mu_h = np.nan, np.nan
                        
                    step_storage[mat]['v'].append(v_int)
                    step_storage[mat]['mu_l'].append(mu_l)
                    step_storage[mat]['mu_h'].append(mu_h)
                    
                    if np.isfinite(mu_l): step_mu_vals.append(mu_l)
                    if np.isfinite(mu_h): step_mu_vals.append(mu_h)
                    if np.isfinite(mu_l) and np.isfinite(mu_h): step_lh_vals.append(mu_l/mu_h)
            
            mats = list(step_mats.values())
            for i in range(len(mats)):
                for j in range(i+1, len(mats)):
                    m1, m2 = mats[i], mats[j]
                    sl1, sl2 = np.array(step_storage[m1]['mu_l']), np.array(step_storage[m2]['mu_l'])
                    sh1, sh2 = np.array(step_storage[m1]['mu_h']), np.array(step_storage[m2]['mu_h'])
                    step_inter_l.extend(sl1[np.isfinite(sl1/sl2)] / sl2[np.isfinite(sl1/sl2)])
                    step_inter_h.extend(sh1[np.isfinite(sh1/sh2)] / sh2[np.isfinite(sh1/sh2)])
                    
                    r1, r2 = sl1 / sh1, sl2 / sh2
                    diff = np.abs(r1 - r2)
                    step_diff.extend(diff[np.isfinite(diff)])
            
            cur_mu_ylim = get_dynamic_ylim(step_mu_vals)
            cur_inter_ratio_l_ylim = get_dynamic_ylim(step_inter_l, default=(0, 10.0))
            cur_inter_ratio_h_ylim = get_dynamic_ylim(step_inter_h, default=(0, 10.0))
            cur_lh_ratio_ylim = get_dynamic_ylim(step_lh_vals, default=(0, 3.0))
            cur_lh_abs_diff_ylim = get_dynamic_ylim(step_diff, default=(0, 1.0))
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(fr"Attenuation Slopes ($\mu$) Analysis - {f_type} (Step: {step_name})", fontsize=18)
            
            sort_idx = np.argsort(step_storage['Cu_step']['v'])
            for mat in mats:
                v = np.array(step_storage[mat]['v'])[sort_idx]
                ml = np.array(step_storage[mat]['mu_l'])[sort_idx]
                mh = np.array(step_storage[mat]['mu_h'])[sort_idx]
                
                axes[0, 0].plot(v, ml, 'o-', label=mat)
                axes[0, 1].plot(v, mh, 'o-', label=mat)
                axes[0, 2].plot(v, ml/mh, 's--', label=f"{mat} L/H")
            
            for i in range(len(mats)):
                for j in range(i+1, len(mats)):
                    m1, m2 = mats[i], mats[j]
                    v = np.array(step_storage[m1]['v'])[sort_idx]
                    r1 = np.array(step_storage[m1]['mu_l'])[sort_idx] / np.array(step_storage[m1]['mu_h'])[sort_idx]
                    r2 = np.array(step_storage[m2]['mu_l'])[sort_idx] / np.array(step_storage[m2]['mu_h'])[sort_idx]
                    r_l = np.array(step_storage[m1]['mu_l'])[sort_idx] / np.array(step_storage[m2]['mu_l'])[sort_idx]
                    r_h = np.array(step_storage[m1]['mu_h'])[sort_idx] / np.array(step_storage[m2]['mu_h'])[sort_idx]
                    
                    line_l, = axes[1, 0].plot(v, r_l, 'v-', label=f"{m1}/{m2} (L)")
                    line_h, = axes[1, 1].plot(v, r_h, '^-', label=f"{m1}/{m2} (H)")
                    axes[1, 2].plot(v, np.abs(r1 - r2), 'D-.', label=f"|{m1}-{m2}| (L/H)")

                    if m1 in mat_physics and m2 in mat_physics:
                        p1, p2 = mat_physics[m1], mat_physics[m2]
                        theo_l = ( (p1['Z']**4.5) / p1['Ar'] * p1['rho'] ) / ( (p2['Z']**4.5) / p2['Ar'] * p2['rho'] )
                        axes[1, 0].axhline(y=theo_l, color=line_l.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (PH)")
                        theo_h = ( p1['Z'] / p1['Ar'] * p1['rho'] ) / ( p2['Z'] / p2['Ar'] * p2['rho'] )
                        axes[1, 1].axhline(y=theo_h, color=line_h.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (C)")

            for ax in axes.flat:
                ax.set_xlabel("Voltage (kV)"); ax.grid(True); ax.legend(fontsize='x-small')
                ax.set_xlim(130, 330) # 强制 X 轴对齐
            
            axes[0, 0].set_title(r"$\mu_L$ vs Voltage"); axes[0, 0].set_ylim(cur_mu_ylim)
            axes[0, 1].set_title(r"$\mu_H$ vs Voltage"); axes[0, 1].set_ylim(cur_mu_ylim)
            axes[0, 2].set_title(r"$\mu_L / \mu_H$ Ratio"); axes[0, 2].set_ylim(cur_lh_ratio_ylim)
            axes[1, 0].set_title(r"Inter-Material Ratio ($\mu_L$)"); axes[1, 0].set_ylim(cur_inter_ratio_l_ylim)
            axes[1, 1].set_title(r"Inter-Material Ratio ($\mu_H$)"); axes[1, 1].set_ylim(cur_inter_ratio_h_ylim)
            axes[1, 2].set_title(r"Inter-Material L/H Abs Diff $|r_1 - r_2|$"); axes[1, 2].set_ylim(cur_lh_abs_diff_ylim)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            suffix = "_with_0331" if (f_type == '0.6mm' and include_0331) else ""
            
            # 保存图片到同一文件夹下，文件名体现不同厚度
            cur_save_dir = base_results_dir
            os.makedirs(cur_save_dir, exist_ok=True)
            plt.savefig(f"{cur_save_dir}/slope_summary_{f_type}{suffix}_{step_name}.png")
            plt.close()

# --- 保存数据到 JSON ---
if analysis_target in ["step", "all"]:
    print(f"\n>>> Saving all JSON results to step folders...")
    max_steps = max(len(t) for t in thicknesses_0429.values())
    for step_idx in range(max_steps):
        cu_th = thicknesses_0429['Cu_step'][step_idx] if step_idx < len(thicknesses_0429['Cu_step']) else "N/A"
        al_th = thicknesses_0429['Al_step'][step_idx] if step_idx < len(thicknesses_0429['Al_step']) else "N/A"
        step_name = f"{cu_th}mm_CuFe_{al_th}mm_Al"
        
        json_data = {}
        for ft in filter_types:
            json_data[ft] = {}
            for mat in step_mats.values():
                json_data[ft][mat] = {}
                v_list = global_raw_data[ft][mat]['v']
                s_idx = np.argsort(v_list)
                
                for i in s_idx:
                    v_str = f"{np.array(v_list)[i]}kV"
                    cur_x = global_raw_data[ft][mat]['cur_x'][i]
                    log_l = global_raw_data[ft][mat]['log_l'][i]
                    log_h = global_raw_data[ft][mat]['log_h'][i]
                    vl = global_raw_data[ft][mat]['vl'][i]
                    vh = global_raw_data[ft][mat]['vh'][i]
                    
                    if step_idx < len(cur_x):
                        t_mm = cur_x[step_idx]
                        ul = float(log_l[step_idx] / t_mm) if vl[step_idx] and t_mm > 0 else None
                        uh = float(log_h[step_idx] / t_mm) if vh[step_idx] and t_mm > 0 else None
                        if ul is not None and not np.isfinite(ul): ul = None
                        if uh is not None and not np.isfinite(uh): uh = None
                    else:
                        ul, uh = None, None
                        
                    json_data[ft][mat][v_str] = {"ul": ul, "uh": uh}
                    
        out_base = os.path.basename(input_dir_0429)
        cur_save_dir_base = os.path.join(script_dir, f'results/thickness_decoupling/H_L_fit/{out_base}')
        os.makedirs(cur_save_dir_base, exist_ok=True)
        
        output_json_path = os.path.join(cur_save_dir_base, f'attenuation_slopes_{step_name}.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
# === 新增：各块矿石的 ul/uh 随电压变化曲线 ===
if analysis_target not in ["ore", "all"]:
    print(f"All analysis complete (0331 + 0429).")
    import sys; sys.exit(0)

print("\n>>> Processing Ore ul/uh ratio over voltage...")

# 1. 确定 0429 中所有的矿石 ID
all_0429_files = glob.glob(f'{input_dir_0429}/pixel_values/*re-*-user_ore_0_data.pkl')
unique_ore_ids = set()
for p in all_0429_files:
    fname = os.path.basename(p)
    # 提取 Ore- 或 ore- 后面的 ID
    match = re.search(r'[Oo]re-([^-]+)-', fname)
    if match:
        oid = match.group(1)
        if oid not in ['0.6mm', '1.2mm']: # 排除误匹配
            unique_ore_ids.add(oid)

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

sorted_ore_ids = sorted(list(unique_ore_ids), key=natural_sort_key)
print(f"Detected 0429 Ores: {sorted_ore_ids}")

ore_ul_uh_storage = {'0.6mm': {oid: {'v': [], 'ratio_old': [], 'ratio_R': [], 'log_l': [], 'log_h': []} for oid in sorted_ore_ids},
                     '1.2mm': {oid: {'v': [], 'ratio_old': [], 'ratio_R': [], 'log_l': [], 'log_h': []} for oid in sorted_ore_ids},
                     '0401': {oid: {'v': [], 'ratio_old': [], 'ratio_R': [], 'log_l': [], 'log_h': []} for oid in sorted_ore_ids}}

# 像素灰度分布收集字典：用于在一张大图里画出每块矿石在不同电压下的灰度值分布
ore_pixels_storage = {'0.6mm': {oid: {} for oid in sorted_ore_ids},
                      '1.2mm': {oid: {} for oid in sorted_ore_ids},
                      '0401': {oid: {} for oid in sorted_ore_ids}}

# 2. 收集 0429 数据
for ft in ['0.6mm', '1.2mm']:
    for voltage in voltages_0429:
        ore_L_list, ore_H_list, ore_ids_list = [], [], []
        v_int = int(re.search(r'(\d+)', voltage).group(1))
        # 扫描该电压和滤片下的所有文件，手动过滤以实现大小写不敏感匹配
        pattern = f'{input_dir_0429}/pixel_values/*-{ft}-{voltage}-2mA-user_ore_0_data.pkl'
        raw_ore_files = glob.glob(pattern)
        for p in raw_ore_files:
            fname = os.path.basename(p)
            if re.search(fr'[Oo]re-([^-]+)-{ft}-{voltage}-2mA-user_ore_0_data.pkl', fname, re.IGNORECASE):
                match = re.search(r'[Oo]re-([^-]+)-', fname, re.IGNORECASE)
                oid = match.group(1) if match else None
                if oid and oid in ore_ul_uh_storage[ft]:
                    with open(p, 'rb') as f:
                        d = pickle.load(f)
                        l_v = d['pixels_low'][0] if isinstance(d['pixels_low'], list) else d['pixels_low']
                        h_v = d['pixels_high'][0] if isinstance(d['pixels_high'], list) else d['pixels_high']
                        
                        v_max = 65535 if l_v.dtype == np.uint16 or np.max(l_v) > 255 else 255
                        lower_th = utils_II.get_ore_lower_threshold(True, v_max)
                        mask = (l_v >= lower_th) & (l_v < v_max) & (h_v >= lower_th) & (h_v < v_max)
                        l_valid = l_v[mask]
                        h_valid = h_v[mask]
                        
                        if len(l_valid) > 0:
                            ore_L_list.append(l_valid)
                            ore_H_list.append(h_valid)
                            ore_ids_list.append(oid)
                            ore_pixels_storage[ft][oid][voltage] = (l_v, h_v)
                            
                            # 1. 计算物理衰减对数均值 (修正为先对每个像素取对数，再求均值，下限设为1.0以符合探测器物理底噪)
                            pixel_logs_l = np.log(I0_0429 / np.maximum(l_valid, 1.0))
                            pixel_logs_h = np.log(I0_0429 / np.maximum(h_valid, 1.0))
                            
                            log_l_mean = np.mean(pixel_logs_l)
                            log_h_mean = np.mean(pixel_logs_h)
                            
                            # 2. 老版本 uL/uH (增加有限值和合理范围过滤)
                            
                            # 过滤掉衰减极低的点（避免分母趋于0导致比值爆炸）
                            valid_signal = pixel_logs_h > 0.02
                            if np.any(valid_signal):
                                pixel_ratios_old = pixel_logs_l[valid_signal] / pixel_logs_h[valid_signal]
                                # 进一步过滤物理上不合理的离群值 (例如 15000)
                                physical_ratios = pixel_ratios_old[(pixel_ratios_old > 0) & (pixel_ratios_old < 20)]
                                ratio_old = np.mean(physical_ratios) if len(physical_ratios) > 0 else np.nan
                            else:
                                ratio_old = np.nan
                            
                            # 3. 新版本 R 值 (同样增加合理范围过滤)
                            r_pixel_array = utils_II.compute_R(l_valid, h_valid, I0_low=I0_0429, I0_high=I0_0429, input='images')
                            r_physical = r_pixel_array[np.isfinite(r_pixel_array) & (r_pixel_array > 0) & (r_pixel_array < 20)]
                            ratio_R = np.mean(r_physical) if len(r_physical) > 0 else np.nan
                            
                            # 调用画图并保存累积分布图的函数
                            out_base = os.path.basename(input_dir_0429)
                            cdf_save_path = os.path.join(script_dir, f"results/thickness_decoupling/H_L_fit/{out_base}/histograms/{ft}/{voltage}/ore_{oid}_ul_cdf.png")
                            utils_II.plot_ul_cdf(
                                pixels_low=l_valid,
                                save_path=cdf_save_path,
                                I0=I0_0429,
                                title=f"Ore {oid} ({ft}) ul Cumulative Distribution (CDF) at {voltage}"
                            )
                            
                            ore_ul_uh_storage[ft][oid]['v'].append(v_int)
                            ore_ul_uh_storage[ft][oid]['ratio_old'].append(ratio_old)
                            ore_ul_uh_storage[ft][oid]['ratio_R'].append(ratio_R)
                            ore_ul_uh_storage[ft][oid]['log_l'].append(log_l_mean)
                            ore_ul_uh_storage[ft][oid]['log_h'].append(log_h_mean)
        
        if ore_L_list:
            ore_data = { "Mixed_Ores": (ore_L_list, ore_H_list) }
            ore_x_coords = { "Mixed_Ores": np.array(ore_ids_list) }
            out_base = os.path.basename(input_dir_0429)
            base_results_dir = os.path.join(script_dir, f'results/thickness_decoupling/H_L_fit/{out_base}/{ft}/ores')
            perform_comprehensive_analysis(voltage, ore_data, base_results_dir, 
                                           f"Ore Sample ({ft})", "Ore ID", ore_x_coords, 
                                           color_by_step=True, plot_mode='means', I0=I0_0429,
                                           raw_lims_global=global_raw_ylim, log_lims_global=global_log_ylim)

    # 绘制该 ft 下每块矿石的灰度分布大图
    out_base = os.path.basename(input_dir_0429)
    for oid in sorted_ore_ids:
        if ore_pixels_storage[ft][oid]:
            save_path = os.path.join(script_dir, f"results/thickness_decoupling/H_L_fit/{out_base}/histograms/{ft}/ore_{oid}_grayscale_distribution.png")
            utils_II.plot_ore_grayscale_distribution(oid, ft, ore_pixels_storage[ft][oid], save_path)

# 3. 收集 0401 的矿石数据
input_dir_0401 = os.path.join(script_dir, 'results/20260401_16bit')
voltages_0401 = ['140kV', '150kV', '160kV', '170kV', '180kV']
for voltage in voltages_0401:
    ore_L_list, ore_H_list, ore_ids_list = [], [], []
    v_int = int(re.search(r'(\d+)', voltage).group(1))
    pattern_0401 = f'{input_dir_0401}/pixel_values/*{voltage}*ore_*_data.pkl'
    files_0401 = glob.glob(pattern_0401)
    for p in files_0401:
        fname = os.path.basename(p)
        match = re.search(r'(?:(.*?)_)?ore_(\d+)_data\.pkl', fname, re.IGNORECASE)
        if match:
            prefix, idx_str = match.groups()
            idx = int(idx_str)
            true_id_int = -1
            if prefix and "1_20" in prefix:
                true_id_int = idx + 1    # 1_20 的 0号是 1号矿石
            elif prefix and "21_38" in prefix:
                true_id_int = idx + 21   # 21_38 的 0号是 21号矿石
            
            if true_id_int != -1:
                # 寻找匹配的 0429 ID (如 "01" 匹配 1)
                for oid in sorted_ore_ids:
                    match_found = False
                    if oid.isdigit() and int(oid) == true_id_int:
                        match_found = True
                    elif oid.lower() == str(true_id_int).lower():
                        match_found = True
                    
                    if match_found:
                        with open(p, 'rb') as f:
                            d = pickle.load(f)
                            l_v = d['pixels_low'][0] if isinstance(d['pixels_low'], list) else d['pixels_low']
                            h_v = d['pixels_high'][0] if isinstance(d['pixels_high'], list) else d['pixels_high']
                            v_max = 65535 if l_v.dtype == np.uint16 or np.max(l_v) > 255 else 255
                            lower_th = utils_II.get_ore_lower_threshold(True, v_max)
                            mask = (l_v >= lower_th) & (l_v < v_max) & (h_v >= lower_th) & (h_v < v_max)
                            l_valid = l_v[mask]
                            h_valid = h_v[mask]
                            
                            if len(l_valid) > 0:
                                ore_L_list.append(l_valid)
                                ore_H_list.append(h_valid)
                                ore_ids_list.append(oid)
                                ore_pixels_storage['0401'][oid][voltage] = (l_v, h_v)
                                
                                # 1. 计算物理衰减对数均值 (修正为先对每个像素取对数，再求均值，下限设为1.0)
                                pixel_logs_l = np.log(I0_0429 / np.maximum(l_valid, 1.0))
                                pixel_logs_h = np.log(I0_0429 / np.maximum(h_valid, 1.0))
                                
                                log_l_mean = np.mean(pixel_logs_l)
                                log_h_mean = np.mean(pixel_logs_h)
                                
                                # 调用画图并保存累积分布图的函数
                                out_base = os.path.basename(input_dir_0429)
                                cdf_save_path = os.path.join(script_dir, f"results/thickness_decoupling/H_L_fit/{out_base}/histograms/0401/{voltage}/ore_{oid}_ul_cdf.png")
                                utils_II.plot_ul_cdf(
                                    pixels_low=l_valid,
                                    save_path=cdf_save_path,
                                    I0=I0_0429,
                                    title=f"Ore {oid} (0401) ul Cumulative Distribution (CDF) at {voltage}"
                                )
                                
                                # 2. 老版本 uL/uH (增加有限值和合理范围过滤)

                                valid_signal = pixel_logs_h > 0.02
                                if np.any(valid_signal):
                                    pixel_ratios_old = pixel_logs_l[valid_signal] / pixel_logs_h[valid_signal]
                                    physical_ratios = pixel_ratios_old[(pixel_ratios_old > 0) & (pixel_ratios_old < 20)]
                                    ratio_old = np.mean(physical_ratios) if len(physical_ratios) > 0 else np.nan
                                else:
                                    ratio_old = np.nan
                                
                                # 3. 新版本 R 值 (先像素 R，再平均)
                                r_pixel_array = utils_II.compute_R(l_valid, h_valid, I0_low=I0_0429, I0_high=I0_0429, input='images')
                                r_physical = r_pixel_array[np.isfinite(r_pixel_array) & (r_pixel_array > 0) & (r_pixel_array < 20)]
                                ratio_R = np.mean(r_physical) if len(r_physical) > 0 else np.nan
                                

                                
                                ore_ul_uh_storage['0401'][oid]['v'].append(v_int)
                                ore_ul_uh_storage['0401'][oid]['ratio_old'].append(ratio_old)
                                ore_ul_uh_storage['0401'][oid]['ratio_R'].append(ratio_R)
                                ore_ul_uh_storage['0401'][oid]['log_l'].append(log_l_mean)
                                ore_ul_uh_storage['0401'][oid]['log_h'].append(log_h_mean)

    if ore_L_list:
        ore_data = { "Mixed_Ores": (ore_L_list, ore_H_list) }
        ore_x_coords = { "Mixed_Ores": np.array(ore_ids_list) }
        out_base = os.path.basename(input_dir_0429)
        base_results_dir = os.path.join(script_dir, f'results/thickness_decoupling/H_L_fit/{out_base}/0401/ores')
        perform_comprehensive_analysis(voltage, ore_data, base_results_dir, 
                                       "Ore Sample (0401)", "Ore ID", ore_x_coords, 
                                       color_by_step=True, plot_mode='means', I0=I0_0331,
                                       raw_lims_global=global_raw_ylim, log_lims_global=global_log_ylim)

    # 绘制该 0401 下每块矿石的灰度分布大图
    out_base = os.path.basename(input_dir_0429)
    for oid in sorted_ore_ids:
        if ore_pixels_storage['0401'][oid]:
            save_path = os.path.join(script_dir, f"results/thickness_decoupling/H_L_fit/{out_base}/histograms/0401/ore_{oid}_grayscale_distribution.png")
            utils_II.plot_ore_grayscale_distribution(oid, '0401', ore_pixels_storage['0401'][oid], save_path)

# 4. 计算统一的 Y 轴范围
all_ratios_old, all_ratios_R, all_logs = [], [], []
for grp in ore_ul_uh_storage.values():
    for oid_data in grp.values():
        all_ratios_old.extend(oid_data['ratio_old'])
        all_ratios_R.extend(oid_data['ratio_R'])
        all_logs.extend(oid_data['log_l'])
        all_logs.extend(oid_data['log_h'])
        
def get_ylim(vals, default=(0.5, 2.0)):
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)] # 过滤掉 inf 和 nan
    if len(vals) == 0: return default
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    pad = (v_max - v_min) * 0.15 if v_max > v_min else 0.1
    return (max(0, v_min - pad), v_max + pad)

unified_ratio_ylim_old = get_ylim(all_ratios_old)
unified_ratio_ylim_R = get_ylim(all_ratios_R)
unified_log_ylim = get_ylim(all_logs, default=(0, 3.0))

# 5. 绘图
for oid in sorted_ore_ids:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    has_16bit = False
    for grp in ['0.6mm', '1.2mm', '0401']:
        if oid in ore_pixels_storage[grp]:
            for voltage, (l, h) in ore_pixels_storage[grp][oid].items():
                if l.dtype == np.uint16 or np.max(l) > 255:
                    has_16bit = True
                    break
            if has_16bit: break
    ex_val = 2560 if has_16bit else 10
    fig.suptitle(fr"Ore {oid} Comprehensive Attenuation Analysis vs Voltage (Excluding Grayscale < {ex_val})", fontsize=18)
    
    # 子图配置: (Subplot, Title, Y-Label, Data-Key, Y-Lim)
    subplots_cfg = [
        (axes[0, 0], "Ratio $u_L/u_H$ (Old)", r"$u_L/u_H$", 'ratio_old', unified_ratio_ylim_old),
        (axes[0, 1], "R-value (via compute_R)", r"$R$", 'ratio_R', unified_ratio_ylim_R),
        (axes[1, 0], r"Low-Energy $\ln(I_0/L)$", r"$\ln(I_0/L)$", 'log_l', unified_log_ylim),
        (axes[1, 1], r"High-Energy $\ln(I_0/H)$", r"$\ln(I_0/H)$", 'log_h', unified_log_ylim)
    ]
    
    colors = {'0.6mm': 'blue', '1.2mm': 'green', '0401': 'red'}
    markers = {'0.6mm': 'o-', '1.2mm': 'o-', '0401': 's--'}
    
    for ax, title, ylabel, key, ylim in subplots_cfg:
        ax.set_title(title)
        ax.set_xlabel("Voltage (kV)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # 存储用于计算分段均值的数据
        seg_data = {'0401': {'v': [], 'y': []}, '0429_06': {'v': [], 'y': []}}
        
        for grp in ['0.6mm', '1.2mm', '0401']:
            v = np.array(ore_ul_uh_storage[grp][oid]['v'])
            data = np.array(ore_ul_uh_storage[grp][oid][key])
            if len(v) > 0:
                si = np.argsort(v)
                label = f"0429 ({grp})" if grp != '0401' else "0401 (Baseline)"
                ax.plot(v[si], data[si], markers[grp], label=label, color=colors[grp])
                
                # 收集用于均值计算的数据
                if grp == '0401':
                    seg_data['0401']['v'].extend(v); seg_data['0401']['y'].extend(data)
                elif grp == '0.6mm':
                    seg_data['0429_06']['v'].extend(v); seg_data['0429_06']['y'].extend(data)
        
        # 在比例图中添加分段均值标注 (Row 0)
        if key in ['ratio_old', 'ratio_R']:
            # 1. 低压段 140-180 (0401)
            v_a, y_a = np.array(seg_data['0401']['v']), np.array(seg_data['0401']['y'])
            mask_a = (v_a >= 140) & (v_a <= 180) & np.isfinite(y_a)
            if np.any(mask_a):
                m_a = np.mean(y_a[mask_a])
                ax.axhline(m_a, color='red', linestyle='--', alpha=0.4, linewidth=1)
                ax.text(325, m_a, f"Mean(140-180): {m_a:.3f}", color='red', fontsize=8, va='center')
            
            # 2. 高压段 200-320 (0429 0.6mm)
            v_b, y_b = np.array(seg_data['0429_06']['v']), np.array(seg_data['0429_06']['y'])
            mask_b = (v_b >= 200) & (v_b <= 320) & np.isfinite(y_b)
            if np.any(mask_b):
                m_b = np.mean(y_b[mask_b])
                ax.axhline(m_b, color='blue', linestyle='--', alpha=0.4, linewidth=1)
                ax.text(325, m_b, f"Mean(200-320): {m_b:.3f}", color='blue', fontsize=8, va='center')
        
        ax.legend(fontsize='x-small')
        ax.set_ylim(ylim)

    out_base = os.path.basename(input_dir_0429)
    out_dir = os.path.join(script_dir, f"results/thickness_decoupling/H_L_fit/{out_base}")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{out_dir}/ore_{oid}_comprehensive_analysis.png")
    plt.close()

print(f"All analysis complete (0331 + 0429).")
