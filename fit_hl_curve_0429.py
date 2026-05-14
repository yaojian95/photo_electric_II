import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import glob

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

    if raw_lims_global is not None:
        raw_lims = raw_lims_global
    else:
        l_min, l_max = get_robust_range(all_L_pts)
        h_min, h_max = get_robust_range(all_H_pts)
        raw_min, raw_max = min(l_min, h_min), max(l_max, h_max)
        pad_raw = (raw_max - raw_min) * 0.15
        raw_lims = (max(0, raw_min - pad_raw), min(255, raw_max + pad_raw))

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
            axes[r, c].grid(True, alpha=0.3)
            leg = axes[r, c].legend(fontsize='x-small')
            if is_categorical and (r, c) in [(0, 1), (0, 2), (1, 1), (1, 2)]:
                axes[r, c].set_xticks(range(len(unique_labels)))
                axes[r, c].set_xticklabels(unique_labels, rotation=45, fontsize=8)
            if leg:
                for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
                    lh.set_alpha(1.0)

    plt.suptitle(f"{title_prefix} Analysis for {voltage} (I0={I0})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_subdir}/{voltage}_analysis.png")
    plt.close()

# --- Main Execution ---
include_0331 = True  # 是否将 0331 数据并入 0.6mm 分析

voltages_0429 = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
voltages_0331 = ['140kV', '160kV', '180kV']
filter_types = ['0.6mm', '1.2mm']
input_dir_0429 = 'results/20260429_mask_generated'
input_dir_0331 = 'results/20260331'
I0_0429 = 204.0
I0_0331 = 204.0

step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
thicknesses_0429 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
thicknesses_0331 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }

# 1. 全局数据采集 (跨越所有 Filter 和 Voltage)
global_slope_storage = {ft: {mat: {'v': [], 'mu_l': [], 'mu_h': []} for mat in step_mats.values()} for ft in filter_types}
global_mu_vals = []
global_inter_ratio_l_vals = [] # 物质间比例 (Subplot 4)
global_inter_ratio_h_vals = [] # 物质间比例 (Subplot 5)
global_lh_ratio_vals = []    # 物质自身 L/H 比例 (Subplot 3)
global_lh_abs_diff_vals = [] # 物质间 L/H 比例差值的绝对值 (Subplot 6)
global_raw_pts = []          # 全局灰度值 (用于统一子图 1, 2, 3 的 Y 轴)
global_log_pts = []          # 全局 ln(I0/I) 值 (用于统一子图 4, 5, 6 的 Y 轴)

print(">>> Scanning all data to unify Y-axis limits...")
for f_type in filter_types:
    # 组合电压列表
    cur_voltages = voltages_0429
    if f_type == '0.6mm' and include_0331:
        cur_voltages = voltages_0331 + voltages_0429
    
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
                
                v_max = 255 # 0331 和 0429 均为 8位
                m_l = np.array([np.mean(px[(px>=0)&(px<v_max)]) if np.any((px>=0)&(px<v_max)) else np.nan for px in l_list])
                m_h = np.array([np.mean(px[(px>=0)&(px<v_max)]) if np.any((px>=0)&(px<v_max)) else np.nan for px in h_list])
                log_l, log_h = np.log(cur_I0 / np.maximum(m_l, 1e-6)), np.log(cur_I0 / np.maximum(m_h, 1e-6))
                
                vl, vh = np.isfinite(log_l), np.isfinite(log_h)
                mu_l = np.polyfit(cur_x[vl][:find_linear_pts(cur_x[vl], log_l[vl])], log_l[vl][:find_linear_pts(cur_x[vl], log_l[vl])], 1)[0] if np.sum(vl)>=3 else np.nan
                mu_h = np.polyfit(cur_x[vh][:find_linear_pts(cur_x[vh], log_h[vh])], log_h[vh][:find_linear_pts(cur_x[vh], log_h[vh])], 1)[0] if np.sum(vh)>=3 else np.nan
                
                global_slope_storage[f_type][name]['v'].append(v_int)
                global_slope_storage[f_type][name]['mu_l'].append(mu_l)
                global_slope_storage[f_type][name]['mu_h'].append(mu_h)
                if np.isfinite(mu_l): global_mu_vals.append(mu_l)
                if np.isfinite(mu_h): global_mu_vals.append(mu_h)
                if np.isfinite(mu_l) and np.isfinite(mu_h):
                    global_lh_ratio_vals.append(mu_l/mu_h)
                
                # 收集全局灰度值和衰减值
                global_raw_pts.extend(m_l[np.isfinite(m_l)])
                global_raw_pts.extend(m_h[np.isfinite(m_h)])
                global_log_pts.extend(log_l[np.isfinite(log_l)])
                global_log_pts.extend(log_h[np.isfinite(log_h)])

# 计算物质间比例及绝对差值的全局极值
mats = list(step_mats.values())
global_lh_abs_diff_vals = []
for ft in filter_types:
    for i in range(len(mats)):
        for j in range(i+1, len(mats)):
            m1, m2 = mats[i], mats[j]
            sl1, sl2 = np.array(global_slope_storage[ft][m1]['mu_l']), np.array(global_slope_storage[ft][m2]['mu_l'])
            sh1, sh2 = np.array(global_slope_storage[ft][m1]['mu_h']), np.array(global_slope_storage[ft][m2]['mu_h'])
            global_inter_ratio_l_vals.extend(sl1[np.isfinite(sl1/sl2)] / sl2[np.isfinite(sl1/sl2)])
            global_inter_ratio_h_vals.extend(sh1[np.isfinite(sh1/sh2)] / sh2[np.isfinite(sh1/sh2)])
            
            # 计算 L/H 比值的绝对差值
            r1, r2 = sl1 / sh1, sl2 / sh2
            diff = np.abs(r1 - r2)
            global_lh_abs_diff_vals.extend(diff[np.isfinite(diff)])

# 确定全局统一的 Y 轴范围
# get_dynamic_ylim 会在内部处理数组转换和无效值过滤

def get_dynamic_ylim(vals, default=(0, 1.0), pad_ratio=0.1):
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0: return default
    v_min, v_max = np.percentile(vals, 1), np.percentile(vals, 99)
    pad = (v_max - v_min) * pad_ratio if v_max > v_min else 0.1
    return (v_min - pad, v_max + pad)

global_mu_ylim = get_dynamic_ylim(global_mu_vals)
global_inter_ratio_l_ylim = get_dynamic_ylim(global_inter_ratio_l_vals, default=(0, 10.0))
global_inter_ratio_h_ylim = get_dynamic_ylim(global_inter_ratio_h_vals, default=(0, 10.0))
global_lh_ratio_ylim = get_dynamic_ylim(global_lh_ratio_vals, default=(0, 3.0))
global_lh_abs_diff_ylim = get_dynamic_ylim(global_lh_abs_diff_vals, default=(0, 1.0))
global_raw_ylim = get_dynamic_ylim(global_raw_pts, default=(0, 255), pad_ratio=0.15)
global_log_ylim = get_dynamic_ylim(global_log_pts, default=(0, 5.0), pad_ratio=0.15)

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
    base_results_dir = f'results/thickness_decoupling/H_L_fit/20260429_mask_generated/{f_type}'
    
    cur_voltages = voltages_0429
    if f_type == '0.6mm' and include_0331:
        cur_voltages = voltages_0331 + voltages_0429
        
    for voltage in cur_voltages:
        is_0331 = voltage in voltages_0331
        cur_input_dir = input_dir_0331 if is_0331 else input_dir_0429
        cur_I0 = I0_0331 if is_0331 else I0_0429
        cur_thick_map = thicknesses_0331 if is_0331 else thicknesses_0429
        
        print(f">>> Processing {voltage}...")
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

        # Ore 分析 (仅针对 0429 文件夹)
        if not is_0331:
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

    # 绘制该 Filter 的汇总图
    print(f">>> Generating Slope Summary Plot for {f_type} (Global Scaling)...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(fr"Attenuation Slopes ($\mu$) Analysis - {f_type} (Unified Scaling)", fontsize=18)
    
    slope_storage = global_slope_storage[f_type]
    sort_idx = np.argsort(slope_storage['Cu_step']['v'])
    
    for mat in mats:
        v = np.array(slope_storage[mat]['v'])[sort_idx]
        ml = np.array(slope_storage[mat]['mu_l'])[sort_idx]
        mh = np.array(slope_storage[mat]['mu_h'])[sort_idx]
        
        axes[0, 0].plot(v, ml, 'o-', label=mat)
        axes[0, 1].plot(v, mh, 'o-', label=mat)
        axes[0, 2].plot(v, ml/mh, 's--', label=f"{mat} L/H")
    
    for i in range(len(mats)):
        for j in range(i+1, len(mats)):
            m1, m2 = mats[i], mats[j]
            v = np.array(slope_storage[m1]['v'])[sort_idx]
            # 计算 L/H 比值和差值
            r1 = np.array(slope_storage[m1]['mu_l'])[sort_idx] / np.array(slope_storage[m1]['mu_h'])[sort_idx]
            r2 = np.array(slope_storage[m2]['mu_l'])[sort_idx] / np.array(slope_storage[m2]['mu_h'])[sort_idx]
            
            r_l = np.array(slope_storage[m1]['mu_l'])[sort_idx] / np.array(slope_storage[m2]['mu_l'])[sort_idx]
            r_h = np.array(slope_storage[m1]['mu_h'])[sort_idx] / np.array(slope_storage[m2]['mu_h'])[sort_idx]
            
            line_l, = axes[1, 0].plot(v, r_l, 'v-', label=f"{m1}/{m2} (L)")
            line_h, = axes[1, 1].plot(v, r_h, '^-', label=f"{m1}/{m2} (H)")
            # 绘制 L/H 的绝对差值到 Subplot 6
            axes[1, 2].plot(v, np.abs(r1 - r2), 'D-.', label=f"|{m1}-{m2}| (L/H)")
            
            if m1 in mat_physics and m2 in mat_physics:
                p1, p2 = mat_physics[m1], mat_physics[m2]
                
                # 低能理论值: 仅考虑光电效应 sigma_ph ~ Z^4.5/E^3
                # 比值 R_L = (Z1^4.5 / Ar1 * rho1) / (Z2^4.5 / Ar2 * rho2)
                theo_l = ( (p1['Z']**4.5) / p1['Ar'] * p1['rho'] ) / ( (p2['Z']**4.5) / p2['Ar'] * p2['rho'] )
                axes[1, 0].axhline(y=theo_l, color=line_l.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (PH)")
                
                # 高能理论值: 仅考虑康普顿散射 sigma_C ~ 0.665 * Z
                # 比值 R_H = (Z1 / Ar1 * rho1) / (Z2 / Ar2 * rho2)
                theo_h = ( p1['Z'] / p1['Ar'] * p1['rho'] ) / ( p2['Z'] / p2['Ar'] * p2['rho'] )
                axes[1, 1].axhline(y=theo_h, color=line_h.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (C)")

    axes[0, 0].set_title(r"$\mu_L$ vs Voltage"); axes[0, 0].set_ylabel(r"Slope ($mm^{-1}$)")
    axes[0, 1].set_title(r"$\mu_H$ vs Voltage"); axes[0, 1].set_ylabel(r"Slope ($mm^{-1}$)")
    axes[0, 2].set_title(r"$\mu_L / \mu_H$ Ratio")
    axes[1, 0].set_title(r"Inter-Material Ratio ($\mu_L$)")
    axes[1, 1].set_title(r"Inter-Material Ratio ($\mu_H$)")
    axes[1, 2].set_title(r"Inter-Material L/H Abs Diff $|r_1 - r_2|$")

    for ax in axes.flat:
        ax.set_xlabel("Voltage (kV)"); ax.grid(True); ax.legend(fontsize='x-small')
        if ax in [axes[0, 0], axes[0, 1]]: ax.set_ylim(global_mu_ylim)
        elif ax == axes[0, 2]: ax.set_ylim(global_lh_ratio_ylim)
        elif ax == axes[1, 0]: ax.set_ylim(global_inter_ratio_l_ylim)
        elif ax == axes[1, 1]: ax.set_ylim(global_inter_ratio_h_ylim)
        elif ax == axes[1, 2]: ax.set_ylim(global_lh_abs_diff_ylim)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    suffix = "_with_0331" if (f_type == '0.6mm' and include_0331) else ""
    plt.savefig(f"{base_results_dir}/slope_summary_{f_type}{suffix}.png")
    plt.close()

# --- 保存数据到 JSON ---
import json

output_json_path = 'attenuation_slopes.json'
json_data = {}

for ft in filter_types:
    json_data[ft] = {}
    for mat in step_mats.values():
        json_data[ft][mat] = {}
        # 获取该材质的所有数据并按电压排序
        v_list = global_slope_storage[ft][mat]['v']
        ul_list = global_slope_storage[ft][mat]['mu_l']
        uh_list = global_slope_storage[ft][mat]['mu_h']
        
        # 排序
        s_idx = np.argsort(v_list)
        for i in s_idx:
            v_str = f"{v_list[i]}kV"
            # 处理 NaN，JSON 不支持 NaN，转换为 None
            ul = float(ul_list[i]) if np.isfinite(ul_list[i]) else None
            uh = float(uh_list[i]) if np.isfinite(uh_list[i]) else None
            json_data[ft][mat][v_str] = {"ul": ul, "uh": uh}

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print(f"\nAll attenuation slopes saved to: {output_json_path}")
print(f"All analysis complete (0331 + 0429).")
