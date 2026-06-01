import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import utils_II

def load_any_dual_pixels(file_path, flip=False):
    """Loads low and high energy pixels and detects if it's a step sample (list) or simple sample (ndarray)."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    low = data['pixels_low']
    high = data['pixels_high']
    is_step = isinstance(low, list)
    
    if is_step and flip:
        return low[::-1], high[::-1], True
    return low, high, is_step

def find_linear_pts(x_pts, y_pts, label=""):
    """
    自动寻找阶梯对数衰减曲线中的最佳线性区间。
    
    参数类型、含义及用法：
    - x_pts (np.ndarray): X 轴坐标数组，一般代表阶梯厚度 (Thickness in mm)。
    - y_pts (np.ndarray): Y 轴坐标数组，一般代表对数衰减值 (Log attenuation ln(I0/I))。
    - label (str, 可选): 该材质/通道的标识，用于分析。
    
    返回值：
    - best_n (int): 最长的线性区域段数（对应前 best_n 个点）。
    """
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
    通用 2x3 综合分析绘图函数。
    
    参数类型、含义及用法：
    - voltage (str): 当前处理的电压或曝光时间描述字符串（如 '160kV', '270us'）。
    - samples_dict (dict): 样本数据字典，格式为 {mat_name: (L_list, H_list)}，其中 L_list 和 H_list 分别为包含各阶梯像素数组的列表。
    - output_subdir (str): 结果保存的子目录路径。
    - title_prefix (str): 图表标题前缀（如 '0331 Yinshan'）。
    - x_label (str): X 轴的物理含义标签（如 'Thickness (mm)'）。
    - x_coords_dict (dict): X 轴坐标字典，格式为 {mat_name: ndarray}，指示各材质阶梯的物理厚度。
    - color_by_step (bool): 是否按照步进的索引使用不同的颜色绘制散点图，默认 False。
    - plot_mode (str): 散点图绘制模式，'all' 绘制全部像素，'means' 仅绘制均值，默认 'all'。
    - I0 (float): 入射背景对数灰度参考值，8位下默认为 204.0，16位下默认为 52428.0。
    - raw_lims_global (tuple/None): 强度坐标轴的自适应全局统一限制，默认为 None。
    - log_lims_global (tuple/None): 衰减对数坐标轴的自适应全局统一限制，默认为 None。
    
    返回值：
    - None (图片会直接保存到磁盘)
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
                    m_l, m_h = np.mean(lv), np.mean(hv)
                    all_L_pts.append(m_l); all_H_pts.append(m_h)
                    all_log_L_pts.append(np.log(I0 / max(m_l, 1e-6))); all_log_H_pts.append(np.log(I0 / max(m_h, 1e-6)))
                else:
                    all_L_pts.append(lv); all_H_pts.append(hv)
                    all_log_L_pts.append(np.log(I0 / np.maximum(lv, 1e-6))); all_log_H_pts.append(np.log(I0 / np.maximum(hv, 1e-6)))

    if not all_L_pts: 
        plt.close()
        return

    def get_robust_range(data_list):
        if isinstance(data_list[0], np.ndarray):
            combined = np.concatenate(data_list)
            return np.percentile(combined, [2, 98])
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

    for mat_name, (L_list, H_list) in samples_dict.items():
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

        eb_alpha = 0.3 if plot_mode == 'means' else 0.6
        line, = axes[0, 1].plot(plot_x, step_L_means, 'o-', markersize=5, label=display_label, linewidth=1.5)
        base_color = line.get_color()
        axes[0, 1].errorbar(plot_x, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)
        
        axes[0, 2].plot(plot_x, step_H_means, 'o-', markersize=5, label=display_label, linewidth=1.5, color=base_color)
        axes[0, 2].errorbar(plot_x, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=eb_alpha, color=base_color)

        for i in range(max(0, len(plot_x)-7), len(plot_x)):
            if not np.isnan(step_L_means[i]):
                axes[0, 1].text(plot_x[i], step_L_means[i]+2, f"{step_L_means[i]:.1f}", fontsize=8, ha='center', va='bottom', color=base_color)
            if not np.isnan(step_H_means[i]):
                axes[0, 2].text(plot_x[i], step_H_means[i]+2, f"{step_H_means[i]:.1f}", fontsize=8, ha='center', va='bottom', color=base_color)

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
                        else:
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
            
            valid_m = ~np.isnan(step_L_means) & ~np.isnan(step_H_means)
            if np.sum(valid_m) > 2:
                coeffs = np.polyfit(step_L_means[valid_m], step_H_means[valid_m], 2)
                x_fit = np.linspace(raw_lims[0], raw_lims[1], 100)
                axes[0, 0].plot(x_fit, np.poly1d(coeffs)(x_fit), color=fit_color, label=f"{display_label} Fit")

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
                        else:
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
                
            valid_m = ~np.isnan(log_L_means) & ~np.isnan(log_H_means) & np.isfinite(log_L_means) & np.isfinite(log_H_means)
            if np.sum(valid_m) > 2:
                l_coeffs = np.polyfit(log_L_means[valid_m], log_H_means[valid_m], 2)
                x_fit_log = np.linspace(log_lims[0], log_lims[1], 100)
                axes[1, 0].plot(x_fit_log, np.poly1d(l_coeffs)(x_fit_log), color=fit_color, label=f"{display_label} Fit")

        line_color = base_color
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

    # 限制设置
    axes[0, 0].set_xlim(raw_lims); axes[0, 0].set_ylim(raw_lims)
    axes[0, 1].set_xlim(x_lims);   axes[0, 1].set_ylim(raw_lims)
    axes[0, 2].set_xlim(x_lims);   axes[0, 2].set_ylim(raw_lims)
    
    axes[1, 0].set_xlim(log_lims); axes[1, 0].set_ylim(log_lims)
    axes[1, 1].set_xlim(x_lims);   axes[1, 1].set_ylim(log_lims)
    axes[1, 2].set_xlim(x_lims);   axes[1, 2].set_ylim(log_lims)

    # 样式
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
    plt.suptitle(full_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_img_path = f"{output_subdir}/{voltage}_analysis.png"
    plt.savefig(save_img_path)
    plt.close()
    print(f"Comprehensive analysis plot saved to {save_img_path}")


def plot_step_means(configs, title_suffix, save_path):
    """Plots means for stepped samples (1-10 steps)."""
    plt.figure(figsize=(12, 7))
    steps = np.arange(1, 11)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, cfg in enumerate(configs):
        low, high = cfg['data']
        label = cfg['label']
        color = colors[i % 10]
        m_low = [np.mean(p) for p in low]
        m_high = [np.mean(p) for p in high]
        plt.plot(steps, m_low, 'o-', color=color, label=f'{label} - Low')
        plt.plot(steps, m_high, 's--', color=color, label=f'{label} - High', alpha=0.6)
    
    plt.title(f'Steps Mean Comparison: {title_suffix}', fontsize=14)
    plt.xlabel('Thickness Step (1-10)')
    plt.ylabel('Mean Intensity')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Step mean plot saved to {save_path}")

def plot_simple_correlation(configs_x, configs_y, label_x, label_y, title_suffix, save_path):
    """Plots a scatter correlation between two sets of simple samples (e.g. 125us vs 270us)."""
    plt.figure(figsize=(8, 8))
    
    # Extract means
    low_x = [np.mean(c['data'][0]) for c in configs_x]
    high_x = [np.mean(c['data'][1]) for c in configs_x]
    low_y = [np.mean(c['data'][0]) for c in configs_y]
    high_y = [np.mean(c['data'][1]) for c in configs_y]
    
    # Plot Scatter
    plt.scatter(low_x, low_y, marker='o', color='blue', s=60, label='Low Energy', edgecolors='white', alpha=0.9)
    plt.scatter(high_x, high_y, marker='s', color='orange', s=60, label='High Energy', edgecolors='white', alpha=0.9)
    
    # Add Identity Line (Reference)
    all_vals = low_x + high_x + low_y + high_y
    v_min, v_max = min(all_vals)*0.95, max(all_vals)*1.05
    plt.plot([v_min, v_max], [v_min, v_max], 'k--', alpha=0.5, label='Identity (y=x)')
    
    # Annotate points with labels
    for i, cfg in enumerate(configs_x):
        plt.annotate(cfg['label'], (low_x[i], low_y[i]), xytext=(5,5), textcoords='offset points', fontsize=9)
        plt.annotate(cfg['label'], (high_x[i], high_y[i]), xytext=(5,5), textcoords='offset points', fontsize=9)

    plt.title(f'Intensity Correlation: {title_suffix}', fontsize=14)
    plt.xlabel(f'{label_x} Mean Intensity')
    plt.ylabel(f'{label_y} Mean Intensity')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Correlation plot saved to {save_path}")

def plot_simple_means(configs, title_suffix, save_path):
    """Plots means for simple samples as a comparative bar chart."""
    plt.figure(figsize=(10, 6))
    
    if len(configs) == 2:
        cfg1, cfg2 = configs[0], configs[1]
        l1, l2 = cfg1['label'], cfg2['label']
        
        m_low1, m_low2 = np.mean(cfg1['data'][0]), np.mean(cfg2['data'][0])
        m_high1, m_high2 = np.mean(cfg1['data'][1]), np.mean(cfg2['data'][1])
        
        # We group by energy level: Group 0 is Low, Group 1 is High
        x = np.array([0.0, 1.2])
        width = 0.35
        
        # Low energy grouped bars
        bar_l1 = plt.bar(x[0] - width/2, m_low1, width, label=f'{l1} - Low', color='#1f77b4', edgecolor='black', linewidth=0.5)
        bar_l2 = plt.bar(x[0] + width/2, m_low2, width, label=f'{l2} - Low', color='#aec7e8', edgecolor='black', linewidth=0.5)
        
        # High energy grouped bars
        bar_h1 = plt.bar(x[1] - width/2, m_high1, width, label=f'{l1} - High', color='#d62728', edgecolor='black', linewidth=0.5)
        bar_h2 = plt.bar(x[1] + width/2, m_high2, width, label=f'{l2} - High', color='#ff9896', edgecolor='black', linewidth=0.5)
        
        # Relative differences
        diff_low_rel = (m_low2 - m_low1) / m_low1 * 100
        diff_high_rel = (m_high2 - m_high1) / m_high1 * 100
        
        # Text annotations above the bars
        plt.text(x[0] - width/2, m_low1 + 100, f"{m_low1:.1f}", ha='center', va='bottom', fontsize=9, color='#1f77b4', fontweight='bold')
        plt.text(x[0] + width/2, m_low2 + 100, f"{m_low2:.1f}", ha='center', va='bottom', fontsize=9, color='#1c5380', fontweight='bold')
        
        sign_l = "+" if diff_low_rel >= 0 else ""
        plt.text(x[0], max(m_low1, m_low2) * 1.05, f"Diff: {sign_l}{diff_low_rel:.2f}%", ha='center', va='bottom', fontsize=11, color='blue', fontweight='bold')
        
        plt.text(x[1] - width/2, m_high1 + 100, f"{m_high1:.1f}", ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold')
        plt.text(x[1] + width/2, m_high2 + 100, f"{m_high2:.1f}", ha='center', va='bottom', fontsize=9, color='#8f1c1c', fontweight='bold')
        
        sign_h = "+" if diff_high_rel >= 0 else ""
        plt.text(x[1], max(m_high1, m_high2) * 1.05, f"Diff: {sign_h}{diff_high_rel:.2f}%", ha='center', va='bottom', fontsize=11, color='red', fontweight='bold')
        
        plt.xticks(x, ['Low Energy (低能)', 'High Energy (高能)'], fontsize=12)
        plt.ylim(0, max(m_low1, m_low2, m_high1, m_high2) * 1.25)
    else:
        labels = [c['label'] for c in configs]
        m_lows = [np.mean(c['data'][0]) for c in configs]
        m_highs = [np.mean(c['data'][1]) for c in configs]
        
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, m_lows, width, label='Low Energy', color='skyblue')
        plt.bar(x + width/2, m_highs, width, label='High Energy', color='coral')
        plt.xticks(x, labels, rotation=45)
        plt.ylim(0, max(max(m_lows), max(m_highs)) * 1.15)
        
    plt.title(f'Sample Means Comparison: {title_suffix}', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Intensity (平均灰度值)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Simple mean plot saved to {save_path}")

def plot_step_hist_grid(configs, channel_name, title_suffix, save_path):
    """Plots density histograms for stepped datasets in a 2x5 grid."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    axes = axes.flatten()
    if len(configs) == 2:
        colors = ['#1f77b4', '#d62728'] # Blue vs Red for high contrast
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        
    idx = 0 if channel_name.lower() == 'low' else 1
    for i in range(10):
        ax = axes[i]
        all_d = [cfg['data'][idx][i] for cfg in configs]
        v_min = min(np.min(d) for d in all_d)
        v_max = max(np.max(d) for d in all_d)
        bins = np.linspace(v_min, v_max, 50)
        
        for j, cfg in enumerate(configs):
            ax.hist(all_d[j], bins=bins, alpha=0.45, label=cfg['label'], color=colors[j], density=True, edgecolor='black', linewidth=0.3)
        
        ax.set_title(f'Step {i+1}')
        ax.set_xlabel('Intensity')
        if i == 0:
            ax.legend(loc='upper right', prop={'size': 9})
    
    fig.suptitle(f'Steps ({title_suffix}): {channel_name} Energy', fontsize=18)
    plt.savefig(save_path)
    print(f"Step histogram grid saved to {save_path}")

def plot_simple_hist_grid(configs_x, configs_y, channel_name, title_suffix, save_path):
    """Plots a grid of histograms where each subplot compares a specific ore from Dataset X vs Dataset Y."""
    num_ores = len(configs_x)
    cols = 2
    rows = (num_ores + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    idx = 0 if channel_name.lower() == 'low' else 1
    
    for i in range(num_ores):
        ax = axes[i]
        d_x = configs_x[i]['data'][idx]
        d_y = configs_y[i]['data'][idx]
        l_x = configs_x[i]['label']
        l_y = configs_y[i]['label']
        
        v_min, v_max = min(np.min(d_x), np.min(d_y)), max(np.max(d_x), np.max(d_y))
        bins = np.linspace(v_min, v_max, 60)
        
        ax.hist(d_x, bins=bins, alpha=0.45, label=f"125us-{l_x}", color='#1f77b4', density=True, edgecolor='black', linewidth=0.3)
        ax.hist(d_y, bins=bins, alpha=0.45, label=f"270us-{l_y}", color='#d62728', density=True, edgecolor='black', linewidth=0.3)
        
        ax.set_title(f'Comparison: {l_x} vs {l_y}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide unused axes
    for i in range(num_ores, len(axes)):
        axes[i].axis('off')
        
    fig.suptitle(f'Simple Sample ({title_suffix}): {channel_name} Energy Per-Ore Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Simple histogram grid saved to {save_path}")

def plot_simple_hist(configs, channel_name, title_suffix, save_path):
    # (Kept for non-XY mode simple comparisons if any)
    plt.figure(figsize=(12, 7))
    if len(configs) == 2:
        colors = ['#1f77b4', '#d62728'] # Blue vs Red for high contrast
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        
    idx = 0 if channel_name.lower() == 'low' else 1
    all_d = [cfg['data'][idx] for cfg in configs]
    v_min, v_max = min(np.min(d) for d in all_d), max(np.max(d) for d in all_d)
    bins = np.linspace(v_min, v_max, 80)
    
    for j, cfg in enumerate(configs):
        plt.hist(all_d[j], bins=bins, alpha=0.45, label=cfg['label'], color=colors[j], density=True, edgecolor='black', linewidth=0.5)
    
    plt.title(f'Histogram ({title_suffix}): {channel_name} Energy', fontsize=14)
    plt.xlabel('Intensity'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path); print(f"Simple overall histogram saved to {save_path}")


def run_comparison(configs_desc, title_suffix, prefix, is_xy=False):
    """Main entry point for routing to different plotting modes."""
    os.makedirs('results/Tube_comparison', exist_ok=True)
    
    if is_xy and len(configs_desc) == 2:
        # Correlation Mode (Group X vs Group Y)
        desc_x, desc_y = configs_desc[0], configs_desc[1]
        configs_x = []
        for cd in desc_x:
            low, high, _ = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            configs_x.append({'data': (low, high), 'label': cd['label']})
        configs_y = []
        for cd in desc_y:
            low, high, _ = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            configs_y.append({'data': (low, high), 'label': cd['label']})
            
        # 1. Scatter Correlation
        plot_simple_correlation(configs_x, configs_y, "125us", "270us", title_suffix, f'results/Tube_comparison/{prefix}_correlation.png')
        
        # 2. Per-Ore Histogram Grid Comparison
        plot_simple_hist_grid(configs_x, configs_y, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
        plot_simple_hist_grid(configs_x, configs_y, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')
    else:
        # Standard Mode (Sequential list)
        configs = []
        is_step_run = False
        for i, cd in enumerate(configs_desc):
            low, high, is_step = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            if i == 0: is_step_run = is_step
            configs.append({'data': (low, high), 'label': cd['label']})
        
        if is_step_run:
            plot_step_means(configs, title_suffix, f'results/Tube_comparison/{prefix}_means.png')
            plot_step_hist_grid(configs, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
            plot_step_hist_grid(configs, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')
        else:
            plot_simple_means(configs, title_suffix, f'results/Tube_comparison/{prefix}_means.png')
            plot_simple_hist(configs, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
            plot_simple_hist(configs, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')

def generate_dataset_slope_summaries(dataset_name, output_subdir, global_raw_data, thicknesses, mu_mode='mu'):
    """
    针对给定的数据集绘制所有阶梯厚度下的衰减系数随电压变化的 2x3 汇总图。
    
    参数类型、含义及用法：
    - dataset_name (str): 数据集名称（如 "0331" 或 "0409"）。
    - output_subdir (str): 图像保存的子目录（如 "results/Tube_comparison/comprehensive_fit/0331"）。
    - global_raw_data (dict): 全局收集的数据，格式为 {mat_name: {'v': [...], 'cur_x': [...], 'log_l': [...], 'log_h': [...], 'vl': [...], 'vh': [...]}}。
    - thicknesses (dict): 物理厚度字典。
    - mu_mode (str): 衰减系数模式，'mu' 代表线衰减系数 (Slope, mm^-1)，'mu_m' 代表质量衰减系数 (cm^2/g)。
    
    返回值：
    - None (图片会直接保存到磁盘)
    """
    os.makedirs(output_subdir, exist_ok=True)
    
    # 物理常数定义
    mat_physics = {
        'Cu_step': {'Z': 29, 'Ar': 63.546, 'rho': 8.96},
        'Fe_step': {'Z': 26, 'Ar': 55.845, 'rho': 7.87},
        'Al_step': {'Z': 13, 'Ar': 26.982, 'rho': 2.70}
    }
    
    def get_dynamic_ylim(vals, default=(0, 1.0), pad_ratio=0.1):
        vals = np.array(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0: return default
        v_min, v_max = np.percentile(vals, 1), np.percentile(vals, 99)
        pad = (v_max - v_min) * pad_ratio if v_max > v_min else 0.1
        return (v_min - pad, v_max + pad)

    max_steps = 10
    mats = ['Cu_step', 'Fe_step', 'Al_step']
    
    for step_idx in range(max_steps):
        cu_th = thicknesses['Cu_step'][step_idx] if step_idx < len(thicknesses['Cu_step']) else "N/A"
        fe_th = thicknesses['Fe_step'][step_idx] if step_idx < len(thicknesses['Fe_step']) else "N/A"
        al_th = thicknesses['Al_step'][step_idx] if step_idx < len(thicknesses['Al_step']) else "N/A"
        step_name = f"{cu_th}mm_Cu_{fe_th}mm_Fe_{al_th}mm_Al"
        
        step_storage = {mat: {'v': [], 'mu_l': [], 'mu_h': []} for mat in mats}
        step_mu_vals, step_lh_vals, step_inter_l, step_inter_h, step_diff = [], [], [], [], []
        
        for mat in mats:
            v_list = global_raw_data[mat]['v']
            for i in range(len(v_list)):
                v_int = v_list[i]
                cur_x = global_raw_data[mat]['cur_x'][i]
                log_l = global_raw_data[mat]['log_l'][i]
                log_h = global_raw_data[mat]['log_h'][i]
                vl = global_raw_data[mat]['vl'][i]
                vh = global_raw_data[mat]['vh'][i]
                
                if step_idx < len(cur_x):
                    t_mm = cur_x[step_idx]
                    if t_mm > 0:
                        if mu_mode == 'mu_m':
                            rho = mat_physics[mat]['rho']
                            mu_l = 10.0 * log_l[step_idx] / (t_mm * rho) if vl[step_idx] else np.nan
                            mu_h = 10.0 * log_h[step_idx] / (t_mm * rho) if vh[step_idx] else np.nan
                        else:
                            mu_l = log_l[step_idx] / t_mm if vl[step_idx] else np.nan
                            mu_h = log_h[step_idx] / t_mm if vh[step_idx] else np.nan
                    else:
                        mu_l, mu_h = np.nan, np.nan
                else:
                    mu_l, mu_h = np.nan, np.nan
                
                step_storage[mat]['v'].append(v_int)
                step_storage[mat]['mu_l'].append(mu_l)
                step_storage[mat]['mu_h'].append(mu_h)
                
                if np.isfinite(mu_l): step_mu_vals.append(mu_l)
                if np.isfinite(mu_h): step_mu_vals.append(mu_h)
                if np.isfinite(mu_l) and np.isfinite(mu_h): step_lh_vals.append(mu_l/np.maximum(mu_h, 1e-9))
        
        for i in range(len(mats)):
            for j in range(i+1, len(mats)):
                m1, m2 = mats[i], mats[j]
                sl1, sl2 = np.array(step_storage[m1]['mu_l']), np.array(step_storage[m2]['mu_l'])
                sh1, sh2 = np.array(step_storage[m1]['mu_h']), np.array(step_storage[m2]['mu_h'])
                
                valid_l = np.isfinite(sl1) & np.isfinite(sl2) & (sl2 != 0)
                valid_h = np.isfinite(sh1) & np.isfinite(sh2) & (sh2 != 0)
                step_inter_l.extend(sl1[valid_l] / sl2[valid_l])
                step_inter_h.extend(sh1[valid_h] / sh2[valid_h])
                
                r1, r2 = sl1 / np.maximum(sh1, 1e-9), sl2 / np.maximum(sh2, 1e-9)
                diff = np.abs(r1 - r2)
                step_diff.extend(diff[np.isfinite(diff)])
        
        cur_mu_ylim = get_dynamic_ylim(step_mu_vals)
        cur_inter_ratio_l_ylim = get_dynamic_ylim(step_inter_l, default=(0, 10.0))
        cur_inter_ratio_h_ylim = get_dynamic_ylim(step_inter_h, default=(0, 10.0))
        cur_lh_ratio_ylim = get_dynamic_ylim(step_lh_vals, default=(0, 3.0))
        cur_lh_abs_diff_ylim = get_dynamic_ylim(step_diff, default=(0, 1.0))
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        if mu_mode == 'mu_m':
            mu_L_symbol = r"\mu_{m, L}"
            mu_H_symbol = r"\mu_{m, H}"
            mu_symbol = r"\mu_m"
        else:
            mu_L_symbol = r"\mu_L"
            mu_H_symbol = r"\mu_H"
            mu_symbol = r"\mu"
        mu_desc = "Mass Attenuation" if mu_mode == 'mu_m' else "Linear Attenuation"
        fig.suptitle(f"{mu_desc} (${mu_symbol}$) Analysis - {dataset_name} (Step: {step_name})", fontsize=18)
        
        sort_idx = np.argsort(step_storage['Cu_step']['v'])
        for mat in mats:
            v = np.array(step_storage[mat]['v'])[sort_idx]
            ml = np.array(step_storage[mat]['mu_l'])[sort_idx]
            mh = np.array(step_storage[mat]['mu_h'])[sort_idx]
            
            axes[0, 0].plot(v, ml, 'o-', label=mat)
            axes[0, 1].plot(v, mh, 'o-', label=mat)
            axes[0, 2].plot(v, ml/np.maximum(mh, 1e-9), 's--', label=f"{mat} {mu_L_symbol}/{mu_H_symbol}")
        
        for i in range(len(mats)):
            for j in range(i+1, len(mats)):
                m1, m2 = mats[i], mats[j]
                v = np.array(step_storage[m1]['v'])[sort_idx]
                r1 = np.array(step_storage[m1]['mu_l'])[sort_idx] / np.maximum(np.array(step_storage[m1]['mu_h'])[sort_idx], 1e-9)
                r2 = np.array(step_storage[m2]['mu_l'])[sort_idx] / np.maximum(np.array(step_storage[m2]['mu_h'])[sort_idx], 1e-9)
                r_l = np.array(step_storage[m1]['mu_l'])[sort_idx] / np.maximum(np.array(step_storage[m2]['mu_l'])[sort_idx], 1e-9)
                r_h = np.array(step_storage[m1]['mu_h'])[sort_idx] / np.maximum(np.array(step_storage[m2]['mu_h'])[sort_idx], 1e-9)
                
                line_l, = axes[1, 0].plot(v, r_l, 'v-', label=f"{m1}/{m2} ({mu_L_symbol})")
                line_h, = axes[1, 1].plot(v, r_h, '^-', label=f"{m1}/{m2} ({mu_H_symbol})")
                axes[1, 2].plot(v, np.abs(r1 - r2), 'D-.', label=f"|{m1}-{m2}| ({mu_symbol})")
 
                if m1 in mat_physics and m2 in mat_physics:
                    p1, p2 = mat_physics[m1], mat_physics[m2]
                    if mu_mode == 'mu_m':
                        theo_l = ( (p1['Z']**4.5) / p1['Ar'] ) / ( (p2['Z']**4.5) / p2['Ar'] )
                        theo_h = ( p1['Z'] / p1['Ar'] ) / ( p2['Z'] / p2['Ar'] )
                    else:
                        theo_l = ( (p1['Z']**4.5) / p1['Ar'] * p1['rho'] ) / ( (p2['Z']**4.5) / p2['Ar'] * p2['rho'] )
                        theo_h = ( p1['Z'] / p1['Ar'] * p1['rho'] ) / ( p2['Z'] / p2['Ar'] * p2['rho'] )
                    
                    axes[1, 0].axhline(y=theo_l, color=line_l.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (PH)")
                    axes[1, 1].axhline(y=theo_h, color=line_h.get_color(), linestyle='--', alpha=0.6, label=f"{m1}/{m2} Theo (C)")
 
        v_min_x = min(step_storage['Cu_step']['v']) - 10
        v_max_x = max(step_storage['Cu_step']['v']) + 10
        for ax in axes.flat:
            ax.set_xlabel("Voltage (kV)"); ax.grid(True); ax.legend(fontsize='x-small')
            ax.set_xlim(v_min_x, v_max_x)
        
        mu_unit = r"$\mathrm{cm}^2/\mathrm{g}$" if mu_mode == 'mu_m' else r"$\mathrm{mm}^{-1}$"
        axes[0, 0].set_title(fr"${mu_L_symbol}$ vs Voltage"); axes[0, 0].set_ylabel(fr"${mu_L_symbol}\ ({mu_unit})$"); axes[0, 0].set_ylim(cur_mu_ylim)
        axes[0, 1].set_title(fr"${mu_H_symbol}$ vs Voltage"); axes[0, 1].set_ylabel(fr"${mu_H_symbol}\ ({mu_unit})$"); axes[0, 1].set_ylim(cur_mu_ylim)
        axes[0, 2].set_title(fr"${mu_L_symbol} / {mu_H_symbol}$ Ratio"); axes[0, 2].set_ylim(cur_lh_ratio_ylim)
        axes[1, 0].set_title(fr"Inter-Material Ratio (${mu_L_symbol}$)"); axes[1, 0].set_ylim(cur_inter_ratio_l_ylim)
        axes[1, 1].set_title(fr"Inter-Material Ratio (${mu_H_symbol}$)"); axes[1, 1].set_ylim(cur_inter_ratio_h_ylim)
        axes[1, 2].set_title(fr"Inter-Material ${mu_L_symbol}/{mu_H_symbol}$ Abs Diff"); axes[1, 2].set_ylim(cur_lh_abs_diff_ylim)
 
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_img_path = f"{output_subdir}/slope_summary_{step_name}.png"
        plt.savefig(save_img_path)
        plt.close()
        print(f"Slope summary saved to {save_img_path}")

def run_stepped_specimen_analysis(mu_mode='mu'):
    """
    对 0331、0407 和 0409 三个数据集进行相同的阶梯试样 H-L 曲线及对数衰减综合分析。
    
    参数类型、含义及用法：
    - mu_mode (str): 衰减系数模式，'mu' 代表线衰减系数 (Slope, mm^-1)，'mu_m' 代表质量衰减系数 (cm^2/g)。
      
    返回值：
    - None
    """
    print("\n==========================================")
    print("RUNNING COMPREHENSIVE STEP SAMPLE ANALYSIS")
    print("==========================================")
    
    out_dir_base = 'results/Tube_comparison/comprehensive_fit'
    os.makedirs(out_dir_base, exist_ok=True)
    
    # 标样厚度坐标 (10 steps)
    thicknesses_std = {
        'Cu_step': np.arange(2, 22, 2),
        'Fe_step': np.arange(2, 22, 2),
        'Al_step': np.arange(12, 32, 2)
    }
    
    # 1. 0331 Dataset (Yinshan) - Multi-voltage (140kV, 160kV, 180kV) 16-bit
    print("\n>>> Processing 0331 Dataset (16-bit)...")
    I0_16bit = 52428.0
    voltages_0331 = ['140kV', '160kV', '180kV']
    input_dir_0331 = 'results/20260331_16bit/pixel_values'
    step_mats_0331 = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
    
    global_raw_0331 = {mat: {'v': [], 'cur_x': [], 'log_l': [], 'log_h': [], 'vl': [], 'vh': []} for mat in thicknesses_std.keys()}
    
    for voltage in voltages_0331:
        step_data = {}
        v_int = int(re.search(r'(\d+)', voltage).group(1))
        
        for idx, name in step_mats_0331.items():
            p = f'{input_dir_0331}/{voltage}_4mA_step_sample_{idx}_data.pkl'
            if os.path.exists(p):
                low, high, _ = load_any_dual_pixels(p, flip=False)
                step_data[name] = (low, high)
                
                # 为 slope_summary 收集衰减对数 (16-bit)
                log_l = np.array([np.mean(np.log(I0_16bit / np.maximum(px, 1.0))) for px in low])
                log_h = np.array([np.mean(np.log(I0_16bit / np.maximum(px, 1.0))) for px in high])
                vl, vh = np.isfinite(log_l), np.isfinite(log_h)
                
                global_raw_0331[name]['v'].append(v_int)
                global_raw_0331[name]['cur_x'].append(thicknesses_std[name])
                global_raw_0331[name]['log_l'].append(log_l)
                global_raw_0331[name]['log_h'].append(log_h)
                global_raw_0331[name]['vl'].append(vl)
                global_raw_0331[name]['vh'].append(vh)
                
        if step_data:
            perform_comprehensive_analysis(
                voltage=voltage,
                samples_dict=step_data,
                output_subdir=f"{out_dir_base}/0331",
                title_prefix="0331 (yinshan) Step (16-bit)",
                x_label="Thickness (mm)",
                x_coords_dict=thicknesses_std,
                plot_mode='all',
                I0=I0_16bit
            )
            
    # 绘制 0331 的 slope_summary
    print(f"\n>>> Generating 0331 Slope Summary Plots ({mu_mode})...")
    generate_dataset_slope_summaries("0331 (yinshan) (16-bit)", f"{out_dir_base}/0331/slope_summary_{mu_mode}", global_raw_0331, thicknesses_std, mu_mode=mu_mode)

    # 2. 0407 Dataset (Home) - 160kV 16-bit (DISABLED per user request)
    # print("\n>>> Processing 0407 Dataset (16-bit)...")
    # runs_0407 = ['test1', 'test2', 'test3']
    # input_dir_0407 = 'results/20260407_Sample_test_16bit/pixel_values'
    # 
    # for run in runs_0407:
    #     step_data = {}
    #     step_mapping = {'Cu_step': 1, 'Fe_step': 3}
    #     if run == 'test1':
    #         step_mapping['Al_step'] = 5
    #         
    #     for name, idx in step_mapping.items():
    #         p = f'{input_dir_0407}/Sample_160kV_{run}_step_sample_{idx}_data.pkl'
    #         if os.path.exists(p):
    #             low, high, _ = load_any_dual_pixels(p, flip=True) # 0407 has flip=True!
    #             step_data[name] = (low, high)
    #     if step_data:
    #         perform_comprehensive_analysis(
    #             voltage=f"160kV_{run}",
    #             samples_dict=step_data,
    #             output_subdir=f"{out_dir_base}/0407",
    #             title_prefix=f"0407 (home) Step {run} (16-bit)",
    #             x_label="Thickness (mm)",
    #             x_coords_dict=thicknesses_std,
    #             plot_mode='all',
    #             I0=I0_16bit
    #         )

    # 3. 0409 Dataset (TYM) - Multi-voltage (160kV, 180kV, 200kV) 125us 16-bit (cropped)
    print("\n>>> Processing 0409 Dataset (160kV, 180kV, 200kV 125us 16-bit cropped)...")
    voltages_0409 = ['160kV', '180kV', '200kV']
    input_dir_0409 = 'results/TYM_test_2_16bit/pixel_values'
    
    # 动态映射各电压的 contour ID
    # 0409 125us 16-bit cropped 排序提取后的索引映射：Al_step=6, Cu_step=8, Fe_step=9
    step_mappings_0409 = {
        '160kV': {'Al_step': 6, 'Cu_step': 8, 'Fe_step': 9},
        '180kV': {'Al_step': 6, 'Cu_step': 8, 'Fe_step': 9},
        '200kV': {'Al_step': 6, 'Cu_step': 8, 'Fe_step': 9}
    }
    
    global_raw_0409 = {mat: {'v': [], 'cur_x': [], 'log_l': [], 'log_h': [], 'vl': [], 'vh': []} for mat in thicknesses_std.keys()}
    
    for voltage in voltages_0409:
        step_data = {}
        v_int = int(re.search(r'(\d+)', voltage).group(1))
        mapping = step_mappings_0409[voltage]
        
        for name, idx in mapping.items():
            # 匹配 16-bit cropped 文件，注意文件名大小写敏感性（160kv vs 160kV）
            v_str_lower = voltage.lower()
            p = f'{input_dir_0409}/{v_str_lower}-2mA-125us-0.5pF-disc-post_calib_cropped_step_sample_{idx}_data.pkl'
            
            if os.path.exists(p):
                low, high, _ = load_any_dual_pixels(p, flip=False)
                step_data[name] = (low, high)
                
                # 为 slope_summary 收集衰减对数 (16-bit)
                log_l = np.array([np.mean(np.log(I0_16bit / np.maximum(px, 1.0))) for px in low])
                log_h = np.array([np.mean(np.log(I0_16bit / np.maximum(px, 1.0))) for px in high])
                vl, vh = np.isfinite(log_l), np.isfinite(log_h)
                
                global_raw_0409[name]['v'].append(v_int)
                global_raw_0409[name]['cur_x'].append(thicknesses_std[name])
                global_raw_0409[name]['log_l'].append(log_l)
                global_raw_0409[name]['log_h'].append(log_h)
                global_raw_0409[name]['vl'].append(vl)
                global_raw_0409[name]['vh'].append(vh)
                
        if step_data:
            perform_comprehensive_analysis(
                voltage=f"{voltage}_125us",
                samples_dict=step_data,
                output_subdir=f"{out_dir_base}/0409",
                title_prefix="0409 (TYM) Step 125us (16-bit)",
                x_label="Thickness (mm)",
                x_coords_dict=thicknesses_std,
                plot_mode='all',
                I0=I0_16bit
            )
            
    # 绘制 0409 的 slope_summary
    print(f"\n>>> Generating 0409 Slope Summary Plots ({mu_mode})...")
    generate_dataset_slope_summaries("0409 (TYM) 125us", f"{out_dir_base}/0409/slope_summary_{mu_mode}", global_raw_0409, thicknesses_std, mu_mode=mu_mode)

def main():
    # Setup plotting aesthetics for Chinese text
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # Choose attenuation coefficient mode: 'mu' (Linear, mm^-1) or 'mu_m' (Mass, cm^2/g)
    mu_mode = 'mu'

    # 1. Comparison of Steps (125us vs 270us) - Stepped Mode
    print("\n=== RUNNING STEP COMPARISON (125us vs 270us) ===")
    configs_step = [
        {"path": r'results/TYM_test_16bit/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_cropped_step_sample_5_data.pkl', "label": "125us"},
        {"path": r'results/TYM_test_16bit/pixel_values/160kv-2mA-270us-0.5pF-disc-post_calib_cropped_step_sample_6_data.pkl', "label": "270us"}
    ]
    run_comparison(configs_step, "Steps: Exposure Time", "TYM_Exposure_Steps")

    # 1b. Comparison of Step Transitions (125us vs 270us) - Transition Mode
    print("\n=== RUNNING STEP TRANSITION COMPARISON (125us vs 270us) ===")
    configs_transition = [
        {"path": r'results/TYM_test_16bit/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_cropped_step_sample_5_transition.pkl', "label": "125us_transition"},
        {"path": r'results/TYM_test_16bit/pixel_values/160kv-2mA-270us-0.5pF-disc-post_calib_cropped_step_sample_6_transition.pkl', "label": "270us_transition"}
    ]
    run_comparison(configs_transition, "Steps Transition: Exposure Time", "TYM_Exposure_Steps_Transition")

    
    # 2. Comparison of Ores (125us vs 270us) - Correlation Mode & Per-Ore Hist Grid
    print("\n=== RUNNING ORE CORRELATION & HIST GRID (125us vs 270us) ===")
    
    # MANUAL INDICES SPECIFICATION
    indices_125 = [0, 1, 3] # Ore numbers extracted from 125us data
    indices_270 = [0, 1, 3] # Ore numbers extracted from 270us data (can match differently)
    
    configs_x = [{"path": f'results/TYM_test_16bit/pixel_values/160kv-2mA-125us-0.5pF-ore-post_calib_cropped_ore_{i}_data.pkl', "label": f"Ore{i}"} for i in indices_125]
    configs_y = [{"path": f'results/TYM_test_16bit/pixel_values/160kv-2mA-270us-0.5pF-ore&step-post_calib_cropped_ore_{j}_data.pkl', "label": f"Ore{j}"} for j in indices_270]
    
    run_comparison([configs_x, configs_y], "Ores: 125us vs 270us", "TYM_Exposure_Ores", is_xy=True)

    # 3. Comprehensive Stepped Specimen Analysis (0331, 0407, 0409)
    run_stepped_specimen_analysis(mu_mode=mu_mode)

if __name__ == "__main__":
    main()
