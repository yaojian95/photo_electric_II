import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# 将父目录加入sys.path以便直接导入根目录的模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import utils_II

def find_linear_pts(x_pts: np.ndarray, y_pts: np.ndarray) -> int:
    """
    自适应寻找对数衰减曲线中最长的优秀物理线性段。
    从最薄台阶（前3个点）开始，逐步加入后续台阶，在决定系数 R^2 出现明显下降时截断。
    
    参数类型、含义及用法：
    ------------------
    - x_pts (np.ndarray): 台阶实际物理厚度的一维数组。
    - y_pts (np.ndarray): 各台阶对应的对数衰减强度均值的一维数组。
    
    返回：
    - best_n (int): 截断后的优秀线性段包含的台阶数（至少为3）。
    """
    best_n = 3
    if len(x_pts) < 3:
        return len(x_pts)
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

def get_clean_ylim(vals: list, default_max: float = 1.0) -> tuple:
    """
    计算干净且物理合理的 Y 轴显示区间限制（下限强制为 0.0，上限根据最大百分位数向上取整到美观的整数值）。
    
    参数类型、含义及用法：
    ------------------
    - vals (list or np.ndarray): 包含所有数据点的数值列表，用于统计最大值以确定 Y 轴上限。
    - default_max (float): 如果数据为空时的默认上限值，默认为 1.0。
    
    返回：
    - (lower_limit, upper_limit) (tuple): 格式为 (0.0, upper_limit) 的 Y 轴范围。
    """
    arr = np.array(vals)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (0.0, default_max)
    v_max = np.percentile(arr, 99.5)
    if v_max <= 6.0:
        upper = float(np.ceil(v_max))
        return (0.0, upper)
    else:
        if v_max > 10000:
            upper = float(np.ceil(v_max / 5000.0) * 5000)
        else:
            upper = float(np.ceil(v_max / 200.0) * 200)
        return (0.0, upper)

def perform_combined_voltage_analysis(
    energy_band_num: int,
    samples_dict_180: dict,
    samples_dict_160: dict,
    output_subdir: str,
    x_coords_dict: dict,
    I0_norm: float = 52428.0,
    I0_raw_180: float = 1500.0,
    I0_raw_160: float = 1500.0,
    raw_lims_norm: tuple = None,
    log_lims_norm: tuple = None,
    raw_lims_raw: tuple = None,
    log_lims_raw: tuple = None
) -> None:
    """
    对 180kV 与 160kV 双电压在同一能量段下的低能归一化与高能原始通道数据进行 2x2 对比分析并绘图。
    通过实线（180kV，圆形标记）与虚线（160kV，方形标记）在同一子图中画出三个材料（Cu、Fe、Al）的强度和对数衰减拟合线。
    
    参数类型、含义及用法：
    ------------------
    - energy_band_num (int): 当前能量通道的中心数值（如 20 代表 20-30keV）。
    - samples_dict_180 (dict): 180kV 样品数据字典，格式为 {mat_name: (L_list, H_list)}，L_list 和 H_list 为 10 阶像素数组列表。
    - samples_dict_160 (dict): 160kV 样品数据字典，格式同 samples_dict_180。
    - output_subdir (str): 结果图像文件保存的目标子目录。
    - x_coords_dict (dict): 各材质的厚度数组映射表，格式为 {mat_name: ndarray}。
    - I0_norm (float): 低能通道归一化背景强度参考值，默认 52428.0。
    - I0_raw_180 (float): 180kV 高能原始通道的背景强度参考值（外部计算传入）。
    - I0_raw_160 (float): 160kV 高能原始通道的背景强度参考值（外部计算传入）。
    - raw_lims_norm (tuple): 低能归一化灰度值的 Y 轴区间限制。
    - log_lims_norm (tuple): 低能归一化对数衰减的 Y 轴区间限制。
    - raw_lims_raw (tuple): 高能原始灰度值的 Y 轴区间限制。
    - log_lims_raw (tuple): 高能原始对数衰减的 Y 轴区间限制。
    """
    os.makedirs(output_subdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mat_colors = {
        'Cu_step': '#d62728',  # 红色
        'Fe_step': '#1f77b4',  # 蓝色
        'al_left': '#2ca02c'   # 绿色
    }

    # 确定横轴范围
    all_x = np.concatenate([x_coords_dict[mat] for mat in x_coords_dict])
    x_min, x_max = all_x.min(), all_x.max()
    pad_x = (x_max - x_min) * 0.1
    x_lims = (x_min - pad_x, x_max + pad_x)

    # 180kV 与 160kV 绘图配置
    configs = [
        (samples_dict_180, I0_raw_180, '-', 'o', '180kV'),
        (samples_dict_160, I0_raw_160, '--', 's', '160kV')
    ]

    for samples_dict, I0_raw, ls, marker, label_suffix in configs:
        for mat_name, (L_list, H_list) in samples_dict.items():
            color = mat_colors[mat_name]
            cur_x_vals = np.array(x_coords_dict[mat_name][:len(L_list)])

            # 1. 归一化低能通道
            step_L_means, step_L_stds = [], []
            step_L_log_means, step_L_log_stds = [], []
            for l in L_list:
                v_max = 65535 if l.dtype == np.uint16 or np.max(l) > 255 else 255
                lower_th = utils_II.get_ore_lower_threshold(False, v_max)
                v_idx = (l >= lower_th) & (l < v_max)
                if np.any(v_idx):
                    lv = l[v_idx].astype(np.float32)
                    step_L_means.append(np.mean(lv))
                    step_L_stds.append(np.std(lv))
                    
                    logs = np.log(I0_norm / np.maximum(lv, 1.0))
                    step_L_log_means.append(np.mean(logs))
                    step_L_log_stds.append(np.std(logs))
                else:
                    step_L_means.append(np.nan); step_L_stds.append(np.nan)
                    step_L_log_means.append(np.nan); step_L_log_stds.append(np.nan)

            step_L_means = np.array(step_L_means)
            step_L_log_means = np.array(step_L_log_means)

            # 1.1 灰度图
            axes[0, 0].plot(cur_x_vals, step_L_means, ls=ls, marker=marker, color=color, markersize=4, 
                            label=f"{mat_name} ({label_suffix})", linewidth=2.2)
            axes[0, 0].errorbar(cur_x_vals, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=0.3, color=color)

            # 1.2 对数拟合图
            axes[0, 1].plot(cur_x_vals, step_L_log_means, marker, color=color, alpha=0.4)
            axes[0, 1].errorbar(cur_x_vals, step_L_log_means, yerr=step_L_log_stds, fmt='none', capsize=3, alpha=0.2, color=color)
            valid_idx_l = np.isfinite(step_L_log_means) & ~np.isnan(step_L_log_means)
            if np.any(valid_idx_l):
                xs = cur_x_vals[valid_idx_l]
                ys = step_L_log_means[valid_idx_l]
                n_lin = find_linear_pts(xs, ys)
                if n_lin > 1:
                    coeffs = np.polyfit(xs[:n_lin], ys[:n_lin], 1)
                    fit_fn = np.poly1d(coeffs)
                    y_fit = fit_fn(xs[:n_lin])
                    ss_res = np.sum((ys[:n_lin] - y_fit)**2)
                    ss_tot = np.sum((ys[:n_lin] - np.mean(ys[:n_lin]))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                    axes[0, 1].plot(xs[:n_lin], y_fit, ls=ls, color=color, label=f"{mat_name} ({label_suffix}) Fit (n={n_lin}, R²={r2:.4f})", linewidth=2.2)
                else:
                    axes[0, 1].plot(xs, ys, ls=ls, color=color, label=f"{mat_name} ({label_suffix})", linewidth=2.2)

            # 2. 原始高能通道
            step_H_means, step_H_stds = [], []
            step_H_log_means, step_H_log_stds = [], []
            for h in H_list:
                v_max = 65535 if h.dtype == np.uint16 or np.max(h) > 255 else 255
                lower_th = utils_II.get_ore_lower_threshold(False, v_max)
                v_idx = (h >= lower_th) & (h < v_max)
                if np.any(v_idx):
                    hv = h[v_idx].astype(np.float32)
                    step_H_means.append(np.mean(hv))
                    step_H_stds.append(np.std(hv))
                    
                    logs = np.log(I0_raw / np.maximum(hv, 1.0))
                    step_H_log_means.append(np.mean(logs))
                    step_H_log_stds.append(np.std(logs))
                else:
                    step_H_means.append(np.nan); step_H_stds.append(np.nan)
                    step_H_log_means.append(np.nan); step_H_log_stds.append(np.nan)

            step_H_means = np.array(step_H_means)
            step_H_log_means = np.array(step_H_log_means)

            # 2.1 灰度图
            axes[1, 0].plot(cur_x_vals, step_H_means, ls=ls, marker=marker, color=color, markersize=4, 
                            label=f"{mat_name} ({label_suffix})", linewidth=2.2)
            axes[1, 0].errorbar(cur_x_vals, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=0.3, color=color)

            # 2.2 对数拟合图
            axes[1, 1].plot(cur_x_vals, step_H_log_means, marker, color=color, alpha=0.4)
            axes[1, 1].errorbar(cur_x_vals, step_H_log_means, yerr=step_H_log_stds, fmt='none', capsize=3, alpha=0.2, color=color)
            valid_idx_h = np.isfinite(step_H_log_means) & ~np.isnan(step_H_log_means)
            if np.any(valid_idx_h):
                xs = cur_x_vals[valid_idx_h]
                ys = step_H_log_means[valid_idx_h]
                n_lin = find_linear_pts(xs, ys)
                if n_lin > 1:
                    coeffs = np.polyfit(xs[:n_lin], ys[:n_lin], 1)
                    fit_fn = np.poly1d(coeffs)
                    y_fit = fit_fn(xs[:n_lin])
                    ss_res = np.sum((ys[:n_lin] - y_fit)**2)
                    ss_tot = np.sum((ys[:n_lin] - np.mean(ys[:n_lin]))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                    axes[1, 1].plot(xs[:n_lin], y_fit, ls=ls, color=color, label=f"{mat_name} ({label_suffix}) Fit (n={n_lin}, R²={r2:.4f})", linewidth=2.2)
                else:
                    axes[1, 1].plot(xs, ys, ls=ls, color=color, label=f"{mat_name} ({label_suffix})", linewidth=2.2)

    # 配置子图样式和限值
    axes[0, 0].set_xlim(x_lims); axes[0, 0].set_ylim(raw_lims_norm)
    axes[0, 0].set_title("Normalized Grayscale Intensity vs Thickness")
    axes[0, 0].set_xlabel("Thickness (mm)"); axes[0, 0].set_ylabel("Normalized Grayscale Value")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend(fontsize='x-small', handlelength=3.5)

    axes[0, 1].set_xlim(x_lims); axes[0, 1].set_ylim(log_lims_norm)
    axes[0, 1].set_title(r"Normalized Log Attenuation $\ln(I_{0,\mathrm{norm}}/I_{\mathrm{norm}})$ vs Thickness")
    axes[0, 1].set_xlabel("Thickness (mm)"); axes[0, 1].set_ylabel("Attenuation")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend(fontsize='x-small', handlelength=3.5)

    axes[1, 0].set_xlim(x_lims); axes[1, 0].set_ylim(raw_lims_raw)
    axes[1, 0].set_title("Raw Grayscale Intensity vs Thickness")
    axes[1, 0].set_xlabel("Thickness (mm)"); axes[1, 0].set_ylabel("Raw Grayscale Value")
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend(fontsize='x-small', handlelength=3.5)

    axes[1, 1].set_xlim(x_lims); axes[1, 1].set_ylim(log_lims_raw)
    axes[1, 1].set_title(r"Raw Log Attenuation $\ln(I_{0,\mathrm{raw}}/I_{\mathrm{raw}})$ vs Thickness")
    axes[1, 1].set_xlabel("Thickness (mm)"); axes[1, 1].set_ylabel("Attenuation")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend(fontsize='x-small', handlelength=3.5)

    plt.suptitle(f"180kV vs 160kV Comparison - Energy Band {energy_band_num} keV\n(I0_norm={I0_norm}, I0_raw_180={I0_raw_180:.1f}, I0_raw_160={I0_raw_160:.1f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, f"compare_E_{energy_band_num}keV.png"))
    plt.close()

def plot_combined_voltage_slope_summaries(
    global_raw_data_norm_180: dict,
    global_raw_data_norm_160: dict,
    global_raw_data_raw_180: dict,
    global_raw_data_raw_160: dict,
    thicknesses: dict,
    mu_mode: str,
    mat_physics: dict,
    step_mats: dict,
    output_dir: str
) -> None:
    """
    绘制随能量变化的 180kV 与 160kV 的衰减系数及比值对比，生成 2x2 折线汇总图并落地保存。
    在每个子图中，180kV 数据以实线圆形画出，160kV 数据以虚线方形画出，实现直观跨电压衰减斜率对比。
    
    参数类型、含义及用法：
    ------------------
    - global_raw_data_norm_180 (dict): 180kV 归一化后的数据汇总。
    - global_raw_data_norm_160 (dict): 160kV 归一化后的数据汇总。
    - global_raw_data_raw_180 (dict): 180kV 原始高能数据汇总。
    - global_raw_data_raw_160 (dict): 160kV 原始高能数据汇总。
    - thicknesses (dict): 各材质的厚度映射。
    - mu_mode (str): 衰减模式（'mu' 或者是 'mu_m'）。
    - mat_physics (dict): 材质的物理参数。
    - step_mats (dict): 材质名称与文件标识对应映射。
    - output_dir (str): 图表保存的目标根目录。
    """
    mat_colors = {
        'Cu_step': '#d62728',
        'Fe_step': '#1f77b4',
        'al_left': '#2ca02c'
    }
    
    pair_colors = {
        ('Cu_step', 'Fe_step'): '#9467bd',
        ('Cu_step', 'al_left'): '#8c564b',
        ('Fe_step', 'al_left'): '#e377c2'
    }
    
    def get_pair_color(m1, m2):
        if (m1, m2) in pair_colors: return pair_colors[(m1, m2)]
        if (m2, m1) in pair_colors: return pair_colors[(m2, m1)]
        return '#7f7f7f'

    max_steps = max(len(t) for t in thicknesses.values())
    mats = list(step_mats.keys())

    # 1. 第一步：确定统一的跨步进全局 Y 轴上下限
    global_mu_norm_vals = []
    global_ratio_norm_vals = []
    global_mu_raw_vals = []
    global_ratio_raw_vals = []

    configs_for_limits = [
        (global_raw_data_norm_180, global_mu_norm_vals, global_ratio_norm_vals),
        (global_raw_data_norm_160, global_mu_norm_vals, global_ratio_norm_vals),
        (global_raw_data_raw_180, global_mu_raw_vals, global_ratio_raw_vals),
        (global_raw_data_raw_160, global_mu_raw_vals, global_ratio_raw_vals)
    ]

    for step_idx in range(max_steps):
        for global_raw_data, mu_list, ratio_list in configs_for_limits:
            step_storage = {mat: {'mu': []} for mat in mats}
            for mat in mats:
                energies = global_raw_data[mat]['energy']
                sort_idx = np.argsort(energies)
                for i in sort_idx:
                    t_mm = thicknesses[mat][step_idx]
                    log_l = global_raw_data[mat]['log_l'][i][step_idx]
                    vl = global_raw_data[mat]['vl'][i][step_idx]
                    if t_mm > 0:
                        if mu_mode == 'mu_m':
                            rho = mat_physics[mat]['rho']
                            mu = 10.0 * log_l / (t_mm * rho) if vl else np.nan
                        else:
                            mu = log_l / t_mm if vl else np.nan
                    else:
                        mu = np.nan
                    step_storage[mat]['mu'].append(mu)
                    if np.isfinite(mu):
                        mu_list.append(mu)
            
            num_pts = len(step_storage[mats[0]]['mu'])
            for k in range(num_pts):
                for i in range(len(mats)):
                    for j in range(i+1, len(mats)):
                        m1, m2 = mats[i], mats[j]
                        mu1 = step_storage[m1]['mu'][k]
                        mu2 = step_storage[m2]['mu'][k]
                        if np.isfinite(mu1) and np.isfinite(mu2) and mu2 != 0:
                            ratio_list.append(mu1 / mu2)

    def get_dynamic_ylim(vals, default=(0, 1.0), pad_ratio=0.15):
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0: return default
        v_min, v_max = np.percentile(arr, 1), np.percentile(arr, 99)
        pad = (v_max - v_min) * pad_ratio if v_max > v_min else 0.1
        return (0.0, float(v_max + pad))

    global_mu_ylim_norm = get_dynamic_ylim(global_mu_norm_vals)
    global_ratio_ylim_norm = get_dynamic_ylim(global_ratio_norm_vals, default=(0, 10.0))
    global_mu_ylim_raw = get_dynamic_ylim(global_mu_raw_vals)
    global_ratio_ylim_raw = get_dynamic_ylim(global_ratio_raw_vals, default=(0, 10.0))

    # 2. 第二步：循环绘制每一档厚度的大图
    for step_idx in range(max_steps):
        cu_th = thicknesses['Cu_step'][step_idx]
        fe_th = thicknesses['Fe_step'][step_idx]
        al_th = thicknesses['al_left'][step_idx]
        step_name = f"{cu_th}mm_Cu_{fe_th}mm_Fe_{al_th}mm_Al"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        mu_symbol = r"\mu_m" if mu_mode == 'mu_m' else r"\mu"
        mu_desc = "Mass Attenuation" if mu_mode == 'mu_m' else "Linear Attenuation"
        fig.suptitle(f"180kV vs 160kV {mu_desc} (${mu_symbol}$) vs Energy - Step: {step_name}", fontsize=16)

        plot_configs = [
            (global_raw_data_norm_180, global_raw_data_norm_160, 'norm', global_mu_ylim_norm, global_ratio_ylim_norm),
            (global_raw_data_raw_180, global_raw_data_raw_160, 'raw', global_mu_ylim_raw, global_ratio_ylim_raw)
        ]

        for row_idx, (raw_180, raw_160, data_type, cur_mu_ylim, cur_inter_ratio_ylim) in enumerate(plot_configs):
            sub_configs = [
                (raw_180, '-', 'o', '180kV'),
                (raw_160, '--', 's', '160kV')
            ]
            
            for raw_data, ls, marker, label_suffix in sub_configs:
                step_storage = {mat: {'energy': [], 'mu': []} for mat in mats}
                for mat in mats:
                    energies = raw_data[mat]['energy']
                    sort_idx = np.argsort(energies)
                    for i in sort_idx:
                        e_val = energies[i]
                        t_mm = thicknesses[mat][step_idx]
                        log_l = raw_data[mat]['log_l'][i][step_idx]
                        vl = raw_data[mat]['vl'][i][step_idx]
                        
                        if t_mm > 0:
                            if mu_mode == 'mu_m':
                                rho = mat_physics[mat]['rho']
                                mu = 10.0 * log_l / (t_mm * rho) if vl else np.nan
                            else:
                                mu = log_l / t_mm if vl else np.nan
                        else:
                            mu = np.nan
                        step_storage[mat]['energy'].append(e_val)
                        step_storage[mat]['mu'].append(mu)
                
                # 2.1 绘制衰减系数 vs 能量
                for mat in mats:
                    energy_arr = np.array(step_storage[mat]['energy'])
                    mu_arr = np.array(step_storage[mat]['mu'])
                    color = mat_colors[mat]
                    axes[row_idx, 0].plot(energy_arr, mu_arr, color=color, ls=ls, marker=marker, 
                                          label=f"{mat} ({label_suffix})", linewidth=2.2, markersize=4)
                
                # 2.2 绘制材质比值 vs 能量
                for i in range(len(mats)):
                    for j in range(i+1, len(mats)):
                        m1, m2 = mats[i], mats[j]
                        energy_arr = np.array(step_storage[m1]['energy'])
                        r_vals = np.array(step_storage[m1]['mu']) / np.maximum(np.array(step_storage[m2]['mu']), 1e-9)
                        color = get_pair_color(m1, m2)
                        axes[row_idx, 1].plot(energy_arr, r_vals, color=color, ls=ls, marker=marker, 
                                              label=f"{m1}/{m2} ({label_suffix})", linewidth=2.2, markersize=4)

            # 2.3 绘制理论极限参考线
            for i in range(len(mats)):
                for j in range(i+1, len(mats)):
                    m1, m2 = mats[i], mats[j]
                    color = get_pair_color(m1, m2)
                    if m1 in mat_physics and m2 in mat_physics:
                        p1, p2 = mat_physics[m1], mat_physics[m2]
                        if mu_mode == 'mu_m':
                            theo_ph = ( (p1['Z']**4.5) / p1['Ar'] ) / ( (p2['Z']**4.5) / p2['Ar'] )
                            theo_c = ( p1['Z'] / p1['Ar'] ) / ( p2['Z'] / p2['Ar'] )
                        else:
                            theo_ph = ( (p1['Z']**4.5) / p1['Ar'] * p1['rho'] ) / ( (p2['Z']**4.5) / p2['Ar'] * p2['rho'] )
                            theo_c = ( p1['Z'] / p1['Ar'] * p1['rho'] ) / ( p2['Z'] / p2['Ar'] * p2['rho'] )
                        
                        axes[row_idx, 1].axhline(y=theo_ph, color=color, linestyle=':', alpha=0.5)
                        axes[row_idx, 1].axhline(y=theo_c, color=color, linestyle='-.', alpha=0.3)

            mu_unit = r"$\mathrm{cm}^2/\mathrm{g}$" if mu_mode == 'mu_m' else r"$\mathrm{mm}^{-1}$"
            axes[row_idx, 0].set_title(fr"${mu_symbol}$ vs Energy ({'Normalized' if data_type == 'norm' else 'Raw'})")
            axes[row_idx, 0].set_xlabel("Energy Channel (keV)"); axes[row_idx, 0].set_ylabel(fr"${mu_symbol}\ ({mu_unit})$")
            axes[row_idx, 0].set_ylim(cur_mu_ylim)
            axes[row_idx, 0].set_xlim(15, 135)
            axes[row_idx, 0].grid(True, alpha=0.3)
            axes[row_idx, 0].legend(fontsize='x-small', ncol=2, handlelength=3.5)

            axes[row_idx, 1].set_title(fr"Inter-Material Attenuation Ratio ({'Normalized' if data_type == 'norm' else 'Raw'})")
            axes[row_idx, 1].set_xlabel("Energy Channel (keV)"); axes[row_idx, 1].set_ylabel("Ratio")
            axes[row_idx, 1].set_ylim(cur_inter_ratio_ylim)
            axes[row_idx, 1].set_xlim(15, 135)
            axes[row_idx, 1].grid(True, alpha=0.3)
            axes[row_idx, 1].legend(fontsize='xx-small', ncol=2, handlelength=3.5)

        plt.tight_layout()
        combined_save_dir = os.path.join(output_dir, "combined", f"slope_summary_{mu_mode}")
        os.makedirs(combined_save_dir, exist_ok=True)
        plt.savefig(os.path.join(combined_save_dir, f"compare_slope_summary_{step_name}.png"))
        plt.close()

def plot_combined_voltage_mu_vs_thickness(
    global_raw_data_norm_180: dict,
    global_raw_data_norm_160: dict,
    global_raw_data_raw_180: dict,
    global_raw_data_raw_160: dict,
    thicknesses: dict,
    mu_mode: str,
    mat_physics: dict,
    step_mats: dict,
    output_dir: str
) -> None:
    """
    绘制随厚度变化的三材质衰减系数，生成 3x2 汇总大图，并在同一子图中对比 180kV 与 160kV 的衰减情况。
    实线圆形代表 180kV 数据，虚线方形代表 160kV 数据，不同颜色曲线代表不同能量段。
    
    参数类型、含义及用法：
    ------------------
    - global_raw_data_norm_180 (dict): 180kV 归一化后的数据汇总。
    - global_raw_data_norm_160 (dict): 160kV 归一化后的数据汇总。
    - global_raw_data_raw_180 (dict): 180kV 原始高能数据汇总。
    - global_raw_data_raw_160 (dict): 160kV 原始高能数据汇总。
    - thicknesses (dict): 各材质的厚度映射。
    - mu_mode (str): 衰减模式（'mu' 或者是 'mu_m'）。
    - mat_physics (dict): 材质的物理参数。
    - step_mats (dict): 材质名称与文件标识对应映射。
    - output_dir (str): 图表保存的目标根目录。
    """
    mats = list(step_mats.keys())
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    mu_symbol = r"\mu_m" if mu_mode == 'mu_m' else r"\mu"
    mu_desc = "Mass Attenuation" if mu_mode == 'mu_m' else "Linear Attenuation"
    fig.suptitle(f"180kV vs 160kV {mu_desc} (${mu_symbol}$) vs Thickness under Different Energy Bands", fontsize=16)
    
    energies_sorted = sorted(list(set(global_raw_data_norm_180[mats[0]]['energy'])))
    num_energies = len(energies_sorted)
    cmap = plt.get_cmap('jet')
    energy_colors = [cmap(i) for i in np.linspace(0, 0.9, num_energies)]

    for row_idx, mat in enumerate(mats):
        t_arr = thicknesses[mat]
        for col_idx, (data_180, data_160, data_type) in enumerate([
            (global_raw_data_norm_180, global_raw_data_norm_160, 'norm'),
            (global_raw_data_raw_180, global_raw_data_raw_160, 'raw')
        ]):
            ax = axes[row_idx, col_idx]
            
            for e_i, e_val in enumerate(energies_sorted):
                # 1. 180kV 曲线
                if e_val in data_180[mat]['energy']:
                    idx = data_180[mat]['energy'].index(e_val)
                    log_l = data_180[mat]['log_l'][idx]
                    vl = data_180[mat]['vl'][idx]
                    
                    mu_vals = []
                    for s_idx, t_mm in enumerate(t_arr):
                        curr_log = log_l[s_idx]
                        curr_vl = vl[s_idx]
                        if t_mm > 0 and curr_vl:
                            if mu_mode == 'mu_m':
                                rho = mat_physics[mat]['rho']
                                mu = 10.0 * curr_log / (t_mm * rho)
                            else:
                                mu = curr_log / t_mm
                        else:
                            mu = np.nan
                        mu_vals.append(mu)
                    ax.plot(t_arr, mu_vals, ls='-', marker='o', color=energy_colors[e_i], linewidth=2.2, markersize=4,
                            label=f"{e_val} keV (180kV)" if row_idx == 0 and col_idx == 0 else "")
                
                # 2. 160kV 曲线
                if e_val in data_160[mat]['energy']:
                    idx = data_160[mat]['energy'].index(e_val)
                    log_l = data_160[mat]['log_l'][idx]
                    vl = data_160[mat]['vl'][idx]
                    
                    mu_vals = []
                    for s_idx, t_mm in enumerate(t_arr):
                        curr_log = log_l[s_idx]
                        curr_vl = vl[s_idx]
                        if t_mm > 0 and curr_vl:
                            if mu_mode == 'mu_m':
                                rho = mat_physics[mat]['rho']
                                mu = 10.0 * curr_log / (t_mm * rho)
                            else:
                                mu = curr_log / t_mm
                        else:
                            mu = np.nan
                        mu_vals.append(mu)
                    ax.plot(t_arr, mu_vals, ls='--', marker='s', color=energy_colors[e_i], linewidth=2.2, markersize=4,
                            label=f"{e_val} keV (160kV)" if row_idx == 0 and col_idx == 0 else "")
            
            mu_unit = r"$\mathrm{cm}^2/\mathrm{g}$" if mu_mode == 'mu_m' else r"$\mathrm{mm}^{-1}$"
            ax.set_title(f"{mat} - {'Normalized' if data_type == 'norm' else 'Raw'} Data")
            ax.set_xlabel("Thickness (mm)")
            ax.set_ylabel(fr"${mu_symbol}\ ({mu_unit})$")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)

    axes[0, 0].legend(title="Energy Channels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2, handlelength=3.5)
    plt.tight_layout()
    combined_save_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_save_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_save_dir, f"compare_mu_vs_thickness_Cu_Fe_Al_{mu_mode}.png"), bbox_inches='tight')
    plt.close()

def main() -> None:
    """
    主控制流程。配置 180kV 与 160kV 结果所在的目录路径，扫描它们共有的能量通道段。
    批量对每个共有能段加载数据包，绘制 steps 图像（实线与虚线对比）；
    随后跨能段汇总，输出 combined 维度下的 slope_summary 大图与 mu vs thickness 大图。
    
    参数：
    - 无参数。
    """
    contour_results_dir_180 = r"E:\photo_electric_II\results\20260512_180kV_1mA_subtracting_noise\contour_results"
    contour_results_dir_160 = r"E:\photo_electric_II\results\20260512_160kV_1mA_subtracting_noise\contour_results"
    output_dir = r"E:\photo_electric_II\results\thickness_decoupling\H_L_fit\compare_180kV_160kV"
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. 扫描两组数据的能量段目录
    subdirs_180 = [d for d in os.listdir(contour_results_dir_180)
                   if os.path.isdir(os.path.join(contour_results_dir_180, d)) and d.endswith('_noNorm_R')]
    subdirs_160 = [d for d in os.listdir(contour_results_dir_160)
                   if os.path.isdir(os.path.join(contour_results_dir_160, d)) and d.endswith('_noNorm_R')]

    # 提取实际对应能量数值（keV）
    def extract_energy(name):
        match_e = re.search(r'_E_(\d+)', name)
        if match_e:
            return int(match_e.group(1))
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 9999

    energy_to_subdir_180 = {extract_energy(d): d for d in subdirs_180}
    energy_to_subdir_160 = {extract_energy(d): d for d in subdirs_160}

    # 寻找共有的能量通道以实现严格的横向对比
    common_energies = sorted(list(set(energy_to_subdir_180.keys()) & set(energy_to_subdir_160.keys())))
    print(f"Common energy bands found: {common_energies}")

    step_mats = {
        'Cu_step': 'step_sample_1',
        'Fe_step': 'step_sample_2',
        'al_left': 'step_sample_3_left'
    }
    
    thicknesses = {
        'Cu_step': np.arange(2, 22, 2),
        'Fe_step': np.arange(2, 22, 2),
        'al_left': np.arange(12, 32, 2)
    }

    mat_physics = {
        'Cu_step': {'Z': 29, 'Ar': 63.546, 'rho': 8.96},
        'Fe_step': {'Z': 26, 'Ar': 55.845, 'rho': 7.87},
        'al_left': {'Z': 13, 'Ar': 26.982, 'rho': 2.70}
    }

    I0_norm = 52428.0

    # 初始化两套数据的全局收集字典
    global_raw_data_norm_180 = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}
    global_raw_data_norm_160 = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}
    global_raw_data_raw_180 = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}
    global_raw_data_raw_160 = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}

    norm_raw_pts, norm_log_pts = [], []
    raw_raw_pts, raw_log_pts = [], []

    # 存储两套数据各个能量段的动态背景参考值
    I0_raw_dict_180 = {}
    I0_raw_dict_160 = {}

    print(">>> Scanning both datasets for limits and dynamic background values...")
    for e_val in common_energies:
        band_180 = energy_to_subdir_180[e_val]
        band_160 = energy_to_subdir_160[e_val]

        # 180kV 动态背景
        max_high_val_180 = 1.0
        for mat, file_label in step_mats.items():
            p = os.path.join(contour_results_dir_180, band_180, "pixel_values", f"{band_180}_{file_label}.pkl")
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                for h in d['pixels_high']:
                    if len(h) > 0:
                        max_high_val_180 = max(max_high_val_180, np.max(h))
        I0_raw_dict_180[e_val] = 1.15 * max_high_val_180

        # 160kV 动态背景
        max_high_val_160 = 1.0
        for mat, file_label in step_mats.items():
            p = os.path.join(contour_results_dir_160, band_160, "pixel_values", f"{band_160}_{file_label}.pkl")
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                for h in d['pixels_high']:
                    if len(h) > 0:
                        max_high_val_160 = max(max_high_val_160, np.max(h))
        I0_raw_dict_160[e_val] = 1.15 * max_high_val_160

        # 分别载入数据计算并汇总
        datasets_configs = [
            (global_raw_data_norm_180, global_raw_data_raw_180, contour_results_dir_180, band_180, I0_raw_dict_180[e_val]),
            (global_raw_data_norm_160, global_raw_data_raw_160, contour_results_dir_160, band_160, I0_raw_dict_160[e_val])
        ]

        for raw_data_norm, raw_data_raw, results_dir, band_dir, I0_raw_band in datasets_configs:
            for mat, file_label in step_mats.items():
                p = os.path.join(results_dir, band_dir, "pixel_values", f"{band_dir}_{file_label}.pkl")
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        d = pickle.load(f)
                    l_list = d['pixels_low']
                    h_list = d['pixels_high']
                    
                    v_max_l = 65535 if l_list[0].dtype == np.uint16 or np.max(l_list[0]) > 255 else 255
                    v_max_h = 65535 if h_list[0].dtype == np.uint16 or np.max(h_list[0]) > 255 else 255

                    # Low Energy
                    log_l_norm = np.array([np.mean(np.log(I0_norm / np.maximum(px[(px>=0)&(px<v_max_l)], 1.0))) if np.any((px>=0)&(px<v_max_l)) else np.nan for px in l_list])
                    m_l_norm = np.array([np.mean(px[(px>=0)&(px<v_max_l)]) if np.any((px>=0)&(px<v_max_l)) else np.nan for px in l_list])
                    vl_norm = np.isfinite(log_l_norm)
                    
                    raw_data_norm[mat]['energy'].append(e_val)
                    raw_data_norm[mat]['log_l'].append(log_l_norm)
                    raw_data_norm[mat]['vl'].append(vl_norm)
                    
                    norm_raw_pts.extend(m_l_norm[np.isfinite(m_l_norm)])
                    norm_log_pts.extend(log_l_norm[np.isfinite(log_l_norm)])

                    # High Energy
                    log_l_raw = np.array([np.mean(np.log(I0_raw_band / np.maximum(px[(px>=0)&(px<v_max_h)], 1.0))) if np.any((px>=0)&(px<v_max_h)) else np.nan for px in h_list])
                    m_l_raw = np.array([np.mean(px[(px>=0)&(px<v_max_h)]) if np.any((px>=0)&(px<v_max_h)) else np.nan for px in h_list])
                    vl_raw = np.isfinite(log_l_raw)

                    raw_data_raw[mat]['energy'].append(e_val)
                    raw_data_raw[mat]['log_l'].append(log_l_raw)
                    raw_data_raw[mat]['vl'].append(vl_raw)
                    
                    raw_raw_pts.extend(m_l_raw[np.isfinite(m_l_raw)])
                    raw_log_pts.extend(log_l_raw[np.isfinite(log_l_raw)])

    # 确定全局一致的坐标区间范围
    raw_ylim_norm = get_clean_ylim(norm_raw_pts, default_max=65535.0)
    log_ylim_norm = get_clean_ylim(norm_log_pts, default_max=5.0)
    raw_ylim_raw = get_clean_ylim(raw_raw_pts, default_max=2048.0)
    log_ylim_raw = get_clean_ylim(raw_log_pts, default_max=5.0)

    # 绘制 steps 对比图
    print("\n>>> Drawing 180kV vs 160kV steps comparison plots...")
    for e_val in common_energies:
        band_180 = energy_to_subdir_180[e_val]
        band_160 = energy_to_subdir_160[e_val]
        
        samples_data_180 = {}
        samples_data_160 = {}

        for mat, file_label in step_mats.items():
            p_180 = os.path.join(contour_results_dir_180, band_180, "pixel_values", f"{band_180}_{file_label}.pkl")
            if os.path.exists(p_180):
                with open(p_180, 'rb') as f:
                    samples_data_180[mat] = pickle.load(f)
            
            p_160 = os.path.join(contour_results_dir_160, band_160, "pixel_values", f"{band_160}_{file_label}.pkl")
            if os.path.exists(p_160):
                with open(p_160, 'rb') as f:
                    samples_data_160[mat] = pickle.load(f)

        if samples_data_180 and samples_data_160:
            perform_combined_voltage_analysis(
                energy_band_num=e_val,
                samples_dict_180={mat: (samples_data_180[mat]['pixels_low'], samples_data_180[mat]['pixels_high']) for mat in samples_data_180},
                samples_dict_160={mat: (samples_data_160[mat]['pixels_low'], samples_data_160[mat]['pixels_high']) for mat in samples_data_160},
                output_subdir=os.path.join(output_dir, "steps"),
                x_coords_dict=thicknesses,
                I0_norm=I0_norm,
                I0_raw_180=I0_raw_dict_180[e_val],
                I0_raw_160=I0_raw_dict_160[e_val],
                raw_lims_norm=raw_ylim_norm,
                log_lims_norm=log_ylim_norm,
                raw_lims_raw=raw_ylim_raw,
                log_lims_raw=log_ylim_raw
            )

    # 绘制 combined 对比图
    for mu_mode in ['mu', 'mu_m']:
        print(f"\n>>> Drawing combined slope summaries for mu_mode: {mu_mode}...")
        plot_combined_voltage_slope_summaries(
            global_raw_data_norm_180=global_raw_data_norm_180,
            global_raw_data_norm_160=global_raw_data_norm_160,
            global_raw_data_raw_180=global_raw_data_raw_180,
            global_raw_data_raw_160=global_raw_data_raw_160,
            thicknesses=thicknesses,
            mu_mode=mu_mode,
            mat_physics=mat_physics,
            step_mats=step_mats,
            output_dir=output_dir
        )
        
        print(f">>> Drawing combined mu vs thickness plots for mu_mode: {mu_mode}...")
        plot_combined_voltage_mu_vs_thickness(
            global_raw_data_norm_180=global_raw_data_norm_180,
            global_raw_data_norm_160=global_raw_data_norm_160,
            global_raw_data_raw_180=global_raw_data_raw_180,
            global_raw_data_raw_160=global_raw_data_raw_160,
            thicknesses=thicknesses,
            mu_mode=mu_mode,
            mat_physics=mat_physics,
            step_mats=step_mats,
            output_dir=output_dir
        )

    print(f"\nAll 180kV vs 160kV comparison analyses completed successfully!")
    print(f"Check results under: {output_dir}")

if __name__ == '__main__':
    main()
