import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import json
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
    
    用法：
    -----
    - 用于灰度值和对数衰减值的 Y 轴范围确定，确保不同能量段间范围一致，且下限不为负数。
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

def perform_dual_channel_analysis(
    energy_band: str,
    samples_dict: dict,
    output_subdir: str,
    title_prefix: str,
    x_label: str,
    x_coords_dict: dict,
    I0_norm: float = 52428.0,
    I0_raw: float = 1500.0,
    raw_lims_norm: tuple = None,
    log_lims_norm: tuple = None,
    raw_lims_raw: tuple = None,
    log_lims_raw: tuple = None
) -> None:
    """
    对双通道（归一化后的低能数据 + 归一化前的原始高能数据）的多材质阶梯样品进行 2x2 衰减特性综合分析和绘图。
    
    参数类型、含义及用法：
    ------------------
    参数：
    - energy_band (str): 当前分析的能量段名称（例如 '20_dual'）。
    - samples_dict (dict): 样品数据字典，格式为 {mat_name: (L_list, H_list)}，
      其中 L_list 和 H_list 为包含 10 个台阶的像素数组列表（分别对应归一化数据和原始数据）。
    - output_subdir (str): 结果图像文件保存的目标子目录。
    - title_prefix (str): 图像标题的前缀，例如 'Step Sample'。
    - x_label (str): X轴标签名，例如 'Thickness (mm)'。
    - x_coords_dict (dict): 每个材质对应的 X 轴坐标数组映射表，格式为 {mat_name: ndarray}。
    - I0_norm (float): 归一化低能数据的背景参考值，默认为 52428.0.
    - I0_raw (float): 原始高能数据的背景参考值（由外部动态计算传入）。
    - raw_lims_norm (tuple): 归一化灰度值的 Y 轴显示区间限制。
    - log_lims_norm (tuple): 归一化对数衰减值的 Y 轴显示区间限制。
    - raw_lims_raw (tuple): 原始高能灰度值的 Y 轴显示区间限制。
    - log_lims_raw (tuple): 原始高能对数衰减值的 Y 轴显示区间限制。
    
    用法：
    -----
    - 在 main 函数循环遍历各能量段时调用，分别生成并保存每个能量段的 2x2 分析图。
    """
    os.makedirs(output_subdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mat_colors = {
        'Cu_step': '#d62728',  # Crimson Red
        'Fe_step': '#1f77b4',  # Slate Blue
        'al_left': '#2ca02c'   # Emerald Green
    }

    # 确定 X 轴网格范围
    all_x_vals = [x_coords_dict[mat][:len(samples_dict[mat][0])] for mat in samples_dict]
    X_glob = np.concatenate(all_x_vals)
    x_min, x_max = X_glob.min(), X_glob.max()
    pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    x_lims = (x_min - pad_x, x_max + pad_x)

    for mat_name, (L_list, H_list) in samples_dict.items():
        color = mat_colors.get(mat_name, '#7f7f7f')
        cur_x_vals = np.array(x_coords_dict[mat_name][:len(L_list)])

        # ------------------ 1. 归一化数据 (低能) ------------------
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
        axes[0, 0].plot(cur_x_vals, step_L_means, 'o-', color=color, markersize=5, label=f"{mat_name}", linewidth=1.5)
        axes[0, 0].errorbar(cur_x_vals, step_L_means, yerr=step_L_stds, fmt='none', capsize=3, alpha=0.5, color=color)
        for i in range(max(0, len(cur_x_vals) - 4), len(cur_x_vals)):
            if not np.isnan(step_L_means[i]):
                axes[0, 0].text(cur_x_vals[i], step_L_means[i] + (raw_lims_norm[1]-raw_lims_norm[0])*0.02, 
                                f"{step_L_means[i]:.0f}", fontsize=8, ha='center', va='bottom', color=color)

        # 1.2 对数拟合图
        axes[0, 1].plot(cur_x_vals, step_L_log_means, 'o', color=color, alpha=0.4)
        axes[0, 1].errorbar(cur_x_vals, step_L_log_means, yerr=step_L_log_stds, fmt='none', capsize=3, alpha=0.3, color=color)
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
                axes[0, 1].plot(xs[:n_lin], y_fit, '--', color=color, label=f"{mat_name} Fit (n={n_lin}, R²={r2:.4f})")
            else:
                axes[0, 1].plot(xs, ys, '-', color=color, label=f"{mat_name}")

        # ------------------ 2. 原始数据 (高能) ------------------
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

        # 2.1 原始灰度图
        axes[1, 0].plot(cur_x_vals, step_H_means, 'o-', color=color, markersize=5, label=f"{mat_name}", linewidth=1.5)
        axes[1, 0].errorbar(cur_x_vals, step_H_means, yerr=step_H_stds, fmt='none', capsize=3, alpha=0.5, color=color)
        for i in range(max(0, len(cur_x_vals) - 4), len(cur_x_vals)):
            if not np.isnan(step_H_means[i]):
                axes[1, 0].text(cur_x_vals[i], step_H_means[i] + (raw_lims_raw[1]-raw_lims_raw[0])*0.02, 
                                f"{step_H_means[i]:.0f}", fontsize=8, ha='center', va='bottom', color=color)

        # 2.2 原始对数拟合图
        axes[1, 1].plot(cur_x_vals, step_H_log_means, 'o', color=color, alpha=0.4)
        axes[1, 1].errorbar(cur_x_vals, step_H_log_means, yerr=step_H_log_stds, fmt='none', capsize=3, alpha=0.3, color=color)
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
                axes[1, 1].plot(xs[:n_lin], y_fit, '--', color=color, label=f"{mat_name} Fit (n={n_lin}, R²={r2:.4f})")
            else:
                axes[1, 1].plot(xs, ys, '-', color=color, label=f"{mat_name}")

    # 设置子图坐标限值与基本样式
    # Row 0: Normalized Data
    axes[0, 0].set_xlim(x_lims); axes[0, 0].set_ylim(raw_lims_norm)
    axes[0, 0].set_title("Normalized Grayscale Intensity vs Thickness")
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel("Normalized Grayscale Value")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend(fontsize='small')

    axes[0, 1].set_xlim(x_lims); axes[0, 1].set_ylim(log_lims_norm)
    axes[0, 1].set_title(r"Normalized Log Attenuation $\ln(I_{0,\mathrm{norm}}/I_{\mathrm{norm}})$ vs Thickness")
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel("Attenuation")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend(fontsize='small')

    # Row 1: Raw Data
    axes[1, 0].set_xlim(x_lims); axes[1, 0].set_ylim(raw_lims_raw)
    axes[1, 0].set_title("Raw Grayscale Intensity vs Thickness")
    axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel("Raw Grayscale Value")
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend(fontsize='small')

    axes[1, 1].set_xlim(x_lims); axes[1, 1].set_ylim(log_lims_raw)
    axes[1, 1].set_title(r"Raw Log Attenuation $\ln(I_{0,\mathrm{raw}}/I_{\mathrm{raw}})$ vs Thickness")
    axes[1, 1].set_xlabel(x_label); axes[1, 1].set_ylabel("Attenuation")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend(fontsize='small')

    plt.suptitle(f"{title_prefix} - Energy Band {energy_band}\n(I0_norm={I0_norm}, I0_raw={I0_raw:.1f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, f"{energy_band}_analysis.png"))
    plt.close()

def plot_combined_slope_summaries(
    global_raw_data_norm: dict,
    global_raw_data_raw: dict,
    thicknesses: dict,
    mu_mode: str,
    mat_physics: dict,
    step_mats: dict,
    output_dir: str
) -> None:
    """
    绘制随能量段变化的两套通道数据（第一行为归一化数据，第二行为原始数据）的衰减系数和相互比值，
    生成 2x2 汇总折线大图，不同厚度档位的大图之间共享统一的纵坐标范围，并输出相应的 JSON 归档文件。
    
    参数类型、含义及用法：
    ------------------
    - global_raw_data_norm (dict): 归一化后低能通道的数据汇总，格式为 {mat: {'energy': [], 'log_l': [], 'vl': []}}。
    - global_raw_data_raw (dict): 归一化前高能原始通道的数据汇总，格式为 {mat: {'energy': [], 'log_l': [], 'vl': []}}。
    - thicknesses (dict): 各材质的厚度数组映射表，格式为 {mat_name: ndarray}。
    - mu_mode (str): 衰减系数类型，可选 'mu' (线衰减 mm^-1) 或 'mu_m' (质量衰减 cm^2/g)。
    - mat_physics (dict): 材质物理性质字典，包含 Z、Ar 和 rho 密度等常数。
    - step_mats (dict): 材质名称与提取文件名标识的映射表。
    - output_dir (str): 图表和 JSON 导出的目标根目录。
    
    用法：
    -----
    - 在 main 函数循环结束后被调用，一次性为 'mu' 或 'mu_m' 模式生成 10 档厚度台阶的 2x2 汇总大图 and 对应的归档 JSON。
    """
    mat_colors = {
        'Cu_step': '#d62728',  # Crimson Red
        'Fe_step': '#1f77b4',  # Slate Blue
        'al_left': '#2ca02c'   # Emerald Green
    }
    
    pair_colors = {
        ('Cu_step', 'Fe_step'): '#9467bd',
        ('Cu_step', 'al_left'): '#8c564b',
        ('Fe_step', 'al_left'): '#e377c2'
    }
    
    def get_pair_color(m1, m2):
        if (m1, m2) in pair_colors:
            return pair_colors[(m1, m2)]
        if (m2, m1) in pair_colors:
            return pair_colors[(m2, m1)]
        return '#7f7f7f'

    max_steps = max(len(t) for t in thicknesses.values())
    mats = list(step_mats.keys())
    
    # 第一步：搜集所有台阶的所有数据，以计算不同厚度大图间完全一致的全局纵坐标范围
    global_mu_norm_vals = []
    global_ratio_norm_vals = []
    global_mu_raw_vals = []
    global_ratio_raw_vals = []

    for step_idx in range(max_steps):
        for global_raw_data, mu_list, ratio_list in [
            (global_raw_data_norm, global_mu_norm_vals, global_ratio_norm_vals),
            (global_raw_data_raw, global_mu_raw_vals, global_ratio_raw_vals)
        ]:
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
            
            # 计算比值
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
        # 强制下限为 0
        return (0.0, float(v_max + pad))

    global_mu_ylim_norm = get_dynamic_ylim(global_mu_norm_vals)
    global_ratio_ylim_norm = get_dynamic_ylim(global_ratio_norm_vals, default=(0, 10.0))
    global_mu_ylim_raw = get_dynamic_ylim(global_mu_raw_vals)
    global_ratio_ylim_raw = get_dynamic_ylim(global_ratio_raw_vals, default=(0, 10.0))
    
    # 第二步：循环生成每一档厚度大图
    for step_idx in range(max_steps):
        cu_th = thicknesses['Cu_step'][step_idx]
        fe_th = thicknesses['Fe_step'][step_idx]
        al_th = thicknesses['al_left'][step_idx]
        step_name = f"{cu_th}mm_Cu_{fe_th}mm_Fe_{al_th}mm_Al"
        
        print(f">>> Generating Combined 2x2 Slope Summary Plot for step index {step_idx}: {step_name} ...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        mu_symbol = r"\mu_m" if mu_mode == 'mu_m' else r"\mu"
        mu_desc = "Mass Attenuation" if mu_mode == 'mu_m' else "Linear Attenuation"
        fig.suptitle(f"{mu_desc} (${mu_symbol}$) vs Energy Channel - Step: {step_name}", fontsize=16)
        
        json_data = {'norm': {}, 'raw': {}}
        
        for row_idx, (global_raw_data, data_type, cur_mu_ylim, cur_inter_ratio_ylim) in enumerate([
            (global_raw_data_norm, 'norm', global_mu_ylim_norm, global_ratio_ylim_norm),
            (global_raw_data_raw, 'raw', global_mu_ylim_raw, global_ratio_ylim_raw)
        ]):
            step_storage = {mat: {'energy': [], 'mu': []} for mat in mats}
            
            for mat in mats:
                energies = global_raw_data[mat]['energy']
                sort_idx = np.argsort(energies)
                
                for i in sort_idx:
                    e_val = energies[i]
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
                    
                    step_storage[mat]['energy'].append(e_val)
                    step_storage[mat]['mu'].append(mu)
            
            # 计算比值
            energy_sorted = step_storage[mats[0]]['energy']
            num_pts = len(energy_sorted)
            
            # 1. 绘制左图：衰减系数 vs 能量段
            for mat in mats:
                energy_arr = np.array(step_storage[mat]['energy'])
                mu_arr = np.array(step_storage[mat]['mu'])
                color = mat_colors[mat]
                axes[row_idx, 0].plot(energy_arr, mu_arr, color=color, ls='-', marker='o', label=f"{mat}")
                
            mu_unit = r"$\mathrm{cm}^2/\mathrm{g}$" if mu_mode == 'mu_m' else r"$\mathrm{mm}^{-1}$"
            axes[row_idx, 0].set_title(fr"${mu_symbol}$ vs Energy Channel ({'Normalized' if data_type == 'norm' else 'Raw'})")
            axes[row_idx, 0].set_xlabel("Energy Channel (keV)"); axes[row_idx, 0].set_ylabel(fr"${mu_symbol}\ ({mu_unit})$")
            axes[row_idx, 0].set_ylim(cur_mu_ylim)
            axes[row_idx, 0].set_xlim(15, 135)
            axes[row_idx, 0].grid(True, alpha=0.3)
            axes[row_idx, 0].legend(fontsize='small')
            
            # 2. 绘制右图：比值 vs 能量段
            for i in range(len(mats)):
                for j in range(i+1, len(mats)):
                    m1, m2 = mats[i], mats[j]
                    energy_arr = np.array(step_storage[m1]['energy'])
                    r_vals = np.array(step_storage[m1]['mu']) / np.maximum(np.array(step_storage[m2]['mu']), 1e-9)
                    
                    color = get_pair_color(m1, m2)
                    axes[row_idx, 1].plot(energy_arr, r_vals, color=color, ls='-', marker='o', label=f"{m1}/{m2} Ratio")
                    
                    # 绘制物理常数极限参考线
                    if m1 in mat_physics and m2 in mat_physics:
                        p1, p2 = mat_physics[m1], mat_physics[m2]
                        if mu_mode == 'mu_m':
                            theo_ph = ( (p1['Z']**4.5) / p1['Ar'] ) / ( (p2['Z']**4.5) / p2['Ar'] )
                            theo_c = ( p1['Z'] / p1['Ar'] ) / ( p2['Z'] / p2['Ar'] )
                        else:
                            theo_ph = ( (p1['Z']**4.5) / p1['Ar'] * p1['rho'] ) / ( (p2['Z']**4.5) / p2['Ar'] * p2['rho'] )
                            theo_c = ( p1['Z'] / p1['Ar'] * p1['rho'] ) / ( p2['Z'] / p2['Ar'] * p2['rho'] )
                        
                        axes[row_idx, 1].axhline(y=theo_ph, color=color, linestyle=':', alpha=0.7, label=f"{m1}/{m2} Theo (PH)")
                        axes[row_idx, 1].axhline(y=theo_c, color=color, linestyle='-.', alpha=0.5, label=f"{m1}/{m2} Theo (Compton)")

            axes[row_idx, 1].set_title(fr"Inter-Material Attenuation Ratio ({'Normalized' if data_type == 'norm' else 'Raw'})")
            axes[row_idx, 1].set_xlabel("Energy Channel (keV)"); axes[row_idx, 1].set_ylabel("Ratio")
            axes[row_idx, 1].set_ylim(cur_inter_ratio_ylim)
            axes[row_idx, 1].set_xlim(15, 135)
            axes[row_idx, 1].grid(True, alpha=0.3)
            axes[row_idx, 1].legend(fontsize='x-small')
            
            # 整理 JSON 数据
            for mat in mats:
                json_data[data_type][mat] = {}
                for k in range(num_pts):
                    e_str = f"{energy_sorted[k]}keV"
                    mu_val = step_storage[mat]['mu'][k]
                    json_data[data_type][mat][e_str] = float(mu_val) if np.isfinite(mu_val) else None

        plt.tight_layout()
        combined_save_dir = os.path.join(output_dir, "combined", f"slope_summary_{mu_mode}")
        os.makedirs(combined_save_dir, exist_ok=True)
        plt.savefig(os.path.join(combined_save_dir, f"slope_summary_combined_{step_name}.png"))
        plt.close()

        # 导出对应的 JSON 归档数据
        for data_type in ['norm', 'raw']:
            output_json_path = os.path.join(output_dir, f'attenuation_slopes_{data_type}_{mu_mode}_{step_name}.json')
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data[data_type], f, indent=4, ensure_ascii=False)

def plot_mu_vs_thickness(
    global_raw_data_norm: dict,
    global_raw_data_raw: dict,
    thicknesses: dict,
    mu_mode: str,
    mat_physics: dict,
    step_mats: dict,
    output_dir: str
) -> None:
    """
    绘制随厚度变化的三材质衰减系数，生成 3x2 的大图（每行代表一种材质 Cu/Fe/Al，第一列为归一化数据，第二列为原始数据），
    不同颜色曲线代表不同能量段。
    
    参数类型、含义及用法：
    ------------------
    - global_raw_data_norm (dict): 归一化数据汇总。
    - global_raw_data_raw (dict): 原始数据汇总。
    - thicknesses (dict): 材质厚度数组映射表。
    - mu_mode (str): 衰减系数类型，可选 'mu' (线衰减 mm^-1) 或 'mu_m' (质量衰减 cm^2/g)。
    - mat_physics (dict): 材质物理性质。
    - step_mats (dict): 材质与标识的映射表。
    - output_dir (str): 图表输出目标目录。
    """
    mats = list(step_mats.keys())
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    mu_symbol = r"\mu_m" if mu_mode == 'mu_m' else r"\mu"
    mu_desc = "Mass Attenuation" if mu_mode == 'mu_m' else "Linear Attenuation"
    fig.suptitle(f"{mu_desc} (${mu_symbol}$) vs Thickness under Different Energy Bands", fontsize=16)
    
    energies_sorted = sorted(list(set(global_raw_data_norm[mats[0]]['energy'])))
    num_energies = len(energies_sorted)
    cmap = plt.get_cmap('jet')
    energy_colors = [cmap(i) for i in np.linspace(0, 0.9, num_energies)]

    for row_idx, mat in enumerate(mats):
        t_arr = thicknesses[mat]
        for col_idx, (global_raw_data, data_type) in enumerate([
            (global_raw_data_norm, 'norm'),
            (global_raw_data_raw, 'raw')
        ]):
            ax = axes[row_idx, col_idx]
            
            energies = global_raw_data[mat]['energy']
            for e_i, e_val in enumerate(energies_sorted):
                if e_val in energies:
                    idx = energies.index(e_val)
                    log_l = global_raw_data[mat]['log_l'][idx]
                    vl = global_raw_data[mat]['vl'][idx]
                    
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
                    
                    ax.plot(t_arr, mu_vals, marker='o', color=energy_colors[e_i], 
                            label=f"{e_val} keV" if row_idx == 0 and col_idx == 0 else "")
            
            mu_unit = r"$\mathrm{cm}^2/\mathrm{g}$" if mu_mode == 'mu_m' else r"$\mathrm{mm}^{-1}$"
            ax.set_title(f"{mat} - {'Normalized' if data_type == 'norm' else 'Raw'} Data")
            ax.set_xlabel("Thickness (mm)")
            ax.set_ylabel(fr"${mu_symbol}\ ({mu_unit})$")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)

    axes[0, 0].legend(title="Energy Channels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    combined_save_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_save_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_save_dir, f"mu_vs_thickness_Cu_Fe_Al_{mu_mode}.png"), bbox_inches='tight')
    plt.close()

def main(contour_results_dir, output_dir) -> None:
    """
    主控制流程。扫描光子计数器各能量段的数据，提取低能通道(归一化数据)与高能通道(原始数据)；
    对每一个能量通道分别绘制 2x2 对数衰减和拟合图，并在能量维度上汇总输出 slope_summary 折线图与归档 JSON。
    
    参数类型、含义及用法：
    ------------------
    - 无参数。
    
    用法：
    -----
    - 直接运行脚本时执行，控制对归一化前后两类数据的全量分析和画图落地。
    """ 
    # 扫描以 _dual 或 _noNorm_R 结尾的能量段目录
    subdirs = [d for d in os.listdir(contour_results_dir) 
               if os.path.isdir(os.path.join(contour_results_dir, d)) and (d.endswith('_dual') or d.endswith('_noNorm_R'))]
    
    if not subdirs:
        print("No valid energy directories found (ending with '_dual' or '_noNorm_R').")
        return

    # 按能量段数值进行排序 (如 20_dual, 30_dual -> 20, 30)
    # 并支持新格式 180kV_1mA+MERGE_E_100-110keV... 中的 _E_100-110keV 提取 100 作为能量值
    def extract_energy(name):
        match_e = re.search(r'_E_(\d+)', name)
        if match_e:
            return int(match_e.group(1))
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 9999

    subdirs = sorted(subdirs, key=extract_energy)
    print(f"Energy channels sorted: {subdirs}")

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

    I0_norm = 52428.0 # 16位图像本底入射强度值
    
    # 初始化全局数据收集字典
    global_raw_data_norm = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}
    global_raw_data_raw = {mat: {'energy': [], 'log_l': [], 'vl': []} for mat in step_mats.keys()}
    
    # 全局坐标轴范围收集列表
    norm_raw_pts, norm_log_pts = [], []
    raw_raw_pts, raw_log_pts = [], []

    # 存储每个 energy_band 的动态 I0_raw 因子
    I0_raw_dict = {}

    print(">>> Scanning dataset for limits and dynamic background values...")
    for energy_band in subdirs:
        e_val = extract_energy(energy_band)
        
        # 1. 扫描当前能量通道中所有的 pixels_high，求得最大值，作为动态 I0_raw 因子，防止 log Attenuation 为负
        max_high_val = 1.0
        for mat, file_label in step_mats.items():
            p = os.path.join(contour_results_dir, energy_band, "pixel_values", f"{energy_band}_{file_label}.pkl")
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                h_list = d['pixels_high']
                for h in h_list:
                    if len(h) > 0:
                        max_high_val = max(max_high_val, np.max(h))
                        
        I0_raw_band = 1.15 * max_high_val
        I0_raw_dict[energy_band] = I0_raw_band
        
        # 2. 逐材质提取和保存均值及对数衰减值
        for mat, file_label in step_mats.items():
            p = os.path.join(contour_results_dir, energy_band, "pixel_values", f"{energy_band}_{file_label}.pkl")
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                l_list = d['pixels_low']
                h_list = d['pixels_high']
                
                v_max_l = 65535 if l_list[0].dtype == np.uint16 or np.max(l_list[0]) > 255 else 255
                v_max_h = 65535 if h_list[0].dtype == np.uint16 or np.max(h_list[0]) > 255 else 255
                
                # 2.1 归一化数据 (低能)
                log_l_norm = np.array([np.mean(np.log(I0_norm / np.maximum(px[(px>=0)&(px<v_max_l)], 1.0))) if np.any((px>=0)&(px<v_max_l)) else np.nan for px in l_list])
                m_l_norm = np.array([np.mean(px[(px>=0)&(px<v_max_l)]) if np.any((px>=0)&(px<v_max_l)) else np.nan for px in l_list])
                vl_norm = np.isfinite(log_l_norm)
                
                global_raw_data_norm[mat]['energy'].append(e_val)
                global_raw_data_norm[mat]['log_l'].append(log_l_norm)
                global_raw_data_norm[mat]['vl'].append(vl_norm)
                
                norm_raw_pts.extend(m_l_norm[np.isfinite(m_l_norm)])
                norm_log_pts.extend(log_l_norm[np.isfinite(log_l_norm)])
                
                # 2.2 原始数据 (高能)
                log_l_raw = np.array([np.mean(np.log(I0_raw_band / np.maximum(px[(px>=0)&(px<v_max_h)], 1.0))) if np.any((px>=0)&(px<v_max_h)) else np.nan for px in h_list])
                m_l_raw = np.array([np.mean(px[(px>=0)&(px<v_max_h)]) if np.any((px>=0)&(px<v_max_h)) else np.nan for px in h_list])
                vl_raw = np.isfinite(log_l_raw)
                
                global_raw_data_raw[mat]['energy'].append(e_val)
                global_raw_data_raw[mat]['log_l'].append(log_l_raw)
                global_raw_data_raw[mat]['vl'].append(vl_raw)
                
                raw_raw_pts.extend(m_l_raw[np.isfinite(m_l_raw)])
                raw_log_pts.extend(log_l_raw[np.isfinite(log_l_raw)])

    # 3. 计算统一的 Y 轴范围限制（强制下限为 0，上限自适应向上取整）
    raw_ylim_norm = get_clean_ylim(norm_raw_pts, default_max=65535.0)
    log_ylim_norm = get_clean_ylim(norm_log_pts, default_max=5.0)
    raw_ylim_raw = get_clean_ylim(raw_raw_pts, default_max=2048.0)
    log_ylim_raw = get_clean_ylim(raw_log_pts, default_max=5.0)

    # 4. 逐能量段绘制 2x2 对数衰减和拟合大图
    print(">>> Generating dual channel 2x2 analysis plots per energy band...")
    for energy_band in subdirs:
        step_data = {}
        for mat, file_label in step_mats.items():
            p = os.path.join(contour_results_dir, energy_band, "pixel_values", f"{energy_band}_{file_label}.pkl")
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                step_data[mat] = (d['pixels_low'], d['pixels_high'])
        
        if step_data:
            perform_dual_channel_analysis(
                energy_band=energy_band,
                samples_dict=step_data,
                output_subdir=os.path.join(output_dir, "steps"),
                title_prefix="Photon Counter Dual-Channel Analysis",
                x_label="Thickness (mm)",
                x_coords_dict=thicknesses,
                I0_norm=I0_norm,
                I0_raw=I0_raw_dict[energy_band],
                raw_lims_norm=raw_ylim_norm,
                log_lims_norm=log_ylim_norm,
                raw_lims_raw=raw_ylim_raw,
                log_lims_raw=log_ylim_raw
            )
            print(f"[{energy_band}] Saved comprehensive 2x2 analysis plot.")

    # 5. 绘制并归档 combined 维度汇总大图及 JSON
    for mu_mode in ['mu', 'mu_m']:
        print(f"\n>>> Generating Combined 2x2 Slope Summaries for mu_mode: {mu_mode}...")
        plot_combined_slope_summaries(
            global_raw_data_norm=global_raw_data_norm,
            global_raw_data_raw=global_raw_data_raw,
            thicknesses=thicknesses,
            mu_mode=mu_mode,
            mat_physics=mat_physics,
            step_mats=step_mats,
            output_dir=output_dir
        )
        
        print(f">>> Generating Mu vs Thickness Plot for mu_mode: {mu_mode}...")
        plot_mu_vs_thickness(
            global_raw_data_norm=global_raw_data_norm,
            global_raw_data_raw=global_raw_data_raw,
            thicknesses=thicknesses,
            mu_mode=mu_mode,
            mat_physics=mat_physics,
            step_mats=step_mats,
            output_dir=output_dir
        )

    print("\nAll analyses completed successfully!")

if __name__ == '__main__':
    # contour_results_dir = r"E:\photo_electric_II\results\20260512_dual_180kV_1mA_no_subtracting_noise\contour_results"
    contour_results_dir = r"E:\photo_electric_II\results\20260512_160kV_1mA_subtracting_noise\contour_results"
    # contour_results_dir = r"E:\photo_electric_II\results\20260512_180kV_1mA_subtracting_noise\contour_results"
    # output_dir = r"E:\photo_electric_II\results\thickness_decoupling\H_L_fit\20260512_180kV_1mA_subtracting_noise"
    # output_dir = r"E:\photo_electric_II\results\thickness_decoupling\H_L_fit\20260512_180kV_1mA_dual"
    output_dir = r"E:\photo_electric_II\results\thickness_decoupling\H_L_fit\20260512_160kV_1mA_subtracting_noise"
   
    main(contour_results_dir, output_dir)
