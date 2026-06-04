import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import json
from scipy.interpolate import interp1d
import get_mu_from_nist_new

# ==========================================
# 设置参数与物理常数
# ==========================================
dataset_date = '20260429'
densities = {
    'Cu_step': 8.96,
    'Fe_step': 7.87,
    'Al_step': 2.70
}

# 阶梯厚度 (mm)
thickness_map = {
    'Cu_step': np.arange(2, 22, 2),
    'Fe_step': np.arange(2, 22, 2),
    'Al_step': np.arange(12, 32, 2)
}

# NIST 符号映射
nist_symbols = {
    'Cu_step': 'Cu',
    'Fe_step': 'Fe',
    'Al_step': 'Al'
}

I0_low = 255 * 0.8
I0_high = 255 * 0.8
voltages = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
filter_types = ['0.6mm', '1.2mm']
base_results_dir = 'results/20260429_mask_generated/pixel_values'

output_dir = 'results/thickness_decoupling/energy_hardening'
os.makedirs(output_dir, exist_ok=True)

def get_energy_from_mu_rho(element_symbol, mu_rho_list, data_dir='nist_data'):
    """
    根据质量衰减系数 mu/rho (cm^2/g) 反向推算光子能量 (keV)。
    
    参数:
        element_symbol (str): 元素符号，如 'Cu', 'Fe', 'Al'。
        mu_rho_list (list/ndarray): 质量衰减系数列表，单位 cm^2/g。
        data_dir (str): NIST 数据存储目录。
        
    返回:
        ndarray: 推算出的等效能量列表，单位 keV。
    """
    # 加载 NIST 数据
    csv_path = os.path.join(data_dir, f"{element_symbol}_mu_rho.csv")
    if not os.path.exists(csv_path):
        get_mu_from_nist_new.save_mu_rho_to_local(element_symbol, data_dir)
        
    e_raw, mu_raw = [], []
    with open(csv_path, 'r') as f:
        import csv
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            e_raw.append(float(row[0]) * 1000.0) # MeV -> keV
            mu_raw.append(float(row[1])) # cm^2/g
            
    e_raw, mu_raw = np.array(e_raw), np.array(mu_raw)
    
    # 保证单调性以便插值 (mu_rho 随能量增加而减小)
    # 对数空间插值
    log_mu = np.log10(mu_raw)
    log_e = np.log10(e_raw)
    
    # 建立 log(mu) -> log(E) 的映射
    # 注意：mu 在低能段可能不单调（吸收边），但在 20keV 以上通常是单调的
    # 我们对数据进行排序以确保 interp1d 正常工作
    sort_idx = np.argsort(log_mu)
    inv_interp = interp1d(log_mu[sort_idx], log_e[sort_idx], kind='linear', fill_value="extrapolate")
    
    res_log_e = inv_interp(np.log10(mu_rho_list))
    return 10**res_log_e

def analyze_hardening():
    """
    主分析函数：计算各材质、各电压下的能量随厚度变化曲线，并外推初始能量。
    """
    summary_results = {f: {v: {} for v in voltages} for f in filter_types}

    for f_type in filter_types:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Energy Hardening Analysis (Effective Energy vs Thickness) - Filter: {f_type}', fontsize=16)
        
        for m_idx, mat_name in enumerate(['Cu_step', 'Fe_step', 'Al_step']):
            ax = axes[m_idx]
            nist_symbol = nist_symbols[mat_name]
            rho = densities[mat_name]
            t_mm = thickness_map[mat_name]
            t_cm = t_mm / 10.0
            
            for v_idx, voltage in enumerate(voltages):
                file_name = f"{mat_name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl"
                file_path = os.path.join(base_results_dir, file_name)
                
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                L_list, H_list = data['pixels_low'], data['pixels_high']
                
                # 清除 pickle data 释放内存
                del data
                
                # 计算每一级的有效能量 (Low Energy)
                e_list_l = []
                valid_t_l = []
                for i in range(min(len(L_list), len(t_cm))):
                    I_L = np.mean(L_list[i])
                    if I_L > 0 and I_L < I0_low:
                        mu_m = -np.log(I_L / I0_low) / (rho * t_cm[i])
                        try:
                            energy = get_energy_from_mu_rho(nist_symbol, [mu_m])[0]
                            e_list_l.append(energy)
                            valid_t_l.append(t_mm[i])
                        except: pass
                
                # 计算每一级的有效能量 (High Energy)
                e_list_h = []
                valid_t_h = []
                for i in range(min(len(H_list), len(t_cm))):
                    I_H = np.mean(H_list[i])
                    if I_H > 0 and I_H < I0_high:
                        mu_m = -np.log(I_H / I0_high) / (rho * t_cm[i])
                        try:
                            energy = get_energy_from_mu_rho(nist_symbol, [mu_m])[0]
                            e_list_h.append(energy)
                            valid_t_h.append(t_mm[i])
                        except: pass
                
                # 释放大列表内存
                del L_list, H_list
                import gc
                gc.collect()
                
                # 绘制数据点与均值
                if e_list_l or e_list_h:
                    # 使用同一个颜色表示同一个电压
                    color = plt.get_cmap('tab10')(v_idx % 10)
                    
                    e_avg_l, e_avg_h = None, None
                    if e_list_l:
                        ax.plot(valid_t_l, e_list_l, 'o', color=color, alpha=0.5, label=f'{voltage} Low')
                        e_avg_l = np.mean(e_list_l)
                        ax.axhline(e_avg_l, color=color, linestyle='--', alpha=0.3)
                    
                    if e_list_h:
                        ax.plot(valid_t_h, e_list_h, 's', color=color, alpha=0.5, label=f'{voltage} High')
                        e_avg_h = np.mean(e_list_h)
                        ax.axhline(e_avg_h, color=color, linestyle=':', alpha=0.3)
                        
                    summary_results[f_type][voltage][mat_name] = {
                        'e_avg_l': float(e_avg_l) if e_avg_l is not None else None,
                        'e_profile_l': [float(e) for e in e_list_l],
                        't_profile_l': [float(t) for t in valid_t_l],
                        'e_avg_h': float(e_avg_h) if e_avg_h is not None else None,
                        'e_profile_h': [float(e) for e in e_list_h],
                        't_profile_h': [float(t) for t in valid_t_h]
                    }

            ax.set_title(f'Material: {nist_symbol}')
            ax.set_xlabel('Thickness (mm)')
            ax.set_ylabel('Effective Energy (keV)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='x-small', loc='lower right')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'energy_hardening_{f_type}.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        gc.collect()

    # 保存 JSON 汇总结果
    json_path = os.path.join(output_dir, 'hardening_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
            
    # 绘制 E_avg 随管电压的变化图
    plot_eavg_summary(summary_results)
    
    # 绘制不同厚度下的能量随管电压变化图
    plot_energy_by_thickness_vs_voltage(summary_results)

def plot_energy_by_thickness_vs_voltage(summary_results):
    """
    针对每一个厚度级别（共10个阶梯），分别画出不同厚度下推出的等效能量随管电压变化的关系，并独立保存。
    图表布局与 plot_eavg_summary 类似，采用 1x2 子图，左边为 0.6mm 滤片结果，右边为 1.2mm 滤片结果，
    在各子图内同时画出铜、铁、铝三材质在该厚度下的变化曲线。
    
    参数类型:
        summary_results (dict): 包含分析结果的嵌套字典。结构为:
            {
                filter_type (str): {
                    voltage (str): {
                        material_name (str): {
                            'e_profile_l': list of float,
                            't_profile_l': list of float,
                            'e_profile_h': list of float,
                            't_profile_h': list of float,
                            ...
                        }
                    }
                }
            }
            
    参数含义:
        summary_results: 阶段分析中提取的各电压、各滤片、各材质及各厚度下的等效能量与厚度对应数据的汇总。
        
    用法:
        在分析完能量硬化数据并生成 summary_results 字典后调用此函数，将会在 output_dir 下的
        `by_thickness/` 目录下生成一系列独立的对比图像：
        `energy_vs_voltage_step{step_idx + 1}_{thicknesses}.png`。
    """
    by_thickness_dir = os.path.join(output_dir, 'by_thickness')
    os.makedirs(by_thickness_dir, exist_ok=True)
    
    # 所有材质的阶梯都是 10 阶 (对应索引 0 到 9)
    for step_idx in range(10):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 收集该步骤中各材质的厚度，以生成文件名描述和标题
        thickness_desc = []
        for mat_name in ['Cu_step', 'Fe_step', 'Al_step']:
            t = thickness_map[mat_name][step_idx]
            thickness_desc.append(f"{nist_symbols[mat_name]}_{t}mm")
        desc_str = "_".join(thickness_desc)
        
        fig.suptitle(f'Effective Energy vs Tube Voltage - Step {step_idx + 1} ({desc_str.replace("_", ", ")})', fontsize=16)
        
        for f_idx, f_type in enumerate(filter_types):
            ax = axes[f_idx]
            
            for m_idx, mat_name in enumerate(['Cu_step', 'Fe_step', 'Al_step']):
                nist_symbol = nist_symbols[mat_name]
                t = thickness_map[mat_name][step_idx]
                
                vs_l, es_l = [], []
                vs_h, es_h = [], []
                
                for voltage in voltages:
                    v_int = int(voltage.replace('kV', ''))
                    if voltage in summary_results[f_type] and mat_name in summary_results[f_type][voltage]:
                        res = summary_results[f_type][voltage][mat_name]
                        
                        # 查找 low energy
                        if t in res.get('t_profile_l', []):
                            idx = res['t_profile_l'].index(t)
                            vs_l.append(v_int)
                            es_l.append(res['e_profile_l'][idx])
                            
                        # 查找 high energy
                        if t in res.get('t_profile_h', []):
                            idx = res['t_profile_h'].index(t)
                            vs_h.append(v_int)
                            es_h.append(res['e_profile_h'][idx])
                
                # 绘制当前材质当前厚度的曲线，铜铁铝各使用不同颜色
                color = plt.get_cmap('tab10')(m_idx % 10)
                if vs_l:
                    ax.plot(vs_l, es_l, 'o-', color=color, label=f"{nist_symbol} ({t}mm) Low")
                if vs_h:
                    ax.plot(vs_h, es_h, 's--', color=color, label=f"{nist_symbol} ({t}mm) High")
            
            ax.set_title(f'Filter: {f_type}')
            ax.set_xlabel('Tube Voltage (kV)')
            ax.set_ylabel('Effective Energy (keV)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small', loc='best')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_name = f'energy_vs_voltage_step{step_idx + 1}_{desc_str}.png'
        save_path = os.path.join(by_thickness_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        plt.close('all')
        
        # 及时释放内存
        import gc
        gc.collect()

def plot_eavg_summary(summary_results):
    """
    绘制平均有效能量 E_avg 随管电压变化的总结图。
    
    参数类型:
        summary_results (dict): 包含分析结果的嵌套字典。
        
    参数含义:
        summary_results: 阶段分析中提取的各电压、各滤片、各材质及各厚度下的等效能量与厚度对应数据的汇总。
        
    用法:
        传入 summary_results，生成 E_avg 随电压变化趋势图并保存为 `eavg_voltage_dependency.png`。
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Average Effective Energy (E_avg) vs Tube Voltage', fontsize=16)
    
    for f_idx, f_type in enumerate(filter_types):
        ax = axes[f_idx]
        for m_idx, mat_name in enumerate(['Cu_step', 'Fe_step', 'Al_step']):
            vs, e_ls, e_hs = [], [], []
            for voltage in voltages:
                if mat_name in summary_results[f_type][voltage]:
                    res = summary_results[f_type][voltage][mat_name]
                    v_int = int(voltage.replace('kV', ''))
                    vs.append(v_int)
                    e_ls.append(res['e_avg_l'])
                    e_hs.append(res['e_avg_h'])
            
            if vs:
                color = plt.get_cmap('tab10')(m_idx % 10)
                ax.plot(vs, e_ls, 'o-', color=color, label=f"{nist_symbols[mat_name]} Low")
                ax.plot(vs, e_hs, 's--', color=color, label=f"{nist_symbols[mat_name]} High")
        
        ax.set_title(f'Filter: {f_type}')
        ax.set_xlabel('Tube Voltage (kV)')
        ax.set_ylabel('Average Effective Energy (keV)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, 'eavg_voltage_dependency.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    analyze_hardening()
