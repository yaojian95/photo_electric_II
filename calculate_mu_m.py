import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 设置数据集参数
# ==========================================
dataset_date = '20260429'  # 修改为 '20260429' 处理新数据

# 密度 rho (g/cm^3)
densities = {
    'Cu_step': 8.96,
    'Fe_step': 7.87,
    'Al_step': 2.70,
    'Al_step_block': 2.70
}

# 阶梯厚度 t (mm) -> 转换为 cm 以后计算单位为 cm^2/g
thickness_cu_fe_mm = np.arange(2, 22, 2)  # 2, 4, ..., 20
thickness_al_mm = np.arange(12, 32, 2)    # 12, 14, ..., 30

# 空气基准强度 I0
if dataset_date == '20260429':
    I0_low = 255 * 0.8
    I0_high = 255 * 0.8
    voltages = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
    materials = ['Cu_step', 'Fe_step', 'Al_step']
    filter_types = ['0.6mm', '1.2mm']
    base_results_dir = 'results/20260429_mask_generated/pixel_values'
elif dataset_date == '20260331':
    I0_low = 204.0
    I0_high = 204.0
    voltages = ['140kV', '160kV', '180kV']
    materials = ['Cu_step', 'Fe_step', 'Al_step_block']
    filter_types = [None]
    base_results_dir = 'results/20260331/pixel_values'
else:
    # 默认或其他日期
    I0_low = 204.0
    I0_high = 204.0
    voltages = ['140kV', '160kV', '180kV']
    materials = ['Cu_step', 'Fe_step', 'Al_step_block']
    filter_types = [None]
    base_results_dir = f'results/{dataset_date}/pixel_values'

output_dir = os.path.join('results', 'thickness_decoupling', 'mu_m')
os.makedirs(output_dir, exist_ok=True)

# 结构: data_store[f_type][voltage][mat_name] = {'t_vis': [], 'mu_l': [], 'mu_h': []}
data_store = {f: {v: {m: {'t_vis_l': [], 'mu_l': [], 't_vis_h': [], 'mu_h': []} for m in materials} for v in voltages} for f in filter_types}

global_mu_min = float('inf')
global_mu_max = float('-inf')

# 1. 第一遍扫描，加载所有数据并计算极限
for f_type in filter_types:
    for voltage in voltages:
        for mat_name in materials:
            if dataset_date == '20260429':
                file_name = f"{mat_name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl"
            else:
                m_idx = 0 if 'Cu' in mat_name else 1 if 'Fe' in mat_name else 2
                file_name = f"{voltage}_4mA_step_sample_{m_idx}_data.pkl"
            
            file_path = os.path.join(base_results_dir, file_name)
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            L_list, H_list = data['pixels_low'], data['pixels_high']
            if not isinstance(L_list, list): 
                continue
            
            rho = densities[mat_name]
            cur_thickness_mm = thickness_al_mm if 'Al' in mat_name else thickness_cu_fe_mm
            cur_thickness_cm = cur_thickness_mm / 10.0
            
            for i in range(min(len(L_list), len(cur_thickness_cm))):
                t = cur_thickness_cm[i]
                # 横向厚度视觉对齐到 2-20mm
                visual_x = thickness_cu_fe_mm[i] if 'Al' in mat_name else cur_thickness_mm[i]
                
                if L_list[i].size > 0 and np.mean(L_list[i]) > 0:
                    I_L = np.mean(L_list[i])
                    if I_L < I0_low:
                        val_l = -np.log(max(I_L, 0.1) / I0_low) / (rho * t)
                        data_store[f_type][voltage][mat_name]['mu_l'].append(val_l)
                        data_store[f_type][voltage][mat_name]['t_vis_l'].append(visual_x)
                        global_mu_min = min(global_mu_min, val_l)
                        global_mu_max = max(global_mu_max, val_l)
                
                if H_list[i].size > 0 and np.mean(H_list[i]) > 0:
                    I_H = np.mean(H_list[i])
                    if I_H < I0_high:
                        val_h = -np.log(max(I_H, 0.1) / I0_high) / (rho * t)
                        data_store[f_type][voltage][mat_name]['mu_h'].append(val_h)
                        data_store[f_type][voltage][mat_name]['t_vis_h'].append(visual_x)
                        global_mu_min = min(global_mu_min, val_h)
                        global_mu_max = max(global_mu_max, val_h)

# 计算统一的 Y 轴范围
if global_mu_min != float('inf'):
    y_pad = (global_mu_max - global_mu_min) * 0.1
    global_ylim = (max(0, global_mu_min - y_pad), global_mu_max + y_pad)
else:
    global_ylim = (0, 1)

# 2. 第二遍扫描，绘图
for f_type in filter_types:
    f_suffix = f_type if f_type else "Default"
    
    # --- 图 1: 子图按 Voltage 划分，画出每种 material ---
    fig_volt_ax, axes_volt_ax = plt.subplots(2, 4, figsize=(24, 12))
    fig_volt_ax.suptitle(f'Mass Attenuation Coefficient - {f_suffix}', fontsize=16)
    
    for v_idx, voltage in enumerate(voltages):
        ax = axes_volt_ax[v_idx // 4, v_idx % 4]
        
        for mat_name in materials:
            d = data_store[f_type][voltage][mat_name]
            if d['mu_l']:
                ax.plot(d['t_vis_l'], d['mu_l'], 'o-', label=f'{mat_name} Low')
            if d['mu_h']:
                ax.plot(d['t_vis_h'], d['mu_h'], 's--', label=f'{mat_name} High')
                
        ax.set_title(f'Voltage: {voltage}')
        ax.set_xlabel('Thickness (mm)')
        ax.set_ylabel(r'$\mu_m \ (\mathrm{cm}^2/\mathrm{g})$')
        ax.set_ylim(global_ylim)
        ax.legend()
        ax.grid(True)
        
    fig_volt_ax.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_v = os.path.join(output_dir, f'mu_m_by_voltage_{dataset_date}_{f_suffix}.png')
    fig_volt_ax.savefig(save_path_v)
    print(f"Plot saved to {save_path_v}")
    plt.close(fig_volt_ax)

    # --- 图 2: 子图按 Material 划分，画出不同 voltage ---
    fig_mat_ax, axes_mat_ax = plt.subplots(1, 3, figsize=(18, 5))
    fig_mat_ax.suptitle(f'Material Attenuation vs Thickness (Varying Voltage) - {f_suffix}', fontsize=16)
    
    cmap = plt.get_cmap('tab10' if len(voltages) <= 10 else 'tab20')
    
    for m_idx, mat_name in enumerate(materials):
        ax = axes_mat_ax[m_idx]
        
        for v_idx, voltage in enumerate(voltages):
            color = cmap(v_idx)
            d = data_store[f_type][voltage][mat_name]
            if d['mu_l']:
                ax.plot(d['t_vis_l'], d['mu_l'], 'o-', color=color, label=f'{voltage} Low' if m_idx == 0 else "")
            if d['mu_h']:
                ax.plot(d['t_vis_h'], d['mu_h'], 's--', color=color, label=f'{voltage} High' if m_idx == 0 else "")
                
        ax.set_title(f'Material: {mat_name}')
        ax.set_xlabel('Thickness (mm)')
        ax.set_ylabel(r'$\mu_m \ (\mathrm{cm}^2/\mathrm{g})$')
        ax.set_ylim(global_ylim)
        ax.grid(True)
        if m_idx == 0:
            ax.legend(fontsize='small')
            
    fig_mat_ax.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_m = os.path.join(output_dir, f'mu_m_by_material_{dataset_date}_{f_suffix}.png')
    fig_mat_ax.savefig(save_path_m)
    print(f"Plot saved to {save_path_m}")
    plt.close(fig_mat_ax)

print("All plots generated successfully.")
