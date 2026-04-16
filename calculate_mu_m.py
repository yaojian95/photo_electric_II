import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义参数
# 密度 rho (g/cm^3)
densities = {
    'Cu_step': 8.96,
    'Fe_step': 7.87,
    'Al_step_block': 2.70
}

# 阶梯厚度 t (mm) -> 转换为 cm 以后计算单位为 cm^2/g
thickness_cu_fe_mm = np.arange(2, 22, 2)
thickness_al_mm = np.arange(22, 42, 2) # 10 steps: 22, 24, ..., 40

# 入射强度 I0 (空气基准)
I0_low = 204.0
I0_high = 204.0

materials = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
voltages = ['140kV', '160kV', '180kV']

output_dir = 'results/thickness_decoupling'
os.makedirs(output_dir, exist_ok=True)

# 存储结果绘图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 收集所有结果
# results[voltage][mat_name] = {'mu_low': [], 'mu_high': [], 't_mm': [], 't_cm': []}
all_results = {v: {} for v in voltages}

for voltage in voltages:
    for m_idx, mat_name in materials.items():
        file_path = f'results/20260331/pixel_values/{voltage}_4mA_step_sample_{m_idx}_data.pkl'
        if not os.path.exists(file_path): continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        L_list, H_list = data['pixels_low'], data['pixels_high']
        if not isinstance(L_list, list): continue
            
        mu_m_low, mu_m_high = [], []
        rho = densities[mat_name]
        cur_thickness_mm = thickness_al_mm if 'Al' in mat_name else thickness_cu_fe_mm
        cur_thickness_cm = cur_thickness_mm / 10.0
        
        for i in range(min(len(L_list), len(cur_thickness_cm))):
            I_L = np.mean(L_list[i]) if L_list[i].size > 0 else 1e-6
            I_H = np.mean(H_list[i]) if H_list[i].size > 0 else 1e-6
            t = cur_thickness_cm[i]
            mu_m_low.append(-np.log(max(I_L, 1e-6) / I0_low) / (rho * t))
            mu_m_high.append(-np.log(max(I_H, 1e-6) / I0_high) / (rho * t))
            
        all_results[voltage][mat_name] = {
            'mu_low': np.array(mu_m_low),
            'mu_high': np.array(mu_m_high),
            't_mm': cur_thickness_mm[:len(mu_m_low)]
        }

# --- 第一排：同一电压下，不同物质对比 ---
for v_idx, voltage in enumerate(voltages):
    ax = axes[0, v_idx]
    for mat_name, res in all_results[voltage].items():
        plot_x = res['t_mm']
        display_label = f"{mat_name}"
        if 'Al' in mat_name: 
            plot_x = plot_x - 20
            display_label += " (t-20mm)"
            
        line, = ax.plot(plot_x, res['mu_low'], 'o-', label=f"{display_label} Low")
        ax.plot(plot_x, res['mu_high'], 's--', color=line.get_color(), label=f"{display_label} High")
        
    ax.set_title(f"Comparison by Voltage: {voltage}")
    ax.set_xlabel("Thickness (mm)")
    ax.set_ylabel(r"$\mu_m$ ($cm^2/g$)")
    ax.legend(fontsize='x-small', ncol=2)
    ax.grid(True)

# --- 第二排：同一物质下，不同电压对比 ---
# 建立从材质名称到列索引的映射
mat_to_col = {'Cu_step': 0, 'Fe_step': 1, 'Al_step_block': 2}
for mat_name, col_idx in mat_to_col.items():
    ax = axes[1, col_idx]
    for voltage in voltages:
        if mat_name not in all_results[voltage]: continue
        res = all_results[voltage][mat_name]
        
        plot_x = res['t_mm']
        if 'Al' in mat_name: plot_x = plot_x - 20
            
        line, = ax.plot(plot_x, res['mu_low'], 'o-', label=f"{voltage} Low")
        ax.plot(plot_x, res['mu_high'], 's--', color=line.get_color(), label=f"{voltage} High")
        
    sub_title = f"Comparison by Material: {mat_name}"
    if 'Al' in mat_name: sub_title += " (X-axis offset -20mm)"
    ax.set_title(sub_title)
    ax.set_xlabel("Thickness (mm)")
    ax.set_ylabel(r"$\mu_m$ ($cm^2/g$)")
    ax.legend(fontsize='x-small', ncol=2)
    ax.grid(True)

plt.suptitle(r"Mass Attenuation Coefficient $\mu_m$ Comprehensive Analysis,  $\mu_m = \frac{\ln(I_0/I)}{\rho \cdot t}$", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{output_dir}/mu_m_analysis.png")
plt.show()

print(f"Analysis complete. Plot saved to {output_dir}/mu_m_analysis.png")
