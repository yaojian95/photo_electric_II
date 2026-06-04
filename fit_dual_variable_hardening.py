import os
import pickle
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import get_mu_from_nist_new

# ==========================================
# 物理常数与管电压/滤片参数设置
# ==========================================
densities = {
    'Cu_step': 8.96,
    'Fe_step': 7.87,
    'Al_step': 2.70
}

thickness_map = {
    'Cu_step': np.arange(2, 22, 2),
    'Fe_step': np.arange(2, 22, 2),
    'Al_step': np.arange(12, 32, 2)
}

nist_symbols = {
    'Cu_step': 'Cu',
    'Fe_step': 'Fe',
    'Al_step': 'Al'
}

I0_val = 52428.0  # 16位透射图像背景入射强度
voltages = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
filter_types = ['0.6mm', '1.2mm']
input_dir = 'results/20260429_mask_generated_16bit/pixel_values'
output_dir = 'results/thickness_decoupling/energy_hardening/dual_variable_fit'
os.makedirs(output_dir, exist_ok=True)


def get_energy_from_mu_rho(element_symbol: str, mu_rho_list: list, data_dir: str = 'nist_data') -> np.ndarray:
    """
    根据给定的质量衰减系数 mu/rho (cm^2/g) 在对数空间逆插值反算出 X 射线等效单能值 (keV)。
    
    参数：
    - element_symbol (str): 元素符号（如 'Cu', 'Fe', 'Al'）。
      用法：传入对应元素的化学符号字符串。
    - mu_rho_list (list 或 np.ndarray): 实测的质量衰减系数一维数组。
      用法：传入包含多个质量衰减系数的实数列表。
    - data_dir (str): 保存 NIST 数据文件的本地文件夹路径。
      用法：默认为 'nist_data'。
      
    返回：
    - np.ndarray: 对应各个质量衰减系数下的等效能量一维数组（单位：keV）。
    """
    csv_path = os.path.join(data_dir, f"{element_symbol}_mu_rho.csv")
    if not os.path.exists(csv_path):
        get_mu_from_nist_new.save_mu_rho_to_local(element_symbol, data_dir)
        
    e_raw, mu_raw = [], []
    with open(csv_path, 'r') as f:
        import csv
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            e_raw.append(float(row[0]) * 1000.0)  # MeV -> keV
            mu_raw.append(float(row[1]))          # cm^2/g
            
    e_raw = np.array(e_raw)
    mu_raw = np.array(mu_raw)
    
    log_mu = np.log10(mu_raw)
    log_e = np.log10(e_raw)
    
    # 按照 log(mu) 排序以确保线性逆插值的单调性
    sort_idx = np.argsort(log_mu)
    inv_interp = interp1d(log_mu[sort_idx], log_e[sort_idx], kind='linear', fill_value="extrapolate")
    
    res_log_e = inv_interp(np.log10(mu_rho_list))
    return 10**res_log_e


def collect_dataset(f_type: str, voltage: str) -> dict:
    """
    加载指定电压和滤片下的铜、铁、铝标样数据，计算并收集每个阶梯层的实测高低能对数衰减及对应的等效能量因变量。
    
    参数：
    - f_type (str): 滤片厚度配置描述符（如 '0.6mm', '1.2mm'）。
      用法：传入包含滤片厚度的字符串。
    - voltage (str): 射线管电压描述符（如 '200kV'）。
      用法：传入包含电压的字符串。
      
    返回：
    - dict: 包含自变量 H, L, L/H 及因变量 E_L, E_H 一维数组的字典。如果无有效数据点则返回空字典。
    """
    H_all, L_all, ratio_all = [], [], []
    E_L_all, E_H_all = [], []
    
    for mat_name in ['Cu_step', 'Fe_step', 'Al_step']:
        file_name = f"{mat_name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl"
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        l_list, h_list = data['pixels_low'], data['pixels_high']
        t_mm = thickness_map[mat_name]
        t_cm = t_mm / 10.0
        nist_symbol = nist_symbols[mat_name]
        rho = densities[mat_name]
        
        # 16位最大灰度阈值，用于剔除死/饱和像素
        v_max = 65535
        
        for s in range(min(len(l_list), len(t_cm))):
            l_v = l_list[s]
            h_v = h_list[s]
            
            mask = (l_v > 0) & (h_v > 0) & (l_v < v_max) & (h_v < v_max)
            l_val = l_v[mask].astype(float)
            h_val = h_v[mask].astype(float)
            
            if len(l_val) == 0:
                continue
                
            # 计算无 Jensen 误差的像素级对数衰减，再求均值
            mu_L_d = np.log(I0_val / (l_val + 1e-6))
            mu_H_d = np.log(I0_val / (h_val + 1e-6))
            
            L_mean = float(np.mean(mu_L_d))
            H_mean = float(np.mean(mu_H_d))
            
            if L_mean <= 0 or H_mean <= 0:
                continue
                
            # 实测质量衰减系数 cm^2/g
            mu_m_L = L_mean / (rho * t_cm[s])
            mu_m_H = H_mean / (rho * t_cm[s])
            
            try:
                # 反向插值获取等效能量 (keV)
                e_L = get_energy_from_mu_rho(nist_symbol, [mu_m_L])[0]
                e_H = get_energy_from_mu_rho(nist_symbol, [mu_m_H])[0]
                
                # 剔除物理上不合理的外推边界点
                v_limit = int(voltage.replace('kV', ''))
                if e_L < 10.0 or e_L > v_limit or e_H < 10.0 or e_H > v_limit:
                    continue
                    
                L_all.append(L_mean)
                H_all.append(H_mean)
                ratio_all.append(L_mean / H_mean)
                E_L_all.append(e_L)
                E_H_all.append(e_H)
            except:
                pass
                
    if len(H_all) < 5:  # 样本数过少无法稳定进行多元拟合
        return {}
        
    return {
        'H': np.array(H_all),
        'L': np.array(L_all),
        'LH_ratio': np.array(ratio_all),
        'E_L_true': np.array(E_L_all),
        'E_H_true': np.array(E_H_all)
    }


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    执行多元线性回归，估计给定特征矩阵和因变量下的拟合系数，并计算拟合优度 R2 和均方根误差 RMSE。
    
    参数：
    - X (np.ndarray): 特征矩阵（自变量），形状为 (N, D)。
      用法：传入自变量特征二维数组。
    - y (np.ndarray): 目标因变量一维数组，形状为 (N,)。
      用法：传入待拟合的目标值向量。
      
    返回：
    - tuple: 包含拟合系数列表 [intercept, c0, c1...]、R2 (float)、RMSE (float) 和预测值一维数组。
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # 系数格式：[intercept, coef_0, coef_1, ...]
    coefs = [reg.intercept_] + reg.coef_.tolist()
    return coefs, r2, rmse, y_pred


def run_dual_variable_fitting():
    """
    主控制流程：遍历电压与滤片组合，提取阶梯衰减数据与等效单能，
    分别拟合模型 1 (基于 Z 代理比值的物理模型) 和模型 2 (通用二元二次多项式)，
    输出拟合系数 JSON，并保存对比分析图表。
    """
    fit_summary = {}
    
    for f_type in filter_types:
        fit_summary[f_type] = {}
        for voltage in voltages:
            print(f"\n>>> Analyzing energy hardening fitting for {f_type} at {voltage}...")
            data_dict = collect_dataset(f_type, voltage)
            if not data_dict:
                print(f"Skipping {f_type}-{voltage}: Insufficient data points.")
                continue
                
            H = data_dict['H']
            L = data_dict['L']
            ratio = data_dict['LH_ratio']
            E_L_true = data_dict['E_L_true']
            E_H_true = data_dict['E_H_true']
            
            # ==========================================
            # 准备自变量特征矩阵
            # ==========================================
            # 模型 1 (物理启发式): E = a0 + a1 * H + a2 * H^2 + a3 * (L/H)
            X_m1 = np.column_stack((H, H**2, ratio))
            
            # 模型 2 (通用二次多项式): E = c0 + c1*H + c2*L + c3*H^2 + c4*L^2 + c5*(H*L)
            X_m2 = np.column_stack((H, L, H**2, L**2, H * L))
            
            # ==========================================
            # 分别进行拟合
            # ==========================================
            # 1. 拟合低能 E_L
            coef_L_m1, r2_L_m1, rmse_L_m1, E_L_pred_m1 = fit_linear_regression(X_m1, E_L_true)
            coef_L_m2, r2_L_m2, rmse_L_m2, E_L_pred_m2 = fit_linear_regression(X_m2, E_L_true)
            
            # 2. 拟合高能 E_H
            coef_H_m1, r2_H_m1, rmse_H_m1, E_H_pred_m1 = fit_linear_regression(X_m1, E_H_true)
            coef_H_m2, r2_H_m2, rmse_H_m2, E_H_pred_m2 = fit_linear_regression(X_m2, E_H_true)
            
            print(f"E_L Model 1: R2 = {r2_L_m1:.4f}, RMSE = {rmse_L_m1:.4f} keV")
            print(f"E_L Model 2: R2 = {r2_L_m2:.4f}, RMSE = {rmse_L_m2:.4f} keV")
            print(f"E_H Model 1: R2 = {r2_H_m1:.4f}, RMSE = {rmse_H_m1:.4f} keV")
            print(f"E_H Model 2: R2 = {r2_H_m2:.4f}, RMSE = {rmse_H_m2:.4f} keV")
            
            # 记录拟合结果
            fit_summary[f_type][voltage] = {
                'points_count': len(H),
                'E_L': {
                    'model_1_启发式': {
                        'formula': 'E_L = a0 + a1*H + a2*H^2 + a3*(L/H)',
                        'coefficients': coef_L_m1,
                        'R2': r2_L_m1,
                        'RMSE_keV': rmse_L_m1
                    },
                    'model_2_多项式': {
                        'formula': 'E_L = c0 + c1*H + c2*L + c3*H^2 + c4*L^2 + c5*H*L',
                        'coefficients': coef_L_m2,
                        'R2': r2_L_m2,
                        'RMSE_keV': rmse_L_m2
                    }
                },
                'E_H': {
                    'model_1_启发式': {
                        'formula': 'E_H = b0 + b1*H + b2*H^2 + b3*(L/H)',
                        'coefficients': coef_H_m1,
                        'R2': r2_H_m1,
                        'RMSE_keV': rmse_H_m1
                    },
                    'model_2_多项式': {
                        'formula': 'E_H = d0 + d1*H + d2*L + d3*H^2 + d4*L^2 + d5*H*L',
                        'coefficients': coef_H_m2,
                        'R2': r2_H_m2,
                        'RMSE_keV': rmse_H_m2
                    }
                }
            }
            
            # ==========================================
            # 可视化评估绘图 (2x2布局)
            # ==========================================
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Energy Hardening Double-Variable Fitting - {f_type} at {voltage}\n(Samples count: {len(H)})", fontsize=14, fontweight='bold')
            
            # 子图 (0,0): E_L Model 1
            ax = axes[0, 0]
            ax.scatter(E_L_true, E_L_pred_m1, color='#E28743', alpha=0.8, edgecolors='k', label='Step Samples')
            ax.plot([E_L_true.min()-1, E_L_true.max()+1], [E_L_true.min()-1, E_L_true.max()+1], 'r--', label='Ideal 1:1')
            ax.set_title(f"E_L Model 1 (Heuristic)\n$R^2$={r2_L_m1:.4f}, RMSE={rmse_L_m1:.3f} keV", fontsize=11)
            ax.set_xlabel("True $E_L$ (keV)"); ax.set_ylabel("Predicted $E_L$ (keV)")
            ax.grid(True, linestyle='--', alpha=0.4); ax.legend(fontsize='small')
            
            # 子图 (0,1): E_L Model 2
            ax = axes[0, 1]
            ax.scatter(E_L_true, E_L_pred_m2, color='#4A90E2', alpha=0.8, edgecolors='k', label='Step Samples')
            ax.plot([E_L_true.min()-1, E_L_true.max()+1], [E_L_true.min()-1, E_L_true.max()+1], 'r--', label='Ideal 1:1')
            ax.set_title(f"E_L Model 2 (2D Poly)\n$R^2$={r2_L_m2:.4f}, RMSE={rmse_L_m2:.3f} keV", fontsize=11)
            ax.set_xlabel("True $E_L$ (keV)"); ax.set_ylabel("Predicted $E_L$ (keV)")
            ax.grid(True, linestyle='--', alpha=0.4); ax.legend(fontsize='small')
            
            # 子图 (1,0): E_H Model 1
            ax = axes[1, 0]
            ax.scatter(E_H_true, E_H_pred_m1, color='#2ECCA7', alpha=0.8, edgecolors='k', label='Step Samples')
            ax.plot([E_H_true.min()-1, E_H_true.max()+1], [E_H_true.min()-1, E_H_true.max()+1], 'r--', label='Ideal 1:1')
            ax.set_title(f"E_H Model 1 (Heuristic)\n$R^2$={r2_H_m1:.4f}, RMSE={rmse_H_m1:.3f} keV", fontsize=11)
            ax.set_xlabel("True $E_H$ (keV)"); ax.set_ylabel("Predicted $E_H$ (keV)")
            ax.grid(True, linestyle='--', alpha=0.4); ax.legend(fontsize='small')
            
            # 子图 (1,1): E_H Model 2
            ax = axes[1, 1]
            ax.scatter(E_H_true, E_H_pred_m2, color='#9B59B6', alpha=0.8, edgecolors='k', label='Step Samples')
            ax.plot([E_H_true.min()-1, E_H_true.max()+1], [E_H_true.min()-1, E_H_true.max()+1], 'r--', label='Ideal 1:1')
            ax.set_title(f"E_H Model 2 (2D Poly)\n$R^2$={r2_H_m2:.4f}, RMSE={rmse_H_m2:.3f} keV", fontsize=11)
            ax.set_xlabel("True $E_H$ (keV)"); ax.set_ylabel("Predicted $E_H$ (keV)")
            ax.grid(True, linestyle='--', alpha=0.4); ax.legend(fontsize='small')
            
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"dual_variable_fit_{f_type}_{voltage}.png")
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)
            
    # 将拟合参数归档持久化写入 JSON 文件
    json_path = os.path.join(output_dir, 'dual_variable_coefficients.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(fit_summary, f, indent=4, ensure_ascii=False)
        
    print(f"\n==========================================")
    print(f"Dual-variable energy hardening fitting analysis completed!")
    print(f"Coefficients JSON saved to: {json_path}")
    print(f"Diagnostic fitting images saved to: {output_dir}")
    print(f"==========================================")


if __name__ == "__main__":
    run_variable_fitting = True
    if run_variable_fitting:
        run_dual_variable_fitting()
