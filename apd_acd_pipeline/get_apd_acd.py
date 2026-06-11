import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.optimize

CONFIG = {
    'E_L': 58, #30,                               # 低能能量 (keV)
    'E_H': 105 #60,                               # 高能能量 (keV)
}

# ===================== a_p和a_c特征计算 =====================

def _fkn(E_keV: np.ndarray) -> np.ndarray:
    """
    计算克莱因-仁科 (Klein-Nishina) 散射截面系数。
    
    参数：
    - E_keV (np.ndarray 或 float): 光子能量，单位为 keV。
    
    返回：
    - np.ndarray: 对应能量下的理论无量纲克莱因-仁科系数。
    """
    alpha = E_keV / 511.0
    term1 = 2.0 * (1.0 + alpha) ** 2 / (alpha ** 2 * (1.0 + 2.0 * alpha))
    term2 = (np.log(1.0 + 2.0 * alpha) / alpha) * (0.5 - (1.0 + alpha) / (alpha ** 2))
    term3 = (1.0 + 3.0 * alpha) / (1.0 + 2.0 * alpha) ** 2
    return term1 + term2 - term3

def calculate_apd(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能通道的灰度值计算：光电效应特征 apd = a_p x d
    
    参数：
    - low (np.ndarray): 低能通道的图像灰度值或像素数组。
    - high (np.ndarray): 高能通道的图像灰度值或像素数组。
    - I0_low (float): 默认值 204.293，代表低能通道未遮挡的入射背景参考灰度值。
    - I0_high (float): 默认值 204.199，代表高能通道未遮挡的入射背景参考灰度值。
    
    返回：
    - np.ndarray: 计算得到的像素级光电效应系数与其厚度的乘积 (apd)。
    """
    E_L = CONFIG['E_L']
    E_H = CONFIG['E_H']
    low = low.astype(float)
    high = high.astype(float)
    mu_L_d = np.log(I0_low / (low + 1e-6))
    mu_H_d = np.log(I0_high / (high + 1e-6))
    t1 = mu_L_d * _fkn(E_H) - mu_H_d * _fkn(E_L)
    t2 = _fkn(E_H) * (E_L ** -3) - _fkn(E_L) * (E_H ** -3)

    return t1 / t2

def calculate_acd(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能通道的灰度值计算：康普顿效应特征 acd = a_c x d
    
    参数：
    - low (np.ndarray): 低能通道的图像灰度值或像素数组。
    - high (np.ndarray): 高能通道的图像灰度值或像素数组。
    - I0_low (float): 默认值 204.293，代表低能通道未遮挡的入射背景参考灰度值。
    - I0_high (float): 默认值 204.199，代表高能通道未遮挡的入射背景参考灰度值。
    
    返回：
    - np.ndarray: 计算得到的像素级康普顿散射系数与其厚度的乘积 (acd)。
    """
    E_L = CONFIG['E_L']
    E_H = CONFIG['E_H']
    low = low.astype(float)
    high = high.astype(float)
    mu_L_d = np.log(I0_low / (low + 1e-6))
    mu_H_d = np.log(I0_high / (high + 1e-6))
    t1 = mu_H_d * (E_L ** -3) - mu_L_d * (E_H ** -3)
    t2 = _fkn(E_H) * (E_L ** -3) - _fkn(E_L) * (E_H ** -3) 
    return t1 / t2

def calculate_Zeff(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能通道的灰度值，采用 apd / mu_H_d 计算近似的有效原子序数 (Zeff)。
    
    参数：
    - low (np.ndarray): 低能通道的图像灰度值或像素数组。
    - high (np.ndarray): 高能通道的图像灰度值或像素数组。
    - I0_low (float): 默认值 204.293。
    - I0_high (float): 默认值 204.199。
    
    返回：
    - np.ndarray: 每个像素对应的近似有效原子序数特征 Zeff。
    """
    low = low.astype(float)
    high = high.astype(float)
    apd = calculate_apd(low, high, I0_low, I0_high)
    mu_H_d = np.log(I0_high / (high + 1e-6))
    return apd / mu_H_d

def calculate_Ze(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能通道的灰度值计算代数有效原子序数：Ze = k * (apd / acd) ** n
    
    参数：
    - low (np.ndarray): 低能通道的图像灰度值或像素数组。
    - high (np.ndarray): 高能通道的图像灰度值或像素数组。
    - I0_low (float): 默认值 204.293。
    - I0_high (float): 默认值 204.199。
    
    返回：
    - np.ndarray: 每个像素对应的代数有效原子序数特征 Ze。
    """
    low = low.astype(float)
    high = high.astype(float)
    n=1
    k=1
    apd = calculate_apd(low, high, I0_low, I0_high)
    acd = calculate_acd(low, high, I0_low, I0_high)
    return k*(apd / acd)**n

def calculate_mu_H_d(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能通道的灰度值计算高能通道的对数衰减厚度值：mu_H_d = log(I0_high / high)
    
    参数：
    - low (np.ndarray): 低能通道的图像灰度值或像素数组。
    - high (np.ndarray): 高能通道的图像灰度值或像素数组。
    - I0_low (float): 默认值 204.293。
    - I0_high (float): 默认值 204.199。
    
    返回：
    - np.ndarray: 每个像素对应的高能对数衰减厚度值 mu_H_d。
    """
    low = low.astype(float)
    high = high.astype(float)
    mu_H_d = np.log(I0_high / (high + 1e-6))
    return mu_H_d


def calculate_apd_acd_mono(T_L, T_H, E_L, E_H):
    """
    使用双单能近似代数公式计算 APD 和 ACD 特征。
    
    参数：
    - T_L (float 或 np.ndarray): 低能通道透射率或透射率数组。
    - T_H (float 或 np.ndarray): 高能通道透射率或透射率数组。
    - E_L (float): 低能等效能量，单位 keV。
    - E_H (float): 高能等效能量，单位 keV。
    
    返回：
    - tuple: (apd, acd) 浮点数或数组元组。
    """
    mu_L_d = -np.log(T_L + 1e-9)
    mu_H_d = -np.log(T_H + 1e-9)
    
    t1 = mu_L_d * _fkn(E_H) - mu_H_d * _fkn(E_L)
    t2 = _fkn(E_H) * (E_L ** -3) - _fkn(E_L) * (E_H ** -3)
    apd = t1 / t2
    
    t1_ac = mu_H_d * (E_L ** -3) - mu_L_d * (E_H ** -3)
    acd = t1_ac / t2
    return apd, acd


def solve_apd_acd_nonlinear(T_L, T_H, S_L, S_H, energies_keV):
    """
    使用 scipy.optimize.root 求解多色能谱积分前向透射方程，反解出 apd 和 acd。
    
    参数：
    - T_L (float 或 np.ndarray): 低能通道测量透射率或透射率数组。
    - T_H (float 或 np.ndarray): 高能通道测量透射率或透射率数组.
    - S_L (np.ndarray): 重建的归一化低能出射能谱概率向量。
    - S_H (np.ndarray): 重建的归一化高能出射能谱概率向量。
    - energies_keV (np.ndarray): 能量网格数组，单位 keV。
    
    返回：
    - tuple: 解耦能谱硬化漂移后的 (apd, acd) 特征元组。
    """
    fkn_vals = _fkn(energies_keV)
    E_cube_inv = energies_keV ** -3
    
    is_array = isinstance(T_L, np.ndarray) or hasattr(T_L, '__len__')
    
    if not is_array:
        def equations(vars_val):
            apd_val, acd_val = vars_val
            exp_term = np.exp(-(apd_val * E_cube_inv + acd_val * fkn_vals))
            pred_T_L = np.sum(S_L * exp_term)
            pred_T_H = np.sum(S_H * exp_term)
            return [pred_T_L - T_L, pred_T_H - T_H]
            
        E_L_est = np.sum(S_L * energies_keV)
        E_H_est = np.sum(S_H * energies_keV)
        apd_init, acd_init = calculate_apd_acd_mono(T_L, T_H, E_L_est, E_H_est)
        
        # Add bounds checking to initial guess to prevent extreme values causing overflow
        apd_init = np.clip(apd_init, 0.0, 50.0)
        acd_init = np.clip(acd_init, 0.0, 50.0)
        
        res = scipy.optimize.root(equations, [apd_init, acd_init], method='hybr')
        if res.success:
            return res.x[0], res.x[1]
        else:
            return apd_init, acd_init
    else:
        T_L_arr = np.array(T_L)
        T_H_arr = np.array(T_H)
        apd_res = []
        acd_res = []
        for tl, th in zip(T_L_arr, T_H_arr):
            ap, ac = solve_apd_acd_nonlinear(tl, th, S_L, S_H, energies_keV)
            apd_res.append(ap)
            acd_res.append(ac)
        return np.array(apd_res), np.array(acd_res)


def calibrate_sirz_coefficients(voltage_data: dict, step_index: int = 0) -> tuple:
    """
    根据标样直接通过对应厚度求解出的 ap 和 ac，结合理论电子密度与原子序数，使用最小二乘与对数线性回归校准估算系统常数 K1, g, nu。
    
    参数：
    - voltage_data (dict): 包含当前电压下所有材料 step 统计特征的字典。键为材料名（如 'Cu_step', 'Al_step', 'Fe_step'），值为包含 step 字典的列表。
      - 类型：dict
      - 含义：已从标样像素提取到的各阶段统计数据，必须包含 APD/ACD 特征及厚度等信息。
      - 用法：传入 _load_and_process_step_pixels 解算出的某一 voltage 与 filter 下的多材质汇总特征字典。
    - step_index (int): 默认为 0，指示直接使用第几个厚度阶梯（0-based，如 0 对应第 1 个阶梯，2 对应第 3 个阶梯，4 对应第 5 个阶梯）的直接求解系数进行系统校准。
      - 类型：int
      - 含义：用于提取 bulk_ap 和 bulk_ac 的特定阶梯的索引。
      - 用法：传入 0, 2, 4 等整数。
      
    返回：
    - tuple: (K1, g, nu) 浮点数三元组，分别代表：
      - K1 (float): 电子密度标定常数，用于方程 rho_e = K1 * acd / d。
      - g (float): 有效原子序数校准系数。
      - nu (float): 有效原子序数幂次系数，与 g 构成方程 Ze = g * (apd / acd) ** (1/nu)。
      如果标样数据缺失或不完整，则返回 (None, None, None)。
    """
    # 理论物理真值 (Ground Truth)
    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888}
    }
    
    # 检查是否包含所有三种标样的数据
    for name in THEORY_DATA.keys():
        if name not in voltage_data or not voltage_data[name]:
            return None, None, None
            
    bulk_ap = {}
    bulk_ac = {}
    
    # 提取指定阶梯的直接求解系数 ap (apd / thickness) 和 ac (acd / thickness) 作为 bulk 属性
    for name in THEORY_DATA.keys():
        stats = voltage_data[name]
        idx = min(step_index, len(stats) - 1)
        bulk_ap[name] = float(stats[idx]['ap_mean'])
        bulk_ac[name] = float(stats[idx]['ac_mean'])
        
    # 1. 拟合电子密度常数 K1: rho_e = K1 * a_c
    numerator_K = 0.0
    denominator_K = 0.0
    for name in THEORY_DATA.keys():
        numerator_K += THEORY_DATA[name]['rho_e'] * bulk_ac[name]
        denominator_K += bulk_ac[name]**2
    K1 = numerator_K / denominator_K
    
    # 2. 拟合有效原子序数常数 g 和 nu: ln(Z) = ln(g) + 1/nu * ln(ap / ac)
    x_coords = []
    y_coords = []
    for name in THEORY_DATA.keys():
        R_m = bulk_ap[name] / (bulk_ac[name] + 1e-6)
        # 对 R_m 实施物理非负安全边界保护，防止能谱硬化导致的负值在 np.log 中产生 nan 引起 SVD 最小二乘求解器崩溃
        R_m = max(1e-3, R_m)
        x_coords.append(np.log(R_m))
        y_coords.append(np.log(THEORY_DATA[name]['Z']))
        
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # 一元线性回归 (y = C1 * x + C0)
    C1, C0 = np.polyfit(x_coords, y_coords, 1)
    nu = 1.0 / C1
    g = np.exp(C0)
    
    return K1, g, nu


def compute_sirz_properties(ap, ac, d_mm, K1, g, nu) -> tuple:
    """
    根据给定的比吸收系数 ap、比康普顿系数 ac、厚度 d_mm 以及标定常数 K1, g, nu，
    计算有效电子密度 rho_e 与代数有效原子序数 Z_e。
    
    参数：
    - ap (np.ndarray 或 float): 像素或阶梯的比吸收特征 ap = apd / d_mm。
      - 类型：np.ndarray 或 float
      - 含义：与厚度无关的光电吸收强度系数。
      - 用法：传入计算好的比吸收系数数组或标量。
    - ac (np.ndarray 或 float): 像素或阶梯的比康普顿特征 ac = acd / d_mm。
      - 类型：np.ndarray 或 float
      - 含义：与厚度无关的康普顿散射强度系数。
      - 用法：传入计算好的比康普顿系数数组或标量。
    - d_mm (np.ndarray 或 float): 物理厚度，单位 mm。
      - 类型：np.ndarray 或 float
      - 含义：样品的实际几何厚度。
      - 用法：传入对应像素或阶梯的厚度。
    - K1 (float): 电子密度标定常数。
      - 类型：float
      - 含义：系统的电子密度标定常数。
      - 用法：传入标定常数浮点数。
    - g (float): 有效原子序数校准系数。
      - 类型：float
      - 含义：系统的有效原子序数标定系数。
      - 用法：传入标定常数浮点数。
    - nu (float): 有效原子序数幂次系数。
      - 类型：float
      - 含义：系统的有效原子序数幂次系数。
      - 用法：传入标定常数浮点数。
      
    返回：
    - tuple: (rho_e, Z_e)，其中：
      - rho_e (np.ndarray 或 float): 对应的有效电子密度特征。
      - Z_e (np.ndarray 或 float): 对应的有效原子序数特征。
    """
    if K1 is None or g is None or nu is None:
        if isinstance(ap, np.ndarray):
            return np.zeros_like(ap), np.zeros_like(ap)
        return 0.0, 0.0
        
    rho_e = K1 * ac
    
    # 物理保护防止 R_m 负值在幂运算中引发 NaN 或虚数
    r_m = ap / (ac + 1e-6)
    if isinstance(r_m, np.ndarray):
        r_m_safe = np.maximum(1e-3, r_m)
    else:
        r_m_safe = max(1e-3, r_m)
        
    Z_e = g * (r_m_safe) ** (1.0 / nu)
    return rho_e, Z_e


# ===================== 辅助计算与绘图子函数 (Modular Sub-functions) =====================

def _load_and_process_step_pixels(filepath: str, thickness_arr: np.ndarray, I0: float) -> list:
    """
    加载单个阶梯样品的像素数据，并计算各厚度阶梯下的像素级 APD, ACD, Zeff, Ze 特征及其统计量。
    
    参数：
    - filepath (str): .pkl 文件的绝对或相对路径。
      用法：传入有效的 pickle 文件路径字符串。
    - thickness_arr (np.ndarray): 包含该样品各阶梯实际物理厚度 (mm) 的一维数组。
      用法：传入对应的阶梯厚度序列，如 np.arange(2, 22, 2)。
    - I0 (float): 用于 logarithmic 衰减计算的入射参考灰度值。
      用法：16位图像通常传入 52428.0，8位图像通常传入 204.0。
      
    返回：
    - list: 包含每个步骤统计结果字典的列表，形如 [{'step_idx': 0, 'thickness_mm': 2.0, 'apd_mean': ...}, ...]。
    """
    import os
    import pickle
    import utils_II
    
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    l_list, h_list = d['pixels_low'], d['pixels_high']
    
    if len(l_list) == 0 or len(h_list) == 0:
        return []
        
    # 动态位深自适应评估 v_max
    v_max = 65535 if l_list[0].dtype == np.uint16 or np.max(l_list[0]) > 255 else 255
    lower_th = utils_II.get_ore_lower_threshold(False, v_max)
    
    step_stats = []
    
    for s in range(len(thickness_arr)):
        if s >= len(l_list) or s >= len(h_list):
            break
        l_v = l_list[s]
        h_v = h_list[s]
        
        # 剔除异常点与盲元像素以保证物理合理性
        mask = (l_v > 0) & (h_v > 0) & (l_v < v_max) & (h_v < v_max)
        l_val = l_v[mask].astype(float)
        h_val = h_v[mask].astype(float)
        
        if len(l_val) > 0:
            # 像素级物理特征映射
            apd_arr = calculate_apd(l_val, h_val, I0_low=I0, I0_high=I0)
            acd_arr = calculate_acd(l_val, h_val, I0_low=I0, I0_high=I0)
            Zeff_arr = calculate_Zeff(l_val, h_val, I0_low=I0, I0_high=I0)
            Ze_arr = calculate_Ze(l_val, h_val, I0_low=I0, I0_high=I0)
            
            # 清除计算过程中产生的非有限数值
            valid_idx = np.isfinite(apd_arr) & np.isfinite(acd_arr) & np.isfinite(Zeff_arr) & np.isfinite(Ze_arr)
            apd_arr = apd_arr[valid_idx]
            acd_arr = acd_arr[valid_idx]
            Zeff_arr = Zeff_arr[valid_idx]
            Ze_arr = Ze_arr[valid_idx]
            
            if len(apd_arr) > 0:
                step_thickness = float(thickness_arr[s])
                # 计算物理厚度相关的比吸收系数 (单位 mm^-1)
                ap_arr = apd_arr / step_thickness
                ac_arr = acd_arr / step_thickness
                ap_ac_arr = apd_arr / (acd_arr + 1e-6)
                
                stat = {
                    'step_idx': s,
                    'thickness_mm': step_thickness,
                    'apd_mean': float(np.mean(apd_arr)),
                    'apd_std': float(np.std(apd_arr)),
                    'acd_mean': float(np.mean(acd_arr)),
                    'acd_std': float(np.std(acd_arr)),
                    'ap_mean': float(np.mean(ap_arr)),
                    'ap_std': float(np.std(ap_arr)),
                    'ac_mean': float(np.mean(ac_arr)),
                    'ac_std': float(np.std(ac_arr)),
                    'Zeff_mean': float(np.mean(Zeff_arr)),
                    'Zeff_std': float(np.std(Zeff_arr)),
                    'Ze_mean': float(np.mean(Ze_arr)),
                    'Ze_std': float(np.std(Ze_arr)),
                    'ap_ac_mean': float(np.mean(ap_ac_arr)),
                    'ap_ac_std': float(np.std(ap_ac_arr)),
                    # 保存原始像素级物理特征数组，用于绘制高精度原始分布点云图
                    'apd_raw': apd_arr,
                    'acd_raw': acd_arr,
                    'Zeff_raw': Zeff_arr,
                    'Ze_raw': Ze_arr
                }
                step_stats.append(stat)
                
    return step_stats


def _plot_detailed_profiling(voltage_data: dict, f_type: str, voltage: str, colors: dict, save_path: str) -> None:
    """
    为指定电压和滤片配置下的所有材料绘制 detailed 2x2 物理剖析曲线（含 APD, ACD, Trajectory, Zeff）。
    
    参数：
    - voltage_data (dict): 包含当前电压下所有材料 step 统计特征的字典。
      用法：键为材料名（如 'Cu_step'），值为 `_load_and_process_step_pixels` 的返回列表。
    - f_type (str): 当前滤片厚度描述（如 '0.6mm', '1.2mm'）。
      用法：传入滤片厚度识别字符串。
    - voltage (str): 当前管电压描述（如 '200kV'）。
      用法：传入电压识别字符串。
    - colors (dict): 用于图线着色的材料色彩配置映射字典。
      用法：传入包含材料颜色值映射的字典。
    - save_path (str): 图像文件的保存绝对或相对路径。
      用法：传入目标落地文件路径字符串。
    """
    import os
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Step Sample APD/ACD Physical Profiling - {f_type} at {voltage}", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. apd (光电特征厚度) vs Thickness
    ax1 = axes[0, 0]
    ax1.set_title(r"Photoelectric Feature $a_p \cdot d$ vs Physical Thickness", fontsize=12, pad=10)
    ax1.set_xlabel("Thickness $d$ (mm)"); ax1.set_ylabel(r"$apd$")
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. acd (康普顿特征厚度) vs Thickness
    ax2 = axes[0, 1]
    ax2.set_title(r"Compton Feature $a_c \cdot d$ vs Physical Thickness", fontsize=12, pad=10)
    ax2.set_xlabel("Thickness $d$ (mm)"); ax2.set_ylabel(r"$acd$")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. apd vs acd 特征空间轨迹
    ax3 = axes[1, 0]
    ax3.set_title(r"$apd$ vs $acd$ Feature Material Trajectory", fontsize=12, pad=10)
    ax3.set_xlabel(r"Compton thickness $acd$"); ax3.set_ylabel(r"Photoelectric thickness $apd$")
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Ze vs Thickness (直接指示能谱硬化漂移)
    ax4 = axes[1, 1]
    ax4.set_title(r"Calibrated Effective Atomic Number $Z_{e}$ vs Physical Thickness", fontsize=12, pad=10)
    ax4.set_xlabel("Thickness $d$ (mm)"); ax4.set_ylabel(r"$Z_{e}$")
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888}
    }
    
    for name, stats in voltage_data.items():
        t_vals = [s['thickness_mm'] for s in stats]
        apd_means = [s['apd_mean'] for s in stats]
        apd_stds = [s['apd_std'] for s in stats]
        acd_means = [s['acd_mean'] for s in stats]
        acd_stds = [s['acd_std'] for s in stats]
        ze_means = [s['Ze_mean'] for s in stats]
        ze_stds = [s['Ze_std'] for s in stats]
        
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        # Subplot 1: APD vs thickness
        ax1.errorbar(t_vals, apd_means, yerr=apd_stds, fmt='o-', color=c, label=f"{lbl} Meas", capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        if len(t_vals) > 1:
            slope = np.sum(np.array(t_vals) * np.array(apd_means)) / np.sum(np.array(t_vals)**2)
            ax1.plot(t_vals, slope * np.array(t_vals), '--', color=c, alpha=0.5, label=f"{lbl} Fit (slope={slope:.4f})")
        
        # Subplot 2: ACD vs thickness
        ax2.errorbar(t_vals, acd_means, yerr=acd_stds, fmt='o-', color=c, label=f"{lbl} Meas", capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        if len(t_vals) > 1:
            slope = np.sum(np.array(t_vals) * np.array(acd_means)) / np.sum(np.array(t_vals)**2)
            ax2.plot(t_vals, slope * np.array(t_vals), '--', color=c, alpha=0.5, label=f"{lbl} Fit (slope={slope:.4f})")
        
        # Subplot 3: APD vs ACD Trajectory
        ax3.errorbar(acd_means, apd_means, xerr=acd_stds, yerr=apd_stds, fmt='o-', color=c, label=lbl, capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        
        # Subplot 4: Ze vs thickness
        ax4.errorbar(t_vals, ze_means, yerr=ze_stds, fmt='o-', color=c, label=f"{lbl} Meas", capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        theo_z = THEORY_DATA.get(name, {}).get('Z', None)
        if theo_z is not None:
            ax4.axhline(y=theo_z, color=c, linestyle=':', alpha=0.7, label=f"{lbl} Theory ({theo_z})")
        
    ax1.legend(fontsize='x-small'); ax2.legend(fontsize='x-small')
    ax3.legend(fontsize='x-small'); ax4.legend(fontsize='x-small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_calibration_fit(voltage_data: dict, K1: float, g: float, nu: float, step_index: int, f_type: str, voltage: str, save_path: str) -> None:
    """
    绘制并保存对数线性拟合关系图 (ln(ap/ac) vs ln(Z))。
    
    参数解释：
    - voltage_data (dict): 包含当前电压下所有材料 step 数据的字典。
      类型：dict
      用法：传入对应 voltage 和 filter 下的多材质汇总特征字典。
    - K1 (float): 电子密度标定常数。
      类型：float
      用法：传入标定好的电子密度系数。
    - g (float): 有效原子序数校准系数。
      类型：float
      用法：传入标定好的原子序数系数。
    - nu (float): 有效原子序数幂次系数。
      类型：float
      用法：传入标定好的幂次系数。
    - step_index (int): 用于标定的阶梯索引。
      类型：int
      用法：例如 0 对应第一阶梯。
    - f_type (str): 滤片厚度配置描述字符串。
      类型：str
      用法：如 '0.6mm'。
    - voltage (str): 管电压描述字符串。
      类型：str
      用法：如 '200kV'。
    - save_path (str): 图像文件的保存磁盘路径。
      类型：str
      用法：指定保存的绝对或相对路径。
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888}
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"SIRZ Z-Calibration Fit - {f_type} at {voltage}\n(g={g:.4f}, $\\nu$={nu:.4f})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("$\\ln(a_p / a_c)$", fontsize=10)
    ax.set_ylabel("$\\ln(Z)$", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    x_points = []
    y_points = []
    
    for name in THEORY_DATA.keys():
        if name in voltage_data and len(voltage_data[name]) > step_index:
            stats = voltage_data[name]
            idx = min(step_index, len(stats) - 1)
            ap = float(stats[idx]['ap_mean'])
            ac = float(stats[idx]['ac_mean'])
            z_val = THEORY_DATA[name]['Z']
            
            r_m = ap / (ac + 1e-6)
            r_m = max(1e-3, r_m)
            
            x_val = np.log(r_m)
            y_val = np.log(z_val)
            
            x_points.append(x_val)
            y_points.append(y_val)
            
            # 绘制散点
            ax.scatter(x_val, y_val, s=80, marker='o', label=f"{name.replace('_step', '')} (Z={z_val})")
            ax.annotate(f"  {name.replace('_step', '')}", (x_val, y_val), fontsize=9, fontweight='semibold')

    if len(x_points) > 0:
        # 绘制拟合直线
        x_min, x_max = min(x_points) - 0.5, max(x_points) + 0.5
        x_line = np.linspace(x_min, x_max, 100)
        y_line = (1.0 / nu) * x_line + np.log(g)
        ax.plot(x_line, y_line, 'k--', label="Fitted Line: $\\ln(Z) = \\ln(g) + \\frac{1}{\\nu}\\ln(a_p/a_c)$")
        
    ax.legend(fontsize='small', loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_apd_acd_histograms(voltage_data: dict, f_type: str, voltage: str, colors: dict, save_path: str) -> None:
    """
    为指定电压和滤片配置下的所有材料绘制 APD 和 ACD 的像素级原始分布直方图。
    
    参数解释：
    - voltage_data (dict): 包含当前电压下所有材料 step 数据的字典（包含像素级 raw 数组）。
      类型：dict (键为材料名，值为包含 step 特征字典的列表)
      用法：直接传递由 _load_and_process_step_pixels 生成的数据结构。
    - f_type (str): 当前滤片配置描述（如 '0.6mm', '1.2mm'）。
      类型：str
      用法：用于直方图大标题的滤片厚度标识。
    - voltage (str): 当前管电压描述（如 '200kV'）。
      类型：str
      用法：用于直方图大标题的电压标识。
    - colors (dict): 用于直方图着色的材料色彩配置映射字典。
      类型：dict (键为材料名，值为颜色十六进制字符串)
      用法：传递全局 colors 配置，如 {'Al_step': '#4A90E2', ...}。
    - save_path (str): 图像文件的保存绝对或相对路径。
      类型：str
      用法：指定直方图大图最终落地的磁盘路径。
    """
    import os
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Step Sample APD/ACD Pixel Grayscale Distributions - {f_type} at {voltage}", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. apd 像素级分布直方图
    ax1 = axes[0]
    ax1.set_title(r"Photoelectric Feature $apd = a_p \cdot d$ Histogram", fontsize=12, pad=10)
    ax1.set_xlabel(r"Pixel $apd$ value")
    ax1.set_ylabel("Pixel Count")
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. acd 像素级分布直方图
    ax2 = axes[1]
    ax2.set_title(r"Compton Feature $acd = a_c \cdot d$ Histogram", fontsize=12, pad=10)
    ax2.set_xlabel(r"Pixel $acd$ value")
    ax2.set_ylabel("Pixel Count")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for name, stats in voltage_data.items():
        if not stats:
            continue
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        # 收集该材料在所有阶梯下的像素值
        apd_all = np.concatenate([s['apd_raw'] for s in stats if 'apd_raw' in s])
        acd_all = np.concatenate([s['acd_raw'] for s in stats if 'acd_raw' in s])
        
        if len(apd_all) > 0:
            # 绘制 apd 直方图
            ax1.hist(apd_all, bins=100, color=c, alpha=0.5, label=f"{lbl} ({len(apd_all)} px)", histtype='stepfilled', edgecolor=c, linewidth=1.2)
        if len(acd_all) > 0:
            # 绘制 acd 直方图
            ax2.hist(acd_all, bins=100, color=c, alpha=0.5, label=f"{lbl} ({len(acd_all)} px)", histtype='stepfilled', edgecolor=c, linewidth=1.2)
            
    ax1.legend(fontsize='small')
    ax2.legend(fontsize='small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_coefficient_dependence(coeff_summary: dict, f_type: str, colors: dict, save_path: str, step_label: str = "Step 1") -> None:
    """
    绘制给定滤片厚度下，Bulk 物理系数（Photoelectric, Compton, Zeff, Ratio）随 X 射线管电压变化的曲线图，并在子图标题中标出 Z_e 与 rho_e 的公式。
    
    参数：
    - coeff_summary (dict): 包含汇总后各管电压点物理系数均值的字典。
      用法：键包含 'voltage' 及各材料名，字典数据按电压对应。
    - f_type (str): 当前滤片配置描述（如 '0.6mm', '1.2mm'）。
      用法：字符串，用于标题标注。
    - colors (dict): 用于图线着色的材料色彩配置映射字典。
      用法：形如 {'Al_step': '#4A90E2', ...} 的色彩字典。
    - save_path (str): 汇总依赖关系曲线大图的保存路径。
      用法：路径字符串，指示图片最终落地位置。
    - step_label (str): 默认为 "Step 1"，指示当前绘图对应的厚度阶梯标签。
      用法：传入 "Step 1"、"Step 3"、"Step 5" 等描述字符串。
    """
    import os
    import matplotlib.pyplot as plt
    
    v_arr = np.array(coeff_summary['voltage'])
    if len(v_arr) == 0:
        return
        
    sort_idx = np.argsort(v_arr)
    v_sorted = v_arr[sort_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Coefficient Energy Dependence - {step_label} (Filter: {f_type})", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. a_p 随电压变化
    ax1 = axes[0, 0]
    ax1.set_title(r"Bulk Photoelectric Coefficient $a_p$ vs Voltage", fontsize=12, pad=10)
    ax1.set_xlabel("Voltage (kV)"); ax1.set_ylabel(r"$a_p \ (\mathrm{mm}^{-1}\mathrm{keV}^3)$")
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. a_c 随电压变化
    ax2 = axes[0, 1]
    ax2.set_title(r"Bulk Compton Coefficient $a_c$ vs Voltage", fontsize=12, pad=10)
    ax2.set_xlabel("Voltage (kV)"); ax2.set_ylabel(r"$a_c \ (\mathrm{mm}^{-1})$")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Ze 随电压变化
    ax3 = axes[1, 0]
    ax3.set_title(r"Bulk Calibrated $Z_e = g \cdot (a_p / a_c)^{1/\nu}$ vs Voltage", fontsize=12, pad=10)
    ax3.set_xlabel("Voltage (kV)"); ax3.set_ylabel(r"$Z_e$")
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. rho_e 随电压变化
    ax4 = axes[1, 1]
    ax4.set_title(r"Bulk Calibrated $\rho_e = K_1 \cdot a_c$ vs Voltage", fontsize=12, pad=10)
    ax4.set_xlabel("Voltage (kV)"); ax4.set_ylabel(r"$\rho_e \ (\mathrm{moles-e}^-/\mathrm{cm}^3)$")
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888}
    }
    
    for name in ['Al_step', 'Fe_step', 'Cu_step']:
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        ap_vals = np.array(coeff_summary[name]['a_p_mean'])[sort_idx]
        ac_vals = np.array(coeff_summary[name]['a_c_mean'])[sort_idx]
        ze_vals = np.array(coeff_summary[name]['Ze_mean'])[sort_idx]
        rho_e_vals = np.array(coeff_summary[name]['rho_e_mean'])[sort_idx]
        
        valid = [x is not None for x in ap_vals]
        if any(valid):
            v_v = v_sorted[valid]
            ax1.plot(v_v, ap_vals[valid], 'o-', color=c, label=f"{lbl} Meas", linewidth=2, markersize=6)
            ax2.plot(v_v, ac_vals[valid], 'o-', color=c, label=f"{lbl} Meas", linewidth=2, markersize=6)
            ax3.plot(v_v, ze_vals[valid], 'o-', color=c, label=f"{lbl} Meas", linewidth=2, markersize=6)
            ax4.plot(v_v, rho_e_vals[valid], 'o-', color=c, label=f"{lbl} Meas", linewidth=2, markersize=6)
            
            # 绘制理论值参考水平线
            theo_z = THEORY_DATA[name]['Z']
            theo_rho = THEORY_DATA[name]['rho_e']
            ax3.axhline(y=theo_z, color=c, linestyle=':', alpha=0.7, label=f"{lbl} Theory ({theo_z})")
            ax4.axhline(y=theo_rho, color=c, linestyle=':', alpha=0.7, label=f"{lbl} Theory ({theo_rho:.4f})")
            
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_combined_coefficient_dependence(coeff_summary_06: dict, coeff_summary_12: dict, colors: dict, save_path: str, step_label: str = "Step 1") -> None:
    r"""
    绘制并合 0.6mm 与 1.2mm 滤片厚度下，M1 方法的 Bulk 物理系数（Photoelectric, Compton, Zeff, Ratio）随 X 射线管电压变化的对比曲线图。
    
    参数：
    - coeff_summary_06 (dict): 包含 0.6mm 滤片下汇总后各管电压点物理系数均值的字典。
      - 类型：dict
      - 含义：0.6mm 滤片下的系数统计汇总字典，键包含 'voltage' 及 'Al_step', 'Fe_step', 'Cu_step'。
      - 用法：传入对应 0.6mm 滤片与指定 step 的 coeff_summary 结构。
    - coeff_summary_12 (dict): 包含 1.2mm 滤片下汇总后各管电压点物理系数均值的字典。
      - 类型：dict
      - 含义：1.2mm 滤片下的系数统计汇总字典，结构与 coeff_summary_06 一致。
      - 用法：传入对应 1.2mm 滤片与指定 step 的 coeff_summary 结构。
    - colors (dict): 用于图线着色的材料色彩配置映射字典。
      - 类型：dict
      - 含义：配置 'Al_step', 'Fe_step', 'Cu_step' 色彩的字典，例如 {'Al_step': '#4A90E2', ...}。
      - 用法：传递全局配色字典。
    - save_path (str): 并合后的依赖关系曲线大图的物理磁盘保存路径。
      - 类型：str
      - 含义：图片保存路径。
      - 用法：例如 'results/.../summary_coefficients_M1_combined_step1.png'。
    - step_label (str): 默认为 "Step 1"，指示当前绘图对应的厚度阶梯标签。
      - 类型：str
      - 含义：厚度级标签描述，如 "Step 1 (M1)"。
      - 用法：传入绘图标题中展示的说明文字。
      
    返回：
    - None (图片直接保存至磁盘)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    v_arr_06 = np.array(coeff_summary_06['voltage']) if (coeff_summary_06 and 'voltage' in coeff_summary_06) else np.array([])
    v_arr_12 = np.array(coeff_summary_12['voltage']) if (coeff_summary_12 and 'voltage' in coeff_summary_12) else np.array([])
    
    if len(v_arr_06) == 0 and len(v_arr_12) == 0:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Coefficient Energy Dependence - {step_label} (Combined Filters)", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. a_p 随电压变化
    ax1 = axes[0, 0]
    ax1.set_title(r"Bulk Photoelectric Coefficient $a_p$ vs Voltage", fontsize=12, pad=10)
    ax1.set_xlabel("Voltage (kV)"); ax1.set_ylabel(r"$a_p \ (\mathrm{mm}^{-1}\mathrm{keV}^3)$")
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. a_c 随电压变化
    ax2 = axes[0, 1]
    ax2.set_title(r"Bulk Compton Coefficient $a_c$ vs Voltage", fontsize=12, pad=10)
    ax2.set_xlabel("Voltage (kV)"); ax2.set_ylabel(r"$a_c \ (\mathrm{mm}^{-1})$")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Ze 随电压变化
    ax3 = axes[1, 0]
    ax3.set_title(r"Bulk Calibrated $Z_e = g \cdot (a_p / a_c)^{1/\nu}$ vs Voltage", fontsize=12, pad=10)
    ax3.set_xlabel("Voltage (kV)"); ax3.set_ylabel(r"$Z_e$")
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. rho_e 随电压变化
    ax4 = axes[1, 1]
    ax4.set_title(r"Bulk Calibrated $\rho_e = K_1 \cdot a_c$ vs Voltage", fontsize=12, pad=10)
    ax4.set_xlabel("Voltage (kV)"); ax4.set_ylabel(r"$\rho_e \ (\mathrm{moles-e}^-/\mathrm{cm}^3)$")
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888}
    }
    
    for name in ['Al_step', 'Fe_step', 'Cu_step']:
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        # 1. Plot 0.6mm (Solid lines with circle markers)
        if len(v_arr_06) > 0:
            sort_idx_06 = np.argsort(v_arr_06)
            v_sorted_06 = v_arr_06[sort_idx_06]
            ap_vals_06 = np.array(coeff_summary_06[name]['a_p_mean'])[sort_idx_06]
            ac_vals_06 = np.array(coeff_summary_06[name]['a_c_mean'])[sort_idx_06]
            ze_vals_06 = np.array(coeff_summary_06[name]['Ze_mean'])[sort_idx_06]
            rho_e_vals_06 = np.array(coeff_summary_06[name]['rho_e_mean'])[sort_idx_06]
            
            valid_06 = [x is not None for x in ap_vals_06]
            if any(valid_06):
                v_v_06 = v_sorted_06[valid_06]
                ax1.plot(v_v_06, ap_vals_06[valid_06], 'o-', color=c, label=f"{lbl} 0.6mm", linewidth=2, markersize=6)
                ax2.plot(v_v_06, ac_vals_06[valid_06], 'o-', color=c, label=f"{lbl} 0.6mm", linewidth=2, markersize=6)
                ax3.plot(v_v_06, ze_vals_06[valid_06], 'o-', color=c, label=f"{lbl} 0.6mm", linewidth=2, markersize=6)
                ax4.plot(v_v_06, rho_e_vals_06[valid_06], 'o-', color=c, label=f"{lbl} 0.6mm", linewidth=2, markersize=6)
                
        # 2. Plot 1.2mm (Dashed lines with triangle markers)
        if len(v_arr_12) > 0:
            sort_idx_12 = np.argsort(v_arr_12)
            v_sorted_12 = v_arr_12[sort_idx_12]
            ap_vals_12 = np.array(coeff_summary_12[name]['a_p_mean'])[sort_idx_12]
            ac_vals_12 = np.array(coeff_summary_12[name]['a_c_mean'])[sort_idx_12]
            ze_vals_12 = np.array(coeff_summary_12[name]['Ze_mean'])[sort_idx_12]
            rho_e_vals_12 = np.array(coeff_summary_12[name]['rho_e_mean'])[sort_idx_12]
            
            valid_12 = [x is not None for x in ap_vals_12]
            if any(valid_12):
                v_v_12 = v_sorted_12[valid_12]
                ax1.plot(v_v_12, ap_vals_12[valid_12], '^--', color=c, label=f"{lbl} 1.2mm", linewidth=1.5, markersize=6)
                ax2.plot(v_v_12, ac_vals_12[valid_12], '^--', color=c, label=f"{lbl} 1.2mm", linewidth=1.5, markersize=6)
                ax3.plot(v_v_12, ze_vals_12[valid_12], '^--', color=c, label=f"{lbl} 1.2mm", linewidth=1.5, markersize=6)
                ax4.plot(v_v_12, rho_e_vals_12[valid_12], '^--', color=c, label=f"{lbl} 1.2mm", linewidth=1.5, markersize=6)
                
        # 绘制理论参考线 (Ze 和 rho_e)
        z_theo = THEORY_DATA[name]['Z']
        rho_theo = THEORY_DATA[name]['rho_e']
        ax3.axhline(z_theo, color=c, linestyle=':', alpha=0.5)
        ax4.axhline(rho_theo, color=c, linestyle=':', alpha=0.5)
        
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(fontsize='small', loc='best')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _load_step_pixels(filepath: str, thickness_arr: np.ndarray, I0: float) -> list:
    """
    加载单个阶梯样品的像素数据，并只过滤出有效且非饱和的低能/高能原始像素对。
    
    参数：
    - filepath (str): .pkl 阶梯样品像素序列路径。
    - thickness_arr (np.ndarray): 阶梯厚度 (mm) 数组。
    - I0 (float): 背景对数灰度参考值。
    
    返回：
    - list: 包含各阶梯有效像素的一维数组字典列表。
    """
    import pickle
    import os
    
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    l_list, h_list = d['pixels_low'], d['pixels_high']
    if len(l_list) == 0 or len(h_list) == 0:
        return []
        
    v_max = 65535 if l_list[0].dtype == np.uint16 or np.max(l_list[0]) > 255 else 255
    steps_data = []
    
    for s in range(len(thickness_arr)):
        if s >= len(l_list) or s >= len(h_list):
            break
        l_v = l_list[s]
        h_v = h_list[s]
        # 剔除异常点与盲元像素以保证物理合理性
        mask = (l_v > 0) & (h_v > 0) & (l_v < v_max) & (h_v < v_max)
        l_val = l_v[mask].astype(float)
        h_val = h_v[mask].astype(float)
        
        if len(l_val) > 0:
            steps_data.append({
                'step_idx': s,
                'thickness_mm': float(thickness_arr[s]),
                'l_val': l_val,
                'h_val': h_val
            })
    return steps_data


def run_step_apd_acd_analysis(include_0331=True, plot_details=True, output_dir='results/thickness_decoupling/apd_acd_analysis'):
    """
    核心分析调度主函数。
    读取能谱反推 JSON，以及陈文反演能谱 CSV，针对 Static (58/105 keV)、Dyn (动态等效单能)、M1 (连续能谱积分) 以及 ChenWen (陈文反演能谱积分) 四种方法，
    并行计算其像素级与阶梯级 APD/ACD 特征、执行 SIRZ 校准、重构有效原子序数 Ze 和电子密度 rho_e，
    生成多方法线性对比图、各算法深度分析大图及 combined summaries 汇总依赖大图。
    
    参数：
    - include_0331 (bool):
      类型：bool
      含义：是否在 0.6mm 分析中并入历史 0331 数据集。
      用法：传入 True 或 False。
    - plot_details (bool):
      类型：bool
      含义：是否绘制详细物理特征、轨迹图、拟合图及直方图。
      用法：传入 True 或 False。
    - output_dir (str):
      类型：str
      含义：图表及 JSON 保存的物理相对/绝对目录路径。
      用法：例如 'results/thickness_decoupling/apd_acd_analysis'。
    """
    import os
    import re
    import json
    import matplotlib.pyplot as plt
    import gc
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir_0429 = os.path.join(script_dir, 'results/20260429_mask_generated_16bit')
    input_dir_0331 = os.path.join(script_dir, 'results/20260331_16bit')
    abs_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # voltages_0429 = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
    # voltages_0331 = ['140kV', '160kV', '180kV']
    voltages_0429 = ['200kV', '280kV']
    voltages_0331 = []
    filter_types = ['0.6mm', '1.2mm']
    
    I0_0429 = 52428.0
    I0_0331 = 52428.0
    
    step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
    thicknesses_0429 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
    thicknesses_0331 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
    
    # 配色统一
    colors = {
        'Al_step': '#4A90E2',  # 现代天空蓝
        'Fe_step': '#2ECCA7',  # 优雅孔雀绿
        'Cu_step': '#E28743'   # 经典黄铜橙
    }
    
    # 尝试加载重构能谱结果以支持 Dyn 和 M1 连续积分计算
    spectra_path = os.path.join(script_dir, 'results/thickness_decoupling/energy_hardening/spectrum_reconstruction/CuFe_4steps/reconstructed_spectra_summary.json')
    reconstructed_spectra = {}
    if os.path.exists(spectra_path):
        try:
            with open(spectra_path, 'r', encoding='utf-8') as f:
                reconstructed_spectra = json.load(f)
            print(f"[+] Successfully loaded reconstructed spectra from {spectra_path}")
        except Exception as e:
            print(f"[-] Failed to load reconstructed spectra: {e}")
    else:
        print(f"[-] Reconstructed spectra JSON not found at {spectra_path}. Dyn and M1 methods will be skipped.")
        
    global_results = {}
    
    # 声明多方法的 coeff_summaries 汇总依赖数据结构 (外层 filter 映射)
    coeff_summaries = {
        f_type: {
            method: {
                step_num: {
                    'voltage': [],
                    'Al_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                    'Fe_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                    'Cu_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []}
                } for step_num in [1, 3, 5]
            } for method in ['Static', 'Dyn', 'M1', 'ChenWen']
        } for f_type in filter_types
    }
    
    for f_type in filter_types:
        global_results[f_type] = {}
        cur_voltages = voltages_0331 + voltages_0429 if (f_type == '0.6mm' and include_0331) else voltages_0429
        
        for voltage in cur_voltages:
            is_0331 = voltage in voltages_0331
            cur_input_dir = input_dir_0331 if is_0331 else input_dir_0429
            cur_I0 = I0_0331 if is_0331 else I0_0429
            cur_thick_map = thicknesses_0331 if is_0331 else thicknesses_0429
            v_int = int(re.search(r'(\d+)', voltage).group(1))
            
            # 读取当前电压和滤片下的各阶梯原始像素值
            raw_voltage_pixels = {}
            has_pixels = False
            for idx, name in step_mats.items():
                if is_0331:
                    p = f'{cur_input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
                else:
                    p = f'{cur_input_dir}/pixel_values/{name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl'
                
                steps_data = _load_step_pixels(p, cur_thick_map[name], cur_I0)
                if steps_data:
                    raw_voltage_pixels[name] = steps_data
                    has_pixels = True
                    
            if not has_pixels:
                continue
                
            # 获取能谱重构参数
            spectrum_info = None
            if f_type in reconstructed_spectra and voltage in reconstructed_spectra[f_type]:
                spectrum_info = reconstructed_spectra[f_type][voltage]
                
            global_results[f_type][voltage] = {}
            voltage_data_methods = {}
            
            for method in ['Static', 'Dyn', 'M1', 'ChenWen']:
                # 只有成功加载能谱参数后才能运行 Dyn 和 M1
                if method in ['Dyn', 'M1'] and spectrum_info is None:
                    continue
                
                cw_S_L = None
                cw_S_H = None
                cw_energies = None
                if method == 'ChenWen':
                    chenwen_csv_path = os.path.join(script_dir, 'chenwen/invsp_results', voltage, 'reconstructed_spectra.csv')
                    if not os.path.exists(chenwen_csv_path):
                        continue
                    import pandas as pd
                    try:
                        cw_df = pd.read_csv(chenwen_csv_path)
                        cw_energies = cw_df['energy_keV'].values
                        cw_S_L = cw_df['low'].values
                        if f_type == '0.6mm':
                            cw_S_H = cw_df['high_0.6mm'].values
                        else:
                            cw_S_H = cw_df['high_1.2mm'].values
                        cw_S_L = cw_S_L / max(np.sum(cw_S_L), 1e-12)
                        cw_S_H = cw_S_H / max(np.sum(cw_S_H), 1e-12)
                    except Exception as e:
                        print(f"[-] Error loading ChenWen spectrum at {voltage}: {e}")
                        continue
                    
                voltage_data = {}
                for name in step_mats.values():
                    if name not in raw_voltage_pixels:
                        continue
                        
                    stats_list = []
                    for step_data in raw_voltage_pixels[name]:
                        s = step_data['step_idx']
                        t_mm = step_data['thickness_mm']
                        l_val = step_data['l_val']
                        h_val = step_data['h_val']
                        
                        T_L = l_val / cur_I0
                        T_H = h_val / cur_I0
                        
                        # 求解各算法对应的像素级 APD/ACD 特征
                        if method == 'Static':
                            apd_arr, acd_arr = calculate_apd_acd_mono(T_L, T_H, E_L=58.0, E_H=105.0)
                        elif method == 'Dyn':
                            E_L_eff = spectrum_info['E_L_eff_keV']
                            E_H_eff = spectrum_info['E_H_eff_keV']
                            apd_arr, acd_arr = calculate_apd_acd_mono(T_L, T_H, E_L=E_L_eff, E_H=E_H_eff)
                        elif method == 'M1':
                            S_L = np.array(spectrum_info['S_L'])
                            S_H = np.array(spectrum_info['S_H'])
                            energies = np.array(spectrum_info['energies_keV'])
                            apd_arr, acd_arr = solve_apd_acd_nonlinear(T_L, T_H, S_L, S_H, energies)
                        elif method == 'ChenWen':
                            apd_arr, acd_arr = solve_apd_acd_nonlinear(T_L, T_H, cw_S_L, cw_S_H, cw_energies)
                            
                        # 剔除无效计算的非有限数据
                        valid_idx = np.isfinite(apd_arr) & np.isfinite(acd_arr)
                        apd_arr = apd_arr[valid_idx]
                        acd_arr = acd_arr[valid_idx]
                        
                        if len(apd_arr) > 0:
                            ap_arr = apd_arr / t_mm
                            ac_arr = acd_arr / t_mm
                            ap_ac_arr = apd_arr / (acd_arr + 1e-6)
                            
                            stats_list.append({
                                'step_idx': s,
                                'thickness_mm': t_mm,
                                'apd_mean': float(np.mean(apd_arr)),
                                'apd_std': float(np.std(apd_arr)),
                                'acd_mean': float(np.mean(acd_arr)),
                                'acd_std': float(np.std(acd_arr)),
                                'ap_mean': float(np.mean(ap_arr)),
                                'ap_std': float(np.std(ap_arr)),
                                'ac_mean': float(np.mean(ac_arr)),
                                'ac_std': float(np.std(ac_arr)),
                                'ap_ac_mean': float(np.mean(ap_ac_arr)),
                                'ap_ac_std': float(np.std(ap_ac_arr)),
                                'apd_raw': apd_arr,
                                'acd_raw': acd_arr,
                                'Zeff_mean': 0.0,
                                'Zeff_std': 0.0,
                                'Ze_mean': 0.0,
                                'Ze_std': 0.0
                            })
                    if stats_list:
                        voltage_data[name] = stats_list
                        
                if not voltage_data:
                    continue
                    
                # 运行系统常数标定 (K1, g, nu)
                calib_results = {}
                for s_idx in [0, 2, 4]:
                    K1_s, g_s, nu_s = calibrate_sirz_coefficients(voltage_data, step_index=s_idx)
                    if K1_s is not None:
                        calib_results[s_idx] = (K1_s, g_s, nu_s)
                        if method in ['M1', 'ChenWen'] and s_idx == 0:
                            print(f"  [+] {method} Spectrum Calibration ({f_type}, {voltage}): K1 = {K1_s:.4f}, g = {g_s:.4f}, nu = {nu_s:.4f}")
                            
                # 应用第一阶梯标定值计算 Ze 和 rho_e
                K1, g, nu = calib_results.get(0, (None, None, None))
                if K1 is not None:
                    for name in step_mats.values():
                        if name in voltage_data:
                            for s in voltage_data[name]:
                                apd_raw = s['apd_raw']
                                acd_raw = s['acd_raw']
                                # compute_sirz_properties needs ap = apd/d and ac = acd/d, and returns (rho_e, Z_e)
                                ap_raw = apd_raw / s['thickness_mm']
                                ac_raw = acd_raw / s['thickness_mm']
                                _, ze_raw = compute_sirz_properties(ap_raw, ac_raw, s['thickness_mm'], K1, g, nu)
                                s['Ze_mean'] = float(np.mean(ze_raw))
                                s['Ze_std'] = float(np.std(ze_raw))
                                s['Ze_raw'] = ze_raw
                else:
                    print(f"  >>> CALIBRATION SKIPPED/FAILED ({f_type}, {voltage}): Incomplete data for 3 materials.")
                
                # 保存材质数据到 global_results
                if method not in global_results[f_type][voltage]:
                    global_results[f_type][voltage][method] = {}
                global_results[f_type][voltage][method]['materials'] = dict(voltage_data)
                
                if K1 is not None:
                    global_results[f_type][voltage][method]['calibration'] = {
                        'K1': float(K1),
                        'g': float(g),
                        'nu': float(nu),
                        'step_1': {
                            'K1': float(calib_results[0][0]),
                            'g': float(calib_results[0][1]),
                            'nu': float(calib_results[0][2])
                        }
                    }
                    if 2 in calib_results:
                        global_results[f_type][voltage][method]['calibration']['step_3'] = {
                            'K1': float(calib_results[2][0]),
                            'g': float(calib_results[2][1]),
                            'nu': float(calib_results[2][2])
                        }
                    if 4 in calib_results:
                        global_results[f_type][voltage][method]['calibration']['step_5'] = {
                            'K1': float(calib_results[4][0]),
                            'g': float(calib_results[4][1]),
                            'nu': float(calib_results[4][2])
                        }
                
                # 对第一、三、五个阶梯分别记录当前电压点下的特征系数
                for s_idx, step_num in [(0, 1), (2, 3), (4, 5)]:
                    # 获取该阶梯的校准参数
                    K1_s, g_s, nu_s = calib_results.get(s_idx, (None, None, None))
                    
                    coeff_summaries[f_type][method][step_num]['voltage'].append(v_int)
                    for name in step_mats.values():
                        if name in voltage_data:
                            stats = voltage_data[name]
                            # 确定索引不越界
                            idx = min(s_idx, len(stats) - 1)
                            s_target = stats[idx]
                            
                            bulk_ap_val = float(s_target['ap_mean'])
                            bulk_ac_val = float(s_target['ac_mean'])
                            
                            # 用该阶梯拟合出来的校准参数重新计算它的 Ze_mean 和 rho_e_mean
                            if K1_s is not None and g_s is not None and nu_s is not None:
                                bulk_rho_e_val, bulk_ze_val = compute_sirz_properties(bulk_ap_val, bulk_ac_val, s_target['thickness_mm'], K1_s, g_s, nu_s)
                                bulk_ze_val = float(bulk_ze_val)
                                bulk_rho_e_val = float(bulk_rho_e_val)
                            else:
                                bulk_ze_val = float(s_target['Ze_mean'])
                                bulk_rho_e_val = float(bulk_ac_val)
                                
                            coeff_summaries[f_type][method][step_num][name]['a_p_mean'].append(bulk_ap_val)
                            coeff_summaries[f_type][method][step_num][name]['a_c_mean'].append(bulk_ac_val)
                            coeff_summaries[f_type][method][step_num][name]['Ze_mean'].append(bulk_ze_val)
                            coeff_summaries[f_type][method][step_num][name]['rho_e_mean'].append(bulk_rho_e_val)
                        else:
                            coeff_summaries[f_type][method][step_num][name]['a_p_mean'].append(None)
                            coeff_summaries[f_type][method][step_num][name]['a_c_mean'].append(None)
                            coeff_summaries[f_type][method][step_num][name]['Ze_mean'].append(None)
                            coeff_summaries[f_type][method][step_num][name]['rho_e_mean'].append(None)
                
                # 调用模块化子函数生成 2x2 深度分析大图
                if plot_details:
                    fig_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_{method}_apd_acd_analysis.png")
                    _plot_detailed_profiling(voltage_data, f_type, voltage, colors, fig_save_path)
                    print(f"Saved physical profiling plot to {fig_save_path}")
                    
                    # 绘制并保存独立的对数线性拟合关系图 (使用第一阶梯)
                    if K1 is not None:
                        fig_fit_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_{method}_sirz_calibration_fit.png")
                        _plot_calibration_fit(voltage_data, K1, g, nu, 0, f_type, voltage, fig_fit_save_path)
                        print(f"Saved calibration fit plot to {fig_fit_save_path}")
                    
                    # 绘制并保存独立的像素级分布直方图，不修改原图
                    hist_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_{method}_apd_acd_histogram.png")
                    _plot_apd_acd_histograms(voltage_data, f_type, voltage, colors, hist_save_path)
                    print(f"Saved physical histogram plot to {hist_save_path}")
                    
            # 绘制不同方法在同一电压下的线性度对比图
            fig_comp, axes_comp = plt.subplots(2, 3, figsize=(14, 8))
            fig_comp.suptitle(f"APD/ACD Linearity Comparison for {f_type} at {voltage}", fontsize=14, fontweight='bold', y=0.98)
            col_map = {'Al_step': 0, 'Fe_step': 1, 'Cu_step': 2}
            has_comp_data = False
            
            for name in ['Al_step', 'Fe_step', 'Cu_step']:
                col = col_map[name]
                ax_ap = axes_comp[0, col]
                ax_ac = axes_comp[1, col]
                
                styles = {'Static': 'ro-', 'Dyn': 'gs-', 'M1': 'b^-', 'ChenWen': 'm*-'}
                for m in ['Static', 'Dyn', 'M1', 'ChenWen']:
                    if m in global_results[f_type][voltage] and 'materials' in global_results[f_type][voltage][m] and name in global_results[f_type][voltage][m]['materials']:
                        stats = global_results[f_type][voltage][m]['materials'][name]
                        d_mm = np.array([s['thickness_mm'] for s in stats])
                        apd_vals = np.array([s['apd_mean'] for s in stats])
                        acd_vals = np.array([s['acd_mean'] for s in stats])
                        
                        def fit_origin_r2(x, y):
                            slope = np.sum(x * y) / np.sum(x ** 2)
                            y_pred = slope * x
                            r2 = 1.0 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                            return slope, r2
                        
                        valid_mask = np.ones(len(d_mm), dtype=bool)
                        if name in ['Fe_step', 'Cu_step']:
                            # Use first 4 steps as requested
                            valid_mask = np.arange(len(d_mm)) < 4
                        
                        slope_ap, r2_ap = fit_origin_r2(d_mm[valid_mask], apd_vals[valid_mask])
                        slope_ac, r2_ac = fit_origin_r2(d_mm[valid_mask], acd_vals[valid_mask])
                        
                        ax_ap.plot(d_mm, apd_vals, styles[m], label=f"{m} ($R^2$={r2_ap:.4f})")
                        ax_ac.plot(d_mm, acd_vals, styles[m], label=f"{m} ($R^2$={r2_ac:.4f})")
                        has_comp_data = True
                
                lbl = name.replace('_step', '')
                ax_ap.set_title(f"{lbl} APD vs Thickness", fontsize=11)
                ax_ap.set_xlabel("Thickness $d$ (mm)")
                ax_ap.set_ylabel("$apd$")
                ax_ap.grid(True, linestyle='--', alpha=0.4)
                ax_ap.legend(fontsize='x-small', loc='best')
                
                ax_ac.set_title(f"{lbl} ACD vs Thickness", fontsize=11)
                ax_ac.set_xlabel("Thickness $d$ (mm)")
                ax_ac.set_ylabel("$acd$")
                ax_ac.grid(True, linestyle='--', alpha=0.4)
                ax_ac.legend(fontsize='x-small', loc='best')
                
            if has_comp_data:
                fig_comp.tight_layout()
                comp_fig_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_apd_acd_linearity.png")
                os.makedirs(os.path.dirname(comp_fig_path), exist_ok=True)
                fig_comp.savefig(comp_fig_path, dpi=150)
                plt.close(fig_comp)
                print(f"Saved linearity comparison plot to {comp_fig_path}")
            else:
                plt.close(fig_comp)
                
            gc.collect()

    # (单滤片 coefficient summaries 绘制已被移除，仅保留 combined summaries)


    # 绘制合并滤片 (0.6mm + 1.2mm) 的 combined summaries 曲线
    for method in ['Static', 'Dyn', 'M1', 'ChenWen']:
        for step_num in [1, 3, 5]:
            summary_06 = coeff_summaries['0.6mm'][method][step_num]
            summary_12 = coeff_summaries['1.2mm'][method][step_num]
            
            has_data_06 = summary_06['voltage'] and any(x is not None for x in summary_06['Al_step']['a_p_mean'])
            has_data_12 = summary_12['voltage'] and any(x is not None for x in summary_12['Al_step']['a_p_mean'])
            
            if has_data_06 or has_data_12:
                summary_path = os.path.join(abs_output_dir, f"summary_coefficients_{method}_combined_step{step_num}.png")
                _plot_combined_coefficient_dependence(
                    summary_06, summary_12, colors, summary_path, step_label=f"Step {step_num} ({method})"
                )
                print(f"Saved combined step {step_num} ({method}) summary plot to {summary_path}")

    # 特征序列持久化导出 (在存入 JSON 之前递归剔除未序列化的像素级原始数组以防报错并压缩体积)
    def _make_json_safe(data):
        if isinstance(data, dict):
            return {k: _make_json_safe(v) for k, v in data.items() if not k.endswith('_raw')}
        elif isinstance(data, list):
            return [_make_json_safe(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    json_path = os.path.join(script_dir, 'results/thickness_decoupling/apd_acd_summary.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_make_json_safe(global_results), f, indent=4, ensure_ascii=False)
        
    print(f"\n==========================================")
    print(f"APD/ACD step physical analysis successfully completed!")
    print(f"Details saved to: {abs_output_dir}")
    print(f"Summary JSON saved to: {json_path}")
    print(f"==========================================")


if __name__ == '__main__':
    run_step_apd_acd_analysis()