import numpy as np

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
def _plot_calibration_fit(voltage_data: dict, K1: float, g: float, nu: float, step_index: int, f_type: str, voltage: str, save_path: str) -> None:
    """
    绘制指定电压和滤片配置下的 SIRZ 有效原子序数 (Ze) 对数线性拟合图（包含散点与拟合曲线及误差分析）。
    
    参数：
    - voltage_data (dict): 包含当前电压下所有材料 step 统计特征的字典。
      - 类型：dict
      - 含义：解算出的多材质统计数据。
      - 用法：直接传递阶段汇总字典。
    - K1 (float): 电子密度标定系数。
      - 类型：float
      - 用法：传入校准出的 K1。
    - g (float): 有效原子序数校准系数。
      - 类型：float
      - 用法：传入校准出的 g。
    - nu (float): 有效原子序数幂次系数。
      - 类型：float
      - 用法：传入校准出的 nu。
    - step_index (int): 指示当前绘图数据所来自的厚度阶梯索引（0-based，如 0 对应第 1 阶梯）。
      - 类型：int
      - 用法：传入 0, 2, 4 等。
    - f_type (str): 当前滤片配置描述（如 '0.6mm', '1.2mm'）。
      - 类型：str
      - 用法：用于图表标题与路径。
    - voltage (str): 当前管电压描述（如 '200kV'）。
      - 类型：str
      - 用法：用于图表标题。
    - save_path (str): 拟合关系图保存的目标物理磁盘路径。
      - 类型：str
      - 用法：传入目标落地文件路径字符串。
      
    返回：
    - None (图片直接保存至磁盘)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    THEORY_DATA = {
        'Al_step': {'Z': 13.0, 'rho_e': 1.3008, 'color': '#4A90E2', 'label': 'Al'},
        'Fe_step': {'Z': 26.0, 'rho_e': 3.6644, 'color': '#2ECCA7', 'label': 'Fe'},
        'Cu_step': {'Z': 29.0, 'rho_e': 4.0888, 'color': '#E28743', 'label': 'Cu'}
    }
    
    # 提取各材料的实测比值 R_m
    mats_x = []
    mats_y = []
    mats_pred_y = []
    mats_colors = []
    mats_labels = []
    
    for name in THEORY_DATA.keys():
        if name in voltage_data:
            stats = voltage_data[name]
            idx = min(step_index, len(stats) - 1)
            bulk_ap = stats[idx]['ap_mean']
            bulk_ac = stats[idx]['ac_mean']
            R_m = bulk_ap / (bulk_ac + 1e-6)
            
            # 物理保护防止 R_m 负值
            R_m_safe = max(1e-3, R_m)
            
            # 使用校准常数预测 Ze
            Ze_pred = g * (R_m_safe) ** (1.0 / nu)
            
            mats_x.append(R_m_safe)
            mats_y.append(THEORY_DATA[name]['Z'])
            mats_pred_y.append(Ze_pred)
            mats_colors.append(THEORY_DATA[name]['color'])
            mats_labels.append(THEORY_DATA[name]['label'])
            
    if not mats_x:
        return
        
    mats_x = np.array(mats_x)
    mats_y = np.array(mats_y)
    mats_pred_y = np.array(mats_pred_y)
    
    # 绘图设计：双对数坐标轴下更具线性直观度
    plt.figure(figsize=(9, 7))
    
    # 1. 绘制拟合出来的 Z_e = g * R ^ (1/nu) 曲线
    R_fit = np.logspace(np.log10(min(mats_x) * 0.5), np.log10(max(mats_x) * 1.5), 300)
    Z_fit = g * (R_fit) ** (1.0 / nu)
    plt.plot(R_fit, Z_fit, '-', color='#E06666', linewidth=2.0, alpha=0.9, label=rf'Fit: $Z_e = {g:.4f} \cdot (a_p/a_c)^{{1/{nu:.4f}}}$')
    
    # 2. 绘制各材质的理论值点（实心圆点）
    for i in range(len(mats_x)):
        plt.scatter(mats_x[i], mats_y[i], color=mats_colors[i], marker='o', s=120, edgecolors='black', linewidths=1.2, zorder=5,
                    label=f'{mats_labels[i]} Theory ({mats_y[i]})')
        
    # 3. 绘制各材质的实际预测点（空心正方形点），并绘制虚线连接两点（代表偏差）
    for i in range(len(mats_x)):
        plt.scatter(mats_x[i], mats_pred_y[i], facecolors='none', edgecolors=mats_colors[i], marker='s', s=100, linewidths=2.0, zorder=4,
                    label=f'{mats_labels[i]} Pred ({mats_pred_y[i]:.2f})')
        plt.plot([mats_x[i], mats_x[i]], [mats_y[i], mats_pred_y[i]], ':', color='#888888', alpha=0.8)
        
    plt.xscale('log')
    plt.yscale('log')
    
    # 设置刻度显示为普通数值而非科学计数法
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
    
    # 设置坐标限制
    plt.xlim(min(mats_x) * 0.4, max(mats_x) * 1.8)
    plt.ylim(10.0, 35.0)
    
    plt.title(f"SIRZ $Z_e$ Calibration Regression Fit (Step {step_index + 1})\nFilter: {f_type} at {voltage}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(r"Ratio $a_p / a_c$ (Log Scale)", fontsize=11)
    plt.ylabel(r"Effective Atomic Number $Z$ (Log Scale)", fontsize=11)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')
    
    # 在图中加注文本框，详细说明误差百分比
    text_lines = [
        r"$\bf{Calibration\ Parameters:}$",
        f"$K_1 = {K1:.6f}$",
        f"$g = {g:.6f}$",
        r"$\nu = " + f"{nu:.6f}$",
        "",
        r"$\bf{Relative\ Estimation\ Errors:}$"
    ]
    for i in range(len(mats_x)):
        err_percent = (mats_pred_y[i] - mats_y[i]) / mats_y[i] * 100.0
        sign = "+" if err_percent >= 0 else ""
        text_lines.append(f"{mats_labels[i]}: Theory {mats_y[i]} vs Pred {mats_pred_y[i]:.2f} ({sign}{err_percent:.2f}%)")
        
    text_str = "\n".join(text_lines)
    plt.text(0.05, 0.05, text_str, transform=plt.gca().transAxes, fontsize=9.5,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D3D3D3', alpha=0.9))
             
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ===================== 核心高层调度入口 (Main Orchestrator Entry) =====================

def run_step_apd_acd_analysis(include_0331=True, plot_details=True, output_dir='results/thickness_decoupling/apd_acd_analysis'):
    """
    核心分析调度主函数，遍历所有实验电压与滤片配置，读取 16 位阶梯校准样本数据，
    调用模块化子函数执行像素级双特征物理算子提取，保存 step 特征汇总 JSON，
    并渲染精美的厚度拟合 2x2 图与电压-衰减系数依赖关系图。
    
    参数：
    - include_0331 (bool): 默认为 True，指示在 0.6mm 滤片厚度的分析中是否并入 0331 实验数据集（包括 140kV, 160kV, 180kV 电压）。
      用法：传入 True 启用，False 禁用。
    - plot_details (bool): 默认为 True，指示是否为每一个电压和滤片组合绘制并保存详细的 2x2 物理特征拟合与轨迹图像。
      用法：传入 True 启用，False 禁用。
    - output_dir (str): 默认值为 'results/thickness_decoupling/apd_acd_analysis'，指定保存所有生成图表、总结曲线和 JSON 数据的目录。
      用法：传入相对或绝对路径的字符串。
    """
    import os
    import re
    import json
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_0429 = os.path.join(script_dir, 'results/20260429_mask_generated_16bit')
    input_dir_0331 = os.path.join(script_dir, 'results/20260331_16bit')
    abs_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    voltages_0429 = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
    voltages_0331 = ['140kV', '160kV', '180kV']
    filter_types = ['0.6mm', '1.2mm']
    
    I0_0429 = 52428.0
    I0_0331 = 52428.0
    
    step_mats = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step'}
    thicknesses_0429 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
    thicknesses_0331 = { 'Cu_step': np.arange(2, 22, 2), 'Fe_step': np.arange(2, 22, 2), 'Al_step': np.arange(12, 32, 2) }
    
    # 专属精美色彩映射字典
    colors = {
        'Al_step': '#4A90E2',  # 现代天空蓝
        'Fe_step': '#2ECCA7',  # 优雅孔雀绿
        'Cu_step': '#E28743'   # 经典黄铜橙
    }
    
    global_results = {}
    
    for f_type in filter_types:
        global_results[f_type] = {}
        cur_voltages = voltages_0331 + voltages_0429 if (f_type == '0.6mm' and include_0331) else voltages_0429
        
        # 初始化当前 filter 厚度下随电压变化的三个厚度阶梯（Step 1, 3, 5）的数据汇总器
        coeff_summaries = {
            1: {
                'voltage': [],
                'Al_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Fe_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Cu_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []}
            },
            3: {
                'voltage': [],
                'Al_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Fe_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Cu_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []}
            },
            5: {
                'voltage': [],
                'Al_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Fe_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []},
                'Cu_step': {'a_p_mean': [], 'a_c_mean': [], 'Ze_mean': [], 'rho_e_mean': []}
            }
        }
        
        for voltage in cur_voltages:
            is_0331 = voltage in voltages_0331
            cur_input_dir = input_dir_0331 if is_0331 else input_dir_0429
            cur_I0 = I0_0331 if is_0331 else I0_0429
            cur_thick_map = thicknesses_0331 if is_0331 else thicknesses_0429
            v_int = int(re.search(r'(\d+)', voltage).group(1))
            
            voltage_data = {}
            
            for idx, name in step_mats.items():
                if is_0331:
                    p = f'{cur_input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
                else:
                    p = f'{cur_input_dir}/pixel_values/{name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl'
                
                if os.path.exists(p):
                    t_arr = cur_thick_map[name]
                    print(f"Processing step data for {name} ({f_type}, {voltage})...")
                    # 调用模块化子函数进行特征解算
                    stats = _load_and_process_step_pixels(p, t_arr, cur_I0)
                    if stats:
                        voltage_data[name] = stats
                        
            if voltage_data:
                # 分别尝试第一、三、五个阶梯（索引为 0, 2, 4）进行校准并打印结果
                calib_results = {}
                for s_idx in [0, 2, 4]:
                    K1_s, g_s, nu_s = calibrate_sirz_coefficients(voltage_data, step_index=s_idx)
                    if K1_s is not None:
                        calib_results[s_idx] = (K1_s, g_s, nu_s)
                        print(f"  >>> CALIBRATION at Step {s_idx + 1} (idx {s_idx}): K1 = {K1_s:.6f}, g = {g_s:.6f}, nu = {nu_s:.6f}")
                
                # 默认使用第一阶梯（索引 0）的校准常数应用于后续的 Ze 重构和绘图
                K1, g, nu = calib_results.get(0, (None, None, None))
                
                if K1 is not None:
                    print(f"  >>> APPLYING Step 1 CALIBRATION ({f_type}, {voltage}): K1 = {K1:.6f}, g = {g:.6f}, nu = {nu:.6f}")
                    # 用校准结果更新 voltage_data 中的 Ze_mean, Ze_std 和 Ze_raw
                    for name in step_mats.values():
                        if name in voltage_data:
                            for s in voltage_data[name]:
                                apd_raw = s['apd_raw']
                                acd_raw = s['acd_raw']
                                ze_raw = g * (apd_raw / (acd_raw + 1e-6)) ** (1.0 / nu)
                                s['Ze_mean'] = float(np.mean(ze_raw))
                                s['Ze_std'] = float(np.std(ze_raw))
                                s['Ze_raw'] = ze_raw
                else:
                    print(f"  >>> CALIBRATION SKIPPED/FAILED ({f_type}, {voltage}): Incomplete data for 3 materials.")
                
                # 保存材质数据到 global_results
                global_results[f_type][voltage] = dict(voltage_data)
                
                if K1 is not None:
                    global_results[f_type][voltage]['calibration'] = {
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
                        global_results[f_type][voltage]['calibration']['step_3'] = {
                            'K1': float(calib_results[2][0]),
                            'g': float(calib_results[2][1]),
                            'nu': float(calib_results[2][2])
                        }
                    if 4 in calib_results:
                        global_results[f_type][voltage]['calibration']['step_5'] = {
                            'K1': float(calib_results[4][0]),
                            'g': float(calib_results[4][1]),
                            'nu': float(calib_results[4][2])
                        }
                
                # 对第一、三、五个阶梯分别记录当前电压点下的特征系数
                for s_idx, step_num in [(0, 1), (2, 3), (4, 5)]:
                    # 获取该阶梯的校准参数
                    K1_s, g_s, nu_s = calib_results.get(s_idx, (None, None, None))
                    
                    coeff_summaries[step_num]['voltage'].append(v_int)
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
                                bulk_ze_val = float(g_s * (bulk_ap_val / (bulk_ac_val + 1e-6)) ** (1.0 / nu_s))
                                bulk_rho_e_val = float(K1_s * bulk_ac_val)
                            else:
                                bulk_ze_val = float(s_target['Ze_mean'])
                                bulk_rho_e_val = float(bulk_ac_val)
                                
                            coeff_summaries[step_num][name]['a_p_mean'].append(bulk_ap_val)
                            coeff_summaries[step_num][name]['a_c_mean'].append(bulk_ac_val)
                            coeff_summaries[step_num][name]['Ze_mean'].append(bulk_ze_val)
                            coeff_summaries[step_num][name]['rho_e_mean'].append(bulk_rho_e_val)
                        else:
                            coeff_summaries[step_num][name]['a_p_mean'].append(None)
                            coeff_summaries[step_num][name]['a_c_mean'].append(None)
                            coeff_summaries[step_num][name]['Ze_mean'].append(None)
                            coeff_summaries[step_num][name]['rho_e_mean'].append(None)
                
                # 调用模块化子函数生成 2x2 深度分析大图
                if plot_details:
                    fig_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_apd_acd_analysis.png")
                    _plot_detailed_profiling(voltage_data, f_type, voltage, colors, fig_save_path)
                    print(f"Saved physical profiling plot to {fig_save_path}")
                    
                    # 绘制并保存独立的对数线性拟合关系图 (使用第一阶梯)
                    if K1 is not None:
                        fig_fit_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_sirz_calibration_fit.png")
                        _plot_calibration_fit(voltage_data, K1, g, nu, 0, f_type, voltage, fig_fit_save_path)
                        print(f"Saved calibration fit plot to {fig_fit_save_path}")
                    
                    # 绘制并保存独立的像素级分布直方图，不修改原图
                    hist_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_apd_acd_histogram.png")
                    _plot_apd_acd_histograms(voltage_data, f_type, voltage, colors, hist_save_path)
                    print(f"Saved physical histogram plot to {hist_save_path}")
                    
        # 针对该滤片，为第一、三、五个阶梯分别调用子函数绘制多电压 bulk 系数能量相关趋势大图
        for step_num in [1, 3, 5]:
            summary_dict = coeff_summaries[step_num]
            if summary_dict['voltage']:
                summary_path = os.path.join(abs_output_dir, f"summary_{f_type}_coefficients_step{step_num}.png")
                _plot_coefficient_dependence(summary_dict, f_type, colors, summary_path, step_label=f"Step {step_num}")
                print(f"Saved step {step_num} coefficient summary plot to {summary_path}")
            
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