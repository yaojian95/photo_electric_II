import numpy as np

CONFIG = {
    'E_L': 30,                               # 低能能量 (keV)
    'E_H': 60,                               # 高能能量 (keV)
    'E_0': 1.0,                              # 参考能量 (keV)
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
    E_0 = CONFIG['E_0']
    low = low.astype(float)
    high = high.astype(float)
    mu_L_d = np.log(I0_low / (low + 1e-6))
    mu_H_d = np.log(I0_high / (high + 1e-6))
    t1 = mu_L_d * _fkn(E_H) - mu_H_d * _fkn(E_L)
    t2 = _fkn(E_H) * (E_0 / E_L) ** 3 - _fkn(E_L) * (E_0 / E_H) ** 3

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
    E_0 = CONFIG['E_0']
    low = low.astype(float)
    high = high.astype(float)
    mu_L_d = np.log(I0_low / (low + 1e-6))
    mu_H_d = np.log(I0_high / (high + 1e-6))
    t1 = mu_H_d*((E_0 / E_L )**3)  - mu_L_d*((E_0 / E_H )**3)
    t2 = _fkn(E_H)*(E_0 / E_L) ** 3 - _fkn(E_L)*(E_0 / E_H) ** 3 
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
    
    # 4. Zeff vs Thickness (直接指示能谱硬化漂移)
    ax4 = axes[1, 1]
    ax4.set_title(r"Effective Atomic Number $Z_{eff}$ vs Physical Thickness", fontsize=12, pad=10)
    ax4.set_xlabel("Thickness $d$ (mm)"); ax4.set_ylabel(r"$Z_{eff}$")
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    for name, stats in voltage_data.items():
        t_vals = [s['thickness_mm'] for s in stats]
        apd_means = [s['apd_mean'] for s in stats]
        apd_stds = [s['apd_std'] for s in stats]
        acd_means = [s['acd_mean'] for s in stats]
        acd_stds = [s['acd_std'] for s in stats]
        zeff_means = [s['Zeff_mean'] for s in stats]
        zeff_stds = [s['Zeff_std'] for s in stats]
        
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        # Subplot 1: APD vs thickness
        ax1.errorbar(t_vals, apd_means, yerr=apd_stds, fmt='o-', color=c, label=f"{lbl} Data", capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        if len(t_vals) > 1:
            slope = np.sum(np.array(t_vals) * np.array(apd_means)) / np.sum(np.array(t_vals)**2)
            ax1.plot(t_vals, slope * np.array(t_vals), '--', color=c, alpha=0.5, label=f"{lbl} Fit (slope={slope:.4f})")
        
        # Subplot 2: ACD vs thickness
        ax2.errorbar(t_vals, acd_means, yerr=acd_stds, fmt='o-', color=c, label=f"{lbl} Data", capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        if len(t_vals) > 1:
            slope = np.sum(np.array(t_vals) * np.array(acd_means)) / np.sum(np.array(t_vals)**2)
            ax2.plot(t_vals, slope * np.array(t_vals), '--', color=c, alpha=0.5, label=f"{lbl} Fit (slope={slope:.4f})")
        
        # Subplot 3: APD vs ACD Trajectory
        ax3.errorbar(acd_means, apd_means, xerr=acd_stds, yerr=apd_stds, fmt='o-', color=c, label=lbl, capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        
        # Subplot 4: Zeff vs thickness
        ax4.errorbar(t_vals, zeff_means, yerr=zeff_stds, fmt='o-', color=c, label=lbl, capsize=3, elinewidth=1, alpha=0.8, markersize=5)
        
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


def _plot_coefficient_dependence(coeff_summary: dict, f_type: str, colors: dict, save_path: str) -> None:
    """
    绘制给定滤片厚度下，Bulk 物理系数（Photoelectric, Compton, Zeff, Ratio）随 X 射线管电压变化的曲线图。
    
    参数：
    - coeff_summary (dict): 包含汇总后各管电压点物理系数均值的字典。
      用法：键包含 'voltage' 及各材料名，字典数据按电压对应。
    - f_type (str): 当前滤片配置描述（如 '0.6mm', '1.2mm'）。
      用法：字符串，用于标题标注。
    - colors (dict): 用于图线着色的材料色彩配置映射字典。
      用法：形如 {'Al_step': '#4A90E2', ...} 的色彩字典。
    - save_path (str): 汇总依赖关系曲线大图的保存路径。
      用法：路径字符串，指示图片最终落地位置。
    """
    import os
    import matplotlib.pyplot as plt
    
    v_arr = np.array(coeff_summary['voltage'])
    if len(v_arr) == 0:
        return
        
    sort_idx = np.argsort(v_arr)
    v_sorted = v_arr[sort_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Coefficient Energy Dependence (Filter: {f_type})", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. a_p 随电压变化
    ax1 = axes[0, 0]
    ax1.set_title(r"Bulk Photoelectric Coefficient $a_p$ vs Voltage", fontsize=12, pad=10)
    ax1.set_xlabel("Voltage (kV)"); ax1.set_ylabel(r"$a_p \ (\mathrm{mm}^{-1})$")
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. a_c 随电压变化
    ax2 = axes[0, 1]
    ax2.set_title(r"Bulk Compton Coefficient $a_c$ vs Voltage", fontsize=12, pad=10)
    ax2.set_xlabel("Voltage (kV)"); ax2.set_ylabel(r"$a_c \ (\mathrm{mm}^{-1})$")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Zeff 随电压变化
    ax3 = axes[1, 0]
    ax3.set_title(r"Bulk Effective Atomic Number $Z_{eff}$ vs Voltage", fontsize=12, pad=10)
    ax3.set_xlabel("Voltage (kV)"); ax3.set_ylabel(r"$Z_{eff}$")
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. ap / ac 比值随电压变化
    ax4 = axes[1, 1]
    ax4.set_title(r"Photo-Compton Ratio $a_p / a_c$ vs Voltage", fontsize=12, pad=10)
    ax4.set_xlabel("Voltage (kV)"); ax4.set_ylabel(r"$a_p / a_c$")
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    for name in ['Al_step', 'Fe_step', 'Cu_step']:
        c = colors.get(name, '#888888')
        lbl = name.replace('_step', '')
        
        ap_vals = np.array(coeff_summary[name]['a_p_mean'])[sort_idx]
        ac_vals = np.array(coeff_summary[name]['a_c_mean'])[sort_idx]
        zeff_vals = np.array(coeff_summary[name]['Zeff_mean'])[sort_idx]
        ap_ac_vals = np.array(coeff_summary[name]['ap_ac_mean'])[sort_idx]
        
        valid = [x is not None for x in ap_vals]
        if any(valid):
            v_v = v_sorted[valid]
            ax1.plot(v_v, ap_vals[valid], 'o-', color=c, label=lbl, linewidth=2, markersize=6)
            ax2.plot(v_v, ac_vals[valid], 'o-', color=c, label=lbl, linewidth=2, markersize=6)
            ax3.plot(v_v, zeff_vals[valid], 'o-', color=c, label=lbl, linewidth=2, markersize=6)
            ax4.plot(v_v, ap_ac_vals[valid], 'o-', color=c, label=lbl, linewidth=2, markersize=6)
            
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
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
        
        # 初始化当前 filter 厚度下随电压变化的系数数据汇总器
        coeff_summary = {
            'voltage': [],
            'Al_step': {'a_p_mean': [], 'a_c_mean': [], 'Zeff_mean': [], 'ap_ac_mean': []},
            'Fe_step': {'a_p_mean': [], 'a_c_mean': [], 'Zeff_mean': [], 'ap_ac_mean': []},
            'Cu_step': {'a_p_mean': [], 'a_c_mean': [], 'Zeff_mean': [], 'ap_ac_mean': []}
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
                global_results[f_type][voltage] = voltage_data
                
                # 计算Bulk材料系数，选用前5个较薄的阶梯以减小硬化效应造成的非线性扭曲
                coeff_summary['voltage'].append(v_int)
                for name in step_mats.values():
                    if name in voltage_data:
                        stats = voltage_data[name]
                        ref_len = min(5, len(stats))
                        ap_vals = [s['ap_mean'] for s in stats[:ref_len]]
                        ac_vals = [s['ac_mean'] for s in stats[:ref_len]]
                        zeff_vals = [s['Zeff_mean'] for s in stats[:ref_len]]
                        ap_ac_vals = [s['ap_ac_mean'] for s in stats[:ref_len]]
                        
                        coeff_summary[name]['a_p_mean'].append(float(np.mean(ap_vals)))
                        coeff_summary[name]['a_c_mean'].append(float(np.mean(ac_vals)))
                        coeff_summary[name]['Zeff_mean'].append(float(np.mean(zeff_vals)))
                        coeff_summary[name]['ap_ac_mean'].append(float(np.mean(ap_ac_vals)))
                    else:
                        coeff_summary[name]['a_p_mean'].append(None)
                        coeff_summary[name]['a_c_mean'].append(None)
                        coeff_summary[name]['Zeff_mean'].append(None)
                        coeff_summary[name]['ap_ac_mean'].append(None)
                
                # 调用模块化子函数生成 2x2 深度分析大图
                if plot_details:
                    fig_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_apd_acd_analysis.png")
                    _plot_detailed_profiling(voltage_data, f_type, voltage, colors, fig_save_path)
                    print(f"Saved physical profiling plot to {fig_save_path}")
                    
                    # 绘制并保存独立的像素级分布直方图，不修改原图
                    hist_save_path = os.path.join(abs_output_dir, f"{f_type}/{voltage}_apd_acd_histogram.png")
                    _plot_apd_acd_histograms(voltage_data, f_type, voltage, colors, hist_save_path)
                    print(f"Saved physical histogram plot to {hist_save_path}")
                    
        # 针对该滤片，调用子函数绘制多电压 bulk 系数能量相关趋势大图
        if coeff_summary['voltage']:
            summary_path = os.path.join(abs_output_dir, f"summary_{f_type}_coefficients.png")
            _plot_coefficient_dependence(coeff_summary, f_type, colors, summary_path)
            print(f"Saved coefficient summary plot to {summary_path}")
            
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