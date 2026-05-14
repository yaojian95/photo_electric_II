import os
import re
import csv
import json
import requests
import urllib3
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from bs4 import BeautifulSoup

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_mu_rho(element_symbol):
    """
    爬取 NIST 原始数据。添加了 SSL 修复、Headers 以及请求间隔。
    """
    symbol_to_z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Mo': 42, 'Ag': 47, 'Sn': 50, 'W': 74, 'Pt': 78, 'Au': 79, 'Pb': 82, 'U': 92
    }
    z = symbol_to_z.get(element_symbol)
    if z is None: raise ValueError(f"Element {element_symbol} not recognized.")

    url = f"https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{z:02d}.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        # 使用 verify=False 绕过 SSL 错误
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Error fetching {element_symbol}: {e}")

    soup = BeautifulSoup(response.text, 'html.parser')
    pre_tag = soup.find('pre')
    if not pre_tag: raise ValueError(f"No data table found for {element_symbol}.")

    energies, mu_rho = [], []
    # 使用正则解析，更稳健地处理空格和吸收边界标注
    for line in pre_tag.text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            try:
                e_val, m_val = float(parts[0]), float(parts[1])
                energies.append(e_val)
                mu_rho.append(m_val)
            except ValueError: continue
            
    return energies, mu_rho

def save_mu_rho_to_local(element_symbol, data_dir='nist_data'):
    """
    保存为 CSV (您的需求) 并同步更新汇总的 JSON (之前提到的功能)。
    """
    os.makedirs(data_dir, exist_ok=True)
    full_energies, mu_rho_values = fetch_mu_rho(element_symbol)

    # 1. 保存 CSV
    csv_path = os.path.join(data_dir, f"{element_symbol}_mu_rho.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Energy_MeV', 'Mu_over_rho_cm2_per_g'])
        for e, mu in zip(full_energies, mu_rho_values):
            writer.writerow([e, mu])
            
    # 2. 同步更新 JSON 汇总文件 (方便跨脚本调用)
    json_path = os.path.join(data_dir, "nist_mu_data.json")
    all_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f: all_data = json.load(f)
    
    all_data[element_symbol] = {
        "energies_mev": full_energies,
        "mu_rho_cm2_g": mu_rho_values
    }
    with open(json_path, 'w') as f: json.dump(all_data, f, indent=4)
    
    print(f"[+] Saved {element_symbol} to {csv_path} and updated {json_path}")
    time.sleep(0.5) # 礼貌延迟

def get_mu_rho_interpolated(element_symbol, target_energies_keV, data_dir='nist_data'):
    """
    使用对数空间下的 Pchip 插值，这是处理衰减系数最准确的方法。
    """
    csv_path = os.path.join(data_dir, f"{element_symbol}_mu_rho.csv")
    if not os.path.exists(csv_path):
        save_mu_rho_to_local(element_symbol, data_dir)
    
    e_raw, mu_raw = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            e_raw.append(float(row[0]) * 1000.0) # 转为 keV
            mu_raw.append(float(row[1]))

    e_raw, mu_raw = np.array(e_raw), np.array(mu_raw)
    # 处理吸收边界重复能量点（加微小偏移以满足插值器要求）
    for i in range(1, len(e_raw)):
        if e_raw[i] <= e_raw[i-1]: e_raw[i] = e_raw[i-1] + 1e-9

    # 核心：对数空间插值
    log_e, log_mu = np.log10(e_raw), np.log10(mu_raw)
    # 使用线性插值（在对数空间），这等同于在原始空间使用幂律插值 (mu = C * E^-alpha)
    # 这会产生在对数坐标轴下完美的直线，更符合 NIST 官网的视觉效果和物理特性
    interp = interp1d(log_e, log_mu, kind='linear', fill_value="extrapolate")
    
    res_log_mu = interp(np.log10(target_energies_keV))
    return e_raw, mu_raw, 10**res_log_mu

def plot_mu_rho_vs_energy(elements, data_dir='nist_data'):
    """
    绘制对比图并保存到 nist_data 目录。
    """
    # 将范围改为 1keV 起始，以显示中低原子序数元素的吸收边 (Fe K-edge ~7.1 keV)
    energies_keV = np.logspace(0, 3, 1000) 
    plt.figure(figsize=(9, 6))

    for symbol in elements:
        try:
            e_raw, mu_raw, mu_interp = get_mu_rho_interpolated(symbol, energies_keV, data_dir)
            plt.loglog(energies_keV, mu_interp, label=symbol, linewidth=2)
        except Exception as e:
            print(f"Error plotting {symbol}: {e}")

    plt.axvspan(90, 350, color='gray', alpha=0.1, label='Range (90-350keV)')
    plt.title(r'X-Ray Mass Attenuation Coefficient $\mu/\rho$', fontsize=14)
    plt.xlabel('Photon Energy (keV)', fontsize=12)
    plt.ylabel(r'Mass Attenuation Coefficient $\mu/\rho$ (cm$^2$/g)', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存图片到 nist_data
    save_path = os.path.join(data_dir, "mu_rho_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[+] Plot saved to {save_path}")
    plt.show()

def get_energy_from_mu(element_symbol, mu_list, data_dir='nist_data'):
    """
    根据输入的线衰减系数 mu (cm^-1) 列表，反推对应的平均 X 射线能量 (keV)。
    
    参数:
        element_symbol: 元素符号 ('Fe', 'Al', 'Cu')
        mu_list: 线衰减系数列表 (list or ndarray)
        data_dir: 数据目录
    返回:
        能量列表 (ndarray, keV)
    """
    # 典型密度定义 (g/cm^3)
    densities = {'Fe': 7.87, 'Al': 2.70, 'Cu': 8.96, 'Si': 2.65, 'S': 2.07}
    rho = densities.get(element_symbol)
    if rho is None:
        raise ValueError(f"Density for {element_symbol} not defined. Please add it to the dictionary.")
    
    # 1. 转换为质量衰减系数 mu/rho
    # 核心修正：由于 JSON 中的 mu 是由 mm 单位的厚度拟合得到的 (mm^-1)，
    # 而密度 rho 是 g/cm^3，NIST 数据是 cm^2/g，因此需要先将 mu 转换为 cm^-1。
    mu_list = np.array(mu_list)
    target_mu_rho = (mu_list * 10.0) / rho
    
    # 2. 加载 NIST 原始数据
    csv_path = os.path.join(data_dir, f"{element_symbol}_mu_rho.csv")
    if not os.path.exists(csv_path):
        save_mu_rho_to_local(element_symbol, data_dir)
        
    e_raw, mu_raw = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            e_raw.append(float(row[0]) * 1000.0) # keV
            mu_raw.append(float(row[1]))
            
    e_raw, mu_raw = np.array(e_raw), np.array(mu_raw)
    
    # 3. 反向插值: E = f(mu_rho)
    # 在对数空间下，mu/rho 随能量增加而单调下降（除吸收边外），适合反向插值
    # 为了保证单调性以便插值，我们对原始数据进行排序
    sort_idx = np.argsort(mu_raw)
    sorted_mu_raw = mu_raw[sort_idx]
    sorted_e_raw = e_raw[sort_idx]
    
    # 对数空间反向插值器
    log_mu = np.log10(sorted_mu_raw)
    log_e = np.log10(sorted_e_raw)
    
    inv_interp = interp1d(log_mu, log_e, kind='linear', fill_value="extrapolate")
    
    res_log_e = inv_interp(np.log10(target_mu_rho))
    return 10**res_log_e

def plot_energy_summary(json_path='attenuation_slopes.json', selected_elements=None):
    """
    读取 attenuation_slopes.json，推算并绘制不同材质在不同管电压下的平均能量对比图。
    
    参数:
        json_path: 数据路径
        selected_elements: 要显示的元素列表 (如 ['Cu', 'Fe'])，None 表示全选。
    """
    if not os.path.exists(json_path):
        print(f"[!] {json_path} 不存在，请先运行拟合脚本生成数据。")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filter_types = list(data.keys())
    mats_map = {
        "Cu_step": "Cu",
        "Fe_step": "Fe",
        "Al_step": "Al"
    }
    
    # 设置中文字体（如果是 Windows 系统通常有黑体或等线）
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {'Cu': 'red', 'Fe': 'blue', 'Al': 'green'}
    
    for ax, f_type in zip(axes, filter_types):
        ax.set_title(f"滤片厚度: {f_type}", fontsize=14)
        ax.set_xlabel("管电压 (kV)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("推算平均能量 (keV)", fontsize=12)
        
        for json_mat, nist_symbol in mats_map.items():
            # 如果指定了元素列表且当前元素不在其中，则跳过
            if selected_elements is not None and nist_symbol not in selected_elements:
                continue

            if json_mat not in data[f_type]:
                continue
                
            mat_data = data[f_type][json_mat]
            voltages = []
            mu_l_list = []
            mu_h_list = []
            
            # 按电压数值排序
            sorted_vs = sorted(mat_data.keys(), key=lambda x: int(x.replace('kV', '')))
            
            for v_str in sorted_vs:
                v_int = int(v_str.replace('kV', ''))
                ul = mat_data[v_str]['ul']
                uh = mat_data[v_str]['uh']
                
                if ul is not None and uh is not None:
                    voltages.append(v_int)
                    mu_l_list.append(ul)
                    mu_h_list.append(uh)
            
            if not voltages:
                continue
                
            # 计算能量
            energy_l = get_energy_from_mu(nist_symbol, mu_l_list)
            energy_h = get_energy_from_mu(nist_symbol, mu_h_list)
            
            # 绘图
            ax.plot(voltages, energy_l, 'o--', color=colors[nist_symbol], label=f"{nist_symbol} (Low)")
            ax.plot(voltages, energy_h, 's-', color=colors[nist_symbol], label=f"{nist_symbol} (High)")
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("energy_vs_voltage_summary.png", dpi=300)
    print("[+] 能量分析图已保存至 energy_vs_voltage_summary.png")
    # plt.show()

if __name__ == "__main__":
    # 1. 仅绘制 Fe, Al, Cu 的衰减曲线对比图
    # target_list = ['Fe', 'Al', 'Cu']
    # plot_mu_rho_vs_energy(target_list)

    # # 2. 演示功能：根据 mu 反推能量
    # # 假设我们在不同电压下测量到了 Fe 的线衰减系数 mu (cm^-1)
    # test_mu_fe = [10.0, 5.0, 2.0] 
    # predicted_energies = get_energy_from_mu('Fe', test_mu_fe)
    
    # print("\n" + "="*50)
    # print("演示功能: 根据线衰减系数 \u03bc 反推平均能量")
    # print("="*50)
    # for mu, e in zip(test_mu_fe, predicted_energies):
    #     print(f"物质: Fe, 输入 \u03bc = {mu:4.1f} cm^-1  =>  推算平均能量: {e:6.2f} keV")

    # 3. 运行能量汇总分析
    plot_energy_summary(selected_elements=['Fe', 'Cu', 'Al'])
