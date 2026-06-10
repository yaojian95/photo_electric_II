import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize
import get_mu_from_nist_new
import get_apd_acd
import gc

# ==============================================================================
# Global Configuration & Physical Constants
# ==============================================================================
DENSITIES = {
    'Al': 2.70,
    'Fe': 7.87,
    'Cu': 8.96
}

THICKNESS_MAP = {
    'Cu_step': np.arange(2, 22, 2),  # mm
    'Fe_step': np.arange(2, 22, 2),  # mm
    'Al_step': np.arange(12, 32, 2)  # mm
}

NIST_SYMBOLS = {
    'Cu_step': 'Cu',
    'Fe_step': 'Fe',
    'Al_step': 'Al'
}

VOLTAGES = ['200kV', '220kV', '240kV', '260kV', '280kV', '300kV', '320kV']
FILTER_TYPES = ['0.6mm', '1.2mm']
I0_VAL = 52428.0  # 16-bit background intensity reference

# ==============================================================================
# Helper Functions
# ==============================================================================

def fkn(E_keV):
    """
    Calculates the Klein-Nishina scattering cross-section coefficient.
    
    Parameters:
    - E_keV (np.ndarray or float): Photon energy in keV.
      Type: np.ndarray or float
      Meaning: The energy or grid of energies for calculation.
      Usage: Pass float or numpy array of energies in keV.
      
    Returns:
    - np.ndarray or float: Dimensionless Klein-Nishina cross-section.
    """
    alpha = E_keV / 511.0
    term1 = 2.0 * (1.0 + alpha) ** 2 / (alpha ** 2 * (1.0 + 2.0 * alpha))
    term2 = (np.log(1.0 + 2.0 * alpha) / alpha) * (0.5 - (1.0 + alpha) / (alpha ** 2))
    term3 = (1.0 + 3.0 * alpha) / (1.0 + 2.0 * alpha) ** 2
    return term1 + term2 - term3


def get_linear_attenuation(element_symbol: str, energies_keV: np.ndarray, density: float) -> np.ndarray:
    """
    Retrieves mass attenuation coefficient from NIST and scales by density to get linear attenuation (cm^-1).
    
    Parameters:
    - element_symbol (str): The element chemical symbol (e.g. 'Al', 'Fe', 'Cu').
      Type: str
      Usage: Pass element symbol.
    - energies_keV (np.ndarray): Energy grid points in keV.
      Type: np.ndarray (float)
      Usage: Pass energy array.
    - density (float): Mass density of the element in g/cm^3.
      Type: float
      Usage: Pass density in g/cm^3.
      
    Returns:
    - np.ndarray: Linear attenuation coefficient in cm^-1.
    """
    # Use log-log interpolation from the local nist file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'nist_data')
    _, _, mu_rho = get_mu_from_nist_new.get_mu_rho_interpolated(element_symbol, energies_keV, data_dir=data_dir)
    return mu_rho * density


def load_transmission_data(f_type: str, voltage: str, data_dir: str, I0: float = I0_VAL, cu_fe_max_steps: int = 10) -> tuple:
    """
    Loads step wedge pickle data and computes transmission ratios for Al, Fe, Cu.
    
    Parameters:
    - f_type (str): Filter thickness descriptor (e.g., '0.6mm', '1.2mm').
      Type: str
      Meaning: The filter type of the dataset.
      Usage: Passed from caller.
    - voltage (str): Tube voltage descriptor (e.g., '200kV').
      Type: str
      Meaning: The tube voltage.
      Usage: Passed from caller.
    - data_dir (str): Path to the pixel values directory.
      Type: str
      Meaning: The path to the folder containing pickle files.
      Usage: E.g., 'results/20260429_mask_generated_16bit/pixel_values'.
    - I0 (float): Incident intensity background reference (16-bit defaults to 52428.0).
      Type: float
      Meaning: The reference intensity before the wedge.
      Usage: E.g., 52428.0.
    - cu_fe_max_steps (int): The maximum number of thickness steps to load for Cu and Fe.
      Type: int
      Meaning: Limits the steps of heavy materials (Cu, Fe) to avoid noise from unpenetrated thick steps.
      Usage: E.g., 1, 3, 5, 7, or 10.
      
    Returns:
    - tuple: (mat_list, thick_list_cm, T_L_list, T_H_list, step_info_list)
      - mat_list: list of str (material names: 'Al', 'Fe', 'Cu')
      - thick_list_cm: np.ndarray (thicknesses in cm)
      - T_L_list: np.ndarray (low-energy transmission ratios)
      - T_H_list: np.ndarray (high-energy transmission ratios)
      - step_info_list: list of dict (metadata about each loaded step)
    """
    mat_list = []
    thick_list_cm = []
    T_L_list = []
    T_H_list = []
    step_info_list = []
    
    for mat_name in ['Al_step', 'Fe_step', 'Cu_step']:
        file_name = f"{mat_name}-calib-{f_type}-{voltage}-2mA-orig_step_sample_0_data.pkl"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        l_list = data['pixels_low']
        h_list = data['pixels_high']
        t_mm = THICKNESS_MAP[mat_name]
        
        max_steps = cu_fe_max_steps if (mat_name in ['Fe_step', 'Cu_step']) else 10
        for s in range(min(len(l_list), len(t_mm), max_steps)):
            l_v = l_list[s]
            h_v = h_list[s]
            
            # Filter out dead or saturated pixels
            mask = (l_v > 0) & (h_v > 0) & (l_v < 65535) & (h_v < 65535)
            l_val = l_v[mask].astype(float)
            h_val = h_v[mask].astype(float)
            
            if len(l_val) == 0:
                continue
                
            T_L = np.mean(l_val) / I0
            T_H = np.mean(h_val) / I0
            
            if T_L > 0 and T_H > 0:
                symbol = NIST_SYMBOLS[mat_name]
                mat_list.append(symbol)
                thick_cm = t_mm[s] / 10.0
                thick_list_cm.append(thick_cm)
                T_L_list.append(T_L)
                T_H_list.append(T_H)
                step_info_list.append({
                    'material': symbol,
                    'thickness_mm': t_mm[s],
                    'thickness_cm': thick_cm,
                    'T_L': T_L,
                    'T_H': T_H,
                    'low_mean': np.mean(l_val),
                    'high_mean': np.mean(h_val)
                })
                
    return mat_list, np.array(thick_list_cm), np.array(T_L_list), np.array(T_H_list), step_info_list


def build_system_matrix(materials: list, thicknesses_cm: np.ndarray, energies_keV: np.ndarray) -> np.ndarray:
    """
    Constructs the system matrix A of size N x M for spectrum reconstruction.
    
    Parameters:
    - materials (list of str): List of element symbols for each measurement.
      Type: list of str
      Meaning: The element symbol corresponding to each measurement row (e.g. ['Al', 'Al', ..., 'Cu']).
      Usage: Passed from the loaded dataset.
    - thicknesses_cm (np.ndarray): Array of step thicknesses in cm.
      Type: np.ndarray (float)
      Meaning: The thickness (cm) of each measurement step.
      Usage: Derived from thickness_map.
    - energies_keV (np.ndarray): Array of energy bin values in keV.
      Type: np.ndarray (float)
      Meaning: The energy grid used for reconstruction.
      Usage: E.g., np.arange(15, 201, 5).
      
    Returns:
    - np.ndarray: System matrix A of shape (N, M) where A[k, i] = exp(-mu_k(E_i) * d_k).
    """
    N = len(materials)
    M = len(energies_keV)
    A = np.zeros((N, M))
    
    unique_mats = list(set(materials))
    mu_dict = {}
    for mat in unique_mats:
        mu_dict[mat] = get_linear_attenuation(mat, energies_keV, DENSITIES[mat])
        
    for k in range(N):
        mat = materials[k]
        d = thicknesses_cm[k]
        mu = mu_dict[mat]
        A[k, :] = np.exp(-mu * d)
        
    return A

# ==============================================================================
# Spectrum Reconstruction Logic
# ==============================================================================

def reconstruct_channel_spectrum(A: np.ndarray, T: np.ndarray, energies_keV: np.ndarray, 
                                 lambda_val: float = 0.005, gamma: float = 20.0, beta: float = 10.0) -> np.ndarray:
    """
    Reconstructs the X-ray spectrum for a single channel (LE or HE) using regularized NNLS,
    incorporating a Duane-Hunt soft decay constraint at high energies to eliminate spurious peaks.
    
    Parameters:
    - A (np.ndarray): The system matrix of shape (N, M).
      Type: np.ndarray (float)
      Meaning: The matrix mapping energy bin intensities to transmission values.
      Usage: Constructed using build_system_matrix.
    - T (np.ndarray): The observed transmission vector of shape (N,).
      Type: np.ndarray (float)
      Meaning: The measured transmission ratio I/I0 for each step.
      Usage: Calculated from average step wedge intensities.
    - energies_keV (np.ndarray): Array of energy bin values in keV.
      Type: np.ndarray (float)
      Meaning: The energy grid.
      Usage: E.g., np.arange(15, 201, 5).
    - lambda_val (float): Regularization parameter for spectrum smoothness (second-difference penalty).
      Type: float
      Meaning: Controls the tradeoff between matching data and spectrum smoothness.
      Usage: E.g., 0.005.
    - gamma (float): Normalization constraint weight.
      Type: float
      Meaning: Forces the sum of spectrum elements to be 1.
      Usage: E.g., 20.0.
    - beta (float): Boundary constraint weight.
      Type: float
      Meaning: Forces the start and end bins to be 0.
      Usage: E.g., 10.0.
      
    中文参数说明：
    - A (np.ndarray): 大小为 (N, M) 的前向系统投影矩阵，A[j, i] 表示能量为 E_i 的光子穿过第 j 阶梯时的理论透射率。
    - T (np.ndarray): 大小为 (N,) 的实测透射率向量 (I/I0)。
    - energies_keV (np.ndarray): 离散重建能谱的能量仓中心坐标数组，单位为 keV。
    - lambda_val (float): Tikhonov 二阶差分平滑正则化参数，用于惩罚能谱剧烈振荡，使谱线平滑。
    - gamma (float): 归一化约束的权重因子，强迫所有能量组分的概率和为 1。
    - beta (float): 两端边界（最低能量和最高能量点）归零约束的权重因子。
      
    Returns:
    - np.ndarray: Reconstructed spectrum S of shape (M,), normalized to sum to 1.
    """
    M = len(energies_keV)
    
    # 2nd difference matrix for smoothness
    D = np.zeros((M - 2, M))
    for i in range(M - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
        
    # Boundary constraints
    row_bound_start = np.zeros((1, M))
    row_bound_start[0, 0] = beta
    row_bound_end = np.zeros((1, M))
    row_bound_end[0, -1] = beta
    
    # Sum constraint (sum(S) = 1)
    row_norm = gamma * np.ones((1, M))
    
    # Duane-Hunt soft decay envelope constraint at high energies (E > 0.85 * E_max)
    E_max = energies_keV[-1]
    E_thresh = 0.85 * E_max
    decay_rows = []
    
    # We add soft decay constraints: for bins with energy E_i > E_thresh, we penalize S_i
    # with a weight that increases quadratically towards E_max.
    for i in range(M):
        E_i = energies_keV[i]
        if E_i > E_thresh:
            row = np.zeros(M)
            # Quadratic decay penalty weight towards the high energy boundary
            weight = beta * ((E_i - E_thresh) / (E_max - E_thresh)) ** 2
            row[i] = weight
            decay_rows.append(row)
            
    # Include low-energy soft decay below 35 keV if available
    E_min = energies_keV[0]
    E_low_thresh = 35.0
    if E_low_thresh > E_min:
        for i in range(M):
            E_i = energies_keV[i]
            if E_i < E_low_thresh:
                row = np.zeros(M)
                # Quadratic decay penalty weight towards the low energy boundary
                weight = beta * ((E_low_thresh - E_i) / (E_low_thresh - E_min)) ** 2
                row[i] = weight
                decay_rows.append(row)
                
    if len(decay_rows) > 0:
        decay_matrix = np.vstack(decay_rows)
        num_decay = decay_matrix.shape[0]
        
        A_aug = np.vstack([
            A,
            row_norm,
            row_bound_start,
            row_bound_end,
            decay_matrix,
            np.sqrt(lambda_val) * D
        ])
        
        T_aug = np.concatenate([
            T,
            [gamma],
            [0.0],
            [0.0],
            np.zeros(num_decay),
            np.zeros(M - 2)
        ])
    else:
        # Augment A and T without decay matrix
        A_aug = np.vstack([
            A,
            row_norm,
            row_bound_start,
            row_bound_end,
            np.sqrt(lambda_val) * D
        ])
        
        T_aug = np.concatenate([
            T,
            [gamma],
            [0.0],
            [0.0],
            np.zeros(M - 2)
        ])
    
    # Solve NNLS
    S, _ = scipy.optimize.nnls(A_aug, T_aug)
    
    # Ensure exact normalization
    sum_S = np.sum(S)
    if sum_S > 0:
        S = S / sum_S
        
    return S


def reconstruct_channel_spectrum_method2(step_info_list: list, energies_keV: np.ndarray, 
                                         voltage_kv: float, channel: str = 'low') -> np.ndarray:
    """
    Reconstructs the X-ray spectrum for a single channel (LE or HE) using Method 2: 
    Adjacent thickness transmission differences mapping to peak sensitivity energy.
    
    Parameters:
    - step_info_list (list): Metadata of all loaded steps.
      Type: list of dict
      Usage: Pass list of step dictionary objects.
    - energies_keV (np.ndarray): Target energy grid.
      Type: np.ndarray (float)
      Usage: E.g. np.arange(15, 201, 5).
    - voltage_kv (float): Max tube voltage.
      Type: float
      Usage: E.g., 200.0.
    - channel (str): Energy channel, 'low' or 'high'.
      Type: str
      Usage: Pass 'low' or 'high'.
      
    Returns:
    - np.ndarray: Reconstructed spectrum S of shape (M,), normalized.
    """
    mats_data = {}
    for info in step_info_list:
        mat = info['material']
        if mat not in mats_data:
            mats_data[mat] = []
        mats_data[mat].append(info)
        
    for mat in mats_data:
        mats_data[mat] = sorted(mats_data[mat], key=lambda x: x['thickness_cm'])
        
    sampled_energies = []
    sampled_S = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'nist_data')
    
    for mat, steps in mats_data.items():
        csv_path = os.path.join(data_dir, f"{mat}_mu_rho.csv")
        if not os.path.exists(csv_path):
            get_mu_from_nist_new.save_mu_rho_to_local(mat, data_dir)
            
        e_raw, mu_raw = [], []
        with open(csv_path, 'r') as f:
            import csv
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                e_raw.append(float(row[0]) * 1000.0)  # MeV -> keV
                mu_raw.append(float(row[1]) * DENSITIES[mat])  # cm^2/g -> cm^-1
                
        e_raw = np.array(e_raw)
        mu_raw = np.array(mu_raw)
        
        for i in range(1, len(e_raw)):
            if e_raw[i] <= e_raw[i-1]:
                e_raw[i] = e_raw[i-1] + 1e-9
                
        sort_idx = np.argsort(mu_raw)
        sorted_mu = mu_raw[sort_idx]
        sorted_e = e_raw[sort_idx]
        mu_to_E = interp1d(np.log10(sorted_mu), sorted_e, kind='linear', fill_value='extrapolate')
        
        for i in range(len(steps) - 1):
            d_j = steps[i]['thickness_cm']
            d_j1 = steps[i+1]['thickness_cm']
            T_j = steps[i]['T_L'] if channel == 'low' else steps[i]['T_H']
            T_j1 = steps[i+1]['T_L'] if channel == 'low' else steps[i+1]['T_H']
            
            delta_T = T_j - T_j1
            if delta_T <= 0:
                continue
                
            mu_star = np.log(d_j1 / d_j) / (d_j1 - d_j)
            
            try:
                E_star = float(mu_to_E(np.log10(mu_star)))
            except:
                continue
                
            E_fine = np.arange(15.0, voltage_kv + 1.0, 1.0)
            _, _, mu_rho_fine = get_mu_from_nist_new.get_mu_rho_interpolated(mat, E_fine, data_dir=data_dir)
            mu_fine = mu_rho_fine * DENSITIES[mat]
            g_fine = np.exp(-mu_fine * d_j) - np.exp(-mu_fine * d_j1)
            C_j = np.trapz(g_fine, E_fine)
            
            if C_j > 0:
                S_est = delta_T / C_j
                sampled_energies.append(E_star)
                sampled_S.append(S_est)
                
    valid_E = []
    valid_S = []
    for e, s in zip(sampled_energies, sampled_S):
        if 15.0 <= e <= voltage_kv and s > 0:
            valid_E.append(e)
            valid_S.append(s)
            
    if len(valid_E) < 2:
        S_fallback = (voltage_kv - energies_keV) * energies_keV
        S_fallback = np.clip(S_fallback, 0, None)
        return S_fallback / np.sum(S_fallback)
        
    valid_E = np.array(valid_E)
    valid_S = np.array(valid_S)
    sort_idx = np.argsort(valid_E)
    valid_E = valid_E[sort_idx]
    valid_S = valid_S[sort_idx]
    
    all_E = []
    all_S = []
    if valid_E[0] > 15.0:
        all_E.append(15.0)
        all_S.append(0.0)
    for e, s in zip(valid_E, valid_S):
        all_E.append(e)
        all_S.append(s)
    if valid_E[-1] < voltage_kv:
        all_E.append(voltage_kv)
        all_S.append(0.0)
        
    all_E = np.array(all_E)
    all_S = np.array(all_S)
    
    try:
        from scipy.interpolate import PchipInterpolator
        unique_E, unique_idx = np.unique(all_E, return_index=True)
        unique_S = all_S[unique_idx]
        interp = PchipInterpolator(unique_E, unique_S)
        S_interp = interp(energies_keV)
    except:
        unique_E, unique_idx = np.unique(all_E, return_index=True)
        unique_S = all_S[unique_idx]
        interp = interp1d(unique_E, unique_S, kind='linear', fill_value='extrapolate')
        S_interp = interp(energies_keV)
        
    S_interp = np.clip(S_interp, 0, None)
    
    sum_S = np.sum(S_interp)
    if sum_S > 0:
        S_interp = S_interp / sum_S
        
    return S_interp


def reconstruct_joint_spectra(A_list: list, T_list: list, energies_list: list,
                              lambda_smooth: float = 0.08, lambda_joint: float = 0.05,
                              gamma: float = 20.0, beta: float = 10.0) -> list:
    """
    Method 3: Joint-NNLS (多电压联合正则化非负最小二乘法)
    同时重构多个电压下的 X 射线能谱，通过跨电压能谱相似性约束进行联合求解，以抑制高能噪声和虚假峰。
    
    参数说明：
    - A_list (list of np.ndarray): 每个电压的前向系统投影矩阵列表。
      类型：list of np.ndarray (float)
      含义：包含不同电压下对应标样透射映射矩阵的列表。
      用法：传入 [A_200kV, A_220kV, ...] 列表。
    - T_list (list of np.ndarray): 每个电压的实测透射率向量列表。
      类型：list of np.ndarray (float)
      含义：对应各电压下的测定透射值。
      用法：传入 [T_200kV, T_220kV, ...] 列表。
    - energies_list (list of np.ndarray): 每个电压对应的离散能量网格列表。
      类型：list of np.ndarray (float)
      含义：不同电压下的能量网格，如 200kV grid, 220kV grid。
      用法：传入 [E_200kV, E_220kV, ...] 列表。
    - lambda_smooth (float): 内部单能谱二阶差分平滑因子。
      类型：float
      含义：调节单个能谱平滑度的惩罚系数，默认为 0.08。
    - lambda_joint (float): 跨电压联合相似性约束因子。
      类型：float
      含义：调节相邻电压之间重合能域处能谱强度差异的惩罚系数，默认为 0.05。
    - gamma (float): 归一化约束权重。
      类型：float
      含义：保证每个能谱面积和为 1。
    - beta (float): 边界归零和 Duane-Hunt 软衰减约束权重。
      类型：float
      含义：控制高低能两端快速归零的惩罚权重。
      
    返回：
    - list of np.ndarray: 包含各电压下归一化重构能谱的列表。
    """
    K = len(A_list)
    M_list = [len(e) for e in energies_list]
    offset = [0] * (K + 1)
    for k in range(K):
        offset[k + 1] = offset[k] + M_list[k]
        
    M_total = offset[K]
    
    # 收集所有的增强方程行和目标值
    data_rows = []
    T_aug_parts = []
    
    # 1. 各个电压独立的数据拟合项
    for k in range(K):
        A_k = A_list[k]
        N_k = A_k.shape[0]
        row_block = np.zeros((N_k, M_total))
        row_block[:, offset[k]:offset[k+1]] = A_k
        data_rows.append(row_block)
        T_aug_parts.append(T_list[k])
        
    # 2. 各个电压独立的归一化约束
    for k in range(K):
        row = np.zeros(M_total)
        row[offset[k]:offset[k+1]] = gamma
        data_rows.append(row.reshape(1, -1))
        T_aug_parts.append([gamma])
        
    # 3. 各个电压独立的边界归零约束与软衰减约束
    for k in range(K):
        energies_k = energies_list[k]
        M_k = M_list[k]
        
        # 边界点 (首尾)
        row_start = np.zeros(M_total)
        row_start[offset[k]] = beta
        data_rows.append(row_start.reshape(1, -1))
        T_aug_parts.append([0.0])
        
        row_end = np.zeros(M_total)
        row_end[offset[k+1] - 1] = beta
        data_rows.append(row_end.reshape(1, -1))
        T_aug_parts.append([0.0])
        
        # Duane-Hunt 软截止衰减 (高能与低能端)
        E_max = energies_k[-1]
        E_thresh = 0.85 * E_max
        E_min = energies_k[0]
        E_low_thresh = 35.0
        
        for i in range(M_k):
            E_i = energies_k[i]
            if E_i > E_thresh:
                row = np.zeros(M_total)
                weight = beta * ((E_i - E_thresh) / (E_max - E_thresh)) ** 2
                row[offset[k] + i] = weight
                data_rows.append(row.reshape(1, -1))
                T_aug_parts.append([0.0])
            if E_i < E_low_thresh:
                row = np.zeros(M_total)
                weight = beta * ((E_low_thresh - E_i) / (E_low_thresh - E_min)) ** 2
                row[offset[k] + i] = weight
                data_rows.append(row.reshape(1, -1))
                T_aug_parts.append([0.0])
                
    # 4. 各个电压独立的平滑约束 (二阶差分)
    for k in range(K):
        M_k = M_list[k]
        if M_k > 2:
            D_k = np.zeros((M_k - 2, M_k))
            for i in range(M_k - 2):
                D_k[i, i] = 1.0
                D_k[i, i + 1] = -2.0
                D_k[i, i + 2] = 1.0
            
            row_smooth = np.zeros((M_k - 2, M_total))
            row_smooth[:, offset[k]:offset[k+1]] = np.sqrt(lambda_smooth) * D_k
            data_rows.append(row_smooth)
            T_aug_parts.append(np.zeros(M_k - 2))
            
    # 5. 跨电压联合相似性约束 (相邻电压重合能域处能谱强度差异惩罚)
    for k in range(K - 1):
        energies_k = energies_list[k]
        M_k = M_list[k]
        energies_next = energies_list[k+1]
        
        # 遍历前一个电压的所有能量仓，并在下一个电压寻找相同的能量仓
        for i in range(M_k):
            E_val = energies_k[i]
            # 假设网格完全对齐
            idx_next = np.where(np.abs(energies_next - E_val) < 1e-3)[0]
            if len(idx_next) > 0:
                j = idx_next[0]
                row_joint = np.zeros(M_total)
                # S_{k+1}(E_val) - S_k(E_val) = 0
                row_joint[offset[k+1] + j] = np.sqrt(lambda_joint)
                row_joint[offset[k] + i] = -np.sqrt(lambda_joint)
                data_rows.append(row_joint.reshape(1, -1))
                T_aug_parts.append([0.0])
                
    # 6. 拼接矩阵并调用 NNLS
    A_aug = np.vstack(data_rows)
    T_aug = np.concatenate(T_aug_parts)
    
    # 运行非负最小二乘
    S_joint, _ = scipy.optimize.nnls(A_aug, T_aug)
    
    # 7. 拆分并分别进行归一化
    S_list = []
    for k in range(K):
        S_k = S_joint[offset[k]:offset[k+1]]
        sum_S = np.sum(S_k)
        if sum_S > 0:
            S_k = S_k / sum_S
        S_list.append(S_k)
        
    return S_list


# ==============================================================================
# Feature Extraction Algorithms (Monoenergetic vs Spectrum-Integrated)
# ==============================================================================

def calculate_apd_acd_mono(T_L, T_H, E_L, E_H):
    """
    Calculates APD and ACD features using the standard monoenergetic dual-energy equations.
    
    Parameters:
    - T_L (float or np.ndarray): Transmission ratio for low-energy channel.
      Type: float or np.ndarray
      Usage: Input values.
    - T_H (float or np.ndarray): Transmission ratio for high-energy channel.
      Type: float or np.ndarray
      Usage: Input values.
    - E_L (float): Low equivalent energy in keV.
      Type: float
      Usage: E.g., 58.0.
    - E_H (float): High equivalent energy in keV.
      Type: float
      Usage: E.g., 105.0.
      
    Returns:
    - tuple: (apd, acd) values.
    """
    mu_L_d = -np.log(T_L + 1e-9)
    mu_H_d = -np.log(T_H + 1e-9)
    
    t1 = mu_L_d * fkn(E_H) - mu_H_d * fkn(E_L)
    t2 = fkn(E_H) * (E_L ** -3) - fkn(E_L) * (E_H ** -3)
    apd = t1 / t2
    
    t1_ac = mu_H_d * (E_L ** -3) - mu_L_d * (E_H ** -3)
    acd = t1_ac / t2
    return apd, acd


def solve_apd_acd_nonlinear(T_L, T_H, S_L, S_H, energies_keV):
    """
    Solves the continuous-spectrum transmission equations for apd and acd using scipy.optimize.root.
    
    Parameters:
    - T_L (float or np.ndarray): Measured low-energy transmission ratio.
      Type: float or np.ndarray
      Meaning: The transmission ratio I_L / I_0.
      Usage: Calculated from low-energy channel.
    - T_H (float or np.ndarray): Measured high-energy transmission ratio.
      Type: float or np.ndarray
      Meaning: The transmission ratio I_H / I_0.
      Usage: Calculated from high-energy channel.
    - S_L (np.ndarray): Reconstructed low-energy spectrum.
      Type: np.ndarray (float)
      Meaning: The normalized spectrum of the low-energy channel.
      Usage: Reconstructed using NNLS.
    - S_H (np.ndarray): Reconstructed high-energy spectrum.
      Type: np.ndarray (float)
      Meaning: The normalized spectrum of the high-energy channel.
      Usage: Reconstructed using NNLS.
    - energies_keV (np.ndarray): Energy grid.
      Type: np.ndarray (float)
      Meaning: The energy bin values.
      Usage: E.g., np.arange(15, 201, 5).
      
    Returns:
    - tuple: (apd, acd) calculated path length features.
    """
    fkn_vals = fkn(energies_keV)
    E_cube_inv = energies_keV ** -3
    
    is_array = isinstance(T_L, np.ndarray) or hasattr(T_L, '__len__')
    
    if not is_array:
        # Single value root finding
        def equations(vars_val):
            apd_val, acd_val = vars_val
            exp_term = np.exp(-(apd_val * E_cube_inv + acd_val * fkn_vals))
def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'results/20260429_mask_generated_16bit/pixel_values')
    
    # We will use exactly 4 steps for Cu and Fe during spectrum reconstruction
    # While fixing Al step wedge to use all 10 steps.
    for cu_fe_max_steps in [4]:
        output_dir = os.path.join(script_dir, f'results/thickness_decoupling/energy_hardening/spectrum_reconstruction/CuFe_{cu_fe_max_steps}steps')
        os.makedirs(output_dir, exist_ok=True)
        
        print("==================================================================")
        print(f"Starting X-ray Tube Spectrum Reconstruction from 0429 Step Wedges")
        print(f"Configuration: Al=10 steps, Cu & Fe = {cu_fe_max_steps} steps max")
        print("==================================================================")
        
        results = {}
        
        for f_type in FILTER_TYPES:
            results[f_type] = {}
            
            # 1. 收集该滤片下所有电压的数据，以便运行 Method 3 (Joint-NNLS)
            voltage_data_list = []
            for voltage in VOLTAGES:
                v_int = int(voltage.replace('kV', ''))
                mats, thicks, T_L, T_H, step_info = load_transmission_data(f_type, voltage, data_dir, cu_fe_max_steps=cu_fe_max_steps)
                if len(mats) == 0:
                    continue
                
                energies = np.arange(15, v_int + 1e-3, 10.0)
                A = build_system_matrix(mats, thicks, energies)
                voltage_data_list.append({
                    'voltage': voltage,
                    'v_int': v_int,
                    'mats': mats,
                    'thicks': thicks,
                    'T_L': T_L,
                    'T_H': T_H,
                    'energies': energies,
                    'A': A
                })
                
            if len(voltage_data_list) == 0:
                print(f"[-] No step wedge data found for filter {f_type}. Skipping.")
                continue
                
            print(f"[+] Starting Joint Reconstruction (Method 3: Joint-NNLS) for filter {f_type}...")
            # 2. 分别为低能通道和高能通道运行 Method 3
            A_list = [item['A'] for item in voltage_data_list]
            T_L_list = [item['T_L'] for item in voltage_data_list]
            T_H_list = [item['T_H'] for item in voltage_data_list]
            energies_list = [item['energies'] for item in voltage_data_list]
            
            S_L_m3_list = reconstruct_joint_spectra(A_list, T_L_list, energies_list, lambda_smooth=0.08, lambda_joint=0.05)
            S_H_m3_list = reconstruct_joint_spectra(A_list, T_H_list, energies_list, lambda_smooth=0.08, lambda_joint=0.05)
            
            # 3. 运行 Method 1，并保存/画图
            fig_spec, axes_spec = plt.subplots(1, 2, figsize=(12, 5))
            fig_spec.suptitle(f"Reconstructed X-Ray Spectrum - Filter: {f_type} (Solid: M1 NNLS, Dashed: M3 Joint-NNLS)", fontsize=13, fontweight='bold')
            axes_spec[0].set_title("Low-Energy Channel Spectrum $S_L(E)$")
            axes_spec[0].set_xlabel("Energy (keV)")
            axes_spec[0].set_ylabel("Intensity Fraction (normalized)")
            axes_spec[0].grid(True, linestyle='--', alpha=0.4)
            axes_spec[0].set_yscale('log')
            axes_spec[0].set_ylim(bottom=1e-4)
            
            axes_spec[1].set_title("High-Energy Channel Spectrum $S_H(E)$")
            axes_spec[1].set_xlabel("Energy (keV)")
            axes_spec[1].set_ylabel("Intensity Fraction (normalized)")
            axes_spec[1].grid(True, linestyle='--', alpha=0.4)
            axes_spec[1].set_yscale('log')
            axes_spec[1].set_ylim(bottom=1e-4)
            
            for idx, item in enumerate(voltage_data_list):
                voltage = item['voltage']
                v_int = item['v_int']
                A = item['A']
                T_L = item['T_L']
                T_H = item['T_H']
                energies = item['energies']
                
                # Method 1
                S_L_m1 = reconstruct_channel_spectrum(A, T_L, energies, lambda_val=0.08)
                S_H_m1 = reconstruct_channel_spectrum(A, T_H, energies, lambda_val=0.08)
                
                E_L_eff_m1 = np.sum(S_L_m1 * energies)
                E_H_eff_m1 = np.sum(S_H_m1 * energies)
                
                # Method 3
                S_L_m3 = S_L_m3_list[idx]
                S_H_m3 = S_H_m3_list[idx]
                
                E_L_eff_m3 = np.sum(S_L_m3 * energies)
                E_H_eff_m3 = np.sum(S_H_m3 * energies)
                
                print(f"    [{voltage}] -> Method 1: LE E_eff={E_L_eff_m1:.1f} keV, HE E_eff={E_H_eff_m1:.1f} keV")
                print(f"    [{voltage}] -> Method 3 (Joint-NNLS): LE E_eff={E_L_eff_m3:.1f} keV, HE E_eff={E_H_eff_m3:.1f} keV")
                
                results[f_type][voltage] = {
                    'energies_keV': energies.tolist(),
                    'S_L': S_L_m1.tolist(),
                    'S_H': S_H_m1.tolist(),
                    'E_L_eff_keV': float(E_L_eff_m1),
                    'E_H_eff_keV': float(E_H_eff_m1),
                    # Method 3 (Joint-NNLS)
                    'S_L_m3': S_L_m3.tolist(),
                    'S_H_m3': S_H_m3.tolist(),
                    'E_L_eff_m3_keV': float(E_L_eff_m3),
                    'E_H_eff_m3_keV': float(E_H_eff_m3)
                }
                
                color = plt.get_cmap('rainbow')((v_int - 200) / 120.0)
                # Method 1 is solid line
                axes_spec[0].plot(energies, np.maximum(S_L_m1, 1e-6), '-', label=f"{voltage} M1 ({E_L_eff_m1:.1f} keV)", color=color, linewidth=1.5)
                axes_spec[1].plot(energies, np.maximum(S_H_m1, 1e-6), '-', label=f"{voltage} M1 ({E_H_eff_m1:.1f} keV)", color=color, linewidth=1.5)
                
                # Method 3 is dashed line
                axes_spec[0].plot(energies, np.maximum(S_L_m3, 1e-6), '--', label=f"{voltage} M3-Joint ({E_L_eff_m3:.1f} keV)", color=color, linewidth=1.5, alpha=0.8)
                axes_spec[1].plot(energies, np.maximum(S_H_m3, 1e-6), '--', label=f"{voltage} M3-Joint ({E_H_eff_m3:.1f} keV)", color=color, linewidth=1.5, alpha=0.8)
                
            axes_spec[0].legend(fontsize='xx-small', loc='upper right')
            axes_spec[1].legend(fontsize='xx-small', loc='upper right')
            fig_spec.tight_layout()
            spec_fig_path = os.path.join(output_dir, f"reconstructed_spectra_{f_type}.png")
            fig_spec.savefig(spec_fig_path, dpi=200)
            plt.close(fig_spec)
            print(f"[+] Saved spectra plots for filter {f_type}: {spec_fig_path}")
            
            plt.close('all')
            gc.collect()
            
        # 保存重隔出的有效能谱参数到 JSON 文件中
        json_path = os.path.join(output_dir, 'reconstructed_spectra_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[+] Saved reconstructed spectra parameters JSON: {json_path}")
        
        # 绘制有效能量对比图
        fig_eff, axes_eff = plt.subplots(1, 2, figsize=(15, 6))
        fig_eff.suptitle("Effective Energy derived from Reconstructed Spectra vs Voltage", fontsize=14, fontweight='bold')
        
        for f_idx, f_type in enumerate(FILTER_TYPES):
            ax = axes_eff[f_idx]
            voltages_f = []
            E_L_effs_m1 = []
            E_H_effs_m1 = []
            E_L_effs_m3 = []
            E_H_effs_m3 = []
            
            for voltage in VOLTAGES:
                if voltage in results[f_type]:
                    voltages_f.append(int(voltage.replace('kV', '')))
                    E_L_effs_m1.append(results[f_type][voltage]['E_L_eff_keV'])
                    E_H_effs_m1.append(results[f_type][voltage]['E_H_eff_keV'])
                    E_L_effs_m3.append(results[f_type][voltage]['E_L_eff_m3_keV'])
                    E_H_effs_m3.append(results[f_type][voltage]['E_H_eff_m3_keV'])
                    
            if voltages_f:
                ax.plot(voltages_f, E_L_effs_m1, 'o-', label="LE $E_{L, eff}$ (M1)", color='#4A90E2', linewidth=2)
                ax.plot(voltages_f, E_H_effs_m1, 's-', label="HE $E_{H, eff}$ (M1)", color='#E28743', linewidth=2)
                ax.plot(voltages_f, E_L_effs_m3, 'o--', label="LE $E_{L, eff}$ (M3: Joint-NNLS)", color='#4A90E2', linewidth=1.5, alpha=0.8)
                ax.plot(voltages_f, E_H_effs_m3, 's--', label="HE $E_{H, eff}$ (M3: Joint-NNLS)", color='#E28743', linewidth=1.5, alpha=0.8)
                # 绘制 58/105 keV 静态对照参考线
                ax.axhline(58.0, color='#4A90E2', linestyle=':', alpha=0.5, label="Static E_L (58 keV)")
                ax.axhline(105.0, color='#E28743', linestyle=':', alpha=0.5, label="Static E_H (105 keV)")
                
            ax.set_title(f"Filter: {f_type}")
            ax.set_xlabel("Tube Voltage (kV)")
            ax.set_ylabel("Effective Energy (keV)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            
        plt.tight_layout()
        eff_fig_path = os.path.join(output_dir, 'effective_energy_vs_voltage.png')
        plt.savefig(eff_fig_path, dpi=200)
        plt.close(fig_eff)
        print(f"[+] Saved effective energy vs voltage plot: {eff_fig_path}")
        print("==================================================================")
        print(f"Spectrum reconstruction completed for CuFe {cu_fe_max_steps} steps!")
        print("==================================================================")


if __name__ == "__main__":
    main()
