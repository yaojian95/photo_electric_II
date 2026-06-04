import os
import pickle
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize
import get_mu_from_nist_new

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
    Reconstructs the X-ray spectrum for a single channel (LE or HE) using regularized NNLS.
    
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
    
    # Augment A and T
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
            pred_T_L = np.sum(S_L * exp_term)
            pred_T_H = np.sum(S_H * exp_term)
            return [pred_T_L - T_L, pred_T_H - T_H]
            
        # Initial guess using dynamic monoenergetic approximations
        E_L_est = np.sum(S_L * energies_keV)
        E_H_est = np.sum(S_H * energies_keV)
        apd_init, acd_init = calculate_apd_acd_mono(T_L, T_H, E_L_est, E_H_est)
        
        res = scipy.optimize.root(equations, [apd_init, acd_init], method='hybr')
        if res.success:
            return res.x[0], res.x[1]
        else:
            return apd_init, acd_init
    else:
        # Array inputs
        T_L_arr = np.array(T_L)
        T_H_arr = np.array(T_H)
        apd_res = []
        acd_res = []
        for tl, th in zip(T_L_arr, T_H_arr):
            ap, ac = solve_apd_acd_nonlinear(tl, th, S_L, S_H, energies_keV)
            apd_res.append(ap)
            acd_res.append(ac)
        return np.array(apd_res), np.array(acd_res)

# ==============================================================================
# Main Execution & Diagnostics Pipeline
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results/20260429_mask_generated_16bit/pixel_values')
    
    # We will vary the number of Cu and Fe steps to be 1, 3, 5, 7, and 10 steps
    # While fixing Al step wedge to use all 10 steps.
    for cu_fe_max_steps in [1, 3, 5, 7, 10]:
        output_dir = os.path.join(script_dir, f'results/thickness_decoupling/energy_hardening/spectrum_reconstruction/CuFe_{cu_fe_max_steps}steps')
        os.makedirs(output_dir, exist_ok=True)
        
        print("==================================================================")
        print(f"Starting X-ray Tube Spectrum Reconstruction from 0429 Step Wedges")
        print(f"Configuration: Al=10 steps, Cu & Fe = {cu_fe_max_steps} steps max")
        print("==================================================================")
        
        results = {}
        
        for f_type in FILTER_TYPES:
            results[f_type] = {}
            
            # We will create a summary plot of spectra for this filter (Method 1 & 2 Comparison)
            fig_spec, axes_spec = plt.subplots(1, 2, figsize=(14, 6))
            fig_spec.suptitle(f"Reconstructed X-Ray Spectrum - Filter: {f_type} (CuFe {cu_fe_max_steps} steps)", fontsize=14, fontweight='bold')
            axes_spec[0].set_title("Low-Energy Channel Spectrum $S_L(E)$")
            axes_spec[0].set_xlabel("Energy (keV)")
            axes_spec[0].set_ylabel("Intensity Fraction (normalized)")
            axes_spec[0].grid(True, linestyle='--', alpha=0.4)
            
            axes_spec[1].set_title("High-Energy Channel Spectrum $S_H(E)$")
            axes_spec[1].set_xlabel("Energy (keV)")
            axes_spec[1].set_ylabel("Intensity Fraction (normalized)")
            axes_spec[1].grid(True, linestyle='--', alpha=0.4)
            
            # We will create a separate summary plot of spectra for this filter (Method 2 Only)
            fig_spec_m2, axes_spec_m2 = plt.subplots(1, 2, figsize=(14, 6))
            fig_spec_m2.suptitle(f"Reconstructed X-Ray Spectrum (Method 2: Difference) - Filter: {f_type} (CuFe {cu_fe_max_steps} steps)", fontsize=14, fontweight='bold')
            axes_spec_m2[0].set_title("Low-Energy Channel Spectrum $S_L(E)$ (Method 2)")
            axes_spec_m2[0].set_xlabel("Energy (keV)")
            axes_spec_m2[0].set_ylabel("Intensity Fraction (normalized)")
            axes_spec_m2[0].grid(True, linestyle='--', alpha=0.4)
            
            axes_spec_m2[1].set_title("High-Energy Channel Spectrum $S_H(E)$ (Method 2)")
            axes_spec_m2[1].set_xlabel("Energy (keV)")
            axes_spec_m2[1].set_ylabel("Intensity Fraction (normalized)")
            axes_spec_m2[1].grid(True, linestyle='--', alpha=0.4)
            
            for voltage in VOLTAGES:
                v_int = int(voltage.replace('kV', ''))
                
                # Load step data (limited for Cu/Fe, full 10 steps for Al)
                mats, thicks, T_L, T_H, step_info = load_transmission_data(f_type, voltage, data_dir, cu_fe_max_steps=cu_fe_max_steps)
                if len(mats) == 0:
                    print(f"[-] No step wedge data found for {f_type} at {voltage}. Skipping.")
                    continue
                    
                print(f"[+] Loaded {len(mats)} steps (CuFe={cu_fe_max_steps}) for {f_type} at {voltage}.")
                
                # Build energy grid
                energies = np.arange(15, v_int + 1e-3, 5.0)
                
                # Build system matrix
                A = build_system_matrix(mats, thicks, energies)
                
                # Solve for low and high energy spectra (Method 1: Regularized NNLS)
                S_L = reconstruct_channel_spectrum(A, T_L, energies, lambda_val=0.005)
                S_H = reconstruct_channel_spectrum(A, T_H, energies, lambda_val=0.005)
                
                # Solve for low and high energy spectra (Method 2: Adjacent thickness difference mapping)
                S_L_m2 = reconstruct_channel_spectrum_method2(step_info, energies, v_int, 'low')
                S_H_m2 = reconstruct_channel_spectrum_method2(step_info, energies, v_int, 'high')
                
                # Compute incident effective energies (d=0) for Method 1
                E_L_eff = np.sum(S_L * energies)
                E_H_eff = np.sum(S_H * energies)
                
                # Compute incident effective energies (d=0) for Method 2
                E_L_eff_m2 = np.sum(S_L_m2 * energies)
                E_H_eff_m2 = np.sum(S_H_m2 * energies)
                
                print(f"    -> Method 1: Reconstructed LE Spectrum E_eff = {E_L_eff:.2f} keV, HE Spectrum E_eff = {E_H_eff:.2f} keV")
                print(f"    -> Method 2: Reconstructed LE Spectrum E_eff = {E_L_eff_m2:.2f} keV, HE Spectrum E_eff = {E_H_eff_m2:.2f} keV")
                
                results[f_type][voltage] = {
                    'energies_keV': energies.tolist(),
                    'S_L': S_L.tolist(),
                    'S_H': S_H.tolist(),
                    'E_L_eff_keV': float(E_L_eff),
                    'E_H_eff_keV': float(E_H_eff),
                    'S_L_m2': S_L_m2.tolist(),
                    'S_H_m2': S_H_m2.tolist(),
                    'E_L_eff_m2_keV': float(E_L_eff_m2),
                    'E_H_eff_m2_keV': float(E_H_eff_m2)
                }
                
                # Plot spectra
                color = plt.get_cmap('rainbow')( (v_int - 200) / 120.0 )
                # Method 1 is solid line
                axes_spec[0].plot(energies, S_L, label=f"{voltage} M1 (E_eff={E_L_eff:.1f} keV)", color=color, linewidth=2)
                # Method 2 is dashed line of the same color
                axes_spec[0].plot(energies, S_L_m2, '--', label=f"{voltage} M2 (E_eff={E_L_eff_m2:.1f} keV)", color=color, linewidth=1.5, alpha=0.8)
                
                axes_spec[1].plot(energies, S_H, label=f"{voltage} M1 (E_eff={E_H_eff:.1f} keV)", color=color, linewidth=2)
                axes_spec[1].plot(energies, S_H_m2, '--', label=f"{voltage} M2 (E_eff={E_H_eff_m2:.1f} keV)", color=color, linewidth=1.5, alpha=0.8)
                
                # Method 2 separate plot (solid lines representing Method 2 only)
                axes_spec_m2[0].plot(energies, S_L_m2, label=f"{voltage} M2 (E_eff={E_L_eff_m2:.1f} keV)", color=color, linewidth=2)
                axes_spec_m2[1].plot(energies, S_H_m2, label=f"{voltage} M2 (E_eff={E_H_eff_m2:.1f} keV)", color=color, linewidth=2)
                
                # Perform APD/ACD feature evaluation on Al, Fe, Cu steps to check linearity
                # We will plot the comparison for 200kV and 280kV as examples
                if voltage in ['200kV', '280kV']:
                    fig_comp, axes_comp = plt.subplots(2, 3, figsize=(18, 10))
                    fig_comp.suptitle(f"APD/ACD Linearity Comparison for {f_type} at {voltage} (CuFe {cu_fe_max_steps} steps)", fontsize=14, fontweight='bold', y=0.98)
                    
                    # Separate linearity comparison plot for Method 2
                    fig_comp_m2, axes_comp_m2 = plt.subplots(2, 3, figsize=(18, 10))
                    fig_comp_m2.suptitle(f"Method 2 APD/ACD Linearity and Fit for {f_type} at {voltage} (CuFe {cu_fe_max_steps} steps)", fontsize=14, fontweight='bold', y=0.98)
                    
                    # Material mapping for columns
                    col_map = {'Al': 0, 'Fe': 1, 'Cu': 2}
                    
                    for mat_symbol in ['Al', 'Fe', 'Cu']:
                        col = col_map[mat_symbol]
                        
                        # Extract step indices for this material
                        mat_indices = [i for i, x in enumerate(mats) if x == mat_symbol]
                        if not mat_indices:
                            continue
                        
                        d_mm = np.array([step_info[i]['thickness_mm'] for i in mat_indices])
                        t_L_sub = np.array([T_L[i] for i in mat_indices])
                        t_H_sub = np.array([T_H[i] for i in mat_indices])
                        
                        # 1. Method 1: Static Monoenergetic (58 / 105 keV)
                        apd_static, acd_static = calculate_apd_acd_mono(t_L_sub, t_H_sub, E_L=58.0, E_H=105.0)
                        
                        # 2. Method 2: Dynamic Incident Monoenergetic (E_L_eff, E_H_eff)
                        apd_dyn, acd_dyn = calculate_apd_acd_mono(t_L_sub, t_H_sub, E_L=E_L_eff, E_H=E_H_eff)
                        
                        # 3. Method 3: Spectrum-Integrated Nonlinear Optimization (Method 1 Spectrum)
                        apd_nl, acd_nl = solve_apd_acd_nonlinear(t_L_sub, t_H_sub, S_L, S_H, energies)
                        
                        # 4. Method 4: Spectrum-Integrated Nonlinear Optimization (Method 2 Spectrum)
                        apd_nl_m2, acd_nl_m2 = solve_apd_acd_nonlinear(t_L_sub, t_H_sub, S_L_m2, S_H_m2, energies)
                        
                        # Fit lines through origin to measure linearity (R2 through origin)
                        def fit_origin_r2(x, y):
                            slope = np.sum(x * y) / np.sum(x ** 2)
                            y_pred = slope * x
                            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                            return slope, r2
                        
                        # Fit APD
                        slope_ap_static, r2_ap_static = fit_origin_r2(d_mm, apd_static)
                        slope_ap_dyn, r2_ap_dyn = fit_origin_r2(d_mm, apd_dyn)
                        slope_ap_nl, r2_ap_nl = fit_origin_r2(d_mm, apd_nl)
                        slope_ap_nl_m2, r2_ap_nl_m2 = fit_origin_r2(d_mm, apd_nl_m2)
                        
                        # Fit ACD
                        slope_ac_static, r2_ac_static = fit_origin_r2(d_mm, acd_static)
                        slope_ac_dyn, r2_ac_dyn = fit_origin_r2(d_mm, acd_dyn)
                        slope_ac_nl, r2_ac_nl = fit_origin_r2(d_mm, acd_nl)
                        slope_ac_nl_m2, r2_ac_nl_m2 = fit_origin_r2(d_mm, acd_nl_m2)
                        
                        # Subplot (0, col) of Comparison plot: APD vs Thickness
                        ax_ap = axes_comp[0, col]
                        ax_ap.set_title(f"{mat_symbol} APD vs Thickness", fontsize=11)
                        ax_ap.plot(d_mm, apd_static, 'ro-', label=f"Static 58/105 ($R^2$={r2_ap_static:.4f})")
                        ax_ap.plot(d_mm, apd_dyn, 'gs-', label=f"Dyn {E_L_eff:.1f}/{E_H_eff:.1f} ($R^2$={r2_ap_dyn:.4f})")
                        ax_ap.plot(d_mm, apd_nl, 'b^-', label=f"Spectrum NL M1 ($R^2$={r2_ap_nl:.4f})")
                        ax_ap.set_xlabel("Thickness $d$ (mm)")
                        ax_ap.set_ylabel("$apd$")
                        ax_ap.grid(True, linestyle='--', alpha=0.4)
                        ax_ap.legend(fontsize='x-small', loc='best')
                        
                        # Subplot (1, col) of Comparison plot: ACD vs Thickness
                        ax_ac = axes_comp[1, col]
                        ax_ac.set_title(f"{mat_symbol} ACD vs Thickness", fontsize=11)
                        ax_ac.plot(d_mm, acd_static, 'ro-', label=f"Static 58/105 ($R^2$={r2_ac_static:.4f})")
                        ax_ac.plot(d_mm, acd_dyn, 'gs-', label=f"Dyn {E_L_eff:.1f}/{E_H_eff:.1f} ($R^2$={r2_ac_dyn:.4f})")
                        ax_ac.plot(d_mm, acd_nl, 'b^-', label=f"Spectrum NL M1 ($R^2$={r2_ac_nl:.4f})")
                        ax_ac.set_xlabel("Thickness $d$ (mm)")
                        ax_ac.set_ylabel("$acd$")
                        ax_ac.grid(True, linestyle='--', alpha=0.4)
                        ax_ac.legend(fontsize='x-small', loc='best')
                        
                        # Subplot (0, col) of Method 2 plot: APD vs Thickness
                        ax_ap_m2 = axes_comp_m2[0, col]
                        ax_ap_m2.set_title(f"{mat_symbol} APD vs Thickness (M2)", fontsize=11)
                        ax_ap_m2.plot(d_mm, apd_nl_m2, 'm*-', label=f"Spectrum NL M2 ($R^2$={r2_ap_nl_m2:.4f})")
                        ax_ap_m2.plot(d_mm, slope_ap_nl_m2 * d_mm, 'k--', label=f"Fit (slope={slope_ap_nl_m2:.2e})")
                        ax_ap_m2.set_xlabel("Thickness $d$ (mm)")
                        ax_ap_m2.set_ylabel("$apd$")
                        ax_ap_m2.grid(True, linestyle='--', alpha=0.4)
                        ax_ap_m2.legend(fontsize='x-small', loc='best')
                        
                        # Subplot (1, col) of Method 2 plot: ACD vs Thickness
                        ax_ac_m2 = axes_comp_m2[1, col]
                        ax_ac_m2.set_title(f"{mat_symbol} ACD vs Thickness (M2)", fontsize=11)
                        ax_ac_m2.plot(d_mm, acd_nl_m2, 'm*-', label=f"Spectrum NL M2 ($R^2$={r2_ac_nl_m2:.4f})")
                        ax_ac_m2.plot(d_mm, slope_ac_nl_m2 * d_mm, 'k--', label=f"Fit (slope={slope_ac_nl_m2:.2e})")
                        ax_ac_m2.set_xlabel("Thickness $d$ (mm)")
                        ax_ac_m2.set_ylabel("$acd$")
                        ax_ac_m2.grid(True, linestyle='--', alpha=0.4)
                        ax_ac_m2.legend(fontsize='x-small', loc='best')
                        
                    fig_comp.tight_layout()
                    comp_fig_path = os.path.join(output_dir, f"apd_acd_linearity_{f_type}_{voltage}.png")
                    fig_comp.savefig(comp_fig_path, dpi=150)
                    plt.close(fig_comp)
                    print(f"    -> Saved linearity comparison plot: {comp_fig_path}")
                    
                    fig_comp_m2.tight_layout()
                    comp_fig_path_m2 = os.path.join(output_dir, f"apd_acd_linearity_method2_{f_type}_{voltage}.png")
                    fig_comp_m2.savefig(comp_fig_path_m2, dpi=150)
                    plt.close(fig_comp_m2)
                    print(f"    -> Saved Method 2 linearity plot: {comp_fig_path_m2}")
                    
            axes_spec[0].legend(fontsize='x-small', loc='upper right')
            axes_spec[1].legend(fontsize='x-small', loc='upper right')
            fig_spec.tight_layout()
            spec_fig_path = os.path.join(output_dir, f"reconstructed_spectra_{f_type}.png")
            fig_spec.savefig(spec_fig_path, dpi=200)
            plt.close(fig_spec)
            print(f"[+] Saved spectra plots for filter {f_type}: {spec_fig_path}")
            
            axes_spec_m2[0].legend(fontsize='x-small', loc='upper right')
            axes_spec_m2[1].legend(fontsize='x-small', loc='upper right')
            fig_spec_m2.tight_layout()
            spec_fig_path_m2 = os.path.join(output_dir, f"reconstructed_spectra_method2_{f_type}.png")
            fig_spec_m2.savefig(spec_fig_path_m2, dpi=200)
            plt.close(fig_spec_m2)
            print(f"[+] Saved spectra plots for filter {f_type} (Method 2): {spec_fig_path_m2}")
            
        # Save the reconstructed spectra parameters to JSON
        json_path = os.path.join(output_dir, 'reconstructed_spectra_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[+] Saved reconstructed spectra parameters JSON: {json_path}")
        
        # Let's generate a summary plot showing incident effective energy (E_L, E_H) vs Voltage for both filters
        fig_eff, axes_eff = plt.subplots(1, 2, figsize=(15, 6))
        fig_eff.suptitle("Effective Energy derived from Reconstructed Spectra vs Voltage", fontsize=14, fontweight='bold')
        
        for f_idx, f_type in enumerate(FILTER_TYPES):
            ax = axes_eff[f_idx]
            voltages_f = []
            E_L_effs = []
            E_H_effs = []
            E_L_effs_m2 = []
            E_H_effs_m2 = []
            
            for voltage in VOLTAGES:
                if voltage in results[f_type]:
                    voltages_f.append(int(voltage.replace('kV', '')))
                    E_L_effs.append(results[f_type][voltage]['E_L_eff_keV'])
                    E_H_effs.append(results[f_type][voltage]['E_H_eff_keV'])
                    E_L_effs_m2.append(results[f_type][voltage]['E_L_eff_m2_keV'])
                    E_H_effs_m2.append(results[f_type][voltage]['E_H_eff_m2_keV'])
                    
            if voltages_f:
                ax.plot(voltages_f, E_L_effs, 'o-', label="LE Channel $E_{L, eff}$ (M1)", color='#4A90E2', linewidth=2)
                ax.plot(voltages_f, E_L_effs_m2, 'o--', label="LE Channel $E_{L, eff}$ (M2)", color='#4A90E2', linewidth=1.5, alpha=0.8)
                ax.plot(voltages_f, E_H_effs, 's-', label="HE Channel $E_{H, eff}$ (M1)", color='#E28743', linewidth=2)
                ax.plot(voltages_f, E_H_effs_m2, 's--', label="HE Channel $E_{H, eff}$ (M2)", color='#E28743', linewidth=1.5, alpha=0.8)
                # Reference lines for the static 58/105 keV
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
