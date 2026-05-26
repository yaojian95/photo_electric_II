import math
import pickle
import numpy as np

# ---- 兼容性补丁：加载 NumPy 2.x 保存的 pickle 文件 ----
import sys
import types
if not hasattr(np, '_core'):
    np_core_compat = types.ModuleType('numpy._core')
    np_core_compat.__dict__.update({k: v for k, v in np.core.__dict__.items() if not k.startswith('_') or k in ('__all__',)})
    np_core_compat.__path__ = []
    sys.modules['numpy._core'] = np_core_compat
    np._core = np_core_compat
    for subname in ['multiarray', 'umath', 'numeric', 'fromnumeric', 'shape_base', 'records', 'defchararray']:
        submod = getattr(np.core, subname, None)
        if submod is not None:
            full_name = f'numpy._core.{subname}'
            if full_name not in sys.modules:
                sys.modules[full_name] = submod
# -------------------------------------------------------

import os
import cv2
import pandas as pd
import warnings
from typing import List
from datetime import datetime
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class BaggedSVR:
    def __init__(self, estimators, kernels, params_list):
        self.estimators = estimators
        self.kernels = kernels
        self.params_list = params_list

    def fit(self, X, y=None, sample_weight=None):
        # No-op: bagged estimators are already trained before wrapping.
        return self

    def predict(self, X):
        preds = [est.predict(X) for est in self.estimators]
        return np.mean(np.vstack(preds), axis=0)


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===================== 配置 =====================

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#粒子群+模拟退火优化SVM超参数
CONFIG = {
    'data_path': os.path.join(SCRIPT_DIR, "input_0219_0224_0225_0226_0227_contour_th_128_pixels_grades_images_aligned.pkl"),
    'mask_threshold': 128,
    'atomic_number_threshold': 12,          # 平均原子序数阈值 (用于精废判别)
    'save_svm_model': True,
    'svm_model_save_dir': os.path.join(SCRIPT_DIR, "SVM_diaoyong", "models"),
    'test_size': 0.3,
    'random_state': 42,
    'relief_features_num': 30,                 # 选择特征的数量
    'svm_c_range': [0.001, 500],                 # C的上下限
    'svm_gamma_range': [0.001, 1],             # Gamma的上下限
    'svm_epsilon_range': [0.001, 0.2],          # epsilon 的上下限
    # PSO参数 (用于SVM超参优化)
    'pso_n_particles': 40,
    'pso_n_iterations': 12,
    'pso_w': 0.7,
    'pso_c1': 1.4,
    'pso_c2': 1.4,
    'pso_cv_folds': 5,
    'E_L': 30,                               # 低能能量 (keV)
    'E_H': 60,                               # 高能能量 (keV)
    'E_0': 1.0,                              # 参考能量 (keV)
    'Pb_grade_low_threshold': 0.15,           # Pb品位阈值(%), 低于此值视为低品位
    'Zn_grade_low_threshold': 0.15,           # Zn品位阈值(%), 低于此值视为低品位
    'low_grade_size': 0.3,                   # 低品位矿石放入测试集的比例
    'high_grade_size': 0.3,                  # 高品位矿石放入测试集的比例
    'epochs': 15,                            # 迭代训练次数，每轮根据误差更新样本权重
    'weight_alpha': 1.0,                    # 误差样本权重放大系数
    'early_stop_patience': 20,              # 早停：最佳MAE连续多少轮未提升则停止
    'sa_initial_temperature': 0.5,          # 模拟退火初始温度
    'sa_final_temperature': 1e-3,           # 模拟退火最终温度
    'bagging_n_estimators': 2,               # 每个SVM内部基学习器数量(Bagging子模型个数)
    'bagging_bootstrap_ratio': .5,          # 每个基学习器从训练集中随机采样的比例(≤1.0)
    'bagging_kernels': ['rbf'],#['linear', 'rbf', 'poly', 'sigmoid']  # 基学习器可选核函数池
    'bagging_poly_degree': 3,                # poly核的度数
    'bagging_coef0': 1.0,                    # poly/sigmoid核的coef0常数项
    'bagging_random_state': None,            # Bagging采样随机种子
    'otsu_intervals': 100,
    'area_intervals': [0, 2000,4000,6000,8000,10000,float('inf')],
}
FEATURE_SWITCHES = {
    'R_mean0': True, 'R_mean1': True, 'R_mean': True,
    'T_mean0': True, 'T_mean1': True, 'T_mean': True,
    'low_mean0': True, 'low_mean1': True, 'low_mean': True,
    'high_mean0': True, 'high_mean1': True, 'high_mean': True,
    'apd_mean0': True, 'apd_mean1': True, 'apd_mean': True,
    'acd_mean0': True, 'acd_mean1': True, 'acd_mean': True,
    'Zeff_mean0': True, 'Zeff_mean1': True, 'Zeff_mean': True,
    'Ze_mean0': True, 'Ze_mean1': True, 'Ze_mean': True,
    'mu_H_d_mean0': True, 'mu_H_d_mean1': True, 'mu_H_d_mean': True,
    'mu_alpha_mean': True, 'mu_beta_mean': True,
    'R_std': True, 'T_std': True,
    'K_mean0': True, 'K_mean1': True, 'K_mean': True, 'K_std': True,
    'alpha_low_mean0': True, 'alpha_low_mean1': True, 'alpha_low_mean': True,
    'alpha_high_mean0': True, 'alpha_high_mean1': True, 'alpha_high_mean': True,
    'R_grad_mean': True, 'T_grad_mean': True, 'Alpha_grad_mean': True,
    'area': True, 'Thickness_mean': False,
    'area_ratio': False, 'gray_ratio': True, 'alpha_ratio': True,
    'R_IQR': True, 'R_skew': True, 'R_kurt': True,
    'T_IQR': True, 'T_skew': True,
    'Low_GLCM_Contrast': False, 'Low_GLCM_Correlation': False,
}


# ===================== a_p和a_c特征计算 =====================

def _fkn(E_keV: np.ndarray) -> np.ndarray:
	alpha = E_keV / 511.0
	term1 = 2.0 * (1.0 + alpha) ** 2 / (alpha ** 2 * (1.0 + 2.0 * alpha))
	term2 = (np.log(1.0 + 2.0 * alpha) / alpha) * (0.5 - (1.0 + alpha) / (alpha ** 2))
	term3 = (1.0 + 3.0 * alpha) / (1.0 + 2.0 * alpha) ** 2
	return term1 + term2 - term3
def calculate_apd(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能图像计算：光电系数 x 厚度 
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
    根据低能和高能图像计算：康普顿系数 x 厚度 
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
    low = low.astype(float)
    high = high.astype(float)
    apd = calculate_apd(low, high, I0_low, I0_high)
    mu_H_d = np.log(I0_high / (high + 1e-6))
    return apd / mu_H_d

def calculate_Ze(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能图像计算：Ze
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
    根据低能和高能图像计算：mu_H_d
    """
    low = low.astype(float)
    high = high.astype(float)
    mu_H_d = np.log(I0_high / (high + 1e-6))
    return mu_H_d

# ===================== CT特征计算 =====================
def calculate_mu_alpha(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能图像计算：mu_alpha
    """
    low = low.astype(float)
    high = high.astype(float)
    t0 = I0_low / (low + 1e-6) 
    t1 = I0_high / (high + 1e-6)
   
    return -np.log(0.5*(t0+t1))

def calculate_mu_beta(low, high, I0_low=204.293, I0_high=204.199) -> np.ndarray:
    """
    根据低能和高能图像计算：mu_beta
    """
    low = low.astype(float)
    high = high.astype(float)
    t0 = I0_low / (low + 1e-6) 
    t1 = I0_high / (high + 1e-6)
   
    return -np.log(np.sqrt(t0*t1))

# ===================== 双能特征计算 =====================

def calculate_R(low, high, I0_low=204.293, I0_high=204.199, method='a', const=[5, 20]):
    """
    根据低能和高能图像计算R值图
    """
    # 避免除以零
    low = low.astype(float)
    high = high.astype(float)
    
    if method == 'a':
        # R = ln(I0_low / (low + const[0])) / ln(I0_high / (high + const[1]))
        return np.log(I0_low / (low + 1e-6) + const[0]) / np.log(I0_high / (high + 1e-6) + const[1])
    else:
        # 默认回退方法
        return np.log(I0_low / (low + 1e-6)) / np.log(I0_high / (high + 1e-6))

def calculate_T(low, high, I0_low=204.293, I0_high=204.199):
    """
    计算T值图（Alpha特征的差值）。
    低能Alpha特征 = ln(I0_low / low)
    高能Alpha特征 = ln(I0_high / high)
    T = 低能Alpha - 高能Alpha
    """
    low = low.astype(float)
    high = high.astype(float)
    
    # 计算Alpha特征
    # 添加微小量epsilon以避免除以零或对0取对数
    alpha_low = np.log(I0_low / (low + 1e-6))
    alpha_high = np.log(I0_high / (high + 1e-6))
    
    return alpha_low - alpha_high, alpha_low, alpha_high

# ===================== 特征工程 =====================

ALL_FEATURES_ORDER = [
    'R_mean0', 'R_mean1', 'R_mean', 'R_std',
    'T_mean0', 'T_mean1', 'T_mean', 'T_std',
    'low_mean0', 'low_mean1', 'low_mean', 'high_mean0', 'high_mean1', 'high_mean',
    'alpha_low_mean0', 'alpha_low_mean1', 'alpha_low_mean',
    'alpha_high_mean0', 'alpha_high_mean1', 'alpha_high_mean',
    'apd_mean0', 'apd_mean1', 'apd_mean', 'acd_mean0', 'acd_mean1', 'acd_mean',
    'Zeff_mean0', 'Zeff_mean1', 'Zeff_mean', 'Ze_mean0', 'Ze_mean1', 'Ze_mean',
    'mu_H_d_mean0', 'mu_H_d_mean1', 'mu_H_d_mean',
    'mu_alpha_mean', 'mu_beta_mean',
    'K_mean0', 'K_mean1', 'K_mean', 'K_std',
    'area', 'area_ratio',
    'gray_ratio', 'alpha_ratio',
    'R_IQR', 'R_skew', 'R_kurt',
    'T_IQR', 'T_skew',
    'Low_GLCM_Contrast', 'Low_GLCM_Correlation',
    'Thickness_mean',
    'R_grad_mean',
    'T_grad_mean',
    'Alpha_grad_mean',
]

def get_feature_names(feature_switches):
    return [name for name in ALL_FEATURES_ORDER if feature_switches.get(name, False)]

FEATURE_NAMES = get_feature_names(FEATURE_SWITCHES)

def resolve_input_path(p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    candidates = [
        os.path.join(script_dir, p),
        os.path.join(parent_dir, p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    name = os.path.basename(p)
    search_roots = [os.getcwd(), script_dir, parent_dir]
    for root in search_roots:
        try:
            for sub in os.listdir(root):
                sub_path = os.path.join(root, sub)
                if not os.path.isdir(sub_path):
                    continue
                c = os.path.join(sub_path, name)
                if os.path.exists(c):
                    return os.path.abspath(c)
        except Exception:
            continue

    checked = [os.path.abspath(p)] + [os.path.abspath(c) for c in candidates]
    raise FileNotFoundError(
        "找不到输入文件。\n"
        f"- 传入: {p}\n"
        "- 已检查:\n  " + "\n  ".join(checked) + "\n"
        "请把文件放到脚本同级/上一级目录，或直接把 CONFIG['data_path'] 改为绝对路径。"
    )


def _format_for_filename(x):
    if x is None:
        return "None"
    try:
        xf = float(x)
        if np.isfinite(xf) and abs(xf - int(xf)) < 1e-12:
            return str(int(xf))
        return f"{xf:.6g}"
    except Exception:
        return str(x)


def save_svm_model_package(save_dir, scaler, selector, reg, feature_names, feature_scores, config):
    os.makedirs(save_dir, exist_ok=True)
    selected_feature_names = (
        feature_scores.sort_values('Score', ascending=False)['Feature']
        .head(config['relief_features_num'])
        .tolist()
    )
    payload = {
        "scaler": scaler,
        "selector": selector,
        "reg": reg,
        "feature_names": list(feature_names),
        "selected_feature_names": selected_feature_names,
        "feature_switches": dict(FEATURE_SWITCHES),
        "config": {
            "mask_threshold": config.get("mask_threshold"),
            "atomic_number_threshold": config.get("atomic_number_threshold"),
            "relief_features_num": config.get("relief_features_num"),
        },
        "calculate_R_params": {
            "I0_low": 204.293,
            "I0_high": 204.199,
            "method": "a",
            "const": [5, 20],
        },
        "calculate_T_params": {
            "I0_low": 204.293,
            "I0_high": 204.199,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    z_thr = _format_for_filename(config.get("atomic_number_threshold"))
    model_filename = f"SVR_Model_{z_thr}.pkl"
    model_path = os.path.join(save_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"SVM模型已保存至: {model_path}")
    return model_path


def _to_percent_series(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    vmax = float(np.nanmax(v.to_numpy())) if len(v) else 0.0
    if vmax <= 1.2:
        return v * 100.0
    return v


def add_mean_atomic_number(
    df: pd.DataFrame,
    pb_col: str = "Pb_grade",
    zn_col: str = "Zn_grade",
    fe_col: str = "Fe_grade",
    s_col: str = "S_grade",
    out_col: str = "平均原子序数",
) -> pd.DataFrame:
    for c in [pb_col, zn_col, fe_col, s_col]:
        if c not in df.columns:
            raise KeyError(f"缺少品位列: {c}")

    P = _to_percent_series(df[pb_col]).to_numpy(dtype=float)
    Z = _to_percent_series(df[zn_col]).to_numpy(dtype=float)
    F = _to_percent_series(df[fe_col]).to_numpy(dtype=float)
    S = _to_percent_series(df[s_col]).to_numpy(dtype=float)

    other = 100.0 - (P + Z + F + S)
    other = np.clip(other, 0.0, 100.0)
    z_bar = (82.0 * P + 30.0 * Z + 26.0 * F + 16.0 * S + 10.8 * other) / 100.0

    out = df.copy()
    out[out_col] = z_bar
    return out


def _normalize_source_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    if s.startswith("source_"):
        return s
    return f"source_{s}"


def _infer_sample_no_offset(sample_no: np.ndarray, n_imgs: int):
    if sample_no.size == 0:
        return 0
    sn_min = np.nanmin(sample_no)
    sn_max = np.nanmax(sample_no)
    if np.isfinite(sn_min) and np.isfinite(sn_max):
        if sn_min == 1 and sn_max <= n_imgs:
            return 1
        if sn_min == 0 and sn_max < n_imgs:
            return 0
    return None


def get_row_indices_for_source(df: pd.DataFrame, source_key: str, n_imgs: int) -> List[int]:
    if "source" not in df.columns:
        return df.index.to_list()[:n_imgs]

    source_key_norm = _normalize_source_value(source_key)
    src_series = df["source"].apply(_normalize_source_value)
    grp = df.loc[src_series == source_key_norm]
    if grp.empty:
        return df.index.to_list()[:n_imgs]

    if "Sample No." in grp.columns:
        sn = pd.to_numeric(grp["Sample No."], errors="coerce").to_numpy()
        offset = _infer_sample_no_offset(sn, n_imgs)
        if offset is not None and np.all(np.isfinite(sn)):
            sn_int = sn.astype(int)
            mapping = dict(zip(sn_int.tolist(), grp.index.tolist()))
            out = []
            for img_i in range(n_imgs):
                sni = img_i + offset
                if sni not in mapping:
                    return grp.index.to_list()[:n_imgs]
                out.append(int(mapping[sni]))
            return out[:n_imgs]

    grp2 = grp.copy()
    if "Sample No." in grp2.columns:
        grp2["_sn"] = pd.to_numeric(grp2["Sample No."], errors="coerce")
        grp2 = grp2.sort_values("_sn", kind="mergesort")
    return grp2.index.to_list()[:n_imgs]

def _grad_mean_in_mask(map_2d, mask_bool):
    m = map_2d.astype(float)
    gx, gy = np.gradient(m)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[mask_bool]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def otsu_threshold_features(values, intervals=20):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    k_min = np.min(vals)
    k_max = np.max(vals)
    mu_total = float(np.mean(vals))
    if k_min == k_max:
        return k_min, mu_total, mu_total, mu_total
    t_candidates = np.linspace(k_min, k_max, intervals)
    best_sigma2 = -1.0
    best_t, best_mu0, best_mu1 = k_min, mu_total, mu_total
    for t in t_candidates:
        mask0 = vals < t
        p0 = np.mean(mask0)
        if p0 == 0.0 or p0 == 1.0:
            continue
        p1 = 1.0 - p0
        mu0 = float(np.mean(vals[mask0]))
        mu1 = float(np.mean(vals[~mask0]))
        sigma2 = p0 * p1 * (mu0 - mu1) ** 2
        if sigma2 > best_sigma2:
            best_sigma2 = sigma2
            best_t = float(t)
            best_mu0 = mu0
            best_mu1 = mu1
    if best_sigma2 < 0:
        return k_min, mu_total, mu_total, mu_total
    return best_t, best_mu0, best_mu1, mu_total


def calculate_ore_features(low_img, high_img, r_map, t_map, alpha_low, alpha_high, apd_map, acd_map, Zeff_map, Ze_map, mu_H_d_map, mu_alpha_map, mu_beta_map, mask_bool, thickness_val):
    valid_low = low_img[mask_bool]
    valid_high = high_img[mask_bool]
    valid_r = r_map[mask_bool]
    valid_t = t_map[mask_bool]
    valid_alpha_low = alpha_low[mask_bool]
    valid_alpha_high = alpha_high[mask_bool]
    valid_apd = apd_map[mask_bool]
    valid_acd = acd_map[mask_bool]
    valid_Zeff = Zeff_map[mask_bool]
    valid_Ze = Ze_map[mask_bool]
    valid_mu_H_d = mu_H_d_map[mask_bool]
    valid_mu_alpha = mu_alpha_map[mask_bool]
    valid_mu_beta = mu_beta_map[mask_bool]

    if len(valid_r) == 0:
        return None, None

    otsu_intervals = CONFIG['otsu_intervals']

    R_t, R_mean0, R_mean1, R_mean = otsu_threshold_features(valid_r, otsu_intervals)
    R_std = float(np.std(valid_r))
    T_t, T_mean0, T_mean1, T_mean = otsu_threshold_features(valid_t, otsu_intervals)
    T_std = float(np.std(valid_t))
    low_t, low_mean0, low_mean1, low_mean = otsu_threshold_features(valid_low, otsu_intervals)
    high_t, high_mean0, high_mean1, high_mean = otsu_threshold_features(valid_high, otsu_intervals)
    alpha_low_t, alpha_low_mean0, alpha_low_mean1, alpha_low_mean = otsu_threshold_features(valid_alpha_low, otsu_intervals)
    alpha_high_t, alpha_high_mean0, alpha_high_mean1, alpha_high_mean = otsu_threshold_features(valid_alpha_high, otsu_intervals)
    apd_t, apd_mean0, apd_mean1, apd_mean = otsu_threshold_features(valid_apd, otsu_intervals)
    acd_t, acd_mean0, acd_mean1, acd_mean = otsu_threshold_features(valid_acd, otsu_intervals)
    Zeff_t, Zeff_mean0, Zeff_mean1, Zeff_mean = otsu_threshold_features(valid_Zeff, otsu_intervals)
    Ze_t, Ze_mean0, Ze_mean1, Ze_mean = otsu_threshold_features(valid_Ze, otsu_intervals)
    mu_H_d_t, mu_H_d_mean0, mu_H_d_mean1, mu_H_d_mean = otsu_threshold_features(valid_mu_H_d, otsu_intervals)

    mu_alpha_mean = float(np.mean(valid_mu_alpha))
    mu_beta_mean = float(np.mean(valid_mu_beta))

    K_pixel = valid_alpha_low / (valid_alpha_high + 1e-6)
    K_t, K_mean0, K_mean1, K_mean = otsu_threshold_features(K_pixel, otsu_intervals)
    K_std = float(np.std(K_pixel))

    area = float(np.sum(mask_bool))
    area_ratio = area / mask_bool.size
    gray_ratio = low_mean / (high_mean + 1e-6)
    alpha_ratio = alpha_low_mean / (alpha_high_mean + 1e-6)

    R_IQR = float(np.percentile(valid_r, 75) - np.percentile(valid_r, 25))
    R_skew = float(skew(valid_r)) if len(valid_r) > 2 else 0.0
    R_kurt = float(kurtosis(valid_r)) if len(valid_r) > 3 else 0.0
    T_IQR = float(np.percentile(valid_t, 75) - np.percentile(valid_t, 25))
    T_skew = float(skew(valid_t)) if len(valid_t) > 2 else 0.0

    feat_contrast = 0.0
    feat_correlation = 0.0
    if FEATURE_SWITCHES.get('Low_GLCM_Contrast', False) or FEATURE_SWITCHES.get('Low_GLCM_Correlation', False):
        try:
            rows, cols = np.where(mask_bool)
            if len(rows) > 1 and len(cols) > 1:
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                roi = low_img[y_min:y_max+1, x_min:x_max+1]
                if roi.shape[0] >= 2 and roi.shape[1] >= 2:
                    roi_q = (roi / 255.0 * 31).astype(np.uint8)
                    glcm = graycomatrix(
                        roi_q,
                        [1],
                        [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32,
                        normed=True,
                        symmetric=True
                    )
                    feat_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
                    feat_correlation = float(np.mean(graycoprops(glcm, 'correlation')))
        except Exception:
            feat_contrast = 0.0
            feat_correlation = 0.0

    r_grad_mean = _grad_mean_in_mask(r_map, mask_bool) if FEATURE_SWITCHES.get('R_grad_mean', False) else 0.0
    t_grad_mean = _grad_mean_in_mask(t_map, mask_bool) if FEATURE_SWITCHES.get('T_grad_mean', False) else 0.0
    alpha_grad_mean = 0.0
    if FEATURE_SWITCHES.get('Alpha_grad_mean', False):
        alpha_map = (alpha_low + alpha_high) / 2.0
        alpha_grad_mean = _grad_mean_in_mask(alpha_map, mask_bool)

    feature_values = {
        'R_mean0': R_mean0, 'R_mean1': R_mean1, 'R_mean': R_mean,
        'R_std': R_std,
        'T_mean0': T_mean0, 'T_mean1': T_mean1, 'T_mean': T_mean,
        'T_std': T_std,
        'low_mean0': low_mean0, 'low_mean1': low_mean1, 'low_mean': low_mean,
        'high_mean0': high_mean0, 'high_mean1': high_mean1, 'high_mean': high_mean,
        'alpha_low_mean0': alpha_low_mean0, 'alpha_low_mean1': alpha_low_mean1, 'alpha_low_mean': alpha_low_mean,
        'alpha_high_mean0': alpha_high_mean0, 'alpha_high_mean1': alpha_high_mean1, 'alpha_high_mean': alpha_high_mean,
        'apd_mean0': apd_mean0, 'apd_mean1': apd_mean1, 'apd_mean': apd_mean,
        'acd_mean0': acd_mean0, 'acd_mean1': acd_mean1, 'acd_mean': acd_mean,
        'Zeff_mean0': Zeff_mean0, 'Zeff_mean1': Zeff_mean1, 'Zeff_mean': Zeff_mean,
        'Ze_mean0': Ze_mean0, 'Ze_mean1': Ze_mean1, 'Ze_mean': Ze_mean,
        'mu_H_d_mean0': mu_H_d_mean0, 'mu_H_d_mean1': mu_H_d_mean1, 'mu_H_d_mean': mu_H_d_mean,
        'mu_alpha_mean': mu_alpha_mean,
        'mu_beta_mean': mu_beta_mean,
        'K_mean0': K_mean0, 'K_mean1': K_mean1, 'K_mean': K_mean,
        'K_std': K_std,
        'area': area,
        'area_ratio': float(area_ratio),
        'gray_ratio': float(gray_ratio),
        'alpha_ratio': float(alpha_ratio),
        'R_IQR': R_IQR,
        'R_skew': R_skew,
        'R_kurt': R_kurt,
        'T_IQR': T_IQR,
        'T_skew': T_skew,
        'Low_GLCM_Contrast': feat_contrast,
        'Low_GLCM_Correlation': feat_correlation,
        'Thickness_mean': float(thickness_val),
        'R_grad_mean': float(r_grad_mean),
        'T_grad_mean': float(t_grad_mean),
        'Alpha_grad_mean': float(alpha_grad_mean),
    }
    thresholds = {
        'R_t': R_t, 'T_t': T_t, 'low_t': low_t, 'high_t': high_t,
        'alpha_low_t': alpha_low_t, 'alpha_high_t': alpha_high_t,
        'apd_t': apd_t, 'acd_t': acd_t,
        'Zeff_t': Zeff_t, 'Ze_t': Ze_t, 'mu_H_d_t': mu_H_d_t,
        'K_t': K_t,
    }
    return [feature_values[name] for name in FEATURE_NAMES], thresholds

# ===================== 指标计算 =====================

def calculate_ore_metrics(y_true, y_pred, weights, grades_pb, grades_zn, grades_fe, grades_s, grades_pbzn):
    """
    计算矿石分选指标：抛废率、回收率、富集比
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)
    grades_pb = np.array(grades_pb)
    grades_zn = np.array(grades_zn)
    grades_fe = np.array(grades_fe)
    grades_s = np.array(grades_s)
    grades_pbzn = np.array(grades_pbzn)
    
    # 预测为废石（0）和精矿（1）的掩膜
    pred_waste_mask = y_pred == 0
    pred_conc_mask = y_pred == 1
    
    # 基础数据提取
    weights_waste = weights[pred_waste_mask]
    weights_conc = weights[pred_conc_mask]
    
    # 品位数据提取
    grades_pb_waste = grades_pb[pred_waste_mask]
    grades_zn_waste = grades_zn[pred_waste_mask]
    grades_fe_waste = grades_fe[pred_waste_mask]
    grades_s_waste = grades_s[pred_waste_mask]
    grades_pbzn_waste = grades_pbzn[pred_waste_mask]
    
    grades_pb_conc = grades_pb[pred_conc_mask]
    grades_zn_conc = grades_zn[pred_conc_mask]
    grades_fe_conc = grades_fe[pred_conc_mask]
    grades_s_conc = grades_s[pred_conc_mask]
    grades_pbzn_conc = grades_pbzn[pred_conc_mask]
    
    # ================= 1. 重量计算 =================
    total_weight = np.sum(weights)
    total_waste_weight = np.sum(weights_waste)
    total_conc_weight = np.sum(weights_conc)
    
    if total_weight == 0:
        return {} # 避免除零错误
        
    # ================= 2. 品位计算 (加权平均) =================
    
    # 精矿品位 = Σ(精矿重量 * 精矿品位) / 精矿总重量
    if total_conc_weight > 0:
        avg_grade_pb_conc = np.sum(weights_conc * grades_pb_conc) / total_conc_weight
        avg_grade_zn_conc = np.sum(weights_conc * grades_zn_conc) / total_conc_weight
        avg_grade_fe_conc = np.sum(weights_conc * grades_fe_conc) / total_conc_weight
        avg_grade_s_conc = np.sum(weights_conc * grades_s_conc) / total_conc_weight
        avg_grade_pbzn_conc = np.sum(weights_conc * grades_pbzn_conc) / total_conc_weight
    else:
        avg_grade_pb_conc = 0
        avg_grade_zn_conc = 0
        avg_grade_fe_conc = 0
        avg_grade_s_conc = 0
        avg_grade_pbzn_conc = 0
        
    # 废石品位 = Σ(废石重量 * 废石品位) / 废石总重量
    if total_waste_weight > 0:
        avg_grade_pb_waste = np.sum(weights_waste * grades_pb_waste) / total_waste_weight
        avg_grade_zn_waste = np.sum(weights_waste * grades_zn_waste) / total_waste_weight
        avg_grade_fe_waste = np.sum(weights_waste * grades_fe_waste) / total_waste_weight
        avg_grade_s_waste = np.sum(weights_waste * grades_s_waste) / total_waste_weight
        avg_grade_pbzn_waste = np.sum(weights_waste * grades_pbzn_waste) / total_waste_weight
    else:
        avg_grade_pb_waste = 0
        avg_grade_zn_waste = 0
        avg_grade_fe_waste = 0
        avg_grade_s_waste = 0
        avg_grade_pbzn_waste = 0
        
    # 原矿品位 = Σ(所有矿石重量 * 品位) / 总重量
    avg_grade_pb_raw = np.sum(weights * grades_pb) / total_weight
    avg_grade_zn_raw = np.sum(weights * grades_zn) / total_weight
    avg_grade_fe_raw = np.sum(weights * grades_fe) / total_weight
    avg_grade_s_raw = np.sum(weights * grades_s) / total_weight
    avg_grade_pbzn_raw = np.sum(weights * grades_pbzn) / total_weight
    
    # ================= 3. 指标计算 =================
    
    # 抛废率 = 废矿重量 / 总重量
    scrap_rate = total_waste_weight / total_weight
    
    # 金属量计算
    # 精矿金属量
    metal_pb_conc = np.sum(weights_conc * grades_pb_conc)
    metal_zn_conc = np.sum(weights_conc * grades_zn_conc)
    metal_fe_conc = np.sum(weights_conc * grades_fe_conc)
    metal_s_conc = np.sum(weights_conc * grades_s_conc)
    metal_pbzn_conc = np.sum(weights_conc * grades_pbzn_conc)
    
    # 废石金属量
    metal_pb_waste = np.sum(weights_waste * grades_pb_waste)
    metal_zn_waste = np.sum(weights_waste * grades_zn_waste)
    metal_fe_waste = np.sum(weights_waste * grades_fe_waste)
    metal_s_waste = np.sum(weights_waste * grades_s_waste)
    metal_pbzn_waste = np.sum(weights_waste * grades_pbzn_waste)
    
    # 回收率 = (精矿金属量) / (精矿金属量 + 废石金属量)
    recovery_rate_pb = metal_pb_conc / (metal_pb_conc + metal_pb_waste) if (metal_pb_conc + metal_pb_waste) > 0 else 0
    recovery_rate_zn = metal_zn_conc / (metal_zn_conc + metal_zn_waste) if (metal_zn_conc + metal_zn_waste) > 0 else 0
    recovery_rate_fe = metal_fe_conc / (metal_fe_conc + metal_fe_waste) if (metal_fe_conc + metal_fe_waste) > 0 else 0
    recovery_rate_s = metal_s_conc / (metal_s_conc + metal_s_waste) if (metal_s_conc + metal_s_waste) > 0 else 0
    recovery_rate_pbzn = metal_pbzn_conc / (metal_pbzn_conc + metal_pbzn_waste) if (metal_pbzn_conc + metal_pbzn_waste) > 0 else 0
    
    # 富集比 = 精矿品位 / 原矿品位
    enrichment_pb = avg_grade_pb_conc / avg_grade_pb_raw if avg_grade_pb_raw > 0 else 0
    enrichment_zn = avg_grade_zn_conc / avg_grade_zn_raw if avg_grade_zn_raw > 0 else 0
    enrichment_fe = avg_grade_fe_conc / avg_grade_fe_raw if avg_grade_fe_raw > 0 else 0
    enrichment_s = avg_grade_s_conc / avg_grade_s_raw if avg_grade_s_raw > 0 else 0
    enrichment_pbzn = avg_grade_pbzn_conc / avg_grade_pbzn_raw if avg_grade_pbzn_raw > 0 else 0
    
    return {
        '抛废率': scrap_rate,
        'Pb回收率': recovery_rate_pb,
        'Zn回收率': recovery_rate_zn,
        'Fe回收率': recovery_rate_fe,
        'S回收率': recovery_rate_s,
        'Pb+Zn综合回收率': recovery_rate_pbzn,
        'Pb富集比': enrichment_pb,
        'Zn富集比': enrichment_zn,
        'Fe富集比': enrichment_fe,
        'S富集比': enrichment_s,
        'Pb+Zn综合富集比': enrichment_pbzn,
        '精矿Pb品位': avg_grade_pb_conc,
        '精矿Zn品位': avg_grade_zn_conc,
        '精矿S品位': avg_grade_s_conc,
        '精矿Fe品位': avg_grade_fe_conc,
        '精矿Pb+Zn品位': avg_grade_pbzn_conc,
        '废石Pb品位': avg_grade_pb_waste,
        '废石Zn品位': avg_grade_zn_waste,
        '废石S品位': avg_grade_s_waste,
        '废石Fe品位': avg_grade_fe_waste,
        '废石Pb+Zn品位': avg_grade_pbzn_waste,
        '原矿Pb品位': avg_grade_pb_raw,
        '原矿Zn品位': avg_grade_zn_raw,
        '原矿S品位': avg_grade_s_raw,
        '原矿Fe品位': avg_grade_fe_raw,
        '原矿Pb+Zn品位': avg_grade_pbzn_raw,
    }

# ===================== 主程序 =====================

def main():
    data_path = resolve_input_path(CONFIG['data_path'])
    print(f"正在尝试加载数据文件: {data_path}")
    with open(data_path, 'rb') as f:
        input_all = pickle.load(f)

    data = input_all[1]
    if isinstance(data, pd.DataFrame):
        data = data.reset_index(drop=True)
    images_dict = input_all[2]

    thickness_col_candidates = [
        'Mean_thickness',
        'Thickness_mean',
        '厚度均值(mm)',
        '厚度(mm)',
        '厚度',
    ]
    thickness_col = next((c for c in thickness_col_candidates if c in data.columns), None)
    if thickness_col is None:
        print("警告: 未找到厚度列，将使用 0 作为厚度值。")
        thickness_values = np.zeros(len(data), dtype=float)
    else:
        thickness_values = pd.to_numeric(data[thickness_col], errors='coerce').fillna(0).values
        print(f"厚度列: {thickness_col}")

    # ===== 回归标签（平均原子序数，连续值） =====
    if '平均原子序数' not in data.columns:
        data = add_mean_atomic_number(data)

    print(f"平均原子序数 范围: [{data['平均原子序数'].min():.4f}, {data['平均原子序数'].max():.4f}], 均值: {data['平均原子序数'].mean():.4f}")

    y_continuous_all = pd.to_numeric(data['平均原子序数'], errors='coerce').values.astype(float)
    if np.isnan(y_continuous_all).any():
        print("错误: '平均原子序数' 列存在无法转换为数值的项（NaN），请先清洗数据。")
        return

    weights = data['weight'].values if 'weight' in data else np.ones(len(y_continuous_all))

    pb_pct = _to_percent_series(data['Pb_grade'])
    zn_pct = _to_percent_series(data['Zn_grade'])
    fe_pct = _to_percent_series(data['Fe_grade'])
    s_pct = _to_percent_series(data['S_grade'])
    pbzn_pct = _to_percent_series(data['Zn_Pb_grade']) if 'Zn_Pb_grade' in data.columns else (pb_pct + zn_pct)

    grades_pb = pb_pct.values / 100.0
    grades_zn = zn_pct.values / 100.0
    grades_fe = fe_pct.values / 100.0
    grades_s = s_pct.values / 100.0
    grades_pbzn = pbzn_pct.values / 100.0

    X = []
    y = []
    indices = [] # 记录有效样本的原始索引
    all_thresholds = []
    
    print("开始构建特征...")
    
    for source_key in sorted(images_dict.keys()):
        source_data = images_dict[source_key]
        if len(source_data) < 2: continue
        num_ores = len(source_data[0])
        row_indices = get_row_indices_for_source(data, source_key, num_ores)

        for i in range(num_ores):
            if i >= len(row_indices):
                break
            row_idx = int(row_indices[i])
            if row_idx < 0 or row_idx >= len(y_continuous_all):
                continue

            low_img = source_data[0][i]
            high_img = source_data[1][i]

            _, binary = cv2.threshold(low_img.astype(np.uint8), CONFIG['mask_threshold'], 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(low_img, dtype=np.uint8)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue

            r_map = calculate_R(low_img, high_img)
            t_map, alpha_low, alpha_high = calculate_T(low_img, high_img)

            # ===================== a_p和a_c特征计算 =====================
            apd_map = calculate_apd(low_img, high_img)
            acd_map = calculate_acd(low_img, high_img)
            Zeff_map = calculate_Zeff(low_img, high_img)
            Ze_map = calculate_Ze(low_img, high_img)
            mu_H_d_map = calculate_mu_H_d(low_img, high_img)
            # ===================== a_p和a_c特征计算 =====================

            # ===================== CT特征计算 =====================
            mu_alpha_map = mu_alpha(low_img, high_img)
            mu_beta_map = mu_beta(low_img, high_img)
            # ===================== CT特征计算 =====================
            
            # 获取当前样本的厚度值
            current_thickness = thickness_values[row_idx] if row_idx < len(thickness_values) else 0

            features, thresholds = calculate_ore_features(
                low_img, high_img, r_map, t_map, alpha_low, alpha_high,
                apd_map, acd_map, Zeff_map, Ze_map, mu_H_d_map,
                mu_alpha_map, mu_beta_map,
                mask_bool, current_thickness
            )

            if features is not None:
                X.append(features)
                y.append(y_continuous_all[row_idx])
                indices.append(row_idx)
                all_thresholds.append(thresholds)

    X = np.array(X)
    y = np.array(y)
    indices = np.array(indices)

    # 清理数值异常，避免 SelectKBest 因 NaN/Inf 报错
    if X.size > 0:
        bad_mask = ~np.isfinite(X)
        bad_count = int(np.sum(bad_mask))
        if bad_count > 0:
            print(f"警告: 特征矩阵存在 {bad_count} 个 NaN/Inf，已替换为 0。")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"有效样本数: {len(y)} / {len(y_continuous_all)}")
    print(f"有效样本 平均原子序数范围: [{np.min(y):.4f}, {np.max(y):.4f}], 均值: {np.mean(y):.4f}")

    valid_weights = weights[indices]
    valid_grades_pb = grades_pb[indices]
    valid_grades_zn = grades_zn[indices]
    valid_grades_fe = grades_fe[indices]
    valid_grades_s = grades_s[indices]
    valid_grades_pbzn = grades_pbzn[indices]
    
    print(f"原始特征维度: {X.shape[1]}")

    area_idx = FEATURE_NAMES.index('area')
    ore_areas = X[:, area_idx]
    area_bins = CONFIG['area_intervals']
    bin_labels = [f"[{area_bins[i]:.0f}, {area_bins[i+1]:.0f})" for i in range(len(area_bins) - 1)]
    bin_indices = np.digitize(ore_areas, bins=area_bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
    print("\n===== 按像素面积区间统计特征均值 =====")
    for bi, label in enumerate(bin_labels):
        mask_bin = bin_indices == bi
        count = int(np.sum(mask_bin))
        if count == 0:
            print(f"\n面积区间 {label}: 无矿石")
            continue
        print(f"\n面积区间 {label}: {count} 块矿石")
        print("  --- 特征均值 ---")
        for fi, fname in enumerate(FEATURE_NAMES):
            if fname == 'area':
                continue
            vals = X[mask_bin, fi]
            mean_val = np.mean(vals)
            print(f"  {fname}: {mean_val:.6f}")
        print("  --- 最优阈值 t 均值 ---")
        thresh_keys = ['R_t', 'T_t', 'low_t', 'high_t', 'alpha_low_t', 'alpha_high_t',
                       'apd_t', 'acd_t', 'Zeff_t', 'Ze_t', 'mu_H_d_t', 'K_t']
        for tk in thresh_keys:
            t_vals = [all_thresholds[idx][tk] for idx in np.where(mask_bin)[0]]
            if t_vals:
                print(f"  {tk}: {np.mean(t_vals):.6f}")

    # ===================== 特征选择（回归：SelectKBest / f_regression） =====================
    print("\n正在进行特征选择 (SelectKBest / f_regression)...")
    
    # 归一化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k = int(min(CONFIG['relief_features_num'], X_scaled.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    scores = selector.scores_
    if scores is None:
        scores = np.zeros(len(FEATURE_NAMES), dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)

    feature_scores = pd.DataFrame({'Feature': FEATURE_NAMES, 'Score': scores}).sort_values('Score', ascending=False)

    selected_idx = selector.get_support(indices=True)
    selected_feature_names = [FEATURE_NAMES[i] for i in selected_idx]
    selected_set = set(selected_feature_names)

    print("\n===== 全部特征 SelectKBest 重要性排名 =====")
    print(f"{'排名':<4} {'特征名称':<25} {'Score':<12} {'是否选中':<8}")
    print("-" * 50)
    for rank, (_, row) in enumerate(feature_scores.iterrows(), 1):
        flag = "Y" if row['Feature'] in selected_set else "N"
        print(f"{rank:<4} {row['Feature']:<25} {row['Score']:<12.4f} {flag:<8}")
    print(f"\n选中特征 ({len(selected_feature_names)} 个): {selected_feature_names}")
    print(f"选择后特征维度: {X_selected.shape[1]}")

    # ===================== BaggedSVR epochs迭代训练（误差驱动） =====================
    epochs = int(CONFIG.get('epochs', 1))
    weight_alpha = float(CONFIG.get('weight_alpha', 1.0))

    # ---- 基于 Pb+Zn 品位的分层拆分（按比例） ----
    n_total = len(y)
    indices_arr = np.arange(n_total)
    pb_threshold_frac = CONFIG['Pb_grade_low_threshold'] / 100.0
    zn_threshold_frac = CONFIG['Zn_grade_low_threshold'] / 100.0
    low_grade_mask = (valid_grades_pb < pb_threshold_frac) & (valid_grades_zn < zn_threshold_frac)
    low_grade_idx = indices_arr[low_grade_mask]
    high_grade_idx = indices_arr[~low_grade_mask]
    n_low_total = len(low_grade_idx)
    n_high_total = len(high_grade_idx)

    rng = np.random.RandomState(CONFIG['random_state'])
    rng.shuffle(low_grade_idx)
    rng.shuffle(high_grade_idx)
    n_low_test = max(1, int(n_low_total * CONFIG['low_grade_size']))
    n_high_test = max(1, int(n_high_total * CONFIG['high_grade_size']))
    low_test_idx = low_grade_idx[:n_low_test]
    low_train_idx = low_grade_idx[n_low_test:]
    high_test_idx = high_grade_idx[:n_high_test]
    high_train_idx = high_grade_idx[n_high_test:]

    train_idx = np.concatenate([low_train_idx, high_train_idx])
    test_idx = np.concatenate([low_test_idx, high_test_idx])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    print(f"\n===== Pb+Zn品位分层拆分 =====")
    low_grade_pb_pct = valid_grades_pb[low_grade_idx] * 100
    low_grade_zn_pct = valid_grades_zn[low_grade_idx] * 100
    high_grade_pb_pct = valid_grades_pb[high_grade_idx] * 100
    high_grade_zn_pct = valid_grades_zn[high_grade_idx] * 100
    print(f"低品位矿石 (Pb<{CONFIG['Pb_grade_low_threshold']:.2f}% 且 Zn<{CONFIG['Zn_grade_low_threshold']:.2f}%): {n_low_total} 块, Pb范围 [{low_grade_pb_pct.min():.4f}%, {low_grade_pb_pct.max():.4f}%], Zn范围 [{low_grade_zn_pct.min():.4f}%, {low_grade_zn_pct.max():.4f}%]")
    print(f"  放入测试集: {len(low_test_idx)} 块 ({CONFIG['low_grade_size']*100:.0f}%), 放入训练集: {len(low_train_idx)} 块")
    print(f"高品位矿石 (Pb>={CONFIG['Pb_grade_low_threshold']:.2f}% 或 Zn>={CONFIG['Zn_grade_low_threshold']:.2f}%): {n_high_total} 块, Pb范围 [{high_grade_pb_pct.min():.4f}%, {high_grade_pb_pct.max():.4f}%], Zn范围 [{high_grade_zn_pct.min():.4f}%, {high_grade_zn_pct.max():.4f}%]")
    print(f"  放入测试集: {len(high_test_idx)} 块 ({CONFIG['high_grade_size']*100:.0f}%), 放入训练集: {len(high_train_idx)} 块")
    print(f"最终训练集: {len(train_idx)} 块, 测试集: {len(test_idx)} 块")

    train_pb_weighted = np.average(valid_grades_pb[train_idx], weights=valid_weights[train_idx]) * 100
    test_pb_weighted = np.average(valid_grades_pb[test_idx], weights=valid_weights[test_idx]) * 100
    all_pb_weighted = np.average(valid_grades_pb, weights=valid_weights) * 100
    train_zn_weighted = np.average(valid_grades_zn[train_idx], weights=valid_weights[train_idx]) * 100
    test_zn_weighted = np.average(valid_grades_zn[test_idx], weights=valid_weights[test_idx]) * 100
    all_zn_weighted = np.average(valid_grades_zn, weights=valid_weights) * 100
    print(f"训练集加权平均品位: Pb={train_pb_weighted:.4f}%, Zn={train_zn_weighted:.4f}%")
    print(f"测试集加权平均品位: Pb={test_pb_weighted:.4f}%, Zn={test_zn_weighted:.4f}%")
    print(f"全体加权平均品位: Pb={all_pb_weighted:.4f}%, Zn={all_zn_weighted:.4f}%")

    X_train = X_selected[train_idx]
    X_test = X_selected[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_weights = valid_weights[train_idx]
    test_weights = valid_weights[test_idx]
    train_grades_pb = valid_grades_pb[train_idx]
    test_grades_pb = valid_grades_pb[test_idx]
    train_grades_zn = valid_grades_zn[train_idx]
    test_grades_zn = valid_grades_zn[test_idx]
    train_grades_fe = valid_grades_fe[train_idx]
    test_grades_fe = valid_grades_fe[test_idx]
    train_grades_s = valid_grades_s[train_idx]
    test_grades_s = valid_grades_s[test_idx]
    train_grades_pbzn = valid_grades_pbzn[train_idx]
    test_grades_pbzn = valid_grades_pbzn[test_idx]
    y_test_check = y_test

    def _bootstrap_sample_indices(train_indices, n_samples, rng):
        return rng.choice(train_indices, size=n_samples, replace=True)

    def _svr_param_grid_for_kernel(kernel):
        grid = {
            'C': np.logspace(np.log10(CONFIG['svm_c_range'][0]), np.log10(CONFIG['svm_c_range'][1]), 5),
            'epsilon': np.linspace(CONFIG['svm_epsilon_range'][0], CONFIG['svm_epsilon_range'][1], 3),
        }
        if kernel in ('rbf', 'poly', 'sigmoid'):
            grid['gamma'] = np.logspace(np.log10(CONFIG['svm_gamma_range'][0]), np.log10(CONFIG['svm_gamma_range'][1]), 5)
        return grid

    def _format_bagged_params(params_list):
        return '; '.join([', '.join([f"{k}={v}" for k, v in p.items()]) for p in params_list])

    def _pso_optimize_svr(X_boot, y_boot, w_boot, kernel, rng):
        c_min, c_max = CONFIG['svm_c_range']
        e_min, e_max = CONFIG['svm_epsilon_range']
        g_min, g_max = CONFIG['svm_gamma_range']
        logc_min, logc_max = np.log10(c_min), np.log10(c_max)
        logg_min, logg_max = np.log10(g_min), np.log10(g_max)

        use_gamma = kernel in ('rbf', 'poly', 'sigmoid')
        dim = 3 if use_gamma else 2

        n_particles = int(CONFIG.get('pso_n_particles', 20))
        n_iterations = int(CONFIG.get('pso_n_iterations', 30))
        w_inertia = float(CONFIG.get('pso_w', 0.7))
        c1 = float(CONFIG.get('pso_c1', 1.4))
        c2 = float(CONFIG.get('pso_c2', 1.4))
        n_folds = int(CONFIG.get('pso_cv_folds', 3))

        if use_gamma:
            pos = np.column_stack([
                rng.uniform(logc_min, logc_max, size=n_particles),
                rng.uniform(logg_min, logg_max, size=n_particles),
                rng.uniform(e_min, e_max, size=n_particles),
            ])
            v = np.zeros((n_particles, 3), dtype=float)
            bounds = np.array([
                [logc_min, logc_max],
                [logg_min, logg_max],
                [e_min, e_max],
            ], dtype=float)
        else:
            pos = np.column_stack([
                rng.uniform(logc_min, logc_max, size=n_particles),
                rng.uniform(e_min, e_max, size=n_particles),
            ])
            v = np.zeros((n_particles, 2), dtype=float)
            bounds = np.array([
                [logc_min, logc_max],
                [e_min, e_max],
            ], dtype=float)

        pbest_pos = pos.copy()
        pbest_score = np.full(n_particles, np.inf, dtype=float)
        gbest_pos = pos[0].copy()
        gbest_score = np.inf

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=CONFIG.get('random_state'))

        def _eval_position(p):
            if use_gamma:
                logc, logg, eps = p
                params = {
                    'C': 10 ** logc,
                    'gamma': 10 ** logg,
                    'epsilon': eps,
                }
            else:
                logc, eps = p
                params = {
                    'C': 10 ** logc,
                    'epsilon': eps,
                }

            maes = []
            for tr_idx, val_idx in kf.split(X_boot):
                X_tr = X_boot[tr_idx]
                y_tr = y_boot[tr_idx]
                w_tr = w_boot[tr_idx]
                X_val = X_boot[val_idx]
                y_val = y_boot[val_idx]

                model = SVR(
                    kernel=kernel,
                    degree=CONFIG.get('bagging_poly_degree', 3),
                    coef0=CONFIG.get('bagging_coef0', 0.0),
                    **params,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                pred = model.predict(X_val)
                maes.append(mean_absolute_error(y_val, pred))

            return float(np.mean(maes)), params

        for _ in range(n_iterations):
            for i in range(n_particles):
                score, params = _eval_position(pos[i])
                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i] = pos[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = pos[i].copy()

            r1 = rng.rand(n_particles, dim)
            r2 = rng.rand(n_particles, dim)
            v = w_inertia * v + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
            pos = pos + v
            for d in range(dim):
                pos[:, d] = np.clip(pos[:, d], bounds[d, 0], bounds[d, 1])

        _, best_params = _eval_position(gbest_pos)
        return best_params

    def train_bagged_svr_once(X_all, y_all, train_indices, sample_weight, rng):
        kernels = CONFIG.get('bagging_kernels', ['rbf'])
        n_estimators = int(CONFIG.get('bagging_n_estimators', 1))
        bootstrap_ratio = float(CONFIG.get('bagging_bootstrap_ratio', 1.0))
        n_boot = max(1, int(len(train_indices) * bootstrap_ratio))
        weight_map = dict(zip(train_indices.tolist(), sample_weight.astype(float).tolist()))

        estimators = []
        kernels_used = []
        params_list = []

        for _ in range(n_estimators):
            kernel = rng.choice(kernels)
            kernels_used.append(kernel)
            boot_idx = _bootstrap_sample_indices(train_indices, n_boot, rng)
            X_boot = X_all[boot_idx]
            y_boot = y_all[boot_idx].astype(float)
            w_boot = np.array([weight_map[i] for i in boot_idx], dtype=float)

            best_params = _pso_optimize_svr(X_boot, y_boot, w_boot, kernel, rng)
            base = SVR(
                kernel=kernel,
                degree=CONFIG.get('bagging_poly_degree', 3),
                coef0=CONFIG.get('bagging_coef0', 0.0),
                **best_params,
            )
            base.fit(X_boot, y_boot, sample_weight=w_boot)
            estimators.append(base)
            params_list.append(best_params)

        bagged = BaggedSVR(estimators, kernels_used, params_list)
        print(f"\nBagging核函数: {kernels_used}")
        print(f"Bagging参数: {_format_bagged_params(params_list)}")
        return bagged, kernels_used, params_list

    sample_weight_train = np.ones(len(train_idx), dtype=float)

    # ---- 模拟退火 (Simulated Annealing) ----
    sa_T_initial = CONFIG.get('sa_initial_temperature', 1.0)
    sa_T_final = CONFIG.get('sa_final_temperature', 0.01)

    # current: 当前"活跃"模型（SA 游走的状态）
    current_reg = None
    current_test_mae = np.inf
    current_kernels = []
    current_params_list = []

    # best: 全局最优
    best_test_mae = np.inf
    best_reg = None
    best_kernels = []
    best_params_list = []
    no_improve_count = 0
    patience = CONFIG.get('early_stop_patience', 50)

    rng_bag = np.random.RandomState(CONFIG.get('bagging_random_state', None))
    sa_rng = np.random.RandomState(CONFIG.get('bagging_random_state', 42))

    for epoch in range(1, epochs + 1):
        # 余弦退火降温
        progress = (epoch - 1) / max(epochs - 1, 1)
        T = sa_T_final + 0.5 * (sa_T_initial - sa_T_final) * (1 + math.cos(math.pi * progress))

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{epochs}  T={T:.6f}")
        print(f"{'=' * 60}")

        candidate_reg, kernels_used, params_list = train_bagged_svr_once(
            X_selected,
            y,
            train_idx,
            sample_weight_train,
            rng_bag,
        )

        test_pred_z = candidate_reg.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred_z)

        # ---- SA 接受准则 ----
        if current_reg is None:
            accept = True
        elif test_mae < current_test_mae:
            accept = True
        else:
            delta = test_mae - current_test_mae
            accept_prob = math.exp(-delta / T) if T > 1e-12 else 0.0
            accept = sa_rng.random() < accept_prob

        if accept:
            current_reg = candidate_reg
            current_test_mae = test_mae
            current_kernels = kernels_used
            current_params_list = params_list

        # 全局最优跟踪
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_reg = candidate_reg
            best_kernels = kernels_used
            best_params_list = params_list
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 早停判断
        if no_improve_count >= patience:
            print(f"\n早停触发：最佳 MAE 连续 {patience} 轮未提升，提前结束训练")
            print(f"当前最优测试集 MAE = {best_test_mae:.4f}")
            break

        # 使用被接受的模型计算训练误差，更新样本权重
        current_train_pred = current_reg.predict(X_train)
        train_mae = mean_absolute_error(y_train, current_train_pred)
        accept_flag = "✓ 接受" if accept else "✗ 拒绝"
        print(f"训练 MAE={train_mae:.4f}  测试 MAE={test_mae:.4f}  {accept_flag}  |  当前最优 MAE={best_test_mae:.4f}")

        abs_err = np.abs(y_train - current_train_pred)
        err_threshold = np.median(abs_err)
        high_err_mask = abs_err > err_threshold
        sample_weight_train = 1.0 + weight_alpha * high_err_mask.astype(float)

    if best_reg is None:
        best_reg = current_reg
        best_kernels = current_kernels
        best_params_list = current_params_list

    # 回归预测（用全局最优模型）
    train_pred_z = best_reg.predict(X_train)
    test_pred_z = best_reg.predict(X_test)

    # 回归指标
    train_mae = mean_absolute_error(y_train, train_pred_z)
    test_mae = mean_absolute_error(y_test, test_pred_z)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred_z)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred_z)))
    train_r2 = r2_score(y_train, train_pred_z)
    test_r2 = r2_score(y_test, test_pred_z)

    # 基于“平均原子序数阈值”的精废判别（分类派生）
    y_train_bin = (y_train >= CONFIG['atomic_number_threshold']).astype(int)
    y_test_bin = (y_test >= CONFIG['atomic_number_threshold']).astype(int)
    train_pred_bin = (train_pred_z >= CONFIG['atomic_number_threshold']).astype(int)
    test_pred_bin = (test_pred_z >= CONFIG['atomic_number_threshold']).astype(int)

    train_acc = accuracy_score(y_train_bin, train_pred_bin)
    test_acc = accuracy_score(y_test_bin, test_pred_bin)
    
    print("\n===== 最终模型性能 =====")
    print("回归指标：")
    print(f"  训练集: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")
    print(f"  测试集: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
    print("派生分类指标（阈值=平均原子序数）：")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_bin, test_pred_bin)
    print("\n混淆矩阵 (测试集):")
    print(cm)
    print(f"TN (真废石): {cm[0][0]}")
    print(f"FP (误判为精矿): {cm[0][1]}")
    print(f"FN (误判为废石): {cm[1][0]}")
    print(f"TP (真精矿): {cm[1][1]}")

    print("\n===== Permutation Importance (基于测试集) =====")
    k_effective = len(selected_feature_names)
    score_map = feature_scores.set_index('Feature')['Score']
    rel_scores_selected = pd.DataFrame({'Feature': selected_feature_names})
    rel_scores_selected['SelectKBest_Score'] = rel_scores_selected['Feature'].map(score_map).astype(float)

    perm = permutation_importance(
        best_reg,
        X_test,
        y_test,
        n_repeats=20,
        random_state=CONFIG['random_state'],
        scoring='neg_mean_absolute_error'
    )
    perm_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'PermutationImportance_mean': perm.importances_mean,
        'PermutationImportance_std': perm.importances_std
    }).sort_values('PermutationImportance_mean', ascending=False)

    compare_df = perm_df.merge(rel_scores_selected, on='Feature', how='left')
    compare_df['PermutationRank'] = np.arange(1, len(compare_df) + 1)
    compare_df['SelectKBestRank'] = compare_df['SelectKBest_Score'].rank(ascending=False, method='min').astype(int)
    compare_df = compare_df[['PermutationRank', 'SelectKBestRank', 'Feature', 'PermutationImportance_mean', 'PermutationImportance_std', 'SelectKBest_Score']]

    print("\nPermutation importance 排名 (Top 10):")
    print(compare_df.head(10).to_string(index=False))
    print("\nSelectKBest vs Permutation importance（仅对已选择特征）:")
    print(compare_df.to_string(index=False))
    
    # ===================== 计算分选指标 =====================
    print("\n===== 矿石分选指标 (测试集) =====")
    assert np.array_equal(y_test, y_test_check), "y_test 不一致，随机状态可能存在问题"
    metrics = calculate_ore_metrics(
        y_test_bin,
        test_pred_bin,
        test_weights,
        test_grades_pb,
        test_grades_zn,
        test_grades_fe,
        test_grades_s,
        test_grades_pbzn,
    )
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if '率' in key or '品位' in key:
                print(f"{key}: {value*100:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # ===================== 测试集Z值点线图 =====================
    y_test_true = y_test
    test_pred_z_plot = np.asarray(test_pred_z).ravel()

    sort_idx = np.argsort(y_test_true)
    x = np.arange(len(y_test_true))

    plt.figure(figsize=(14, 6))
    plt.plot(x, y_test_true[sort_idx], 'o-', color='black', linewidth=1.5, markersize=3, label='真实Z值')
    plt.plot(x, test_pred_z_plot[sort_idx], 's--', color='blue', linewidth=1, markersize=3, label='SVM预测Z值')
    plt.axhline(y=CONFIG['atomic_number_threshold'], color='green', linestyle='--', linewidth=2, label=f'阈值(Z={CONFIG["atomic_number_threshold"]})')
    plt.xlabel('测试集样本 (按真实Z值升序排列)')
    plt.ylabel('Z值 (原子序数)')
    plt.title('测试集Z值预测对比')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===================== 保存最优模型（测试集 MAE 最小） =====================
    if CONFIG['save_svm_model']:
        save_svm_model_package(
            save_dir=CONFIG['svm_model_save_dir'],
            scaler=scaler,
            selector=selector,
            reg=best_reg,
            feature_names=FEATURE_NAMES,
            feature_scores=feature_scores,
            config=CONFIG,
        )

    # ===================== 保存结果到 Excel =====================
    print("\n正在保存结果到 Excel...")

    X_final = selector.transform(X_scaled)
    all_pred_z = best_reg.predict(X_final)
    all_true_z = y.astype(float)
    all_true_bin = (all_true_z >= CONFIG['atomic_number_threshold']).astype(int)
    all_pred_bin = (all_pred_z >= CONFIG['atomic_number_threshold']).astype(int)

    r_mean_values = np.full(len(y), np.nan, dtype=float)
    if 'R_mean' in FEATURE_NAMES:
        r_mean_values = X[:, FEATURE_NAMES.index('R_mean')]

    valid_thickness = thickness_values[indices]
    valid_atomic_z = pd.to_numeric(data.loc[indices, '平均原子序数'], errors='coerce').values if '平均原子序数' in data.columns else np.full(len(indices), np.nan)
    ore_ids = indices + 1

    indices_arr = np.arange(len(y))
    # 复用分层划分的 train_idx / test_idx
    split_labels = np.empty(len(y), dtype=object)
    split_labels[train_idx] = '训练集'
    split_labels[test_idx] = '测试集'

    is_correct = (all_true_bin == all_pred_bin).astype(int)
    abs_error = np.abs(all_true_z - all_pred_z)

    results_df = pd.DataFrame({
        '矿石序号': ore_ids,
        '平均原子序数(真实)': all_true_z,
        '平均原子序数(预测)': all_pred_z,
        '平均原子序数绝对误差': abs_error,
        '厚度(mm)': valid_thickness,
        'Pb品位': valid_grades_pb * 100,
        'Zn品位': valid_grades_zn * 100,
        'Fe品位': valid_grades_fe * 100,
        'S品位': valid_grades_s * 100,
        'Pb+Zn品位': valid_grades_pbzn * 100,
        'R值均值': r_mean_values,
        '真实标签(1精0废)': all_true_bin,
        '预测标签(1精0废)': all_pred_bin,
        '是否预测正确': is_correct,
        '数据划分': split_labels
    })

    perf_df = pd.DataFrame({
        'Item': [
            'atomic_number_threshold',
            'Pb_grade_low_threshold',
            'Zn_grade_low_threshold',
            'low_grade_size',
            'high_grade_size',
            'random_state',
            'relief_features_num',
            'E_L',
            'E_H',
            'E_0',
            'bagging_kernels',
            'bagging_params',
            'train_mae',
            'test_mae',
            'train_rmse',
            'test_rmse',
            'train_r2',
            'test_r2',
            'train_accuracy_by_threshold',
            'test_accuracy_by_threshold',
            'TN',
            'FP',
            'FN',
            'TP',
        ],
        'Value': [
            CONFIG['atomic_number_threshold'],
            CONFIG['Pb_grade_low_threshold'],
            CONFIG['Zn_grade_low_threshold'],
            CONFIG['low_grade_size'],
            CONFIG['high_grade_size'],
            CONFIG['random_state'],
            CONFIG['relief_features_num'],
            CONFIG['E_L'],
            CONFIG['E_H'],
            CONFIG['E_0'],
            ';'.join(best_kernels),
            _format_bagged_params(best_params_list),
            train_mae,
            test_mae,
            train_rmse,
            test_rmse,
            train_r2,
            test_r2,
            train_acc,
            test_acc,
            int(cm[0][0]),
            int(cm[0][1]),
            int(cm[1][0]),
            int(cm[1][1]),
        ]
    })

    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })

    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PbZn_SVR_Prediction_Results_{CONFIG['atomic_number_threshold']}_{current_time}.xlsx"
    save_path = os.path.join(results_dir, filename)

    with pd.ExcelWriter(save_path) as writer:
        results_df.to_excel(writer, index=False, sheet_name='预测明细')
        compare_df.to_excel(writer, index=False, sheet_name='特征重要性')
        perf_df.to_excel(writer, index=False, sheet_name='模型性能')
        metrics_df.to_excel(writer, index=False, sheet_name='分选指标')

    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
