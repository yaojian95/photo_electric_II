import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure parent directory is in path to import utils_II
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils_II

class SinglePC2ThresholdClassifier:
    """
    单变量 PC2 均值阈值分类器，通过在训练集网格搜索最佳划分阈值来进行精废二分类。
    参数含义：
    - optimize_metric: 优化指标，可为 'count_acc' (数量准确率), 'mass_acc' (质量加权准确率), 或 'balanced_acc' (平衡准确率)。
    """
    def __init__(self, optimize_metric='count_acc', greater_than=False):
        self.threshold = 0.0
        self.optimize_metric = optimize_metric
        self.greater_than = greater_than
        
    def fit(self, X, y, sample_weight=None):
        pc2_vals = X[:, 0]
        best_acc = -1.0
        best_theta = 0.0
        
        # 寻找使得分类效果最好的阈值 θ (判定为精矿条件为 PC2_mean < theta)
        threshold_candidates = np.linspace(np.min(pc2_vals), np.max(pc2_vals), 500)
        for theta in threshold_candidates:
            if self.greater_than:
                y_pred = (pc2_vals > theta).astype(int)
            else:
                y_pred = (pc2_vals < theta).astype(int)
            
            if self.optimize_metric == 'mass_acc' and sample_weight is not None:
                acc = np.sum(sample_weight[y_pred == y]) / np.sum(sample_weight) * 100
            elif self.optimize_metric == 'balanced_acc':
                # 计算平衡准确率：(Sensitivity + Specificity) / 2
                tp = np.sum((y_pred == 1) & (y == 1))
                fn = np.sum((y_pred == 0) & (y == 1))
                tn = np.sum((y_pred == 0) & (y == 0))
                fp = np.sum((y_pred == 1) & (y == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                acc = (tpr + tnr) / 2.0 * 100
            else:
                acc = np.mean(y_pred == y) * 100
                
            if acc > best_acc:
                best_acc = acc
                best_theta = theta
                
        self.threshold = best_theta
        return self
        
    def predict(self, X):
        pc2_vals = X[:, 0]
        if self.greater_than:
            return (pc2_vals > self.threshold).astype(int)
        else:
            return (pc2_vals < self.threshold).astype(int)
        
    def get_params(self, deep=True):
        return {'optimize_metric': self.optimize_metric}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def load_step_data(step_config):
    """
    加载指定阶梯数据字典。
    参数含义：
    - step_config: 配置字典，形如:
      {
          'Cu': {'file_path': '...', 'steps': [0, 1, 2, 3]},
          'Fe': {'file_path': ['path1.pkl', 'path2.pkl'], 'steps': [0, 1]},
          'Al': {'file_path': '...', 'steps': None}
      }
    """
    step_data = {}
    
    for name, config in step_config.items():
        p_paths = config['file_path']
        
        # 兼容单文件和多文件列表模式
        if not isinstance(p_paths, list):
            p_paths = [p_paths]
            
        L_list = []
        H_list = []
        
        for p_path in p_paths:
            if os.path.exists(p_path):
                with open(p_path, 'rb') as f:
                    d = pickle.load(f)
                    # 如果是包含单个图像数组的新格式
                    if isinstance(d['pixels_low'], np.ndarray):
                        L_list.append(d['pixels_low'])
                        H_list.append(d['pixels_high'])
                    # 如果是包含多个图像的列表的旧格式
                    elif isinstance(d['pixels_low'], list):
                        L_list.extend(d['pixels_low'])
                        H_list.extend(d['pixels_high'])
            else:
                print(f"Warning: Step file not found: {p_path}")
                
        if config['steps'] is not None:
            L_list = [L_list[i] for i in config['steps']]
            H_list = [H_list[i] for i in config['steps']]
            
        T_list = []
        for i, L in enumerate(L_list):
            if config.get('thicknesses'):
                t = config['thicknesses'][i]
            else:
                t = 1.0 # Default fallback
            T_list.append(np.full(L.shape, t, dtype=np.float32))
            
        if len(L_list) > 0:
            L_all = np.concatenate(L_list).astype(np.float32)
            H_all = np.concatenate(H_list).astype(np.float32)
            T_all = np.concatenate(T_list).astype(np.float32)
            
            v_max = 65535 if L_all.dtype == np.uint16 or np.max(L_all) > 255 else 255
            lower_th = utils_II.get_ore_lower_threshold(False, v_max)
            valid = (L_all >= lower_th) & (H_all >= lower_th) & (L_all < v_max) & (H_all < v_max)
            
            step_data[name] = (L_all[valid], H_all[valid], T_all[valid])
            
    return step_data

def load_ore_dataset():
    pkl_path = r'E:\photo_electric_II\data\0325_0519_0520_input_cleaned_dataset_le2.pkl'
    try:
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
    except Exception:
        import numpy.core.numeric
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = numpy.core.numeric
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        with open(pkl_path, 'rb') as f:
            input_all = pickle.load(f)
            
    return input_all

def compute_log_attenuation(L, H, I0=204.0):
    """
    计算对数衰减值。对灰度值做限幅以避免除零或对数域发散。

    参数类型、含义及用法：
    - L (np.ndarray): 低能灰度像素值数组。
    - H (np.ndarray): 高能灰度像素值数组。
    - I0 (float): X光未穿过岩石时的空带入射光强度，此处默认为 204.0。
    """
    u_L = np.log(I0 / np.maximum(L, 1.0))
    u_H = np.log(I0 / np.maximum(H, 1.0))
    return u_L, u_H

def fit_reference_pca(step_data, I0=204.0):
    """
    利用铜铁铝标样阶梯像素构建基准 PCA 物理空间。

    参数类型、含义及用法：
    - step_data (dict): 标样像素字典，格式为 {材质名: (L_array, H_array, T_array)}。
    - I0 (float): 空带强度。
    """
    X_ref_all = []
    
    for mat_name, (L, H, T_v) in step_data.items():
        u_L, u_H = compute_log_attenuation(L, H, I0)
        u_L_norm = u_L / T_v
        u_H_norm = u_H / T_v
        X_ref_all.append(np.column_stack([u_L_norm, u_H_norm]))
        
    X_ref = np.vstack(X_ref_all)
    
    pca = PCA(n_components=2)
    pca.fit(X_ref)
    
    print("\n=== Fitted Reference PCA Components ===")
    print(f"Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    print(f"PC1 (Thickness Axis) Vector: [u_L: {pca.components_[0, 0]:.4f}, u_H: {pca.components_[0, 1]:.4f}]")
    print(f"PC2 (Z-eff/Material Axis) Vector: [u_L: {pca.components_[1, 0]:.4f}, u_H: {pca.components_[1, 1]:.4f}]")
    
    return pca

def extract_ore_features(pixels_df_list, info_df, pca_model, I0=204.0):
    """
    提取每块矿石的原始 XRT 灰度统计特征与投影解耦后的 PCA 统计特征。

    参数类型、含义及用法：
    - pixels_df_list (list): 矿石灰度像素 Series 列表（包含 Channel 0 和 Channel 1）。
    - info_df (pd.DataFrame): 矿石特征数据库 DataFrame。
    - pca_model (sklearn.decomposition.PCA): 已训练的标样 PCA 模型。
    - I0 (float): 空带强度。
    """
    ore_features = []
    
    for row_idx in range(len(info_df)):
        ore_row = info_df.iloc[row_idx]
        
        ore_L = pixels_df_list[0].iloc[row_idx]
        ore_H = pixels_df_list[1].iloc[row_idx]
        
        # 1. 过滤无效和饱和像素
        v_max = 65535 if ore_L.dtype == np.uint16 or np.max(ore_L) > 255 else 255
        lower_th = utils_II.get_ore_lower_threshold(False, v_max)
        valid = (ore_L >= lower_th) & (ore_H >= lower_th) & (ore_L < v_max) & (ore_H < v_max)
        ore_L_v, ore_H_v = ore_L[valid].astype(np.float32), ore_H[valid].astype(np.float32)
        
        if len(ore_L_v) == 0:
            # 异常兜底，全零填充
            features = {
                'global_id': ore_row['global_id'],
                'raw_LE_mean': 0, 'raw_LE_std': 0, 'raw_HE_mean': 0, 'raw_HE_std': 0,
                'PC1_mean': 0, 'PC1_std': 0, 'PC1_skew': 0, 'PC1_kurt': 0,
                'PC2_mean': 0, 'PC2_std': 0, 'PC2_skew': 0, 'PC2_kurt': 0
            }
            ore_features.append(features)
            continue
            
        # 2. 提取原始灰度统计特征 (4维)
        raw_LE_mean = np.mean(ore_L_v)
        raw_LE_std = np.std(ore_L_v)
        raw_HE_mean = np.mean(ore_H_v)
        raw_HE_std = np.std(ore_H_v)
        
        # 3. 转换到对数衰减平面并进行 PCA 旋转
        u_L, u_H = compute_log_attenuation(ore_L_v, ore_H_v, I0)
        X_ore = np.column_stack([u_L, u_H])
        X_pca = pca_model.transform(X_ore) # 投影到 (PC1, PC2) 空间
        
        pc1_vals = X_pca[:, 0]
        pc2_vals = X_pca[:, 1]
        
        # 4. 提取 PC 特征的四阶矩 (8维)
        features = {
            'global_id': ore_row['global_id'],
            'raw_LE_mean': raw_LE_mean,
            'raw_LE_std': raw_LE_std,
            'raw_HE_mean': raw_HE_mean,
            'raw_HE_std': raw_HE_std,
            'PC1_mean': np.mean(pc1_vals),
            'PC1_std': np.std(pc1_vals),
            'PC1_skew': skew(pc1_vals) if len(pc1_vals) > 2 else 0.0,
            'PC1_kurt': kurtosis(pc1_vals) if len(pc1_vals) > 2 else 0.0,
            'PC2_mean': np.mean(pc2_vals),
            'PC2_std': np.std(pc2_vals),
            'PC2_skew': skew(pc2_vals) if len(pc2_vals) > 2 else 0.0,
            'PC2_kurt': kurtosis(pc2_vals) if len(pc2_vals) > 2 else 0.0
        }
        ore_features.append(features)
        
    return pd.DataFrame(ore_features)

def run_cross_validation(X, y, weights, model_name, classifier):
    """
    进行分层 5 折交叉验证，统计数量准确率与重量加权准确率、查全率和排废率。

    参数类型、含义及用法：
    - X (np.ndarray): 输入特征矩阵。
    - y (np.ndarray): 标签向量 (0代表废石，1代表精矿)。
    - weights (np.ndarray): 样本物理重量 (weight_g) 数组，用于计算质量加权准确率。
    - model_name (str): 模型标记名称。
    - classifier (sklearn estimator): 待评估的分类器实例。
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = []
    mass_acc_scores = []
    sens_scores = [] # Sensitivity (Recall on Concentrate)
    spec_scores = [] # Specificity (TNR on Waste)
    
    oof_preds = np.zeros(len(y), dtype=int)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_test = weights[test_idx]
        
        # 建立带标准化 Pipeline
        pipeline = make_pipeline(StandardScaler(), classifier)
        
        # 动态检测并传入样本权重支持自定义阈值分类器优化
        fit_params = {}
        clf_name_lower = classifier.__class__.__name__.lower()
        if 'singlepc2threshold' in clf_name_lower:
            fit_params = {f'{clf_name_lower}__sample_weight': weights[train_idx]}
            
        pipeline.fit(X_train, y_train, **fit_params)
        
        y_pred = pipeline.predict(X_test)
        oof_preds[test_idx] = y_pred
        
        # 1. 数量准确率
        acc = accuracy_score(y_test, y_pred) * 100
        
        # 2. 质量加权准确率 (以物理重量进行加权评分)
        correct_mask = (y_pred == y_test)
        mass_acc = (np.sum(w_test[correct_mask]) / np.sum(w_test)) * 100
        
        # 3. 混淆矩阵成分计算
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        sens = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        
        acc_scores.append(acc)
        mass_acc_scores.append(mass_acc)
        sens_scores.append(sens)
        spec_scores.append(spec)
        
    return {
        'count_acc': (np.mean(acc_scores), np.std(acc_scores)),
        'mass_acc': (np.mean(mass_acc_scores), np.std(mass_acc_scores)),
        'sens': (np.mean(sens_scores), np.std(sens_scores)),
        'spec': (np.mean(spec_scores), np.std(spec_scores)),
        'oof_preds': oof_preds
    }

def plot_step_attenuation_space(step_data, abs_output_dir, I0=204.0):
    """
    绘制阶梯数据在对数衰减量 (u_H vs u_L) 原始空间的分布。
    分为各个子图，避免金属重叠导致的视觉混乱。
    """
    import math
    num_metals = len(step_data)
    cols = 3
    rows = math.ceil(num_metals / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    if num_metals == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    import matplotlib as mpl
    colors = mpl.colormaps['tab10'].colors
    
    z_map = {'Al': 13, 'Ti': 22, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Sn': 50, 'W': 74, 'Pb': 82}
    def get_z(name):
        return z_map.get(name.split('_')[0], 999)
        
    sorted_items = sorted(step_data.items(), key=lambda x: get_z(x[0]))
    
    for i, (mat_name, (L_v, H_v, T_v)) in enumerate(sorted_items):
        ax = axes[i]
        u_L, u_H = compute_log_attenuation(L_v, H_v, I0)
        u_L = u_L / T_v
        u_H = u_H / T_v
        
        max_plot_points = 1000
        if len(u_L) > max_plot_points:
            indices = np.random.choice(len(u_L), max_plot_points, replace=False)
            u_L_plot = u_L[indices]
            u_H_plot = u_H[indices]
        else:
            u_L_plot = u_L
            u_H_plot = u_H
            
        color = colors[i % len(colors)]
        ax.scatter(u_L_plot, u_H_plot, color=color, alpha=0.3, s=2.0)
        
        base_name = mat_name.split('_')[0]
        z_num = z_map.get(base_name, '?')
        ax.set_title(f"Metal: {mat_name} (Z={z_num})", fontsize=12, fontweight='bold')
        ax.set_xlabel("u_L/t = (1/t) * ln(I0/I_L)", fontsize=10)
        ax.set_ylabel("u_H/t = (1/t) * ln(I0/I_H)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    # 隐藏多余的子图
    for j in range(len(step_data), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    fig.suptitle("原始试样在高低能对数衰减空间的分布图 (分金属子图)", fontsize=16, fontweight='bold', y=1.02)
            
    save_path = os.path.join(abs_output_dir, "step_attenuation_space_subplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Step attenuation distribution subplots saved to: {save_path}")

def plot_step_pca_space(step_data, pca_model, abs_output_dir, I0=204.0):
    """
    绘制铜铁铝参考阶梯数据在 PC1-PC2 平面上的投影分布，直观验证其线性拉平效果。
    """
    plt.figure(figsize=(10, 8))
    
    import matplotlib as mpl
    colors = mpl.colormaps['tab10'].colors
    
    z_map = {'Al': 13, 'Ti': 22, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Sn': 50, 'W': 74, 'Pb': 82}
    def get_z(name):
        return z_map.get(name.split('_')[0], 999)
        
    sorted_items = sorted(step_data.items(), key=lambda x: get_z(x[0]))
    
    for i, (mat_name, (L_v, H_v, T_v)) in enumerate(sorted_items):
        u_L, u_H = compute_log_attenuation(L_v, H_v, I0)
        u_L_norm = u_L / T_v
        u_H_norm = u_H / T_v
        X_ref = np.column_stack([u_L_norm, u_H_norm])
        X_pca = pca_model.transform(X_ref)
        
        max_plot_points = 1000
        if len(X_pca) > max_plot_points:
            indices = np.random.choice(len(X_pca), max_plot_points, replace=False)
            X_plot = X_pca[indices]
        else:
            X_plot = X_pca
            
        color = colors[i % len(colors)]
        
        base_name = mat_name.split('_')[0]
        z_num = z_map.get(base_name, '?')
        plt.scatter(X_plot[:, 0], X_plot[:, 1], color=color, alpha=0.3, s=2.0, label=f"{mat_name} (Z={z_num})")
        
    plt.title("DE-XRT 标样阶梯在 PC1-PC2 解耦平面的投影走势图", fontsize=14, fontweight='bold')
    plt.xlabel("PC1 (Thickness / Attenuation Magnitude)", fontsize=12)
    plt.ylabel("PC2 (Composition / Effective Z Deviation)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    leg = plt.legend(fontsize=10, loc='upper left')
    if leg:
        for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
            lh.set_alpha(1.0)
            lh.set_sizes([20])
            
    save_path = os.path.join(abs_output_dir, "step_pca_space.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Step PCA distribution plot saved to: {save_path}")

def plot_ore_attenuation_comparison(step_data, pixels_df_list, info_df, target_ids, group_name, abs_output_dir, I0=204.0):
    """
    在原始 u_L - u_H 对数衰减空间绘制 2x2 网格，展示 4 类代表性矿石的像素投影与标样参考线的分布重叠对比。
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()
    
    import matplotlib as mpl
    colors = mpl.colormaps['tab10'].colors
    
    z_map = {'Al': 13, 'Ti': 22, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Sn': 50, 'W': 74, 'Pb': 82}
    def get_z(name):
        return z_map.get(name.split('_')[0], 999)
        
    for idx_sp, (key, cfg) in enumerate(target_ids.items()):
        g_id = cfg['id']
        title = cfg['title']
        ax = axes_flat[idx_sp]
        
        sorted_items = sorted(step_data.items(), key=lambda x: get_z(x[0]))
        
        # 1. 绘制背景标样线在 原始 空间的投影
        for i, (mat_name, (L_v, H_v, T_v)) in enumerate(sorted_items):
            u_L, u_H = compute_log_attenuation(L_v, H_v, I0)
            u_L_norm = u_L / T_v
            u_H_norm = u_H / T_v
            X_ref = np.column_stack([u_L_norm, u_H_norm])
            
            max_plot_points = 1000
            if len(X_ref) > max_plot_points:
                indices = np.random.choice(len(X_ref), max_plot_points, replace=False)
                X_plot = X_ref[indices]
            else:
                X_plot = X_ref
                
            color = colors[i % len(colors)]
            base_name = mat_name.split('_')[0]
            z_num = z_map.get(base_name, '?')
            ax.scatter(X_plot[:, 0], X_plot[:, 1], color=color, alpha=0.15, s=1.0, label=f"{mat_name} (Z={z_num})")
            
        # 2. 提取并转换目标矿石像素点
        rows = info_df[info_df['global_id'] == g_id]
        if len(rows) == 0:
            continue
        row_idx = rows.index[0]
        ore_row = rows.iloc[0]
        ore_L = pixels_df_list[0].iloc[row_idx]
        ore_H = pixels_df_list[1].iloc[row_idx]
        
        v_max = 65535 if ore_L.dtype == np.uint16 or np.max(ore_L) > 255 else 255
        lower_th = utils_II.get_ore_lower_threshold(False, v_max)
        valid = (ore_L >= lower_th) & (ore_H >= lower_th) & (ore_L < v_max) & (ore_H < v_max)
        ore_L_v, ore_H_v = ore_L[valid].astype(np.float32), ore_H[valid].astype(np.float32)
        
        if len(ore_L_v) > 0:
            mean_thickness = float(ore_row['mean_thickness'])
            u_L, u_H = compute_log_attenuation(ore_L_v, ore_H_v, I0)
            u_L_norm = u_L / mean_thickness
            u_H_norm = u_H / mean_thickness
            X_ore = np.column_stack([u_L_norm, u_H_norm])
            
            max_ore_plot = 1000
            if len(X_ore) > max_ore_plot:
                indices = np.random.choice(len(X_ore), max_ore_plot, replace=False)
                X_plot_ore = X_ore[indices]
            else:
                X_plot_ore = X_ore
                
            ax.scatter(X_plot_ore[:, 0], X_plot_ore[:, 1], color='#e377c2', alpha=0.3, s=1.5, label='Ore Pixels')
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("u_L/t = (1/t) * ln(I0/I_L)", fontsize=10)
        ax.set_ylabel("u_H/t = (1/t) * ln(I0/I_H)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 优化 Legend
        leg = ax.legend(fontsize='small', loc='upper left')
        if leg:
            for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
                lh.set_alpha(1.0)
                if lh.get_label() == 'Ore Pixels':
                    lh.set_sizes([20])
                else:
                    lh.set_sizes([10])
                    
    plt.tight_layout()
    save_path = os.path.join(abs_output_dir, f"ore_attenuation_comparison_{group_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved attenuation comparison figure to: {save_path}")

def plot_ore_pca_comparison(step_data, pca_model, pixels_df_list, info_df, target_ids, group_name, abs_output_dir, I0=204.0):
    """
    在 PC1-PC2 空间绘制 2x2 网格，展示 4 类代表性矿石的像素投影与标样参考线的分布重叠对比。
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()
    
    import matplotlib as mpl
    colors = mpl.colormaps['tab10'].colors
    
    z_map = {'Al': 13, 'Ti': 22, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Sn': 50, 'W': 74, 'Pb': 82}
    def get_z(name):
        return z_map.get(name.split('_')[0], 999)
        
    for idx_sp, (key, cfg) in enumerate(target_ids.items()):
        g_id = cfg['id']
        title = cfg['title']
        ax = axes_flat[idx_sp]
        
        sorted_items = sorted(step_data.items(), key=lambda x: get_z(x[0]))
        
        # 1. 绘制背景标样线在 PC 空间的投影
        for i, (mat_name, (L_v, H_v, T_v)) in enumerate(sorted_items):
            u_L, u_H = compute_log_attenuation(L_v, H_v, I0)
            u_L_norm = u_L / T_v
            u_H_norm = u_H / T_v
            X_ref = np.column_stack([u_L_norm, u_H_norm])
            X_pca = pca_model.transform(X_ref)
            
            max_plot_points = 1000
            if len(X_pca) > max_plot_points:
                indices = np.random.choice(len(X_pca), max_plot_points, replace=False)
                X_plot = X_pca[indices]
            else:
                X_plot = X_pca
                
            color = colors[i % len(colors)]
            base_name = mat_name.split('_')[0]
            z_num = z_map.get(base_name, '?')
            ax.scatter(X_plot[:, 0], X_plot[:, 1], color=color, alpha=0.15, s=1.0, label=f"{mat_name} (Z={z_num})")
            
        # 2. 提取并转换目标矿石像素点到 PC 空间
        rows = info_df[info_df['global_id'] == g_id]
        if len(rows) == 0:
            continue
        row_idx = rows.index[0]
        ore_row = rows.iloc[0]
        
        ore_L = pixels_df_list[0].iloc[row_idx]
        ore_H = pixels_df_list[1].iloc[row_idx]
        
        v_max = 65535 if ore_L.dtype == np.uint16 or np.max(ore_L) > 255 else 255
        lower_th = utils_II.get_ore_lower_threshold(False, v_max)
        valid = (ore_L >= lower_th) & (ore_H >= lower_th) & (ore_L < v_max) & (ore_H < v_max)
        ore_L_v, ore_H_v = ore_L[valid].astype(np.float32), ore_H[valid].astype(np.float32)
        
        if len(ore_L_v) > 0:
            mean_thickness = float(ore_row['mean_thickness'])
            u_L, u_H = compute_log_attenuation(ore_L_v, ore_H_v, I0)
            u_L_norm = u_L / mean_thickness
            u_H_norm = u_H / mean_thickness
            X_ore = np.column_stack([u_L_norm, u_H_norm])
            X_ore_pca = pca_model.transform(X_ore)
            
            max_ore_plot = 1000
            if len(X_ore_pca) > max_ore_plot:
                indices = np.random.choice(len(X_ore_pca), max_ore_plot, replace=False)
                X_plot_ore = X_ore_pca[indices]
            else:
                X_plot_ore = X_ore_pca
                
            ax.scatter(X_plot_ore[:, 0], X_plot_ore[:, 1], color='#e377c2', alpha=0.3, s=1.5, label='Ore Pixels')
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("PC1 (Thickness / Mass)", fontsize=10)
        ax.set_ylabel("PC2 (Composition / Z)", fontsize=10)
        ax.set_xlim(-4.5, 4.5) # 对数衰减映射的最大物理区间
        ax.set_ylim(-1.5, 1.5) # 偏差分量范围
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 优化 Legend
        leg = ax.legend(fontsize='small', loc='upper left')
        if leg:
            for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
                lh.set_alpha(1.0)
                if lh.get_label() == 'Ore Pixels':
                    lh.set_sizes([20])
                else:
                    lh.set_sizes([10])
                    
        # 信息文本框
        text_info = (
            f"Ore ID: #{ore_row['global_id']} (Sample {ore_row['sample_id']})\n"
            f"Cu Grade: {ore_row['Cu']:.3f} %\n"
            f"Fe Grade: {ore_row['Fe']:.2f} %\n"
            f"S Grade: {ore_row['S']:.2f} %\n"
            f"Mean Thickness: {ore_row['mean_thickness']:.2f} mm"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='#cccccc')
        ax.text(0.95, 0.05, text_info, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, fontfamily='monospace')
                
    plt.suptitle(f"DE-XRT PC1-PC2 投影平面矿石与标样对比图 ({group_name})", fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(abs_output_dir, f"ore_pca_comparison_{group_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure to: {save_path}")

def run_classification_pipeline(step_config, output_dir_name="fit_PCA", I0=204.0, pc2_greater_than=False):
    # 设中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    base_dir = os.path.dirname(__file__)
    # 创建专属输出文件夹，并统一放在 results/fit_pca 下
    abs_output_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'fit_pca', output_dir_name))
    os.makedirs(abs_output_dir, exist_ok=True)
    
    print(f"\n[{output_dir_name}] Loading step wedge data...")
    step_data = load_step_data(step_config)
    
    # 动态检测是 16位 还是 8位 图像，设置对应的 I0 背景本底值
    max_val = 0.0
    for L_v, H_v, T_v in step_data.values():
        max_val = max(max_val, np.max(L_v), np.max(H_v))
        
    if max_val > 500:
        I0 = 65536 * 0.8
        print(f"[{output_dir_name}] Detected 16-bit data (max_val={max_val:.1f}). Setting I0 = {I0}.")
    else:
        I0 = 204.0
        print(f"[{output_dir_name}] Detected 8-bit data (max_val={max_val:.1f}). Setting I0 = {I0}.")
        
    print(f"[{output_dir_name}] Fitting reference PCA on steps...")
    pca_model = fit_reference_pca(step_data, I0)
    
    # 2. 绘制标样在 PCA 平面的投影以及原始衰减空间的分布
    plot_step_attenuation_space(step_data, abs_output_dir, I0)
    plot_step_pca_space(step_data, pca_model, abs_output_dir, I0)
    
    # 3. 加载矿石清洗后数据集并提取特征
    print(f"\n[{output_dir_name}] Loading ore cleaned dataset...")
    ore_dataset = load_ore_dataset()
    pixels_df_list = ore_dataset[0]
    info_df = ore_dataset[1].copy()
    
    print(f"[{output_dir_name}] Extracting features from ore pixels...")
    features_df = extract_ore_features(pixels_df_list, info_df, pca_model, I0)
    
    # 合并特征到主 info DataFrame
    merged_df = pd.merge(info_df, features_df, on='global_id')
    
    # 4. 绘制 Group1 和 Group2 在 PCA 平面下的投影对比图
    ore_groups = {
        'group1': {
            'subplot1': {'id': 293, 'title': '1. 综合品位最低且挺薄的矿石'},
            'subplot2': {'id': 107, 'title': '2. 品位低且厚的矿石'},
            'subplot3': {'id': 211, 'title': '3. 品位高且薄的矿石'},
            'subplot4': {'id': 23,  'title': '4. 品位高且厚的矿石'}
        },
        'group2': {
            'subplot1': {'id': 271, 'title': '1. 综合品位最低且挺薄的矿石'},
            'subplot2': {'id': 112, 'title': '2. 品位低且厚的矿石'},
            'subplot3': {'id': 164, 'title': '3. 品位高且薄的矿石'},
            'subplot4': {'id': 10,  'title': '4. 品位高且厚的矿石'}
        }
    }
    for group_name, target_ids in ore_groups.items():
        # 绘制原始衰减量对比网格图
        plot_ore_attenuation_comparison(step_data, pixels_df_list, info_df, target_ids, group_name, abs_output_dir, I0)
        # 绘制 PCA 投影网格图
        plot_ore_pca_comparison(step_data, pca_model, pixels_df_list, info_df, target_ids, group_name, abs_output_dir, I0)
        
    # 5. 构建分类数据集
    # 设定分类阈值 (Cu >= 0.10% 为精矿 concentrate 标签为 1，否则为废石 0)
    y = (merged_df['Cu'] >= 0.10).astype(int).values
    weights = merged_df['weight_g'].values
    
    # 特征子集 1: Baseline (Raw XRT, 4维)
    X_baseline = merged_df[['raw_LE_mean', 'raw_LE_std', 'raw_HE_mean', 'raw_HE_std']].values
    
    # 特征子集 2: Fusion (PCA + 3D Geometry, 10维)
    # PCA(8维) + 3D几何(mean_thickness, weight_g)
    X_fusion = merged_df[[
        'PC1_mean', 'PC1_std', 'PC1_skew', 'PC1_kurt',
        'PC2_mean', 'PC2_std', 'PC2_skew', 'PC2_kurt',
        'mean_thickness', 'weight_g'
    ]].values
    
    # 定义待评估分类器
    classifiers = {
        'Random Forest (RF)': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'Gradient Boosting (GB)': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42),
        'Support Vector Machine (SVC)': SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42)
    }
    
    print("\n" + "="*70)
    print(f"Dataset Size: {len(merged_df)} Ores (Waste={np.sum(y==0)}, Concentrate={np.sum(y==1)})")
    print("="*70)
    
    # 执行交叉验证评估
    all_results = []
    
    # 5.1 评估单变量 PC2 均值阈值分类器 (包含两种优化目标)
    X_pc2_only = merged_df[['PC2_mean']].values
    
    print("\nTraining PC2 Threshold Classifier (Optimize Count Acc)...")
    clf_pc2_count = SinglePC2ThresholdClassifier(optimize_metric='count_acc', greater_than=pc2_greater_than)
    res_pc2_count = run_cross_validation(X_pc2_only, y, weights, 'PC2 Threshold (Count Acc)', clf_pc2_count)
    all_results.append({
        'Model': 'PC2 Threshold (Count Acc)',
        'Feat': 'PC2_mean only',
        'Acc': f"{res_pc2_count['count_acc'][0]:.2f}% ± {res_pc2_count['count_acc'][1]:.2f}%",
        'Mass_Acc': f"{res_pc2_count['mass_acc'][0]:.2f}% ± {res_pc2_count['mass_acc'][1]:.2f}%",
        'Recall(Sens)': f"{res_pc2_count['sens'][0]:.2f}% ± {res_pc2_count['sens'][1]:.2f}%",
        'Reject(Spec)': f"{res_pc2_count['spec'][0]:.2f}% ± {res_pc2_count['spec'][1]:.2f}%"
    })
    
    print("Training PC2 Threshold Classifier (Optimize Mass Acc)...")
    clf_pc2_mass = SinglePC2ThresholdClassifier(optimize_metric='mass_acc', greater_than=pc2_greater_than)
    res_pc2_mass = run_cross_validation(X_pc2_only, y, weights, 'PC2 Threshold (Mass Acc)', clf_pc2_mass)
    all_results.append({
        'Model': 'PC2 Threshold (Mass Acc)',
        'Feat': 'PC2_mean only',
        'Acc': f"{res_pc2_mass['count_acc'][0]:.2f}% ± {res_pc2_mass['count_acc'][1]:.2f}%",
        'Mass_Acc': f"{res_pc2_mass['mass_acc'][0]:.2f}% ± {res_pc2_mass['mass_acc'][1]:.2f}%",
        'Recall(Sens)': f"{res_pc2_mass['sens'][0]:.2f}% ± {res_pc2_mass['sens'][1]:.2f}%",
        'Reject(Spec)': f"{res_pc2_mass['spec'][0]:.2f}% ± {res_pc2_mass['spec'][1]:.2f}%"
    })
    
    print("Training PC2 Threshold Classifier (Optimize Balanced Acc)...")
    clf_pc2_bal = SinglePC2ThresholdClassifier(optimize_metric='balanced_acc', greater_than=pc2_greater_than)
    res_pc2_bal = run_cross_validation(X_pc2_only, y, weights, 'PC2 Threshold (Balanced Acc)', clf_pc2_bal)
    all_results.append({
        'Model': 'PC2 Threshold (Balanced Acc)',
        'Feat': 'PC2_mean only',
        'Acc': f"{res_pc2_bal['count_acc'][0]:.2f}% ± {res_pc2_bal['count_acc'][1]:.2f}%",
        'Mass_Acc': f"{res_pc2_bal['mass_acc'][0]:.2f}% ± {res_pc2_bal['mass_acc'][1]:.2f}%",
        'Recall(Sens)': f"{res_pc2_bal['sens'][0]:.2f}% ± {res_pc2_bal['sens'][1]:.2f}%",
        'Reject(Spec)': f"{res_pc2_bal['spec'][0]:.2f}% ± {res_pc2_bal['spec'][1]:.2f}%"
    })
    
    # 5.2 评估多变量机器学习分类器
    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name}...")
        
        # 评估 Baseline (Raw XRT)
        res_base = run_cross_validation(X_baseline, y, weights, 'Raw XRT', clf)
        
        # 评估 Fusion (PCA + Geometry)
        res_fuse = run_cross_validation(X_fusion, y, weights, 'Fusion', clf)
        
        all_results.append({
            'Model': clf_name,
            'Feat': 'Raw XRT',
            'Acc': f"{res_base['count_acc'][0]:.2f}% ± {res_base['count_acc'][1]:.2f}%",
            'Mass_Acc': f"{res_base['mass_acc'][0]:.2f}% ± {res_base['mass_acc'][1]:.2f}%",
            'Recall(Sens)': f"{res_base['sens'][0]:.2f}% ± {res_base['sens'][1]:.2f}%",
            'Reject(Spec)': f"{res_base['spec'][0]:.2f}% ± {res_base['spec'][1]:.2f}%"
        })
        all_results.append({
            'Model': clf_name,
            'Feat': 'PCA Fusion',
            'Acc': f"{res_fuse['count_acc'][0]:.2f}% ± {res_fuse['count_acc'][1]:.2f}%",
            'Mass_Acc': f"{res_fuse['mass_acc'][0]:.2f}% ± {res_fuse['mass_acc'][1]:.2f}%",
            'Recall(Sens)': f"{res_fuse['sens'][0]:.2f}% ± {res_fuse['sens'][1]:.2f}%",
            'Reject(Spec)': f"{res_fuse['spec'][0]:.2f}% ± {res_fuse['spec'][1]:.2f}%"
        })
        
    # 格式化输出表格
    results_df = pd.DataFrame(all_results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n" + "="*100)
    print("                      SUMMARY OF CLASSIFICATION RESULTS (5-FOLD CV)")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)
    
    # 保存结果到 csv 方便后续查阅
    results_csv = os.path.join(abs_output_dir, "classification_summary.csv")
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"\n[{output_dir_name}] Classification results summary saved to: {results_csv}")
    print(f"[{output_dir_name}] Done pipeline execution.")
    
    return pca_model, merged_df, classifiers, weights
