import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# 假设三种材料的等效原子序数 (Z) 
# Cu = 29, Fe = 26, Al = 13
material_Z = {0: 29, 1: 26, 2: 13}
material_names = {0: 'Cu', 1: 'Fe', 2: 'Al'}

# 定义厚度子集场景: (Al_steps, CuFe_steps)
analysis_cases = [
    (6, 4),  # 场景1
    (8, 6),  # 场景2
    (10, 8)  # 场景3
]

voltages = ['140kV', '160kV', '180kV']
input_dir = 'results/20260331/'
output_dir = 'results/thickness_decoupling/z_decouple'
os.makedirs(output_dir, exist_ok=True)

# 定义要测试的模型形式
# 方案 1: z(H, L) = a(H/L)^2 + b(H/L) + c
def extract_feature_HL_ratio(L, H):
    ratio = H / L
    return np.column_stack([ratio, ratio**2])

# 方案 2: z(H, L) 是对 H 和 L 的 2 阶多项式 (包含 H, L, H^2, L^2, H*L)
def extract_feature_poly(L, H):
    return np.column_stack([L, H])

for voltage in voltages:
    for al_steps, cufe_steps in analysis_cases:
        print(f"--- Processing Voltage: {voltage} | Al steps: {al_steps}, Cu/Fe steps: {cufe_steps} ---")
        
        X_ratio_list = []
        X_poly_list = []
        y_list = []
        
        # 1. 加载数据并组合
        for idx, Z in material_Z.items():
            file_path = f'{input_dir}/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            L_data = data['pixels_low']
            H_data = data['pixels_high']
            
            # 根据场景筛选厚度阶梯 (如果是阶梯样本的 list 格式)
            n_target_steps = al_steps if Z == 13 else cufe_steps
            
            if isinstance(L_data, list):
                # 只取前 n 个阶梯（通常是穿透更好的较薄阶梯）
                L = np.concatenate(L_data[:n_target_steps]).astype(np.float32)
                H = np.concatenate(H_data[:n_target_steps]).astype(np.float32)
            else:
                L = L_data.astype(np.float32)
                H = H_data.astype(np.float32)
            
            # 过滤无效背景像素
            valid = (L > 10) & (H > 10) & (L < 250) & (H < 250)
            L = L[valid]
            H = H[valid]
            
            # 随机采样，防止数据量过大内存不足
            if len(L) > 50000:
                indices = np.random.choice(len(L), 50000, replace=False)
                L = L[indices]
                H = H[indices]
                
            # 提取特征
            ratio_feat = extract_feature_HL_ratio(L, H)
            poly_feat = extract_feature_poly(L, H)
            
            X_ratio_list.append(ratio_feat)
            X_poly_list.append(poly_feat)
            
            # 目标值 Z，因为包含多个厚度，但Z是常数，模型会自动解耦厚度
            y_list.append(np.full(len(L), Z))
        
        if not y_list:
            continue
            
        X_ratio = np.vstack(X_ratio_list)
        X_poly = np.vstack(X_poly_list)
        y = np.concatenate(y_list)
        
        # 2. 模型训练
        # Model 1: 基于 H/L 比例的二次模型 + StandardScaler
        model_ratio = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model_ratio.fit(X_ratio, y)
        
        # Model 2: 基于 H和L 的常规二阶多项式模型 + StandardScaler
        model_poly = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
        model_poly.fit(X_poly, y)
        
        # 3. 评估
        preds_ratio = model_ratio.predict(X_ratio)
        preds_poly = model_poly.predict(X_poly)
        
        # 提取并还原原始公式系数 (Unscaling)
        # 只有还原到输入空间的公式才有物理意义
        scaler_r = model_ratio.named_steps['standardscaler']
        ridge_r = model_ratio.named_steps['ridge']
        
        coef_orig = ridge_r.coef_ / scaler_r.scale_
        intercept_orig = ridge_r.intercept_ - np.sum(ridge_r.coef_ * scaler_r.mean_ / scaler_r.scale_)
        
        formula_ratio = f"Z = {coef_orig[1]:.4f}*(H/L)^2 + {coef_orig[0]:.4f}*(H/L) + {intercept_orig:.4f}"
        
        # 可视化结果 (精简为 1x3 布局)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        material_colors = {29: 'red', 26: 'green', 13: 'blue'}
        import seaborn as sns

        # Ax1: Model 1 散点图 (Z vs H/L Ratio)
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            axes[0].scatter(X_ratio[mask, 0], preds_ratio[mask], color=material_colors[Z_val], alpha=0.1, s=1, label=f'{name} (True Z={Z_val})')
            axes[0].axhline(Z_val, color=material_colors[Z_val], linestyle='--', alpha=0.5)
        
        axes[0].set_title(f"Model 1 Scatter\n{formula_ratio}")
        axes[0].set_xlabel("H / L Ratio")
        axes[0].set_ylabel("Predicted Z")
        axes[0].legend()
        
        # Ax2: Model 1 分布图 (Z Distribution)
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            sns.kdeplot(preds_ratio[mask], ax=axes[1], label=f'{name} (True={Z_val})', fill=True, color=material_colors[Z_val])
        axes[1].set_title(f"Model 1 Z Distribution")
        axes[1].set_xlabel("Predicted Z")
        axes[1].axvline(29, color='r', linestyle='--')
        axes[1].axvline(26, color='g', linestyle='--')
        axes[1].axvline(13, color='b', linestyle='--')
        axes[1].legend()

        # Ax3: Model 2 分布图 (Z Distribution)
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            sns.kdeplot(preds_poly[mask], ax=axes[2], label=f'{name} (True={Z_val})', fill=True, color=material_colors[Z_val])
            
        axes[2].set_title(f"Model 2 (L, H Poly) Z Distribution\n2nd-order f(L, H)")
        axes[2].set_xlabel("Predicted Z")
        axes[2].axvline(29, color='r', linestyle='--')
        axes[2].axvline(26, color='g', linestyle='--')
        axes[2].axvline(13, color='b', linestyle='--')
        axes[2].legend()
        
        plt.tight_layout()
        case_name = f"Al{al_steps}_CuFe{cufe_steps}"
        plt.savefig(f"{output_dir}/{voltage}_{case_name}_Z_decoupling.png")
        plt.close()
        
        # 验证解耦程度
        print(f"Variance of Predicted Z ({case_name}):")
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            std_r = np.std(preds_ratio[mask])
            std_p = np.std(preds_poly[mask])
            print(f"  {name}: M1 std={std_r:.3f}, M2 std={std_p:.3f}")
        print("\n")

print(f"Done. Please check the charts in {output_dir}")
