import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 假设三种材料的等效原子序数 (Z) 
# Cu = 29, Fe = 26, Al = 13
material_Z = {0: 29, 1: 26, 2: 13}
material_names = {0: 'Cu', 1: 'Fe', 2: 'Al'}

voltages = ['140kV', '160kV', '180kV']
output_dir = 'results/thickness_decoupling'
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
    print(f"--- Processing Voltage: {voltage} ---")
    
    X_ratio_list = []
    X_poly_list = []
    y_list = []
    
    # 1. 加载数据并组合
    for idx, Z in material_Z.items():
        file_path = f'results/20260331/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        L_data = data['pixels_low']
        H_data = data['pixels_high']
        
        # 兼容阶梯样本的 list 格式以及纯方块的 ndarray 格式
        if isinstance(L_data, list):
            L = np.concatenate(L_data).astype(np.float32)
            H = np.concatenate(H_data).astype(np.float32)
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
    # Model 1: 基于 H/L 比例的二次模型
    model_ratio = Ridge(alpha=1.0)
    model_ratio.fit(X_ratio, y)
    
    # Model 2: 基于 H和L 的常规二阶多项式模型 (a*H^2 + b*L^2 + c*H*L + d*H + e*L + f)
    model_poly = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0))
    model_poly.fit(X_poly, y)
    
    # 3. 评估并在散点图中展示
    preds_ratio = model_ratio.predict(X_ratio)
    preds_poly = model_poly.predict(X_poly)
    
    # 提取参数打印
    print(f"Ratio Model (H/L): Z = {model_ratio.coef_[1]:.4f}*(H/L)^2 + {model_ratio.coef_[0]:.4f}*(H/L) + {model_ratio.intercept_:.4f}")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制 Model 1 结果 (X轴可以是 H/L，Y轴预测Z)
    axes[0].scatter(X_ratio[:, 0], preds_ratio, c=y, cmap='viridis', alpha=0.1, s=1)
    axes[0].set_title(f"Model 1 (H/L Poly): Pred Z vs H/L ({voltage})")
    axes[0].set_xlabel("H / L Ratio")
    axes[0].set_ylabel("Predicted Z (Should be flat per material)")
    axes[0].axhline(29, color='r', linestyle='--', label='Cu (29)')
    axes[0].axhline(26, color='g', linestyle='--', label='Fe (26)')
    axes[0].axhline(13, color='b', linestyle='--', label='Al (13)')
    axes[0].legend()
    
    # 绘制 Model 2 结果 (因为有H和L两个输入变量，我们直接画真实Z和预测Z的分布，或者画密度图)
    # 使用直方图展示预测Z的集中度 (解耦越好，方差越小)
    import seaborn as sns
    for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
        mask = (y == Z_val)
        sns.kdeplot(preds_poly[mask], ax=axes[1], label=f'{name} (True={Z_val})', fill=True)
        
    axes[1].set_title(f"Model 2 (H & L 2nd-order Poly) Z Distribution ({voltage})")
    axes[1].set_xlabel("Predicted Z (Thickness Decoupled)")
    axes[1].set_ylabel("Density")
    axes[1].axvline(29, color='r', linestyle='--')
    axes[1].axvline(26, color='g', linestyle='--')
    axes[1].axvline(13, color='b', linestyle='--')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{voltage}_Z_decoupling_results.png")
    plt.close()
    
    # 验证解耦程度（方差越小越好）
    print("Variance of Predicted Z (Smaller means better thickness decoupling):")
    for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
        mask = (y == Z_val)
        std_r = np.std(preds_ratio[mask])
        std_p = np.std(preds_poly[mask])
        print(f"  {name}: Model1(H/L) std={std_r:.3f}, Model2(H,L) std={std_p:.3f}")
    print("\n")

print(f"Done. Please check the charts in {output_dir}")
