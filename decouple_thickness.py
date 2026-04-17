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
# 方案 1: z(H, L) = a(H/L)^2 + b(H/L) + c# 用于汇总各阶段结果的数据列表
results_data = []

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
        step_id_list = []
        
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
            
            m_X_ratio = []
            m_X_poly = []
            m_y = []
            m_step = []

            if isinstance(L_data, list):
                for s_idx in range(n_target_steps):
                    l_s = L_data[s_idx].astype(np.float32)
                    h_s = H_data[s_idx].astype(np.float32)
                    # 过滤无效背景像素 (阈值降低至 1 以观察极厚层)
                    valid = (l_s > 1) & (h_s > 1) & (l_s < 250) & (h_s < 250)
                    l_s, h_s = l_s[valid], h_s[valid]
                    if len(l_s) == 0: continue

                    m_X_ratio.append(extract_feature_HL_ratio(l_s, h_s))
                    m_X_poly.append(extract_feature_poly(l_s, h_s))
                    m_y.append(np.full(len(l_s), Z))
                    m_step.append(np.full(len(l_s), s_idx))
            else:
                L = L_data.astype(np.float32)
                H = H_data.astype(np.float32)
                valid = (L > 1) & (H > 1) & (L < 250) & (H < 250)
                L, H = L[valid], H[valid]
                m_X_ratio.append(extract_feature_HL_ratio(L, H))
                m_X_poly.append(extract_feature_poly(L, H))
                m_y.append(np.full(len(L), Z))
                m_step.append(np.full(len(L), 0))
            
            # 合并当前材质的数据以便统一采样
            X_r_m = np.vstack(m_X_ratio)
            X_p_m = np.vstack(m_X_poly)
            y_m = np.concatenate(m_y)
            step_m = np.concatenate(m_step)
            
            # 材质内随机采样，防止数据量过大
            if len(y_m) > 50000:
                indices = np.random.choice(len(y_m), 50000, replace=False)
                X_r_m, X_p_m, y_m, step_m = X_r_m[indices], X_p_m[indices], y_m[indices], step_m[indices]
                
            X_ratio_list.append(X_r_m)
            X_poly_list.append(X_p_m)
            y_list.append(y_m)
            step_id_list.append(step_m)
        
        if not y_list:
            continue
            
        X_ratio = np.vstack(X_ratio_list)
        X_poly = np.vstack(X_poly_list)
        y = np.concatenate(y_list)
        step_ids = np.concatenate(step_id_list)
        
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
        scaler_r = model_ratio.named_steps['standardscaler']
        ridge_r = model_ratio.named_steps['ridge']
        coef_orig = ridge_r.coef_ / scaler_r.scale_
        intercept_orig = ridge_r.intercept_ - np.sum(ridge_r.coef_ * scaler_r.mean_ / scaler_r.scale_)
        formula_ratio = f"Z = {coef_orig[1]:.4f}*(H/L)^2 + {coef_orig[0]:.4f}*(H/L) + {intercept_orig:.4f}"
        
        # 可视化结果 (4x3 布局)
        fig, axes = plt.subplots(4, 3, figsize=(20, 20))
        material_colors = {29: 'red', 26: 'green', 13: 'blue'}
        material_cmaps = {29: 'Reds', 26: 'Greens', 13: 'Blues'}
        import seaborn as sns

        # --- Row 0: Global Performance Overview ---
        # Ax0,0: Model 1 Scatter
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            axes[0, 0].scatter(X_ratio[mask, 0], preds_ratio[mask], color=material_colors[Z_val], alpha=0.1, s=1, label=f'{name} (True Z={Z_val})')
            axes[0, 0].axhline(Z_val, color=material_colors[Z_val], linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f"M1 Global Scatter\n{formula_ratio}")
        axes[0, 0].set_xlabel("H/L Ratio")
        axes[0, 0].set_ylabel("Predicted Z")
        axes[0, 0].legend(loc='upper right', fontsize='x-small')

        # Ax0,1 & Ax0,2: Global KDE
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask_m = (y == Z_val)
            for ax_col, preds, title in zip([1, 2], [preds_ratio, preds_poly], ["M1 Global KDE", "M2 Global KDE"]):
                sns.kdeplot(preds[mask_m], ax=axes[0, ax_col], label=f'{name} (Z={Z_val})', color=material_colors[Z_val], linewidth=1.5)
                axes[0, ax_col].axvline(Z_val, color=material_colors[Z_val], linestyle=':', alpha=0.8)
                axes[0, ax_col].set_title(title)
                axes[0, ax_col].legend(loc='upper right', fontsize='x-small')

        # --- Row 1 & 2: Material-Specific Thickness Breakdown (KDE Distribution) ---
        for row_idx, preds, model_name in zip([1, 2], [preds_ratio, preds_poly], ["Model 1", "Model 2"]):
            for col_idx, (Z_val, name) in enumerate(zip([13, 26, 29], ['Al', 'Fe', 'Cu'])):
                ax = axes[row_idx, col_idx]
                mask_m = (y == Z_val)
                cmap_name = material_cmaps[Z_val]
                sns.kdeplot(preds[mask_m], ax=ax, label='Total', color='black', alpha=0.3, linestyle='--', linewidth=1.2, zorder=10)
                
                cm = plt.get_cmap(cmap_name)
                m_steps = np.unique(step_ids[mask_m])
                max_step_idx = m_steps.max() if len(m_steps) > 0 else 1
                for s_idx in m_steps:
                    mask_s = mask_m & (step_ids == s_idx)
                    if np.count_nonzero(mask_s) < 100: continue
                    color_idx = 0.3 + 0.6 * (s_idx / max_step_idx)
                    sns.kdeplot(preds[mask_s], ax=ax, label=f'Step {s_idx}', color=cm(color_idx), linewidth=1.0, alpha=0.7)
                
                ax.set_title(f"{model_name} - {name} Decay")
                ax.axvline(Z_val, color=material_colors[Z_val], linestyle=':', alpha=0.8)
                ax.legend(loc='upper right', fontsize='xx-small', ncol=2)

        # --- Row 3: Systematic Bias Analysis (Mean Z vs Step Index) ---
        for col_idx, (Z_val, name) in enumerate(zip([13, 26, 29], ['Al', 'Fe', 'Cu'])):
            ax = axes[3, col_idx]
            mask_m = (y == Z_val)
            m_steps = np.unique(step_ids[mask_m])
            
            plot_steps = []
            means_m1 = []
            means_m2 = []
            for s_idx in m_steps:
                mask_s = mask_m & (step_ids == s_idx)
                if np.count_nonzero(mask_s) < 100: continue
                plot_steps.append(s_idx)
                means_m1.append(np.mean(preds_ratio[mask_s]))
                means_m2.append(np.mean(preds_poly[mask_s]))
            
            ax.plot(plot_steps, means_m1, 'o-', color='tab:red', label='Model 1 Mean', linewidth=2)
            ax.plot(plot_steps, means_m2, 's--', color='tab:blue', label='Model 2 Mean', linewidth=2)
            ax.axhline(Z_val, color='black', linestyle=':', alpha=0.8, label=f'True Z={Z_val}')
            
            ax.set_title(f"{name} Mean Predicted Z vs Step")
            ax.set_xlabel("Step Index (0=Thinnest)")
            ax.set_ylabel("Mean Z")
            ax.set_xticks(m_steps)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize='x-small')

        plt.tight_layout()
        case_name = f"Al{al_steps}_CuFe{cufe_steps}"
        plt.savefig(f"{output_dir}/{voltage}_{case_name}_Z_decoupling.png")
        plt.close()
        
        # 验证解耦程度
        print(f"Variance of Predicted Z ({case_name}):")
        v_num = int(voltage.replace('kV', ''))
        for Z_val, name in zip([29, 26, 13], ['Cu', 'Fe', 'Al']):
            mask = (y == Z_val)
            std_r = np.std(preds_ratio[mask])
            std_p = np.std(preds_poly[mask])
            print(f"  {name}: M1 std={std_r:.3f}, M2 std={std_p:.3f}")
            
            # 记录数据用于汇总绘图
            results_data.append({'Voltage': v_num, 'Scenario': case_name, 'Material': name, 'Model': 'Model 1', 'Std': std_r})
            results_data.append({'Voltage': v_num, 'Scenario': case_name, 'Material': name, 'Model': 'Model 2', 'Std': std_p})
        print("\n")

# --- 生成汇总对比图 ---
if results_data:
    import pandas as pd
    df_acc = pd.DataFrame(results_data)
    
    plt.figure(figsize=(18, 6))
    materials = ['Al', 'Fe', 'Cu']
    for i, mat in enumerate(materials):
        plt.subplot(1, 3, i+1)
        sub_df = df_acc[df_acc['Material'] == mat]
        
        # 使用 seaborn 绘制折线图，区分模型和场景
        sns.lineplot(data=sub_df, x='Voltage', y='Std', hue='Model', style='Scenario', 
                     markers=True, markersize=8, linewidth=2, palette="Set1")
        
        plt.title(f"{mat} Decoupling Accuracy vs Voltage")
        plt.xlabel("Voltage (kV)")
        plt.ylabel("Prediction Std (Lower is Better)")
        plt.xticks([140, 160, 180])
        plt.grid(True, linestyle='--', alpha=0.6)
        if i == 2: # 最后一个子图显示图例
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            plt.legend().remove()

    plt.tight_layout()
    summary_path = f"{output_dir}/Z_accuracy_summary_comparison.png"
    plt.savefig(summary_path)
    print(f"Accuracy summary plot saved to: {summary_path}")

print(f"Done. Please check the charts in {output_dir}")
