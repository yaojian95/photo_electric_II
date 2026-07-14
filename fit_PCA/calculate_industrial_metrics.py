import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Ensure paths are correct
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))

import fit_pca_classifier
from fit_pca_classifier import SinglePC2ThresholdClassifier

def evaluate_industrial_metrics(merged_df, output_dir_name="comprehensive_run", pc2_greater_than=False):
    abs_output_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'fit_pca', output_dir_name))
    os.makedirs(abs_output_dir, exist_ok=True)
    
    print(f"\n[{output_dir_name}] Evaluating industrial metrics...")
    
    # 分类标签与样本重量
    y = (merged_df['Cu'] >= 0.10).astype(int).values
    weights = merged_df['weight_g'].values
    
    # 特征子集
    X_pc2_only = merged_df[['PC2_mean']].values
    X_fusion = merged_df[[
        'PC1_mean', 'PC1_std', 'PC1_skew', 'PC1_kurt',
        'PC2_mean', 'PC2_std', 'PC2_skew', 'PC2_kurt',
        'mean_thickness', 'weight_g'
    ]].values
    
    # 待评估模型配置
    models_config = {
        'PC2 Threshold (Count Acc)': {
            'clf': SinglePC2ThresholdClassifier(optimize_metric='count_acc', greater_than=pc2_greater_than),
            'X': X_pc2_only
        },
        'PC2 Threshold (Mass Acc)': {
            'clf': SinglePC2ThresholdClassifier(optimize_metric='mass_acc', greater_than=pc2_greater_than),
            'X': X_pc2_only
        },
        'PC2 Threshold (Balanced Acc)': {
            'clf': SinglePC2ThresholdClassifier(optimize_metric='balanced_acc', greater_than=pc2_greater_than),
            'X': X_pc2_only
        },
        'Random Forest (RF) - PCA Fusion': {
            'clf': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'X': X_fusion
        },
        'Gradient Boosting (GB) - PCA Fusion': {
            'clf': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42),
            'X': X_fusion
        },
        'SVC - PCA Fusion': {
            'clf': SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42),
            'X': X_fusion
        }
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 计算原矿总指标
    W_total = np.sum(weights)
    Cu_total_mass = np.sum(weights * merged_df['Cu'] / 100.0)
    Fe_total_mass = np.sum(weights * merged_df['Fe'] / 100.0)
    S_total_mass = np.sum(weights * merged_df['S'] / 100.0)
    
    Cu_feed_grade = Cu_total_mass / W_total * 100.0
    Fe_feed_grade = Fe_total_mass / W_total * 100.0
    S_feed_grade = S_total_mass / W_total * 100.0
    
    print(f"\n=======================================================")
    print(f"原矿(Feed)总重: {W_total:.2f} g")
    print(f"原矿平均品位: Cu={Cu_feed_grade:.3f} %, Fe={Fe_feed_grade:.2f} %, S={S_feed_grade:.2f} %")
    print(f"=======================================================\n")
    
    industrial_results = []
    
    for model_name, config in models_config.items():
        print(f"Evaluating {model_name}...")
        clf = config['clf']
        X = config['X']
        
        # Out-of-fold predictions
        y_pred_oof = np.zeros(len(merged_df), dtype=int)
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            pipeline = make_pipeline(StandardScaler(), clf)
            
            fit_params = {}
            clf_name_lower = clf.__class__.__name__.lower()
            if 'singlepc2threshold' in clf_name_lower:
                fit_params = {f'{clf_name_lower}__sample_weight': weights[train_idx]}
                
            pipeline.fit(X_train, y_train, **fit_params)
            y_pred_oof[test_idx] = pipeline.predict(X_test)
            
        # 计算工业分选指标
        pred_conc = (y_pred_oof == 1)
        pred_waste = (y_pred_oof == 0)
        
        # 个数指标
        N_total = len(merged_df)
        N_conc = np.sum(pred_conc)
        N_waste = np.sum(pred_waste)
        
        count_yield_conc = N_conc / N_total * 100.0
        count_reject_rate = N_waste / N_total * 100.0
        
        # 质量指标
        W_conc = np.sum(weights[pred_conc])
        W_waste = np.sum(weights[pred_waste])
        
        mass_yield_conc = W_conc / W_total * 100.0
        mass_reject_rate = W_waste / W_total * 100.0
        
        # 金属回收率与品位
        # 1. 铜 (Cu)
        Cu_conc_mass = np.sum(weights[pred_conc] * merged_df.loc[pred_conc, 'Cu'] / 100.0)
        Cu_rec = (Cu_conc_mass / Cu_total_mass * 100.0) if Cu_total_mass > 0 else 0.0
        Cu_conc_grade = (Cu_conc_mass / W_conc * 100.0) if W_conc > 0 else 0.0
        Cu_er = (Cu_conc_grade / Cu_feed_grade) if Cu_feed_grade > 0 else 0.0
        
        # 铜尾矿品位
        Cu_waste_mass = np.sum(weights[pred_waste] * merged_df.loc[pred_waste, 'Cu'] / 100.0)
        Cu_waste_grade = (Cu_waste_mass / W_waste * 100.0) if W_waste > 0 else 0.0
        
        # 2. 铁 (Fe)
        Fe_conc_mass = np.sum(weights[pred_conc] * merged_df.loc[pred_conc, 'Fe'] / 100.0)
        Fe_rec = (Fe_conc_mass / Fe_total_mass * 100.0) if Fe_total_mass > 0 else 0.0
        Fe_conc_grade = (Fe_conc_mass / W_conc * 100.0) if W_conc > 0 else 0.0
        Fe_er = (Fe_conc_grade / Fe_feed_grade) if Fe_feed_grade > 0 else 0.0
        
        # 3. 硫 (S)
        S_conc_mass = np.sum(weights[pred_conc] * merged_df.loc[pred_conc, 'S'] / 100.0)
        S_rec = (S_conc_mass / S_total_mass * 100.0) if S_total_mass > 0 else 0.0
        S_conc_grade = (S_conc_mass / W_conc * 100.0) if W_conc > 0 else 0.0
        S_er = (S_conc_grade / S_feed_grade) if S_feed_grade > 0 else 0.0
        
        industrial_results.append({
            'Model': model_name,
            'Mass_Yield_Conc%': f"{mass_yield_conc:.2f}%",
            'Cu_Conc_Grade%': f"{Cu_conc_grade:.3f}%",
            'Cu_Rec%': f"{Cu_rec:.2f}%",
            'Cu_ER': f"{Cu_er:.2f}x",
            'Cu_Waste_Grade%': f"{Cu_waste_grade:.3f}%",
            'Fe_Conc_Grade%': f"{Fe_conc_grade:.2f}%",
            'Fe_Rec%': f"{Fe_rec:.2f}%",
            'Fe_ER': f"{Fe_er:.2f}x",
            'S_Conc_Grade%': f"{S_conc_grade:.2f}%",
            'S_Rec%': f"{S_rec:.2f}%",
            'S_ER': f"{S_er:.2f}x"
        })
        
    results_df = pd.DataFrame(industrial_results)
    
    print("\n" + "="*140)
    print("                                      INDUSTRIAL SORTING INDICATORS COMPARISON TABLE (OOF)")
    print("="*140)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print("="*140)
    
    # Save to CSV
    results_csv = os.path.join(abs_output_dir, "industrial_sorting_indicators.csv")
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"[{output_dir_name}] Industrial sorting indicators saved to: {results_csv}")
    
    # 绘制 PC2 抛废-回收率权衡曲线
    plot_pc2_tradeoff_curve(merged_df, abs_output_dir, output_dir_name, pc2_greater_than)
    
    return results_df

def plot_pc2_tradeoff_curve(merged_df, abs_output_dir, output_dir_name, pc2_greater_than=False):
    """
    遍历 PC2 阈值，绘制抛废率 vs 铜回收率的权衡曲线（类似于 ROC 曲线）。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    pc2_vals = merged_df['PC2_mean'].values
    weights = merged_df['weight_g'].values
    cu_grades = merged_df['Cu'].values
    
    total_mass = np.sum(weights)
    total_cu_mass = np.sum(weights * cu_grades / 100)
    
    if total_cu_mass == 0 or total_mass == 0:
        return
        
    min_val, max_val = np.min(pc2_vals), np.max(pc2_vals)
    # 生成 500 个候选阈值
    thresholds = np.linspace(min_val, max_val, 500)
    
    reject_rates = []
    cu_recoveries = []
    yield_rates = []
    cu_ers = []
    
    feed_cu_grade = (total_cu_mass / total_mass * 100)
    
    for theta in thresholds:
        if pc2_greater_than:
            pred_conc = pc2_vals > theta
        else:
            pred_conc = pc2_vals < theta
            
        conc_mass = np.sum(weights[pred_conc])
        conc_cu_mass = np.sum(weights[pred_conc] * cu_grades[pred_conc] / 100)
        
        yield_rate = conc_mass / total_mass if total_mass > 0 else 0
        cu_rec = conc_cu_mass / total_cu_mass if total_cu_mass > 0 else 0
        
        reject_rate = 1.0 - yield_rate
        cu_conc_grade = (conc_cu_mass / conc_mass * 100) if conc_mass > 0 else 0
        cu_er = cu_conc_grade / feed_cu_grade if feed_cu_grade > 0 else 0
        
        reject_rates.append(reject_rate * 100)
        cu_recoveries.append(cu_rec * 100)
        yield_rates.append(yield_rate * 100)
        cu_ers.append(cu_er)
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制百分比曲线 (左轴)
    ax1.plot(thresholds, cu_recoveries, color='darkorange', lw=2, label='铜回收率 (Cu Recovery %)')
    ax1.plot(thresholds, reject_rates, color='navy', lw=2, linestyle='-', label='抛废率 (Reject Rate %)')
    ax1.plot(thresholds, yield_rates, color='green', lw=2, linestyle='--', label='产率 (Yield Rate %)')
    
    ax1.set_ylim([-2.0, 102.0])
    ax1.set_xlabel('PC2 判定阈值 ($\\theta$)', fontsize=12)
    ax1.set_ylabel('百分比 (%)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 绘制富集比曲线 (右轴)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, cu_ers, color='firebrick', lw=2, linestyle='-.', label='铜富集比 (Cu ER)')
    ax2.set_ylabel('富集比 (倍)', fontsize=12, color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    
    direction_str = "PC2 > $\\theta$" if pc2_greater_than else "PC2 < $\\theta$"
    plt.title(f'PC2 阈值 vs 分选指标 趋势图\n判定精矿条件: {direction_str}', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=11)
    
    save_path = os.path.join(abs_output_dir, "pc2_threshold_tradeoff_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{output_dir_name}] PC2 Trade-off curve saved to: {save_path}")
