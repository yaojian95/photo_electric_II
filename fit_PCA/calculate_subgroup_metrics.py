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

def evaluate_subgroup_metrics(merged_df, output_dir_name="fit_PCA", pc2_greater_than=False):
    abs_output_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'fit_pca', output_dir_name))
    os.makedirs(abs_output_dir, exist_ok=True)
    
    print(f"\n[{output_dir_name}] Evaluating subgroup metrics...")
    
    # 2. 确定分类基础
    y = (merged_df['Cu'] >= 0.10).astype(int).values
    weights = merged_df['weight_g'].values
    
    # 特征集
    X_pc2_only = merged_df[['PC2_mean']].values
    X_fusion = merged_df[[
        'PC1_mean', 'PC1_std', 'PC1_skew', 'PC1_kurt',
        'PC2_mean', 'PC2_std', 'PC2_skew', 'PC2_kurt',
        'mean_thickness', 'weight_g'
    ]].values
    
    # 待评估模型
    models_config = {
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
    oof_predictions = {}
    
    # 运行 5 折交叉验证以获取各模型 OOF 预测
    for model_name, config in models_config.items():
        clf = config['clf']
        X = config['X']
        
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
            
        oof_predictions[model_name] = y_pred_oof
        
    # 3. 按中位数和品位边界划分 8 类矿石
    # 3.1 厚度中位数和铁品位中位数
    t_median = merged_df['mean_thickness'].median()
    fe_median = merged_df['Fe'].median()
    
    print(f"\n厚度中位数: {t_median:.2f} mm")
    print(f"铁品位中位数: {fe_median:.2f} %")
    
    # 3.2 增加标记
    merged_df['t_group'] = np.where(merged_df['mean_thickness'] <= t_median, 'Thin', 'Thick')
    merged_df['fe_group'] = np.where(merged_df['Fe'] <= fe_median, 'Low Fe', 'High Fe')
    
    # 3.3 筛选 Cu < 0.05% 和 Cu > 0.1% 的样本
    valid_mask = (merged_df['Cu'] < 0.05) | (merged_df['Cu'] > 0.10)
    filtered_df = merged_df[valid_mask].copy()
    filtered_df['cu_group'] = np.where(filtered_df['Cu'] < 0.05, 'Cu<0.05%', 'Cu>0.1%')
    
    print(f"原始数据集大小: {len(merged_df)}")
    print(f"剔除中间过渡带 (0.05% <= Cu <= 0.1%) 后样本数: {len(filtered_df)}")
    
    # 3.4 划分 8 个子组
    subgroups = [
        # (Thickness, Fe, Cu)
        ('Thin', 'Low Fe', 'Cu<0.05%'),
        ('Thin', 'Low Fe', 'Cu>0.1%'),
        ('Thin', 'High Fe', 'Cu<0.05%'),
        ('Thin', 'High Fe', 'Cu>0.1%'),
        ('Thick', 'Low Fe', 'Cu<0.05%'),
        ('Thick', 'Low Fe', 'Cu>0.1%'),
        ('Thick', 'High Fe', 'Cu<0.05%'),
        ('Thick', 'High Fe', 'Cu>0.1%')
    ]
    
    subgroup_results = []
    
    for t_g, fe_g, cu_g in subgroups:
        sub_mask = (filtered_df['t_group'] == t_g) & (filtered_df['fe_group'] == fe_g) & (filtered_df['cu_group'] == cu_g)
        sub_df = filtered_df[sub_mask]
        
        count = len(sub_df)
        total_w = np.sum(sub_df['weight_g'])
        
        # 获取各模型的预测准确率
        row_result = {
            'Thickness': t_g,
            'Fe_Grade': fe_g,
            'Cu_Grade': cu_g,
            'Count': count,
            'Total_Weight(g)': f"{total_w:.1f}"
        }
        
        for model_name in models_config.keys():
            y_pred_oof = oof_predictions[model_name][sub_df.index]
            y_true = (sub_df['Cu'] >= 0.10).astype(int).values
            w_sub = sub_df['weight_g'].values
            
            if len(sub_df) > 0:
                # 质量准确率
                correct_mask = (y_pred_oof == y_true)
                mass_acc = np.sum(w_sub[correct_mask]) / np.sum(w_sub) * 100.0
                row_result[model_name] = f"{mass_acc:.2f}%"
            else:
                row_result[model_name] = "N/A"
                
        subgroup_results.append(row_result)
        
    results_df = pd.DataFrame(subgroup_results)
    
    print("\n" + "="*120)
    print("                                      SUBGROUP MASS ACCURACY SUMMARY TABLE (OOF)")
    print("="*120)
    print(results_df.to_string(index=False))
    print("="*120)
    
    # Save to CSV
    csv_path = os.path.join(abs_output_dir, "subgroup_mass_accuracy.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[{output_dir_name}] Saved subgroup mass accuracy to: {csv_path}")
    
    return results_df
