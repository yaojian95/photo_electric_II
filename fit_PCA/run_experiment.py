import os
import sys
import pandas as pd

# Ensure module is discoverable
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from fit_pca_classifier import run_classification_pipeline
from calculate_industrial_metrics import evaluate_industrial_metrics
from calculate_subgroup_metrics import evaluate_subgroup_metrics
from metal_mapping import METAL_MAPPING

def run_experiment(exp_name, step_config, pc2_greater_than=False):
    print("\n" + "="*80)
    print(f"Starting Experiment: {exp_name}")
    print("="*80)
    
    # 1. 运行核心 PCA 分类流程
    # 返回：pca模型，含特征的矿石数据集，模型配置字典，样本权重
    pca_model, merged_df, classifiers, weights = run_classification_pipeline(
        step_config=step_config, 
        output_dir_name=exp_name,
        pc2_greater_than=pc2_greater_than
    )
    
    # 2. 评估工业指标 (Industrial Metrics)
    # 内部将读取 merged_df 并计算对应指标
    evaluate_industrial_metrics(merged_df, output_dir_name=exp_name, pc2_greater_than=pc2_greater_than)
    
    # 3. 评估子组质量准确率指标 (Subgroup Metrics)
    # 内部根据 merged_df 进行重新 5 折预测，并划分细分子组
    evaluate_subgroup_metrics(merged_df, output_dir_name=exp_name, pc2_greater_than=pc2_greater_than)
    
    print(f"\n[SUCCESS] Experiment {exp_name} completed.")

def generate_comparison_report(exp_names, output_filename="experiment_comparison_report.md"):
    """
    读取多个实验的结果文件并生成对比的 Markdown 报告。
    """
    report_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'fit_pca'))
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, output_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 实验结果综合对比报告\n\n")
        
        # 1. 工业选别指标对比
        f.write("## 1. 工业选别指标对比 (全模型)\n\n")
        f.write("| 实验名称 | 模型 | 总体质量准确率 (Mass_Acc) | 产率 (Mass_Yield_Conc%) | 铜回收率 (Cu_Rec%) | 铜富集比 (Cu_ER) | 抛废率 (1-Yield) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for exp in exp_names:
            csv_path = os.path.join(report_dir, exp, 'industrial_sorting_indicators.csv')
            summary_path = os.path.join(report_dir, exp, 'classification_summary.csv')
            if os.path.exists(csv_path):
                mass_acc_dict = {}
                if os.path.exists(summary_path):
                    try:
                        summary_df = pd.read_csv(summary_path, encoding='utf-8-sig')
                        for _, r in summary_df.iterrows():
                            # classification_summary.csv model name + feature combo might not perfectly match, 
                            # but for RF/GB/SVC it's "Model" column. Wait, classification summary has Model and Feat.
                            # For PCA Fusion, it's e.g., "Random Forest (RF)" and "PCA Fusion".
                            # Industrial metrics has "Random Forest (RF) - PCA Fusion".
                            if r['Feat'] == 'PCA Fusion':
                                m_name = f"{r['Model']} - PCA Fusion"
                            else:
                                m_name = r['Model']
                            mass_acc_dict[m_name] = r['Mass_Acc']
                    except Exception as e:
                        pass
                
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    for _, row in df.iterrows():
                        model_name = row['Model']
                        yield_val = row['Mass_Yield_Conc%']
                        cu_rec = row['Cu_Rec%']
                        cu_er = row['Cu_ER']
                        try:
                            yield_float = float(str(yield_val).strip('%')) / 100
                            reject_rate = f"{(1 - yield_float)*100:.2f}%"
                        except:
                            reject_rate = "N/A"
                            
                        mass_acc = mass_acc_dict.get(model_name, "N/A")
                        f.write(f"| {exp} | {model_name} | {mass_acc} | {yield_val} | {cu_rec} | {cu_er} | {reject_rate} |\n")
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    
        # 2. 极端子组质量准确率对比
        f.write("\n## 2. 子组质量准确率综合对比\n\n")
        f.write("下表反映了各模型在不同厚度、铁品位及铜品位区间下的详细分类表现。\n\n")
        
        for exp in exp_names:
            csv_path = os.path.join(report_dir, exp, 'subgroup_mass_accuracy.csv')
            if os.path.exists(csv_path):
                # 获取总体质量加权准确率
                summary_path = os.path.join(report_dir, exp, 'classification_summary.csv')
                mass_acc_summary = []
                if os.path.exists(summary_path):
                    try:
                        summary_df = pd.read_csv(summary_path, encoding='utf-8-sig')
                        for _, r in summary_df.iterrows():
                            if r['Feat'] == 'PCA Fusion':
                                m_name = f"{r['Model']} - PCA Fusion"
                            else:
                                m_name = r['Model']
                            mass_acc_summary.append(f"- **{m_name}**: {r['Mass_Acc']}")
                    except Exception as e:
                        pass
                
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    f.write(f"### 实验: {exp}\n\n")
                    if mass_acc_summary:
                        f.write("**总体质量加权准确率 (Mass_Acc)**:\n")
                        f.write("\n".join(mass_acc_summary) + "\n\n")
                    
                    headers = df.columns.tolist()
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                    for _, row in df.iterrows():
                        f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
                    f.write("\n")
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    
    print(f"\n[REPORT] Comparison report generated at: {report_path}")

if __name__ == '__main__':
    # 基准输入目录
    input_base = os.path.abspath(os.path.join(base_dir, '..', 'results', '20260407_Sample_test', 'pixel_values'))
    
    # ==============================================================
    # 实验 1: 基准，所有金属均为默认阶梯（全加载，10阶）
    # ==============================================================
    config_10_10_10 = {
        'Cu_step': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_1_data.pkl'), 
            'steps': None,
            'thicknesses': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        },
        'Fe_step': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_3_data.pkl'), 
            'steps': None,
            'thicknesses': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        },
        'Al_step_block': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_5_data.pkl'), 
            'steps': None,
            'thicknesses': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        }
    }
    
    # ==============================================================
    # 实验 2: 铜前4阶、铁前4阶、铝全10阶
    # ==============================================================
    config_4_4_10 = {
        'Cu_step': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_1_data.pkl'), 
            'steps': [0, 1, 2, 3],
            'thicknesses': [2.0, 4.0, 6.0, 8.0]
        },
        'Fe_step': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_3_data.pkl'), 
            'steps': [0, 1, 2, 3],
            'thicknesses': [2.0, 4.0, 6.0, 8.0]
        },
        'Al_step_block': {
            'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_5_data.pkl'), 
            'steps': None,
            'thicknesses': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        }
    }
    
    # ==============================================================
    # 实验 3: 使用20260611研究源提供的新金属片（8种全金属，PCA基础正交基）
    # ==============================================================
    new_dataset_base = r"E:\multi_source_info\data_dir\20260611_metal_sheet_yanjiuyuan\160kV_4mA\pixel_values"
    config_all_metals = {}
    for metal, mapping in METAL_MAPPING.items():
        if metal == 'Other':
            continue
        # 为了与原本从薄到厚排列的要求一致，反转 indices
        indices = mapping['indices'][::-1]
        file_paths = [os.path.join(new_dataset_base, f"160kV_4mA_block_{i}.pkl") for i in indices]
        
        config_all_metals[f"{metal}_step"] = {
            'file_path': file_paths,
            'steps': None,
            'thicknesses': mapping['thicknesses'][::-1]
        }
    
    # ==============================================================
    # 实验 4: 仅使用 Z <= 30 的金属（融合新旧阶梯）
    # ==============================================================
    config_Z_le_30 = {}
    
    # 1. 放入旧的 10阶 铜铁铝
    config_Z_le_30['Cu_step_old'] = {
        'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_1_data.pkl'), 
        'steps': [0, 1, 2, 3],
        'thicknesses': [2.0, 4.0, 6.0, 8.0]
    }
    config_Z_le_30['Fe_step_old'] = {
        'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_3_data.pkl'), 
        'steps': [0, 1, 2, 3],
        'thicknesses': [2.0, 4.0, 6.0, 8.0]
    }
    config_Z_le_30['Al_step_old'] = {
        'file_path': os.path.join(input_base, 'Sample_160kV_test1_step_sample_5_data.pkl'), 
        'steps': None,
        'thicknesses': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    }
    
    # 2. 放入新的 Z <= 30 的金属片
    z_map_local = {'Ti': 22, 'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Sn': 50, 'W': 74, 'Pb': 82}
    for metal, mapping in METAL_MAPPING.items():
        if metal == 'Other':
            continue
        z_val = z_map_local.get(metal, 999)
        if z_val <= 30:
            indices = mapping['indices'][::-1]
            file_paths = [os.path.join(new_dataset_base, f"160kV_4mA_block_{i}.pkl") for i in indices]
            
            config_Z_le_30[f"{metal}_step_new"] = {
                'file_path': file_paths,
                'steps': None,
                'thicknesses': mapping['thicknesses'][::-1]
            }
    
    # ==============================================================
    # 实验 5: 仅使用 Z <= 30 的新金属片（不包含旧铜铁铝阶梯）
    # ==============================================================
    config_metals_Z_le_30_new_only = {}
    for metal, mapping in METAL_MAPPING.items():
        if metal == 'Other':
            continue
        z_val = z_map_local.get(metal, 999)
        if z_val <= 30:
            indices = mapping['indices'][::-1]
            file_paths = [os.path.join(new_dataset_base, f"160kV_4mA_block_{i}.pkl") for i in indices]
            
            config_metals_Z_le_30_new_only[f"{metal}_step_new"] = {
                'file_path': file_paths,
                'steps': None,
                'thicknesses': mapping['thicknesses'][::-1]
            }

    # 依次执行对比实验
    # run_experiment("exp_10_10_10", config_10_10_10)
    # run_experiment("exp_4_4_10", config_4_4_10)
    # run_experiment("exp_all_metals_160kV_4mA", config_all_metals)
    # run_experiment("exp_metals_Z_le_30", config_Z_le_30)
    run_experiment("exp_metals_Z_le_30_new_only", config_metals_Z_le_30_new_only)
    
    print("\n" + "="*80)
    print("All experiments completed successfully! Results are isolated in their respective folders under results/fit_pca/.")
    # print("Skipping exp_metals_Z_le_30_new_only for now.")

    # ==============================================================
    # 实验 4: 仅使用 Z<=30 的新金属片 (大于阈值为精矿)
    # ==============================================================
    run_experiment("exp_metals_Z_le_30_new_only_greater_than", config_metals_Z_le_30_new_only, pc2_greater_than=True)

    # 4. 生成综合对比报告
    generate_comparison_report([
        "exp_10_10_10", 
        "exp_metals_Z_le_30", 
        "exp_metals_Z_le_30_new_only",
        "exp_metals_Z_le_30_new_only_greater_than"
    ])
