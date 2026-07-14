import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure parent directory is in path to import utils_II
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils_II

def plot_hl_scatter(voltage, samples_dict, output_subdir, title_prefix):
    """
    绘制阶梯样品的低能(Low-E)与高能(High-E)强度的H-L原始散点分布图（不进行拟合）。

    参数类型、含义及用法：
    - voltage (str): 当前处理的管电压（例如 '160kV'），用于标题和输出文件名区分。
    - samples_dict (dict): 样本数据字典。格式为 {材质名: (低能像素列表, 高能像素列表)}。
                           其中低/高能像素列表为各厚度阶梯对应的二维/一维像素值数组。
    - output_subdir (str): 结果图片保存的目标文件夹路径。
    - title_prefix (str): 散点图标题前缀，如 '0407 Step Sample'。
    """
    os.makedirs(output_subdir, exist_ok=True)
    
    # 使用学术风格的高对比度色彩配置
    colors = {
        'Cu_step': '#d62728',       # 红色
        'Fe_step': '#1f77b4',       # 蓝色
        'Al_step_block': '#2ca02c'  # 绿色
    }

    plt.figure(figsize=(10, 8))
    
    for mat_name, (L_list, H_list) in samples_dict.items():
        L_all = np.concatenate(L_list).astype(np.float32)
        H_all = np.concatenate(H_list).astype(np.float32)
        
        v_max = 65535 if L_all.dtype == np.uint16 or np.max(L_all) > 255 else 255
        lower_th = utils_II.get_ore_lower_threshold(False, v_max)
        
        # 过滤底噪、饱和和异常像素
        valid = (L_all >= lower_th) & (H_all >= lower_th) & (L_all < v_max) & (H_all < v_max)
        L_v, H_v = L_all[valid], H_all[valid]
        
        if len(L_v) > 0:
            color = colors.get(mat_name, '#7f7f7f')
            plt.scatter(L_v, H_v, color=color, alpha=0.08, s=0.6, label=mat_name)

    plt.title(f"{title_prefix} - H vs L Scatter Plot ({voltage})", fontsize=14, fontweight='bold')
    plt.xlabel("Low Energy Intensity (L)", fontsize=12)
    plt.ylabel("High Energy Intensity (H)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 优化 Legend 显示
    leg = plt.legend(fontsize=10, loc='upper left')
    if leg:
        for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
            lh.set_alpha(1.0)
            
    plt.tight_layout()
    save_path = os.path.join(output_subdir, f"{voltage}_hl_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"H-L Scatter plot saved to {save_path}")

def main():
    # 0407 Dataset Configuration
    voltages = ['160kV']
    input_dir = '../results/20260407_Sample_test/'
    output_dir = '../results/fit_PCA/'
    
    # 解决路径相对于当前执行目录的问题
    base_dir = os.path.dirname(__file__)
    abs_input_dir = os.path.abspath(os.path.join(base_dir, input_dir))
    abs_output_dir = os.path.abspath(os.path.join(base_dir, output_dir))
    
    step_mats = {1: 'Cu_step', 3: 'Fe_step', 5: 'Al_step_block'}
    
    for voltage in voltages:
        print(f"\n>>> Processing {voltage} for PCA plot ...")
        step_data = {}
        for idx, name in step_mats.items():
            p_path = os.path.join(abs_input_dir, 'pixel_values', f'Sample_{voltage}_test1_step_sample_{idx}_data.pkl')
            if os.path.exists(p_path):
                with open(p_path, 'rb') as f:
                    d = pickle.load(f)
                    step_data[name] = (d['pixels_low'], d['pixels_high'])
            else:
                print(f"File not found: {p_path}")
        
        if step_data:
            plot_hl_scatter(voltage, step_data, abs_output_dir, "0407 Step Sample")
            
if __name__ == "__main__":
    main()
