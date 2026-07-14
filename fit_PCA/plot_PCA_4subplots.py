import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure parent directory is in path to import utils_II
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils_II

def load_step_data(abs_input_dir, voltage='160kV'):
    step_mats = {1: 'Cu_step', 3: 'Fe_step', 5: 'Al_step_block'}
    step_data = {}
    
    for idx, name in step_mats.items():
        p_path = os.path.join(abs_input_dir, 'pixel_values', f'Sample_{voltage}_test1_step_sample_{idx}_data.pkl')
        if os.path.exists(p_path):
            with open(p_path, 'rb') as f:
                d = pickle.load(f)
                
                # Concatenate all pixels across the 10 steps to draw the full trajectory
                L_all = np.concatenate(d['pixels_low']).astype(np.float32)
                H_all = np.concatenate(d['pixels_high']).astype(np.float32)
                
                # Filter out background and saturated pixels
                v_max = 65535 if L_all.dtype == np.uint16 or np.max(L_all) > 255 else 255
                lower_th = utils_II.get_ore_lower_threshold(False, v_max)
                valid = (L_all >= lower_th) & (H_all >= lower_th) & (L_all < v_max) & (H_all < v_max)
                
                step_data[name] = (L_all[valid], H_all[valid])
        else:
            print(f"Warning: Step file not found: {p_path}")
            
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

def plot_ore_overlay_subplot(ax, step_data, ore_L, ore_H, info_dict, title_text, mode='grayscale'):
    """
    在指定的 Axes 子图上绘制基准阶梯曲线并叠加矿石像素点分布。

    参数类型、含义及用法：
    - ax (matplotlib.axes.Axes): 当前绘图的子图子坐标系对象。
    - step_data (dict): 基准阶梯像素字典，格式为 {材质名: (L_array, H_array)}。
    - ore_L (np.ndarray): 矿石的低能灰度像素值数组。
    - ore_H (np.ndarray): 矿石的高能灰度像素值数组。
    - info_dict (dict): 包含矿石品位与厚度等属性的字典，用于在图中绘制文本框。
    - title_text (str): 当前子图的标题名称。
    - mode (str): 绘图模式，支持 'grayscale'（灰度值）和 'attenuation'（对数衰减值）。
    """
    # 1. 过滤矿石的死像素与饱和像素（基于原始灰度值）
    v_max = 65535 if ore_L.dtype == np.uint16 or np.max(ore_L) > 255 else 255
    lower_th = utils_II.get_ore_lower_threshold(False, v_max)
    valid = (ore_L >= lower_th) & (ore_H >= lower_th) & (ore_L < v_max) & (ore_H < v_max)
    ore_L_v, ore_H_v = ore_L[valid].astype(np.float32), ore_H[valid].astype(np.float32)

    # 2. 根据模式转换数据
    if mode == 'attenuation':
        # 衰减值计算：u = ln(I0 / I)，这里对于8位数据，空带强度I0=204.0
        I0 = 204.0
        ore_x = np.log(I0 / np.maximum(ore_L_v, 1.0))
        ore_y = np.log(I0 / np.maximum(ore_H_v, 1.0))
    else:
        ore_x = ore_L_v
        ore_y = ore_H_v

    # 3. 绘制背景基准阶梯标样
    step_colors = {
        'Cu_step': '#d62728',       # 红色
        'Fe_step': '#1f77b4',       # 蓝色
        'Al_step_block': '#2ca02c'  # 绿色
    }
    
    for mat_name, (L_v, H_v) in step_data.items():
        if mode == 'attenuation':
            I0 = 204.0
            step_x = np.log(I0 / np.maximum(L_v, 1.0))
            step_y = np.log(I0 / np.maximum(H_v, 1.0))
        else:
            step_x = L_v
            step_y = H_v
            
        color = step_colors.get(mat_name, '#7f7f7f')
        ax.scatter(step_x, step_y, color=color, alpha=0.03, s=0.3, label=f"{mat_name} Ref")

    # 4. 叠加矿石像素散点
    if len(ore_x) > 0:
        ax.scatter(ore_x, ore_y, color='#e377c2', alpha=0.3, s=1.5, label='Ore Pixels')
        
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    
    # 5. 坐标限制与标签
    if mode == 'attenuation':
        ax.set_xlabel(r"Low Energy Attenuation $u_L$", fontsize=10)
        ax.set_ylabel(r"High Energy Attenuation $u_H$", fontsize=10)
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 5.5)
    else:
        ax.set_xlabel("Low Energy (L)", fontsize=10)
        ax.set_ylabel("High Energy (H)", fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 6. 优化图例
    leg = ax.legend(fontsize='small', loc='upper left')
    if leg:
        for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
            lh.set_alpha(1.0)
            if lh.get_label() == 'Ore Pixels':
                lh.set_sizes([20])
            else:
                lh.set_sizes([10])

    # 7. 添加理化性质的文本框
    text_info = (
        f"Ore ID: {info_dict['ore_id']}\n"
        f"Cu Grade: {info_dict['cu']:.3f} %\n"
        f"Fe Grade: {info_dict['fe']:.2f} %\n"
        f"S Grade: {info_dict['s']:.2f} %\n"
        f"Mean Thickness: {info_dict['thickness']:.2f} mm"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='#cccccc')
    ax.text(0.95, 0.05, text_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, fontfamily='monospace')

def main():
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    base_dir = os.path.dirname(__file__)
    abs_input_dir = os.path.abspath(os.path.join(base_dir, '../results/20260407_Sample_test/'))
    abs_output_dir = os.path.abspath(os.path.join(base_dir, '../results/fit_PCA/'))
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # 1. 载入 0407 阶梯样数据作为基准线
    print("Loading 0407 step wedge data...")
    step_data = load_step_data(abs_input_dir, voltage='160kV')
    
    # 2. 载入 0325+0519+0520 矿石清洗后数据集
    print("Loading ore cleaned dataset...")
    ore_dataset = load_ore_dataset()
    pixels_df_list = ore_dataset[0]
    info_df = ore_dataset[1]
    
    # 定义多个样本组 (每一组包含 4 类满足分类条件的矿石)
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
    
    # 选项：灰度值模式 ('grayscale') 与 对数衰减值模式 ('attenuation')
    # 我们同时为这两种模式生成所有的对比大图，以便进行直观物理比较
    active_modes = ['grayscale', 'attenuation']
    
    for mode in active_modes:
        print(f"\n===== Plotting in {mode.upper()} Mode =====")
        for group_name, target_ids in ore_groups.items():
            print(f"Processing {group_name}...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes_flat = axes.flatten()
            
            for idx_sp, (key, cfg) in enumerate(target_ids.items()):
                g_id = cfg['id']
                title = cfg['title']
                ax = axes_flat[idx_sp]
                
                # 寻找对应的 DataFrame 行
                rows = info_df[info_df['global_id'] == g_id]
                if len(rows) == 0:
                    print(f"Error: global_id {g_id} not found in DataFrame!")
                    continue
                    
                row_idx = rows.index[0]
                ore_row = rows.iloc[0]
                
                # 提取像素数组
                ore_L = pixels_df_list[0].iloc[row_idx]
                ore_H = pixels_df_list[1].iloc[row_idx]
                
                # 构建信息描述
                info_dict = {
                    'ore_id': f"#{ore_row['global_id']} (Sample {ore_row['sample_id']})",
                    'cu': ore_row['Cu'],
                    'fe': ore_row['Fe'],
                    's': ore_row['S'],
                    'thickness': ore_row['mean_thickness']
                }
                
                # 绘图
                plot_ore_overlay_subplot(ax, step_data, ore_L, ore_H, info_dict, title, mode=mode)
                print(f"  Plotted Subplot {idx_sp+1} for global_id {g_id}")
                
            title_suffix = "高低能对数衰减值平面" if mode == 'attenuation' else "高低能灰度值平面"
            plt.suptitle(f"DE-XRT 阶梯标样与不同类型矿石的散点分布对比图 ({group_name} - {title_suffix})", 
                         fontsize=16, fontweight='bold', y=0.96)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(abs_output_dir, f"ore_step_hl_comparison_{group_name}_{mode}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved figure to: {save_path}")
            
    print("\nAll plots for all modes complete.")

if __name__ == '__main__':
    main()
