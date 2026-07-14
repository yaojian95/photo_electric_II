import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA

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
                
                L_all = np.concatenate(d['pixels_low']).astype(np.float32)
                H_all = np.concatenate(d['pixels_high']).astype(np.float32)
                
                v_max = 65535 if L_all.dtype == np.uint16 or np.max(L_all) > 255 else 255
                lower_th = utils_II.get_ore_lower_threshold(False, v_max)
                valid = (L_all >= lower_th) & (H_all >= lower_th) & (L_all < v_max) & (H_all < v_max)
                
                step_data[name] = (L_all[valid], H_all[valid])
    return step_data

def main():
    # 设置中文显示，防止图例和标签乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    base_dir = os.path.dirname(__file__)
    abs_input_dir = os.path.abspath(os.path.join(base_dir, '../results/20260407_Sample_test/'))
    abs_output_dir = os.path.abspath(os.path.join(base_dir, '../results/fit_PCA/'))
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # 1. 载入标梯数据并计算对数衰减
    print("Loading 0407 step data...")
    step_data = load_step_data(abs_input_dir, voltage='160kV')
    I0 = 204.0
    
    all_u_L = []
    all_u_H = []
    mat_pixels = {}
    
    for mat_name, (L, H) in step_data.items():
        u_L = np.log(I0 / np.maximum(L, 1.0))
        u_H = np.log(I0 / np.maximum(H, 1.0))
        all_u_L.append(u_L)
        all_u_H.append(u_H)
        mat_pixels[mat_name] = (u_L, u_H)
        
    X_ref = np.column_stack([np.concatenate(all_u_L), np.concatenate(all_u_H)])
    
    # 2. 拟合 PCA
    print("Fitting PCA on combined step pixels...")
    pca = PCA(n_components=2)
    pca.fit(X_ref)
    
    mean_u = pca.mean_  # 数据中心 (均值点)
    v1 = pca.components_[0]  # PC1 方向向量
    v2 = pca.components_[1]  # PC2 方向向量
    
    print(f"Data Mean: u_L={mean_u[0]:.4f}, u_H={mean_u[1]:.4f}")
    print(f"PC1 Vector: {v1}")
    print(f"PC2 Vector: {v2}")
    
    # 3. 绘图
    fig, ax = plt.subplots(figsize=(10, 10)) # 设置正方形画布
    
    # 绘制标样对数衰减散点
    step_colors = {
        'Cu_step': '#d62728',       # 红色
        'Fe_step': '#1f77b4',       # 蓝色
        'Al_step_block': '#2ca02c'  # 绿色
    }
    
    for mat_name, (u_L, u_H) in mat_pixels.items():
        color = step_colors.get(mat_name, '#7f7f7f')
        ax.scatter(u_L, u_H, color=color, alpha=0.03, s=0.3, label=f"{mat_name} 标样像素")
        
    # 绘制数据中心点
    ax.scatter(mean_u[0], mean_u[1], color='black', marker='X', s=120, zorder=5, label='数据均值中心 (Mean)')
    
    # 4. 绘制 PC1 和 PC2 轴向量线（以均值点为起点画正交箭头）
    # 为保证在图中清晰且成比例地显示：
    # 我们对向量长度（用其标准差std表示）进行缩放，并通过对坐标组件直接进行乘法缩放，
    # 从而保持角度不变。使用相同的 quiver 标尺并在等轴比（equal aspect）下绘制，确保它们在视觉上完全垂直！
    std1 = np.sqrt(pca.explained_variance_[0])
    std2 = np.sqrt(pca.explained_variance_[1])
    
    len_pc1 = std1 * 1.1
    len_pc2 = std2 * 5.0  # 适当缩短以和 PC1 保持视觉协调
    
    # 绘制 PC1 轴向量箭头（红色）
    ax.quiver(mean_u[0], mean_u[1], len_pc1 * v1[0], len_pc1 * v1[1], 
              angles='xy', scale_units='xy', scale=1.0,
              color='#d62728', width=0.006, headwidth=5, headlength=7, zorder=6,
              label=f'PC1 轴 (厚度方向, 权重=[{v1[0]:.3f}, {v1[1]:.3f}])')
               
    # 绘制 PC2 轴向量箭头（紫色）
    ax.quiver(mean_u[0], mean_u[1], len_pc2 * v2[0], len_pc2 * v2[1], 
              angles='xy', scale_units='xy', scale=1.0,
              color='#9467bd', width=0.006, headwidth=5, headlength=7, zorder=6,
              label=f'PC2 轴 (材质方向, 放大5倍, 权重=[{v2[0]:.3f}, {v2[1]:.3f}])')
               
    # 5. 坐标系等比控制与限制 (Crucial to make orthogonal vectors look 90 degrees!)
    ax.set_aspect('equal', adjustable='box')
    
    # 补充文本框说明向量方程 (去掉 fontfamily='monospace' 以防中文乱码)
    text_info = (
        f"数据中心 (Mean): ({mean_u[0]:.2f}, {mean_u[1]:.2f})\n"
        f"PC1 (厚度): 0.726 * u_L + 0.688 * u_H (方差解释比 {pca.explained_variance_ratio_[0]*100:.2f}%)\n"
        f"PC2 (材质): -0.688 * u_L + 0.726 * u_H (方差解释比 {pca.explained_variance_ratio_[1]*100:.2f}%)"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#bbbbbb')
    ax.text(0.05, 0.95, text_info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    ax.set_title("双能 XRT 对数衰减平面 $u_L$-$u_H$ 与 PCA 正交基向量图 (已做等轴校正)", fontsize=13, fontweight='bold')
    ax.set_xlabel(r"低能对数衰减值 $u_L$", fontsize=12)
    ax.set_ylabel(r"高能对数衰减值 $u_H$", fontsize=12)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 优化图例显示
    leg = ax.legend(fontsize=10, loc='lower right')
    if leg:
        for lh in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
            lh.set_alpha(1.0)
            if '标样像素' in lh.get_label():
                lh.set_sizes([20])
                
    save_path = os.path.join(abs_output_dir, "step_uluh_with_pca_vectors.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to: {save_path}")

if __name__ == '__main__':
    main()
