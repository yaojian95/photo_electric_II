import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

materials = {0: 'Cu_step', 1: 'Fe_step', 2: 'Al_step_block'}
voltages = ['140kV', '160kV', '180kV']

output_dir = 'results/thickness_decoupling'
os.makedirs(output_dir, exist_ok=True)

for voltage in voltages:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    I0 = 204.0
    
    thickness_cu_fe_mm = np.arange(2, 22, 2)
    thickness_al_mm = np.arange(22, 42, 2)
    
    for idx, mat_name in materials.items():
        file_path = f'results/20260331/pixel_values/{voltage}_4mA_step_sample_{idx}_data.pkl'
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        L_raw = data['pixels_low']
        H_raw = data['pixels_high']
        
        # 记录每个阶梯的均值和标准差
        step_L_means, step_L_stds = [], []
        step_H_means, step_H_stds = [], []
        
        # 兼容阶梯样本的 list 格式以及纯方块的 ndarray 格式
        if isinstance(L_raw, list):
            for l_step, h_step in zip(L_raw, H_raw):
                step_L_means.append(np.mean(l_step) if l_step.size > 0 else np.nan)
                step_L_stds.append(np.std(l_step) if l_step.size > 0 else np.nan)
                step_H_means.append(np.mean(h_step) if h_step.size > 0 else np.nan)
                step_H_stds.append(np.std(h_step) if h_step.size > 0 else np.nan)
            
            L = np.concatenate(L_raw).astype(np.float32)
            H = np.concatenate(H_raw).astype(np.float32)
        else:
            L = L_raw.astype(np.float32)
            H = H_raw.astype(np.float32)
        
        # Filter valid pixels
        valid = (L > 1) & (H > 1) & (L < 255) & (H < 255)
        L_v, H_v = L[valid], H[valid]
        # print(len(L_v))
        
        # # Subsample
        # if len(L_v) > 100000:
        #     print(len(L_v))
        #     indices = np.random.choice(len(L_v), 100000, replace=False)
        #     L_sub, H_sub = L_v[indices], H_v[indices]
        # else:
        L_sub, H_sub = L_v, H_v
            
        # 准备厚度坐标 (绘图偏移标注已按指示处理)
        cur_thickness_mm = thickness_al_mm if 'Al' in mat_name else thickness_cu_fe_mm
        plot_thickness = cur_thickness_mm - 20 if 'Al' in mat_name else cur_thickness_mm
        display_label = f"{mat_name}" + (" (t-20mm)" if 'Al' in mat_name else "")

        # --- 第一排：原始强度 ---
        if len(L_sub) > 0:
            # 1. H vs L
            axes[0, 0].scatter(L_sub, H_sub, alpha=0.1, s=1)
            coeffs = np.polyfit(L_sub, H_sub, 2)
            axes[0, 0].plot(np.sort(L_sub), np.poly1d(coeffs)(np.sort(L_sub)), label=display_label)
        
        # 2. t vs L (带误差棒)
        if step_L_means:
            axes[0, 1].errorbar(plot_thickness, step_L_means, yerr=step_L_stds, fmt='o-', 
                               capsize=3, label=display_label, alpha=0.8)
        
        # 3. t vs H (带误差棒)
        if step_H_means:
            axes[0, 2].errorbar(plot_thickness, step_H_means, yerr=step_H_stds, fmt='o-', 
                               capsize=3, label=display_label, alpha=0.8)

        # --- 第二排：对数变换 (衰减量 ln(I0/I)) ---
        log_L_sub = np.log(I0 / L_sub)
        log_H_sub = np.log(I0 / H_sub)
        log_L_means = np.log(I0 / np.array(step_L_means))
        log_H_means = np.log(I0 / np.array(step_H_means))

        # 4. ln(I0/H) vs ln(I0/L)
        if len(log_L_sub) > 0:
            axes[1, 0].scatter(log_L_sub, log_H_sub, alpha=0.1, s=1)
            l_coeffs = np.polyfit(log_L_sub, log_H_sub, 2)
            axes[1, 0].plot(np.sort(log_L_sub), np.poly1d(l_coeffs)(np.sort(log_L_sub)), label=display_label)

        # --- 自动寻找线性区间算法 ---
        def find_linear_pts(x_pts, y_pts, label=""):
            # 至少需要 3 个点来验证线性度
            best_n = 3
            if len(x_pts) < 3: return len(x_pts)
            
            # 初始拟合
            prev_r2 = 1.0
            for n in range(3, len(x_pts) + 1):
                cur_x = x_pts[:n]
                cur_y = y_pts[:n]
                c = np.polyfit(cur_x, cur_y, 1)
                f = np.poly1d(c)
                # 计算 R^2
                ss_res = np.sum((cur_y - f(cur_x))**2)
                ss_tot = np.sum((cur_y - np.mean(cur_y))**2)
                cur_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # 判定条件：如果 R^2 显著下降（变得变差很多），则停止并回退
                # 这里的 0.99 和 0.005 是灵敏度阈值，可以根据需要调整
                if n > 3 and (cur_r2 < 0.99 or cur_r2 < prev_r2 - 0.005):
                    print(f"  [Linear Range Alert] {label} stopped at n={n-1} due to R2 drop ({prev_r2:.4f} -> {cur_r2:.4f})")
                    break
                best_n = n
                prev_r2 = cur_r2
            return best_n

        # 5. t vs ln(I0/L) 及其受限拟合
        axes[1, 1].plot(plot_thickness, log_L_means, 'o-', alpha=0.3) # 原始点画淡色
        line_color = axes[1, 1].get_lines()[-1].get_color()
        cur_t_mm = cur_thickness_mm[:len(log_L_means)]
        
        n_l = find_linear_pts(cur_t_mm, log_L_means, f"{mat_name} Low-E")
        l_coeffs = np.polyfit(cur_t_mm[:n_l], log_L_means[:n_l], 1)
        l_fit_fn = np.poly1d(l_coeffs)
        axes[1, 1].plot(plot_thickness[:n_l], l_fit_fn(cur_t_mm[:n_l]), '--', color=line_color, 
                        label=f"{display_label} (n={n_l})")

        # 6. t vs ln(I0/H) 及其受限拟合
        axes[1, 2].plot(plot_thickness, log_H_means, 'o-', alpha=0.3)
        n_h = find_linear_pts(cur_t_mm, log_H_means, f"{mat_name} High-E")
        h_coeffs = np.polyfit(cur_t_mm[:n_h], log_H_means[:n_h], 1)
        h_fit_fn = np.poly1d(h_coeffs)
        axes[1, 2].plot(plot_thickness[:n_h], h_fit_fn(cur_t_mm[:n_h]), '--', color=line_color, 
                        label=f"{display_label} (n={n_h})")
        
    # 设置样式
    for r in range(2):
        for c in range(3):
            axes[r, c].grid(True)
            axes[r, c].legend(fontsize='x-small')

    # Row 1 Labels & Limits
    axes[0, 0].set_title("H vs L Fit")
    axes[0, 0].set_xlim(0, 120); axes[0, 0].set_ylim(0, 130)
    axes[0, 1].set_title("Thickness vs Low Energy"); axes[0, 1].set_ylim(0, 130)
    axes[0, 2].set_title("Thickness vs High Energy"); axes[0, 2].set_ylim(0, 130)
    
    # Row 2 Labels
    axes[1, 0].set_title(r"$\ln(I_0/H)$ vs $\ln(I_0/L)$ Fit")
    axes[1, 0].set_xlim(0.5, 3.0); axes[1, 0].set_ylim(0.5, 2.8)
    axes[1, 1].set_title(r"Thickness vs $\ln(I_0/L)$")
    axes[1, 1].set_ylim(0.5, 5.0)
    axes[1, 2].set_title(r"Thickness vs $\ln(I_0/H)$")
    axes[1, 2].set_ylim(0.5, 5.0)
    
    plt.suptitle(f"Comprehensive Analysis for {voltage} (I0={I0})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/{voltage}_thickness_analysis.png")
    plt.close()
    
print(f"Analysis complete. Plots saved to {output_dir}")
