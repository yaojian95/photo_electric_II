import os
import sys
import pickle
import numpy as np
import pandas as pd
import json

# 添加父目录到系统路径以便导入 get_apd_acd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import get_apd_acd

def calculate_ores_properties_pipeline(
    pickle_path: str = r"E:\multi_source_info\data_dir\0325_input.pkl",
    filter_type: str = "0.6mm",
    voltage: str = "200kV",
    method: str = "M1"
) -> pd.DataFrame:
    """
    运行矿石属性计算管线，提取 apd/acd 并反算有效原子序数 (Ze) 与电子密度 (rho_e)。
    
    参数说明：
    - pickle_path (str): 矿石 0325_input.pkl 文件的物理路径。
      类型：str
      含义：包含原始双能通道像素点、灰度、化验数据的 pickle 数据包。
      用法：传入 pickle 文件的绝对路径。
    - filter_type (str): 滤片厚度。
      类型：str
      含义：用于过滤出对应的能谱重建和标定常数。
      用法：'0.6mm' 或 '1.2mm'。
    - voltage (str): 射线管管电压。
      类型：str
      含义：计算能谱和标定所用电压。
      用法：'200kV' 或 '280kV'。
    - method (str): 计算 APD/ACD 特征的方法。
      类型：str
      含义：可选项有 'Static'（静态单能近似）、'Dyn'（动态单能近似）、'M1'（连续能谱积分法）、'ChenWen'（陈文反演能谱积分法）。
      用法：'M1' 或 'ChenWen'。
      
    返回：
    - pd.DataFrame: 合并计算出的 Ze_mean, Ze_std, rho_e_mean, rho_e_std 后的完整矿石 DataFrame。
    """
    # 动态确定工作区根目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. 加载系统标定常数 (K1, g, nu)
    calib_path = os.path.join(script_dir, "results/thickness_decoupling/apd_acd_summary.json")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"未找到系统标定汇总文件：{calib_path}。请先运行 get_apd_acd.py 进行标定计算！")
        
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
        
    try:
        calib_info = calib_data[filter_type][voltage][method]["calibration"]
        K1 = calib_info["K1"]
        g = calib_info["g"]
        nu = calib_info["nu"]
    except KeyError:
        raise KeyError(f"标定 JSON 中未找到对应配置：{filter_type} -> {voltage} -> {method}")
        
    print(f"[+] 成功载入系统常数（配置 {filter_type} / {voltage} / {method}）：K1={K1}, g={g}, nu={nu}")
    
    # 2. 读取矿石数据集 0325_input.pkl
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"未找到矿石数据集文件：{pickle_path}")
        
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
        
    low_pixels_series = d[0][0]   # 低能像素 Series (长度 114)
    high_pixels_series = d[0][1]  # 高能像素 Series (长度 114)
    info_df = d[1].copy()         # 化验 DataFrame (长度 114)
    
    # 3. 载入重建能谱以获取能谱分布与网格
    if method == "ChenWen":
        chenwen_csv_path = os.path.join(script_dir, "chenwen/invsp_results", voltage, "reconstructed_spectra.csv")
        if not os.path.exists(chenwen_csv_path):
            raise FileNotFoundError(f"未找到陈文反演能谱文件：{chenwen_csv_path}")
        try:
            cw_df = pd.read_csv(chenwen_csv_path)
            energies = cw_df['energy_keV'].values
            S_L = cw_df['low'].values
            if filter_type == '0.6mm':
                S_H = cw_df['high_0.6mm'].values
            else:
                S_H = cw_df['high_1.2mm'].values
            
            # 归一化能谱分布
            S_L = S_L / max(np.sum(S_L), 1e-12)
            S_H = S_H / max(np.sum(S_H), 1e-12)
        except Exception as e:
            raise RuntimeError(f"加载陈文反演能谱失败: {e}")
    else:
        spectra_path = os.path.join(script_dir, f"results/thickness_decoupling/energy_hardening/spectrum_reconstruction/CuFe_4steps/reconstructed_spectra_summary.json")
        if not os.path.exists(spectra_path):
            raise FileNotFoundError(f"未找到能谱重建总结文件：{spectra_path}。请先运行 reconstruct_spectrum.py！")
            
        with open(spectra_path, "r", encoding="utf-8") as f:
            spectra_data = json.load(f)
            
        try:
            spec_info = spectra_data[filter_type][voltage]
            S_L = np.array(spec_info["S_L"])
            S_H = np.array(spec_info["S_H"])
            energies = np.array(spec_info["energies_keV"])
        except KeyError:
            raise KeyError(f"能谱数据中未找到对应配置：{filter_type} -> {voltage}")
        
    I0 = 52428.0  # 16位图像背景基准值
    results = []
    
    # 4. 循环计算每块矿石的物理属性
    print(f"[+] 开始遍历 {len(low_pixels_series)} 块矿石进行 APD/ACD 特征反演...")
    for idx in range(len(low_pixels_series)):
        le_raw = low_pixels_series[idx].astype(float)
        he_raw = high_pixels_series[idx].astype(float)
        
        # 兼容 Mean_thickness 列的获取
        d_mm = info_df.loc[idx, "Mean_thickness"]
        
        # 过滤无效像素 (剔除死像素、零点与饱和点)
        mask = (le_raw > 0) & (he_raw > 0) & (le_raw < 65535) & (he_raw < 65535)
        T_L = le_raw[mask] / I0
        T_H = he_raw[mask] / I0
        
        if len(T_L) == 0:
            results.append({
                "Ze_mean": np.nan, "Ze_std": np.nan,
                "rho_e_mean": np.nan, "rho_e_std": np.nan
            })
            continue
            
        # a. 解算像素级 apd 和 acd 特征值
        if method in ["M1", "ChenWen"]:
            apd_arr, acd_arr = get_apd_acd.solve_apd_acd_nonlinear(T_L, T_H, S_L, S_H, energies)
        else:
            # 兼容非连续能谱积分的常规 Dyn 或 Static 近似
            if method == "Dyn":
                E_L_mono = float(spec_info["E_L_eff_keV"])
                E_H_mono = float(spec_info["E_H_eff_keV"])
            else:  # Static
                E_L_mono = 58.0
                E_H_mono = 105.0
            apd_arr, acd_arr = get_apd_acd.calculate_apd_acd_mono(T_L, T_H, E_L_mono, E_H_mono)
            
        # 剔除无效或异常的解
        valid = np.isfinite(apd_arr) & np.isfinite(acd_arr)
        apd_arr, acd_arr = apd_arr[valid], acd_arr[valid]
        
        if len(apd_arr) == 0:
            results.append({
                "Ze_mean": np.nan, "Ze_std": np.nan,
                "rho_e_mean": np.nan, "rho_e_std": np.nan
            })
            continue
            
        # b. 比厚度归一化：获取比系数 ap 和 ac
        ap_arr = apd_arr / d_mm
        ac_arr = acd_arr / d_mm
        
        # c. 使用标定好的常数反算 Ze 与 rho_e
        rho_e_arr, Ze_arr = get_apd_acd.compute_sirz_properties(ap_arr, ac_arr, d_mm, K1, g, nu)
        
        results.append({
            "Ze_mean": float(np.mean(Ze_arr)),
            "Ze_std": float(np.std(Ze_arr)),
            "rho_e_mean": float(np.mean(rho_e_arr)),
            "rho_e_std": float(np.std(rho_e_arr))
        })
        
    # 5. 合并并保存结果 DataFrame
    res_df = pd.DataFrame(results)
    final_df = pd.concat([info_df, res_df], axis=1)
    
    # 重命名乱码列以确保输出友好
    # d[1] 第一列 '' 为 ID，第 10 列 '' 为序号，第 11 列 'ܶ' 为密度，第 12 列 'ƽԭ' 为平均原子序数
    rename_cols = {}
    orig_cols = final_df.columns.tolist()
    if orig_cols[0] == '':
        rename_cols[orig_cols[0]] = 'ID'
    if len(orig_cols) > 9 and orig_cols[9] == '':
        rename_cols[orig_cols[9]] = 'Index'
    if len(orig_cols) > 10 and orig_cols[10] == 'ܶ':
        rename_cols[orig_cols[10]] = 'Density'
    if len(orig_cols) > 11 and orig_cols[11] == 'ƽԭ':
        rename_cols[orig_cols[11]] = 'Theoretical_Zeff'
        
    final_df = final_df.rename(columns=rename_cols)
    return final_df

def plot_ze_comparison(df: pd.DataFrame, save_path: str, method: str = "M1") -> None:
    """
    绘制并计算有效原子序数 Ze_mean 与理论有效原子序数 Theoretical_Zeff 的对比散点图。
    
    参数说明：
    - df (pd.DataFrame): 包含计算得到的 Ze_mean 和 Theoretical_Zeff 字段的数据框。
      类型：pd.DataFrame
      含义：包含矿石物理特征反解及化验结果的数据。
      用法：传入计算完的 final_df 数据框。
    - save_path (str): 图像保存的磁盘物理路径。
      类型：str
      含义：生成的 PNG 图表路径。
      用法：传入合法的物理路径。
    - method (str): 计算特征的方法名称。
      类型：str
      含义：可选方法名如 'M1' 或 'ChenWen' 等，用于在 Y 轴标注。
      用法：传入 'M1' 或 'ChenWen'。
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    # 按照公式计算理论原子序数 Zeff
    w_cu = df['Cu_grade'] / 100.0
    w_fe = df['Fe_grade'] / 100.0
    w_s = df['S_grade'] / 100.0
    w_gangue = np.clip(1.0 - w_cu - w_fe - w_s, 0.0, 1.0)
    df['Theoretical_Zeff'] = w_cu * 29.0 + w_fe * 26.0 + w_s * 16.0 + w_gangue * 11.0

    valid_mask = df['Ze_mean'].notna() & df['Theoretical_Zeff'].notna()
    x = df.loc[valid_mask, 'Theoretical_Zeff']
    y = df.loc[valid_mask, 'Ze_mean']
    
    if len(x) == 0:
        print("[-] 没有有效数据可用于绘制对比图！")
        return
        
    r_val, _ = pearsonr(x, y)
    
    plt.figure(figsize=(8, 7))
    plt.scatter(x, y, color='#E28743', edgecolors='k', alpha=0.8, s=50, label='Ores')
    
    # 绘制 y = x 恒等线
    lims = [min(min(x), min(y)) - 0.5, max(max(x), max(y)) + 0.5]
    plt.plot(lims, lims, '--', color='gray', alpha=0.7, label='y = x')
    
    plt.title("Calculated $Z_e$ vs Theoretical $Z_{eff}$ (0325 Yinshan Ores)", fontsize=13, fontweight='bold')
    plt.xlabel("Theoretical $Z_{eff}$ ($w_{Cu} \cdot 29 + w_{Fe} \cdot 26 + w_{S} \cdot 16 + (1 - w_{Cu} - w_{Fe} - w_{S}) \cdot 11$)", fontsize=10)
    plt.ylabel(f"Calculated $Z_e$ ({method} Spectrum-Integrated)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(lims)
    plt.ylim(lims)
    
    # 标注相关系数
    plt.text(0.05, 0.95, f"Pearson r = {r_val:.4f}\nSamples Count = {len(x)}", 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] 对比图已成功保存至：{save_path}")


if __name__ == "__main__":
    # 动态确定工作区根目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 运行 M1 与 ChenWen 两种方法进行对比
    for m in ["M1", "ChenWen"]:
        try:
            print(f"\n==================== 运行方法: {m} ====================")
            final_df = calculate_ores_properties_pipeline(
                pickle_path=r"E:\multi_source_info\data_dir\0325_input.pkl",
                filter_type="0.6mm",
                voltage="200kV",
                method=m
            )
            
            # 确保 Theoretical_Zeff 被正确计算并加入到 dataframe 中
            w_cu = final_df['Cu_grade'] / 100.0
            w_fe = final_df['Fe_grade'] / 100.0
            w_s = final_df['S_grade'] / 100.0
            w_gangue = np.clip(1.0 - w_cu - w_fe - w_s, 0.0, 1.0)
            final_df['Theoretical_Zeff'] = w_cu * 29.0 + w_fe * 26.0 + w_s * 16.0 + w_gangue * 11.0

            print(f"\n[+] {m} 计算成功！前 5 行数据预览：")
            cols_to_show = ["Cu_grade", "Fe_grade", "Mean_thickness", "Ze_mean", "Theoretical_Zeff", "rho_e_mean"]
            print(final_df[cols_to_show].head())
            
            # 绘制 Z_e vs Z_eff 对比图
            comparison_plot_path = os.path.join(script_dir, f"results/thickness_decoupling/0325_ores_ze_vs_zeff_{m}.png")
            plot_ze_comparison(final_df, comparison_plot_path, method=m)
            
            # 结果保存至 results 目录中
            output_csv = os.path.join(script_dir, f"results/thickness_decoupling/0325_ores_calculated_properties_{m}.csv")
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[+] {m} 数据分析结果已成功持久化至：{output_csv}")
        except Exception as e:
            print(f"\n[-] {m} 管线运行遇到错误: {e}")

