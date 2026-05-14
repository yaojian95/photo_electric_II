import numpy as np
import pandas as pd
from get_mu_from_nist_new import get_mu_rho_interpolated

# --- 物理常量与计算逻辑封装 ---

def get_mineral_properties(energy_kev):
    """
    根据给定能量，获取所有元素的质量衰减系数并计算矿物组分的属性。
    """
    # 1. 获取基础元素的质量衰减系数 (cm^2/g)
    _, _, mu_m_cu = get_mu_rho_interpolated('Cu', energy_kev)
    _, _, mu_m_fe = get_mu_rho_interpolated('Fe', energy_kev)
    _, _, mu_m_al = get_mu_rho_interpolated('Al', energy_kev)
    _, _, mu_m_s  = get_mu_rho_interpolated('S',  energy_kev)
    _, _, mu_m_si = get_mu_rho_interpolated('Si', energy_kev)
    _, _, mu_m_o  = get_mu_rho_interpolated('O',  energy_kev)
    
    # 2. 纯金属密度与线衰减系数
    rho_cu, rho_fe, rho_al = 8.96, 7.87, 2.70
    mu_cu = mu_m_cu * rho_cu
    mu_fe = mu_m_fe * rho_fe
    mu_al = mu_m_al * rho_al

    # 3. 矿物组分计算 (质量分数固定，mu_m 随能量变)
    
    # 黄铜矿 CuFeS2
    w_cu_in_cp = 63.55 / (63.55 + 55.85 + 2 * 32.06)
    w_fe_in_cp = 55.85 / (63.55 + 55.85 + 2 * 32.06)
    w_s_in_cp  = (2 * 32.06) / (63.55 + 55.85 + 2 * 32.06)
    mu_m_cp = w_cu_in_cp * mu_m_cu + w_fe_in_cp * mu_m_fe + w_s_in_cp * mu_m_s
    rho_cp = 4.2

    # 黄铁矿 FeS2
    w_fe_in_py = 55.85 / (55.85 + 2 * 32.06)
    w_s_in_py  = (2 * 32.06) / (55.85 + 2 * 32.06)
    mu_m_py = w_fe_in_py * mu_m_fe + w_s_in_py * mu_m_s
    rho_py = 5.0

    # 脉石 (以石英 SiO2 为例)
    w_si_in_qz = 28.085 / (28.085 + 2 * 15.999)
    w_o_in_qz  = (2 * 15.999) / (28.085 + 2 * 15.999)
    mu_m_qz = w_si_in_qz * mu_m_si + w_o_in_qz * mu_m_o
    rho_qz = 2.65
    
    return {
        'mu_cu': mu_cu, 'mu_fe': mu_fe, 'mu_al': mu_al,
        'mu_m_cp': mu_m_cp, 'rho_cp': rho_cp,
        'mu_m_py': mu_m_py, 'rho_py': rho_py,
        'mu_m_qz': mu_m_qz, 'rho_qz': rho_qz,
        'w_cu_in_cp': w_cu_in_cp
    }

def calc_ore_properties(cu_grade_percent, py_grade_percent, porosity, props):
    """
    根据品位和预计算的矿物属性计算矿石整体衰减系数。
    """
    w_cu = cu_grade_percent / 100.0
    w_py = py_grade_percent / 100.0
    w_cp = w_cu / props['w_cu_in_cp']
    w_qz = 1.0 - w_cp - w_py
    
    if w_qz < 0:
        raise ValueError("Grade too high! Sum of minerals exceeds 100%.")
        
    mu_m_ore = w_cp * props['mu_m_cp'] + w_py * props['mu_m_py'] + w_qz * props['mu_m_qz']
    inv_rho_ore = (w_cp / props['rho_cp']) + (w_py / props['rho_py']) + (w_qz / props['rho_qz'])
    rho_ore_theory = 1.0 / inv_rho_ore
    rho_ore_actual = rho_ore_theory * (1.0 - porosity)
    
    return mu_m_ore * rho_ore_actual, rho_ore_actual

def run_thickness_analysis(energy_kev, cu_grade, py_grade, porosity):
    print(f"\n" + "="*60)
    print(f"设定平均能量: {energy_kev:.1f} keV")
    print(f"设定矿石参数: 铜品位 {cu_grade}%, 伴生黄铁矿 {py_grade}%, 孔隙率 {porosity*100}%")
    print("="*60)
    
    props = get_mineral_properties(energy_kev)
    mu_ore, rho_ore = calc_ore_properties(cu_grade, py_grade, porosity, props)
    
    print(f"-> 矿石实际平均密度: {rho_ore:.2f} g/cm^3")
    print(f"-> 矿石平均线衰减系数 (\u03bc): {mu_ore:.3f} cm^-1")
    print("-" * 60)
    
    step_thicknesses_mm = np.arange(2.0, 22.0, 2.0)
    
    # 1. 矿石转换表格
    results = []
    for t_mm in step_thicknesses_mm:
        results.append({
            "阶梯厚度 (mm)": t_mm,
            "等效矿石-Cu (mm)": round(t_mm * (props['mu_cu'] / mu_ore), 1),
            "等效矿石-Fe (mm)": round(t_mm * (props['mu_fe'] / mu_ore), 1),
            "等效矿石-Al (mm)": round(t_mm * (props['mu_al'] / mu_ore), 1),
        })
    # print(pd.DataFrame(results).to_string(index=False))
    
    # 2. 金属间转换表格
    # print("\n" + "-"*30 + " 金属间转换 " + "-"*30)
    metal_results = []
    for t_mm in step_thicknesses_mm:
        metal_results.append({
            "原阶梯 (mm)": t_mm,
            "Cu -> Fe (mm)": round(t_mm * (props['mu_cu'] / props['mu_fe']), 2),
            "Cu -> Al (mm)": round(t_mm * (props['mu_cu'] / props['mu_al']), 2),
            "Fe -> Al (mm)": round(t_mm * (props['mu_fe'] / props['mu_al']), 2)
        })
    # print(pd.DataFrame(metal_results).to_string(index=False))

    # 3. 换算倍率汇总
    print("\n" + "-"*30 + " 换算倍率汇总 " + "-"*30)
    print(f"纯铜 (Cu) -> 矿石: {props['mu_cu'] / mu_ore:.2f} 倍")
    print(f"纯铁 (Fe) -> 矿石: {props['mu_fe'] / mu_ore:.2f} 倍")
    print(f"纯铝 (Al) -> 矿石: {props['mu_al'] / mu_ore:.2f} 倍")
    print(f"纯铜 (Cu) -> 纯铁 (Fe): {props['mu_cu'] / props['mu_fe']:.2f} 倍")
    print(f"纯铜 (Cu) -> 纯铝 (Al): {props['mu_cu'] / props['mu_al']:.2f} 倍")
    print(f"纯铁 (Fe) -> 纯铝 (Al): {props['mu_fe'] / props['mu_al']:.2f} 倍")
    print("\n注: 基于单能近似。真实宽能谱下大厚度建议进行硬化修正。")

if __name__ == "__main__":
    # 用户可以在此修改能量参数
    TARGET_ENERGY_KEV = 100.0  
    
    # 运行分析
    run_thickness_analysis(
        energy_kev = TARGET_ENERGY_KEV,
        cu_grade = 0.2,
        py_grade = 10.0,
        porosity = 0.05
    )
