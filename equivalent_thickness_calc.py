import numpy as np
import pandas as pd

# 物理常量定义 (在 ~100 keV 能量下的近似值，基于 NIST XCOM 数据库)
# 质量衰减系数 (Mass attenuation coefficients) \mu_m: cm^2/g
MU_M_CU = 0.456
MU_M_FE = 0.371
MU_M_AL = 0.170
MU_M_S  = 0.191
MU_M_SI = 0.183
MU_M_O  = 0.155

# 纯金属密度 (g/cm^3)
RHO_CU = 8.96
RHO_FE = 7.87
RHO_AL = 2.70

# 纯金属的线衰减系数 (Linear attenuation coefficients) \mu: cm^-1
MU_CU = MU_M_CU * RHO_CU
MU_FE = MU_M_FE * RHO_FE
MU_AL = MU_M_AL * RHO_AL

# 矿物成分计算
# 黄铜矿 CuFeS2
w_cu_in_cp = 63.55 / (63.55 + 55.85 + 2 * 32.06)
w_fe_in_cp = 55.85 / (63.55 + 55.85 + 2 * 32.06)
w_s_in_cp  = (2 * 32.06) / (63.55 + 55.85 + 2 * 32.06)
MU_M_CP = w_cu_in_cp * MU_M_CU + w_fe_in_cp * MU_M_FE + w_s_in_cp * MU_M_S
RHO_CP = 4.2

# 黄铁矿 FeS2
w_fe_in_py = 55.85 / (55.85 + 2 * 32.06)
w_s_in_py  = (2 * 32.06) / (55.85 + 2 * 32.06)
MU_M_PY = w_fe_in_py * MU_M_FE + w_s_in_py * MU_M_S
RHO_PY = 5.0

# 脉石 (以石英 SiO2 为例)
w_si_in_qz = 28.085 / (28.085 + 2 * 15.999)
w_o_in_qz  = (2 * 15.999) / (28.085 + 2 * 15.999)
MU_M_QZ = w_si_in_qz * MU_M_SI + w_o_in_qz * MU_M_O
RHO_QZ = 2.65


def calc_ore_properties(cu_grade_percent: float, py_grade_percent: float = 0.0, porosity: float = 0.0) -> tuple:
    """
    根据铜品位、黄铁矿含量和孔隙率，计算矿石的整体质量衰减系数和密度。
    
    参数:
        cu_grade_percent: 铜品位 (%)
        py_grade_percent: 伴生黄铁矿含量 (%)
        porosity: 岩石孔隙率 (0.0~1.0)
        
    返回:
        mu_ore (线衰减系数 cm^-1), rho_ore_actual (实际密度 g/cm^3)
    """
    # 将质量百分比转化为分数
    w_cu = cu_grade_percent / 100.0
    w_py = py_grade_percent / 100.0
    
    # 假设所有的铜都来自黄铜矿
    w_cp = w_cu / w_cu_in_cp
    
    # 脉石比例
    w_qz = 1.0 - w_cp - w_py
    if w_qz < 0:
        raise ValueError(f"品位设定过高！黄铜矿占比 {w_cp:.2%}，黄铁矿占比 {w_py:.2%}，总和超过100%。")
        
    # 计算矿石的综合质量衰减系数 (成分的线性组合)
    mu_m_ore = w_cp * MU_M_CP + w_py * MU_M_PY + w_qz * MU_M_QZ
    
    # 计算矿石的理论密度 (假设体积可加性)
    inv_rho_ore = (w_cp / RHO_CP) + (w_py / RHO_PY) + (w_qz / RHO_QZ)
    rho_ore_theory = 1.0 / inv_rho_ore
    
    # 考虑孔隙率后的实际密度
    rho_ore_actual = rho_ore_theory * (1.0 - porosity)
    
    # 计算线衰减系数
    mu_ore = mu_m_ore * rho_ore_actual
    
    return mu_ore, rho_ore_actual

def calc_equivalent_thickness(metal_type: str, metal_thickness_mm: float, 
                              cu_grade_percent: float, py_grade_percent: float = 0.0, 
                              porosity: float = 0.0) -> float:
    """
    计算特定金属厚度对应的矿石等效厚度。
    """
    metal_type = metal_type.upper()
    if metal_type == 'CU':
        mu_metal = MU_CU
    elif metal_type == 'FE':
        mu_metal = MU_FE
    elif metal_type == 'AL':
        mu_metal = MU_AL
    else:
        raise ValueError("Unsupported metal type. Choose 'CU', 'FE', or 'AL'.")
        
    mu_ore, _ = calc_ore_properties(cu_grade_percent, py_grade_percent, porosity)
    
    # 核心换算公式
    ore_thickness_mm = metal_thickness_mm * (mu_metal / mu_ore)
    return ore_thickness_mm

if __name__ == "__main__":
    # 模拟输入参数
    ASSUMED_CU_GRADE = 2.0  # 假设铜品位 2%
    ASSUMED_PY_GRADE = 5.0  # 假设伴生黄铁矿 5%
    POROSITY = 0.05         # 假设孔隙率 5%
    
    print(f"=== 矿石等效厚度换算 ===")
    print(f"假设矿石参数: 铜品位 {ASSUMED_CU_GRADE}%, 伴生黄铁矿 {ASSUMED_PY_GRADE}%, 孔隙率 {POROSITY*100}%")
    
    mu_ore, rho_ore = calc_ore_properties(ASSUMED_CU_GRADE, ASSUMED_PY_GRADE, POROSITY)
    print(f"-> 矿石实际平均密度: {rho_ore:.2f} g/cm^3")
    print(f"-> 矿石平均线衰减系数 (\u03bc): {mu_ore:.3f} cm^-1")
    print("-" * 50)
    
    # 阶梯厚度范围 (2mm 到 20mm，步长 2mm)
    step_thicknesses_mm = np.arange(2.0, 22.0, 2.0)
    
    # 创建表格
    results = []
    for t_mm in step_thicknesses_mm:
        ore_cu = calc_equivalent_thickness('CU', t_mm, ASSUMED_CU_GRADE, ASSUMED_PY_GRADE, POROSITY)
        ore_fe = calc_equivalent_thickness('FE', t_mm, ASSUMED_CU_GRADE, ASSUMED_PY_GRADE, POROSITY)
        ore_al = calc_equivalent_thickness('AL', t_mm, ASSUMED_CU_GRADE, ASSUMED_PY_GRADE, POROSITY)
        
        results.append({
            "阶梯厚度 (mm)": t_mm,
            "等效矿石厚度 - 用纯铜标定 (mm)": round(ore_cu, 1),
            "等效矿石厚度 - 用纯铁标定 (mm)": round(ore_fe, 1),
            "等效矿石厚度 - 用纯铝标定 (mm)": round(ore_al, 1),
        })
        
    df = pd.DataFrame(results)
    
    # 控制台打印
    print(df.to_string(index=False))
    
    # 计算换算倍率
    ratio_cu = MU_CU / mu_ore
    ratio_fe = MU_FE / mu_ore
    ratio_al = MU_AL / mu_ore
    
    print("-" * 50)
    print("【换算倍率参考 (厚度放大倍数)】")
    print(f"纯铜 (Cu) -> 矿石: {ratio_cu:.2f} 倍")
    print(f"纯铁 (Fe) -> 矿石: {ratio_fe:.2f} 倍")
    print(f"纯铝 (Al) -> 矿石: {ratio_al:.2f} 倍")
    
    print("\n注: 由于射束硬化效应，以上为理论线性换算（基于~100keV单能近似）。在真实宽能谱XRT设备中，大厚度下实际矿石厚度可能略低于理论计算值。")
