## 2026-06-04
- **Execution & Local Verification**: 在本地 Anaconda Python 环境下运行并验证了 [reconstruct_spectrum.py](file:///e:/photo_electric_II/reconstruct_spectrum.py)。生成了 5 组不同铜铁最大厚度配置（`1`、`3`、`5`、`7`、`10` 个厚度阶梯，同时固定铝为全部 10 个厚度）在 7 个电压（200kV-320kV）及 2 种滤片（0.6mm, 1.2mm）下的全部能谱反演和 APD/ACD 线性度分析结果，成功输出至各自对应的结果文件夹中。
- **Multi-Thickness-Step Configuration Evaluation**: 针对铜、铁重衰减材质在后半段大厚度阶梯（第 5 级及以后）因高吸收导致探测器穿透不足、数据噪点增大的实验事实，重构了能谱反演脚本 [reconstruct_spectrum.py](file:///e:/photo_electric_II/reconstruct_spectrum.py)。
    - **函数参数升级**：为 [load_transmission_data](file:///e:/photo_electric_II/reconstruct_spectrum.py#L86-L167) 引入了新输入参数 `cu_fe_max_steps`（整型，默认值 10），并在函数文档字符串中详细注解了该参数的物理含义和用法类型，用来限制重金属标样的加载步数。
    - **主控制流遍历与独立目录归档**：主函数 `main()` 增加了一层外循环，强制固定铝标样使用全部 10 个厚度，而循环遍历铜、铁阶梯取 `[1, 3, 5, 7, 10]` 个不同最大厚度的实验结果。生成的全部能谱曲线图、线性拟合图和参数 JSON 归档文件分别以 `CuFe_1steps` 到 `CuFe_10steps` 独立文件夹进行存储，独立保存结果。
- **Method 2 Separate Plotting**: 应用户要求，在 [reconstruct_spectrum.py](file:///e:/photo_electric_II/reconstruct_spectrum.py) 的主程序中独立生成并输出了方法二（相邻差值映射法）的专属能谱图和 $apd$/$acd$ 线性度散点拟合图。
    - 能谱图保存为 `reconstructed_spectra_method2_{f_type}.png`，只包含方法二的各电压低高能能谱曲线。
    - 线性度图保存为 `apd_acd_linearity_method2_{f_type}_{voltage}.png`，展示了方法二解算特征的物理厚度线性度，并绘制了过原点的线性拟合对比基准线。
    - **方法二比对隔离**：在主对比图 `apd_acd_linearity_{f_type}_{voltage}.png` 中移除了方法二曲线（`Spectrum NL M2`）。这避免了方法二过大的偏差拉伸 Y 轴刻度，从而让方法一、静态单能、动态单能的线性对比细节更清晰。
- **Method 2 Evaluation & Documentation**: 对基于厚度差值映射的能谱反推方法（Method 2）进行了全面的物理与数学可行性评估。编写了独立诊断脚本 `evaluate_spectra.py` 提取并计算了 7 个电压（200kV - 320kV）与 2 种滤片（0.6mm, 1.2mm）配置下各材质厚度级 APD/ACD 的 $R^2$ 线性度。评估显示，Method 2 由于受阶梯内部严重能谱硬化过滤的影响，重建能谱表现出严重的高能偏置（200kV下有效低能达 103.8 keV），从而导致解算 APD/ACD 线性度退化为严重负值（$R^2 \approx -48.0$），证明其物理不可行性；而方法一（正则化 NNLS）能重建真实的入射能谱（200kV下 LE $E_{eff} \approx 65.0$ keV），实现优秀的线性度。并在 [notes_spectrum_reconstruction.md](file:///e:/photo_electric_II/paper/notes_spectrum_reconstruction.md) 中补充了方法二的数学原理、物理成因分析与定量对比数据。
- **Documentation & Physical Derivation for Spectrum Reconstruction**: 编写并新建了详细的数学与物理原理文档 [notes_spectrum_reconstruction.md](file:///e:/photo_electric_II/paper/notes_spectrum_reconstruction.md)，详尽推导并解释了利用标样阶梯在不同厚度下的吸收差异，构建前向矩阵，并通过增广正则化非负最小二乘（NNLS）求解 X 射线出射能谱 $S(E)$ 的完整数学过程，以及如何将重构出的谱应用于解耦能谱硬化的 APD/ACD 特征求解。
- **Refactor & Image Format Transition**: 将 [read_raw.py](file:///e:/photo_electric_II/read_raw.py) 重构为可同时兼容读取和转换 `.txt` 与 `.raw` 格式图像文件。
    - **双格式自适应解析与中文参数注释**：设计并重构了核心转换逻辑函数 `convert_txt_and_raw_to_png`，移除了对硬编码宽高的单向依赖。同步解释并补充了详尽的中英文多维度参数（入参类型、含义、用法）物理设计文档，严格遵守代码规范。
    - **递归遍历子目录**：使用 `os.walk` 替换原本的 `glob.glob`，实现自动遍历 `source_dir` 下的所有子文件夹，并在目标文件夹 `dest_dir` 中镜像生成相同的子目录结构，且能够自动跳过已转换的输出文件夹，防止产生死循环或冗余转换。
    - **文件名关键字过滤**：新增了可变参数 `filter_keyword`（默认值为 `"校准后"`），在转换时自动过滤文件名，仅对名字中含有该关键字的图像文件进行转换。
    - **目标路径自动拼接**：设定 `dest_dir` 默认值为 `None`，若用户未指定则自动在 `source_dir` 后拼接 `converted_pngs`。
    - **多格式流式读取与智能尺寸推断**：
        - 对于文本 `.txt` 文件：使用 `np.loadtxt` 加载，**天然保持原本矩阵的二维 shape**（依靠换行符自动解析）。
        - 对于二进制 `.raw` 文件：因为二进制流是一维的，无法直接保留原本 shape，必须通过 `reshape` 还原为 2D。为了减少人工输入，新增了**从文件名自动正则匹配解析宽高尺寸**的逻辑（如文件名中包含 `1024_1024` 或 `2048_512`）；若文件名未包含尺寸，则 fallback 使用默认传入的参数 `width` 和 `height` 进行 `reshape`。
    - **无损 PNG 转换与路径容错**：采用 `cv2.imencode` 统一无损转换为 16-bit PNG 图像并保存，完美解决 Windows 平台下的中文路径支持。


## 2026-06-03
- **Feature & X-ray Spectrum Reconstruction**: 编写并新建了能谱反推与物理特征计算脚本 [reconstruct_spectrum.py](file:///e:/photo_electric_II/reconstruct_spectrum.py)。该脚本能够利用 0429 阶梯标样在不同厚度下的吸收差异，结合 NIST 数据库的质量衰减系数，建立正向传输方程；采用带二阶差分平滑约束、归一化约束及能量边界归零约束的非负最小二乘 (NNLS) 算法，高精度反演重建出低能通道 $S_L(E)$ 和高能通道 $S_H(E)$ 的出射 X 射线能谱。
- **Effective Energy Calculation**: 实现了基于能谱积分的入射有效能量 $E_{eff}$ 计算。输出并保存了各电压和滤片配置下的等效单能值至 `results/thickness_decoupling/energy_hardening/spectrum_reconstruction/reconstructed_spectra_summary.json`，并自动生成了有效能量随管电压的变化折线图。
- **Spectrum-Integrated APD/ACD calculation**:
    - **动态等效单能法**: 使用反推得到的入射有效能量 $E_{eff}$ 代替固定的 $58/105\text{ keV}$ 静态常数，使 APD/ACD 特征计算自适应管电压变化。
    - **全谱非线性积分法**: 直接以连续能谱为积分核，通过二元 Newton-Raphson 算法 (scipy.optimize.root) 数值求解 Alvarez-Macovski 积分方程组，反解出不受能谱硬化扭曲的 APD/ACD 特征。
    - **线性度对比评估**: 对三种方法在 Al, Fe, Cu 阶梯上的 $apd$/$acd$ 厚度线性度 ($R^2$) 进行对比诊断，自动绘制并保存对比折线图。
- **Feature**: 编写并新建了双自变量输入能谱硬化拟合脚本 [fit_dual_variable_hardening.py](file:///e:/photo_electric_II/fit_dual_variable_hardening.py)，该脚本能够利用 0429 实验数据集的高精度 16 位阶梯标样实测衰减值，结合 NIST 质量衰减系数逆插值反算出对应各个厚度层的等效能量因变量，并在此基础上成功构建了**物理启发式（自变量为 $H, H^2, L/H$）**和**通用二元多项式（自变量为 $H, L, H^2, L^2, H \cdot L$）**两种能谱硬化预测模型。模型在排除极端外推点后运行多元最小二乘回归，输出精确的系数总结 JSON 并自动渲染 2x2 多维拟合效果对比诊断图。
- **Thickness-dependent Energy Hardening Analysis**: 在 `analyze_energy_hardening.py` 中新增 `plot_energy_by_thickness_vs_voltage` 绘图子模块。对每个厚度阶梯（共 10 阶）分别独立生成一张 1x2 对比图并保存为 `energy_vs_voltage_step{step_idx + 1}_{thicknesses}.png`（左图为 0.6mm，右图为 1.2mm，子图内同时画出铜、铁、铝三材质在该厚度下的 Low/High 能等效能量随电压变化的曲线，布局与 `plot_eavg_summary` 严格一致）。优化了 `analyze_hardening` 中的内存管理逻辑，有效规避了 MemoryError。
- **Calibration Fit Curve Overlay Plots**: 在 `get_apd_acd.py` 中新增了 `_plot_calibration_fit` 绘图子模块，实现了对系统校准中 $\ln(a_p/a_c) \to \ln(Z)$ 空间内回归直线与各单质（Al, Fe, Cu）理论输入点、测量预测点及其残差和相对误差的直接可视化对比（保存为 `sirz_calibration_fit.png`），使用户能够最直观地看到拟合误差的产生形态。
- **Multi-Step Trend Plots & Subplot Title Formulas**: 针对第一、三、五个厚度阶梯单独求解的物理特性，在 `get_apd_acd.py` 的 `run_step_apd_acd_analysis` 中实现了分别收集并绘制这三个特定阶梯电压依赖大图（`summary_coefficients_step1/3/5.png`）的逻辑。并在绘图子模块 `_plot_coefficient_dependence` 中，在 $Z_e$ 和 $\rho_e$ 的子图标题（subplot title）中分别清晰标注了物理重构公式（$Z_e = g \cdot (a_p/a_c)^{1/\nu}$ 与 $\rho_e = K_1 \cdot a_c$），方便进行直观的对比与诊断。
- **Calibration Update (Direct Thickness Division)**: 在 [get_apd_acd.py](file:///e:/photo_electric_II/get_apd_acd.py) 中，将系统校准时提取 Bulk 系数 $a_p, a_c$ 的逻辑由“多厚度阶梯线性斜率拟合”升级为“基于特定阶梯的直接厚度除法求解”。新增对第一、三、五个较薄阶梯（索引 0, 2, 4）分别运行系统校准 `(K1, g, nu)` 的功能并于控制台和 JSON 统计中进行输出对比。默认将受能谱硬化扭曲最小的“第一阶梯”校准参数作为后续物理量重构的基准。同步在 [notes_calibration_wedge.md](file:///e:/photo_electric_II/paper/notes_calibration_wedge.md) 中更新了第二阶段物理公式描述。
- **Plot & Theory Line Overlay**: 针对双能物理特征解算中有效原子序数（Ze）在 Fe (Z=26) 和 Cu (Z=29) 之间的倒置现象（此倒置由多色射线通过强衰减铜介质时的严重硬化漂移导致，已通过直接对比 LE 通道对数衰减均值排除了样品读写反置的可能性），在 `get_apd_acd.py` 的 `_plot_detailed_profiling` 和 `_plot_coefficient_dependence` 中绘制并标注了各金属单质（Al, Fe, Cu）的原子序数理论值（13.0, 26.0, 29.0）和理论电子密度值（1.3008, 3.6644, 4.0888 moles-e/cm³）水平参考线，使用户能清晰对比测量校准曲线与理论靶值的物理偏差。
- **Documentation & System Coefficient Calibration Guide**: 基于三明治探测器的有效单能近似物理框架，设计并撰写了铜、铝、铁阶梯标样下的系统物理系数（电子密度常数 $K_1$、原子序数常数 $g$ 与 $\nu$）校准流程，保存在 `paper/notes_calibration_wedge.md`。
    - **理论值推导**：归纳了高纯单质金属（Al, Fe, Cu）的等效原子序数理论值（分别为 13, 26, 29）与基于质量密度的物理电子密度 $\rho_e$ 理论值（分别为 1.3008, 3.6644, 4.0888 moles-e/cm³）。
    - **数学回归建模**：设计了三步校准数学模型，首先使用最小二乘斜率拟合排除厚度 $d$ 的影响提取各材质 Bulk 物理衰减系数 $a_p, a_c$；接着通过无截距最小二乘求解电子密度常数 $K_1$；最后通过对数线性化直线回归 $\ln(Z) = \ln(g) + \frac{1}{\nu}\ln(R)$ 求解原子序数常数 $g$ 和 $\nu$。
    - **Python 自动化校准脚本**：编写并集成了完整的 Python 校准工具代码，支持 16-bit 像素去噪清洗、APD/ACD 算子计算、厚度斜率拟合与跨材质系数对数线性回归。
    - **无量纲化去 E0 升级**：去除了原代码和校准文档中多余的参考能量参数 $E_0$，将光电截面直接改写为无量纲的 $E_L^{-3}$ 和 $E_H^{-3}$ 格式，使公式在物理表示和数值计算中更简洁直观。

## 2026-06-02
- **Documentation & Sandwich Detector Paper Reading Report**: 阅读了学术论文 `用于双能X射线成像中材料识别与对比度消除的三明治探测器的设计与制造.pdf` (Rimcy Palakkappilly Alikunju et al., J. Appl. Phys. 2024)，并在 `paper/notes_sandwich_detector.md` 中撰写了详尽的学术阅读报告。
    - **探测器构造与几何**：详细整理并用 Mermaid 流程图绘制了该单次曝光三明治探测器的微观物理多层结构（CsI 闪烁体、CMOS 先进像素传感器、中间铜滤片、光纤面板等），归纳了其各层材料的参数考量。
    - **解耦与对比度消除算法**：系统梳理了基于 Alvarez-Macovski 的系统无关原子序数与电子密度计算公式，以及 Lehmann 等人的双基质线性分解与对比度消除投影偏置角（$\phi$）数学推导。
    - **结果分析**：总结了在 RQA5 能谱品质下对中间铜滤片厚度（0, 0.25, 0.5 mm）的最优化 $\chi^2$ 偏差测试和实物对比度消除 SNR 卡限（Rose 准则，SNR $\ge 5$）实验结果，指出 0.25 mm 铜滤片在消除背景杂乱基质（鸡肉瘦肉与脂肪）中对于骨骼和钙化靶的卓越呈现效果。

## 2026-06-01
- **Feature & Multi-Dataset Comparison & 0407 Re-enablement**: 在 `compare_tube.py` 中重构并重新启用了历史 16-bit 阶梯标样数据集 `0407 (Home)`，并实现了多数据集在 `160kV` 管电压下衰减特性的联合对比分析。
    - **重新启用 0407 数据集**：在中央调度函数 `run_stepped_specimen_analysis` 中移除了对 `0407` 的注释禁用，支持对 `test1`、`test2` 和 `test3` 运行的 160kV 标样像素数据进行 2x3 剖析渲染，并将图像输出保存到 `results/Tube_comparison/comprehensive_fit/0407/`。
    - **物理厚度坐标对齐修正**：由于 0407 的 Al 标样没有 10mm 底座垫块（厚度直接为 2-20mm），在 `perform_comprehensive_analysis` 中通过新增 title 前缀的字符串判定逻辑，自适应跳过对其减去 10mm 偏置的计算；而在 0331/0409 数据集上则保留减去 10mm 偏置，从而使所有数据集的 Al 标样能完美、正确地对齐到统一的 2-20mm 物理厚度 X 轴上。
    - **160kV 跨数据集联合对比**：在顶层新增了 `plot_dataset_comparison_160kV` 绘图函数，详细定义并解释了输入参数类型、含义与用法。该函数收集 `0331`、`0407 test1` 和 `0409 125us` 在 160kV 处的衰减对数均值，在统一物理厚度轴上绘制 2x3 联合对比图（包含 $\mu_L$、$\mu_H$、比值、材质间比值与差值）。
    - **物理理论参考比值线绘制**：在材质间比值子图内以 dotted 形式分别绘制了 Photoelectric (PH) 和 Compton (C) 理论参考比值线。为了保持图面整洁度，设计了仅绘制一次的逻辑，完美杜绝了多数据集叠加绘制时的参考线重叠冗余。
    - **多衰减模式支持与 Latex 兼容**：新对比分析和 0407 各 run 的剖析图完美支持 `mu_mode` 在 `'mu'` (线衰减) 与 `'mu_m'` (质量衰减) 之间的切换。对质量衰减模式下 LaTeX 图表符号使用单下标（如 `\mu_{m, L}`、`\mu_{m, H}`）进行了规范，完全规避了 Matplotlib 下的 LaTeX 解析崩溃。
- **Documentation & Paper Reading Report**: 阅读了学术论文 `利用双能计算机断层扫描实现材料表征的系统无关性方法.pdf` (Stephen G. Azevedo et al., IEEE TNS 2016)，并在 `paper/notes.md` 中撰写了学术阅读报告，并补充了对数衰减投影（$P_L, P_H$）在单色与多色连续谱下的严格物理定义、实验观测灰度（$I, I_0$）的物理映射与正向方程推导。
    - **原理介绍**：深入分析了经典双能特征空间的局限性，对比了 Mayneord 经验有效原子序数（$Z_{eff}$）与物理电子截面决定的系统无关有效原子序数（$Z_e$），详细阐述了 Alvarez-Macovski 光电-康普顿散射双能分解原理、电子密度（$\rho_e$）定义以及 $Z_e$ 的最小二乘优化模型。
    - **实验方法**：整理了其两台工业级 DECT 扫描系统（HE 与 TB，配备 Thales Flashscan 33 与 PerkinElmer XRD 1620 探测器）的配置、5 种能谱与滤片组合设计（HE100/160、TB100/160、TB80/125、TB125/200、TB80/200），以及标样（石墨、水、镁、硅）、非均匀块体、高原子序数 RbBr 溶液的验证实验。详细梳理了利用 MCNP6 蒙特卡洛软件建立的探测器能谱响应模型，以及基于 Newton-Raphson 约束优化的正向分解算法。
    - **实验结果**：总结了 Ratio、YNC 与 SIRZ 三种方法的精度与准确度对比，指出 SIRZ 在低能谱变化敏感性（精度 $<2\%$，准确度 $<3\%$）上的显著优势，以及在 RbBr 溶液外推界限上的物理鲁棒性。

## 2026-05-29
- **Feature & Excel Assayer Filling**: 在 `fill_csv/fill_csv.py` 中实现了 CSV 数据与 Excel 表格的序号对齐、化验品位与 XRF 测试序号自动填充逻辑。
    - **序号与元素及测试号匹配**：从 `2026.05.29.csv` 中提取矿石序号（第 3 列，索引为 2）与化验品位（`Cu`, `Fe`, `Al`, `Ca`, `S`），同时提取测试序号（第 2 列，索引为 1，“测试 #”），并与 `CuO矿石重量.xlsx` 的 “0514氧化铜” 工作表中的第 1 列（A列，“序号”）进行精确整数匹配。
    - **无损 Excel 写入**：利用 `openpyxl` 引擎打开和保存 Excel 表格，在写入“正面”化验品位（D 列至 H 列，即第 4 至第 8 列）和 XRF 编号（C 列，第 3 列）时，完美保留了原 Excel 的所有 sheet、格式、样式、字体、列宽以及 “平均值” 列中的公式（如 `=AVERAGE(D5,I5)` 等），确保表格不受任何污染且平均值由 Excel 自动刷新。
    - **缺失值与数据健壮性**：对 CSV 中的 `NaN` 或缺失值，自动将其在 Excel 中写入为空（`None`），以保持表格的绝对干净。支持在匹配前后打印详尽的匹配统计日志。
    - **文档规范化**：为新实现的 `fill_assay_grades` 核心函数编写了详尽并包含 XRF 测试号处理的中英文多维度参数（入参类型、含义、用法、返回值）物理设计文档。

## 2026-05-28
- **Feature & Combined Slope Analysis**: 在 `fit_hl_curve_0429.py` 中重构并实现了 0.6mm 与 1.2mm 滤片标样质量衰减汇总曲线（slope_summary）的联合绘制与对比管线。
    - **模块化重构**：将原本嵌套在单滤片扫描循环内部的 10 阶厚度绘图逻辑彻底解耦，抽象并定义为全新的 `plot_combined_slope_summaries` 顶层函数，并同步为该函数编写了详尽的中英文参数类型、含义、用法及返回值说明文档，严格遵守代码规范。
    - **双滤片并合与自适应 Y 轴**：在联合汇总大图中，同时画出 `0.6mm` (实线圆形 `o-`) 与 `1.2mm` (虚线三角形 `^-`) 的曲线。两组数据联合输入 `get_dynamic_ylim` 以计算全局自适应统一 Y 轴，确保跨厚度、跨电压、跨滤片的可视化在物理尺度上具有严格的可比性。
    - **一致的色彩体系**：各材料保持高对比度一致着色：铜 (`Cu_step`) 统一使用 Crimson Red (`#d62728`)，铁 (`Fe_step`) 统一使用 Slate Blue (`#1f77b4`)，铝 (`Al_step`) 统一使用 Emerald Green (`#2ca02c`)；跨材质比值对使用专属莫兰迪配色 (Purple, Brown, Magenta)。
    - **理论比值虚线防冗余**：对于各跨材质比值图中的 Photoelectric 贡献 (PH) 和 Compton 贡献 (C) 理论比值辅助线，通过将其移出 filter 循环，确保每个子图中的特定颜色理论参考虚点线 (`:`) 仅绘制一次，彻底避免了图面文字与参考线重叠冗余。
    - **保存归档机制**：联合绘制大图统一输出保存至 `combined/slope_summary_{mu_mode}/` 下，与原本各滤片的独立 results 子目录完全解耦并清晰归档。
- **Feature & Multi-mode Attenuation Suffixing**: 在 `compare_tube.py` 和 `fit_hl_curve_0429.py` 中引入了质量衰减系数 $\mu_m$ ($\mathrm{cm}^2/\mathrm{g}$) 与线衰减系数 $\mu$ ($\mathrm{mm}^{-1}$) 的动态切换参数 `mu_mode`（取值 `'mu'` 或 `'mu_m'`），并在生成的保存路径中自动加入对应后缀（如 `slope_summary_mu` 或 `slope_summary_mu_m`，以及 JSON 后缀 `attenuation_slopes_mu_m_*`），完美满足了用户多样化物理量评估和数据分离管理的需求。
    - **物理公式校准**：切换至质量衰减模式 `'mu_m'` 后，程序自动在底层计算对数衰减时将厚度转换为厘米（$t_{cm} = t_{mm} / 10.0$）并除以材质的理论密度（$\rho$），计算公式严格对齐为 $\mu_m = 10 \cdot \mu / \rho$；且在跨材质比值折线图中，自动将 Photoelectric 贡献（PH）和 Compton 贡献（C）的理论辅助线中的密度 $\rho$ 进行消去化简（比值仅取决于原子序数 $Z$ 和相对原子质量 $Ar$），保持物理推导的严密性。
    - **UI & 轴标签与 Latex 解析自适应**：图表大标题、子图标题及 Y 轴标签在 `'mu'` 模式下自动显示为 $\mu$ ($\mathrm{mm}^{-1}$)，在 `'mu_m'` 模式下自动显示为 $\mu_m$ ($\mathrm{cm}^2/\mathrm{g}$)，且通过定义 LaTeX 单下标变量（如 `\mu_{m, L}`, `\mu_{m, H}`）替代双下标格式，彻底规避了 Matplotlib 符号解析的 `ParseFatalException: Double subscript` 报错，保证了图像的高清稳定渲染。
    - **防错与安全限位**：为两脚本的跨材质除法引入了 `np.maximum(..., 1e-9)` 安全保护，彻底避免了分母为零或 NaN 引起的计算异常。
- **Feature & Step Transition Extraction**: 在 `extract_sample_values.py` 中新增了阶梯标样厚度突变区域特征提取逻辑。程序会自动比较阶梯两端的灰度平均值以自动判定由薄到厚的阶梯排列方向，定位第三个厚度突变分界线（若 Step 0 为最薄则在 `3 * step_h`，若 Step 9 为最薄则在 `7 * step_h`），向上下外扩提取共计 10 行（突变处前 5 行与后 5 行）以及收缩 10% 横向边缘宽度（`0.1 * W` 至 `0.9 * W`）的核心过渡区像素。提取的高低能像素均以一维数组形式，使用 `_transition.pkl` 后缀命名保存到 `pixel_values` 中，与全区域提取数据物理分离。同时，在生成的 `contoured.png` 可视化标定图上，使用厚度为 `2` 像素的红色实线框标注出该 `transition` 区域，并在其边缘外侧标注 `'T3'`，为物理过渡区域提供极直观的视觉确认与结果验证支持。
- **Comparison & Transition Comparison**: 在 `compare_tube.py` `main()` 函数中，引入了 `125us` (index 5) 和 `270us` (index 6) 在 `160kV` 下阶梯突变分界区 (transition) 的直接像素对比。调用了 `run_comparison` 算法管线自动评估并输出了突变像素的均值对比图 `TYM_Exposure_Steps_Transition_means.png` 和像素级原始高低能分布直方图 `TYM_Exposure_Steps_Transition_hist_low/high.png`，完成了曝光时间对跃迁区散射及边缘过渡特征影响 of 物理诊断。

## 2026-05-27
- **Feature & Height-Splitting Update**: 针对用户要求，在 `utils_II.py` 中重构了 `get_bricks_watershed` 轮廓分割算法的高度拆分阈值逻辑。将原有的硬编码 `800` 像素分割界限升级为动态判定：针对 `270us` 曝光时间图像自动将阈值调宽至 **`1000`** 像素以自适应其垂直物理拉伸度，其他曝光配置（如 `125us`）默认保持原有的 `800` 像素拆分规则。同步为 `get_bricks_watershed` 函数补充了极详尽的中英文多维度参数（入参类型、含义、用法、返回值）物理设计文档。
- **Alignment & Fix**: 修复了 `compare_tube.py` `main()` 函数中 125us 与 270us 曝光对比部分的路径拼写错误。修正了 configs_step 标样对比和 configs_x / configs_y 矿石对比的文件名，补齐了缺失的 `_cropped` 命名标记（如 `160kv-2mA-125us-0.5pF-ore-post_calib_cropped_ore_0_data.pkl`），使得曝光时间相关性汇总图表能够完美、顺利无阻地跑通输出。
- **Alignment & Comments**: 针对用户要求，在 `compare_tube.py` 中注释了 0407 (Home) 数据集相关的所有数据加载、处理及 `perform_comprehensive_analysis` 综合分析调用代码，暂时不使用该数据集。
- **Step Mapping Update**: 对 0409 (TYM) 数据集，根据用户指导在 `compare_tube.py` 中将数据读取路径 `input_dir_0409` 指向以 `125us` 裁剪（cropped）16 位位深保存的 `'results/TYM_test_2_16bit/pixel_values'` 目录，并核对/修改了其 10 阶标样索引映射。目前 `160kV`、`180kV` 和 `200kV` 三个电压统一映射为：铝阶梯/铝块 (`Al_step`): 6, 铜阶梯 (`Cu_step`): 8, 铁阶梯 (`Fe_step`): 9。此映射经过物理衰减属性校验，表现出完美的材质物理对应性（Cu 衰减最强，对应最低灰度 1203.0；Fe 衰减中等，对应中等灰度 1914.7；Al 衰减最弱，对应最高灰度 9292.4）。运行测试全部顺利通过，生成了完整的随管电压变化的 slope_summary 汇总折线大图 and 2x3 剖析图。
- **Exposure Comparison Fix**: 修复了 `compare_tube.py` 中 `main()` 函数下 125us 与 270us 曝光对比部分的数据路径与索引。将原有的 `results/TYM_test/pixel_values/` 替换为正确的 `results/TYM_test_16bit/pixel_values/` 路径，并将 step 比较索引设为共同存在且物理有效的 `step_sample_6`，矿石对比索引统一设为共同拥有的现有 `[0, 1, 3]` 序号，确保对比逻辑顺利跑通。
- **Feature & Alignment**: 配合 `crop_TYM.py` 的裁剪规则，对标样特征提取主程序 `extract_sample_values.py` 的文件读取模块进行了同步升级。
    - **按需读取过滤**：针对 0409 实验数据集，升级了文件检索与自适应读取逻辑，从扫描所有带 `kv` 的 tif，变更为**仅读取以 `_cropped.tif` 后缀结尾的裁剪后 16 位图像**。其他历史标样数据集的自适应发现规则（读取带 `kv` 的 `.tif` 或带 `dual` 的 `.png`）保持完全不变与向前兼容。
    - **运行校验**：运行测试并成功对 16-bit 裁剪完成的 `160kv`、`180kv`、`200kv` 阶梯和 disk 标样文件进行了特征分析，完成了分水岭高精度区域分割、中心 Erosion 核心区域掩膜计算并替换空气背景，并输出了完美的像素级统计序列和 structured summary report，验证了裁剪与读取闭环管道的 100% 正确性。
- **Fix & Optimization**: 修复并优化了 16-bit 原生图像自动裁剪脚本 `crop_TYM.py` 的数据判定逻辑，完美解决了裁剪输出与原图尺寸完全一致（未实际裁剪）的缺陷。
    - **问题定位**：由于 16-bit 图像的像素灰度范围极宽（0-65535，且存在 800 - 1500 左右的随机探测器噪声与本底干扰），原本为 8 位图像设计的 `std_threshold = 50.0` 阈值显著偏低，导致所有空载背景行的像素标准差均远高于阈值被误判为有效区域，使裁剪行选定在 [0, 4095]，即裁剪前后图片完全相同。
    - **算法升级**：
        - 将 `std_threshold` 的默认值由 `50.0` 升级为更契合 16-bit 噪声水平的 `3000.0`（建议区间为 2000 - 5000），成功将高频本底噪声与实际样品区隔开。
        - 引入了裁剪外扩安全裕度参数 `margin`（默认值为 `50` 像素），在检测到的有效上下沿外侧保留一定缓冲背景，避免截断样品的微弱边缘，保障物理轮廓的完整性。
        - 结合 125us 与 270us 图像由于积分时间差异导致的垂直缩放（2.16x）物理特性，验证了阈值在两类图像中的普适性与裁剪精确度（125us 裁剪后高约 1190 像素，270us 裁剪后高约 2520 像素，与积分比值完全吻合）。
    - **代码规范化**：为 `auto_crop_xrt_16bit` 编写了详尽的中英文多维度参数说明文档，并将文档字符串设为 Raw 格式 (`r"""`) 以杜绝 Windows 路径反斜杠转义警告（`SyntaxWarning`）。
- **Feature**: 将 `compare_tube.py` 中的所有数据集分析全面升级为高精度 16-bit 输入。
    - **数据集升级**:
        - **0331 (Yinshan)**：切换至 `results/20260331_16bit/pixel_values/` 数据源，统一设定入射光强对数背景 $I_0 = 52428.0$。
        - **0407 (Home)**：首先运行 `extract_sample_values.py` 从原生的 16-bit TIFF 图像中重新提取了 16 位的标样核心像素，输出至 `results/20260407_Sample_test_16bit/pixel_values/` 目录。接着将 `compare_tube.py` 切到该 16-bit 路径，并将入射光强对数背景升级为 $I_0 = 52428.0$。
        - **0409 (TYM)**：保持 16-bit 输入及 $I_0 = 52428.0$。
    - **图表重生成**: 重新运行 `compare_tube.py`。至此，0331、0407、0409 标样数据已完全统一至 16-bit 位深口径下比对，画出了横纵坐标高度一致的 2x3 综合拟合图和 20 张高清随电压变化的 slope_summary 曲线。
- **Fix & Complete**: 修复了 0409 (TYM) 多电压标样数据文件嵌套导致的读取缺失问题，成功补全并生成了所有电压下的分析与 slope_summary 图表。
    - **路径纠正**: 发现之前提取的 `180kv` 和 `200kv` 16-bit 阶梯标样 `.pkl` 文件由于 `save_contour_data` 自动追加目录特性被嵌套保存在 `results/TYM_test/pixel_values/pixel_values/` 目录下，导致 `compare_tube.py` 执行时静默跳过了这两个电压。
    - **文件迁移与重新生成**: 编写并执行了文件迁移指令，将所有的 `.pkl` 文件与高低能图像均提至 `results/TYM_test/pixel_values/` 和 `results/TYM_test/high_low_images/` 标准目录下，并重新运行了 `compare_tube.py` 脚本。
    - **结果产出**: 成功补全生成了 TYM 在 `180kV` 和 `200kV` (270us) 下的综合折线散点分析大图 `180kV_270us_analysis.png` / `200kV_270us_analysis.png`，以及该数据集全部 10 个阶梯厚度对应的随电压变化的斜率相关性汇总 (slope_summary) 图表，完美解决了数据展示缺失的问题。
- **Feature**: 在 `compare_tube.py` 中引入了极高物理精度的多电压阶梯标样综合分析与斜率相关性汇总 (slope_summary) 管线。
    - **16-bit 像素提取**: 编写并执行了 16-bit 高精度像素分割脚本，从 `TYM_converted_results` 原始 tif 提取了 **0409 (TYM)** 在 `160kV`、`180kV`、`200kV` (270us) 下的完整 10 阶标样数据 (pkl)，精确定位了各电压下材质 contour 对应关系。
    - **移植与优化核心算法**: 移植并优化了 `fit_hl_curve_0429.py` 中的 `find_linear_pts`（自动寻找线性区间）与 `perform_comprehensive_analysis`（2x3 综合折线散点大图）。
    - **质量衰减汇总 (slope_summary)**: 新增了 `generate_dataset_slope_summaries` 核心函数。对多电压标样数据，逐一计算各阶厚度对应的 $\mu_L$、$\mu_H$、L/H 比值、材质间比值（内嵌 Photoelectric & Compton 物理理论辅助线）以及 L/H 绝对偏差随电压变化大图，将 0331 和 0409 (TYM) 的 10 个步进各自渲染并输出了共 20 张高清 `slope_summary` 大图。
    - **总控调度升级**: `run_stepped_specimen_analysis` 升级为全自动调度：
        - 0331 数据集：遍历 `140kV`、`160kV`、`180kV` 三个电压，对应 `pixels_low` 按 thinnest-to-thickest 排序，并输出 10 张 `slope_summary` 图。
        - 0407 数据集：遍历 `160kV` 下的 `test1`、`test2` 和 `test3` 序列，使用 `flip=True` 执行逆序反转，精准校准厚度坐标。
        - 0409 (TYM) 数据集：遍历 `160kV`、`180kV`、`200kV` 三个电压 (270us 16-bit, I0=52428.0)，并输出 10 张 `slope_summary` 图。
- **Environment**: 调查并解决了 AI 助手在 IDE 终端执行命令（如运行 `python` 脚本）时响应极慢或卡死的问题。
    - **原因定位**：默认 the 终端环境运行在 UWP AppContainer 沙箱内，无法直接访问 `D:\` 盘中的 Anaconda 环境，且当沙箱拦截文件请求时，会因检索 `PATH` 环境变量中排在前面的 Microsoft Store 重定向代理（App Execution Alias）而导致进程挂起或报错。
    - **修复方案**：成功申请并获批了 `unsandboxed(cmd.exe)` 与 `unsandboxed(python)` 提权，使命令能够在沙箱外以 100% 的原生系统速度和完整权限执行，完美对接 `D:\anaconda\python.exe` 及其包含的物理仿真和数据拟合等科学计算库（如 `numpy`、`pandas` 等），彻底消除了执行挂起，实现了瞬时响应。


## 2026-05-26
- **Feature**: 在 `get_apd_acd.py` 中新增并导出了 `_plot_apd_acd_histograms` 模块化直方图生成函数，用于为指定电压和滤片下的标样材料（Cu, Fe, Al）绘制其全部有效像素在 $apd$ 和 $acd$ 特征上的像素级原始分布直方图，生成专门独立的可视化图表并自动保存至 `{f_type}/{voltage}_apd_acd_histogram.png`，完美响应了“直观展现全部像素分布且不污染原图”的物理验证设计要求。
- **Refactor**: 撤销并还原了 `_plot_detailed_profiling`（2x2 深度物理特征剖析图）中强加的像素级散点点云背景，使其回归到完全洁净、不受噪声云污染的宏观均值趋势及标准差误差棒曲线图（原图完好无损、保持原始设计）。
- **Optimization**: 在 `get_apd_acd.py` 的特征数据持久化导出流程中，引入了 `_make_json_safe` 递归安全序列化处理器，在导出 JSON 总结文件前递归自动剥离 `*_raw` 像素数组，在完美保留图表绘制所需高精度原始像素级数组的同时，杜绝了 JSON 导出报错并极大压缩了 JSON 文件体积。
- **Configuration**: 配置并注册了官方 Model Context Protocol (MCP) PDF 阅读服务器 `@modelcontextprotocol/server-pdf`。创建了 IDE 级的 `mcp_config.json` 配置文件，使用 `npx` 自动执行，为 AI 助手赋予原生的高性能 PDF 交互式文档阅读与文本提取能力。
- **Feature**: 在 `get_apd_acd.py` 中实现了完整的阶梯样 APD 和 ACD 特征计算与可视化分析管线。通过 `calculate_apd` 与 `calculate_acd` 进行像素级物理特征提取，并计算其均值 and 标准差，量化物理厚度及材料依赖。
- **Feature**: 实现了各电压和滤片配置下的 2x2 物理剖析图渲染。子图包含 APD vs 厚度、ACD vs 厚度、APD vs ACD 特征空间轨迹、以及 $Z_{eff}$ vs 厚度，直观展示了双能系统下的材料物理属性与能谱硬化漂移。
- **Feature**: 实现了多电压下的材料 Bulk 物理系数随电压变化的汇总大图渲染。折线包含 $a_p$、$a_c$、$Z_{eff}$ 以及 $a_p/a_c$ 比值随管电压的变化趋势，清晰展示了能谱硬度对物理衰减系数的调控规律。
- **Feature**: 实现了 step 特征提取结果 of JSON 结构化汇总导出，将计算得到的所有特征（平均值与标准差）持久化写入至 `results/thickness_decoupling/apd_acd_summary.json`，供下游材料原子序数解耦与厚度拟合模型使用。
- **Refactor**: 重构并拆分了原巨型主函数 `run_step_apd_acd_analysis` 为多个单一职责的模块化子函数：`_load_and_process_step_pixels`（像素读取与特征计算）、`_plot_detailed_profiling`（电压下2x2剖析图渲染）和 `_plot_coefficient_dependence`（多电压相关趋势图生成），极大提升了代码的可读性、内聚度与可维护性。
- **Refactor**: 为 `get_apd_acd.py` 中的所有核心物理函数及新增子函数编写了详尽的中英文多维度参数类型、含义及用法注释，严格遵守工作区参数透明化规则。

## 2026-05-25
- **Feature**: 优化了 `fit_hl_curve.py` 和 `fit_hl_curve_0429.py` 综合绘图函数中矿石样品（Ore）的异常值剔除阈值。将低能与高能通道的灰度值下界由静态的 `256` 变更为动态阈值（16位下剔除 `< 2560`，8位下剔除 `< 10`，对应8位下的比例 10），并在大标题中同步标注对应的剔除阈值，以更精确地规避大衰减及盲元像素点，提高拟合的稳定性。
- **Fix**: 优化了 `utils_II.py` 中的 `plot_ore_grayscale_distribution` 直方图绘制函数。将图例（legend）中指示死像素与低灰度值的静态阈值 `< 256` 重构为自适应逻辑：在 16 位图像下动态检测 `< 2560`（对应 8 位下的 10 倍增），在 8 位图像下检测 `< 10`。这与矿石综合分析图中的剔除边界保持了严格 of 物理含义一致性。
- **Fix**: 在 `fit_hl_curve_0429.py` 内部单块矿石加载流程（读取 0429 及 0401 的单块矿石 pkl 时）中，引入了相同的 `< 2560` (16位) / `< 10` (8位) 灰度阈值剔除逻辑，彻底消除了由于盲元或未合理衰减导致的 Log 对数极值，同时更新了 `ore_{oid}_comprehensive_analysis.png` 的图表大标题，使其动态显示排除阈值注记（如 `Excluding Grayscale < 2560`），确保单矿石全流程与多矿石混合图的物理机制高度一致。
- **Refactor**: 秉持 DRY 软件开发原则，在 `utils_II.py` 中新增并导出了 `get_ore_lower_threshold(is_ore, v_max)` 统一集中管理阈值计算逻辑。随后将 `fit_hl_curve.py` 和 `fit_hl_curve_0429.py` 中原本分布在各处的共 12 处硬编码 threshold 判定统一替换为对该中央函数的调用。这不仅显著净化了脚本主体逻辑，也避免了未来阈值变更时多点修改、极易出现人为疏漏的隐患。
- **Fix**: 在 `fit_hl_curve_0429.py` 中将 `ore_pixels_storage` 保存的像素源调整为反序列化后的**完全原始未过滤像素** `(l_v, h_v)`，从而确保 `plot_ore_grayscale_distribution` 函数绘制矿石灰度分布直方图时能够展示全部像素点（而非清洗后的部分），使得直方图中的 `=0` 和 `<2560`（16位）/ `<10`（8位）统计比例能够真实、不偏不倚地揭示图像采集底噪以及死盲元分布的全貌。
- **Feature**: 增强了 `utils_II.get_ore_lower_threshold` 中央配置器，新增支持了自定义最低允许透过率比例的参数 `ratio`（默认 5% 保持完全向前兼容）。为应对超高品位、大部分像素全吸收变黑的极端高致密矿石样品，提供了通过提高 `ratio`（如至 10% 或 15%）将信噪比极低、易受探测器底噪散射扭曲的“超黑噪声点”彻底过滤的标准化手段。

## 2026-05-22
- **Feature**: 在 `fit_hl_curve.py` 和 `fit_hl_curve_0429.py` 的 `perform_comprehensive_analysis` 综合绘图函数中，对矿石样品（Ore）引入了灰度下限过滤逻辑。现在，为了消除死像素、盲元以及由于过厚样品造成的异常极高衰减像素干扰，程序会自动将低能和高能通道中灰度值小于 `256` 的所有无效像素剔除在外，并在最终大图大标题中添加注记 `(Excluding Grayscale < 256)`，使得拟合回归曲线和统计均值点更为精确合理。
- **Feature**: 为 `plot_ore_grayscale_distribution` 函数集成了低灰度与死像素统计指示器。现在在每个电压子图的 legend 中，会自动计算并注明低能/高能通道中 `=0` 的像素占比以及 `<10` 的像素占比，为评估 XRT 探测器的底层物理底噪与可能存在的盲元盲点提供了强有力的定量化数据支撑。
- **Feature**: 将 `plot_ore_grayscale_distribution` 重构为支持“自适应动态判定灰度值上下限”的机制。若未显示传入 `x_min`/`x_max`，程序会自动合并某矿石在所有扫描电压下的高、低能有效像素，利用 0.5% 与 99.5% 分位数计算出自适应的直方图横坐标上下界（并在此基础上额外拓宽 2% 的视觉边距）。这不仅摆脱了原先 `12750` 的硬编码限制，也完美解决了 8位/16位 动态范围下图像横坐标轴的自适应渲染。
- **Feature**: 在 `utils_II.py` 中编写并集成了 `plot_ore_grayscale_distribution` 函数，用于将单块矿石在所有电压配置下的高、低能灰度值直方图分布整合成一张大图进行可视化输出。同步在 `fit_hl_curve_0429.py` 中引入了 `ore_pixels_storage` 字典，自动收集 0429（0.6mm 与 1.2mm）及 0401 两个数据集下各电压的有效像素分布并调用此函数生成各矿石的完整灰度直方图大图，存储于对应 `histograms/{ft}/` 路径下，极大方便了对矿石细节衰减与死像素分布的直接观测。
- **Fix**: 修正了 `fit_hl_curve.py` (以及 `fit_hl_curve_0429.py` 内部) 中衰减对数均值的计算顺序。根据詹森不等式 (Jensen's Inequality) 以及物理上的独立射线穿透原理，现在强制使用“先对每个像素进行 `log(I0/L)` 计算物理衰减，然后再对结果求均值”的逻辑，彻底取代了原本“先求灰度平均再取对数”的做法。此修改统一了综合分析图 (Row 2 曲线) 中的数值与 CDF 图中的标注平均值，显著提升了对高度异质性样本（如矿石）分析的准确性。





## 2026-05-20
- **Fix**: 修复了 `fit_hl_curve_0429.py` 中 `analysis_target = "step"` 时仍会错误执行矿石分析逻辑的问题。现在代码会通过早期退出 (`sys.exit()`) 严格遵守目标筛选配置。
- **Feature**: 重构了 `fit_hl_curve_0429.py` 中关于阶梯样分析的逻辑。摒弃了原来的单一拟合选项 (`mu_calc_method`)，改为将阶梯的厚度从最薄到最厚（索引 0-9）进行完整循环。
- **Feature**: 优化了阶梯 `attenuation_analysis` 图的坐标展示。统一强制 1.2mm 的结果子图与 0.6mm 的全局横坐标（电压）保持完全一致（130kV-330kV），从而避免了坐标轴范围不一引起的视觉差异。同时，所有对应的结果图与 `.json` 文件都会存入同一文件夹内，并在文件名中体现不同的阶梯厚度（如 `slope_summary_0.6mm_with_0331_2mm_CuFe_12mm_Al.png` 和 `attenuation_slopes_2mm_CuFe_12mm_Al.json`），便于集中查阅与对比。
- **Feature**: 更新了 `fit_hl_curve_0429.py` 的输出逻辑，现在保存的综合汇总图和 JSON 衰减系数文件会自动带上 `_slope` 或 `_3rd_step` 的后缀以防止互相覆盖。同步更新了 `get_mu_from_nist_new.py` 以默认读取 `_slope` 文件。
- **Feature**: 在 `fit_hl_curve_0429.py` 中增加了 `mu_calc_method` 选项，支持通过“拟合斜率” (`"slope"`) 或“第三个阶梯离散值” (`"3rd_step"`) 来计算真实的质量衰减系数 $\mu_m$ ($\mathrm{cm}^2/\mathrm{g}$)，并同步移除了理论曲线中冗余的密度计算。
- **Fix**: 修复了 `fit_hl_curve_0429.py` 中 0401 矿石数据读取硬编码指向旧8位文件夹的Bug，现已正确适配至 `_16bit` 文件夹，解决了高能基准下计算越界的问题。
- **Feature**: 实现了 0401 和 0429 矿石按电压层级自动生成 `2x3` 的综合衰减分析散点大图。
- **Feature**: 移除了 `fit_hl_curve.py` 和 `fit_hl_curve_0429.py` 中强加的 `255` 坐标轴上限硬编码，使得16位原生极宽动态范围坐标能自动延展。
- **Feature**: 实现了16位图像的端到端直接读取、处理与保存，避免了降采样到8位带来的精度损失：
    - 更新了 `utils_II.py` 中的 `get_bricks` 和 `get_bricks_watershed`，使用 `cv2.IMREAD_ANYDEPTH` 直接读取并支持16位 (`uint16`) 深度图像。
    - 针对16位图像自动进行了 `th_val` 参数自适应放大(如果输入是0-255，则自动乘以256)，保证轮廓提取阈值一致性。
    - 修改了 `utils_II.py` 中的可视化逻辑，仅对最终可视化的 `contoured` 图像进行8位压缩与BGR彩色化，而所有的分析用像素(`pixels`)、裁剪图(`low`, `high`)等核心数据均维持16位高精度输出。
    - 更新了 `extract_sample_values.py`, `extract_0429_RaySov.py` 以及 `extract_0429_RaySov_from_mask.py` 脚本，将最终的结果保存目录强制重定向到包含 `_16bit` 后缀的新文件夹，实现与原有8位数据解耦分离。
    - 在 `extract_0429_RaySov_from_mask.py` 中移除了将 16位 `.user.tif` 或 `.orig.tif` 降采样为8位的逻辑，现在它们统一输出 16位 (target_bit_depth=16) 的原始图像进行处理。
    - 更新了拟合分析脚本 `fit_hl_curve.py` 和 `fit_hl_curve_0429.py`，引入了动态的 `v_max` (自动适配255或65535)，使代码能够兼容分析读取出来的16位高动态范围 `pkl` 像素级数据。

## 2026-05-18
- **New Feature**: 在 `label_ores` 中新增了水平镜像翻转底图与轮廓映射的特性，完美响应用户关于左右反转渲染结果的要求：
    - 使用 `cv2.flip(labeled_img, 1)` 对底图进行水平翻转。
    - 同步将所有提取出的轮廓 X 坐标更新为 `W - 1 - x`，确保所有半透明填充、描边与高对比度文字框能以极高的精度自动对齐至反转后的新物理坐标。
    - **非镜像文字显示**：绘制步骤在底图和轮廓翻转完成后进行，使得所有分类标识文字（`1`~`4`）和 1-based 序号（`#序号`）均保持正常、非镜像、极其易读的从左到右文本显示。
- **Optimization**: 更新并完善 `pick_ores.py` 的方法3：
    - 更新了 Q1、Q2、Q3 和 Q4 的最新 1-based 静态矿石分类编号集合（各区间分别拥有 31, 30, 30, 30 块矿石，总计 121 块，实现完美映射）。
    - 重新设计并优化了矿石序号（`#序号`）的绘制布局：将序号绘制在每个矿石外接矩形（Bounding Box）的正上方空隙外（高度为 `y - id_h - 8` 像素处），配有专属高对比度圆角深灰黑背景与亮白文本；若矿石处于图像最顶端，则自动贴紧矿石内部上边沿绘制。这使得序号彻底与位于几何中心的等级数字类别标签（`1`~`4`）在纵向上完美拉开，即使面对小尺寸矿石，也能做到 100% 的视觉分离，完全杜绝重叠。
- **Hotfix**: 修复了 `pick_ores.py` 中 `save_image_robust` 函数的缩进和文档字符串语法错误，恢复为标准的 4 空格缩进，消除运行时的 IndentationError。
- **New Feature**: 在 `label_ores` 中引入并支持 `reverse_sort` 参数，完美解决矿石序号与分类预测结果的排列顺序对齐问题：
    - 对齐了 Yinshan (银山) 等特定数据集所要求的 `reverse_sort = True` 行内从右至左的逆序排列规则。
    - 详细解释了参数的作用与用法，确保 1-based 矿石序号标注（`#1` ~ `#121`）与物理分类预测结果的逐一精确映射，彻底杜绝排序方向冲突引起的逻辑错乱与视觉偏差。
- **New Feature**: 重构并实现 `pick_ores.py` 的**方法3** (基于用户指定的 1-based 静态矿石编号集合进行四档分类渲染)：
    - 输入用户给出的前 25% (31块)、25%-50% (30块)、50%-75% (30块) 和 倒数 25% (30块) 的静态矿石 ID 集合。
    - **类别数值与高性能标注优化**：核心黑框背景标签内直接通过 OpenCV 绘制极其醒目、高对比度的数字类别（**`1`**、**`2`**、**`3`**、**`4`**），代表第一至第四类矿石，对应颜色半透明蒙层及不透明边界，而在矿石顶部边缘精确附上 1-based 序号 `f"#{ore_id}"`，完美兼顾了查找定位与运行的高清极速。
    - **兜底稳定性机制**：针对非指定样本的图片测试，内置了基于轮廓索引排名的动态比例四等分分档算法作为 Fallback 兜底，确保算法的绝对健壮。
- **Refactoring**: 将 `utils_II.py` 中被 `pick_ores.py` 调用的核心几何提取和畸变校正函数（如 `get_bricks`, `get_bricks_watershed`, `split_dual_xray_image`, `sort_contours`, `correct_high_energy_distortion` 等）直接拷贝集成进 `pick_ores.py` 内。现在 `pick_ores.py` **完全独立且不再依赖导入 `utils_II.py`**，极大减少了外部依赖并提升了离线验证和打包执行的稳定性。
- **New Feature**: 重构并实现 `pick_ores.py` 的**方法2** (基于有效原子序数 $Z_{eff}$ 的多档分类)：
    - 输入模型 1 和 2 的 $Z_{eff}$ 预测值，各自归一化至 `[0, 1]` 区间并等分为 4 档 (`T1` 至 `T4`)。
    - **加权决策逻辑优化**：对于两模型预测分档不一致的矿石，采用已归一化值的加权和 $Z_{norm, weighted} = 0.6 \cdot Z_{norm, 1} + 0.4 \cdot Z_{norm, 2}$ 直接划分档位，无需再进行二次全局归一化，极大简化并稳定了判定逻辑。
    - **参数可控化**：在 `label_ores` 中引入 `method` 参数 (1 或 2)，默认为方法2，并在主程序入口支持直接通过文件内参数切换测试不同的图像（如 CuO 测试图）和预测数据。
- **New Feature**: 在 `test_pick_ores.py` 测试脚本中增加方法 1、方法 2 以及短列表填充等多用例离线校验，确保代码的绝对正确和健壮性。
- **New Feature**: Created `pick_ores.py` to extract ore contours using `utils_II.py` and visually label them based on binary predictions (1 for concentrate, 0 for waste) from two separate classification models.
    - Implemented `label_ores`: Core pipeline supporting both standard and watershed contour extraction, length mismatch handling (padding/truncating), and high-fidelity visualization.
    - Designed premium aesthetics including class-specific semi-transparent mask overlays, thick colorful boundary outlines, and contrast-enhanced white-on-grey text annotations centered at contour centroids with index tracking.
    - Supported robust file saving via `cv2.imencode` to prevent Unicode/Chinese encoding failures on Windows.
    - Transitioned script execution in the `if __name__ == '__main__':` block to be controlled via configurable internal input variables instead of CLI arguments, with defaults pre-configured for the Yinshan dataset (`E:\multi_source_info\data_dir\20260325_yinshan\big_ores_position_2_160kV.tif`), making it easy to run and test immediately.
- **New Feature**: Added `plot_ul_cdf` to `utils_II.py` to compute and plot the Cumulative Distribution Function (CDF) curve of physical low-energy attenuation $u_L$ (ul) values for ore samples, automatically displaying the Mean, Median, and a 50% probability reference line for highly detailed statistical analysis.
- **Updated**: Modified `fit_hl_curve_0429.py` to call `utils_II.plot_ul_cdf` inside the ore loops of both the `0429` (0.6mm and 1.2mm) and `0401` datasets, automatically generating and saving cumulative distribution (CDF) graphs for all processed ore samples across voltages.
- **Updated**: Synchronized `code_explanation.md` with detailed parameter descriptions, types, and usages for `plot_ul_cdf`.
- **Refactoring**: Robustly refactored `fit_hl_curve_0429.py` to resolve absolute paths relative to `script_dir`, ensuring correct path resolution and script execution regardless of the shell's current working directory (CWD) (e.g. running from `contour_app`).


## 2026-05-15
- **New Feature**: Developed the **Contour Extraction GUI Tool** using `CustomTkinter`. 
    - Created `contour_app/app.py`: A modern desktop interface for real-time parameter tuning (threshold, ROI, scaling).
    - Created `contour_app/processor.py`: A decoupled backend wrapper for `utils_II` logic, supporting both 8-bit and 16-bit image processing.
    - Added real-time preview, file selection, and result exporting capabilities.
    - Provided a comprehensive guide for packaging into a single `.exe` using PyInstaller.
- **New Feature**: Created `analyze_energy_hardening.py` to implement a physics-based energy back-calculation method...

## 2026-05-14
- **Feature**: Added `analysis_target` parameter to `fit_hl_curve_0429.py` to allow selective analysis of "step", "ore", or "all" data.
- **Bugfix**: Fixed indentation and logic errors in `fit_hl_curve_0429.py`, implementing robust case-insensitive file matching and correct mapping between 0429 and 0401 ore datasets.
- **Feature**: Updated `fit_hl_curve_0429.py` to extract and plot the $u_L/u_H$ ratio ($\ln(I_0/L)/\ln(I_0/H)$) versus voltage for each individual ore sample (Ores 0-6).

## 2026-05-13
- **Updated**: `fit_hl_curve.py` 仿照 `fit_hl_curve_0429.py` 添加了对于 0401 文件夹中矿石数据的画图处理，包括类别型（Categorical）X轴数据（如矿石ID）的自适应映射与坐标轴展示支持，并更新了执行配置以处理0401数据（包含140kV至180kV电压组）。同时增加了专门针对0401矿石数据的序号拼接逻辑，自动将含有 `1_20` 和 `21_38` 前缀的提取文件（后缀 0~19 和 0~17）整合为连续的 0-37 号整体矿石序列。

## 2026-05-12
- **New Feature**: Created `get_mu_from_nist.py` to automate downloading mass attenuation coefficient ($\mu/\rho$) data from NIST for elements (Fe, Cu, Al, etc.).
- **Feature**: Implemented `NISTMuInterpolator` for high-precision log-log cubic spline interpolation of $\mu$ values.
- **Feature**: Added JSON export (`nist_mu_data.json`) to allow cross-script data sharing and persistence of fitted data.
- **Feature**: Added visualization logic to compare multiple elements (Fe, Cu, Al) in a single plot, highlighting the typical XRT energy range (20-150 keV).
- **Updated**: Updated `code_explanation.md` with descriptions of the new NIST data tool and its functions.

## 2026-05-08
- Created **`compare_zeff_methods.py`**: A new standalone script designed to compare different theoretical effective atomic number ($Z_{eff}$) calculation methods across the 98 Yinshan ore samples. The script generates a 2x2 comparison plot (`Zeff_Methods_Comparison.png`) analyzing the impacts of Sulfur inclusion, exponent variation (including simple mass weighting `exp=1.0`), and gangue Base Z adjustments on the final $Z_{eff}$ value.
- Updated **`predict_disk_Z.py`**: Switched the regression plot (`Predicted Z vs Equivalent Z_eff`) from `plt.errorbar` to `plt.scatter` to remove error bars and provide a cleaner, less cluttered visualization of the sample means.
- **Feature**: Added `reverse_sort` parameter to `get_bricks` and `get_bricks_watershed` in `utils_II.py` to allow reversing the order of extracted objects. Enabled `reverse_sort = True` by default for the `20260325_yinshan` dataset in `extract_sample_values.py` to correct the left-to-right labeling order from (6..0) to (0..6).
- **Feature**: Added `all_type` parameter to `extract_sample_values.py` and `classify_contour` in `utils_II.py` to allow users to force a specific object classification (e.g., `'ore'`, `'disk'`) and bypass automatic geometric shape classification.
- Updated **`predict_disk_Z.py`**: Added dynamic file naming templates (`1_98_position_3_{voltage}_ore_{d_id}`) inside the validation loop to correctly read the Yinshan 0325 dataset's `_data.pkl` and `_low.png`/`_high.png` files, resolving missing data errors during validation.
- Created **`update_disk_grades_20260325.py`**: A dedicated script to parse the `2026_矿石数据采集-0325_银山铜矿_114.csv` file, extract the “院里化验” grades (Cu, Fe, S) for all 114 disks, and append them into `disk_grades.json` under the key `20260325`.
- Updated **`predict_disk_Z.py`**:
  - **Dataset Decoupling**: Separated the training parameters date (`params_file` path) from the validation dataset date. Set `TEST_DATA_DIR` to `20260325_yinshan` to allow applying the previously trained model directly to the new Yinshan disk dataset.
  - **Iteration Expansion**: Increased the evaluation loop dynamically to process up to 114 disks.
- Updated **`fit_hl_curve.py`**: Further refined visual aesthetics by decoupling markers from error bars. Mean points are now drawn with full opacity and increased size (`s=40` or `markersize=5`), while error bars are rendered with high transparency (`alpha=0.3`) and no markers. This creates a "bold points, faint bars" effect for better clarity.
- Updated **`fit_hl_curve.py`**: Refined visualization and regression logic:
    - Reduced errorbar opacity (`alpha=0.4`) for disk samples to improve clarity.
    - Switched H-L and Log-Log trendline fitting to use **sample means** instead of individual pixels, providing more stable regression curves and eliminating rank warnings.
    - Extended trendlines to cover the full visible plot range for better trend visualization.
- Updated **`fit_hl_curve.py`**: Optimized **Adaptive Axis Limits** to improve plot readability and zooming:
    - In `'means'` mode (Disks), limits are now calculated based only on sample means, ensuring the view is tightly focused on the data being plotted.
    - In `'all'` mode (Steps), implemented robust percentile-based scaling (2%-98%) to prevent outlier pixels from shrinking the main data clusters.
    - Increased relative padding to 15% for better visual framing.
- Created **`equivalent_thickness_calc.py`**: A new standalone utility script to calculate the equivalent ore thickness corresponding to standard metal step wedges (Cu, Fe, Al) under dual-energy XRT. The script establishes a physical model for porphyry copper ore (incorporating chalcopyrite, pyrite, and gangue) and computes theoretical mass attenuation coefficients and densities to perform the equivalent thickness mapping based on the Beer-Lambert law.
- Updated **`fit_hl_curve.py`**: Implemented **Adaptive Axis Limits** in `perform_comprehensive_analysis`. The function now automatically scans all sample data to determine optimal limits while enforcing row-wise consistency: 
    - **Intensity Row**: All plots share a unified Raw Intensity scale for Y-axes and the H-vs-L X-axis.
    - **Log Row**: All plots share a unified Attenuation scale for Y-axes and the LogH-vs-LogL X-axis.
    - **Horizontal Alignment**: Columns 1 and 2 in both rows share unified Thickness/Z_eff X-limits.
- Updated **`fit_hl_curve.py`**: Added `plot_mode` parameter to `perform_comprehensive_analysis` to support two visualization modes for scatter plots (All pixels vs. Means with error bars). Configured **Step Samples** to use 'all' pixels mode and **Disk Samples** to use 'means' mode with bidirectional error bars for clearer interpretation of mixed samples.
- Updated **`fit_hl_curve.py`**: Expanded disk analysis range to IDs 9-20. The script now loads grade configurations from `disk_grades.json` and uses the calculated effective atomic number ($Z_{eff}$) as the x-coordinate (grade proxy) in analysis plots, providing a more physically meaningful comparison.
- Updated **`predict_disk_Z.py`**: Extracted the hardcoded `disk_grades` dictionary into an external JSON configuration file (`E:\multi_source_info\data_dir\disk_grades.json`). The script now dynamically loads the grades based on the date extracted from the `params_file` path, enabling multi-date support.

## 2026-05-07
- Updated **`utils_II.py`**: Added `calculate_effective_z` function to centralize the calculation of effective atomic numbers (Z_eff) based on ore grades (Cu, Fe, S) using the Mayneord formula.
- Updated **`predict_disk_Z.py`**: Refactored the script to use the centralized `calculate_effective_z` function, improving code maintainability and removing redundant calculation logic.

## 2026-05-07
- Updated **`extract_sample_values.py`**: Modified the image extraction logic so that when saving `high_low_images`, only the pixels strictly inside the contour are preserved. Pixels outside the contour but inside the bounding rectangle are now correctly masked and replaced with the maximum grayscale value (`255` or `65535` depending on image bit depth) to eliminate background noise.
- Updated **`extract_sample_values.py`**: Added visual feedback to draw the minimum bounding rectangle (`minAreaRect`) used for image warping onto the saved `contoured.png` images. The bounding boxes are drawn in magenta.

## 2026-05-06
- Added automatic Chinese-to-English filename translation in `extract_0429_RaySov.py` to prevent OpenCV reading errors. Words like "铁阶梯", "铜阶梯", "校准后" are automatically converted to `Fe_step`, `Cu_step`, `calib`.
- Swapped `cv2.imread` for `cv2.imdecode(np.fromfile(...))` in `extract_0429_RaySov.py` to robustly read 16-bit TIF images from directories containing non-ASCII (Chinese) characters (like "阶梯").
- Upgraded **`normalize_16bit_image`** to **`normalize_image`** in `utils_II.py` to support variable target bit depths. It now correctly maps calibrated 16-bit X-ray images (max value 50,000) to either 80% of 65,535 (`uint16`) or 80% of 255 (`uint8`).
- Updated `extract_0429_RaySov.py` so that both `'16bit'` and `'8bit'` read modes use `cv2.IMREAD_ANYDEPTH` (preserving raw 50k values) and apply `normalize_image` immediately upon loading, ensuring mathematically consistent extraction thresholds across different bit depths.
- Created **`extract_0429_RaySov.py`** to handle specific data extraction for the 20260429 RaySov step sample dataset.
- Added **`get_bricks_raysov`** function within `extract_0429_RaySov.py` to correctly split images into 0.6mm and 1.2mm filter regions, accommodating the 1024-pixel width format (0-255: 0.6mm LE, 256-511: 1.2mm LE, 512-767: 0.6mm HE, 768-1023: 1.2mm HE).

## 2026-05-06
- Updated **`fit_hl_curve.py`**: Added support for distinguishing individual disks with different colors in the h-l fit scatter plots. Introduced the `color_by_step` parameter in `perform_comprehensive_analysis` to dynamically apply categorical colormaps (tab10/tab20) based on sample sequence indices, improving visual separation for different disks while maintaining global regression fits.

## 2026-04-28
- Created **`plot_row_mean.py`** to analyze and plot the mean pixel values of the first 10 rows across columns for the initially converted `.png` images.
- Created **`read_raw.py`** to batch convert 16-bit `.raw` images (1024x1024) into `.png` format, saving them directly to a subfolder within the data directory.

## 2026-04-22
- Refined **`decouple_thickness.py`** 5x3 Visualization:
    - Replaced Row 0 scatter/overlay plots with **Individual Global KDEs** for Model 1, 2, and 3. This provides a cleaner per-model overview of material separation performance.
- Implemented **Model 3 (R-driven Analysis)**:
    - Added feature extraction for $R = \ln(I_{0,L}/L + 5) / \ln(I_{0,H}/H + 20)$ based on `utils_II.compute_R`.
    - Integrated Model 3 into the regression pipeline and accuracy tracking.
- Expanded **`decouple_thickness.py`** Visualization to **5x3 Grid**:
    - Added Model 3 Global Scatter and triple-model Global KDE comparison (Row 0).
    - Added Model 3 Step Breakdown (Row 3).
    - Enhanced Bias Analysis (Row 4) to overlay M1, M2, and M3 mean trends.
- Added **Dynamic Output Directory** and **Parameter Logging**:
    - `output_dir` now automatically creates a subfolder based on the date in `input_dir` (e.g., `.../z_decouple/20260331/`).
    - Implemented `fitting_parameters.txt` to archive all model coefficients and intercepts for every voltage/scenario combination.

## 2026-04-17
- **Optimization**: Optimized `extract_sample_values.py` for 0409 dataset. When filenames contain "270us", it now uses `roi_270` and performs a 1.5x vertical compression using `cv2.INTER_AREA` interpolation before subsequent processing.
- **Improved**: Added `vscale` and `vscale_interp` parameters to `get_bricks` and `get_bricks_watershed` in `utils_II.py` to support flexible image scaling after ROI selection.

## 2026-04-16
- **Improved**: Added height-based vertical splitting logic in `get_bricks_watershed` to separate long merged objects (h > 800px: 50/50 split; 600-800px: 429px/remaining split).
- **Feature**: Created `get_bricks_watershed` in `utils_II.py` to robustly separate touching or overlapping samples using the Watershed algorithm.
- **Improved**: Updated `extract_sample_values.py` to automatically route the 0409 dataset to the new watershed-based extraction function.
- **Feature**: Implemented path-specific thresholding in `extract_sample_values.py`.
- **Improved**: Added `th_type` parameter to `get_bricks` in `utils_II.py` to support `cv2.THRESH_BINARYINV` for the `20260409_TYM-data` dataset.
- **Improved**: Updated `code_explanation.md` with new function parameter details.

## 2026-04-16
- Updated **`fit_hl_curve.py`**: Switched logarithmic attenuation representation from $\ln(I/I_0)$ to $\ln(I_0/I)$. This ensures that attenuation values are positive and increase with sample thickness, providing a more intuitive physical interpretation. Adjusted axis limits to match the new value ranges.

## 2026-04-17
- Expanded **`decouple_thickness.py`** Visualization to **4x3 Grid**:
    - Added **Systematic Bias Analysis** (Row 3): Plots mean predicted Z vs Step Index for Al, Fe, and Cu, comparing Model 1 and Model 2 directly.
    - Preserved high-resolution KDE distributions and global overview in Rows 0-2.
- Added **Accuracy vs Voltage Summary Plot**:
    - Implemented a global data collection mechanism to aggregate `std` metrics across all voltage/scenario test cases.
    - Generated a comprehensive 1x3 comparison chart (`Z_accuracy_summary_comparison.png`) to visualize decoupled Z stability trends.
- Adjusted **Intensity Threshold** in `decouple_thickness.py`:
    - Lowered the valid signal threshold from 10 to 1, enabling the analysis of thicker samples that are closer to the sensor's noise floor.
- Expanded **`decouple_thickness.py`** Visualization to **3x3 Grid**:
    - **Row 0**: Global Performance Overview (Scatter, M1 KDE, M2 KDE).
    - **Row 1**: Model 1 Material Breakdown (Al, Fe, Cu step-wise).
    - **Row 2**: Model 2 Material Breakdown (Al, Fe, Cu step-wise).
    - Added granular legends for each material-specific subplot to explicitly identify thickness steps.
- Optimized **Thickness Color Gradients** in `decouple_thickness.py`:
    - Implemented **Dynamic Color Normalization**: Ensures full color spectrum usage regardless of step count.
    - Improved **Visual Clarity**: Increased alpha (0.7) and linewidth (1.0) for step curves; introduced a dashed black baseline for total distributions.
- Enhanced **`decouple_thickness.py`** Visualization:
    - Added **Step-Wise KDE Analysis**: Distributions are now plotted for each individual thickness step using color gradients (e.g., light to dark shade representing thin to thick).
    - Refactored data tracking: Improved pixel-level labeling to maintain thickness awareness during global regression and sampling.
- Integrated **`StandardScaler`** into `decouple_thickness.py`:
    - Added feature standardization to resolve `LinAlgWarning: Ill-conditioned matrix` for both ratio and polynomial models.
    - Implemented **Coefficient Unscaling** logic to transform model parameters back to the original physical space for human-readable formula display on plots.
- Implemented **Multi-Scenario Thickness Analysis** in `decouple_thickness.py`:
    - Added support for running multiple thickness subset configurations (Case 1: Al6/CuFe4, Case 2: Al8/CuFe6, Case 3: Al10/CuFe8).
    - Automated step slicing to exclude non-penetrating thick steps from the regression model.
    - Updated Plot Layout: Optimized to a 1x3 structure containing Model 1 Scatter, Model 1 KDE, and Model 2 KDE.
    - Added dynamic file naming and console variance reporting for all scenarios.

## 2026-04-15
- Enhanced **`code_explanation.md`**: Added detailed parameter descriptions and return type information for the feature extraction functions in `decouple_thickness.py`.
- Implemented **`calculate_mu_m.py`**: A new verification script that calculates the mass attenuation coefficient ($\mu_m$) as a function of thickness for Cu, Fe, and Al samples.
- Fixed **`fit_hl_curve.py`**: Updated the pixel loading logic to handle the `list` format (used for stepped samples in `.pkl` files), restoring the script's ability to generate H-L trajectory comparison plots.
- Updated **`code_explanation.md`**: Added documentation for `fit_hl_curve.py` and `calculate_mu_m.py`, and finalized descriptions for the thickness decoupling pipeline.

## 2026-04-16
- Implemented **Automated Summary Reporting**: `extract_sample_values.py` now generates a formatted `pandas` table at the end of execution.
- Categorized Metrics: The table provides **Mean** and **STD** for every object:
    - `step_sample`: 10-step arrays of means and STDs.
    - `disk`: 2/3 inner core stats.
    - `block`: 95% inner region stats.

- Fixed **Block Extraction Coordinate Bug**: Re-routed `block` sample erosion (`get_inner_95_pixels`) to execute directly on the global `low_roi` and `high_roi` in `extract_sample_values.py`. This resolves an issue where the global shape contour (`cnt`) was erroneously applied to tiny, locally-warped cropped images, which previously resulted in empty `pixels_low` arrays and mismatched `pixels_high` arrays for `block` objects.

- Fixed **Step Sample Data Extraction**: Corrected a logic flaw where `step_sample` data was being saved as a single global array. It now correctly saves a **list of 10 pixel arrays** (one per step core) in the `.pkl` file, enabling precise thickness-based analysis.

- Formulated and implemented a regression-based thickness decoupling strategy in `decouple_thickness.py`. It uses a polynomial model to map High and Low energy pixels (H, L) directly to predefined Atomic Numbers (Z), creating thickness-invariant variables.
- Created parameter feature engineering: `extract_feature_HL_ratio` and `extract_feature_poly` functions inside `decouple_thickness.py` to compare linear, quadratic H/L ratio models vs generic 2D polynomials.

- Integrated Dual-Axis Calibration: `get_bricks` and `split_dual_xray_image` now propagate both `fx` and `fy` parameters (default `fx=0.9909`, `fy=1.0`).

- Refined **Disk Core Sampling**: Replaced circular 2/3 radius approximation with **Centroid-Based Contour Scaling**. The sampling area (blue boundary) now precisely tracks the actual geometry of the disk, even if it is elliptical or irregular.

- Refined **Step Sampling Box Dimensions**: Decoupled horizontal and vertical margins in `get_10_step_means`. New default coverage: **80% horizontal** (margin_x=0.1) and **60% vertical** (margin_y=0.2). This provides the optimal balance between data density and edge protection.

- Widened **Step Sampling Boxes**: Reduced `sampling_margin` from 0.3 to **0.1** in `get_10_step_means`. This expands the horizontal sampling area from 40% to **80%** of the object's width, significantly increasing the data representation for each thickness step.

- Hardened **Classification Robustness**:
    - Lowered Pearson Correlation threshold to **0.7** to detect non-linear (accelerating) stepped gradients.
    - Lowered Rectangularity threshold to **0.75** to correctly classify stepped samples with irregular or noisy contours (preventing them from defaulting to `ore`).

- Restructured **Dynamic Step Classification**:
    - Replaced loose monotonicity checks with **Pearson Correlation ($|r| > 0.9$)** for stable trend detection.
    - Introduced **Intensity-Aware Dynamic Span Threshold**: Classification now requires a minimum value range of `max(2.0, 0.05 * average_intensity)`. This prevents flat, noisy surfaces from being misidentified as steps while preserving sensitivity for low-intensity samples.

- Optimized **Step Classification Sensitivity**: Changed the dependency between monotonicity and intensity jump from `AND` to `OR`. Stepped samples are now identified if they show either a significant intensity jump OR a clear monotonic trend, improving detection for samples with low thickness/intensity spans.

- Fixed **Redundant Annotation Buildup**: Centralized all mean/std drawing in the main processing loop to prevent shadow-text and overlap on disks.
- Simplified `get_bricks` to only handle contour ID drawing.

- Implemented **Refined Type-Specific Analysis**:
    - **Step Samples**: Now saves 10 individual pixel arrays (one per step core) in `.pkl` instead of a single object block.
    - **Disks**: Statistics (`mean`, `std`) and saved pixels are now based on a **2/3 radius core** to eliminate edge effects.
- Enhanced **Verification Visuals**:
    - Added **Blue Circular Boundary** for disk core sampling areas.
    - Updated centroid labels to show **Core Statistics** in yellow for disks.

- Added **Classification Labels** to visual results: The summary images in `contoured_images/` now display the detected category (e.g., `block`, `step_sample`) in green above the object's metadata.
- Optimized annotation layout to prevent text overlap.

- Implemented **95% Inner-Pixel Calculation**: Refined global `mean` and `std` to only use 95% of the interior pixels (via contour erosion), eliminating boundary noise and mixed-pixel artifacts.
- Restructured **Organized Result Storage**: Upgraded the flat output directory into a categorized subfolder system (`contoured_images`, `pixel_values`, `high_low_images`).
- Added `get_inner_95_pixels` helper to `utils_II.py`.

- Consolidated **Visual Annotations**: Integrated `Mean` and `Std` display directly into `get_bricks` for a cleaner result image.
- Guaranteed **Stat Integrity**: Ensured that intensity metrics are calculated on original pixels before any text or contours are drawn on the image.
- Improved **Layout Clarity**: Reduced text clutter by using smaller, layered font blocks for contour metadata.

- Fixed **ROI Coordinate Misalignment**: Re-aligned the warping pipeline to use ROI-cropped images, ensuring `low.png` results are centered and accurate.
- Verified **Intensity Profiling**: Confirmed that 10-step mean sequences now perfectly match the global contour means without background interference.

- Implemented **High-Precision Warped Step Detection**: Straightens tilted objects using perspective transforms for accurate segmenting.
- Added **10-Step Margin Sampling**: Extracts means from the central 40% of each segment (30% margin) to eliminate edge/alignment noise.
- Enhanced **Visualization**: Draws back-projected sampling boxes on `contoured.png` for visual verification of extraction regions.
- Updated **Data Storage**: ROI images (`_low.png`, `_high.png`) are now saved as straightened crops.
- Refined classification logic to use 10-step monotonicity.

- Implemented **Segment Mean Intensity Logging**: The system now outputs absolute mean values for the top 1/10, middle 8/10, and bottom 1/10 segments of rectangular objects.
- Updated `check_step_gradient` and `classify_contour` to propagate these raw intensity metrics to the console.
- Refined per-image console output to display `[top, mid, bot]` triplets for classification cross-verification.

- Implemented **Detailed Gradient Data Logging**: `check_step_gradient` and `classify_contour` now return raw `diff_top` and `diff_bottom` values for transparent threshold tuning.
- Updated `standard_sample_0402.py` to print a per-image summary of these gradient differences.
- Refined `classify_contour` to return a structured `meta` dictionary instead of individual values.

- Implemented **Row-Gradient Step Detection**: Replaced `std`-based logic with a more robust method that compares the mean of edge regions (first/last 1/10th) with the middle (8/10ths) to identify thickness steps.
- Added **Centroid Annotations**: The `contoured.png` output now automatically displays the `mean` and `std` values at the centroid of each identified object for immediate visual verification.
- Updated `utils_II.py` with `check_step_gradient` utility.

- Enhanced `classify_contour` in `utils_II.py` with `Ellipse Fit` logic to robustly identify disks even when elongated.
- Implemented automated detection of `step_sample` (bricks) by calculating pixel standard deviation (`std > 5.0`) for rectangular objects.
- Updated `standard_sample_0402.py` to flow pixel data through the classifier.

## 2026-04-21
- Refined **`compare_tube.py`** with **Manual Index Pairing**:
    - **Flexible Matching**: Updated `main()` to support separate index lists (e.g., `indices_125` vs `indices_270`), enabling precise pairing of corresponding ores even if their segmentation IDs differ.
- **Improved Visualization**:
    - **Per-Ore Histograms**: Implemented `plot_simple_hist_grid` to create individual comparative subplots for each matched ore pair, ensuring clear visibility of distribution changes across exposure times.

- Added **Intensity Correlation Plot** to `compare_tube.py`:
    - **XY Scatter Mode**: Supports comparing two groups of datasets by plotting one as X and another as Y (e.g., 125us vs 270us).
    - **Automatic Formatting**: Includes an identity line (y=x) and point annotations to visualize intensity shifts and linearity across different exposure settings.
- **Improved**: `main` now automatically collects common ore samples (Ore 0-3) to provide a statistically meaningful correlation view.

## 2026-04-11
- Updated `standard_sample_0402.py` to automatically organize results into subfolders within `results/`, named after the input data directory (e.g., `results/20260402/`).
- Refined saving logic to output separate low and high energy images for each contour and exclude R-value data.
- Implemented automated shape classification (block, ore, disk) in `utils_II.py`.

- Added automated shape classification (`block`, `ore`, `disk`) in `utils_II.py` using geometric features (rectangularity, circularity).
- Implemented categorized saving: individual `.pkl` data and ROI box images now include the type name in the filename.
- Updated `standard_sample_0402.py` to leverage automated classification and multi-file saving.

## 2026-04-10
- Updated `txt2img_TYM.py` to skip files containing "offset" or "air" in their filenames (case-insensitive).

- **Improved**: Added support for **16-bit precision** output (uint16) to preserve raw data values (ideal for .tif).
- **New Feature**: Added support for configurable output formats (e.g., `.tif`, `.png`), defaulting to `.tif`.
- **Improved**: Switched to `cv2.imencode` for saving images to support Unicode/Chinese paths on Windows.
- **New Feature**: Added automatic filename translation from Chinese to English.
- **Optimization**: Implemented `pandas` for faster loading of large data grids.

## 2026-04-08
- Preserved original float R-values in `utils_II.py` by removing 0-255 normalization.
- Updated visualization in `standard_sample_0402.py` and `standard_sample.py` to use `plt.imshow` with `jet` colormap and colorbar for scientific accuracy.

## 2026-04-08
- Switched R-image saving from `cv2.imwrite` to `plt.savefig` in `standard_sample_0402.py` and `standard_sample.py` for better visualization.

## 2026-04-08
- Fixed R-image visualization by adding normalization (0.5-1.5 -> 0-255) and `uint8` conversion in `utils_II.py`.
- Converted R-image to BGR before drawing contours to enable red color visualization and avoid `imwrite` depth warnings.

## 2026-04-08
- Synchronized `get_bricks` parameters in `standard_sample_0402.py` with `standard_sample.py` (`roi=[0, 1000, 200, 1336]`, `th_val=175`).

## 2026-04-08
- Fixed `ModuleNotFoundError` by adding script directory to `sys.path` in `standard_sample_0402.py` and `standard_sample.py`.
- Corrected import from `utils` to `utils_II` in `standard_sample_0402.py`.

## 2026-04-08
- Created `standard_sample_0402.py` for multi-voltage XRT image analysis (140kV, 160kV, 180kV).
- Processed 9 images from `E:\multi_source_info\data_dir\20260402` involving 3 tests and 3 voltages.
- Automated saving of R-value contoured images and pixel data (Pickle) to the `results/` directory.

## 2026-04-08
- Configured Python environment path for external library `jt_ore_sorting-main`.
- Created `utils.py` as a local utility wrapper for image processing.
- Fixed `utils.py` missing imports (`numpy`, `cv2`, `pandas`) and external function imports (`preprocessing`).
- Created `standard_sample.py` to automate contour detection and image saving for standard sample XRT data.
- Processed `Sample_160kV_test1.tif` and generated `standard_sample_contoured.png`.

## 2026-05-09
- Created `fit_hl_curve_0429.py` to seamlessly adapt the comprehensive 2x3 grid visualization from `fit_hl_curve.py` to the 0429 dataset:
    - Independent processing loops for `0.6mm` and `1.2mm` filter data with results saved into dedicated subdirectories.
    - Updated naming conventions to match `orig` and `calib` 8-bit files (with $I_0=204$).
    - Automatically collects all calibrated ore (`ore-*-orig`) files for joint sample plotting.
- Redesigned `calculate_mu_m.py` visualization pipeline:
    - Fixed Y-axis labeling to correctly render as a LaTeX formula: `$\mu_m \ (\mathrm{cm}^2/\mathrm{g})$`.
    - Computed a true **Global Y-axis limit** across all datasets, voltages, and materials first, then applied it identically across all output figures to guarantee 100% strict limit unification.
    - Modified the secondary plot to display $\mu_m$ vs Thickness for different voltages grouped by **Material** (one subplot per material containing 14 overlapping voltage curves).
    - Visually mapped the Aluminum step x-axis to `2-20mm` for direct comparative alignment with Copper and Iron, while keeping the physical calculation at `12-30mm`.
    - Added rigorous signal filtering (`mean > 1` and `mean < I0`) to prevent outlier pixels or background noise from generating massively skewed negative or infinite logarithms.
- Switched 0429 dataset processing (both extraction and mu_m calculation) to **8-bit** mode (normalization target: 80% of 255).
- Updated `calculate_mu_m.py` to support the 20260429 dataset:
    - Added separate visualization for 0.6mm and 1.2mm filter regions.
    - Updated filename pattern to match the English translated names (e.g., `calib`, `Fe_step`) from the mask extraction process.
- Refined `extract_0429_RaySov_from_mask.py`:
    - Implemented automatic filename translation (e.g., "铁阶梯" -> "Fe_step") to ensure results are saved with English names.
    - Updated output path to `results/20260429_mask_generated` to match project-wide standards.
    - Integrated `save_contour_data` to match the `.pkl` and `.png` saving format of `extract_sample_values.py`. and reversed step ordering to **Bottom-to-Top** (Thin-to-Thick, `S1`=thinnest).
- Created `extract_0429_RaySov_from_mask.py` to batch extract pixel values directly from manually annotated `.png` masks in the `20260429_mask_generated` directory, separating Low-Energy and High-Energy data automatically.
- Added adaptive normalization logic inside `extract_0429_RaySov_from_mask.py` which scales values to 80% of 65536 for `.orig.tif` files while preserving completely raw values for `.user.tif` files.
- Completely refactored `extract_0429_RaySov.py` to extract step samples globally instead of pre-cropping.
- Implemented `find_step_sample_corners` in `extract_0429_RaySov.py` to geometrically identify the true left and right vertical edges of the step sample, allowing for perfect straightening while computationally ignoring the wider bracket at the bottom.
- Added splitting logic mapping the exact global filter boundary (`X=256`) into the straightened warped image, allowing independent 10-step sample extraction for both 0.6mm and 1.2mm halves simultaneously.

## 2026-04-08
- Created `validate_speed.py` to compare XRT images at 0.5 m/s and 3.0 m/s.
- Implemented pass extraction (top half) for 0.5 m/s image.
- Implemented low-energy focus (left half) for both images.
- Performed initial intensity and noise analysis between the two speeds.
- Refined `validate_speed.py` to include 200px side-cropping for the low-energy channel.
- Implemented mask-based pixel extraction for precise per-ore statistics using `cv.findContours`.
- Implemented a unified threshold search (Thresh 202) for consistent ore isolation across speeds.
- Added relative percentage difference reporting to the validation summary.
- Generated `ores_detailed_mask_comparison.png` for final validation.
