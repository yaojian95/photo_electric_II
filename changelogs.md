## 2026-07-14
- **新增 PC2 抛废率与回收率权衡曲线 (ROC-style)**: 在 `calculate_industrial_metrics.py` 中新增 `plot_pc2_tradeoff_curve` 评估函数。该模块会自动扫描 500 个 PC2 判定阈值，严谨计算每一个阈值界线下的“矿石铜回收率”和“总体抛废率”，并绘制出一张直观反映两者博弈关系的“折线图”。生成结果自动保存在各实验目录下的 `pc2_threshold_tradeoff_curve.png` 中。
- **增加 PC2 大于阈值分类选项**: 在 `SinglePC2ThresholdClassifier` 及其相关流水线函数中，增加了 `greater_than` 和 `pc2_greater_than` 参数。通过设置此参数为 `True`，可以使单变量 PC2 阈值分类器将 PC2 **大于** 阈值的像素点判定为精矿。
- **新增反向判定实验**: 在 `run_experiment.py` 中新跑了基于大于阈值为精矿的纯新金属片实验 (`exp_metals_Z_le_30_new_only_greater_than`)，以对比阈值符号对平衡准确率的影响。
- **报告新增总体质量准确率**: 在 `run_experiment.py` 的报告生成器中，从 `classification_summary.csv` 提取了每个模型的 `Mass_Acc` (总体质量加权准确率) 拼接到 markdown 对比表中。
- **新增纯新金属片实验流**: 在 `run_experiment.py` 中新增并执行了 `exp_metals_Z_le_30_new_only` 实验流。该实验排除了所有旧版铜、铁、铝标样，**仅使用有效原子序数 $Z \le 30$ 的最新批次金属片（Ti, Fe, Ni, Cu, Zn）** 进行 PCA 正交基训练和特征提取。实验产生的全套可视化图表和数据指标现已完全独立保存在 `results/fit_pca/exp_metals_Z_le_30_new_only/` 目录中，方便您与之前的融合实验进行最直观的对比。
- **修复绘图 OOM 错误**: 在 `fit_pca_classifier.py` 中进一步下调了散点图抽样的最大点数（从 20000 降至 1000），彻底修复了因为在同一超大图像绘制 8 个金属材质各数十万像素点从而导致底层 matplotlib 发生 `MemoryError: bad allocation` 的系统内存崩溃问题。

## 2026-07-13
- **厚度归一化 (Thickness Normalization)**: 彻底重构了提取特征的流水线，使得主成分分析 (PCA) 完全剥离了样本物理厚度的影响。在 `run_experiment.py` 中补全了所有参考金属的物理厚度信息（包含旧版铝10阶、铜铁4阶，以及新版金属片），并在 `fit_pca_classifier.py` 的特征提取和可视化环节中，统一将高低能对数衰减量除以各自的物理厚度 $t$（矿石取 `mean_thickness`，标样取对应厚度）。坐标系全面变更为“单位厚度衰减系数” ($u_L/t$ vs $u_H/t$)，使第一主成分向着纯粹反映密度的方向收敛，第二主成分纯粹反映有效原子序数 (Z-eff)。
- **衰减原始空间可视化**: 新增 `plot_step_attenuation_space` 函数，在保存 PCA 转换后的平面投影图外，也会在 `results/fit_pca/` 的各实验目录下额外生成试样在高低能衰减量空间（$u_L$ 对应 $u_H$）中的原始分布图 `step_attenuation_space.png`。
- **支持新金属片数据集扩展**: 在 `fit_pca_classifier.py` 中更新了 `load_step_data` 的兼容逻辑，现在支持直接传入文件路径的列表。基于此新增了 [fit_PCA/metal_mapping.py](file:///e:/photo_electric_II/fit_PCA/metal_mapping.py)，对最新的 `160kV_4mA` 金属序列（Zn, Ni, Sn, Fe, Ti, Pb, Cu, W）的 `pkl` 文件进行了物理厚度和种类的精准映射，并在 orchestrator 中添加了 `exp_all_metals_160kV_4mA` 实验配置。
- **自动适配 16位/8位 $I_0$ 背景强度**: 在 `fit_pca_classifier.py` 中新增了输入数据位深自适应检测逻辑。当检测到最大灰度值超过 500（16位原生传感器数据）时，自动将 $I_0$ 修正为 $65536 \times 0.8 = 52428.8$；否则保持原有的 8位 归一化强度 $204.0$。此举不仅保证了高低能衰减量 $u_L, u_H$ 的计算结果处于物理直觉的正数区间，也完美统一了原始空间与 PCA 空间的坐标尺度体系。
- **绘图内存溢出保护 (OOM Fix)**: 考虑到原生金属试样和矿石像素点量级数以百万计，在 `plot_step_pca_space`、`plot_step_attenuation_space` 和 `plot_ore_pca_comparison` (标样与矿石叠加网格图) 中全面部署了防内存溢出（MemoryError: bad allocation）机制。通过对 `matplotlib.pyplot.scatter` 引入最大 $20000 \sim 50000$ 个像素点的 Numpy 无放回随机降采样 (`np.random.choice`)，确保在生成包含 8 种高密度元素的复杂分布图时既能完美呈现密度趋势轮廓，又能使后台渲染速度极快、不崩溃。
- **矿石原始衰减空间双重映射对比图**: 新增了 `plot_ore_attenuation_comparison` 绘图模块。现在系统在输出矿石的 PCA 投影对比图之前，会额外先生成对应矿石组与纯金属标样在**原始衰减量空间 ($u_L$ vs $u_H$)** 中的叠加分布网格图（如 `ore_attenuation_comparison_group1.png`）。这让您能够在没有任何坐标变换的前提下，最直观地比对出由于原子序数不同所带来的矿石衰减弧线偏转趋势。
- **构建 Z<=30 低原子序数金属融合实验**: 在 `run_experiment.py` 中新增 `exp_metals_Z_le_30` 实验流。该实验提取了 `METAL_MAPPING` 中所有 $Z \le 30$ 的新金属（Ti, Fe, Ni, Cu, Zn），并将其与原版实验中的 10 阶旧版金属（Cu_step, Fe_step, Al_step_block）进行了混合加载，用于专门针对低低原子序数区间求解更精细的 PCA 正交基并重新评测模型的分选性能。
- **金属序列按原子序数排序与标注**: 在 `plot_step_attenuation_space`、`plot_step_pca_space` 和 `plot_ore_pca_comparison` 所有图中，现已将各阶梯金属的显示顺序（包括子图排列、图例排序）根据其真实的原子序数 (Z 值) 进行了升序排列（例如 Al, Ti, Fe, Ni, Cu, Zn, Sn, W, Pb），并在标题及图例中清晰地标注出了具体的 `Z=...` 数值，从而方便直观分析有效原子序数（Z-eff）规律。
- **Auto-Generate Markdown Report**: 在 `run_experiment.py` 中增加了 `generate_comparison_report` 函数。程序运行结束后，会自动读取各实验隔离目录下的 `industrial_sorting_indicators.csv` 和 `subgroup_mass_accuracy.csv` 生成 `experiment_comparison_report.md` 输出，并且统一保存所有图表至 `results/fit_pca/` 独立子目录下杜绝相互覆盖。

## 2026-07-04
- **Refine mu_L/mu_H Ratio Plot Layout**: 根据要求优化了 `get_mu_from_nist_new.py` 中的 `plot_mu_ratio_difference` 函数。移除了相对差异曲线图；重新布局为左右双图并排（1x2）：左图固定展示高低能差 $\Delta E = 30$ keV 下 Cu 与 Al 的 $\mu_L / \mu_H$ 比值随低能升高的下降曲线；右图展示在不同的高低能间隔（$\Delta E = 20, 30, 40, 50$ keV）下 Cu 与 Al 比值的绝对差异变化趋势。继续保留所有学术论文级别的图表渲染规范。同步更新了 `code_explanation.md`。

## 2026-07-02
- **Save Low-Energy Step 1 Separately**: 在 [compare_tube.py](file:///e:/photo_electric_II/compare_tube.py) 中新增了 `plot_single_step_hist` 函数，用于单独绘制并保存单个特定阶梯的灰度分布直方图。修改了 `run_comparison` 函数，在进行阶梯标样比对时，除输出原有的 2x5 阶梯直方图网格大图外，同步调用该函数将低能通道的第一个阶梯（Step 1）的直方图单独保存为 `{prefix}_hist_low_step1.png`。同步更新了 [code_explanation.md](file:///e:/photo_electric_II/code_explanation.md)。

## 2026-06-15
- **180kV vs 160kV Comparison Script**: 新增脚本 [PCD/compare_180kV_160kV.py](file:///e:/photo_electric_II/PCD/compare_180kV_160kV.py)，用于加载 `20260512_180kV` 和 `160kV` 两个电压数据集，在 12 个共有能量通道段（20keV - 130keV）下进行跨电压比对。在 `steps/` 中生成 12 张 2x2 对比图叠加展现两电压在归一化低能和原始高能下的强度和对数衰减趋势；在 `combined/` 中生成随能量变化的 2x2 汇总比值折线大图（`slope_summary`）与随厚度变化的衰减系数 3x2 折线大图。**修正了铝阶梯（al_left）的实际物理厚度范围为 `12-32mm` (先前误为 2-20mm)；优化了绘图线宽（增至 2.2）、缩减了数据点标记大小（降至 4）且增加了图例 handle length（设为 3.5），解决了虚线不明显、被数据点标记遮挡的问题。**同步更新了 [code_explanation.md](file:///e:/photo_electric_II/code_explanation.md) 的相关解析。
- **Support for Dual and noNorm_R Suffixes**: 优化了 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py) 和 [PCD/divide_al_sheet.py](file:///e:/photo_electric_II/PCD/divide_al_sheet.py)，使其同时支持以 `_dual` 和 `_noNorm_R` 结尾的能量通道目录。在 `fit_hl_curve_dual.py` 中重构了 `extract_energy` 函数，使其能从小能谱通道名称 `180kV_1mA+MERGE_E_100-110keV_post_L__post_noNorm_R` 中提取实际能段 `100` 作为有效能量值（原先单纯寻找首个数字会返回 180 导致曲线合并计算错乱）。同时实现了 `divide_al_sheet.py` 末尾校验打印的 `demo_band` 动态自适应回退选择。
- **Cell-Level Scaling in Al Step Partitioning**: 优化了 [PCD/divide_al_sheet.py](file:///e:/photo_electric_II/PCD/divide_al_sheet.py) 中的 `scale` 缩放逻辑。现在整体区域不再按 `scale` 进行全局缩小（避免了分割界线的偏移以及裁剪后图像 of 面积压缩），而是将 `scale` 作用于各个具体的阶梯单元格内部。在横向等分 3 列、纵向等分 10 阶所得的单元格中心点上进行 `scale`（默认 `0.9`）缩放，再叠加 `margin_x`/`margin_y` 锁定采样核心区域，同时裁剪图像也恢复为完整真实面积的分割列。同时同步更新了 [code_explanation.md](file:///e:/photo_electric_II/code_explanation.md)。

## 2026-06-12
- **Unified Global Y-Limits for Summary Plots**: 优化了 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py) 中的汇总图纵坐标。现在通过两阶段计算逻辑，搜集并统计所有 10 个厚度台阶下的衰减系数和材质比值，计算出统一的全局 Y 轴上下限。这使得不同厚度台阶（不同大图）之间的 Y 轴范围和跨度完全一致，更加方便横向数据对比。
- **New mu vs Thickness Plot**: 在 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py) 中新增了 `plot_mu_vs_thickness` 函数，生成了展示线衰减 $\mu$ (和质量衰减 $\mu_m$) 随厚度变化的 3x2 大图（包含 Cu、Fe、Al_left 三种材质，归一化和原始两套数据，不同颜色曲线代表不同能量段），落地保存在 `combined/` 目录下。
- **Combined Slope Summary 2x2 Grid**: 优化了 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py) 中的汇总折线图绘制逻辑。将原先独立输出的 `norm`（归一化）和 `raw`（原始通道）两套 1x2 汇总图合并到了同一个 2x2 网格的大图（第一行为归一化数据，第二行为原始数据，左侧为衰减系数随能量段变化，右侧为两两材质衰减比值），并保存在 `combined/slope_summary_{mu_mode}` 目录下，大大方便了数据的直观横向对比，同时仍然保留了对 `norm` 和 `raw` 双通道 JSON 参数文件的分别归档导出。
- **Y-Axis Limits Correction**: 优化了 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py) 中的 Y 轴上下限计算逻辑。引入了 `get_clean_ylim` 函数，强行将灰度强度值和对数衰减值的 Y 轴下限约束为 `0.0`，从而避免出现物理上不合理的负数轴，并自适应将上限向上取整为美观干净的整数值（如 5.0、1600.0 或 50000.0），确保多能量通道折线图和 2x2 剖析图在不同能量段下的纵坐标区间高度统一和严谨对比。
- **HL Curve Analysis for Photon Counter**: 将脚本移动并放置在 [PCD/fit_hl_curve_dual.py](file:///e:/photo_electric_II/PCD/fit_hl_curve_dual.py)。根据最新 analysis 需求，同时支持对归一化数据（低能通道 `pixels_low`）和归一化前的原始数据（高能通道 `pixels_high`）的拟合分析。各能量段（20keV-130keV）的绘图结构升级为 2x2 网格（第一行为归一化数据，第二行为原始数据，各包含灰度衰减和对数衰减自适应线性拟合），且原始数据的对数衰减采用了基于每个能量段最大灰度值动态计算的未衰减背景值 $I_{0,\mathrm{raw}} = 1.15 \times I_{\mathrm{raw,max}}$，确保数值物理合理性且不同通道间纵坐标高度一致以便横向对比。多能量通道的 slope_summary 汇总折线图与归档 JSON 相应为 `norm` 和 `raw` 两套数据分别输出，流程仅耗时 27 秒即完成全量计算。
- **Al Step Partitioning for Iron Sheets**: 新增脚本 [divide_al_sheet.py](file:///e:/photo_electric_II/PCD/divide_al_sheet.py) 用于将 Al 阶梯（step3）数据在横向上划分为左（纯 Al）、中（Al + 0.3mm Fe 铁片）、右（Al + 0.6mm Fe 铁片）三个区域，在整体缩放到 `0.9` 的核心采样区域内提取 10 个厚度台阶 of 像素数据。仅保存全称形式的 `.pkl` 文件（如 `*_left.pkl`、`*_mid.pkl`、`*_right.pkl`，并自动清理旧的 `_L`、`_M`、`_R` 缩写后缀文件）及对应裁剪的 `.png` 图片。同时在各 `_dual` 能量段子文件夹下输出带有整体 0.9 边界线、区域划分黄色分界线 and 10 阶采样框的 `*_step_sample_3_division.png` 可视化图像用于物理校验。

## 2026-06-11
- **Inverse Square Law Analysis**: 新增脚本 detector_raw_intensity/evaluate_inverse_square.py 用于评估 X 射线探测器灰度值是否满足平方反比定律，并确定皮带边缘在探测器上的像素位置 (左右侧分别对应像素 111 和 1419)。
- **R2 Metric & Artifact Analysis**: 在平方反比拟合曲线的图例中新增了决定系数 ^2$ 评估拟合优度，并在报告中揭示了高能图像正中心(像素 767/768 处)由于双能探测器双模块物理拼接所导致的特征凹点/缝隙伪影(Module Seam/Gap)。
- **R2 Summary Table**: 在分析报告中补充了详细的 ^2$ 拟合公式与各个电压/能量下的拟合结果对照表格。
## 2026-06-11
- **相关性热力图（01c）列与顺序更新**：更新了 [generate_combined_figures.py](file:///e:/photo_electric_II/density_with_grade/generate_combined_figures.py) 和 [generate_supplemental_figures.py](file:///e:/photo_electric_II/density_with_grade/generate_supplemental_figures.py)，将合并相关性热力图 `01c` 的变量调整为与原版一致，包含`密度`、`Cu`、`Fe`、`S`和`总金属品位`，并将列顺序对齐。
- **补充合并版图像**：新增 [generate_supplemental_figures.py](file:///e:/photo_electric_II/density_with_grade/generate_supplemental_figures.py)，修正 `01c` 相关热力图列顺序（Cu→Fe→密度→质量，与原版 `01` 保持一致）；新增 `09c`（合并批次 Cu/Fe 品位质量分布 + 密度分段箱线图）和 `10c`（合并批次质量 vs 品位散点）。报告 §7 图表索引补充 09c/10c 链接，§7.1 可疑样本表格加入扫描体积列，密度计算方式修正为激光线扫描仪。

## 2026-06-10
- **Combined Figures (00c~08c) + Cu/Fe 联合分选曲线**: 新增 [generate_combined_figures.py](file:///e:/photo_electric_II/density_with_grade/generate_combined_figures.py)，为合并批次（0325+0520）生成全套 `00c~05c` 图像（密度段统计、相关热力图、预测vs实测、散点图、分布直方图、KDE 轮廓），并为 0325 单批次（`06a/07a/08a`）和合并批次（`06c/07c/08c`）分别生成 Cu+Fe **联合分选曲线**（选别曲线、指标 vs 阈值、典型阈值柱状对比）。报告 §6.3 同步扩展为 Cu/Fe/S 三元素各自分选指标子表。
- **Combined Batch Analysis (0325 + 0520)**: 新增 [analyze_combined.py](file:///e:/photo_electric_II/density_with_grade/analyze_combined.py)，合并分析 0325（114块含Cu/Fe/S）与 0520（100块含Cu/Fe，无S）两批次矿石数据（合计 209 块，剔除 5 个密度物理异常值）。主要结论：0520 批次矿石 96% 集中于密度 < 2.5 g/ml，整体为贫矿区，与 0325 差异显著；0520 质量加权 Cu=0.14%，约为 0325（0.44%）的 1/3；合并回归 R²=0.216；5个密度可疑样本（含12.7 g/ml 和 0.30 g/ml 的极端值，激光扫描误差所致）标记需重测。报告新增 §七。
- **Grade Distribution Plot Updates**: 优化了 [plot_grade_distribution.py](file:///e:/photo_electric_II/density_with_grade/plot_grade_distribution.py) 脚本。
  - 将去极值算法从 1.5×IQR 更改为 **99% 分位数截断**，避免极度右偏分布下过多合法高品位样本被误删。在分析报告中详细列出了仅剔除的 2~3 个极端离群值样本（如 #32531 和 #325114）。
  - 直方图由"品位-频数"分布改为**"品位-质量"分布**（基于样本质量加权），并在右侧 Y 轴引入了紫色的**累积质量百分比曲线**，使得各品位区间的实际质量占比更直观。
- **Density Threshold Sorting Analysis**: 新增分选模拟脚本 [sorting_analysis.py](file:///e:/photo_electric_II/density_with_grade/sorting_analysis.py)。对密度范围进行 500 点全局扫描，计算产率、各元素回收率、富集比和尾矿品位，生成选别曲线和指标随阈值变化图（`06~08`）。典型结论：ρ>3.0 g/ml 可在仅 23.8% 产率下实现 Cu 回收率 78.2%、富集比 3.29×；ρ>3.2 g/ml 可将 Cu 品位从 0.44% 提升至 1.90%（富集 4.28×）。报告 §六 同步更新了完整分选指标对比表与策略建议。
- **Density and Grade Analysis (深度版)**: 在 `density_with_grade/` 目录下新增并升级了分析脚本 [analyze_density_grade.py](file:///e:/photo_electric_II/density_with_grade/analyze_density_grade.py)，同时编写了详细的书面分析报告 [密度与品位分析报告.md](file:///e:/photo_electric_II/density_with_grade/密度与品位分析报告.md)。

