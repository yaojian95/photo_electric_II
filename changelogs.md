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

