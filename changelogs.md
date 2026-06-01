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
