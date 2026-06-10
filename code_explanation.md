# Code Explanation - Photo Electric II

## Overview
This workspace focus on validating XRT image quality and extracting standard sample features using an external library pipeline.

## External Library Integration
- **Modules used**: `preprocessing`. The dual-energy splitting logic has been localized to `utils_II.py` to eliminate external dependencies.

## Local Scripts

### `apd_acd_pipeline/` (Alvarez-Macovski & Spectrum Calibration Pipeline)
- **Purpose**: 包含双能光电效应 (APD) 与康普顿散射 (ACD) 特征计算、能谱重建以及矿石物理特征反解的中央管道。
- **Components**:
    - `get_apd_acd.py`: 物理计算与 SIRZ 系统标定工具。支持 Static（静态）、Dyn（动态）与 M1（连续能谱积分）三种算法提取 apd/acd。
    - `reconstruct_spectrum.py`: X 射线管出射有效能谱重建求解器，基于已知厚度梯度的阶梯块在各电压下的透射率进行 NNLS 反演（默认能量 bin 宽度为 10.0 keV，平滑因子为 0.08，结合 Duane-Hunt 渐进物理截止约束抑制高能多重峰与 0 值）。
    - `calculate_ores_properties.py`: 针对 114 块标样圆盘（`0325_input.pkl`），通过引入重建能谱与标定系统常数 $(K_1, g, \nu)$ 解算像素级 APD/ACD 特征，反算各矿石的代数有效原子序数 ($Z_e$) 和电子密度 ($\rho_e$)，并提供 `plot_ze_comparison` 绘制反算 $Z_e$ 与基于元素品位加权的理论有效原子序数 $Z_{eff}$ 的散点对比图。

### `contour_app/` (Desktop Application)
- **Purpose**: A standalone GUI tool for interactive contour extraction and parameter tuning.
- **Components**:
    - `app.py`: The main GUI entry point built with `CustomTkinter`. Handles the interface layout, event loops, and interactive widgets (sliders, switches).
    - `processor.py`: The bridge between the GUI and `utils_II.py`. Wraps high-level image processing tasks like loading, splitting, and preview generation to ensure the GUI remains responsive.
- **Workflow**:
    1. User selects an image.
    2. The app splits the dual-energy image and displays the low-energy channel.
    3. User adjusts sliders (Threshold, ROI, Scaling) to see real-time updates of detected contours.
    4. User saves the final contoured image as a PNG.

### `equivalent_thickness_calc.py`
- **Purpose**: A standalone utility script to calculate the equivalent ore thickness corresponding to standard metal step wedges (Cu, Fe, Al) under dual-energy XRT.
- **Functions**:
    - `get_mineral_properties(energy_kev)`: 根据给定能量（keV）动态从 NIST 获取元素的质量衰减系数，并计算黄铜矿、黄铁矿和脉石的物理属性。
    - `calc_ore_properties(cu_grade_percent, py_grade_percent, porosity, props)`: 
        - 计算给定品位下的矿石的理论密度和质量衰减系数。
        - 内部构建了斑岩型铜矿的物理模型，考虑了黄铜矿(CuFeS2)、伴生黄铁矿(FeS2)和脉石(SiO2)的比例。
    - `run_thickness_analysis(energy_kev, cu_grade, py_grade, porosity)`: 
        - 主分析函数，输出不同能量下的厚度转换表和倍率。
- **Workflow**:
    1. 从 `get_mu_from_nist_new` 动态获取物理常数 ($\mu_m$)。
    2. 计算矿石混合物的综合衰减特性。
    3. 输出金属与矿石、以及金属与金属之间的等效厚度表。

### `get_mu_from_nist_new.py`
- **Purpose**: NIST 官方 X 射线质量衰减系数爬取与插值工具。
- **Functions**:
    - `fetch_mu_rho(element_symbol)`: 爬取指定元素的 NIST 原始衰减数据。
    - `get_mu_rho_interpolated(element_symbol, target_energies_keV)`: 在对数空间执行 Pchip 插值，获取指定能量下的 $\mu/\rho$。
    - `get_energy_from_mu(element_symbol, mu_list)`: 
        - **核心功能**: 根据输入的线衰减系数 $\mu$ (cm^-1) 列表，反向推算出对应的平均 X 射线能量 (keV)。
        - 参数 `element_symbol`: 物质种类 ('Fe', 'Al', 'Cu')；`mu_list`: 线衰减系数列表。

### `get_apd_acd.py`
- **Purpose**: 双能 X 射线光电效应 (APD) 与康普顿散射 (ACD) 物理特征的像素级计算工具，并提供标准阶梯样品的特征演化分析与可视化管线。
- **Functions**:
    - `calculate_apd(low, high, I0_low=204.293, I0_high=204.199)`: 
        - 计算光电效应乘厚度特征 $apd = a_p \cdot d$。
        - 参数 `low`/`high`: 低能与高能通道灰度像素数组；`I0_low`/`I0_high`: 低能与高能通道的入射背景灰度参考值。
    - `calculate_acd(low, high, I0_low=204.293, I0_high=204.199)`: 
        - 计算康普顿散射乘厚度特征 $acd = a_c \cdot d$。
        - 参数同 `calculate_apd`。
    - `calculate_Zeff(low, high, I0_low=204.293, I0_high=204.199)`: 
        - 采用比例法计算近似有效原子序数特征 $Z_{eff} = apd / \mu_H$。
        - 参数同 `calculate_apd`。
    - `calculate_Ze(low, high, I0_low=204.293, I0_high=204.199)`: 
        - 计算代数有效原子序数特征 $Z_e = k \cdot (apd / acd)^n$。
        - 参数同 `calculate_apd`。
    - `calculate_mu_H_d(low, high, I0_low=204.293, I0_high=204.199)`: 
        - 计算高能对数衰减厚度值 $\mu_H \cdot d = \ln(I0\_high / high)$。
        - 参数同 `calculate_apd`。
    - `calculate_apd_acd_mono(T_L, T_H, E_L, E_H)`:
        - 使用双单能近似代数公式计算 APD 和 ACD 特征。
        - **参数解释**：
            - `T_L` (float 或 np.ndarray): 低能通道透射率。
            - `T_H` (float 或 np.ndarray): 高能通道透射率。
            - `E_L` (float): 低能等效能量，单位 keV。
            - `E_H` (float): 高能等效能量，单位 keV。
    - `solve_apd_acd_nonlinear(T_L, T_H, S_L, S_H, energies_keV)`:
        - 结合能谱分布通过 root 寻优求解 Alvarez-Macovski 积分方程组，反解出 APD 与 ACD。
        - **参数解释**：
            - `T_L`, `T_H` (float 或 np.ndarray): 实测低能与高能透射率。
            - `S_L`, `S_H` (np.ndarray): 归一化低能与高能有效能谱分布。
            - `energies_keV` (np.ndarray): 对应的能量网格数组，单位 keV。
    - `run_step_apd_acd_analysis(include_0331=True, plot_details=True, output_dir='results/thickness_decoupling/apd_acd_analysis')`: 
        - 核心分析调度主函数。批量启动物理算子解算、2x2物理剖析大图生成、以及多电压 Bulk 系数依赖曲线的生成，并将最终的统计序列持久化为 JSON。
        - 参数 `include_0331`: 0.6mm 下是否并入历史的 0331 阶梯数据点；`plot_details`: 是否绘制各电压/滤片组合的 2x2 深度大图；`output_dir`: 指定的落地输出目录。
    - `_load_and_process_step_pixels(filepath, thickness_arr, I0)`:
        - 阶梯标样像素级特征计算子模块。加载单个材料在指定电压/滤片下的像素数据，进行盲元与边界异常过滤，解算像素级 APD/ACD 特征，统计其均值、标准差，并**保留高精度的像素级原始物理特征数组**。
        - **参数解释**：
            - `filepath` (str): 阶梯标样像素序列 pkl 文件的存储路径。
            - `thickness_arr` (np.ndarray): 包含该样品各级阶梯实际物理厚度 (mm) 的一维数组。
            - `I0` (float): 入射通道的背景参考对数灰度参考值（16位默认为 52428.0）。
        - **返回值**：
            - `list`: 包含各厚度级阶梯物理属性统计量（如均值、标准差）和原始像素级一维数组（如 `apd_raw`, `acd_raw`, `Zeff_raw`, `Ze_raw`）的字典列表。
    - `_plot_detailed_profiling(voltage_data, f_type, voltage, colors, save_path)`:
        - 2x2 物理剖析多图绘制子模块。绘制单电压/滤片组合下，三材料的 APD vs 厚度、ACD vs 厚度、特征空间轨迹、以及 $Z_{eff}$ vs 厚度的剖析关系图（包含宏观均值趋势及标准差误差棒曲线，保持原图结构不受污染）。
        - **参数解释**：
            - `voltage_data` (dict): 阶段统计字典。
            - `f_type` (str): 滤片类型。
            - `voltage` (str): 电压。
            - `colors` (dict): 颜色配置映射。
            - `save_path` (str): 图表落地磁盘路径。
    - `_plot_calibration_fit(voltage_data, K1, g, nu, step_index, f_type, voltage, save_path)`:
        - 绘制并保存对数线性拟合关系图 (ln(ap/ac) vs ln(Z))。
        - **参数解释**：
            - `voltage_data` (dict): 阶段统计数据字典。
            - `K1` (float): 电子密度标定常数。
            - `g` (float): 有效原子序数校准系数。
            - `nu` (float): 有效原子序数幂次系数。
            - `step_index` (int): 阶梯标样厚度索引。
            - `f_type` (str): 滤片厚度。
            - `voltage` (str): 管电压。
            - `save_path` (str): 图片保存路径。
    - `_plot_apd_acd_histograms(voltage_data, f_type, voltage, colors, save_path)`:
        - APD & ACD 像素级原始分布直方图独立绘制子模块。**将各材料（Cu, Fe, Al）在所有厚度阶梯下的全部有效像素的原始计算物理特征分别提取出来，绘制为独立的 $apd$ 直方图与 $acd$ 直方图大图**，保存为独立的分析结果图，完全不修改、不污染原来的 2x2 剖析折线大图。
        - **参数解释**：
            - `voltage_data` (dict): 包含像素级原始物理特征数组的阶段汇总数据字典。
            - `f_type` (str): 滤片厚度配置描述字符串。
            - `voltage` (str): 管电压描述字符串。
            - `colors` (dict): 直方图着色的材料色彩配置映射字典。
            - `save_path` (str): 独立的直方图分析大图保存路径。
    - `_plot_coefficient_dependence(coeff_summary, f_type, colors, save_path)`:
        - 随管电压变化的 bulk 物理系数依赖曲线绘制子模块。
        - 参数 `coeff_summary`: 系数阶段汇总字典；`f_type`: 滤片厚度识别符；`colors`: 材料颜色字典；`save_path`: 图表落地磁盘路径。
- **Workflow**:
    1. 动态过滤各阶梯的核心像素，剔除饱和与非有效区间。
    2. 计算像素级 $apd$ 与 $acd$ 独立物理贡献，规避非均匀介质在宏观均值上的 Jensen 不等式误差。
    3. 生成 2x2 深度分析大图，从 $apd$/$acd$ 与物理厚度 $d$ 的严格线性映射、特征空间轨迹、以及 $Z_{eff}$ 的硬化漂移对系统进行全方位评估。
    4. 提取各电压下的 bulk 材料衰减比值，汇总为合并滤片（0.6mm + 1.2mm）随管电压变化的能量相关特性图，并将特征数据集序列化为 JSON 导出。

### `utils_II.py`
- **Purpose**: Local wrapper for dual-energy XRT processing.
- **Functions**:
    - `split_dual_xray_image`: Locally implemented splitting logic with integrated geometric distortion correction for high-energy images.
    - `compute_R`: Calculates the R-value image (float).
    - `get_step_pixels_list`: Extracts individual arrays for each of the 10 step cores.
    - `get_disk_core_info`: Calculates core pixels and boundary for centroid-based contour scaling.
    - `get_inner_95_pixels`: Helper for general 95% area erosion.
    - `classify_contour`: Classifies the contour into ['block', 'step_sample', 'disk', 'ore'] based on geometric properties.
        - 参数 `cnt`: 轮廓；`ellipse_limit`: 椭圆率阈值；`box_image_low`: 裁剪后的低能图像块；`pixels_low`: 低能像素集合；`all_type`: 强制指定的物体类型 (如 'ore', 'disk')，若提供此参数则跳过自动几何形状推断。
        - 返回: `label` (类别字符串) 和 `meta` (包含提取的统计像素和边界框信息)。
    - `get_bricks`: Main pipeline for batch feature extraction. Supports configurable thresholding methods (`th_type`) and vertical scaling (`vscale`).
        - 参数 `path`: 图像路径 (str)；`roi`: 感兴趣区域 [y1, y2, x1, x2] (list)；`th_val`: 阈值 (int)；`th_type`: 阈值类型 (cv2)；`fx`/`fy`: 畸变校正系数 (float)；`vscale`: 纵向缩放系数 (float, 默认1.0)；`vscale_interp`: 缩放插值方法 (cv2, 默认INTER_LINEAR)；`reverse_sort`: 是否反转排序结果 (bool, 默认False)。
    - `get_bricks_watershed`: Enhanced pipeline using Distance Transform and Watershed algorithm. Supports the same scaling parameters as `get_bricks`。
        - 参数同 `get_bricks`；并内置了高度拆分阈值的动态自适应逻辑：对于文件名包含 '270us' 的长曝光拉伸图像，其分割判定高度阈值自动设为 `1000` 像素，而普通配置图像（如 125us）默认沿用 `800` 像素的分割上限。
    - `check_step_gradient`: Analyzes row-wise mean gradients using Pearson Correlation and dynamic thresholds.
    - `warp_straighten`: Aligns tilted objects using perspective transforms.
    - `get_10_step_means`: Multi-axis core sampling (80% width, 60% height).
    - `save_contour_data`: Organized saving of warped images and pickle data.
    - `plot_ul_cdf`: Calculates and plots the cumulative distribution (CDF) for low-energy attenuation.
    - `plot_ore_grayscale_distribution`: 绘制每块矿石在不同电压下的高低能灰度值分布直方图，并将所有电压的子图整合进一张大图中保存。灰度值上下限（`x_min` 和 `x_max`）默认开启自适应动态判定（基于 0.5% 和 99.5% 分位数），且在每个子图的 legend 中标注 `=0` 和 `<2560`（16位）/ `<10`（8位）的低灰度/死像素所占的百分比。
        - 参数 `oid` (str): 矿石唯一标识符；`ft` (str): 滤片或数据集配置；`voltage_data` (dict): 各电压对应的有效像素对字典；`save_path` (str/Path): 保存路径；`x_min` (float/int, 可选): 横坐标下限，为 None 时自动动态计算；`x_max` (float/int, 可选): 横坐标上限，为 None 时自动动态计算。
    - `get_ore_lower_threshold`: 统一的矿石灰度过滤阈值集中管理器。
        - 参数 `is_ore` (bool): 当前是否为矿石数据；`v_max` (int/float): 图像最大可能灰度值（255 或 65535）；`ratio` (float, 可选): 最低允许透过率比例，默认为 0.05 (即 5%)。
        - 返回: `lower_th` (int, 动态计算得到的过滤下限值)。



### `pick_ores.py`
- **Purpose**: 用于矿石轮廓提取与多模型分类结果的可视化标注。为了保证脚本的完全独立性与极高的执行效率，本脚本已直接完整集成了原 `utils_II.py` 中的关键几何提取和畸变校正算法（如 `get_bricks`, `get_bricks_watershed` 等），**已完全剥离对外部模块 `utils_II.py` 的导入依赖**。支持接收摆放矿石的 dual-energy 或低能图像，提取出所有矿石轮廓。本模块在图像最显眼位置的背景框内，使用 OpenCV 绘制极高对比度的数字类别（**`1`**、**`2`**、**`3`**、**`4`**）代表第一至第四类，并在顶部附带 1-based 序号 `f"#{ore_id}"` 进行精确追踪。支持以下三种分类划分机制：
    - **方法 1 (二分类配对方法)**：基于两模型预测的 0/1 类别（1代表精矿，0代表废矿），组合为四类标签（`11` -> 1，`10` -> 2，`01` -> 3，`00` -> 4）。
    - **方法 2 (原子序数 $Z_{eff}$ 多档分类与加权决策方法)**：输入两个模型预测的 $Z_{eff}$ 浮点数值，首先将各自在图中的预测值分别归一化到 `[0, 1]` 区间，再等分为四档 (`T1` 至 `T4`)。当两模型分档不一致时，采用已归一化值的加权和 $Z_{norm, weighted} = 0.6 \cdot Z_{norm, 1} + 0.4 \cdot Z_{norm, 2}$ 直接判定所属档位，输出对应的数字类别标签。
    - **方法 3 (基于静态矿石编号的方法)**：直接根据用户给出的前 25%、25%-50%、50%-75% 和 倒数 25% 的 1-based 静态矿石编号集合将图中的矿石轮廓分类到四类档位（分别对应第一至第四类并绘制 `1` 到 `4`），内置动态兜底 Fallback。
- **Functions**:
    - `label_ores`: 主入口函数，负责调用集成的几何分割算法提取轮廓并进行半透明与彩色描边、高对比度文本框绘制。
        - **参数解释**：
            - `image_path` (str): 输入摆放矿石的图片文件路径 (可以是 8 位或 16 位 stacked dual-energy 图像)。
            - `model1_results` (list 或 np.ndarray): 
                - 方法 1 下：模型 1 给出每个提取出的矿石轮廓的二分类预测结果 (精矿为 1，废矿为 0)。
                - 方法 2 下：模型 1 预测每个矿石的有效原子序数 $Z_{eff}$ 浮点数值。
                - 方法 3 下：不使用模型数据，可传入空列表 `[]`。
            - `model2_results` (list 或 np.ndarray): 
                - 方法 1 下：模型 2 给出每个提取出的矿石轮廓的二分类预测结果 (精矿为 1，废矿为 0)。
                - 方法 2 下：模型 2 预测每个矿石的有效原子序数 $Z_{eff}$ 浮点数值。
                - 方法 3 下：不使用模型数据，可传入空列表 `[]`。
            - `output_path` (str, 可选): 标注后的图像保存路径。如果为 None，则不进行磁盘写入，仅返回处理后的图像。
            - `roi` (list, 默认 `[200, -1, 600, 800]`): 感兴趣的区域 `[y1, y2, x1, x2]`。若 y2 或 x2 为 -1 代表提取到图像边缘。
            - `th_val` (int, 默认 `175`): 图像二值化时的灰度阈值，用于检测矿石轮廓。
            - `use_watershed` (bool, 默认 `False`): 是否启用基于分水岭算法的 `get_bricks_watershed` 提取轮廓。如果为 False，则使用传统的 `get_bricks`。
            - `fx` (float, 默认 `0.99`): 高能图像几何校正的横向比例参数。
            - `fy` (float, 默认 `1.0`): 高能图像几何校正的纵向比例参数。
            - `sort_direction` (str, 默认 `'y'`): 轮廓排序方向，'y' 代表列优先 (从上到下，从左到右)，'x' 代表行优先。
            - `max_colwidth` (int, 默认 `35`): 排序时的横向或纵向聚类容差距离。
            - `vscale` (float, 默认 `1.0`): 纵向缩放系数。
            - `alpha` (float, 默认 `0.4`): 半透明填充遮罩 (mask overlay) 的不透明度，范围在 0.0 到 1.0 之间。
            - `method` (int, 默认 `3`): 选择的分类标记方法：`1` 代表二分类配对；`2` 代表原子序数归一化多档及加权判定；`3` 代表基于 1-based 静态矿石 ID 列表的分类标记方法。
            - `reverse_sort` (bool, 默认 `False`): 是否对提取的每一档轮廓进行逆向排列。对于 Yinshan (银山数据行排列，X 轴从右到左) 必须设为 `True`，对于其他数据集默认为 `False`。
        - **返回值**：
            - `(labeled_img, cnt_filtered)` 元组。其中 `labeled_img` 为同样大小的 BGR 彩色标注图像，`cnt_filtered` 为排序后的过滤轮廓列表。
- **Workflow**:
    1. 调用 `utils_II.get_bricks` 或 `utils_II.get_bricks_watershed` 提取图像中的矿石轮廓并排序。随后对底图进行水平翻转（`cv2.flip(labeled_img, 1)`），并同步镜像翻转所有已提取轮廓的 X 坐标，以响应左右反转渲染要求并确保绘图与文本的对齐和文字正向显示。
    2. 对分类预测结果（二分类或 $Z_{eff}$ 数值）进行对齐与长度容错处理。若预测列表长度少于提取的轮廓，在方法 1 中以二分类 `0` 进行填充，在方法 2 中以预测序列的均值 (如空则为 `12.0`) 进行均值填充，以保证多图像大批量处理时的稳定性。
    3. 根据两模型组合结果或 Zeff 分档映射为 Class 1 (绿色), Class 2 (橙色), Class 3 (蓝色), 或 Class 4 (红色) 4 种级别。
    4. 采用融合算法在原图上叠加透明度为 `alpha` 的精美色块，用不透明边界包围矿石。在矿石几何中心放置高对比度黑底背景的白色类别文本（显示为等级数字 `1`、`2`、`3`、`4`）；同时在各矿石外接矩形（Bounding Box）的正上方空隙外 `(y - id_h - 8)` 绘制配有微型暗底背景和亮白文本的 1-based 序号 `f"#{ore_id}"`（若最顶部越界则置于矿石内部上沿），完全杜绝重叠与遮挡，清晰美观。
    5. 使用 `cv2.imencode` 写入保存，防止在 Windows 系统中因非 ASCII 或中文路径出错。
用户只需运行 `python pick_ores.py` 即可在 `results/` 目录下得到渲染后的图片。


### `extract_sample_values.py`
- **Purpose**: Batch analysis of standard samples using relative paths.
- **Workflow**:
    1. Finds target TIF files in relative data directories. (Specifically for 0409 datasets, it selectively and exclusively reads files with the suffix '_cropped.tif' as produced by `crop_TYM.py` to ensure only pre-cropped images are analyzed; other datasets use standard discovery rules).
    2. Synchronized `get_bricks` parameters.
    3. Performs type-specific refined analysis (Disk scaling, Step 10-segmentation).
    4. Outputs categorized results to `results/`.
    5. Visualizes bounding rectangles (`minAreaRect`) used for ROI extraction directly on the output contoured images.
    6. Masks `high_low_images` during extraction to keep only pixels inside the contour, replacing the background with the maximum grayscale value (`255` or `65535`).
    7. Extracted step sample transition zones (the 3rd thickness mutation boundaries +/- 5 rows, with 10% horizontal margins) and persisted them separately to `_transition.pkl` files for physical boundary transition analysis.


### `extract_0429_RaySov_from_mask.py`
- **Purpose**: An auxiliary script for the 0429 dataset designed to extract precise pixel values strictly using manually drawn `.png` masks, skipping algorithmic geometric contouring entirely.
- **Workflow**:
    1. Traverses the `ore` and `steps` subdirectories within `20260429_mask_generated`.
    2. Pairs raw `.tif` files with their `.png` masks based on identical filename stems.
    3. Examines the `.tif` filename: applies `normalize_image` to scale up to 80% of 65536 if the file is labeled `orig`; leaves pixels fully raw if labeled `user`.
    4. Slices both the image and the mask perfectly in half down the horizontal center (Left = Low Energy, Right = High Energy).
    5. For ores, extracts the mean/std for the entire global masked area.
    6. For steps, mathematically sorts the 10 separate step mask contours **bottom-to-top** (ordering from **Thin to Thick**) and computes the LE and HE values per individual step.
    7. Uses `save_contour_data` to persist results in `.pkl` and `.png` formats, ensuring compatibility with the downstream analysis pipeline.
    8. Writes all data to a compiled `extracted_values_summary.csv`.

### `extract_0429_RaySov.py`
- **Purpose**: Specialized extraction script for the 20260429 RaySov step sample dataset.
- **Workflow**:
    1. Reads 16-bit `.tif` images directly from the `阶梯` directory, applying automatic English translation to filenames (e.g., `Fe_step`, `calib`) to avoid OpenCV decoding errors.
    2. Identifies images as step samples unconditionally if the filename contains "step" or "阶梯", bypassing generic geometric classification.
    3. Extracts the global contour from the full `0-511` low-energy region.
    4. Computes the true bounding box using `find_step_sample_corners` to completely ignore wider bottom brackets.
    5. Maps the mid-boundary line (`X=256`) into the straightened step sample, slices it perfectly into 0.6mm (left) and 1.2mm (right) zones, and runs 10-step extractions on both independently.
- **Functions**:
    - `find_step_sample_corners(cnt)`: Parses contour polygon segments to isolate the two longest vertical edges, identifying the true inner corner coordinates and discarding the bracket beneath.
    - `warp_step_sample(image, corners, width_ratio)`: Perfectly straightens the targeted sample using the identified corners.
    - `get_bricks_raysov_global(path, read_mode, roi, th_val, th_type)`: The main extraction pipeline integrating the above steps and mapping the splitting boundary across raw and straightened domains.
    - `get_bricks_raysov(path, filter_type, read_mode, roi, th_val, th_type, fx, fy, sort_direction, max_colwidth, vscale, vscale_interp)`: Custom extraction pipeline handling the unique 4-zone layout of the 0429 RaySov detector.
        - 参数 `path`: 图像路径 (str)。
        - 参数 `filter_type`: 滤片类型 ('0.6mm' 或 '1.2mm')，用于定位图像中的对应区域 (str)。
        - 参数 `read_mode`: 读取模式 ('16bit' 或 '8bit')，决定了如何读取图片并是否应用自动归一化处理 (str)。
        - 参数 `roi`: 感兴趣区域 [y1, y2, x1, x2] (list)。
        - 参数 `th_val`: 二值化阈值 (int)。
        - 参数 `th_type`: OpenCV阈值类型 (cv2)。
        - 参数 `fx, fy`: 高能图像畸变校正系数 (float)。
        - 参数 `sort_direction`: 轮廓排序方向 (str)。
        - 参数 `vscale`: 纵向缩放系数 (float)。
        - 参数 `vscale_interp`: 缩放插值方法 (cv2)。
        - 返回: 包含像素列表、绘制轮廓图像、低/高能区域等数据的元组。

### `decouple_thickness.py`
- **Purpose**: Fits a multivariate polynomial regression model to decouple thickness from dual-energy XRT signals, mapping them directly to Equivalent Atomic Numbers (Z).
- **Functions**:
    - `extract_feature_HL_ratio(L, H)`: Generates input features `[H/L, (H/L)^2]` to model the ratio-based non-linear transformation.
        - 参数 `L`: 低能像素值 (ndarray)；`H`: 高能像素值 (ndarray)。
        - 返回: 包含两列 `[ratio, ratio^2]` 的特征矩阵。
    - `extract_feature_poly(L, H)`: Generates simple base features `[L, H]` for feeding into `PolynomialFeatures(2)` which creates `[L, H, L^2, H^2, LH]`.
        - 参数 `L`: 低能像素值 (ndarray)；`H`: 高能像素值 (ndarray)。
        - 返回: 包含 `[L, H]` 的特征矩阵。
- **Workflow**:
    1. Loads step-sample data from `results/20260331/pixel_values/` for Cu, Fe, and Al.
    2. Assigns target atomic numbers: Cu=29, Fe=26, Al=13.
    3. Iteratively fits Ridge regression models for three thickness scenarios (Al 6/8/10 steps) to evaluate how step inclusion impacts model robustness.
    4. **Visualization Engine**: Evaluates the model accuracy across all thickness levels and voltage configurations.
    5. **Output**: `results/thickness_decoupling/z_decouple/[DATE]/`: Contains output charts and parameter logs for a specific dataset.
    6. `results/thickness_decoupling/z_decouple/[DATE]/fitting_parameters.txt`: Centralized log of all regression coefficients.
    7. **Multi-Model Comparison**: Evaluates three distinct regression strategies (Model 1: Linear Ratio, Model 2: Polynomial DE, Model 3: R-Value Physics-based) in a **5x3 grid**:
        - **Row 0**: Global overview showing separate KDE plots for Model 1, 2, and 3.
        - **Row 1**: Model 1 distribution breakdown per material.
        - **Row 2**: Model 2 distribution breakdown per material.
        - **Row 3**: Model 3 distribution breakdown per material.
        - **Row 4**: Systematic bias analysis (Mean Predicted Z vs Step Index) comparing all three models.
    8. **Step-Wise Visualization**: KDE plots include granular distributions per thickness level.
    9. **Mean Bias Analysis**: Row 4 subplots visualize the drift in mean prediction across physical thickness steps for M1, M2, and M3.
    10. **Model 3 Physics**: Uses the $R$-value formula $R = \ln(I_{0,L}/L + 5) / \ln(I_{0,H}/H + 20) for robust feature extraction.
    11. **Accuracy Summary**: Generates `Z_accuracy_summary_comparison.png` comparing precision (std) of all three models.
    12. **Parameter Logging**: Automatically archives all fitted coefficients and intercepts to `fitting_parameters.txt` in the timestamped output directory.
    13. **Data Traceability**: `output_dir` is now dynamically synchronized with the `input_dir` date string to prevent results from being overwritten when testing different datasets.
    14. **Optimization**: Incorporates `StandardScaler` with unscaling logic for physically accurate formula display.

### `calculate_mu_m.py`
- **Purpose**: Calculates the mass attenuation coefficient ($\mu_m$) for standard samples (Cu, Fe, Al) using the exponential attenuation law.
- **Workflow**:
    1. Loads mean pixel data and solves for $\mu_m = -\ln(I/I_0) / (\rho \cdot t)$.
    2. Uses defined densities: Cu=8.96, Fe=7.87, Al=2.70 g/cm³.
    3. Analyzes the consistency of $\mu_m$ across different thickness steps to verify beam hardening effects.
    4. Generates comparison plots (`mu_m_analysis.png`) across voltages.

### `analyze_energy_hardening.py`
- **Purpose**: Implements a thickness-dependent energy back-calculation method to find the incident effective energy.
- **Functions**:
    - `get_energy_from_mu_rho(element_symbol, mu_rho_list, data_dir)`: 
        - **核心逻辑**: 根据质量衰减系数 $\mu_m$ ($cm^2/g$) 在对数空间反向推算光子能量 ($keV$)。
        - 参数 `element_symbol`: 元素符号；`mu_rho_list`: $\mu_m$ 列表。
    - `analyze_hardening()`: 
        - 对每个材质和电压，计算各阶梯的有效能量并绘制 $E$ vs $Thickness$ 曲线。
        - 执行线性拟合并外推到 $t=0$，从而获得不受硬化影响的入射光束等效能量 $E_0$。
- **Workflow**:
    1. 提取阶梯像素值并计算逐点 $\mu_m$。
    2. 利用 NIST 逆插值将 $\mu_m$ 转换为能量 $E$。
    3. 拟合硬化趋势线并保存 `hardening_summary.json` 与对比图。
    
### `reconstruct_spectrum.py`
- **Purpose**: 从阶梯样在不同厚度下的吸收差异反推出射 X 射线低能/高能通道有效能谱，支持方法一（正则化增广 NNLS）与方法二（相邻差值映射法），并利用重建谱求解 $apd$ 与 $acd$ 物理特征以解耦能谱硬化。
- **Functions**:
    - `fkn(E_keV)`:
        - **核心逻辑**: 计算并返回给定能量下的无量纲 Klein-Nishina 康普顿散射截面系数。
        - **参数解释**: `E_keV` (float 或 np.ndarray)：光子能量，单位 keV。
    - `get_linear_attenuation(element_symbol, energies_keV, density)`:
        - **核心逻辑**: 从 NIST 数据库插值质量衰减系数 $\mu/\rho$，并乘以密度获取各能量的线衰减系数 $\mu$ ($cm^{-1}$)。
        - **参数解释**: 
            - `element_symbol` (str): 材质化学符号（'Al', 'Fe', 'Cu'）。
            - `energies_keV` (np.ndarray): 能量网格数组，单位 keV。
            - `density` (float): 材质密度，单位 $g/cm^3$。
    - `load_transmission_data(f_type, voltage, data_dir, I0=52428.0, cu_fe_max_steps=10)`:
        - **核心逻辑**: 载入对应滤片与电压下的阶梯样 pkl 灰度值，过滤死像素或饱和区，计算低能与高能通道的实测透射率 $T = I/I_0$。可通过限制重材质（Cu, Fe）的最高阶梯数来规避厚端穿透不足的噪声数据影响。
        - **参数解释**:
            - `f_type` (str): 滤片厚度描述字符串（如 '0.6mm', '1.2mm'）。
            - `voltage` (str): 管电压描述字符串（如 '200kV'）。
            - `data_dir` (str): pkl 数据包所在的存储物理目录。
            - `I0` (float): 空载入射背景对数灰度参考值，默认 16 位下为 52428.0。
            - `cu_fe_max_steps` (int): 限制铜、铁阶梯加载的最大层数（默认 10 层，可设为 1, 3, 5, 7 等以规避探测器透不过的厚端）。
    - `build_system_matrix(materials, thicknesses_cm, energies_keV)`:
        - **核心逻辑**: 构造能谱离散前向投影的系统映射矩阵 $\mathbf{A}$，使得 $\mathbf{A} \mathbf{S} \approx \mathbf{T}$。
        - **参数解释**:
            - `materials` (list of str): 每个测量阶梯对应的材料列表（如 `['Al', ..., 'Cu']`）。
            - `thicknesses_cm` (np.ndarray): 每个测量阶梯的实际物理厚度，单位 cm。
            - `energies_keV` (np.ndarray): 离散重建能谱的能量仓中心坐标数组，单位 keV。
    - `reconstruct_channel_spectrum(A, T, energies_keV, lambda_val=0.005, gamma=20.0, beta=10.0)`:
        - **核心逻辑**: 【方法一】使用增广正则化非负最小二乘 (NNLS) 求解单通道归一化有效出射谱。结合二阶差分平滑约束 $\lambda$，归一化约束 $\gamma$，以及低能与高能截止边界归零约束 $\beta$。
        - **参数解释**:
            - `A` (np.ndarray): 系统的正向透射投影矩阵，大小 (N, M)。
            - `T` (np.ndarray): 实测透射率向量，大小 (N,)。
            - `energies_keV` (np.ndarray): 重建能量网格数组。
            - `lambda_val` (float): 二阶差分平滑正则化惩罚因子。
            - `gamma` (float): 强迫能谱总和等于 1 的归一化约束权重。
            - `beta` (float): 强迫两端截止点强度归零的边界权重。
    - `reconstruct_channel_spectrum_method2(step_info_list, energies_keV, voltage_kv, channel='low')`:
        - **核心逻辑**: 【方法二】基于相邻阶梯透射率差值 $\Delta T_j = T_j - T_{j+1}$ 映射到其敏感带通峰值能量 $E^*_j = \mu^{-1}(\ln(d_{j+1}/d_j)/(d_{j+1}-d_j))$ 的能谱估算与 PCHIP 插值归一化算法。
        - **参数解释**:
            - `step_info_list` (list of dict): 各阶梯 of 物理信息与测量透射率明细字典列表。
            - `energies_keV` (np.ndarray): 重建能谱的目标能量网格数组。
            - `voltage_kv` (float): 射线管的最大管电压（能量仓上限），单位 kV。
            - `channel` (str): 电能通道，可选 `'low'` 或 `'high'`。

### `predict_disk_Z.py`
- **Purpose**: Fits Model 2 (multivariate polynomial) and generates heatmaps and histograms for predicted atomic numbers (Z) of disks.
- **Workflow**:
    1. Parses Model 2 parameters from `fitting_parameters.txt` (which tracks the training dataset date).
    2. Uses a decoupled `TEST_DATA_DIR` to allow applying the previously trained model directly to new datasets (e.g., `20260325_yinshan`).
    3. Conditionally constructs file paths (`1_98_position_3_{voltage}_ore_{d_id}`) for Yinshan ore data to dynamically switch between standard disk calibration and real ore validation.
    4. Loads `disk_grades` from an external JSON config (`E:\multi_source_info\data_dir\disk_grades.json`) based on the test dataset date.
    5. Predicts Z for up to 114 disks using Low and High energy signals (from both `.pkl` and `.png`).
    6. Generates robustness statistics (mean, std) by filtering extreme 1% outliers, and plots Predicted Z vs Theoretical $Z_{eff}$ using `plt.scatter` for clear mean-value regression analysis.

### `update_disk_grades_20260325.py`
- **Purpose**: Automates the extraction of disk assay grades (Cu, Fe, S) from raw data CSVs and updates the centralized `disk_grades.json` configuration file.
- **Workflow**:
    1. Reads the `2026_矿石数据采集-0325_银山铜矿_114.csv` file using `utf-8-sig` or `gbk` encoding.
    2. Skips the first three header rows and iterates over the 114 disk records.
    3. Extracts the Disk ID (Column 0) and the "院里化验" assay results for Cu (Col 10), Fe (Col 11), and S (Col 12).
    4. Dynamically appends or updates the `"20260325"` key in `E:\multi_source_info\data_dir\disk_grades.json`.

### `compare_zeff_methods.py`
- **Purpose**: A standalone script to compare and visualize the differences between various theoretical calculation methods for the effective atomic number ($Z_{eff}$).
- **Workflow**:
    1. Reads the assay grades (Cu, Fe, S) for the 98 Yinshan ore samples from `disk_grades.json`.
    2. Calculates $Z_{eff}$ using 5 different models/parameters:
       - **Baseline**: Mayneord formula with $S$ included (exp=2.94, z_base=11.0).
       - **No Sulphur**: Mayneord formula without $S$ (exp=2.94, z_base=11.0).
       - **Simple Weighted**: Direct mass-fraction weighted average (exp=1.0).
       - **High Exponent**: Mayneord formula with higher power (exp=3.5, z_base=11.0).
       - **Si Base**: Mayneord formula using pure Silicon as gangue (exp=2.94, z_base=14.0).
    3. Generates a 2x2 multi-subplot visualization (`Zeff_Methods_Comparison.png`) to show absolute value trends and relative prediction errors between the models.


### `compare_tube.py`
- **Purpose**: 用于比较不同探测管参数、曝光时间及电压下，阶梯样和矿石特征强度的脚本。现已集成阶梯标样 H-L 曲线与对数衰减的 2x3 深度分析以及质量衰减系数随电压变化的 slope_summary 汇总分析管线。
- **Functions**:
    - `load_any_dual_pixels(file_path, flip=False)`:
        - **核心逻辑**: 加载双能通道的像素值并检测是阶梯标样还是普通材质。若 `flip=True` 则自动执行步进反向反转 `low[::-1]`，用于将 thickest-to-thinnest 排列标样反转为标准的 thinnest-to-thickest 物理厚度轴。
        - 参数 `file_path`: 阶梯像素数据 pkl 的存储路径 (str)；`flip`: 是否对阶梯排序执行翻转 (bool)。
    - `find_linear_pts(x_pts, y_pts, label="")`:
        - **核心逻辑**: 在对数衰减曲线中，从最薄阶梯开始进行多阶拟合，并在 $R^2$ 衰减速率超过阈值时截断，自动定位最长的优秀物理线性衰减段数。
        - 参数 `x_pts`: 阶梯标样物理厚度数组；`y_pts`: 高/低能对数衰减值；`label`: 调试材质名称。
    - `perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, color_by_step=False, plot_mode='all', I0=204.0, raw_lims_global=None, log_lims_global=None)`:
        - **核心逻辑**: 绘制通用 2x3 的衰减特征与硬化特性大图。包含 H vs L 均值多项式拟合、Log-Log 拟合、Low/High 能与物理厚度的折线剖析（含 std 误差棒）以及对数衰减在自动线性区间内的斜率拟合，保存为图片。
        - 参数同 `fit_hl_curve_0429.py` 对应定义。
    - `generate_dataset_slope_summaries(dataset_name, output_subdir, global_raw_data, thicknesses)`:
        - **核心逻辑**: 对指定数据集的多电压标样均值进行全厚度提取，计算各阶厚度对应的低能/高能质量衰减系数 $\mu_L$ 和 $\mu_H$、L/H 比值、材质间比值（内嵌 Photoelectric & Compton 物理理论辅助线）以及 L/H 绝对偏差，绘制 10 阶厚度的 2x3 曲线大图并保存。
        - 参数 `dataset_name`: 数据集标识 (str)；`output_subdir`: 输出物理目录 (str)；`global_raw_data`: 收集的多电压衰减对数汇总字典 (dict)；`thicknesses`: 标样物理厚度坐标字典 (dict)。
    - `run_stepped_specimen_analysis()`:
        - **核心逻辑**: 针对 0331 及 0409 标样数据集的 H-L 特征拟合与衰减评估的中央调度总控模块。现已全面升级为统一使用高精度 16-bit 数据源（0407 数据集被注释禁用）。
        - **工作流**:
            1. **0331 (yinshan)**: 读取 `20260331_16bit` 像素数据包，处理 `140kV`、`160kV`、`180kV` 的三材质 10 阶数据，采用 `flip=False` 自适应加载。在 $I_0=52428.0$ （16-bit）下计算，并自动调用 `generate_dataset_slope_summaries` 输出 10 张质量衰减随电压的 `slope_summary` 大图。
            2. **0407 (home) [已注释禁用]**: 原为读取 `20260407_Sample_test_16bit` 像素数据包进行 `160kV` 10 阶数据 2x3 剖析。现已根据用户要求注释，暂不参与 analysis 流程。
            3. **0409 (TYM)**: 读取 `TYM_test_2_16bit` 像素数据包，使用高精度 16-bit 像素数据遍历 `160kV`、`180kV`、`200kV` 三个电压的 `125us` 曝光序列，标样材质统一映射为 `Al_step=6` (铝阶梯/铝块)、`Cu_step=8` (铜阶梯)、`Fe_step=9` (铁阶梯)。在 $I_0=52428.0$ 下执行 2x3 剖析。自动调用 `generate_dataset_slope_summaries` 输出 10 张质量衰减随电压的 `slope_summary` 大图。
- **Workflow**:
    1. 比较不同曝光时间（125us 与 270us）下的阶梯均值 and 直方图分布（使用 `results/TYM_test_16bit/pixel_values/` 路径下的 `step_sample_6`）。
    2. 对多块矿石在 125us 和 270us 之间的灰度相关性进行散点拟合，并绘制同种矿石分布直方图（选用共同现存的矿石编号 `[0, 1, 3]`）。
    3. 调用 `run_stepped_specimen_analysis()` 全面评估 0331（三个电压，16-bit）和 0409（三个电压 16-bit，125us，Cu/Fe/Al 精准对齐至统一索引映射 Al=6, Cu=8, Fe=9）的 2x3 物理衰减特征与 20 张 `slope_summary` 汇总大图，并将大图输出至 `results/Tube_comparison/comprehensive_fit/`。
    4. 比较不同曝光时间下第三个厚度跃迁突变处（transition）的跃迁散射特征，输出跃迁区域的均值对比图 `TYM_Exposure_Steps_Transition_means.png` 和高低能像素分布直方图 `_hist_low.png` / `_hist_high.png`。



### `fit_hl_curve.py`
- **Purpose**: Modular analysis pipeline that performs a 2x3 grid evaluation for both stepped samples and graded disks.
- **Output Structure**:
    - `results/thickness_decoupling/h-l-fit/steps/`: Contains Cu, Fe, and Al step-sample analysis.
    - `results/thickness_decoupling/h-l-fit/disks/`: Contains analysis for graded disks (IDs 9-17).
    - `results/thickness_decoupling/h-l-fit/ores/`: Contains mixed ore samples plotting against their IDs.
- **Functions**:
    - `perform_comprehensive_analysis(voltage, samples_dict, output_subdir, title_prefix, x_label, x_coords_dict, color_by_step=False, plot_mode='all')`: 通用 2x3 综合分析绘图函数。
        - 参数 `voltage`: 当前处理的电压值 (str, 例如 '140kV')。
        - 参数 `samples_dict`: 样本数据字典，格式为 `{mat_name: (L_list, H_list)}` (dict)。
        - 参数 `output_subdir`: 结果保存的子目录路径 (str)。
        - 参数 `title_prefix`: 图表标题前缀 (str)。
        - 参数 `x_label`: X轴标签 (str)。
        - 参数 `x_coords_dict`: X轴坐标字典，格式为 `{mat_name: ndarray}` (dict)。
        - 参数 `color_by_step`: 是否按照步进/圆盘的索引使用不同的颜色绘制散点图 (bool, 默认False)。
        - 参数 `plot_mode`: 散点图绘制模式 ('all' 或 'means')。
        - **特性**: 自动扫描数据确定自适应坐标轴限制，并强制每一行的 Y 轴（及相关 X 轴）保持一致以增强可比性。**现已支持自动识别 and 映射类别型 (Categorical) X 轴变量 (如矿石 ID)。同时，在分析矿石样品（Ore）时，为了排除硬件底噪、无效盲元和极高衰减所致异常像素，算法会自动排除灰度值小于 2560（16位，对应8位下的10）或小于 10（8位）的所有像素，并在图表大标题 (suptitle) 中添加注记说明。**
- **Workflow**:
    1. **Iterative Loading**: Dynamically loads data from single files (steps), multiple files (disks), and wildcard matching for ores. Now supports datasets like `20260401` up to 5 voltages (140kV-180kV), including logic to combine multi-image ore extracts (e.g. `1_20` and `21_38`) into a continuous 0-37 ID sequence.
    2. **Z_eff Integration**: Loads disk grades from `disk_grades.json` and calculates $Z_{eff}$ (using `utils_II.calculate_effective_z`) to use as the x-coordinate (grade proxy).
    3. **Group-Specific Calibration**: Applies different axis limits for step samples (low gray value, high attenuation) and disk samples (high gray value, low attenuation) to ensure visibility.
    3. **Total Visualization**: Disables subsampling to plot every valid pixel in H-L space.
    4. **Grid Metrics**: Computes means, standard deviations, log-attenuation, and adaptive linear ranges.

### `fit_hl_curve_0429.py`
- **Purpose**: 针对 2026-04-29 实验数据集的综合衰减曲线分析脚本，整合了 0.6mm 与 1.2mm 滤片的对比分析，并支持与 2026-04-01 矿石数据的跨数据集对比。
- **Configuration Parameters**:
    - `analysis_target`: 控制分析范围。可选值：`"step"` (仅处理阶梯样块), `"ore"` (仅处理矿石样块), `"all"` (处理全部)。
    - `mu_mode`: 衰减系数计算模式。可选值：`"mu"` (线衰减系数 $\mathrm{mm}^{-1}$，默认) 或是 `"mu_m"` (质量衰减系数 $\mathrm{cm}^2/\mathrm{g}$，厚度转换为厘米并除以材质密度 $\rho$)。
- **Key Features**:
    - **Global Scaling**: 自动扫描阶梯样块数据以确定统一的横轴(X轴)对齐范围(130kV-330kV)，并为每一个厚度分别计算自适应全局统一 Y 轴，确保跨电压、跨材质、跨滤片的图表具有严格的视觉可比性。
    - **核心流程**：
      - **动态获取 Y 轴上限**: `get_dynamic_ylim` 会自动根据每个厚度的结果动态放开/收紧坐标限制。
      - **阶梯厚度遍历循环**：不再使用单一的“拟合斜率”或“指定单层”，而是强制遍历从最薄到最厚（0-9）的所有阶梯层，分别计算各层对应的衰减特性。
      - **物质比例与差值分析**：计算同种物质的高低能衰减系数比例 ($L/H$) 及不同物质之间的衰减系数比例。
      - **单块矿石综合衰减特性多通道分析**：在单块矿石对比分析中，进行 $\mu$ 衰减值计算和散点多项式回归时，会应用中央控制的 `< 2560` (16位) / `< 10` (8位) 阈值过滤，并对 `ore_{oid}_comprehensive_analysis.png` 的大标题追加剔除注记；而专门收集来绘制 `ore_{oid}_grayscale_distribution.png` 灰度直方图大图的像素集合则保持为**完全未经过滤的原始像素**，以供完整、真实地反映探测器底噪和死盲元分布状态。
      - **联合阶梯衰减汇总图 (Combined Slope Summary)**: 遍历所有厚度后，调用 `plot_combined_slope_summaries` 函数，将 `0.6mm` (实线 `o-`) 与 `1.2mm` (虚线 `^-`) 的衰减及比值偏差曲线绘制在同一张 2x3 大图内。两滤片共用统一的 Y 轴动态范围以确保严格的物理可比性。材质对的理论比值线仅作为特定颜色的虚点线 (`:`) 绘制一次，以保持图面整洁。汇总图保存至 `combined/slope_summary_{mu_mode}/` 下。
      - **LaTeX 语法兼容**: 在 `mu_mode = 'mu_m'` (质量衰减系数) 模式下，将 LaTeX 双下标表示法规范化为单层下标（如 `\mu_{m, L}`），彻底避免 Matplotlib LaTeX 编译器在解析多重嵌套下标时抛出解析异常。
      - **独立 JSON 归档**：各层厚度对应的详细物理衰减 JSON 参数数据仍独立生成，保存为 `attenuation_slopes_{mu_mode}_{step_name}.json` 以实现详细数值的精确查阅。

### `read_raw.py`
- **Purpose**: 图像格式批量转换工具，用于递归地将 16-bit text 图像文件（以 tab 或空格分隔的整数灰度矩阵）以及 16-bit 二进制 RAW 格式图片文件无损地转换为标准的 16-bit PNG 图像。
- **Workflow**:
    1. 递归地遍历指定源目录下的子文件夹，查找匹配的文件。
    2. 对于每个匹配的文件，应用文件名关键字过滤。
    3. 自动解析和提取 `.txt`（二维形状自保留）与 `.raw`（从文件名解析尺寸或使用参数）的图像像素。
    4. 保留原有的目录结构层级镜像输出至 `converted_pngs` 目录中。
- **Functions**:
    - `convert_txt_and_raw_to_png(src_dir, dst_dir=None, filter_keyword="校准后", width=1024, height=1024)`:
        - **核心逻辑**：自动递归扫描源目录及所有子文件夹下的 `*.txt` 和 `*.raw` 文件。
            - **路径层级保留**：在输出的目标文件夹 `dst_dir` 下镜像生成相同的子目录树，保存转换后的 PNG，并自动跳过已存在的 `converted_pngs` 输出文件夹。
            - **关键字过滤**：提供参数 `filter_keyword`（默认 `"校准后"`），只转换文件名包含该关键字的图片。
            - 对于文本 `.txt` 文件：直接利用 `np.loadtxt` 载入其二维整数矩阵，天然保持二维形状。
            - 对于二进制 `.raw` 文件：通过正则从文件名中匹配尺寸信息（如 `1024_1024` 或 `2048_512`）进行 `reshape`，或使用默认的 `width` 和 `height` 进行 `reshape`。
            - 最终利用 OpenCV 的 `cv2.imencode` 接口无损打包成 16-bit `.png` 文件并写入目标路径，支持 Windows 下包含中文及特殊字符的路径。
        - **参数解释**：
            - `src_dir` (str): 包含待转换文件的源目录路径。
            - `dst_dir` (str, 可选): 保存生成的 `.png` 图像的目标目录路径，为 None 时默认拼接 `"converted_pngs"`。
            - `filter_keyword` (str, 可选): 文件名中必须包含的关键字（默认 `"校准后"`）。
            - `width` (int): 二进制 `.raw` 图像的默认期望宽度（默认 1024）。
            - `height` (int): 二进制 `.raw` 图像的默认期望高度（默认 1024）。

### `plot_row_mean.py`
- **Purpose**: Analyze and visualize the column-wise mean of specific row segments (e.g., first 10 rows) for the converted PNG images.
- **Workflow**:
    1. Loads the first converted PNG image from the specified directory.
    2. Extracts the first 10 rows of the image.
    3. Calculates the mean pixel intensity across these rows for each column.
    4. Generates and saves a curve plot of the result.
- **Functions**:
    - `plot_first_10_rows_mean(img_dir, save_path)`: Reads the first PNG image, calculates the mean of its first 10 rows along the columns, and plots the result.
        - 参数 `img_dir`: 包含待分析PNG图像的目录 (str)。
        - 参数 `save_path`: 图像输出的完整路径 (str, 默认为None, 默认保存在 `img_dir` 内)。

### `crop_TYM.py`
- **Purpose**: 自动识别并裁剪16位高动态范围双能XRT图像上下部的均匀空载背景区域，仅保留包含标样或矿石有效物理信号的垂直行，从而显著减少图像存储空间并提升后续轮廓分割与特征提取算法的执行效率。
- **Workflow**:
    1. 批量遍历扫描指定输入目录下的所有 16 位 `.tif` / `.tiff` 格式的 stacked 图像。
    2. 对每一行计算像素的标准差以区分本底空气背景与实体样品。
    3. 利用适配 16-bit 像素范围的标准差阈值（默认 `3000.0`，避开 16 位下 800 - 1500 左右的高频随机探测器本底噪声）定位图像中包含有效样品的起始行与终止行。
    4. 对获取的上下边界额外外扩安全保留边距 `margin`（默认 50 像素），保障物理样品弱边缘不被误剪。
    5. 沿着高度方向对原始高动态范围图像进行无损物理裁剪，并将裁剪后的 16 位图片以原生 `uint16` 格式保存至输出文件夹。
- **Functions**:
    - `auto_crop_xrt_16bit(input_dir, output_dir, std_threshold=3000.0, margin=50)`:
        - **核心逻辑**: 对指定文件夹下的 16 位 XRT 图像进行无损空载背景行裁剪，安全外扩 margin 后保存至目标路径。
        - **参数解释**：
            - `input_dir` (str): 存放待裁剪原始 16 位 TIF 图片文件夹的物理路径。
            - `output_dir` (str): 裁剪完毕后图片的目标存储路径（不存在时会自动创建）。
            - `std_threshold` (float, 默认 `3000.0`): 区分背景与实际样品的行标准差判定阈值。16 位图像的背景行在高动态像素范围（0-65535）下带有 `800 - 1500` 左右的本底噪标准差，因此该阈值应设定在 `2000 - 5000` 之间以实现稳健分割。
            - `margin` (int, 默认 `50`): 上下边界的外扩安全缓冲裕度（像素行数）。在识别出的有效行外侧各多保留 margin 行像素，防止切除样品的微弱边缘。

### `fill_csv/fill_csv.py`
- **Purpose**: 用于将化验品位 CSV 数据自动匹配并填入主 Excel 数据表格的特定 Sheet “正面”列及 XRF 编号列的自动化工具。
- **Functions**:
    - `fill_assay_grades(excel_path, csv_path, sheet_name='0514氧化铜')`:
        - **核心逻辑**: 将 `csv_path` 指定的 CSV 文件中的化验数据与 `excel_path` 指定的 Excel 工作簿中的“0514氧化铜” Sheet 进行序号匹配。提取 CSV 中的 `Cu`, `Fe`, `Al`, `Ca`, `S` 化验品位填入 Excel 对应序号的“正面”列（D至H列），并将 CSV 的第二列（测试 #）作为 XRF 测试序号填入 Excel 的“XRF编号”列（C列，第 3 列），完美保留原 Excel 文件的所有公式（平均值公式 `=AVERAGE(...)`）、多 Sheet 结构、格式和列宽设置。
        - **参数解释**：
            - `excel_path` (str): 目标 Excel 文件的相对或绝对路径。
            - `csv_path` (str): 包含源化验品位数据的 CSV 文件路径。
            - `sheet_name` (str): 目标 Excel 工作表名称，默认为 `'0514氧化铜'`。
- **Workflow**:
    1. 使用 pandas 读取源 CSV 数据，并以 `gbk` 编码加载。识别其第 2 列（索引 1）为 XRF 测试序号，第 3 列（索引 2）为矿石序号，并收集 `Cu`, `Fe`, `Al`, `Ca`, `S` 品位值。
    2. 使用 `openpyxl` 引擎以 `data_only=False` 模式打开目标 Excel 文件以保留单元格公式。
    3. 遍历 Excel 指定 Sheet（如 `0514氧化铜`）的每一行，将 Column A（第 1 列）的单元格值转换为整数与 CSV 序号匹配。
    4. 匹配成功后，将 CSV 中的对应品位填入该行的 `D` 到 `H` 列（即第 4 到 8 列，对应正面品位），同时将 CSV 的测试序号填入 `C` 列（第 3 列，XRF编号）。如果某项数据在 CSV 中缺失（NaN），则在 Excel 中写入 `None` 以清空单元格。
    5. 调用 `wb.save(excel_path)` 写入并保存 Excel 更改。

### `pkl_reader/reader_app.py`
- **Purpose**: 一个功能丰富的跨平台 PKL 文件结构与数据细节可视化阅读器，基于 CustomTkinter 构建。支持加载任意 PKL 文件、解析多层嵌套对象（字典、列表等）、提取 Numpy 数组和 Pandas Dataframe 进行数学统计描述、可视化呈现二维表格及 1D/2D 数值数组的可视化 Matplotlib 图表。
- **Functions**:
    - `__init__()`: 
        - **核心逻辑**: 初始化 GUI 主窗口，配置窗口尺寸、居中位置以及数据缓存。
    - `setup_ui()`: 
        - **核心逻辑**: 搭建控制栏、数据层级目录树（Treeview）、选项卡（概览、文本、二维表格、绘图）及底部状态栏。
    - `open_file()`: 
        - **核心逻辑**: 弹出文件选择对话框，用户选中文件后调用加载函数。
    - `load_pkl_file(filepath)`: 
        - **核心逻辑**: 反序列化读取 PKL 文件并调用 populate 方法构建左侧结构树。
        - **参数解释**：
            - `filepath` (str): PKL 文件的绝对或相对路径。
    - `populate_tree_node(parent, name, data)`: 
        - **核心逻辑**: 递归解析 Python 数据结构并将其节点化插入至层级 Treeview 控件中。
        - **参数解释**：
            - `parent` (str): 父节点 ID。
            - `name` (str): 节点名称（字典键或列表索引）。
            - `data` (any): 节点代表的原始数据对象。
    - `on_tree_select(event)`: 
        - **核心逻辑**: 节点选中响应，分发数据进行统计展示、文本美化、表格渲染与数据绘图。
        - **参数解释**：
            - `event` (tk.Event): 触发事件对象。
    - `update_info_panel(data)`: 
        - **核心逻辑**: 计算并输出所选节点的基础物理与数学统计指标（如均值、最大值、标准差等）。
        - **参数解释**：
            - `data` (any): 待统计概览的数据对象。
    - `update_detail_text_panel(data)`: 
        - **核心逻辑**: pretty-print 美化文本或以 DataFrame.head 形式输出限制长度的文本细节。
        - **参数解释**：
            - `data` (any): 待展示的文本或数据对象。
    - `update_table_panel(data)`: 
        - **核心逻辑**: 若为二维数组或 DataFrame，提取前 300 行并在 Tkinter Table 表格网格中显示。
        - **参数解释**：
            - `data` (any): 二维表格、矩阵或数据列。
    - `update_plot_panel(data)`: 
        - **核心逻辑**: 检查数据是否可绘图。若是一维或多维数值型数组，使用 Matplotlib 绘制折线趋势、分布直方图或热图并嵌入 Tkinter 界面。
        - **参数解释**：
            - `data` (any): 数值型数组、序列或 DataFrame 对象。
    - `export_current_data()`: 
        - **核心逻辑**: 将当前选中的节点数据导出为外部 CSV 表格或纯文本 TXT 配置文件。

### `pkl_reader/build.bat`
- **Purpose**: 用于将 `reader_app.py` 一键打包成免安装 `.exe` 文件的批处理脚本。
- **Workflow**:
    1. 自动调用 pip 安装 `customtkinter`、`pyinstaller`、`matplotlib` 等所需包。
    2. 执行 `pyinstaller` 打包命令。为了解决 Anaconda 环境下 `torchaudio` 库因缺失部分 DLL 入口导致打包进程报错中断的问题，在打包指令中添加了 `--exclude-module torchaudio --exclude-module torch --exclude-module torchvision` 参数，完全绕过与音频、神经网络库相关的多余二进制扫描，确保纯净安全打包。



## 2026-05-29
- **Feature**: Excel Assayer Data & XRF Number Filling.
    - Created `fill_csv/fill_csv.py` to match ore IDs from CSV (`2026.05.29.csv`) to Excel (`CuO矿石重量.xlsx`) and write target assay grades (`Cu`, `Fe`, `Al`, `Ca`, `S`) into the "正面" (Front) columns and the XRF test numbers into "XRF编号" column (Column C).
    - Used the `openpyxl` engine with `data_only=False` to fully preserve cell styling, formatting, multi-sheet structures, and existing average calculation formulas (e.g. `=AVERAGE(D5, I5)`).

## 2026-05-20
- **Feature**: End-to-End 16-bit Image Processing Pipeline.
    - Updated `utils_II.py` (`get_bricks`, `get_bricks_watershed`) to use `cv2.IMREAD_ANYDEPTH` for natively reading 16-bit TIF/PNG files without downsampling to 8-bit.
    - Added dynamic scaling for the binarization threshold `th_val` to automatically adapt to 16-bit (0-65535) ranges.
    - Modified `extract_sample_values.py`, `extract_0429_RaySov.py`, and `extract_0429_RaySov_from_mask.py` to direct output saving into dedicated `_16bit` folders (e.g., `results/20260429_RaySov_16bit`).
    - Changed `extract_0429_RaySov_from_mask.py` normalization logic to target 16-bit depth instead of 8-bit.
    - Updated `fit_hl_curve.py` and `fit_hl_curve_0429.py` to use dynamic `v_max` masking (255 or 65535), ensuring compatibility with high-dynamic-range `.pkl` data from the 16-bit pipeline.

## 2026-04-17
- **Optimization**: Optimized `extract_sample_values.py` for 0409 dataset. When filenames contain "270us", it now uses `roi_270` and performs a 1.5x vertical compression using `cv2.INTER_AREA` interpolation before subsequent processing.
- **Improved**: Added `vscale` and `vscale_interp` parameters to `get_bricks` and `get_bricks_watershed` in `utils_II.py` to support flexible image scaling after ROI selection.

## 2026-05-08
- **Feature**: Added `all_type` parameter to `extract_sample_values.py` and `classify_contour` in `utils_II.py` to allow users to force a specific object classification (e.g., `'ore'`, `'disk'`) and bypass automatic geometric shape classification.

## 2026-04-16
- Updated `txt2img_TYM.py` to centralize all generated images into a single `converted_results` folder.
- Updated `txt2img_TYM.py` to skip files containing "offset" or "air" in their filenames (case-insensitive).

## 2026-04-10
- Updated `txt2img_TYM.py` to centralize all generated images into a single `converted_results` folder.
- Updated `txt2img_TYM.py` to skip files containing "offset" or "air" in their filenames (case-insensitive).

## 2026-04-09
- Created `txt2img_TYM.py` to recursively convert 2D TXT XRT data to images.
- **Improved**: Added support for **16-bit precision** output (uint16) to preserve raw data values (ideal for .tif).
- **New Feature**: Added support for configurable output formats (e.g., `.tif`, `.png`), defaulting to `.tif`.
- **Improved**: Switched to `cv2.imencode` for saving images to support Unicode/Chinese paths on Windows.
- **New Feature**: Added automatic filename translation from Chinese to English.
- **Optimization**: Implemented `pandas` for faster loading of large data grids.

### `txt2img_TYM.py`
- **Purpose**: Batch conversion of TXT-formatted XRT data into images (TIF, PNG, etc.).
- **Workflow**:
    1. Recursively traverses the specified data directory.
    2. Creates a centralized output folder (e.g., `converted_results`) and saves all images there.
    3. Filters out files containing "offset" or "air" in their filenames.

    3. Detects 2D data arrays using `pandas` for performance.
    4. Translates Chinese terms in filenames to English for better compatibility.
    5. Uses `cv2.imencode` to robustly save images (default format: `.tif`) to paths containing Chinese characters.
    6. Supports **16-bit precision** (preserving raw pixel values in `uint16`) or 8-bit normalization.
## Paper Notes & Guidelines
- [notes_calibration_wedge.md](file:///e:/photo_electric_II/paper/notes_calibration_wedge.md): 铜、铝、铁阶梯标样下的双能物理系数校准指南。详尽整理了基于系统无关（SIRZ）三阶段标定算法对电子密度常数 $K_1$、原子序数常数 $g$ 与幂次 $\nu$ 的最小二乘与对数线性拟合回归数学模型。
- [notes_sandwich_detector.md](file:///e:/photo_electric_II/paper/notes_sandwich_detector.md): 用于双能X射线成像中材料识别与对比度消除的三明治探测器设计学术阅读报告。
- [notes_spectrum_reconstruction.md](file:///e:/photo_electric_II/paper/notes_spectrum_reconstruction.md): 基于标样阶梯吸收差异反推出射 X 射线有效能谱的数学物理模型与增广非负最小二乘 (NNLS) 反演正则化求解原理指南。

## Data Paths
- Standard Samples: `data/` or relevant relative path.

## IDE Configuration
- **MCP Config**: `C:\Users\yaoji\.gemini\antigravity-ide\mcp_config.json` 注册并运行了官方 `@modelcontextprotocol/server-pdf` MCP 服务器（使用 `npx` 自动执行），赋予 AI 助手原生、高性能阅读本地和远程 PDF 文档、解析文本 and 交互式文档处理的能力。
- **Unsandboxed Environment**: 针对默认沙箱环境（UWP AppContainer）无法访问 `D:\` 盘 Anaconda 环境且极易触发 App Execution Alias 导致命令挂起或卡死的问题，成功申请并启用了 `unsandboxed` 的 `cmd.exe` 和 `python` 提权。这使得 AI 助手在执行终端命令时能够脱离沙箱限制，直接使用原生环境执行 `D:\anaconda\python.exe` 及其相关科学计算库，实现 100% 的原生系统速度与瞬时响应。
