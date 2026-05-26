# Code Explanation - Photo Electric II

## Overview
This workspace focus on validating XRT image quality and extracting standard sample features using an external library pipeline.

## External Library Integration
- **Modules used**: `preprocessing`. The dual-energy splitting logic has been localized to `utils_II.py` to eliminate external dependencies.

## Local Scripts

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
    4. 提取各电压下的 bulk 材料衰减比值，汇总为随管电压变化的能量相关特性图，并将特征数据集序列化为 JSON 导出。

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
    - `get_bricks_watershed`: Enhanced pipeline using Distance Transform and Watershed algorithm. Supports the same scaling parameters as `get_bricks`.
        - 参数同 `get_bricks`。
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
    1. Finds target TIF files in relative data directories.
    2. Synchronized `get_bricks` parameters.
    3. Performs type-specific refined analysis (Disk scaling, Step 10-segmentation).
    4. Outputs categorized results to `results/`.
    5. Visualizes bounding rectangles (`minAreaRect`) used for ROI extraction directly on the output contoured images.
    6. Masks `high_low_images` during extraction to keep only pixels inside the contour, replacing the background with the maximum grayscale value (`255` or `65535`).

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
    3. Iteratively fits Ridge regression models for three thickness scenarios (Al 6/8/10 steps) to evaluate how step inclusion- `results/thickness_decoupling/z_decouple/[DATE]/`: Contains output charts and parameter logs for a specific dataset.
- `results/thickness_decoupling/z_decouple/[DATE]/fitting_parameters.txt`: Centralized log of all regression coefficients.
 in a **5x3 grid**:
        - **Row 0**: Global overview showing separate KDE plots for Model 1, 2, and 3.
        - **Row 1**: Model 1 distribution breakdown per material.
        - **Row 2**: Model 2 distribution breakdown per material.
        - **Row 3**: Model 3 distribution breakdown per material.
        - **Row 4**: Systematic bias analysis (Mean Predicted Z vs Step Index) comparing all three models.
    5. **Step-Wise Visualization**: KDE plots include granular distributions per thickness level.
    6. **Mean Bias Analysis**: Row 4 subplots visualize the drift in mean prediction across physical thickness steps for M1, M2, and M3.
    7. **Model 3 Physics**: Uses the $R$-value formula $R = \ln(I_{0,L}/L + 5) / \ln(I_{0,H}/H + 20)$ for robust feature extraction.
    8. **Accuracy Summary**: Generates `Z_accuracy_summary_comparison.png` comparing precision (std) of all three models.
    9. **Parameter Logging**: Automatically archives all fitted coefficients and intercepts to `fitting_parameters.txt` in the timestamped output directory.
    10. **Data Traceability**: `output_dir` is now dynamically synchronized with the `input_dir` date string to prevent results from being overwritten when testing different datasets.
    8. **Optimization**: Incorporates `StandardScaler` with unscaling logic for physically accurate formula display.

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
    
### `predict_disk_Z.py`
- **Purpose**: Fits Model 2 (multivariate polynomial) and generates heatmaps and histograms for predicted atomic numbers (Z) of disks.
- **Workflow**:
    1. Parses Model 2 parameters from `fitting_parameters.txt` (which tracks the training dataset date).
    2. Uses a decoupled `TEST_DATA_DIR` to allow applying the previously trained model directly to new datasets (e.g., `20260325_yinshan`).
    3. Conditionally constructs file paths (`1_98_position_3_{voltage}_ore_{d_id}`) for Yinshan ore data to dynamically switch between standard disk calibration and real ore validation.
    3. Loads `disk_grades` from an external JSON config (`E:\multi_source_info\data_dir\disk_grades.json`) based on the test dataset date.
    4. Predicts Z for up to 114 disks using Low and High energy signals (from both `.pkl` and `.png`).
    5. Generates robustness statistics (mean, std) by filtering extreme 1% outliers, and plots Predicted Z vs Theoretical $Z_{eff}$ using `plt.scatter` for clear mean-value regression analysis.

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
- **Key Features**:
    - **Global Scaling**: 自动扫描阶梯样块数据以确定统一的横轴(X轴)对齐范围(130kV-330kV)，并为每一个厚度分别计算自适应全局统一 Y 轴，确保跨电压、跨材质、跨滤片的图表具有严格的视觉可比性。
    - **核心流程**：
  - **动态获取 Y 轴上限**: `get_dynamic_ylim` 会自动根据每个厚度的结果动态放开/收紧坐标限制。
  - **阶梯厚度遍历循环**：不再使用单一的“拟合斜率”或“指定单层”，而是强制遍历从最薄到最厚（0-9）的所有阶梯层，分别计算各层对应的衰减特性。
  - **物质比例与差值分析**：计算同种物质的高低能衰减系数比例 ($L/H$) 及不同物质之间的衰减系数比例。
  - **单块矿石综合衰减特性多通道分析**：在单块矿石对比分析中，进行 $\mu$ 衰减值计算和散点多项式回归时，会应用中央控制的 `< 2560` (16位) / `< 10` (8位) 阈值过滤，并对 `ore_{oid}_comprehensive_analysis.png` 的大标题追加剔除注记；而专门收集来绘制 `ore_{oid}_grayscale_distribution.png` 灰度直方图大图的像素集合则保持为**完全未经过滤的原始像素**，以供完整、真实地反映探测器底噪和死盲元分布状态。
  - **独立结果归档**：遍历所有厚度后，每层厚度的 $2\times3$ 综合大图 (`slope_summary`) 以及详细参数数据 (`attenuation_slopes.json`) 将会自动保存到统一的文件夹内，并在文件名中体现具体的厚度信息（如 `attenuation_slopes_2mm_CuFe_12mm_Al.json`），实现全厚度数据的直观对比与查阅。

### `read_raw.py`
- **Purpose**: Batch convert 16-bit `.raw` XRT images into `.png` format.
- **Workflow**:
    1. Traverses the specified source directory for `.raw` files.
    2. Reads 1024x1024 raw pixel data as `uint16`.
    3. Reshapes to 2D array and saves as `.png` using `cv2.imencode` to support Unicode file paths.
- **Functions**:
    - `convert_raw_to_png(src_dir, dst_dir, width, height)`: Reads 16-bit RAW images from the source directory, converts them to PNG, and saves them in the destination directory.
        - 参数 `src_dir`: 包含 .raw 文件的源目录 (str)。
        - 参数 `dst_dir`: 保存转换后 .png 文件的目标目录 (str)。
        - 参数 `width`: 图像预期宽度 (int，默认为1024)。
        - 参数 `height`: 图像预期高度 (int，默认为1024)。

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


## Data Paths
- Standard Samples: `data/` or relevant relative path.

## IDE Configuration
- **MCP Config**: `C:\Users\yaoji\.gemini\antigravity-ide\mcp_config.json` 注册并运行了官方 `@modelcontextprotocol/server-pdf` MCP 服务器（使用 `npx` 自动执行），赋予 AI 助手原生、高性能阅读本地和远程 PDF 文档、解析文本和交互式文档处理的能力。
