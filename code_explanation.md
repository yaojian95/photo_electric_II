# Code Explanation - Photo Electric II

## Overview
This workspace focus on validating XRT image quality and extracting standard sample features using an external library pipeline.

## External Library Integration
- **Modules used**: `preprocessing`. The dual-energy splitting logic has been localized to `utils_II.py` to eliminate external dependencies.

## Local Scripts

### `equivalent_thickness_calc.py`
- **Purpose**: A standalone utility script to calculate the equivalent ore thickness corresponding to standard metal step wedges (Cu, Fe, Al) under dual-energy XRT.
- **Functions**:
    - `calc_ore_properties(cu_grade_percent, py_grade_percent, porosity)`: 
        - 计算给定品位下的矿石的理论密度和质量衰减系数。
        - 内部构建了斑岩型铜矿的物理模型，考虑了黄铜矿(CuFeS2)、伴生黄铁矿(FeS2)和脉石(SiO2)的比例。
    - `calc_equivalent_thickness(metal_type, metal_thickness_mm, cu_grade_percent, py_grade_percent, porosity)`: 
        - 基于等效衰减原理（比尔-朗伯定律），计算纯金属（Cu, Fe, Al）厚度所对应的实际矿石厚度。
- **Workflow**:
    1. Defines physical constants ($\mu_m$, $\rho$) for key elements and minerals at ~100 keV.
    2. Uses `calc_ore_properties` to determine the aggregate attenuation properties of the ore mixture.
    3. Iterates over standard step wedge thicknesses (2mm to 20mm).
    4. Prints a formatted table and the conversion ratios (e.g., Cu step translates to ~7.5x thicker ore).


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
    - `save_contour_data`: Organized saving of warped images and pixel data.
    - `normalize_image`: 将校准后的图像灰度值从理论最大值(如50000)重新映射到目标位深(8位或16位)的指定比例(如80%)。
        - 参数 `image`: 输入的原始高深度图像 (ndarray)；`current_max`: 当前最大值 (float, 默认50000.0)；`target_ratio`: 目标比例 (float, 默认0.8)；`target_bit_depth`: 目标位深 (int, 8或16)。
        - 返回: 归一化后的图像 (ndarray, dtype=np.uint8 或 np.uint16)。
    - `calculate_effective_z`: 根据矿石品位计算有效原子序数 (Z_eff)。
        - 参数 `cu, fe, s`: 铜、铁、硫品位 (%) (float)；`z_base`: 脉石基体原子序数 (float, 默认11.0)；`exponent`: 指数因子 (float, 默认2.94)。
        - 返回: (z_eff_cufe, z_eff_cufes) 元组。

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
        - **特性**: 自动扫描数据确定自适应坐标轴限制，并强制每一行的 Y 轴（及相关 X 轴）保持一致以增强可比性。
- **Workflow**:
    1. **Iterative Loading**: Dynamically loads data from single files (steps) or multiple files (disks). Now supports disks **9-20**.
    2. **Z_eff Integration**: Loads disk grades from `disk_grades.json` and calculates $Z_{eff}$ (using `utils_II.calculate_effective_z`) to use as the x-coordinate (grade proxy).
    3. **Group-Specific Calibration**: Applies different axis limits for step samples (low gray value, high attenuation) and disk samples (high gray value, low attenuation) to ensure visibility.
    3. **Total Visualization**: Disables subsampling to plot every valid pixel in H-L space.
    4. **Grid Metrics**: Computes means, standard deviations, log-attenuation, and adaptive linear ranges.

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
