# Code Explanation - Photo Electric II

## Overview
This workspace focus on validating XRT image quality and extracting standard sample features using an external library pipeline.

## External Library Integration
- **Modules used**: `preprocessing`. The dual-energy splitting logic has been localized to `utils_II.py` to eliminate external dependencies.

## Local Scripts

### `utils_II.py`
- **Purpose**: Local wrapper for dual-energy XRT processing.
- **Functions**:
    - `split_dual_xray_image`: Locally implemented splitting logic with integrated geometric distortion correction for high-energy images.
    - `compute_R`: Calculates the R-value image (float).
    - `get_step_pixels_list`: Extracts individual arrays for each of the 10 step cores.
    - `get_disk_core_info`: Calculates core pixels and boundary for centroid-based contour scaling.
    - `get_inner_95_pixels`: Helper for general 95% area erosion.
    - `get_bricks`: Main pipeline for batch feature extraction. Supports configurable thresholding methods (`th_type`) and vertical scaling (`vscale`).
        - 参数 `path`: 图像路径 (str)；`roi`: 感兴趣区域 [y1, y2, x1, x2] (list)；`th_val`: 阈值 (int)；`th_type`: 阈值类型 (cv2)；`fx`/`fy`: 畸变校正系数 (float)；`vscale`: 纵向缩放系数 (float, 默认1.0)；`vscale_interp`: 缩放插值方法 (cv2, 默认INTER_LINEAR)。
    - `get_bricks_watershed`: Enhanced pipeline using Distance Transform and Watershed algorithm. Supports the same scaling parameters as `get_bricks`.
        - 参数同 `get_bricks`。
    - `check_step_gradient`: Analyzes row-wise mean gradients using Pearson Correlation and dynamic thresholds.
    - `warp_straighten`: Aligns tilted objects using perspective transforms.
    - `get_10_step_means`: Multi-axis core sampling (80% width, 60% height).
    - `save_contour_data`: Organized saving of warped images and pixel data.

### `extract_sample_values.py`
- **Purpose**: Batch analysis of standard samples using relative paths.
- **Workflow**:
    1. Finds target TIF files in relative data directories.
    2. Synchronized `get_bricks` parameters.
    3. Performs type-specific refined analysis (Disk scaling, Step 10-segmentation).
    4. Outputs categorized results to `results/`.

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


### `fit_hl_curve.py`
- **Purpose**: Modular analysis pipeline that performs a 2x3 grid evaluation for both stepped samples and graded disks.
- **Output Structure**:
    - `results/thickness_decoupling/h-l-fit/steps/`: Contains Cu, Fe, and Al step-sample analysis.
    - `results/thickness_decoupling/h-l-fit/disks/`: Contains analysis for graded disks (IDs 9-17).
- **Workflow**:
    1. **Iterative Loading**: Dynamically loads data from single files (steps) or multiple files (disks).
    2. **Group-Specific Calibration**: Applies different axis limits for step samples (low gray value, high attenuation) and disk samples (high gray value, low attenuation) to ensure visibility.
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
