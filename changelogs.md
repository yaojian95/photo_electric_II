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
