## 2026-04-13
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
