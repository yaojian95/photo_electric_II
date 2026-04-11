# Code Explanation - Photo Electric II

## Overview
This workspace focus on validating XRT image quality and extracting standard sample features using an external library pipeline.

## External Library Integration
- **Path**: `E:\photo_electric\jt_ore_sorting-main\jt_ore_sorting-main`
- **Modules used**: `dataloader`, `preprocessing`.

## Local Scripts

### `utils_II.py`
- **Purpose**: Local wrapper for dual-energy XRT processing.
- **Functions**:
    - `compute_R`: Calculates the R-value image (float).
    - `get_bricks`: Main pipeline to segment items. Now returns `cnt_filtered` for downstream processing.
    - `classify_contour`: Geometric classifier to distinguish between `block`, `ore`, and `disk`.
    - `save_contour_data`: Helper to save categorized pixels and bounding box visuals.

### `standard_sample_0402.py`
- **Purpose**: Batch analysis of standard samples at different voltages (140kV, 160kV, 180kV).
- **Workflow**:
    1. Ensures script directory is in `sys.path`.
    2. Dynamically creates an output subfolder in `results/` based on the `data_dir` basename.
    3. Iterates through the specified tests and voltages.
    4. Calls `get_bricks` from `utils_II.py`.
    5. Automatically **classifies each contour** as a block, ore, or disk.
    6. Saves categorized individual data (`.pkl`) and separate low/high energy images for each identified object.

### 2026-04-10
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
- 0407 Samples: `E:\multi_source_info\data_dir\20260407_Sample_test`
- 0402 Samples: `E:\multi_source_info\data_dir\20260402`
