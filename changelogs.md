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
