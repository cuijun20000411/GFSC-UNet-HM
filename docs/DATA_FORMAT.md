# Processed data contract

For each pollutant (`no2`, `o3`, `pm25`), the HM program expects:

| File | Meaning |
| --- | --- |
| `conc_img_raw.npy` | Sparse raw concentration images in physical units; shape `(T, 32, 32)` |
| `cov_img_full.npy` | Full-grid covariates; shape `(T, F, 32, 32)`, F = 39 for NO2/O3 and 40 for PM2.5 |
| `coord_img.npy` | Two grid coordinate channels, shape `(32, 32, 2)` |
| `station_positions.npy` | Row/column indices in the fixed original station order, shape `(55, 2)` |
| `valid_time_indices.npy` | Original hourly time indices for aligned samples |
| `time_splits.npy` | Dictionary with `train`, `val`, `test` array-index lists, saved with NumPy |
| `valid_grid_mask.npy` | Valid-cell mask, shape `(32, 32)`, with exactly 999 valid cells |

For legacy concentration input, `conc_img_full.npy` plus the matching `conc_scaler.pkl` can replace `conc_img_raw.npy`. The loader restores raw units before refitting concentration scaling within each outer fold. Do not substitute globally standardized values without their matching scaler.

The current manuscript uses 720 aligned hourly samples (504 train, 108 validation, 108 test), 55 stations and the captured fixed spatial split. Array-index lists and original time IDs are different fields.

Also publish the exact covariate names, units, order, grouped channel indices, coordinate convention and data provenance. The three original protocol JSON files preserve grouping and parameter checks; they are missing from this preparation bundle.

The recovered `zh.py` is a legacy preprocessing script. It writes normalized concentration arrays but does not write the valid-grid mask required by the HM preflight. Recover the actual final preprocessing pipeline used to create the current data rather than declaring that the legacy script generates every required file.

Use a data repository or an appropriate download location for the original arrays and weights. If sharing restrictions apply, describe the restriction and provide a compatible synthetic/example dataset. No synthetic data in this bundle should be mistaken for measurements; this bundle currently contains no input arrays.
