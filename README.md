# GFSC-UNet-HM

Single-timestamp reconstruction of air pollutant concentration fields under sparse monitoring using hybrid spatial masking and multiscale covariate fusion.

## Preparation status

This repository snapshot contains 22 selected Python source files, a captured five-fold spatial split, and selected supporting results. The source files are preserved byte for byte. Original processed data, the three reference protocol JSON files, model checkpoints and the final spatial-anomaly/pairwise analysis generator are not included. Complete `docs/RELEASE_CHECKLIST.md` before presenting the repository as an independently reproducible paper release.

`D3` and `D3HM` are retained in source filenames and output labels for compatibility. The paper model is **GFSC-UNet-HM**. Files required by the random-mask reference are retained as dependencies; they should not be deleted merely because their filenames use a historical model label.

## Main source files

| File | Function |
| --- | --- |
| `gfsc_unet_d3_hybrid_mask_main.py` | HM training, strict LOSO, fixed spatial-block validation, diagnostics and full-field inference |
| `rh4_final_strict_light.py` | Data loading, fold-specific concentration scaling, observation features and spatial regularization |
| `fusion_final_strict.py` | Covariate grouping, encoders and multiscale conditioning |
| `hybrid_masking.py` | Balanced random/contiguous masking and partition checks |

Keep the Python files together at the repository root. See `docs/source_manifest.csv` for each source file's role and SHA-256. The older `spatial_skill_analysis.py` is provided for provenance and diagnostics; it does **not** implement the latest spatial-anomaly/pairwise analysis reported by the captured summaries.

## Environment

Python 3.10 or later is needed by the type annotations; the exact original experiment environment still needs to be recorded. `requirements.txt` lists imported packages, without claiming these are the original version pins. PyTorch/CUDA versions and hardware should match the recorded paper environment when reproducing runtime measurements. `requirements-optional.txt` lists PyKrige, which is optional in the spatial classical-baseline runner.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

## Data and reference protocols

Read `docs/DATA_FORMAT.md`. Put the original processed pollutant files in `data/processed/{no2,o3,pm25}/`. Put the original audited files `protocol_no2.json`, `protocol_o3.json` and `protocol_pm25.json` in `configs/reference/D3/`.

The main HM program reads these protocol files to recover and verify the exact covariate grouping and reference configuration. The protocol files are needed even when training from scratch; old D3 model weights are not required for this grouping check. Do not invent replacement protocols or remove provenance checks.

## Preflight and training

Run from the repository root, after adding the missing data and reference protocols:

```bash
python gfsc_unet_d3_hybrid_mask_main.py \
  --mode loso \
  --data-root data/processed \
  --d3-reference-root configs/reference/D3 \
  --spatial-split-json configs/spatial_split.json \
  --output results/runs/hm_seed42 \
  --preflight
```

For a short configuration trial, use the same paths with `--mode loso --pollutants no2 --max-folds 1 --epochs 2 --patience 2`, and a separate trial output directory. Remove these trial limits when running the full paper experiment; trial results are not paper results.

To run the fixed spatial blocks, set `--mode spatial-block`. To generate full-region fields, set `--mode full-field`. `--mode all` runs several expensive stages and should be used only when all are intended. Matched-seed runners need the completed seed-42 reference results. Inspect each runner's `--help` and explicitly supply portable input/output paths; the preserved sources still contain server-specific defaults.

## Existing results and evidence

`results/hm_multiseed/` contains the captured HM seed metrics; the matched RM/HM experiment outputs and all station-time predictions still need to be added from the original server. `results/spatial_pattern/` contains the latest captured summary tables and pair definitions. It is not sufficient to reproduce those tables without their generating script and prediction inputs.

Use held-out monitoring stations for quantitative accuracy. Full-field maps display unmonitored-cell estimates without independent full-grid ground truth. Training a full-station visualization model does not replace strict LOSO validation.

## License and citation

Add a project license approved by the code rights holders and a finalized citation once the authors and publication metadata are confirmed. No license grant or publication DOI is supplied by this preparation snapshot.
