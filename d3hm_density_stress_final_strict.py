"""D3-HM fixed-model observation-density stress test.

This program evaluates the already trained strict-LOSO D3-HM checkpoints when
additional input stations are removed at inference time.  It does not retrain
or overwrite D3-HM.  For every outer held-out station, the target station is
kept unavailable and 0%, 25%, 50%, or 75% of the remaining 54 stations are
additionally hidden.  Every positive ratio uses ten deterministic subsets by
default.  D3-HM, IDW, KNN, SpatialMean, and optional NearestNeighbor/RBF use
the identical visible-station set in every scenario.

All observation-derived inputs (filled concentration, local mean/std,
observation mask, distance transform, and neighbourhood statistics) are
recomputed after each additional hiding operation.  The 0%-hidden predictions
must reproduce the stored D3-HM LOSO fold predictions before a fold is
accepted.

Required neighbouring files on the server
-----------------------------------------
* gfsc_unet_d3_hybrid_mask_main.py
* hybrid_masking.py
* rh4_final_strict_light.py
* fusion_final_strict.py
* density_stress_final_strict.py

The default paths match the earlier D3-HM and D3 experiments.  They can be
changed with environment variables documented in README_运行说明.md.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import gfsc_unet_d3_hybrid_mask_main as d3hm
    import density_stress_final_strict as density
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Place d3hm_density_stress_final_strict.py beside "
        "gfsc_unet_d3_hybrid_mask_main.py, hybrid_masking.py, "
        "rh4_final_strict_light.py, fusion_final_strict.py, and "
        "density_stress_final_strict.py."
    ) from exc


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be 0/1, false/true, or no/yes")
    return value in {"1", "true", "yes"}


class Config(d3hm.D3Config):
    """D3-HM inference configuration plus density-stress controls."""

    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    D3_REFERENCE_ROOT = os.environ.get(
        "D3_REFERENCE_ROOT", "/etc/cj/boundary/f3_direct_ablation_final_strict_v1"
    )
    D3HM_OUTPUT_ROOT = os.environ.get(
        "D3HM_OUTPUT_ROOT", "/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1"
    )
    MODEL_ROOT = os.environ.get(
        "D3HM_MODEL_ROOT", str(Path(D3HM_OUTPUT_ROOT, "loso", "D3HM"))
    )
    RESULT_DIR = os.environ.get(
        "DENSITY_RESULT_ROOT",
        str(Path(D3HM_OUTPUT_ROOT, "density_stress", "D3HM")),
    )
    D3_DENSITY_ROOT = os.environ.get(
        "D3_DENSITY_RESULT_ROOT",
        "/etc/cj/boundary/d3_density_stress_final_strict_v1",
    )
    POLLUTANTS = [
        item.strip().lower()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    HIDE_RATIOS = density.parse_float_list(
        os.environ.get("DENSITY_HIDE_RATIOS", "0,0.25,0.50,0.75")
    )
    REPEATS = int(os.environ.get("DENSITY_REPEATS", "10"))
    MODELS = [
        item.strip()
        for item in os.environ.get(
            "DENSITY_MODELS", "D3HM,SpatialMean,NearestNeighbor,IDW,KNN"
        ).split(",")
        if item.strip()
    ]
    IDW_POWER_CANDIDATES = tuple(
        float(item)
        for item in os.environ.get(
            "DENSITY_IDW_POWER_CANDIDATES", "1.0,1.5,2.0"
        ).split(",")
        if item.strip()
    )
    KNN_CANDIDATES = tuple(
        int(item)
        for item in os.environ.get("DENSITY_KNN_CANDIDATES", "3,4,6,8").split(",")
        if item.strip()
    )
    RBF_FUNCTION = "linear"
    RBF_SMOOTH = 0.1
    EVAL_BATCH_SIZE = int(os.environ.get("DENSITY_BATCH_SIZE", "1"))
    NUM_WORKERS = int(os.environ.get("DENSITY_NUM_WORKERS", "2"))
    FOLD_START = int(os.environ.get("DENSITY_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("DENSITY_MAX_STATIONS", "0"))
    REUSE_COMPLETED = env_flag("DENSITY_REUSE_COMPLETED", True)
    SAVE_RAW_PREDICTIONS = env_flag("DENSITY_SAVE_RAW_PREDICTIONS", True)
    BOOTSTRAP_REPEATS = int(os.environ.get("DENSITY_BOOTSTRAP_REPEATS", "2000"))
    REQUIRE_D3HM_REFERENCE = env_flag("DENSITY_REQUIRE_D3HM_REFERENCE", True)
    ZERO_REFERENCE_ATOL = float(
        os.environ.get("DENSITY_ZERO_REFERENCE_ATOL", "1e-5")
    )
    COMPARE_D3 = env_flag("DENSITY_COMPARE_D3", True)
    REQUIRE_D3_COMPARISON = env_flag("DENSITY_REQUIRE_D3_COMPARISON", True)
    EXPECTED_MODEL_VERSION = d3hm.D3Config.MODEL_VERSION
    EXPERIMENT_VERSION = "d3hm_fixed_model_density_stress_v1"


ALLOWED_MODELS = {
    "D3HM",
    "SpatialMean",
    "NearestNeighbor",
    "IDW",
    "KNN",
    "RBF",
}


def validate_config(config: Config) -> None:
    unknown = sorted(set(config.MODELS) - ALLOWED_MODELS)
    if unknown:
        raise ValueError(f"Unknown DENSITY_MODELS entries: {unknown}")
    if "D3HM" not in config.MODELS:
        raise ValueError("D3HM must be included in DENSITY_MODELS")
    if config.REPEATS < 1:
        raise ValueError("DENSITY_REPEATS must be at least 1")
    if not any(np.isclose(ratio, 0.0) for ratio in config.HIDE_RATIOS):
        raise ValueError("DENSITY_HIDE_RATIOS must include 0")
    if not 0 <= config.FOLD_START < 55:
        raise ValueError("DENSITY_FOLD_START must be in 0..54")
    if config.MAX_STATIONS < 0:
        raise ValueError("DENSITY_MAX_STATIONS cannot be negative")
    if config.EVAL_BATCH_SIZE < 1:
        raise ValueError("DENSITY_BATCH_SIZE must be at least 1")
    if config.REQUIRE_D3HM_REFERENCE and config.EVAL_BATCH_SIZE != 1:
        raise ValueError(
            "DENSITY_BATCH_SIZE must equal 1 when the strict stored-D3HM "
            "prediction check is enabled"
        )
    if config.ZERO_REFERENCE_ATOL < 0:
        raise ValueError("DENSITY_ZERO_REFERENCE_ATOL cannot be negative")
    if not config.IDW_POWER_CANDIDATES:
        raise ValueError("DENSITY_IDW_POWER_CANDIDATES cannot be empty")
    if not config.KNN_CANDIDATES or min(config.KNN_CANDIDATES) < 1:
        raise ValueError("DENSITY_KNN_CANDIDATES must contain positive integers")

    model_root = Path(config.MODEL_ROOT)
    if model_root.name != "D3HM":
        warnings.warn(f"D3HM_MODEL_ROOT does not end in D3HM: {model_root}")
    if not model_root.exists():
        raise FileNotFoundError(f"D3-HM LOSO model root not found: {model_root}")
    if config.COMPARE_D3 and config.REQUIRE_D3_COMPARISON:
        d3_root = Path(config.D3_DENSITY_ROOT)
        if not d3_root.exists():
            raise FileNotFoundError(
                "Existing D3 density-stress result root is required for the "
                f"paired comparison but was not found: {d3_root}"
            )


def prepare_output(config: Config) -> None:
    for name in ("folds", "metrics", "predictions", "figures"):
        Path(config.RESULT_DIR, name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_version": config.EXPERIMENT_VERSION,
        "density_seed": density.GLOBAL_SEED,
        "d3hm_training_seed": d3hm.GLOBAL_SEED,
        "device": str(d3hm.DEVICE),
        "candidate_model": "D3HM",
        "candidate_definition": (
            "D3 architecture trained with balanced random/contiguous masking"
        ),
        "model_root": str(Path(config.MODEL_ROOT).resolve()),
        "d3_reference_root": str(Path(config.D3_REFERENCE_ROOT).resolve()),
        "expected_model_version": config.EXPECTED_MODEL_VERSION,
        "pollutants": config.POLLUTANTS,
        "models": config.MODELS,
        "additional_hidden_station_ratios": config.HIDE_RATIOS,
        "positive_ratio_repeats": config.REPEATS,
        "zero_ratio_repeats": 1,
        "evaluation_batch_size": config.EVAL_BATCH_SIZE,
        "outer_protocol": "station-wise strict LOSO",
        "outer_target_history_used": False,
        "additional_hidden_station_history_used_during_original_training": True,
        "experiment_interpretation": "fixed-model observation sparsification test",
        "retraining": False,
        "prediction_clipping": False,
        "fixed_subset_within_scenario": True,
        "subset_shared_by_all_methods": True,
        "derived_features_recomputed_after_hiding": True,
        "idw_knn_tuning": "inner validation stations only; outer station excluded",
        "zero_hidden_check": (
            "inputs and raw predictions must reproduce the stored D3-HM "
            "strict-LOSO outputs"
        ),
        "require_stored_d3hm_reference": config.REQUIRE_D3HM_REFERENCE,
        "zero_reference_atol": config.ZERO_REFERENCE_ATOL,
        "paired_d3_comparison": config.COMPARE_D3,
        "d3_density_result_root": str(Path(config.D3_DENSITY_ROOT).resolve()),
    }
    path = Path(config.RESULT_DIR, "protocol.json")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


class D3HMDensityStressDataset(d3hm.HybridMaskFusionDataset):
    """D3-HM LOSO test data with a fixed context-station subset removed."""

    def __init__(
        self,
        config: Config,
        pollutant: str,
        heldout_station: int,
        additionally_hidden: tuple[int, ...],
    ):
        self.additionally_hidden = tuple(
            sorted(int(value) for value in additionally_hidden)
        )
        super().__init__(
            config,
            pollutant,
            outer_heldout=[int(heldout_station)],
            mode="test",
            variant=d3hm.d3_variant(config),
            mask_view="none",
        )
        self.mask_station_idx = int(heldout_station)
        if self.mask_station_idx in self.additionally_hidden:
            raise ValueError("Outer held-out station cannot be additionally hidden")
        valid_ids = set(range(self.total_stations)) - {self.mask_station_idx}
        if not set(self.additionally_hidden).issubset(valid_ids):
            raise ValueError("Additional hidden-station set contains an invalid id")

    @property
    def available_station_ids(self) -> tuple[int, ...]:
        excluded = set(self.additionally_hidden) | {self.mask_station_idx}
        return tuple(
            station for station in range(self.total_stations) if station not in excluded
        )

    def normalized_image(self, array_idx: int) -> np.ndarray:
        return self._to_fold_normalized(
            np.asarray(self.conc_img_raw[array_idx], dtype=np.float32)
        ).copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        array_idx = int(self.time_indices[idx])
        original_time_idx = int(self.valid_time_indices_original[array_idx])
        sparse = self.normalized_image(array_idx)
        covariates = np.asarray(self.cov_img_full[array_idx], dtype=np.float32).copy()
        if not np.isfinite(covariates).all():
            raise ValueError(
                f"{self.pollutant}: non-finite covariates at array index {array_idx}"
            )
        target = np.full(self.config.IMG_SHAPE, np.nan, dtype=np.float32)

        target_row, target_col = self.station_positions[self.mask_station_idx]
        target_value = sparse[target_row, target_col]
        sparse[target_row, target_col] = np.nan
        target[target_row, target_col] = target_value
        for station in self.additionally_hidden:
            row, col = self.station_positions[station]
            sparse[row, col] = np.nan

        observed_mask = np.isfinite(sparse) & (self.valid_grid_mask > 0)
        known_values = sparse.copy()
        observed_values = sparse[observed_mask]
        sample_mean = float(observed_values.mean()) if observed_values.size else 0.0
        sample_std = float(observed_values.std()) if observed_values.size > 1 else 1.0
        if sample_std < 1e-6:
            sample_std = 1.0

        filled = np.where(observed_mask, sparse, sample_mean).astype(np.float32)
        neighbour = d3hm.core.observed_neighbor_features(
            sparse, observed_mask, self.radius, sample_mean, sample_std
        )
        distance = distance_transform_edt(~observed_mask).astype(np.float32)
        valid_distance = distance[self.valid_grid_mask > 0]
        maximum = (
            float(valid_distance.max())
            if valid_distance.size
            else float(distance.max())
        )
        distance /= max(maximum, 1.0)
        distance *= self.valid_grid_mask

        base_features = np.concatenate(
            [filled[None], neighbour, self.coord_feat, distance[None]], axis=0
        ).astype(np.float32)
        model_input = np.concatenate([base_features, covariates], axis=0)
        infer_mask = ((self.valid_grid_mask > 0) & (~observed_mask)).astype(np.float32)
        known_values = np.nan_to_num(
            known_values,
            nan=sample_mean,
            posinf=sample_mean,
            neginf=sample_mean,
        ).astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "channel_mask": torch.from_numpy(
                observed_mask[None].astype(np.float32)
            ),
            "known_values": torch.from_numpy(known_values[None]),
            "target": torch.from_numpy(target[None]),
            "infer_mask": torch.from_numpy(infer_mask[None]),
            "valid_grid_mask": torch.from_numpy(self.valid_grid_mask[None]),
            "time_idx": original_time_idx,
            "array_idx": array_idx,
            "mask_type": self.MASK_CODES["none"],
            "masked_station_count": len(self.additionally_hidden),
        }


def reference_dataset(
    config: Config, pollutant: str, station_idx: int
) -> d3hm.HybridMaskFusionDataset:
    return d3hm.HybridMaskFusionDataset(
        config,
        pollutant,
        outer_heldout=[station_idx],
        mode="test",
        variant=d3hm.d3_variant(config),
        mask_view="none",
    )


def verify_zero_hidden_equivalence(
    config: Config,
    pollutant: str,
    station_idx: int,
    stress_dataset: D3HMDensityStressDataset,
) -> None:
    reference = reference_dataset(config, pollutant, station_idx)
    if len(reference) != len(stress_dataset):
        raise RuntimeError("D3-HM zero-hidden and strict-test lengths differ")
    if not (
        np.isclose(reference.fold_mean, stress_dataset.fold_mean)
        and np.isclose(reference.fold_std, stress_dataset.fold_std)
    ):
        raise RuntimeError("D3-HM zero-hidden and strict fold scalers differ")

    check_indices = sorted({0, len(reference) // 2, len(reference) - 1})
    tensor_keys = (
        "input",
        "channel_mask",
        "known_values",
        "target",
        "infer_mask",
        "valid_grid_mask",
    )
    for index in check_indices:
        expected = reference[index]
        actual = stress_dataset[index]
        if int(expected["time_idx"]) != int(actual["time_idx"]):
            raise RuntimeError("D3-HM zero-hidden test-time indices differ")
        for key in tensor_keys:
            if not torch.allclose(
                expected[key], actual[key], atol=1e-6, rtol=1e-6, equal_nan=True
            ):
                raise RuntimeError(
                    "D3-HM zero-hidden input differs from the original test input: "
                    f"station={station_idx}, sample={index}, tensor={key}"
                )


def checkpoint_paths(
    config: Config, pollutant: str, station_idx: int
) -> tuple[Path, Path]:
    checkpoint = Path(
        config.MODEL_ROOT,
        "models",
        f"{pollutant}_station_{station_idx:02d}_D3HM.pth",
    )
    return checkpoint, Path(str(checkpoint) + ".complete.json")


def checkpoint_signature(
    checkpoint: Path, fingerprint: str, protocol_path: Path
) -> dict[str, int | str]:
    stat = checkpoint.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "fingerprint": fingerprint,
        "model_version": d3hm.D3Config.MODEL_VERSION,
        "source_protocol_size": int(protocol_path.stat().st_size),
        "source_protocol_mtime_ns": int(protocol_path.stat().st_mtime_ns),
    }


def load_model_context(
    config: Config, pollutant: str, probe: D3HMDensityStressDataset
) -> tuple[list, Path]:
    reference_root = d3hm.resolve_reference_root(config.D3_REFERENCE_ROOT)
    groups, protocol_path, _ = d3hm.load_groups(
        reference_root, pollutant, probe.F_dim, config
    )
    return groups, protocol_path


def load_completed_model(
    config: Config,
    pollutant: str,
    station_idx: int,
    probe: D3HMDensityStressDataset,
) -> tuple[d3hm.GFSCUNetD3, dict[str, int | str]]:
    groups, protocol_path = load_model_context(config, pollutant, probe)
    expected_fingerprint = d3hm.model_fingerprint(
        config,
        pollutant,
        station_idx,
        groups,
        protocol_path,
        outer_heldout=[station_idx],
    )
    checkpoint, marker_path = checkpoint_paths(config, pollutant, station_idx)
    if not checkpoint.exists() or not marker_path.exists():
        raise FileNotFoundError(
            f"Completed D3-HM checkpoint is missing for {pollutant} station "
            f"{station_idx}: {checkpoint}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid D3-HM marker JSON: {marker_path}") from exc
    compatible = (
        marker.get("status") == "completed"
        and marker.get("fingerprint") == expected_fingerprint
        and marker.get("model_version") == config.EXPECTED_MODEL_VERSION
        and marker.get("pollutant") == pollutant
        and int(marker.get("station_idx", -1)) == station_idx
        and marker.get("outer_heldout") == [station_idx]
        and marker.get("masking_protocol") == config.MASKING_PROTOCOL_VERSION
    )
    if not compatible:
        raise RuntimeError(
            f"D3-HM checkpoint marker/config mismatch for {pollutant} "
            f"station {station_idx}: {marker_path}"
        )

    model = d3hm.GFSCUNetD3(
        probe.base_channels,
        probe.F_dim,
        groups,
        d3hm.d3_variant(config),
        config,
    ).to(d3hm.DEVICE)
    try:
        state = torch.load(checkpoint, map_location=d3hm.DEVICE)
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"D3-HM state_dict mismatch for {pollutant} station "
            f"{station_idx}: {checkpoint}"
        ) from exc
    return model.eval(), checkpoint_signature(
        checkpoint, expected_fingerprint, protocol_path
    )


def d3hm_predictions(
    model: d3hm.GFSCUNetD3,
    dataset: D3HMDensityStressDataset,
    config: Config,
) -> dict[int, tuple[float, float]]:
    loader = d3hm.core.make_loader(
        dataset,
        config,
        shuffle=False,
        batch_size=config.EVAL_BATCH_SIZE,
    )
    row, col = dataset.station_positions[dataset.mask_station_idx]
    result: dict[int, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, targets, _ = d3hm.core.move_batch(batch)
            output = model(inputs, observed, known)
            predictions = output[:, 0, row, col].cpu().numpy()
            truths = targets[:, 0, row, col].cpu().numpy()
            times = batch["time_idx"].cpu().numpy()
            for time_idx, true_value, prediction in zip(times, truths, predictions):
                if np.isfinite(true_value) and np.isfinite(prediction):
                    result[int(time_idx)] = (
                        float(true_value),
                        float(prediction),
                    )
    return result


def scenario_rows(
    config: Config,
    pollutant: str,
    station_idx: int,
    hide_ratio: float,
    repeat: int,
    hidden: tuple[int, ...],
    model: d3hm.GFSCUNetD3,
    best_idw_power: float,
    best_knn_k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset = D3HMDensityStressDataset(
        config,
        pollutant,
        heldout_station=station_idx,
        additionally_hidden=hidden,
    )
    available_count = len(dataset.available_station_ids)
    neural = d3hm_predictions(model, dataset, config)
    rows: list[dict[str, object]] = []

    for array_idx_raw in dataset.time_indices:
        array_idx = int(array_idx_raw)
        original_time = int(dataset.valid_time_indices_original[array_idx])
        normalized = dataset.normalized_image(array_idx)
        distances, coords, known_values, true_normalized = density.distance_and_values(
            dataset, normalized
        )
        if len(known_values) < 2 or not np.isfinite(true_normalized):
            continue
        target_coord = dataset.station_positions[station_idx].astype(np.float64)
        predictors: dict[str, Callable[[], float]] = {
            "SpatialMean": lambda: density.spatial_mean(distances, known_values),
            "NearestNeighbor": lambda: density.nearest_neighbor(
                distances, known_values
            ),
            "IDW": lambda: density.idw_predict(
                distances, known_values, best_idw_power
            ),
            "KNN": lambda: density.knn_predict(
                distances, known_values, best_knn_k
            ),
            "RBF": lambda: density.rbf_predict(
                coords,
                known_values,
                target_coord,
                config.RBF_FUNCTION,
                config.RBF_SMOOTH,
            ),
        }
        normalized_predictions: dict[str, float] = {}
        if "D3HM" in config.MODELS and original_time in neural:
            neural_true, neural_prediction = neural[original_time]
            if not np.isclose(
                neural_true, true_normalized, atol=1e-5, rtol=1e-5
            ):
                raise RuntimeError(
                    "D3-HM and spatial-baseline test targets are misaligned"
                )
            normalized_predictions["D3HM"] = neural_prediction
        for model_name in config.MODELS:
            if model_name in predictors:
                normalized_predictions[model_name] = predictors[model_name]()

        raw_true = true_normalized * dataset.fold_std + dataset.fold_mean
        for model_name, prediction_normalized in normalized_predictions.items():
            if not np.isfinite(prediction_normalized):
                continue
            raw_prediction = (
                prediction_normalized * dataset.fold_std + dataset.fold_mean
            )
            rows.append(
                {
                    "pollutant": pollutant,
                    "station": station_idx,
                    "hide_ratio": hide_ratio,
                    "repeat": repeat,
                    "n_available_stations": available_count,
                    "model": model_name,
                    "time_idx": original_time,
                    "true": float(raw_true),
                    "prediction": float(raw_prediction),
                    "absolute_error": float(abs(raw_prediction - raw_true)),
                }
            )

    subset_record = {
        "pollutant": pollutant,
        "station": station_idx,
        "hide_ratio": hide_ratio,
        "repeat": repeat,
        "n_hidden_stations": len(hidden),
        "n_available_stations": available_count,
        "hidden_station_ids": ";".join(str(item) for item in hidden),
        "selected_idw_power": best_idw_power,
        "selected_knn_k": best_knn_k,
    }
    return rows, subset_record


def reference_fold_path(config: Config, pollutant: str, station_idx: int) -> Path:
    return Path(
        config.MODEL_ROOT,
        "folds",
        f"{pollutant}_station_{station_idx:02d}_predictions.csv",
    )


def verify_zero_hidden_predictions(
    config: Config,
    pollutant: str,
    station_idx: int,
    predictions: pd.DataFrame,
) -> bool:
    path = reference_fold_path(config, pollutant, station_idx)
    if not path.exists():
        if config.REQUIRE_D3HM_REFERENCE:
            raise FileNotFoundError(
                f"Stored D3-HM fold predictions are required: {path}"
            )
        warnings.warn(f"Skipping stored D3-HM prediction check: {path}")
        return False
    reference = pd.read_csv(path)
    required = {"station", "time_idx", "true", "prediction"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(
            f"D3-HM reference prediction file lacks columns {sorted(missing)}"
        )
    zero = predictions[
        (predictions["model"] == "D3HM")
        & np.isclose(predictions["hide_ratio"], 0.0)
        & (predictions["repeat"] == 0)
    ][["station", "time_idx", "true", "prediction"]].copy()
    merged = reference[["station", "time_idx", "true", "prediction"]].merge(
        zero,
        on=["station", "time_idx"],
        how="outer",
        suffixes=("_reference", "_density"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(
            f"D3-HM zero-hidden prediction keys differ at station {station_idx}"
        )
    for column in ("true", "prediction"):
        difference = np.abs(
            merged[f"{column}_reference"].to_numpy(dtype=np.float64)
            - merged[f"{column}_density"].to_numpy(dtype=np.float64)
        )
        maximum = float(np.nanmax(difference)) if difference.size else 0.0
        if maximum > config.ZERO_REFERENCE_ATOL:
            raise RuntimeError(
                f"D3-HM zero-hidden {column} mismatch at station {station_idx}: "
                f"max_abs_difference={maximum:.8g}, "
                f"allowed={config.ZERO_REFERENCE_ATOL:.8g}"
            )
    return True


_BASE_RUN_FOLD = density.run_fold


def run_fold(
    config: Config, pollutant: str, station_idx: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions, subsets, runtime = _BASE_RUN_FOLD(config, pollutant, station_idx)
    verified = verify_zero_hidden_predictions(
        config, pollutant, station_idx, predictions
    )
    runtime = dict(runtime)
    runtime["zero_hidden_d3hm_reference_verified"] = verified
    _, _, runtime_path, marker_path = density.fold_paths(
        config, pollutant, station_idx
    )
    runtime_path.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["zero_hidden_d3hm_reference_verified"] = verified
        marker_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return predictions, subsets, runtime


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_row(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    truth = frame["true"].to_numpy(dtype=np.float64)
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    return {
        f"{prefix}_MAE": float(mean_absolute_error(truth, prediction)),
        f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        f"{prefix}_R2": safe_r2(truth, prediction),
    }


def station_average_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = frame[frame["model"] == model].copy()
    rows = []
    for keys, group in subset.groupby(
        ["pollutant", "hide_ratio", "station", "repeat"], sort=True
    ):
        truth = group["true"].to_numpy(dtype=np.float64)
        prediction = group["prediction"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "pollutant": keys[0],
                "hide_ratio": float(keys[1]),
                "station": int(keys[2]),
                "repeat": int(keys[3]),
                "MAE": float(mean_absolute_error(truth, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
            }
        )
    by_repeat = pd.DataFrame(rows)
    return (
        by_repeat.groupby(
            ["pollutant", "hide_ratio", "station"], as_index=False
        )[["MAE", "RMSE"]]
        .mean()
    )


def paired_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    finite = np.isfinite(difference)
    difference = difference[finite]
    if difference.size < 2 or np.allclose(difference, 0.0):
        return 1.0
    return float(wilcoxon(difference, zero_method="wilcox").pvalue)


def verify_subset_identity(
    config: Config, pollutant: str, d3_subsets: pd.DataFrame
) -> None:
    current_path = Path(
        config.RESULT_DIR, "metrics", f"{pollutant}_hidden_subsets.csv"
    )
    if not current_path.exists():
        raise FileNotFoundError(f"D3-HM subset file is missing: {current_path}")
    current = pd.read_csv(current_path)
    keys = ["pollutant", "station", "hide_ratio", "repeat"]
    fields = keys + [
        "n_hidden_stations",
        "n_available_stations",
        "hidden_station_ids",
    ]
    merged = d3_subsets[fields].merge(
        current[fields],
        on=keys,
        how="outer",
        suffixes=("_D3", "_D3HM"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"D3/D3-HM scenario keys differ for {pollutant}")
    for field in ("n_hidden_stations", "n_available_stations", "hidden_station_ids"):
        left = merged[f"{field}_D3"].fillna("").astype(str)
        right = merged[f"{field}_D3HM"].fillna("").astype(str)
        if not left.equals(right):
            raise RuntimeError(
                f"D3 and D3-HM do not use identical hidden subsets: "
                f"pollutant={pollutant}, field={field}"
            )


def compare_with_existing_d3(config: Config) -> None:
    if not config.COMPARE_D3:
        return
    d3_root = Path(config.D3_DENSITY_ROOT)
    if not d3_root.exists():
        if config.REQUIRE_D3_COMPARISON:
            raise FileNotFoundError(f"D3 density result root not found: {d3_root}")
        warnings.warn(f"Skipping D3 paired comparison: {d3_root}")
        return

    comparison_rows: list[dict[str, object]] = []
    for pollutant in config.POLLUTANTS:
        d3_prediction_path = d3_root / "predictions" / f"{pollutant}_predictions.csv"
        d3_subset_path = d3_root / "metrics" / f"{pollutant}_hidden_subsets.csv"
        hm_prediction_path = (
            Path(config.RESULT_DIR) / "predictions" / f"{pollutant}_predictions.csv"
        )
        required_paths = [d3_prediction_path, d3_subset_path, hm_prediction_path]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            if config.REQUIRE_D3_COMPARISON:
                raise FileNotFoundError(
                    f"Files required for D3/D3-HM comparison are missing: {missing}"
                )
            warnings.warn(f"Skipping {pollutant} D3 comparison; missing {missing}")
            continue

        d3_predictions = pd.read_csv(d3_prediction_path)
        hm_predictions = pd.read_csv(hm_prediction_path)
        verify_subset_identity(config, pollutant, pd.read_csv(d3_subset_path))
        d3_only = d3_predictions[d3_predictions["model"] == "D3"].copy()
        hm_only = hm_predictions[hm_predictions["model"] == "D3HM"].copy()
        keys = ["pollutant", "station", "hide_ratio", "repeat", "time_idx"]
        merged = d3_only[keys + ["true", "prediction"]].merge(
            hm_only[keys + ["true", "prediction"]],
            on=keys,
            how="outer",
            suffixes=("_D3", "_D3HM"),
            indicator=True,
        )
        if not (merged["_merge"] == "both").all():
            raise RuntimeError(f"D3/D3-HM prediction keys differ for {pollutant}")
        truth_difference = np.abs(
            merged["true_D3"].to_numpy(dtype=np.float64)
            - merged["true_D3HM"].to_numpy(dtype=np.float64)
        )
        if truth_difference.size and float(np.nanmax(truth_difference)) > 1e-5:
            raise RuntimeError(f"D3/D3-HM truths differ for {pollutant}")

        d3_station = station_average_metrics(d3_only, "D3")
        hm_station = station_average_metrics(hm_only, "D3HM")
        station_pair = d3_station.merge(
            hm_station,
            on=["pollutant", "hide_ratio", "station"],
            suffixes=("_D3", "_D3HM"),
            validate="one_to_one",
        )
        for hide_ratio in sorted(merged["hide_ratio"].unique()):
            selected = merged[np.isclose(merged["hide_ratio"], hide_ratio)]
            d3_frame = selected.rename(
                columns={"true_D3": "true", "prediction_D3": "prediction"}
            )
            hm_frame = selected.rename(
                columns={"true_D3HM": "true", "prediction_D3HM": "prediction"}
            )
            record: dict[str, object] = {
                "pollutant": pollutant,
                "hide_ratio": float(hide_ratio),
                "n_available_stations_mean": float(
                    hm_only[np.isclose(hm_only["hide_ratio"], hide_ratio)][
                        "n_available_stations"
                    ].mean()
                ),
                "n_prediction_rows": int(len(selected)),
            }
            record.update(metric_row(d3_frame, "D3"))
            record.update(metric_row(hm_frame, "D3HM"))
            record["D3HM_minus_D3_MAE"] = (
                record["D3HM_MAE"] - record["D3_MAE"]
            )
            record["D3HM_minus_D3_RMSE"] = (
                record["D3HM_RMSE"] - record["D3_RMSE"]
            )
            record["D3HM_minus_D3_R2"] = (
                record["D3HM_R2"] - record["D3_R2"]
            )
            record["relative_MAE_change_percent"] = 100.0 * (
                record["D3HM_MAE"] / record["D3_MAE"] - 1.0
            )
            record["relative_RMSE_change_percent"] = 100.0 * (
                record["D3HM_RMSE"] / record["D3_RMSE"] - 1.0
            )
            station_selected = station_pair[
                np.isclose(station_pair["hide_ratio"], hide_ratio)
            ]
            record["n_paired_stations"] = int(len(station_selected))
            record["paired_wilcoxon_MAE_p"] = paired_pvalue(
                station_selected["MAE_D3HM"], station_selected["MAE_D3"]
            )
            record["paired_wilcoxon_RMSE_p"] = paired_pvalue(
                station_selected["RMSE_D3HM"], station_selected["RMSE_D3"]
            )
            record["RMSE_winner"] = (
                "D3HM"
                if record["D3HM_RMSE"] < record["D3_RMSE"]
                else "D3"
            )
            comparison_rows.append(record)

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["pollutant", "hide_ratio"]
        )
        output = Path(
            config.RESULT_DIR,
            "metrics",
            "d3hm_vs_d3_density_comparison.csv",
        )
        comparison.to_csv(output, index=False)
        print("\nD3-HM versus D3 paired density comparison")
        columns = [
            "pollutant",
            "hide_ratio",
            "D3_RMSE",
            "D3HM_RMSE",
            "relative_RMSE_change_percent",
            "D3_R2",
            "D3HM_R2",
            "paired_wilcoxon_RMSE_p",
            "RMSE_winner",
        ]
        print(comparison[columns].to_string(index=False))
        print(f"Paired comparison saved to: {output}")


def install_adapter() -> None:
    """Install D3-HM components into the shared density-test engine."""

    density.Config = Config
    density.ALLOWED_MODELS = ALLOWED_MODELS
    density.DensityStressDataset = D3HMDensityStressDataset
    density.validate_config = validate_config
    density.prepare_output = prepare_output
    density.verify_zero_hidden_equivalence = verify_zero_hidden_equivalence
    density.load_completed_model = load_completed_model
    density.scunet_predictions = d3hm_predictions
    density.scenario_rows = scenario_rows
    density.run_fold = run_fold
    density.DEVICE = d3hm.DEVICE
    density.core = d3hm.core


def main() -> None:
    install_adapter()
    density.main()
    compare_with_existing_d3(Config())


if __name__ == "__main__":
    main()
