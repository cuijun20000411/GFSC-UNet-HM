"""Zero-shot observation-density stress test for strict SC-UNet.

This experiment evaluates an already trained outer-LOSO SC-UNet when fewer of
the remaining monitoring stations are available at test time.  It does not
retrain the network.

Task definition
---------------
For outer fold ``s``:

* Station ``s`` is absent from training, validation, fold normalization, model
  selection and every test input.  Its test value is used only as final truth.
* A deterministic subset of the other 54 stations is additionally hidden from
  the test input.  The subset is fixed across all test times in one repeat,
  simulating a fixed observation-availability pattern.
* Additional hidden stations are context removals only; they are not prediction
  targets.  Their earlier training histories may have influenced the already
  trained model.  Therefore this is an inference-time availability stress test,
  not a claim that those additional stations never existed historically.
* SC-UNet, SpatialMean, NearestNeighbor, IDW and KNN use exactly the same
  available station subset and the same test samples.  Optional RBF is provided
  but disabled by default because repeated fitting is expensive.
* IDW power and KNN k are selected using only inner validation pseudo-targets
  drawn from non-held-out stations.  Outer test truth is never used for tuning.
* Predictions are evaluated without non-negative clipping.

Default additional hidden-station ratios are 0%, 25%, 50% and 75%.  The 0%
condition is evaluated once; every positive ratio uses ten deterministic
station subsets by default.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import rh4_final_strict_light as core
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Place density_stress_final_strict.py in the same directory as "
        "rh4_final_strict_light.py."
    ) from exc


GLOBAL_SEED = int(os.environ.get("DENSITY_SEED", "42"))
DEVICE = core.DEVICE
core.GLOBAL_SEED = GLOBAL_SEED


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((GLOBAL_SEED + offset) % (2**31 - 1))


seed_everything(GLOBAL_SEED)


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("DENSITY_HIDE_RATIOS cannot be empty")
    for ratio in result:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"Invalid hidden-station ratio: {ratio}")
    return sorted(set(result))


class Config(core.Config):
    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    MODEL_ROOT = os.environ.get(
        "SCUNET_MODEL_ROOT",
        os.environ.get(
            "SCUNET_RESULT_ROOT", "/etc/cj/boundary/rh4_final_strict_light_v1"
        ),
    )
    RESULT_DIR = os.environ.get(
        "DENSITY_RESULT_ROOT",
        "/etc/cj/boundary/density_stress_final_strict_v1",
    )
    POLLUTANTS = [
        item.strip()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    HIDE_RATIOS = parse_float_list(
        os.environ.get("DENSITY_HIDE_RATIOS", "0,0.25,0.50,0.75")
    )
    REPEATS = int(os.environ.get("DENSITY_REPEATS", "10"))
    MODELS = [
        item.strip()
        for item in os.environ.get(
            "DENSITY_MODELS", "SCUNet,SpatialMean,NearestNeighbor,IDW,KNN"
        ).split(",")
        if item.strip()
    ]
    IDW_POWER_CANDIDATES = (1.0, 1.5, 2.0)
    KNN_CANDIDATES = (3, 4, 6, 8)
    RBF_FUNCTION = "linear"
    RBF_SMOOTH = 0.1
    EVAL_BATCH_SIZE = int(os.environ.get("DENSITY_BATCH_SIZE", "32"))
    NUM_WORKERS = int(
        os.environ.get("DENSITY_NUM_WORKERS", str(core.Config.NUM_WORKERS))
    )
    FOLD_START = int(os.environ.get("DENSITY_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("DENSITY_MAX_STATIONS", "0"))
    REUSE_COMPLETED = os.environ.get("DENSITY_REUSE_COMPLETED", "1") == "1"
    SAVE_RAW_PREDICTIONS = (
        os.environ.get("DENSITY_SAVE_RAW_PREDICTIONS", "1") == "1"
    )
    BOOTSTRAP_REPEATS = int(os.environ.get("DENSITY_BOOTSTRAP_REPEATS", "2000"))
    EXPECTED_MODEL_VERSION = os.environ.get(
        "SCUNET_MODEL_VERSION", "strict_light_architecture_v1"
    )
    EXPERIMENT_VERSION = "zero_shot_density_stress_v1"


ALLOWED_MODELS = {
    "SCUNet",
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
    if "SCUNet" not in config.MODELS:
        raise ValueError("SCUNet must be included in DENSITY_MODELS")
    if config.REPEATS < 1:
        raise ValueError("DENSITY_REPEATS must be at least 1")
    if not any(np.isclose(ratio, 0.0) for ratio in config.HIDE_RATIOS):
        raise ValueError(
            "DENSITY_HIDE_RATIOS must include 0 so degradation has a reference"
        )


def prepare_output(config: Config) -> None:
    for name in ("folds", "metrics", "predictions", "figures"):
        Path(config.RESULT_DIR, name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_version": config.EXPERIMENT_VERSION,
        "seed": GLOBAL_SEED,
        "device": str(DEVICE),
        "model_root": str(config.MODEL_ROOT),
        "expected_model_version": config.EXPECTED_MODEL_VERSION,
        "pollutants": config.POLLUTANTS,
        "models": config.MODELS,
        "additional_hidden_station_ratios": config.HIDE_RATIOS,
        "positive_ratio_repeats": config.REPEATS,
        "zero_ratio_repeats": 1,
        "subset_definition": (
            "fixed station subset within each outer-fold/ratio/repeat; "
            "shared by all methods and all test times"
        ),
        "outer_target_history_used": False,
        "additional_hidden_station_history_used_during_original_training": True,
        "interpretation": "zero-shot inference-time observation availability stress test",
        "retraining": False,
        "prediction_clipping": False,
        "idw_knn_tuning": "inner validation stations only; outer station excluded",
        "zero_hidden_equivalence_check": (
            "custom stress-test input is checked against the original strict "
            "test dataset at the first, middle and last test samples"
        ),
    }
    Path(config.RESULT_DIR, "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class DensityStressDataset(core.SpatialInpaintDatasetLOOCV):
    """Strict outer test dataset with an additional fixed hidden-station set."""

    def __init__(
        self,
        config: Config,
        pollutant: str,
        heldout_station: int,
        additionally_hidden: tuple[int, ...],
    ):
        self.additionally_hidden = tuple(sorted(additionally_hidden))
        super().__init__(
            config,
            pollutant,
            mask_station_idx=heldout_station,
            mode="test",
        )
        if heldout_station in self.additionally_hidden:
            raise ValueError("Outer held-out station cannot be an additional hidden station")
        valid_station_ids = set(range(self.total_stations)) - {heldout_station}
        if not set(self.additionally_hidden).issubset(valid_station_ids):
            raise ValueError("Additional hidden-station set contains an invalid station")

    @property
    def available_station_ids(self) -> tuple[int, ...]:
        excluded = set(self.additionally_hidden) | {self.mask_station_idx}
        return tuple(
            station
            for station in range(self.total_stations)
            if station not in excluded
        )

    def normalized_image(self, array_idx: int) -> np.ndarray:
        return self._to_fold_normalized(
            np.asarray(self.conc_img_raw[array_idx], dtype=np.float32)
        ).copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        array_idx = int(self.time_indices[idx])
        original_time_idx = int(self.valid_time_indices_original[array_idx])
        sparse = self.normalized_image(array_idx)
        covariates = np.asarray(
            self.cov_img_full[array_idx], dtype=np.float32
        ).copy()
        target = np.full(self.config.IMG_SHAPE, np.nan, dtype=np.float32)

        target_row, target_col = self.station_positions[self.mask_station_idx]
        heldout_value = sparse[target_row, target_col]
        sparse[target_row, target_col] = np.nan
        target[target_row, target_col] = heldout_value

        for station in self.additionally_hidden:
            row, col = self.station_positions[station]
            sparse[row, col] = np.nan

        observed_mask = np.isfinite(sparse) & (self.valid_grid_mask > 0)
        known_values = sparse.copy()
        observed_values = sparse[observed_mask]
        sample_mean = (
            float(observed_values.mean())
            if observed_values.size
            else self.fold_global_mean
        )
        sample_std = (
            float(observed_values.std())
            if observed_values.size > 1
            else self.fold_global_std
        )
        if sample_std < 1e-6:
            sample_std = self.fold_global_std

        filled = np.where(observed_mask, sparse, sample_mean).astype(np.float32)
        neighbor = core.observed_neighbor_features(
            sparse,
            observed_mask,
            self.radius,
            sample_mean,
            sample_std,
        )
        distance = distance_transform_edt(~observed_mask).astype(np.float32)
        valid_distance = distance[self.valid_grid_mask > 0]
        max_distance = (
            float(valid_distance.max())
            if valid_distance.size
            else float(distance.max())
        )
        distance /= max(max_distance, 1.0)
        distance *= self.valid_grid_mask

        base_features = np.concatenate(
            [
                filled[None],
                neighbor,
                self.coord_feat,
                distance[None],
            ],
            axis=0,
        ).astype(np.float32)
        model_input = np.concatenate([base_features, covariates], axis=0)
        infer_mask = (
            (self.valid_grid_mask > 0) & (~observed_mask)
        ).astype(np.float32)
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
        }


def verify_zero_hidden_equivalence(
    config: Config,
    pollutant: str,
    station_idx: int,
    stress_dataset: DensityStressDataset,
) -> None:
    """Fail fast if the 0%-hidden input differs from the strict main test."""

    reference = core.SpatialInpaintDatasetLOOCV(
        config,
        pollutant,
        mask_station_idx=station_idx,
        mode="test",
    )
    if len(reference) != len(stress_dataset):
        raise RuntimeError("Zero-hidden stress and strict test lengths differ")
    if not (
        np.isclose(reference.fold_mean, stress_dataset.fold_mean)
        and np.isclose(reference.fold_std, stress_dataset.fold_std)
    ):
        raise RuntimeError("Zero-hidden stress and strict fold scalers differ")

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
        if expected["time_idx"] != actual["time_idx"]:
            raise RuntimeError("Zero-hidden stress and strict test times differ")
        for key in tensor_keys:
            if not torch.allclose(
                expected[key], actual[key], atol=1e-6, rtol=1e-6, equal_nan=True
            ):
                raise RuntimeError(
                    f"Zero-hidden stress input differs from strict test: "
                    f"station={station_idx}, sample={index}, key={key}"
                )


def checkpoint_paths(
    config: Config, pollutant: str, station_idx: int
) -> tuple[Path, Path]:
    checkpoint = Path(
        config.MODEL_ROOT,
        "models",
        f"{pollutant}_station_{station_idx}_scunet.pth",
    )
    return checkpoint, Path(str(checkpoint) + ".complete.json")


def checkpoint_signature(checkpoint: Path) -> dict[str, int]:
    stat = checkpoint.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_completed_model(
    config: Config,
    pollutant: str,
    station_idx: int,
    probe: DensityStressDataset,
) -> tuple[core.SCUNet, dict[str, int]]:
    checkpoint, marker_path = checkpoint_paths(config, pollutant, station_idx)
    if not checkpoint.exists() or not marker_path.exists():
        raise FileNotFoundError(
            f"Completed strict-light checkpoint is missing for {pollutant} "
            f"station {station_idx}: {checkpoint}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid checkpoint marker: {marker_path}") from exc
    compatible = (
        marker.get("status") == "completed"
        and marker.get("model_version") == config.EXPECTED_MODEL_VERSION
        and marker.get("pollutant") == pollutant
        and marker.get("mask_station_idx") == station_idx
        and marker.get("feature_dim") == probe.F_dim
    )
    if not compatible:
        raise RuntimeError(
            f"Incompatible checkpoint marker for {pollutant} station {station_idx}: "
            f"{marker_path}"
        )
    model = core.SCUNet(probe.base_channels, probe.F_dim).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model.eval(), checkpoint_signature(checkpoint)


def hidden_subset(
    total_stations: int,
    heldout_station: int,
    hide_ratio: float,
    repeat: int,
) -> tuple[int, ...]:
    candidates = np.asarray(
        [station for station in range(total_stations) if station != heldout_station],
        dtype=np.int64,
    )
    hide_count = int(round(len(candidates) * hide_ratio))
    hide_count = min(max(hide_count, 0), len(candidates) - 1)
    if hide_count == 0:
        return ()
    rng = np.random.default_rng(
        stable_seed("subset", heldout_station, f"{hide_ratio:.8f}", repeat)
    )
    selected = rng.choice(candidates, size=hide_count, replace=False)
    return tuple(sorted(int(item) for item in selected))


def distance_and_values(
    dataset: DensityStressDataset,
    normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    available = np.asarray(dataset.available_station_ids, dtype=np.int64)
    coords = dataset.station_positions[available].astype(np.float64)
    rows = dataset.station_positions[available, 0]
    cols = dataset.station_positions[available, 1]
    values = normalized[rows, cols].astype(np.float64)
    finite = np.isfinite(values)
    coords = coords[finite]
    values = values[finite]
    target_coord = dataset.station_positions[dataset.mask_station_idx].astype(
        np.float64
    )
    distances = np.linalg.norm(coords - target_coord[None, :], axis=1)
    target_row, target_col = dataset.station_positions[dataset.mask_station_idx]
    target_value = float(normalized[target_row, target_col])
    return distances, coords, values, target_value


def spatial_mean(distances: np.ndarray, values: np.ndarray) -> float:
    del distances
    return float(np.mean(values))


def nearest_neighbor(distances: np.ndarray, values: np.ndarray) -> float:
    return float(values[int(np.argmin(distances))])


def idw_predict(distances: np.ndarray, values: np.ndarray, power: float) -> float:
    zero = np.flatnonzero(distances < 1e-8)
    if zero.size:
        return float(values[int(zero[0])])
    weights = 1.0 / np.power(np.maximum(distances, 1e-8), power)
    return float(np.sum(weights * values) / np.sum(weights))


def knn_predict(distances: np.ndarray, values: np.ndarray, k: int) -> float:
    order = np.argsort(distances)[: min(k, len(distances))]
    selected_distances = distances[order]
    selected_values = values[order]
    zero = np.flatnonzero(selected_distances < 1e-8)
    if zero.size:
        return float(selected_values[int(zero[0])])
    weights = 1.0 / np.maximum(selected_distances, 1e-8)
    return float(np.sum(weights * selected_values) / np.sum(weights))


def rbf_predict(
    coords: np.ndarray,
    values: np.ndarray,
    target_coord: np.ndarray,
    function: str,
    smooth: float,
) -> float:
    try:
        interpolator = Rbf(
            coords[:, 0],
            coords[:, 1],
            values,
            function=function,
            smooth=smooth,
        )
        prediction = float(interpolator(target_coord[0], target_coord[1]))
        return prediction if np.isfinite(prediction) else float(np.mean(values))
    except Exception:
        return float(np.mean(values))


def select_inner_hyperparameters(
    dataset: DensityStressDataset, config: Config
) -> tuple[float, int]:
    """Tune IDW/KNN only on validation times and non-outer stations."""

    splits = np.load(
        Path(dataset.data_dir, "time_splits.npy"), allow_pickle=True
    ).item()
    validation_indices = np.asarray(splits["val"], dtype=np.int64)
    safe_stations = [
        station
        for station in range(dataset.total_stations)
        if station != dataset.mask_station_idx
    ]
    idw_errors = {power: [] for power in config.IDW_POWER_CANDIDATES}
    knn_errors = {k: [] for k in config.KNN_CANDIDATES}

    for array_idx in validation_indices:
        normalized = dataset.normalized_image(int(array_idx))
        for pseudo_target in safe_stations:
            target_row, target_col = dataset.station_positions[pseudo_target]
            target_value = float(normalized[target_row, target_col])
            if not np.isfinite(target_value):
                continue
            known_ids = [
                station for station in safe_stations if station != pseudo_target
            ]
            rows = dataset.station_positions[known_ids, 0]
            cols = dataset.station_positions[known_ids, 1]
            values = normalized[rows, cols].astype(np.float64)
            coords = dataset.station_positions[known_ids].astype(np.float64)
            finite = np.isfinite(values)
            values = values[finite]
            coords = coords[finite]
            if len(values) < 4:
                continue
            target_coord = dataset.station_positions[pseudo_target].astype(np.float64)
            distances = np.linalg.norm(coords - target_coord[None, :], axis=1)
            for power in config.IDW_POWER_CANDIDATES:
                idw_errors[power].append(
                    abs(idw_predict(distances, values, power) - target_value)
                )
            for k in config.KNN_CANDIDATES:
                knn_errors[k].append(
                    abs(knn_predict(distances, values, k) - target_value)
                )

    best_power = min(
        idw_errors,
        key=lambda power: float(np.mean(idw_errors[power]))
        if idw_errors[power]
        else float("inf"),
    )
    best_k = min(
        knn_errors,
        key=lambda k: float(np.mean(knn_errors[k]))
        if knn_errors[k]
        else float("inf"),
    )
    return float(best_power), int(best_k)


def scunet_predictions(
    model: core.SCUNet,
    dataset: DensityStressDataset,
    config: Config,
) -> dict[int, tuple[float, float]]:
    loader = core.make_loader(
        dataset,
        config,
        shuffle=False,
        batch_size=config.EVAL_BATCH_SIZE,
    )
    target_row, target_col = dataset.station_positions[dataset.mask_station_idx]
    result: dict[int, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, targets, _ = core.move_batch(batch)
            output = model(inputs, observed, known)
            predictions = output[:, 0, target_row, target_col].cpu().numpy()
            truths = targets[:, 0, target_row, target_col].cpu().numpy()
            times = batch["time_idx"].cpu().numpy()
            for time_idx, true_value, prediction in zip(times, truths, predictions):
                if np.isfinite(true_value) and np.isfinite(prediction):
                    result[int(time_idx)] = (float(true_value), float(prediction))
    return result


def scenario_rows(
    config: Config,
    pollutant: str,
    station_idx: int,
    hide_ratio: float,
    repeat: int,
    hidden: tuple[int, ...],
    model: core.SCUNet,
    best_idw_power: float,
    best_knn_k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset = DensityStressDataset(
        config,
        pollutant,
        heldout_station=station_idx,
        additionally_hidden=hidden,
    )
    available_count = len(dataset.available_station_ids)
    rows: list[dict[str, object]] = []
    neural = scunet_predictions(model, dataset, config)

    for local_idx, array_idx in enumerate(dataset.time_indices):
        array_idx = int(array_idx)
        original_time = int(dataset.valid_time_indices_original[array_idx])
        normalized = dataset.normalized_image(array_idx)
        distances, coords, known_values, true_normalized = distance_and_values(
            dataset, normalized
        )
        if len(known_values) < 2 or not np.isfinite(true_normalized):
            continue
        target_coord = dataset.station_positions[station_idx].astype(np.float64)

        predictors: dict[str, Callable[[], float]] = {
            "SpatialMean": lambda: spatial_mean(distances, known_values),
            "NearestNeighbor": lambda: nearest_neighbor(distances, known_values),
            "IDW": lambda: idw_predict(
                distances, known_values, best_idw_power
            ),
            "KNN": lambda: knn_predict(distances, known_values, best_knn_k),
            "RBF": lambda: rbf_predict(
                coords,
                known_values,
                target_coord,
                config.RBF_FUNCTION,
                config.RBF_SMOOTH,
            ),
        }
        normalized_predictions = {}
        if "SCUNet" in config.MODELS and original_time in neural:
            neural_true, neural_prediction = neural[original_time]
            if not np.isclose(neural_true, true_normalized, atol=1e-5, rtol=1e-5):
                raise RuntimeError("SCUNet and spatial baseline test targets are misaligned")
            normalized_predictions["SCUNet"] = neural_prediction
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


def fold_paths(
    config: Config, pollutant: str, station_idx: int
) -> tuple[Path, Path, Path, Path]:
    stem = f"{pollutant}_station_{station_idx:02d}"
    return (
        Path(config.RESULT_DIR, "folds", f"{stem}_predictions.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}_subsets.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}_runtime.json"),
        Path(config.RESULT_DIR, "folds", f"{stem}.complete.json"),
    )


def reusable_fold(
    config: Config,
    pollutant: str,
    station_idx: int,
    expected_checkpoint_signature: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    prediction_path, subset_path, runtime_path, marker_path = fold_paths(
        config, pollutant, station_idx
    )
    if not config.REUSE_COMPLETED:
        return None
    if not all(
        path.exists()
        for path in (prediction_path, subset_path, runtime_path, marker_path)
    ):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        compatible = (
            marker.get("status") == "completed"
            and marker.get("experiment_version") == config.EXPERIMENT_VERSION
            and marker.get("seed") == GLOBAL_SEED
            and marker.get("pollutant") == pollutant
            and marker.get("station") == station_idx
            and marker.get("hide_ratios") == config.HIDE_RATIOS
            and marker.get("repeats") == config.REPEATS
            and marker.get("models") == config.MODELS
            and marker.get("checkpoint_signature")
            == expected_checkpoint_signature
        )
        if not compatible:
            return None
        return (
            pd.read_csv(prediction_path),
            pd.read_csv(subset_path),
            json.loads(runtime_path.read_text(encoding="utf-8")),
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def save_completed_fold(
    config: Config,
    pollutant: str,
    station_idx: int,
    predictions: pd.DataFrame,
    subsets: pd.DataFrame,
    runtime: dict,
    model_signature: dict[str, int],
) -> None:
    prediction_path, subset_path, runtime_path, marker_path = fold_paths(
        config, pollutant, station_idx
    )
    predictions.to_csv(prediction_path, index=False)
    subsets.to_csv(subset_path, index=False)
    runtime_path.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = {
        "status": "completed",
        "experiment_version": config.EXPERIMENT_VERSION,
        "seed": GLOBAL_SEED,
        "pollutant": pollutant,
        "station": station_idx,
        "hide_ratios": config.HIDE_RATIOS,
        "repeats": config.REPEATS,
        "models": config.MODELS,
        "checkpoint_signature": model_signature,
        "n_prediction_rows": len(predictions),
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_fold(
    config: Config, pollutant: str, station_idx: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    probe = DensityStressDataset(
        config,
        pollutant,
        heldout_station=station_idx,
        additionally_hidden=(),
    )
    verify_zero_hidden_equivalence(
        config, pollutant, station_idx, probe
    )
    model, model_signature = load_completed_model(
        config, pollutant, station_idx, probe
    )
    reused = reusable_fold(
        config, pollutant, station_idx, model_signature
    )
    if reused is not None:
        print(f"  station {station_idx:02d}: reuse completed density fold")
        del model, probe
        return reused

    started = time.time()
    best_idw_power, best_knn_k = select_inner_hyperparameters(probe, config)
    prediction_rows = []
    subset_rows = []
    print(
        f"  station {station_idx:02d}: IDW power={best_idw_power}, "
        f"KNN k={best_knn_k}"
    )

    for hide_ratio in config.HIDE_RATIOS:
        repeats = 1 if np.isclose(hide_ratio, 0.0) else config.REPEATS
        for repeat in range(repeats):
            hidden = hidden_subset(
                probe.total_stations, station_idx, hide_ratio, repeat
            )
            rows, subset_record = scenario_rows(
                config,
                pollutant,
                station_idx,
                hide_ratio,
                repeat,
                hidden,
                model,
                best_idw_power,
                best_knn_k,
            )
            prediction_rows.extend(rows)
            subset_rows.append(subset_record)
        available = probe.total_stations - 1 - int(
            round((probe.total_stations - 1) * hide_ratio)
        )
        print(
            f"    hide={hide_ratio:.0%} | available≈{available} | "
            f"repeats={repeats}"
        )

    predictions = pd.DataFrame(prediction_rows)
    subsets = pd.DataFrame(subset_rows)
    actual_models = sorted(predictions["model"].unique().tolist())
    expected_models = sorted(config.MODELS)
    if actual_models != expected_models:
        raise RuntimeError(
            f"Density fold model mismatch: expected {expected_models}, got {actual_models}"
        )
    runtime = {
        "pollutant": pollutant,
        "station": station_idx,
        "seconds": time.time() - started,
        "n_scenarios": len(subsets),
        "n_prediction_rows": len(predictions),
    }
    save_completed_fold(
        config,
        pollutant,
        station_idx,
        predictions,
        subsets,
        runtime,
        model_signature,
    )
    del model, probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, subsets, runtime


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) < 1e-12:
        return np.nan
    return float(r2_score(y_true, y_pred))


def scenario_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "pollutant",
        "model",
        "hide_ratio",
        "station",
        "repeat",
        "n_available_stations",
    ]
    for keys, group in predictions.groupby(group_columns, sort=True):
        y_true = group["true"].to_numpy(dtype=np.float64)
        y_pred = group["prediction"].to_numpy(dtype=np.float64)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_test": len(group),
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "R2": safe_r2(y_true, y_pred),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2 or repeats <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def cluster_bootstrap_pooled_ci(
    group: pd.DataFrame, repeats: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Bootstrap pooled metrics by resampling complete outer-station blocks."""

    blocks = []
    for _, station_group in group.groupby("station", sort=True):
        y_true = station_group["true"].to_numpy(dtype=np.float64)
        y_pred = station_group["prediction"].to_numpy(dtype=np.float64)
        residual = y_pred - y_true
        blocks.append(
            (
                float(len(station_group)),
                float(np.abs(residual).sum()),
                float(np.square(residual).sum()),
                float(y_true.sum()),
                float(np.square(y_true).sum()),
            )
        )
    if len(blocks) < 2 or repeats <= 0:
        nan_ci = (np.nan, np.nan)
        return {"MAE": nan_ci, "RMSE": nan_ci, "R2": nan_ci}

    statistics = np.asarray(blocks, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(statistics), size=(repeats, len(statistics))
    )
    sampled = statistics[indices].sum(axis=1)
    n = sampled[:, 0]
    mae = sampled[:, 1] / n
    rmse = np.sqrt(sampled[:, 2] / n)
    centered_sum = sampled[:, 4] - np.square(sampled[:, 3]) / n
    r2 = np.full(repeats, np.nan, dtype=np.float64)
    valid = centered_sum > 1e-12
    r2[valid] = 1.0 - sampled[valid, 2] / centered_sum[valid]

    def interval(values: np.ndarray) -> tuple[float, float]:
        values = values[np.isfinite(values)]
        if values.size < 2:
            return np.nan, np.nan
        return (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )

    return {
        "MAE": interval(mae),
        "RMSE": interval(rmse),
        "R2": interval(r2),
    }


def aggregate_density_metrics(
    predictions: pd.DataFrame,
    scenario_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    station_average = (
        scenario_df.groupby(
            ["pollutant", "model", "hide_ratio", "station"], as_index=False
        )[["MAE", "RMSE", "R2"]]
        .mean()
    )
    rows = []
    for (pollutant, model, hide_ratio), group in predictions.groupby(
        ["pollutant", "model", "hide_ratio"], sort=True
    ):
        y_true = group["true"].to_numpy(dtype=np.float64)
        y_pred = group["prediction"].to_numpy(dtype=np.float64)
        station_group = station_average[
            (station_average["pollutant"] == pollutant)
            & (station_average["model"] == model)
            & np.isclose(station_average["hide_ratio"], hide_ratio)
        ]
        ci = {}
        for metric in ("MAE", "RMSE", "R2"):
            ci[metric] = bootstrap_mean_ci(
                station_group[metric].to_numpy(dtype=np.float64),
                config.BOOTSTRAP_REPEATS,
                stable_seed(pollutant, model, hide_ratio, metric),
            )
        pooled_ci = cluster_bootstrap_pooled_ci(
            group,
            config.BOOTSTRAP_REPEATS,
            stable_seed(pollutant, model, hide_ratio, "pooled"),
        )
        rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "hide_ratio": hide_ratio,
                "mean_available_stations": group["n_available_stations"].mean(),
                "n_outer_stations": station_group["station"].nunique(),
                "n_network_repeats": group[["station", "repeat"]]
                .drop_duplicates()
                .shape[0],
                "n_prediction_rows": len(group),
                "pooled_MAE": mean_absolute_error(y_true, y_pred),
                "pooled_MAE_CI_low": pooled_ci["MAE"][0],
                "pooled_MAE_CI_high": pooled_ci["MAE"][1],
                "pooled_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "pooled_RMSE_CI_low": pooled_ci["RMSE"][0],
                "pooled_RMSE_CI_high": pooled_ci["RMSE"][1],
                "pooled_R2": safe_r2(y_true, y_pred),
                "pooled_R2_CI_low": pooled_ci["R2"][0],
                "pooled_R2_CI_high": pooled_ci["R2"][1],
                "mean_station_MAE": station_group["MAE"].mean(),
                "mean_station_MAE_CI_low": ci["MAE"][0],
                "mean_station_MAE_CI_high": ci["MAE"][1],
                "mean_station_RMSE": station_group["RMSE"].mean(),
                "mean_station_RMSE_CI_low": ci["RMSE"][0],
                "mean_station_RMSE_CI_high": ci["RMSE"][1],
                "mean_station_R2": station_group["R2"].mean(),
                "mean_station_R2_CI_low": ci["R2"][0],
                "mean_station_R2_CI_high": ci["R2"][1],
            }
        )
    aggregate = pd.DataFrame(rows)

    degradation_rows = []
    for (pollutant, model), group in aggregate.groupby(
        ["pollutant", "model"], sort=True
    ):
        reference = group[np.isclose(group["hide_ratio"], 0.0)]
        if reference.empty:
            continue
        base = reference.iloc[0]
        for _, row in group.iterrows():
            record = row.to_dict()
            record["MAE_increase_percent_vs_0"] = (
                (row["pooled_MAE"] / base["pooled_MAE"] - 1.0) * 100.0
                if base["pooled_MAE"] > 1e-12
                else np.nan
            )
            record["RMSE_increase_percent_vs_0"] = (
                (row["pooled_RMSE"] / base["pooled_RMSE"] - 1.0) * 100.0
                if base["pooled_RMSE"] > 1e-12
                else np.nan
            )
            record["R2_drop_vs_0"] = base["pooled_R2"] - row["pooled_R2"]
            degradation_rows.append(record)
    return pd.DataFrame(degradation_rows).sort_values(
        ["pollutant", "model", "hide_ratio"]
    )


def plot_density_curves(
    aggregate: pd.DataFrame, config: Config, pollutant: str
) -> None:
    specifications = [
        ("pooled_MAE", "pooled_MAE_CI_low", "pooled_MAE_CI_high", "MAE"),
        (
            "pooled_RMSE",
            "pooled_RMSE_CI_low",
            "pooled_RMSE_CI_high",
            "RMSE",
        ),
        ("pooled_R2", "pooled_R2_CI_low", "pooled_R2_CI_high", "$R^2$"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for model_name, model_rows in aggregate.groupby("model", sort=True):
        model_rows = model_rows.sort_values("hide_ratio")
        x = model_rows["hide_ratio"].to_numpy(dtype=np.float64) * 100.0
        for axis, (metric, low, high, label) in zip(axes, specifications):
            y = model_rows[metric].to_numpy(dtype=np.float64)
            line = axis.plot(x, y, marker="o", linewidth=1.8, label=model_name)[0]
            lower = model_rows[low].to_numpy(dtype=np.float64)
            upper = model_rows[high].to_numpy(dtype=np.float64)
            axis.fill_between(
                x,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.10,
                linewidth=0,
            )
            axis.set_xlabel("Additionally hidden stations (%)")
            axis.set_ylabel(label)
            axis.set_xticks(x)
            axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(5, max(1, len(labels))),
        frameon=False,
    )
    figure.suptitle(
        f"{pollutant.upper()} zero-shot observation-density stress test",
        y=1.12,
    )
    figure.tight_layout()
    output = Path(
        config.RESULT_DIR,
        "figures",
        f"{pollutant}_density_stress_curve.png",
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_pollutant(config: Config, pollutant: str) -> None:
    probe = DensityStressDataset(
        config,
        pollutant,
        heldout_station=0,
        additionally_hidden=(),
    )
    total_stations = probe.total_stations
    del probe
    stop = total_stations
    if config.MAX_STATIONS > 0:
        stop = min(total_stations, config.FOLD_START + config.MAX_STATIONS)

    all_predictions = []
    all_subsets = []
    runtimes = []
    print(
        f"\n{pollutant.upper()}: zero-shot density stress folds "
        f"{config.FOLD_START}..{stop - 1}"
    )
    for station_idx in range(config.FOLD_START, stop):
        predictions, subsets, runtime = run_fold(
            config, pollutant, station_idx
        )
        all_predictions.append(predictions)
        all_subsets.append(subsets)
        runtimes.append(runtime)

    prediction_df = pd.concat(all_predictions, ignore_index=True)
    subset_df = pd.concat(all_subsets, ignore_index=True)
    scenario_df = scenario_metrics(prediction_df)
    aggregate_df = aggregate_density_metrics(prediction_df, scenario_df, config)

    if config.SAVE_RAW_PREDICTIONS:
        prediction_df.to_csv(
            Path(config.RESULT_DIR, "predictions", f"{pollutant}_predictions.csv"),
            index=False,
        )
    subset_df.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_hidden_subsets.csv"),
        index=False,
    )
    scenario_df.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_scenario_metrics.csv"),
        index=False,
    )
    aggregate_df.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_density_metrics.csv"),
        index=False,
    )
    plot_density_curves(aggregate_df, config, pollutant)
    pd.DataFrame(runtimes).to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_runtime.csv"),
        index=False,
    )
    display_columns = [
        "model",
        "hide_ratio",
        "mean_available_stations",
        "pooled_MAE",
        "pooled_RMSE",
        "pooled_R2",
        "RMSE_increase_percent_vs_0",
    ]
    print(aggregate_df[display_columns].to_string(index=False))


def main() -> None:
    config = Config()
    validate_config(config)
    prepare_output(config)
    print(
        f"device={DEVICE} | seed={GLOBAL_SEED} | "
        f"model_root={config.MODEL_ROOT} | models={','.join(config.MODELS)}"
    )
    for pollutant in config.POLLUTANTS:
        run_pollutant(config, pollutant)


if __name__ == "__main__":
    main()
