"""Strict and reproducible baselines for zero-history station-to-grid inference.

This script is aligned with ``rh4_final_strict_light.py``:

* The outer LOSO station is absent from training, inner validation, target
  scaling and model selection for every time step.
* Only same-time observations from the remaining stations are used.  No
  temporal history of the outer station is used.
* Concentration normalization is fitted independently in each outer fold from
  training times and non-held-out stations only.
* Deep baselines use deterministic inner validation masks drawn only from the
  non-held-out stations and early stopping on that inner validation loss.
* Test truth is accessed only after each model has been fully specified.
* Predictions are evaluated without non-negative clipping.  Raw per-time,
  station-level and pooled metrics are saved separately.

The implemented comparison groups are:

1. Spatial interpolation: SpatialMean, NearestNeighbor, IDW, KNN, RBF and
   optional OrdinaryKriging (requires ``pykrige``).
2. Tabular learning: RandomForest, ExtraTrees and optional XGBoost.
3. Deep learning: NeighborTransformer, SparseGridCNN, CovariateUNet and a real
   mask-renormalized PartialConvUNet.

``SparseGridCNN`` is an adapted sparse-grid CNN baseline, not an exact CMSI
reproduction.  The published CMSI model uses a spatially complete proxy field;
calling this network "CMSI" without that proxy would be misleading.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import Rbf
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from pykrige.ok import OrdinaryKriging
except Exception:
    OrdinaryKriging = None


warnings.filterwarnings("ignore")

GLOBAL_SEED = int(os.environ.get("BASELINE_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible fold-level training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((GLOBAL_SEED + offset) % (2**31 - 1))


seed_everything(GLOBAL_SEED)


class Config:
    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    RESULT_DIR = os.environ.get(
        "BASELINE_RESULT_ROOT", "/etc/cj/boundary/baselines_final_strict_v1"
    )
    POLLUTANTS = [
        item.strip()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    EXPECTED_FEATURE_DIMS = {"no2": 39, "o3": 39, "pm25": 40}
    IMG_SHAPE = (32, 32)
    NUM_GRIDS = 999

    DEFAULT_MODELS = (
        "SpatialMean,NearestNeighbor,IDW,KNN,RBF,OrdinaryKriging,"
        "RandomForest,ExtraTrees,XGBoost,NeighborTransformer,"
        "SparseGridCNN,CovariateUNet,PartialConvUNet"
    )
    MODELS = [
        item.strip()
        for item in os.environ.get("BASELINE_MODELS", DEFAULT_MODELS).split(",")
        if item.strip()
    ]

    K_NEIGHBORS = 8
    IDW_POWER_CANDIDATES = (1.0, 1.5, 2.0)
    KNN_CANDIDATES = (3, 4, 6, 8)
    RBF_FUNCTION = "linear"
    RBF_SMOOTH = 0.1

    FIXED_NEIGHBOR_RADIUS = 2
    BATCH_SIZE = 32
    EPOCHS = 150
    START_MASK_RATIO = 0.05
    FINAL_MASK_RATIO = 0.25
    CURRICULUM_EPOCH = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    NUM_WORKERS = int(os.environ.get("BASELINE_NUM_WORKERS", "4"))
    PREFETCH_FACTOR = 4
    LOG_INTERVAL = 10

    HUBER_WEIGHT = 0.8
    MSE_WEIGHT = 0.2
    IMPORTANCE_BETA = 2.0

    RF_ESTIMATORS = 300
    ET_ESTIMATORS = 300
    ML_N_JOBS = int(os.environ.get("BASELINE_ML_N_JOBS", "-1"))

    FOLD_START = int(os.environ.get("BASELINE_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("BASELINE_MAX_STATIONS", "0"))
    REUSE_COMPLETED_FOLDS = (
        os.environ.get("BASELINE_REUSE_COMPLETED", "1") == "1"
    )
    MODEL_VERSION = "strict_baselines_v1"
    BOOTSTRAP_REPEATS = int(os.environ.get("BASELINE_BOOTSTRAP_REPEATS", "2000"))


ALL_MODEL_NAMES = {
    "SpatialMean",
    "NearestNeighbor",
    "IDW",
    "KNN",
    "RBF",
    "OrdinaryKriging",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "NeighborTransformer",
    "SparseGridCNN",
    "CovariateUNet",
    "PartialConvUNet",
}

INTERPOLATION_MODELS = {
    "SpatialMean",
    "NearestNeighbor",
    "IDW",
    "KNN",
    "RBF",
    "OrdinaryKriging",
}
TABULAR_MODELS = {"RandomForest", "ExtraTrees", "XGBoost", "NeighborTransformer"}
IMAGE_MODELS = {"SparseGridCNN", "CovariateUNet", "PartialConvUNet"}


def validate_config(config: Config) -> None:
    unknown = sorted(set(config.MODELS) - ALL_MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown BASELINE_MODELS entries: {unknown}")
    if "XGBoost" in config.MODELS and xgb is None:
        print("[warning] xgboost is unavailable; XGBoost will be skipped")
        config.MODELS = [m for m in config.MODELS if m != "XGBoost"]
    if "OrdinaryKriging" in config.MODELS and OrdinaryKriging is None:
        print("[warning] pykrige is unavailable; OrdinaryKriging will be skipped")
        config.MODELS = [m for m in config.MODELS if m != "OrdinaryKriging"]
    if not config.MODELS:
        raise ValueError("No runnable baseline model was selected")


def prepare_output_dirs(config: Config) -> None:
    for name in ("metrics", "predictions", "models", "folds"):
        Path(config.RESULT_DIR, name).mkdir(parents=True, exist_ok=True)

    protocol = {
        "version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "device": str(DEVICE),
        "models": config.MODELS,
        "outer_protocol": "station-wise LOSO",
        "inner_validation": "non-held-out stations only",
        "time_split": "chronological train/val/test from preprocessing",
        "fold_target_scaling": "training times and non-held-out stations only",
        "temporal_history_at_outer_target": False,
        "prediction_clipping": False,
        "deep_mask_curriculum": [
            config.START_MASK_RATIO,
            config.FINAL_MASK_RATIO,
            config.CURRICULUM_EPOCH,
        ],
        "notes": {
            "SparseGridCNN": "adapted sparse-grid CNN; not exact CMSI",
            "PartialConvUNet": "mask-renormalized partial convolutions",
            "CovariateUNet": "plain U-Net using the same gridded inputs but no ASF/SCB",
        },
    }
    with open(
        Path(config.RESULT_DIR, "protocol.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)


@dataclass
class FoldScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class PollutantData:
    """Memory-mapped pollutant arrays with a raw-concentration fallback."""

    def __init__(self, config: Config, pollutant: str):
        self.config = config
        self.pollutant = pollutant
        self.data_dir = Path(config.DATA_ROOT_DIR, pollutant)

        raw_path = self.data_dir / "conc_img_raw.npy"
        if raw_path.exists():
            self.conc_stored = np.load(raw_path, mmap_mode="r")
            self.legacy_scaler = None
        else:
            self.conc_stored = np.load(
                self.data_dir / "conc_img_full.npy", mmap_mode="r"
            )
            self.legacy_scaler = joblib.load(self.data_dir / "conc_scaler.pkl")

        self.covariates = np.load(
            self.data_dir / "cov_img_full.npy", mmap_mode="r"
        )
        self.coord_img = np.load(self.data_dir / "coord_img.npy").astype(np.float32)
        self.coord_feat = self.coord_img.transpose(2, 0, 1).astype(np.float32)
        self.station_positions = np.load(
            self.data_dir / "station_positions.npy"
        ).astype(np.int64)
        self.time_splits = np.load(
            self.data_dir / "time_splits.npy", allow_pickle=True
        ).item()

        original_path = self.data_dir / "valid_time_indices.npy"
        if original_path.exists():
            self.original_time_indices = np.load(original_path).astype(np.int64)
        else:
            self.original_time_indices = np.arange(len(self.conc_stored), dtype=np.int64)

        valid_path = self.data_dir / "valid_grid_mask.npy"
        if valid_path.exists():
            self.valid_grid_mask = np.load(valid_path).astype(np.float32)
        else:
            flat = np.zeros(np.prod(config.IMG_SHAPE), dtype=np.float32)
            flat[: config.NUM_GRIDS] = 1.0
            self.valid_grid_mask = flat.reshape(config.IMG_SHAPE)

        self.feature_dim = int(self.covariates.shape[1])
        expected = config.EXPECTED_FEATURE_DIMS.get(pollutant)
        if expected is not None and self.feature_dim != expected:
            raise ValueError(
                f"{pollutant}: expected {expected} covariates, got {self.feature_dim}"
            )
        if tuple(self.conc_stored.shape[1:]) != tuple(config.IMG_SHAPE):
            raise ValueError(
                f"Unexpected concentration grid shape: {self.conc_stored.shape[1:]}"
            )

    @property
    def n_stations(self) -> int:
        return len(self.station_positions)

    def raw_image(self, array_idx: int) -> np.ndarray:
        stored = np.asarray(self.conc_stored[array_idx], dtype=np.float32)
        if self.legacy_scaler is None:
            return stored.copy()
        return (
            stored * float(self.legacy_scaler.scale_[0])
            + float(self.legacy_scaler.mean_[0])
        ).astype(np.float32)

    def fold_scaler(self, heldout_station: int) -> FoldScaler:
        train_indices = np.asarray(self.time_splits["train"], dtype=np.int64)
        fit_stations = [i for i in range(self.n_stations) if i != heldout_station]
        rows = self.station_positions[fit_stations, 0]
        cols = self.station_positions[fit_stations, 1]

        values = []
        for array_idx in train_indices:
            raw = self.raw_image(int(array_idx))
            values.append(raw[rows, cols])
        fit_values = np.concatenate(values)
        fit_values = fit_values[np.isfinite(fit_values)]
        if fit_values.size == 0:
            raise ValueError("No finite values available for fold normalization")
        mean = float(fit_values.mean())
        std = float(fit_values.std())
        return FoldScaler(mean=mean, std=std if std >= 1e-8 else 1.0)


@dataclass
class SpatialRecord:
    array_idx: int
    original_time_idx: int
    target_station: int
    target_coord: np.ndarray
    known_coords: np.ndarray
    known_values: np.ndarray
    target_value: float


def normalized_concentration(
    data: PollutantData, array_idx: int, scaler: FoldScaler
) -> np.ndarray:
    return scaler.transform(data.raw_image(array_idx))


def nearest_features(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    target_coord: np.ndarray,
    target_coord_features: np.ndarray,
    target_covariates: np.ndarray,
    k: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    distances = np.linalg.norm(known_coords - target_coord[None, :], axis=1)
    order = np.argsort(distances)
    use = order[: min(k, len(order))]
    values = known_values[use].astype(np.float32)
    max_distance = max(float(np.hypot(grid_shape[0] - 1, grid_shape[1] - 1)), 1.0)
    dists = (distances[use] / max_distance).astype(np.float32)

    if len(use) < k:
        pad = k - len(use)
        values = np.pad(values, (0, pad), constant_values=0.0)
        dists = np.pad(dists, (0, pad), constant_values=1.0)

    weights = 1.0 / np.maximum(distances, 1e-6)
    weighted_mean = float(np.sum(weights * known_values) / np.sum(weights))
    summary = np.asarray(
        [
            weighted_mean,
            float(np.mean(known_values)),
            float(np.std(known_values)),
            float(np.min(known_values)),
            float(np.max(known_values)),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            values,
            dists,
            summary,
            np.nan_to_num(target_coord_features, nan=0.0).astype(np.float32),
            np.nan_to_num(target_covariates, nan=0.0).astype(np.float32),
        ]
    ).astype(np.float32)


def build_tabular_split(
    data: PollutantData,
    scaler: FoldScaler,
    heldout_station: int,
    mode: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray, list[SpatialRecord]]:
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported mode: {mode}")

    features: list[np.ndarray] = []
    targets: list[float] = []
    records: list[SpatialRecord] = []
    safe_stations = [i for i in range(data.n_stations) if i != heldout_station]
    time_indices = np.asarray(data.time_splits[mode], dtype=np.int64)

    for array_idx in time_indices:
        normalized = normalized_concentration(data, int(array_idx), scaler)
        cov_t = np.asarray(data.covariates[int(array_idx)], dtype=np.float32)
        target_stations = [heldout_station] if mode == "test" else safe_stations

        for target_station in target_stations:
            tr, tc = data.station_positions[target_station]
            target_value = float(normalized[tr, tc])
            if not np.isfinite(target_value):
                continue

            known_station_ids = [
                station
                for station in safe_stations
                if station != target_station
            ]
            known_coords = data.station_positions[known_station_ids].astype(np.float32)
            rows = data.station_positions[known_station_ids, 0]
            cols = data.station_positions[known_station_ids, 1]
            known_values = normalized[rows, cols].astype(np.float32)
            finite = np.isfinite(known_values)
            known_coords = known_coords[finite]
            known_values = known_values[finite]
            if len(known_values) < 4:
                continue

            target_coord = data.station_positions[target_station].astype(np.float32)
            feat = nearest_features(
                known_coords,
                known_values,
                target_coord,
                data.coord_img[tr, tc],
                cov_t[:, tr, tc],
                k,
                data.config.IMG_SHAPE,
            )
            features.append(feat)
            targets.append(target_value)
            records.append(
                SpatialRecord(
                    array_idx=int(array_idx),
                    original_time_idx=int(data.original_time_indices[int(array_idx)]),
                    target_station=int(target_station),
                    target_coord=target_coord,
                    known_coords=known_coords,
                    known_values=known_values,
                    target_value=target_value,
                )
            )

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        records,
    )


def distance_vector(record: SpatialRecord) -> np.ndarray:
    return np.linalg.norm(
        record.known_coords - record.target_coord[None, :], axis=1
    ).astype(np.float64)


def predict_spatial_mean(record: SpatialRecord) -> float:
    return float(np.mean(record.known_values))


def predict_nearest(record: SpatialRecord) -> float:
    distances = distance_vector(record)
    return float(record.known_values[int(np.argmin(distances))])


def predict_idw(record: SpatialRecord, power: float) -> float:
    distances = distance_vector(record)
    zero = np.flatnonzero(distances < 1e-8)
    if zero.size:
        return float(record.known_values[int(zero[0])])
    weights = 1.0 / np.power(np.maximum(distances, 1e-8), power)
    return float(np.sum(weights * record.known_values) / np.sum(weights))


def predict_knn(record: SpatialRecord, k: int) -> float:
    distances = distance_vector(record)
    order = np.argsort(distances)[: min(k, len(distances))]
    selected_distances = distances[order]
    selected_values = record.known_values[order]
    zero = np.flatnonzero(selected_distances < 1e-8)
    if zero.size:
        return float(selected_values[int(zero[0])])
    weights = 1.0 / np.maximum(selected_distances, 1e-8)
    return float(np.sum(weights * selected_values) / np.sum(weights))


def predict_rbf(record: SpatialRecord, function: str, smooth: float) -> float:
    try:
        interpolator = Rbf(
            record.known_coords[:, 0],
            record.known_coords[:, 1],
            record.known_values,
            function=function,
            smooth=smooth,
        )
        prediction = float(
            interpolator(record.target_coord[0], record.target_coord[1])
        )
        return prediction if np.isfinite(prediction) else predict_spatial_mean(record)
    except Exception:
        return predict_spatial_mean(record)


def predict_ordinary_kriging(record: SpatialRecord) -> float:
    if OrdinaryKriging is None:
        return np.nan
    try:
        model = OrdinaryKriging(
            record.known_coords[:, 1],
            record.known_coords[:, 0],
            record.known_values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
        prediction, _ = model.execute(
            "points",
            np.asarray([record.target_coord[1]]),
            np.asarray([record.target_coord[0]]),
        )
        value = float(np.asarray(prediction).reshape(-1)[0])
        return value if np.isfinite(value) else predict_spatial_mean(record)
    except Exception:
        return predict_spatial_mean(record)


def select_interpolation_hyperparameters(
    validation_records: Sequence[SpatialRecord], config: Config
) -> tuple[float, int]:
    if not validation_records:
        return 1.5, 4
    y_true = np.asarray([record.target_value for record in validation_records])

    idw_scores = {}
    for power in config.IDW_POWER_CANDIDATES:
        prediction = np.asarray(
            [predict_idw(record, power) for record in validation_records]
        )
        idw_scores[power] = mean_absolute_error(y_true, prediction)

    knn_scores = {}
    for k in config.KNN_CANDIDATES:
        prediction = np.asarray(
            [predict_knn(record, k) for record in validation_records]
        )
        knn_scores[k] = mean_absolute_error(y_true, prediction)

    best_power = min(idw_scores, key=idw_scores.get)
    best_k = min(knn_scores, key=knn_scores.get)
    return float(best_power), int(best_k)


def build_ml_models(config: Config, fold_seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    if "RandomForest" in config.MODELS:
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=config.RF_ESTIMATORS,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=fold_seed,
            n_jobs=config.ML_N_JOBS,
        )
    if "ExtraTrees" in config.MODELS:
        models["ExtraTrees"] = ExtraTreesRegressor(
            n_estimators=config.ET_ESTIMATORS,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=fold_seed,
            n_jobs=config.ML_N_JOBS,
        )
    if "XGBoost" in config.MODELS and xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=fold_seed,
            n_jobs=config.ML_N_JOBS,
        )
    return models


class NeighborTransformer(nn.Module):
    """Attention over same-time nearest-station value-distance tokens."""

    def __init__(self, k_neighbors: int, auxiliary_dim: int):
        super().__init__()
        embed_dim = 32
        self.k_neighbors = k_neighbors
        self.token_embedding = nn.Linear(2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.randn(1, k_neighbors + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.auxiliary = nn.Sequential(
            nn.Linear(auxiliary_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        k = self.k_neighbors
        values = features[:, :k].unsqueeze(-1)
        distances = features[:, k : 2 * k].unsqueeze(-1)
        tokens = self.token_embedding(torch.cat([values, distances], dim=-1))
        cls = self.cls_token.expand(features.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.position)
        auxiliary = self.auxiliary(features[:, 2 * k :])
        return self.head(torch.cat([encoded[:, 0], auxiliary], dim=1)).squeeze(-1)


def make_tensor_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    config: Config,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": DEVICE.type == "cuda",
    }
    if config.NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(**kwargs)


def train_neighbor_transformer(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: Config,
    fold_seed: int,
) -> tuple[NeighborTransformer, float, int]:
    seed_everything(fold_seed)
    auxiliary_dim = train_x.shape[1] - 2 * config.K_NEIGHBORS
    model = NeighborTransformer(config.K_NEIGHBORS, auxiliary_dim).to(DEVICE)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = make_tensor_loader(
        train_ds, config.BATCH_SIZE, True, fold_seed, config
    )
    val_loader = make_tensor_loader(
        val_ds, config.BATCH_SIZE, False, fold_seed, config
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.SmoothL1Loss()
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    started = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bx)
            loss = criterion(prediction, by)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                loss = criterion(
                    model(bx.to(DEVICE, non_blocking=True)),
                    by.to(DEVICE, non_blocking=True),
                )
                if torch.isfinite(loss):
                    losses.append(float(loss.item()))
        mean_val = float(np.mean(losses)) if losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("NeighborTransformer did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), time.time() - started, best_epoch


def observed_neighbor_features(
    sparse: np.ndarray,
    observed_mask: np.ndarray,
    radius: int,
    fallback_mean: float,
    fallback_std: float,
) -> np.ndarray:
    """Observed-only local mean/std; missing-pixel fill never enters statistics."""

    kernel_size = 2 * radius + 1
    values = np.where(observed_mask, sparse, 0.0).astype(np.float32)
    mask_float = observed_mask.astype(np.float32)
    local_sum = uniform_filter(
        values, size=kernel_size, mode="constant", cval=0.0
    ) * (kernel_size**2)
    local_sq_sum = uniform_filter(
        values * values, size=kernel_size, mode="constant", cval=0.0
    ) * (kernel_size**2)
    local_count = uniform_filter(
        mask_float, size=kernel_size, mode="constant", cval=0.0
    ) * (kernel_size**2)
    safe_count = np.maximum(local_count, 1.0)
    local_mean = local_sum / safe_count
    local_var = np.maximum(local_sq_sum / safe_count - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var)
    no_neighbor = local_count < 0.5
    local_mean[no_neighbor] = fallback_mean
    local_std[no_neighbor] = fallback_std
    return np.stack(
        [
            local_mean,
            local_std,
            np.full_like(local_mean, fallback_mean),
        ],
        axis=0,
    ).astype(np.float32)


class GridMaskDataset(Dataset):
    """Same strict gridded input/masking protocol as the final SC-UNet."""

    def __init__(
        self,
        data: PollutantData,
        scaler: FoldScaler,
        heldout_station: int,
        mode: str,
        config: Config,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.data = data
        self.scaler = scaler
        self.heldout_station = heldout_station
        self.mode = mode
        self.config = config
        self.mask_ratio = config.START_MASK_RATIO
        self.time_indices = np.asarray(data.time_splits[mode], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.time_indices)

    def set_mask_ratio(self, ratio: float) -> None:
        self.mask_ratio = float(np.clip(ratio, 0.0, 1.0))

    def _mask_inner_targets(
        self, array_idx: int, sparse: np.ndarray, target: np.ndarray
    ) -> None:
        candidates = [
            station
            for station in range(self.data.n_stations)
            if station != self.heldout_station
        ]
        count = max(1, int(round(len(candidates) * self.mask_ratio)))
        count = min(count, len(candidates))
        if self.mode == "train":
            selected = random.sample(candidates, count)
        else:
            rng = np.random.default_rng(GLOBAL_SEED + int(array_idx))
            selected = rng.choice(candidates, size=count, replace=False).tolist()
        for station in selected:
            row, col = self.data.station_positions[station]
            target[row, col] = sparse[row, col]
            sparse[row, col] = np.nan

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        array_idx = int(self.time_indices[index])
        sparse = normalized_concentration(self.data, array_idx, self.scaler).copy()
        target = np.full(self.config.IMG_SHAPE, np.nan, dtype=np.float32)
        row, col = self.data.station_positions[self.heldout_station]
        heldout_value = sparse[row, col]
        sparse[row, col] = np.nan

        if self.mode == "test":
            target[row, col] = heldout_value
        else:
            self._mask_inner_targets(array_idx, sparse, target)

        observed = np.isfinite(sparse) & (self.data.valid_grid_mask > 0)
        observed_values = sparse[observed]
        sample_mean = float(observed_values.mean()) if observed_values.size else 0.0
        sample_std = float(observed_values.std()) if observed_values.size > 1 else 1.0
        if sample_std < 1e-6:
            sample_std = 1.0

        filled = np.where(observed, sparse, sample_mean).astype(np.float32)
        neighbor = observed_neighbor_features(
            sparse,
            observed,
            self.config.FIXED_NEIGHBOR_RADIUS,
            sample_mean,
            sample_std,
        )
        distance = distance_transform_edt(~observed).astype(np.float32)
        valid_distances = distance[self.data.valid_grid_mask > 0]
        max_distance = (
            float(valid_distances.max())
            if valid_distances.size
            else float(distance.max())
        )
        distance /= max(max_distance, 1.0)
        distance *= self.data.valid_grid_mask

        base_features = np.concatenate(
            [
                filled[None],
                neighbor,
                self.data.coord_feat,
                distance[None],
            ],
            axis=0,
        ).astype(np.float32)
        covariates = np.asarray(
            self.data.covariates[array_idx], dtype=np.float32
        ).copy()
        model_input = np.concatenate([base_features, covariates], axis=0)
        known = np.nan_to_num(
            sparse, nan=sample_mean, posinf=sample_mean, neginf=sample_mean
        ).astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "observed": torch.from_numpy(observed[None].astype(np.float32)),
            "known": torch.from_numpy(known[None]),
            "target": torch.from_numpy(target[None]),
            "valid": torch.from_numpy(self.data.valid_grid_mask[None]),
            "array_idx": array_idx,
            "original_time_idx": int(self.data.original_time_indices[array_idx]),
        }


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def enforce_known_pixels(
    raw_prediction: torch.Tensor,
    observed: torch.Tensor,
    known: torch.Tensor,
) -> torch.Tensor:
    return observed * known + (1.0 - observed) * raw_prediction


class SparseGridCNN(nn.Module):
    """Adapted CNN baseline using all gridded input channels."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor, known: torch.Tensor
    ) -> torch.Tensor:
        return enforce_known_pixels(self.net(x), observed, known)


class CovariateUNet(nn.Module):
    """Plain U-Net with identical gridded inputs but no ASF/SCB/attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.enc1 = ConvNormAct(in_channels, 32)
        self.enc2 = ConvNormAct(32, 64)
        self.enc3 = ConvNormAct(64, 128)
        self.mid = ConvNormAct(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvNormAct(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvNormAct(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvNormAct(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor, known: torch.Tensor
    ) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        middle = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(middle), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return enforce_known_pixels(self.output(d1), observed, known)


class PartialConv2d(nn.Module):
    """Partial convolution conditioned only on valid pixels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        updater = torch.ones(1, in_channels, kernel_size, kernel_size)
        self.register_buffer("mask_updater", updater)
        self.window_size = float(in_channels * kernel_size * kernel_size)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.shape[1] == 1:
            expanded_mask = mask.expand(-1, x.shape[1], -1, -1)
        elif mask.shape[1] == x.shape[1]:
            expanded_mask = mask
        else:
            raise ValueError(
                f"PartialConv mask has {mask.shape[1]} channels for "
                f"an input with {x.shape[1]} channels"
            )
        masked_x = x * expanded_mask
        raw = self.input_conv(masked_x)
        with torch.no_grad():
            mask_sum = F.conv2d(
                expanded_mask,
                self.mask_updater,
                stride=self.stride,
                padding=self.padding,
            )
            new_mask = (mask_sum > 0).to(x.dtype)
            ratio = self.window_size / mask_sum.clamp_min(1.0)
            ratio = ratio * new_mask
        if self.input_conv.bias is not None:
            bias = self.input_conv.bias.view(1, -1, 1, 1)
            output = (raw - bias) * ratio + bias
        else:
            output = raw * ratio
        return output * new_mask, new_mask


class PartialConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pconv1 = PartialConv2d(in_channels, out_channels)
        self.pconv2 = PartialConv2d(out_channels, out_channels)
        self.norm1 = nn.GroupNorm(max(1, min(8, out_channels)), out_channels)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_channels)), out_channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv1(x, mask)
        x = F.leaky_relu(self.norm1(x), 0.2, inplace=True)
        x, mask = self.pconv2(x, mask)
        x = F.leaky_relu(self.norm2(x), 0.2, inplace=True)
        return x, mask


class PartialConvUNet(nn.Module):
    """U-Net whose convolutions are renormalized by propagated masks."""

    def __init__(self, _in_channels: int):
        super().__init__()
        self.enc1 = PartialConvBlock(1, 32)
        self.enc2 = PartialConvBlock(32, 64)
        self.enc3 = PartialConvBlock(64, 128)
        self.mid = PartialConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = PartialConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = PartialConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = PartialConvBlock(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor, known: torch.Tensor
    ) -> torch.Tensor:
        concentration = x[:, :1]
        e1, m1 = self.enc1(concentration, observed)
        e2, m2 = self.enc2(self.pool(e1), self.pool(m1))
        e3, m3 = self.enc3(self.pool(e2), self.pool(m2))
        middle, mm = self.mid(self.pool(e3), self.pool(m3))

        u3 = self.up3(middle)
        um3 = F.interpolate(mm, size=e3.shape[-2:], mode="nearest")
        input3 = torch.cat([u3, e3], dim=1)
        mask3 = torch.cat(
            [um3.expand_as(u3), m3.expand_as(e3)], dim=1
        )
        d3, dm3 = self.dec3(input3, mask3)
        u2 = self.up2(d3)
        um2 = F.interpolate(dm3, size=e2.shape[-2:], mode="nearest")
        input2 = torch.cat([u2, e2], dim=1)
        mask2 = torch.cat(
            [um2.expand_as(u2), m2.expand_as(e2)], dim=1
        )
        d2, dm2 = self.dec2(input2, mask2)
        u1 = self.up1(d2)
        um1 = F.interpolate(dm2, size=e1.shape[-2:], mode="nearest")
        input1 = torch.cat([u1, e1], dim=1)
        mask1 = torch.cat(
            [um1.expand_as(u1), m1.expand_as(e1)], dim=1
        )
        d1, _ = self.dec1(input1, mask1)
        return enforce_known_pixels(self.output(d1), observed, known)


def make_grid_loader(
    dataset: GridMaskDataset,
    config: Config,
    shuffle: bool,
    seed: int,
    batch_size: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size or config.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": DEVICE.type == "cuda",
    }
    if config.NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def move_grid_batch(batch: dict) -> tuple[torch.Tensor, ...]:
    return (
        batch["input"].to(DEVICE, non_blocking=True),
        batch["observed"].to(DEVICE, non_blocking=True),
        batch["known"].to(DEVICE, non_blocking=True),
        batch["target"].to(DEVICE, non_blocking=True),
        batch["valid"].to(DEVICE, non_blocking=True),
    )


def supervised_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model_input: torch.Tensor,
    valid_grid_mask: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    mask = torch.isfinite(target) & (valid_grid_mask > 0)
    if not mask.any():
        return prediction.sum() * 0.0

    covariates = model_input[:, 7:]
    importance = covariates.abs().mean(dim=1, keepdim=True)
    valid_importance = importance * valid_grid_mask
    per_sample_max = valid_importance.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    weights = 1.0 + config.IMPORTANCE_BETA * (
        valid_importance / per_sample_max.clamp_min(1e-8)
    )
    huber = F.smooth_l1_loss(
        prediction[mask], target[mask], beta=1.0, reduction="none"
    )
    mse = F.mse_loss(prediction[mask], target[mask], reduction="none")
    return (
        (config.HUBER_WEIGHT * huber + config.MSE_WEIGHT * mse) * weights[mask]
    ).mean()


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def checkpoint_paths(
    config: Config, pollutant: str, heldout_station: int, model_name: str
) -> tuple[Path, Path]:
    safe_name = model_name.lower().replace(" ", "_")
    checkpoint = Path(
        config.RESULT_DIR,
        "models",
        f"{pollutant}_station_{heldout_station:02d}_{safe_name}.pth",
    )
    marker = Path(str(checkpoint) + ".complete.json")
    return checkpoint, marker


def train_image_model(
    model_name: str,
    data: PollutantData,
    scaler: FoldScaler,
    heldout_station: int,
    config: Config,
) -> tuple[nn.Module, float, int, int]:
    fold_seed = stable_seed(data.pollutant, heldout_station, model_name)
    seed_everything(fold_seed)
    in_channels = 7 + data.feature_dim
    constructors = {
        "SparseGridCNN": SparseGridCNN,
        "CovariateUNet": CovariateUNet,
        "PartialConvUNet": PartialConvUNet,
    }
    model = constructors[model_name](in_channels).to(DEVICE)
    parameter_count = model_parameter_count(model)
    checkpoint, marker_path = checkpoint_paths(
        config, data.pollutant, heldout_station, model_name
    )

    if config.REUSE_COMPLETED_FOLDS and checkpoint.exists() and marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            compatible = (
                marker.get("version") == config.MODEL_VERSION
                and marker.get("seed") == GLOBAL_SEED
                and marker.get("model") == model_name
                and marker.get("pollutant") == data.pollutant
                and marker.get("heldout_station") == heldout_station
                and marker.get("feature_dim") == data.feature_dim
                and marker.get("status") == "completed"
            )
        except (OSError, json.JSONDecodeError):
            compatible = False
        if compatible:
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            return (
                model.eval(),
                float(marker.get("train_seconds", 0.0)),
                int(marker.get("best_epoch", -1)),
                parameter_count,
            )

    checkpoint.unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)
    train_dataset = GridMaskDataset(data, scaler, heldout_station, "train", config)
    val_dataset = GridMaskDataset(data, scaler, heldout_station, "val", config)
    train_loader = make_grid_loader(train_dataset, config, True, fold_seed)
    val_loader = make_grid_loader(val_dataset, config, False, fold_seed)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=config.EPOCHS,
    )
    amp_scaler = GradScaler(enabled=USE_AMP)
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    started = time.time()

    for epoch in range(config.EPOCHS):
        mask_ratio = (
            config.START_MASK_RATIO
            if epoch < config.CURRICULUM_EPOCH
            else config.FINAL_MASK_RATIO
        )
        train_dataset.set_mask_ratio(mask_ratio)
        val_dataset.set_mask_ratio(config.FINAL_MASK_RATIO)
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs, observed, known, target, valid = move_grid_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                prediction = model(inputs, observed, known)
                loss = supervised_reconstruction_loss(
                    prediction, target, inputs, valid, config
                )
            if torch.isfinite(loss):
                train_losses.append(float(loss.item()))
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, observed, known, target, valid = move_grid_batch(batch)
                prediction = model(inputs, observed, known)
                loss = supervised_reconstruction_loss(
                    prediction, target, inputs, valid, config
                )
                if torch.isfinite(loss):
                    val_losses.append(float(loss.item()))
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint)
            patience = 0
        else:
            patience += 1

        if epoch == 0 or (epoch + 1) % config.LOG_INTERVAL == 0:
            train_mean = float(np.mean(train_losses)) if train_losses else np.nan
            print(
                f"    {model_name} epoch {epoch + 1:03d} | "
                f"train={train_mean:.6f} val={mean_val:.6f} best={best_val:.6f}"
            )
        if patience >= config.PATIENCE:
            break

    if not checkpoint.exists():
        raise RuntimeError(f"{model_name} did not produce a finite checkpoint")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    elapsed = time.time() - started
    marker = {
        "status": "completed",
        "version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "model": model_name,
        "pollutant": data.pollutant,
        "heldout_station": heldout_station,
        "feature_dim": data.feature_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "train_seconds": elapsed,
        "parameter_count": parameter_count,
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return model.eval(), elapsed, best_epoch, parameter_count


def predict_image_model(
    model: nn.Module,
    data: PollutantData,
    scaler: FoldScaler,
    heldout_station: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = GridMaskDataset(data, scaler, heldout_station, "test", config)
    loader = make_grid_loader(dataset, config, False, GLOBAL_SEED, batch_size=1)
    row, col = data.station_positions[heldout_station]
    predictions = []
    targets = []
    original_times = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, target, _ = move_grid_batch(batch)
            prediction = model(inputs, observed, known)
            pred_value = float(prediction[0, 0, row, col].cpu())
            true_value = float(target[0, 0, row, col].cpu())
            if np.isfinite(pred_value) and np.isfinite(true_value):
                predictions.append(pred_value)
                targets.append(true_value)
                original_times.append(int(batch["original_time_idx"].item()))
    return (
        np.asarray(predictions, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(original_times, dtype=np.int64),
    )


def predict_neighbor_transformer(
    model: NeighborTransformer, features: np.ndarray, config: Config
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features))
    loader = make_tensor_loader(dataset, 256, False, GLOBAL_SEED, config)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch.to(DEVICE, non_blocking=True)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def prediction_rows(
    pollutant: str,
    heldout_station: int,
    model_name: str,
    normalized_prediction: np.ndarray,
    normalized_target: np.ndarray,
    original_times: np.ndarray,
    scaler: FoldScaler,
) -> list[dict[str, object]]:
    raw_prediction = scaler.inverse(normalized_prediction.astype(np.float64))
    raw_target = scaler.inverse(normalized_target.astype(np.float64))
    rows = []
    for time_idx, true_value, predicted_value in zip(
        original_times, raw_target, raw_prediction
    ):
        if np.isfinite(true_value) and np.isfinite(predicted_value):
            rows.append(
                {
                    "pollutant": pollutant,
                    "station": heldout_station,
                    "time_idx": int(time_idx),
                    "model": model_name,
                    "true": float(true_value),
                    "prediction": float(predicted_value),
                }
            )
    return rows


def fold_file_paths(
    config: Config, pollutant: str, heldout_station: int
) -> tuple[Path, Path, Path]:
    stem = f"{pollutant}_fold_{heldout_station:02d}"
    return (
        Path(config.RESULT_DIR, "folds", f"{stem}_predictions.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}_efficiency.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}.complete.json"),
    )


def reusable_fold(
    config: Config, pollutant: str, heldout_station: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    prediction_path, efficiency_path, marker_path = fold_file_paths(
        config, pollutant, heldout_station
    )
    if not config.REUSE_COMPLETED_FOLDS:
        return None
    if not prediction_path.exists() or not efficiency_path.exists() or not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_models = sorted(config.MODELS)
        if not (
            marker.get("status") == "completed"
            and marker.get("version") == config.MODEL_VERSION
            and marker.get("seed") == GLOBAL_SEED
            and marker.get("pollutant") == pollutant
            and marker.get("heldout_station") == heldout_station
            and marker.get("models") == expected_models
        ):
            return None
        predictions = pd.read_csv(prediction_path)
        efficiency = pd.read_csv(efficiency_path)
        if sorted(predictions["model"].unique().tolist()) != expected_models:
            return None
        return predictions, efficiency
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None


def save_completed_fold(
    config: Config,
    pollutant: str,
    heldout_station: int,
    predictions: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> None:
    prediction_path, efficiency_path, marker_path = fold_file_paths(
        config, pollutant, heldout_station
    )
    predictions.to_csv(prediction_path, index=False)
    efficiency.to_csv(efficiency_path, index=False)
    marker = {
        "status": "completed",
        "version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "pollutant": pollutant,
        "heldout_station": heldout_station,
        "models": sorted(config.MODELS),
        "n_rows": len(predictions),
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")


def run_fold(
    config: Config, data: PollutantData, heldout_station: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reused = reusable_fold(config, data.pollutant, heldout_station)
    if reused is not None:
        print(f"  station {heldout_station:02d}: reuse completed fold")
        return reused

    print(f"  station {heldout_station:02d}/{data.n_stations - 1:02d}")
    scaler = data.fold_scaler(heldout_station)
    need_tabular = bool(set(config.MODELS) & (INTERPOLATION_MODELS | TABULAR_MODELS))
    prediction_output: list[dict[str, object]] = []
    efficiency_output: list[dict[str, object]] = []

    if need_tabular:
        train_x, train_y, _ = build_tabular_split(
            data, scaler, heldout_station, "train", config.K_NEIGHBORS
        )
        val_x, val_y, val_records = build_tabular_split(
            data, scaler, heldout_station, "val", config.K_NEIGHBORS
        )
        test_x, test_y, test_records = build_tabular_split(
            data, scaler, heldout_station, "test", config.K_NEIGHBORS
        )
        if len(test_records) == 0:
            raise RuntimeError(f"No test observations for station {heldout_station}")
        test_times = np.asarray(
            [record.original_time_idx for record in test_records], dtype=np.int64
        )
        best_idw_power, best_knn_k = select_interpolation_hyperparameters(
            val_records, config
        )

        interpolation_functions = {
            "SpatialMean": lambda record: predict_spatial_mean(record),
            "NearestNeighbor": lambda record: predict_nearest(record),
            "IDW": lambda record: predict_idw(record, best_idw_power),
            "KNN": lambda record: predict_knn(record, best_knn_k),
            "RBF": lambda record: predict_rbf(
                record, config.RBF_FUNCTION, config.RBF_SMOOTH
            ),
            "OrdinaryKriging": lambda record: predict_ordinary_kriging(record),
        }
        for model_name in config.MODELS:
            if model_name not in interpolation_functions:
                continue
            started = time.time()
            prediction = np.asarray(
                [interpolation_functions[model_name](record) for record in test_records],
                dtype=np.float32,
            )
            elapsed = time.time() - started
            prediction_output.extend(
                prediction_rows(
                    data.pollutant,
                    heldout_station,
                    model_name,
                    prediction,
                    test_y,
                    test_times,
                    scaler,
                )
            )
            efficiency_output.append(
                {
                    "pollutant": data.pollutant,
                    "station": heldout_station,
                    "model": model_name,
                    "train_seconds": 0.0,
                    "inference_seconds": elapsed,
                    "best_epoch": np.nan,
                    "parameter_count": 0,
                    "selected_idw_power": best_idw_power if model_name == "IDW" else np.nan,
                    "selected_knn_k": best_knn_k if model_name == "KNN" else np.nan,
                }
            )

        fold_seed = stable_seed(data.pollutant, heldout_station, "tabular_ml")
        for model_name, model in build_ml_models(config, fold_seed).items():
            started = time.time()
            model.fit(train_x, train_y)
            train_seconds = time.time() - started
            started = time.time()
            prediction = np.asarray(model.predict(test_x), dtype=np.float32)
            inference_seconds = time.time() - started
            prediction_output.extend(
                prediction_rows(
                    data.pollutant,
                    heldout_station,
                    model_name,
                    prediction,
                    test_y,
                    test_times,
                    scaler,
                )
            )
            efficiency_output.append(
                {
                    "pollutant": data.pollutant,
                    "station": heldout_station,
                    "model": model_name,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "best_epoch": np.nan,
                    "parameter_count": np.nan,
                    "selected_idw_power": np.nan,
                    "selected_knn_k": np.nan,
                }
            )

        if "NeighborTransformer" in config.MODELS:
            fold_seed = stable_seed(
                data.pollutant, heldout_station, "NeighborTransformer"
            )
            transformer, train_seconds, best_epoch = train_neighbor_transformer(
                train_x, train_y, val_x, val_y, config, fold_seed
            )
            started = time.time()
            prediction = predict_neighbor_transformer(transformer, test_x, config)
            inference_seconds = time.time() - started
            prediction_output.extend(
                prediction_rows(
                    data.pollutant,
                    heldout_station,
                    "NeighborTransformer",
                    prediction,
                    test_y,
                    test_times,
                    scaler,
                )
            )
            efficiency_output.append(
                {
                    "pollutant": data.pollutant,
                    "station": heldout_station,
                    "model": "NeighborTransformer",
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "best_epoch": best_epoch,
                    "parameter_count": model_parameter_count(transformer),
                    "selected_idw_power": np.nan,
                    "selected_knn_k": np.nan,
                }
            )
            del transformer

        del train_x, train_y, val_x, val_y, test_x, test_y
        del val_records, test_records

    for model_name in config.MODELS:
        if model_name not in IMAGE_MODELS:
            continue
        model, train_seconds, best_epoch, parameter_count = train_image_model(
            model_name, data, scaler, heldout_station, config
        )
        started = time.time()
        prediction, target, test_times = predict_image_model(
            model, data, scaler, heldout_station, config
        )
        inference_seconds = time.time() - started
        prediction_output.extend(
            prediction_rows(
                data.pollutant,
                heldout_station,
                model_name,
                prediction,
                target,
                test_times,
                scaler,
            )
        )
        efficiency_output.append(
            {
                "pollutant": data.pollutant,
                "station": heldout_station,
                "model": model_name,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "best_epoch": best_epoch,
                "parameter_count": parameter_count,
                "selected_idw_power": np.nan,
                "selected_knn_k": np.nan,
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_df = pd.DataFrame(prediction_output)
    efficiency_df = pd.DataFrame(efficiency_output)
    actual_models = sorted(prediction_df["model"].unique().tolist())
    expected_models = sorted(config.MODELS)
    if actual_models != expected_models:
        raise RuntimeError(
            f"Fold {heldout_station} model mismatch: expected {expected_models}, "
            f"got {actual_models}"
        )
    save_completed_fold(
        config, data.pollutant, heldout_station, prediction_df, efficiency_df
    )
    return prediction_df, efficiency_df


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) < 1e-12:
        return np.nan
    return float(r2_score(y_true, y_pred))


def station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pollutant, model, station), group in predictions.groupby(
        ["pollutant", "model", "station"], sort=True
    ):
        y_true = group["true"].to_numpy(dtype=np.float64)
        y_pred = group["prediction"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "station": int(station),
                "n_test": len(group),
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "R2": safe_r2(y_true, y_pred),
            }
        )
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


def aggregate_metrics(
    predictions: pd.DataFrame, station_df: pd.DataFrame, config: Config
) -> pd.DataFrame:
    rows = []
    for (pollutant, model), group in predictions.groupby(
        ["pollutant", "model"], sort=True
    ):
        y_true = group["true"].to_numpy(dtype=np.float64)
        y_pred = group["prediction"].to_numpy(dtype=np.float64)
        station_group = station_df[
            (station_df["pollutant"] == pollutant) & (station_df["model"] == model)
        ]
        mae_low, mae_high = bootstrap_mean_ci(
            station_group["MAE"].to_numpy(),
            config.BOOTSTRAP_REPEATS,
            stable_seed(pollutant, model, "MAE"),
        )
        rmse_low, rmse_high = bootstrap_mean_ci(
            station_group["RMSE"].to_numpy(),
            config.BOOTSTRAP_REPEATS,
            stable_seed(pollutant, model, "RMSE"),
        )
        r2_low, r2_high = bootstrap_mean_ci(
            station_group["R2"].to_numpy(),
            config.BOOTSTRAP_REPEATS,
            stable_seed(pollutant, model, "R2"),
        )
        rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "n_stations": station_group["station"].nunique(),
                "n_test_observations": len(group),
                "pooled_MAE": mean_absolute_error(y_true, y_pred),
                "pooled_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "pooled_R2": safe_r2(y_true, y_pred),
                "mean_station_MAE": station_group["MAE"].mean(),
                "mean_station_MAE_CI_low": mae_low,
                "mean_station_MAE_CI_high": mae_high,
                "mean_station_RMSE": station_group["RMSE"].mean(),
                "mean_station_RMSE_CI_low": rmse_low,
                "mean_station_RMSE_CI_high": rmse_high,
                "mean_station_R2": station_group["R2"].mean(),
                "mean_station_R2_CI_low": r2_low,
                "mean_station_R2_CI_high": r2_high,
                "median_station_R2": station_group["R2"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["pollutant", "pooled_RMSE"])


def run_pollutant(config: Config, pollutant: str) -> None:
    data = PollutantData(config, pollutant)
    stop = data.n_stations
    if config.MAX_STATIONS > 0:
        stop = min(stop, config.FOLD_START + config.MAX_STATIONS)
    folds = range(config.FOLD_START, stop)
    all_predictions = []
    all_efficiency = []
    print(
        f"\n{pollutant.upper()}: strict LOSO baselines over folds "
        f"{config.FOLD_START}..{stop - 1} | models={','.join(config.MODELS)}"
    )

    for heldout_station in folds:
        prediction_df, efficiency_df = run_fold(config, data, heldout_station)
        all_predictions.append(prediction_df)
        all_efficiency.append(efficiency_df)

    predictions = pd.concat(all_predictions, ignore_index=True)
    efficiency = pd.concat(all_efficiency, ignore_index=True)
    stations = station_metrics(predictions)
    aggregate = aggregate_metrics(predictions, stations, config)

    predictions.to_csv(
        Path(config.RESULT_DIR, "predictions", f"{pollutant}_test_predictions.csv"),
        index=False,
    )
    stations.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_station_metrics.csv"),
        index=False,
    )
    aggregate.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_aggregate_metrics.csv"),
        index=False,
    )
    efficiency.to_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_efficiency.csv"),
        index=False,
    )
    print(
        aggregate[
            ["model", "pooled_MAE", "pooled_RMSE", "pooled_R2", "mean_station_R2"]
        ].to_string(index=False)
    )


def main() -> None:
    config = Config()
    validate_config(config)
    prepare_output_dirs(config)
    print(f"device={DEVICE} | seed={GLOBAL_SEED}")
    for pollutant in config.POLLUTANTS:
        run_pollutant(config, pollutant)


if __name__ == "__main__":
    main()
