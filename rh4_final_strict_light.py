"""Lightweight SC-UNet for zero-history station-to-grid reconstruction.

This release combines the task-aligned lightweight architecture with the
strict data and evaluation protocol required by the paper:

* Dense covariates are compressed asymmetrically with inexpensive 1x1
  convolutions while the seven sparse-observation channels are preserved.
* Group-normalized residual blocks use projected shortcuts whenever channel
  dimensions change, including every decoder stage.
* The distance channel is the normalized Euclidean distance to the nearest
  currently observed station; 0 denotes an observed pixel and larger values
  denote increasingly remote pixels.

1. The outer held-out station is absent from training and validation inputs and
   targets.  Validation masks only stations that remain in the training fold.
2. Concentration normalization is fitted per LOSO fold using training times and
   non-held-out stations, then applied to validation/test data.
3. LOSO and full-region models use the same supervised and spatial losses.
4. Spatial penalties ignore the 32x32 padding pixels outside the 999-grid domain.
5. Known station values are hard-preserved in the reconstructed output.
6. Raw, pooled and station-level test metrics are all saved without clipping.

No concentration values from unmonitored grids are used as targets.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
    }
)

GLOBAL_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def seed_everything(seed: int = GLOBAL_SEED) -> None:
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


seed_everything()


class Config:
    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    RESULT_DIR = os.environ.get(
        "SCUNET_RESULT_ROOT", "/etc/cj/boundary/rh4_final_strict_light_v1"
    )
    IMG_SHAPE = (32, 32)
    NUM_GRIDS = 999
    EXPECTED_FEATURE_DIMS = {"no2": 39, "o3": 39, "pm25": 40}
    POLLUTANTS = [
        x.strip()
        for x in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if x.strip()
    ]

    FIXED_NEIGHBOR_RADIUS = 2
    BATCH_SIZE = 32
    EPOCHS = 150
    START_MASK_RATIO = 0.05
    FINAL_MASK_RATIO = 0.25
    CURRICULUM_EPOCH = 50

    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    VAL_INTERVAL = 1
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 4
    LOG_INTERVAL = 10

    HUBER_WEIGHT = 0.8
    MSE_WEIGHT = 0.2
    IMPORTANCE_BETA = 2.0
    DIFFUSION_COEFF = 0.1

    # TV values preserve the original pollutant-specific settings.  The
    # Laplacian coefficient is explicitly common because the old code used
    # PINN_WEIGHT=0.01 for every pollutant.  Tune only inside validation data if
    # the manuscript claims pollutant-specific Laplacian coefficients.
    TV_WEIGHTS = {"no2": 0.01, "o3": 0.10, "pm25": 0.05}
    LAPLACIAN_WEIGHTS = {"no2": 0.01, "o3": 0.01, "pm25": 0.01}

    REUSE_CHECKPOINTS = os.environ.get("SCUNET_REUSE_CHECKPOINTS", "0") == "1"
    MODEL_VERSION = "strict_light_architecture_v1"


def prepare_output_dirs(config: Config) -> None:
    for name in ("models", "figures", "metrics", "full_predict_results"):
        Path(config.RESULT_DIR, name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": GLOBAL_SEED,
        "device": str(DEVICE),
        "outer_protocol": "station-wise LOSO",
        "validation_protocol": "mask non-held-out stations only",
        "time_split": "70/15/15 chronological (from preprocessing)",
        "target_source": "monitoring-station label values only",
        "prediction_clipping": False,
        "model_version": config.MODEL_VERSION,
        "architecture": {
            "asf": "two 1x1 covariate convolutions + residual spatial weighting",
            "scb": "two 3x3 Conv-GroupNorm layers + channel attention + projected residual",
            "unet_channels": [32, 64, 128, 256],
            "hard_known_pixel_consistency": True,
        },
        "distance_encoding": "normalized Euclidean distance to nearest observed station",
        "tv_weights": config.TV_WEIGHTS,
        "laplacian_weights": config.LAPLACIAN_WEIGHTS,
        "diffusion_coefficient": config.DIFFUSION_COEFF,
    }
    with open(Path(config.RESULT_DIR, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)


def observed_neighbor_features(
    sparse_img: np.ndarray,
    observed_mask: np.ndarray,
    radius: int,
    fallback_mean: float,
    fallback_std: float,
) -> np.ndarray:
    """Observed-only local mean/std; missing pixels never enter the statistic."""

    size = 2 * radius + 1
    area = float(size * size)
    values = np.where(observed_mask, sparse_img, 0.0).astype(np.float32)
    weights = observed_mask.astype(np.float32)

    local_count = uniform_filter(weights, size=size, mode="constant") * area
    local_sum = uniform_filter(values, size=size, mode="constant") * area
    local_sq_sum = uniform_filter(values * values, size=size, mode="constant") * area

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


class SpatialInpaintDatasetLOOCV(Dataset):
    def __init__(
        self,
        config: Config,
        pollutant: str,
        mask_station_idx: int = -1,
        mode: str = "train",
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.config = config
        self.pollutant = pollutant
        self.mask_station_idx = mask_station_idx
        self.mode = mode
        self.radius = config.FIXED_NEIGHBOR_RADIUS
        self.mask_ratio = config.START_MASK_RATIO
        self.data_dir = Path(config.DATA_ROOT_DIR, pollutant)

        raw_conc_path = self.data_dir / "conc_img_raw.npy"
        if raw_conc_path.exists():
            self.conc_img_raw = np.load(raw_conc_path, mmap_mode="r")
            self.conc_scaler = None
        else:
            # Backward-compatible fallback for outputs from older zh scripts.
            normalized = np.load(
                self.data_dir / "conc_img_full.npy", mmap_mode="r"
            )
            self.conc_scaler = joblib.load(self.data_dir / "conc_scaler.pkl")
            self.conc_img_raw = normalized
        self.cov_img_full = np.load(self.data_dir / "cov_img_full.npy", mmap_mode="r")
        self.coord_img = np.load(self.data_dir / "coord_img.npy")
        self.valid_time_indices_original = np.load(
            self.data_dir / "valid_time_indices.npy"
        )
        self.station_positions = np.load(
            self.data_dir / "station_positions.npy"
        ).astype(np.int64)
        time_splits = np.load(
            self.data_dir / "time_splits.npy", allow_pickle=True
        ).item()
        self.time_indices = np.asarray(time_splits[mode], dtype=np.int64)
        self.train_time_indices = np.asarray(time_splits["train"], dtype=np.int64)

        valid_mask_path = self.data_dir / "valid_grid_mask.npy"
        if valid_mask_path.exists():
            self.valid_grid_mask = np.load(valid_mask_path).astype(np.float32)
        else:
            flat = np.zeros(np.prod(config.IMG_SHAPE), dtype=np.float32)
            flat[: config.NUM_GRIDS] = 1.0
            self.valid_grid_mask = flat.reshape(config.IMG_SHAPE)

        self.coord_feat = self.coord_img.transpose(2, 0, 1).astype(np.float32)
        self.total_stations = len(self.station_positions)
        self.F_dim = int(self.cov_img_full.shape[1])
        expected_dim = config.EXPECTED_FEATURE_DIMS.get(pollutant)
        if expected_dim is not None and self.F_dim != expected_dim:
            raise ValueError(
                f"{pollutant}: expected {expected_dim} feature channels, "
                f"got {self.F_dim}"
            )
        self.base_channels = 7
        self.total_in_dim = self.base_channels + self.F_dim

        if not (-1 <= mask_station_idx < self.total_stations):
            raise IndexError("mask_station_idx is outside the station list")

        # Fit the fold scaler using training times and non-held-out stations
        # only.  New preprocessing stores raw sparse concentrations directly.
        train_raw = np.asarray(self.conc_img_raw[self.train_time_indices])
        if self.conc_scaler is not None:
            train_raw = (
                train_raw * float(self.conc_scaler.scale_[0])
                + float(self.conc_scaler.mean_[0])
            )

        fit_station_indices = [
            i for i in range(self.total_stations) if i != mask_station_idx
        ]
        rows = self.station_positions[fit_station_indices, 0]
        cols = self.station_positions[fit_station_indices, 1]
        fit_values = train_raw[:, rows, cols].reshape(-1)
        fit_values = fit_values[np.isfinite(fit_values)]
        if fit_values.size == 0:
            raise ValueError("No finite training observations for fold normalization")
        self.fold_mean = float(fit_values.mean())
        self.fold_std = float(fit_values.std())
        if self.fold_std < 1e-8:
            self.fold_std = 1.0

        self.fold_global_mean = 0.0
        self.fold_global_std = 1.0

    def __len__(self) -> int:
        return len(self.time_indices)

    def set_mask_ratio(self, ratio: float) -> None:
        self.mask_ratio = float(np.clip(ratio, 0.0, 1.0))

    def _to_fold_normalized(self, stored_img: np.ndarray) -> np.ndarray:
        raw = stored_img
        if self.conc_scaler is not None:
            raw = (
                stored_img * float(self.conc_scaler.scale_[0])
                + float(self.conc_scaler.mean_[0])
            )
        return ((raw - self.fold_mean) / self.fold_std).astype(np.float32)

    def _mask_training_targets(
        self,
        current_t: int,
        sparse: np.ndarray,
        target: np.ndarray,
    ) -> None:
        candidates = [
            i for i in range(self.total_stations) if i != self.mask_station_idx
        ]
        num_to_mask = max(1, int(round(len(candidates) * self.mask_ratio)))
        num_to_mask = min(num_to_mask, len(candidates))

        if self.mode == "train":
            selected = random.sample(candidates, num_to_mask)
        else:
            # Stable inner validation masks; the outer held-out station is never
            # selected and therefore never used for early stopping.
            rng = np.random.default_rng(GLOBAL_SEED + int(current_t))
            selected = rng.choice(candidates, size=num_to_mask, replace=False).tolist()

        for station_idx in selected:
            row, col = self.station_positions[station_idx]
            target[row, col] = sparse[row, col]
            sparse[row, col] = np.nan

    def __getitem__(self, idx: int):
        array_idx = int(self.time_indices[idx])
        original_time_idx = int(self.valid_time_indices_original[array_idx])
        sparse = self._to_fold_normalized(
            np.asarray(self.conc_img_raw[array_idx], dtype=np.float32)
        ).copy()
        covariates = np.asarray(self.cov_img_full[array_idx], dtype=np.float32).copy()
        target = np.full(self.config.IMG_SHAPE, np.nan, dtype=np.float32)

        # Outer LOSO station is removed in every mode.  It becomes a target only
        # in the test split, never during training or validation.
        if self.mask_station_idx >= 0:
            row, col = self.station_positions[self.mask_station_idx]
            heldout_value = sparse[row, col]
            sparse[row, col] = np.nan
            if self.mode == "test":
                target[row, col] = heldout_value

        if self.mode in {"train", "val"}:
            self._mask_training_targets(array_idx, sparse, target)

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
        neighbor = observed_neighbor_features(
            sparse,
            observed_mask,
            self.radius,
            sample_mean,
            sample_std,
        )
        # Preserve distance resolution across the full sparse-monitoring
        # domain.  Observed pixels are 0; more remote valid pixels approach 1.
        distance = distance_transform_edt(~observed_mask).astype(np.float32)
        valid_distance = distance[self.valid_grid_mask > 0]
        max_distance = (
            float(valid_distance.max()) if valid_distance.size else float(distance.max())
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
            known_values, nan=sample_mean, posinf=sample_mean, neginf=sample_mean
        ).astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "channel_mask": torch.from_numpy(observed_mask[None].astype(np.float32)),
            "known_values": torch.from_numpy(known_values[None]),
            "target": torch.from_numpy(target[None]),
            "infer_mask": torch.from_numpy(infer_mask[None]),
            "valid_grid_mask": torch.from_numpy(self.valid_grid_mask[None]),
            "time_idx": original_time_idx,
            "array_idx": array_idx,
        }


class ChannelAttention(nn.Module):
    """Light channel recalibration used inside each residual block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialConstraintBlock(nn.Module):
    """Group-normalized block with an always-valid residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels))
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.attention = ChannelAttention(out_channels, reduction=8)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.2, inplace=True)
        out = self.attention(self.norm2(self.conv2(out)))
        return F.leaky_relu(out + residual, 0.2, inplace=True)


class AsymmetricSqueezeFusion(nn.Module):
    """Preserve sparse channels while cheaply compressing dense covariates."""

    def __init__(
        self,
        base_dim: int,
        cov_dim: int,
        squeezed_dim: int = 8,
    ):
        super().__init__()
        hidden = max(16, squeezed_dim)
        self.cov_squeeze = nn.Sequential(
            nn.Conv2d(cov_dim, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, squeezed_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        merged_dim = base_dim + squeezed_dim
        attention_hidden = max(8, merged_dim // 2)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(merged_dim, attention_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(attention_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, base_features: torch.Tensor, covariates: torch.Tensor
    ) -> torch.Tensor:
        compressed_covariates = self.cov_squeeze(covariates)
        merged = torch.cat([base_features, compressed_covariates], dim=1)
        # Residual weighting preserves all original information and lets the
        # learned map amplify locations that need more capacity.
        return merged * (1.0 + self.spatial_attention(merged))


class SCUNet(nn.Module):
    """Light U-Net backbone with strict known-pixel consistency."""

    def __init__(self, base_channels: int, cov_channels: int, out_channels: int = 1):
        super().__init__()
        squeezed_cov_dim = 8
        self.base_channels = base_channels
        self.asym_fusion = AsymmetricSqueezeFusion(
            base_channels, cov_channels, squeezed_cov_dim
        )
        merged_dim = base_channels + squeezed_cov_dim

        self.enc1 = SpatialConstraintBlock(merged_dim, 32)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(32, 64)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(64, 128)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(128, 256)
        )

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = SpatialConstraintBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = SpatialConstraintBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = SpatialConstraintBlock(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        observed_mask: torch.Tensor,
        known_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_features = x[:, : self.base_channels]
        covariates = x[:, self.base_channels :]
        fused = self.asym_fusion(base_features, covariates)

        enc1 = self.enc1(fused)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec3 = self.dec3(torch.cat([self.up4(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up3(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up2(dec2), enc1], dim=1))
        raw_prediction = self.out_conv(dec1)

        # Known station pixels remain exact; no clipping is applied to unknowns.
        if known_values is None:
            known_values = base_features[:, 0:1]
        return observed_mask * known_values + (1.0 - observed_mask) * raw_prediction


def masked_total_variation(
    pred: torch.Tensor, valid_grid_mask: torch.Tensor
) -> torch.Tensor:
    vertical_mask = valid_grid_mask[:, :, 1:, :] * valid_grid_mask[:, :, :-1, :]
    horizontal_mask = valid_grid_mask[:, :, :, 1:] * valid_grid_mask[:, :, :, :-1]
    vertical = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    horizontal = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    return (vertical * vertical_mask).sum() / vertical_mask.sum().clamp_min(1.0) + (
        horizontal * horizontal_mask
    ).sum() / horizontal_mask.sum().clamp_min(1.0)


def masked_laplacian_penalty(
    pred: torch.Tensor,
    valid_grid_mask: torch.Tensor,
    diffusion_coeff: float,
) -> torch.Tensor:
    if pred.shape[-2] < 3 or pred.shape[-1] < 3:
        return pred.new_tensor(0.0)
    center = pred[:, :, 1:-1, 1:-1]
    laplacian = (
        pred[:, :, 2:, 1:-1]
        + pred[:, :, :-2, 1:-1]
        + pred[:, :, 1:-1, 2:]
        + pred[:, :, 1:-1, :-2]
        - 4.0 * center
    )
    stencil_mask = (
        valid_grid_mask[:, :, 1:-1, 1:-1]
        * valid_grid_mask[:, :, 2:, 1:-1]
        * valid_grid_mask[:, :, :-2, 1:-1]
        * valid_grid_mask[:, :, 1:-1, 2:]
        * valid_grid_mask[:, :, 1:-1, :-2]
    )
    residual_sq = (diffusion_coeff * laplacian).pow(2)
    return (residual_sq * stencil_mask).sum() / stencil_mask.sum().clamp_min(1.0)


def reconstruction_loss(
    config: Config,
    pollutant: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    model_input: torch.Tensor,
    valid_grid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supervised_mask = torch.isfinite(target) & (valid_grid_mask > 0)
    if supervised_mask.any():
        covariates = model_input[:, 7:]
        importance = covariates.abs().mean(dim=1, keepdim=True)
        valid_importance = importance * valid_grid_mask
        per_sample_max = valid_importance.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        normalized_importance = valid_importance / per_sample_max.clamp_min(1e-8)
        weights = 1.0 + config.IMPORTANCE_BETA * normalized_importance

        huber = F.smooth_l1_loss(
            pred[supervised_mask], target[supervised_mask], beta=1.0, reduction="none"
        )
        mse = F.mse_loss(
            pred[supervised_mask], target[supervised_mask], reduction="none"
        )
        supervised = (
            (config.HUBER_WEIGHT * huber + config.MSE_WEIGHT * mse)
            * weights[supervised_mask]
        ).mean()
    else:
        supervised = pred.sum() * 0.0

    tv = masked_total_variation(pred, valid_grid_mask)
    laplacian = masked_laplacian_penalty(
        pred, valid_grid_mask, config.DIFFUSION_COEFF
    )
    total = (
        supervised
        + config.TV_WEIGHTS[pollutant] * tv
        + config.LAPLACIAN_WEIGHTS[pollutant] * laplacian
    )
    return total, {"supervised": supervised, "tv": tv, "laplacian": laplacian}


def make_loader(
    dataset: Dataset,
    config: Config,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(GLOBAL_SEED)
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
        # Workers must be recreated each epoch so the updated curriculum
        # mask_ratio is visible inside Dataset.__getitem__.
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def move_batch(batch: dict) -> tuple[torch.Tensor, ...]:
    return (
        batch["input"].to(DEVICE, non_blocking=True),
        batch["channel_mask"].to(DEVICE, non_blocking=True),
        batch["known_values"].to(DEVICE, non_blocking=True),
        batch["target"].to(DEVICE, non_blocking=True),
        batch["valid_grid_mask"].to(DEVICE, non_blocking=True),
    )


def completion_marker_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".complete.json")


def completed_checkpoint_is_compatible(
    config: Config,
    pollutant: str,
    mask_station_idx: int,
    feature_dim: int,
    checkpoint_path: Path,
) -> bool:
    marker_path = completion_marker_path(checkpoint_path)
    if not checkpoint_path.exists() or not marker_path.exists():
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("model_version") == config.MODEL_VERSION
        and marker.get("pollutant") == pollutant
        and marker.get("mask_station_idx") == mask_station_idx
        and marker.get("feature_dim") == feature_dim
        and marker.get("status") == "completed"
    )


def train_one_model(
    config: Config,
    pollutant: str,
    mask_station_idx: int,
    checkpoint_path: Path,
) -> tuple[SCUNet, float]:
    train_dataset = SpatialInpaintDatasetLOOCV(
        config, pollutant, mask_station_idx, mode="train"
    )
    val_dataset = SpatialInpaintDatasetLOOCV(
        config, pollutant, mask_station_idx, mode="val"
    )
    model = SCUNet(train_dataset.base_channels, train_dataset.F_dim).to(DEVICE)

    if config.REUSE_CHECKPOINTS and completed_checkpoint_is_compatible(
        config,
        pollutant,
        mask_station_idx,
        train_dataset.F_dim,
        checkpoint_path,
    ):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        fold_name = "full" if mask_station_idx < 0 else f"station {mask_station_idx:02d}"
        print(f"  reuse completed checkpoint: {pollutant.upper()} {fold_name}")
        return model, 0.0

    # A weight file without a compatible completion marker may come from an
    # interrupted fold or another code version.  It is never silently reused.
    checkpoint_path.unlink(missing_ok=True)
    completion_marker_path(checkpoint_path).unlink(missing_ok=True)

    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=config.EPOCHS,
    )
    scaler = GradScaler(enabled=USE_AMP)
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
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
            inputs, observed, known, targets, valid_domain = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                predictions = model(inputs, observed, known)
                loss, _ = reconstruction_loss(
                    config,
                    pollutant,
                    predictions,
                    targets,
                    inputs,
                    valid_domain,
                )
            if torch.isfinite(loss):
                train_losses.append(float(loss.item()))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

        if (epoch + 1) % config.VAL_INTERVAL != 0:
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, observed, known, targets, valid_domain = move_batch(batch)
                predictions = model(inputs, observed, known)
                loss, _ = reconstruction_loss(
                    config,
                    pollutant,
                    predictions,
                    targets,
                    inputs,
                    valid_domain,
                )
                if torch.isfinite(loss):
                    val_losses.append(float(loss.item()))
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

        if (
            epoch == 0
            or (epoch + 1) % config.LOG_INTERVAL == 0
            or patience_counter >= config.PATIENCE
        ):
            fold_name = (
                "full" if mask_station_idx < 0 else f"station {mask_station_idx:02d}"
            )
            mean_train_loss = (
                float(np.mean(train_losses)) if train_losses else float("nan")
            )
            print(
                f"  {pollutant.upper()} {fold_name} | epoch {epoch + 1:03d} | "
                f"train={mean_train_loss:.6f} | val={mean_val_loss:.6f} | "
                f"best={best_val_loss:.6f}"
            )

    if not checkpoint_path.exists():
        raise RuntimeError("Training ended without a finite validation checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    elapsed = time.time() - started
    marker = {
        "status": "completed",
        "model_version": config.MODEL_VERSION,
        "pollutant": pollutant,
        "mask_station_idx": mask_station_idx,
        "feature_dim": train_dataset.F_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "train_seconds": elapsed,
    }
    with open(completion_marker_path(checkpoint_path), "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)
    return model, elapsed


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def evaluate_loso_fold(
    config: Config,
    pollutant: str,
    mask_station_idx: int,
    model: SCUNet,
) -> tuple[dict, pd.DataFrame]:
    dataset = SpatialInpaintDatasetLOOCV(
        config, pollutant, mask_station_idx, mode="test"
    )
    loader = make_loader(dataset, config, shuffle=False, batch_size=1)
    row, col = dataset.station_positions[mask_station_idx]
    records = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, targets, _ = move_batch(batch)
            predictions = model(inputs, observed, known)
            true_normalized = float(targets[0, 0, row, col].cpu())
            pred_normalized = float(predictions[0, 0, row, col].cpu())
            true_value = true_normalized * dataset.fold_std + dataset.fold_mean
            pred_value = pred_normalized * dataset.fold_std + dataset.fold_mean
            records.append(
                {
                    "pollutant": pollutant,
                    "station_index": mask_station_idx,
                    "row": int(row),
                    "col": int(col),
                    "original_time_idx": int(batch["time_idx"].item()),
                    "true": true_value,
                    "prediction": pred_value,
                    "error": pred_value - true_value,
                }
            )

    samples = pd.DataFrame(records)
    y_true = samples["true"].to_numpy()
    y_pred = samples["prediction"].to_numpy()
    summary = {
        "pollutant": pollutant,
        "station_index": mask_station_idx,
        "row": int(row),
        "col": int(col),
        "n": len(samples),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
        "fold_train_mean": dataset.fold_mean,
        "fold_train_std": dataset.fold_std,
    }
    return summary, samples


def predict_full_region(config: Config, pollutant: str) -> None:
    checkpoint = Path(
        config.RESULT_DIR, "models", f"{pollutant}_full_region_scunet.pth"
    )
    model, train_seconds = train_one_model(config, pollutant, -1, checkpoint)
    dataset = SpatialInpaintDatasetLOOCV(config, pollutant, -1, mode="test")
    loader = make_loader(dataset, config, shuffle=False, batch_size=1)

    prediction_maps = []
    known_maps = []
    original_times = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, _, valid_domain = move_batch(batch)
            prediction = model(inputs, observed, known)
            prediction = prediction[0, 0].cpu().numpy()
            prediction = prediction * dataset.fold_std + dataset.fold_mean
            prediction[dataset.valid_grid_mask == 0] = np.nan

            known_raw = known[0, 0].cpu().numpy() * dataset.fold_std + dataset.fold_mean
            observed_np = observed[0, 0].cpu().numpy().astype(bool)
            known_raw[~observed_np] = np.nan
            known_raw[dataset.valid_grid_mask == 0] = np.nan

            prediction_maps.append(prediction.astype(np.float32))
            known_maps.append(known_raw.astype(np.float32))
            original_times.append(int(batch["time_idx"].item()))

    predictions = np.stack(prediction_maps)
    known_observations = np.stack(known_maps)
    np.savez_compressed(
        Path(
            config.RESULT_DIR,
            "full_predict_results",
            f"{pollutant}_full_region_predict.npz",
        ),
        predictions=predictions,
        known_observations=known_observations,
        original_time_indices=np.asarray(original_times, dtype=np.int64),
        station_positions=dataset.station_positions,
        valid_grid_mask=dataset.valid_grid_mask,
        fold_train_mean=dataset.fold_mean,
        fold_train_std=dataset.fold_std,
        train_seconds=train_seconds,
    )

    first_prediction = predictions[0]
    first_known = known_observations[0]
    finite = np.concatenate(
        [first_prediction[np.isfinite(first_prediction)], first_known[np.isfinite(first_known)]]
    )
    vmin, vmax = float(finite.min()), float(finite.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axes[0].imshow(first_known, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].scatter(
        dataset.station_positions[:, 1],
        dataset.station_positions[:, 0],
        c="red",
        s=15,
        edgecolors="white",
        linewidths=0.3,
    )
    axes[0].set_title(f"{pollutant.upper()} observed stations")
    im1 = axes[1].imshow(first_prediction, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].scatter(
        dataset.station_positions[:, 1],
        dataset.station_positions[:, 0],
        c="red",
        s=15,
        edgecolors="white",
        linewidths=0.3,
    )
    axes[1].set_title(f"{pollutant.upper()} reconstructed grid")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    fig.savefig(
        Path(config.RESULT_DIR, "figures", f"{pollutant}_full_predict_vis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_paper_experiment(config: Config) -> None:
    prepare_output_dirs(config)
    print(f"Using device: {DEVICE}")
    for pollutant in config.POLLUTANTS:
        data_dir = Path(config.DATA_ROOT_DIR, pollutant)
        station_positions = np.load(data_dir / "station_positions.npy")
        total_stations = len(station_positions)
        station_metrics = []
        all_samples = []

        print(f"\n{pollutant.upper()}: strict LOSO over {total_stations} stations")
        for station_idx in range(total_stations):
            checkpoint = Path(
                config.RESULT_DIR,
                "models",
                f"{pollutant}_station_{station_idx}_scunet.pth",
            )
            model, train_seconds = train_one_model(
                config, pollutant, station_idx, checkpoint
            )
            summary, samples = evaluate_loso_fold(
                config, pollutant, station_idx, model
            )
            summary["train_seconds"] = train_seconds
            station_metrics.append(summary)
            all_samples.append(samples)
            print(
                f"  station {station_idx:02d}: "
                f"MAE={summary['MAE']:.4f}, RMSE={summary['RMSE']:.4f}, "
                f"R2={summary['R2']:.4f}"
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        station_df = pd.DataFrame(station_metrics)
        sample_df = pd.concat(all_samples, ignore_index=True)
        y_true = sample_df["true"].to_numpy()
        y_pred = sample_df["prediction"].to_numpy()
        aggregate_df = pd.DataFrame(
            [
                {
                    "pollutant": pollutant,
                    "n_stations": total_stations,
                    "n_test_observations": len(sample_df),
                    "pooled_MAE": mean_absolute_error(y_true, y_pred),
                    "pooled_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "pooled_R2": safe_r2(y_true, y_pred),
                    "mean_station_MAE": station_df["MAE"].mean(),
                    "mean_station_RMSE": station_df["RMSE"].mean(),
                    "mean_station_R2": station_df["R2"].mean(),
                    "median_station_R2": station_df["R2"].median(),
                }
            ]
        )

        metrics_dir = Path(config.RESULT_DIR, "metrics")
        station_df.to_csv(
            metrics_dir / f"{pollutant}_loocv_station_metrics.csv", index=False
        )
        sample_df.to_csv(
            metrics_dir / f"{pollutant}_loocv_test_predictions.csv", index=False
        )
        aggregate_df.to_csv(
            metrics_dir / f"{pollutant}_loocv_aggregate_metrics.csv", index=False
        )
        print(aggregate_df.to_string(index=False))

        predict_full_region(config, pollutant)


if __name__ == "__main__":
    run_paper_experiment(Config())
