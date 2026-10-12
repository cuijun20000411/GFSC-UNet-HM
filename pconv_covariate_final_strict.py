"""Standalone strict-LOSO runner for a covariate-conditioned PConv U-Net.

Why this file exists
--------------------
``baseline_final_strict.py`` intentionally implemented its original
``PartialConvUNet`` as a concentration-only image-inpainting baseline.  That
model is valid when it is labelled as concentration-only, but it must not be
described as receiving the same dense covariates as SC-UNet.

This runner adds one *adapted* comparison model, ``PConvUNetCov``.  It keeps
the exact data splits, fold-specific scaler, training-mask curriculum,
supervised loss, early stopping and strict outer LOSO protocol from
``baseline_final_strict.py`` while giving PConv the complete 7+F input.

The first partial-convolution mask is channel-aware:

* sparse concentration channel: valid only at currently observed stations;
* all auxiliary channels: valid over the study area's valid-grid mask.

This distinction is essential.  Expanding the sparse station mask across all
channels would erase dense covariates precisely at the unmonitored pixels to
be predicted.

Put this script beside ``baseline_final_strict.py`` and run it directly.  It
trains only ``PConvUNetCov`` and writes to a separate result directory, so an
already-running baseline experiment is neither modified nor restarted.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

try:
    import baseline_final_strict as base
except ImportError as exc:  # pragma: no cover - user-facing environment check
    raise ImportError(
        "Place pconv_covariate_final_strict.py in the same directory as "
        "baseline_final_strict.py, then run it from that directory."
    ) from exc


MODEL_NAME = "PConvUNetCov"
MODEL_VERSION = "strict_pconv_covariate_channel_mask_v1"
GLOBAL_SEED = int(
    os.environ.get(
        "PCONV_COV_SEED", os.environ.get("BASELINE_SEED", "42")
    )
)

# GridMaskDataset and stable_seed read these globals from the imported module.
# Setting and reseeding them makes this standalone runner deterministic while
# retaining exactly the same dataset implementation as the baseline script.
base.GLOBAL_SEED = GLOBAL_SEED
base.seed_everything(GLOBAL_SEED)


class Config(base.Config):
    """Configuration with a separate namespace and result directory."""

    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    RESULT_DIR = os.environ.get(
        "PCONV_COV_RESULT_ROOT",
        "/etc/cj/boundary/pconv_covariate_final_strict_v1",
    )
    POLLUTANTS = [
        item.strip()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]

    EPOCHS = int(os.environ.get("PCONV_COV_EPOCHS", str(base.Config.EPOCHS)))
    BATCH_SIZE = int(
        os.environ.get("PCONV_COV_BATCH_SIZE", str(base.Config.BATCH_SIZE))
    )
    NUM_WORKERS = int(
        os.environ.get(
            "PCONV_COV_NUM_WORKERS",
            os.environ.get("BASELINE_NUM_WORKERS", str(base.Config.NUM_WORKERS)),
        )
    )
    EVAL_BATCH_SIZE = int(os.environ.get("PCONV_COV_EVAL_BATCH_SIZE", "1"))
    FOLD_START = int(os.environ.get("PCONV_COV_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("PCONV_COV_MAX_STATIONS", "0"))
    REUSE_COMPLETED_FOLDS = (
        os.environ.get("PCONV_COV_REUSE_COMPLETED", "1") == "1"
    )
    BOOTSTRAP_REPEATS = int(
        os.environ.get(
            "PCONV_COV_BOOTSTRAP_REPEATS",
            str(base.Config.BOOTSTRAP_REPEATS),
        )
    )
    MODEL_NAME = MODEL_NAME
    MODEL_VERSION = MODEL_VERSION


def validate_config(config: Config) -> None:
    unknown_pollutants = sorted(
        set(config.POLLUTANTS) - set(config.EXPECTED_FEATURE_DIMS)
    )
    if unknown_pollutants:
        raise ValueError(f"Unsupported pollutants: {unknown_pollutants}")
    if not config.POLLUTANTS:
        raise ValueError("SCUNET_POLLUTANTS selected no pollutants")
    if config.EPOCHS < 1 or config.BATCH_SIZE < 1 or config.EVAL_BATCH_SIZE < 1:
        raise ValueError("Epoch and batch-size settings must be positive")
    if config.NUM_WORKERS < 0 or config.FOLD_START < 0 or config.MAX_STATIONS < 0:
        raise ValueError("Worker/fold settings must be non-negative")
    if not 0.0 <= config.START_MASK_RATIO <= 1.0:
        raise ValueError("START_MASK_RATIO must be in [0, 1]")
    if not 0.0 <= config.FINAL_MASK_RATIO <= 1.0:
        raise ValueError("FINAL_MASK_RATIO must be in [0, 1]")
    if config.CURRICULUM_EPOCH < 0:
        raise ValueError("CURRICULUM_EPOCH must be non-negative")
    if not Path(config.DATA_ROOT_DIR).exists():
        raise FileNotFoundError(f"Data root not found: {config.DATA_ROOT_DIR}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def prepare_output_dirs(config: Config) -> None:
    for name in ("metrics", "predictions", "models", "folds"):
        Path(config.RESULT_DIR, name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "device": str(base.DEVICE),
        "model": config.MODEL_NAME,
        "paper_label": "PConv-UNet+Cov (adapted)",
        "dependency": str(Path(base.__file__).resolve()),
        "outer_protocol": "strict station-wise LOSO",
        "outer_station_seen_during_training_or_selection": False,
        "temporal_history_at_outer_target": False,
        "time_split": "preprocessed chronological train/validation/test split",
        "fold_target_scaling": "training times and non-held-out stations only",
        "inner_validation": "deterministic masks on non-held-out stations only",
        "prediction_clipping": False,
        "input": "all 7 handcrafted/sparse channels plus all F covariates",
        "first_layer_mask": {
            "concentration_channel": "current observed-station mask",
            "auxiliary_channels": "valid study-grid mask",
        },
        "loss": "same supervised reconstruction loss as baseline_final_strict.py",
        "mask_curriculum": {
            "start_ratio": config.START_MASK_RATIO,
            "final_ratio": config.FINAL_MASK_RATIO,
            "switch_epoch": config.CURRICULUM_EPOCH,
        },
        "training": {
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "patience": config.PATIENCE,
        },
    }
    atomic_write_json(Path(config.RESULT_DIR, "protocol.json"), protocol)


class PConvUNetCov(nn.Module):
    """Partial-convolution U-Net with channel-specific input validity.

    Concentration is sparse, whereas auxiliary channels are known throughout
    the valid study grid.  After the first partial convolution, the propagated
    single-channel validity masks follow the original PConv implementation.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels < 2:
            raise ValueError("PConvUNetCov requires concentration and auxiliaries")
        self.in_channels = int(in_channels)
        self.enc1 = base.PartialConvBlock(self.in_channels, 32)
        self.enc2 = base.PartialConvBlock(32, 64)
        self.enc3 = base.PartialConvBlock(64, 128)
        self.mid = base.PartialConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = base.PartialConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = base.PartialConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = base.PartialConvBlock(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def initial_channel_mask(
        self, observed: torch.Tensor, valid_grid: torch.Tensor
    ) -> torch.Tensor:
        if observed.ndim != 4 or valid_grid.ndim != 4:
            raise ValueError("observed and valid_grid must be BCHW tensors")
        if observed.shape[1] != 1 or valid_grid.shape[1] != 1:
            raise ValueError("observed and valid_grid must each have one channel")
        if observed.shape != valid_grid.shape:
            raise ValueError(
                f"Mask shape mismatch: observed={observed.shape}, "
                f"valid_grid={valid_grid.shape}"
            )
        valid = (valid_grid > 0).to(observed.dtype)
        concentration_valid = (observed > 0).to(observed.dtype) * valid
        auxiliary_valid = valid.expand(
            -1, self.in_channels - 1, -1, -1
        )
        return torch.cat([concentration_valid, auxiliary_valid], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        known: torch.Tensor,
        valid_grid: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, "
                f"got {tuple(x.shape)}"
            )
        input_mask = self.initial_channel_mask(observed, valid_grid)
        # Invalid-grid cells must never affect a partial convolution.  torch.where
        # also avoids NaN*0 propagation if padded cells contain non-finite values.
        safe_x = torch.where(input_mask > 0, x, torch.zeros_like(x))

        e1, m1 = self.enc1(safe_x, input_mask)
        e2, m2 = self.enc2(self.pool(e1), self.pool(m1))
        e3, m3 = self.enc3(self.pool(e2), self.pool(m2))
        middle, mm = self.mid(self.pool(e3), self.pool(m3))

        u3 = self.up3(middle)
        um3 = F.interpolate(mm, size=e3.shape[-2:], mode="nearest")
        d3, dm3 = self.dec3(
            torch.cat([u3, e3], dim=1),
            torch.cat([um3.expand_as(u3), m3.expand_as(e3)], dim=1),
        )
        u2 = self.up2(d3)
        um2 = F.interpolate(dm3, size=e2.shape[-2:], mode="nearest")
        d2, dm2 = self.dec2(
            torch.cat([u2, e2], dim=1),
            torch.cat([um2.expand_as(u2), m2.expand_as(e2)], dim=1),
        )
        u1 = self.up1(d2)
        um1 = F.interpolate(dm2, size=e1.shape[-2:], mode="nearest")
        d1, _ = self.dec1(
            torch.cat([u1, e1], dim=1),
            torch.cat([um1.expand_as(u1), m1.expand_as(e1)], dim=1),
        )
        return base.enforce_known_pixels(self.output(d1), observed, known)


def sha256_json(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def data_signature(data: base.PollutantData) -> str:
    files: dict[str, dict[str, int]] = {}
    for name in (
        "conc_img_raw.npy",
        "conc_img_full.npy",
        "conc_scaler.pkl",
        "cov_img_full.npy",
        "coord_img.npy",
        "station_positions.npy",
        "time_splits.npy",
        "valid_time_indices.npy",
        "valid_grid_mask.npy",
    ):
        path = data.data_dir / name
        if path.exists():
            stat = path.stat()
            files[name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    split_payload = {
        key: array_digest(np.asarray(value, dtype=np.int64))
        for key, value in sorted(data.time_splits.items())
    }
    return sha256_json(
        {
            "pollutant": data.pollutant,
            "concentration_shape": list(data.conc_stored.shape),
            "covariate_shape": list(data.covariates.shape),
            "feature_dim": data.feature_dim,
            "station_positions": array_digest(data.station_positions),
            "time_splits": split_payload,
            "original_time_indices": array_digest(data.original_time_indices),
            "valid_grid_mask": array_digest(data.valid_grid_mask),
            "files": files,
        }
    )


def training_signature(config: Config, input_channels: int) -> str:
    return sha256_json(
        {
            "version": config.MODEL_VERSION,
            "model": config.MODEL_NAME,
            "input_channels": input_channels,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "start_mask_ratio": config.START_MASK_RATIO,
            "final_mask_ratio": config.FINAL_MASK_RATIO,
            "curriculum_epoch": config.CURRICULUM_EPOCH,
            "learning_rate": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "patience": config.PATIENCE,
            "huber_weight": config.HUBER_WEIGHT,
            "mse_weight": config.MSE_WEIGHT,
            "importance_beta": config.IMPORTANCE_BETA,
            "fixed_neighbor_radius": config.FIXED_NEIGHBOR_RADIUS,
        }
    )


def checkpoint_paths(
    config: Config, pollutant: str, heldout_station: int
) -> tuple[Path, Path]:
    checkpoint = Path(
        config.RESULT_DIR,
        "models",
        f"{pollutant}_station_{heldout_station:02d}_pconv_unet_cov.pth",
    )
    return checkpoint, Path(str(checkpoint) + ".complete.json")


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(
            checkpoint, map_location=base.DEVICE, weights_only=True
        )
    except TypeError:  # PyTorch < 2.0
        return torch.load(checkpoint, map_location=base.DEVICE)


def checkpoint_fingerprint(checkpoint: Path) -> dict[str, int]:
    stat = checkpoint.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def marker_matches(
    marker: dict[str, Any],
    data: base.PollutantData,
    heldout_station: int,
    data_sig: str,
    train_sig: str,
) -> bool:
    return bool(
        marker.get("status") == "completed"
        and marker.get("version") == MODEL_VERSION
        and marker.get("seed") == GLOBAL_SEED
        and marker.get("model") == MODEL_NAME
        and marker.get("pollutant") == data.pollutant
        and marker.get("heldout_station") == heldout_station
        and marker.get("feature_dim") == data.feature_dim
        and marker.get("input_channels") == 7 + data.feature_dim
        and marker.get("data_signature") == data_sig
        and marker.get("training_signature") == train_sig
    )


def save_checkpoint_atomic(model: nn.Module, checkpoint: Path) -> None:
    temporary = Path(str(checkpoint) + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)


def train_model(
    data: base.PollutantData,
    scaler: base.FoldScaler,
    heldout_station: int,
    config: Config,
    data_sig: str,
    train_sig: str,
) -> tuple[PConvUNetCov, float, int, int, dict[str, int]]:
    fold_seed = base.stable_seed(data.pollutant, heldout_station, MODEL_NAME)
    base.seed_everything(fold_seed)
    input_channels = 7 + data.feature_dim
    model = PConvUNetCov(input_channels).to(base.DEVICE)
    parameter_count = base.model_parameter_count(model)
    checkpoint, marker_path = checkpoint_paths(
        config, data.pollutant, heldout_station
    )

    if (
        config.REUSE_COMPLETED_FOLDS
        and checkpoint.exists()
        and marker_path.exists()
    ):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker_matches(
                marker, data, heldout_station, data_sig, train_sig
            ) and marker.get("checkpoint") == checkpoint_fingerprint(checkpoint):
                model.load_state_dict(load_state_dict(checkpoint))
                return (
                    model.eval(),
                    float(marker.get("train_seconds", 0.0)),
                    int(marker.get("best_epoch", -1)),
                    parameter_count,
                    checkpoint_fingerprint(checkpoint),
                )
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            pass

    # A partially trained fold is deliberately restarted.  Fully completed
    # folds are reused, so interruption never sends the whole LOSO run to fold 0.
    checkpoint.unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)
    Path(str(checkpoint) + ".tmp").unlink(missing_ok=True)

    train_dataset = base.GridMaskDataset(
        data, scaler, heldout_station, "train", config
    )
    val_dataset = base.GridMaskDataset(
        data, scaler, heldout_station, "val", config
    )
    train_loader = base.make_grid_loader(
        train_dataset, config, True, fold_seed
    )
    val_loader = base.make_grid_loader(
        val_dataset, config, False, fold_seed
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=config.EPOCHS,
    )
    amp_scaler = GradScaler(enabled=base.USE_AMP)
    best_validation = float("inf")
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
        train_losses: list[float] = []

        for batch in train_loader:
            inputs, observed, known, target, valid = base.move_grid_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=base.USE_AMP):
                prediction = model(inputs, observed, known, valid)
                loss = base.supervised_reconstruction_loss(
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
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, observed, known, target, valid = base.move_grid_batch(batch)
                prediction = model(inputs, observed, known, valid)
                loss = base.supervised_reconstruction_loss(
                    prediction, target, inputs, valid, config
                )
                if torch.isfinite(loss):
                    validation_losses.append(float(loss.item()))

        mean_validation = (
            float(np.mean(validation_losses))
            if validation_losses
            else float("inf")
        )
        if mean_validation < best_validation:
            best_validation = mean_validation
            best_epoch = epoch + 1
            save_checkpoint_atomic(model, checkpoint)
            patience = 0
        else:
            patience += 1

        if epoch == 0 or (epoch + 1) % config.LOG_INTERVAL == 0:
            mean_train = (
                float(np.mean(train_losses)) if train_losses else float("nan")
            )
            print(
                f"    {MODEL_NAME} epoch {epoch + 1:03d} | "
                f"train={mean_train:.6f} val={mean_validation:.6f} "
                f"best={best_validation:.6f}",
                flush=True,
            )
        if patience >= config.PATIENCE:
            break

    if not checkpoint.exists():
        raise RuntimeError(f"{MODEL_NAME} did not produce a finite checkpoint")
    model.load_state_dict(load_state_dict(checkpoint))
    elapsed = time.time() - started
    fingerprint = checkpoint_fingerprint(checkpoint)
    marker = {
        "status": "completed",
        "version": MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "model": MODEL_NAME,
        "pollutant": data.pollutant,
        "heldout_station": heldout_station,
        "feature_dim": data.feature_dim,
        "input_channels": input_channels,
        "data_signature": data_sig,
        "training_signature": train_sig,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "train_seconds": elapsed,
        "parameter_count": parameter_count,
        "checkpoint": fingerprint,
    }
    atomic_write_json(marker_path, marker)
    return model.eval(), elapsed, best_epoch, parameter_count, fingerprint


def predict_model(
    model: PConvUNetCov,
    data: base.PollutantData,
    scaler: base.FoldScaler,
    heldout_station: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = base.GridMaskDataset(
        data, scaler, heldout_station, "test", config
    )
    loader = base.make_grid_loader(
        dataset,
        config,
        False,
        base.stable_seed(data.pollutant, heldout_station, "test"),
        batch_size=config.EVAL_BATCH_SIZE,
    )
    row, col = data.station_positions[heldout_station]
    predictions: list[float] = []
    targets: list[float] = []
    original_times: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, target, valid = base.move_grid_batch(batch)
            prediction = model(inputs, observed, known, valid)
            pred_values = prediction[:, 0, row, col].detach().cpu().numpy()
            true_values = target[:, 0, row, col].detach().cpu().numpy()
            time_values = np.asarray(batch["original_time_idx"], dtype=np.int64)
            finite = np.isfinite(pred_values) & np.isfinite(true_values)
            predictions.extend(pred_values[finite].astype(float).tolist())
            targets.extend(true_values[finite].astype(float).tolist())
            original_times.extend(time_values[finite].astype(int).tolist())
    return (
        np.asarray(predictions, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(original_times, dtype=np.int64),
    )


def fold_paths(
    config: Config, pollutant: str, heldout_station: int
) -> tuple[Path, Path, Path]:
    stem = f"{pollutant}_fold_{heldout_station:02d}"
    return (
        Path(config.RESULT_DIR, "folds", f"{stem}_predictions.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}_efficiency.csv"),
        Path(config.RESULT_DIR, "folds", f"{stem}.complete.json"),
    )


def reusable_fold(
    config: Config,
    data: base.PollutantData,
    heldout_station: int,
    data_sig: str,
    train_sig: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    prediction_path, efficiency_path, marker_path = fold_paths(
        config, data.pollutant, heldout_station
    )
    checkpoint, checkpoint_marker = checkpoint_paths(
        config, data.pollutant, heldout_station
    )
    if not config.REUSE_COMPLETED_FOLDS:
        return None
    required = (
        prediction_path,
        efficiency_path,
        marker_path,
        checkpoint,
        checkpoint_marker,
    )
    if not all(path.exists() for path in required):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint_meta = json.loads(
            checkpoint_marker.read_text(encoding="utf-8")
        )
        fingerprint = checkpoint_fingerprint(checkpoint)
        if not marker_matches(
            marker, data, heldout_station, data_sig, train_sig
        ):
            return None
        if not marker_matches(
            checkpoint_meta, data, heldout_station, data_sig, train_sig
        ):
            return None
        if marker.get("checkpoint") != fingerprint:
            return None
        predictions = pd.read_csv(prediction_path)
        efficiency = pd.read_csv(efficiency_path)
        required_prediction_columns = {
            "pollutant",
            "station",
            "time_idx",
            "model",
            "true",
            "prediction",
        }
        if not required_prediction_columns.issubset(predictions.columns):
            return None
        if len(predictions) != int(marker.get("n_predictions", -1)):
            return None
        if set(predictions["model"].unique()) != {MODEL_NAME}:
            return None
        return predictions, efficiency
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def run_fold(
    config: Config,
    data: base.PollutantData,
    heldout_station: int,
    data_sig: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_channels = 7 + data.feature_dim
    train_sig = training_signature(config, input_channels)
    reused = reusable_fold(
        config, data, heldout_station, data_sig, train_sig
    )
    if reused is not None:
        print(f"  station {heldout_station:02d}: reuse completed fold", flush=True)
        return reused

    print(
        f"  station {heldout_station:02d}/{data.n_stations - 1:02d} | "
        f"input_channels={input_channels}",
        flush=True,
    )
    scaler = data.fold_scaler(heldout_station)
    model, train_seconds, best_epoch, parameter_count, fingerprint = train_model(
        data,
        scaler,
        heldout_station,
        config,
        data_sig,
        train_sig,
    )
    started = time.time()
    normalized_prediction, normalized_target, original_times = predict_model(
        model, data, scaler, heldout_station, config
    )
    inference_seconds = time.time() - started
    rows = base.prediction_rows(
        data.pollutant,
        heldout_station,
        MODEL_NAME,
        normalized_prediction,
        normalized_target,
        original_times,
        scaler,
    )
    if not rows:
        raise RuntimeError(
            f"No finite test predictions for station {heldout_station}"
        )
    predictions = pd.DataFrame(rows)
    efficiency = pd.DataFrame(
        [
            {
                "pollutant": data.pollutant,
                "station": heldout_station,
                "model": MODEL_NAME,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "best_epoch": best_epoch,
                "parameter_count": parameter_count,
                "input_channels": input_channels,
                "mask_policy": "concentration=observed; auxiliaries=valid_grid",
            }
        ]
    )

    prediction_path, efficiency_path, marker_path = fold_paths(
        config, data.pollutant, heldout_station
    )
    atomic_write_csv(prediction_path, predictions)
    atomic_write_csv(efficiency_path, efficiency)
    marker = {
        "status": "completed",
        "version": MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "model": MODEL_NAME,
        "pollutant": data.pollutant,
        "heldout_station": heldout_station,
        "feature_dim": data.feature_dim,
        "input_channels": input_channels,
        "data_signature": data_sig,
        "training_signature": train_sig,
        "checkpoint": fingerprint,
        "n_predictions": len(predictions),
    }
    atomic_write_json(marker_path, marker)

    fold_metric = base.station_metrics(predictions).iloc[0]
    print(
        f"    station {heldout_station:02d}: "
        f"MAE={fold_metric['MAE']:.4f}, "
        f"RMSE={fold_metric['RMSE']:.4f}, "
        f"R2={fold_metric['R2']:.4f}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, efficiency


def run_pollutant(config: Config, pollutant: str) -> None:
    data = base.PollutantData(config, pollutant)
    if config.FOLD_START >= data.n_stations:
        raise ValueError(
            f"FOLD_START={config.FOLD_START} but {pollutant} has only "
            f"{data.n_stations} stations"
        )
    stop = data.n_stations
    if config.MAX_STATIONS > 0:
        stop = min(stop, config.FOLD_START + config.MAX_STATIONS)
    folds = range(config.FOLD_START, stop)
    data_sig = data_signature(data)
    all_predictions: list[pd.DataFrame] = []
    all_efficiency: list[pd.DataFrame] = []
    print(
        f"\n{pollutant.upper()}: strict LOSO {MODEL_NAME} over folds "
        f"{config.FOLD_START}..{stop - 1} | stations={data.n_stations} | "
        f"features={data.feature_dim}",
        flush=True,
    )

    for heldout_station in folds:
        prediction_frame, efficiency_frame = run_fold(
            config, data, heldout_station, data_sig
        )
        all_predictions.append(prediction_frame)
        all_efficiency.append(efficiency_frame)

    predictions = pd.concat(all_predictions, ignore_index=True)
    efficiency = pd.concat(all_efficiency, ignore_index=True)
    stations = base.station_metrics(predictions)
    aggregate = base.aggregate_metrics(predictions, stations, config)

    atomic_write_csv(
        Path(
            config.RESULT_DIR,
            "predictions",
            f"{pollutant}_test_predictions.csv",
        ),
        predictions,
    )
    atomic_write_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_station_metrics.csv"),
        stations,
    )
    atomic_write_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_aggregate_metrics.csv"),
        aggregate,
    )
    atomic_write_csv(
        Path(config.RESULT_DIR, "metrics", f"{pollutant}_efficiency.csv"),
        efficiency,
    )
    print(
        "\n"
        + aggregate[
            [
                "model",
                "n_stations",
                "pooled_MAE",
                "pooled_RMSE",
                "pooled_R2",
                "mean_station_R2",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    config = Config()
    validate_config(config)
    prepare_output_dirs(config)
    print(
        f"device={base.DEVICE} | seed={GLOBAL_SEED} | "
        f"result_dir={config.RESULT_DIR}",
        flush=True,
    )
    for pollutant in config.POLLUTANTS:
        run_pollutant(config, pollutant)


if __name__ == "__main__":
    main()
