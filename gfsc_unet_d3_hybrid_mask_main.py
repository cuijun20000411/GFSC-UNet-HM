"""GFSC-UNet D3-HM独立主程序。

D3-HM保留D3的分组协变量编码、多尺度门控残差融合、观测保持和污染物特异的
TV/拉普拉斯权重，仅将训练代理任务改为双形态遮蔽：

* random：在可见站点中分散随机遮蔽；
* block：随机选择一个中心站，遮蔽与其空间距离最近的若干站点。

训练集在每个epoch中均衡交替两种遮蔽；内部验证分别固定random与block遮蔽，
用两者的等权平均损失选择检查点。外层留出站/空间块在训练输入、训练监督、
内部验证和浓度标准化中始终不可见。

模式：
  spatial-block  在已固定的5个空间块上训练D3-HM，用于与原D3/IDW直接比较（默认）。
  loso           从头运行55折严格LOSO。
  full-field     训练/复用全站模型，输出完整浓度场。
  diagnostics    读取本程序已完成的D3-HM LOSO预测并生成定量图。
  all            依次执行spatial-block、loso、diagnostics和full-field（计算量很大）。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast

import rh4_final_strict_light as core
import fusion_final_strict as fusion
from hybrid_masking import (
    balanced_mask_kind,
    deterministic_validation_seed,
    select_masked_stations,
    validate_spatial_partition,
)


POLLUTANT_LABELS = {"no2": r"NO$_2$", "o3": r"O$_3$", "pm25": r"PM$_{2.5}$"}
GLOBAL_SEED = 42
DEVICE = core.DEVICE
USE_AMP = core.USE_AMP


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


class D3Config(fusion.Config):
    """固定后的论文主模型配置；不再从消融选择器读取结构。"""

    MODEL_VERSION = "gfsc_unet_d3_hybrid_mask_v1"
    MASKING_PROTOCOL_VERSION = "balanced_random_nearest_block_v1"
    VALIDATION_MASK_SEED = 42
    USE_SCB_ATTENTION = False
    GROUP_WIDTH = 8
    GROUP_DROPOUT = 0.25
    GATE_BIAS = -3.0
    IMPORTANCE_BETA = 0.0
    START_MASK_RATIO = 0.05
    FINAL_MASK_RATIO = 0.25
    CURRICULUM_EPOCH = 50
    FIXED_NEIGHBOR_RADIUS = 2
    HUBER_WEIGHT = 0.8
    MSE_WEIGHT = 0.2
    DIFFUSION_COEFF = 0.1
    TV_WEIGHTS = {"no2": 0.01, "o3": 0.10, "pm25": 0.05}
    LAPLACIAN_WEIGHTS = {"no2": 0.01, "o3": 0.01, "pm25": 0.01}
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 150
    PATIENCE = 15
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LOG_INTERVAL = 10
    REUSE_CHECKPOINTS = True


class GFSCUNetD3(fusion.FusionSCUNet):
    """D3网络本体不变；HM只改变训练遮蔽和内部验证任务。"""


def d3_variant(config: D3Config) -> fusion.FusionVariant:
    return fusion.FusionVariant(
        code="D3HM",
        description="D3 with balanced random and contiguous spatial masking",
        use_covariates=True,
        bottleneck_fusion=True,
        decoder_fusion=True,
        group_dropout=float(config.GROUP_DROPOUT),
        ablate_group=None,
    )


class HybridMaskFusionDataset(fusion.FusionDataset):
    """Strict outer holdout with balanced random/block inner masking.

    ``outer_heldout`` may contain one station (LOSO), several stations
    (spatial-block validation), or be empty (full-region model).  The fold
    scaler is always re-fitted with training-time values from non-held-out
    stations only.  ``mask_view`` is ``mixed`` for training and fixed to
    ``random``/``block`` for the two deterministic validation views.
    """

    MASK_CODES = {"none": 0, "random": 1, "block": 2}

    def __init__(
        self,
        config: D3Config,
        pollutant: str,
        outer_heldout: list[int] | tuple[int, ...],
        mode: str,
        variant: fusion.FusionVariant,
        mask_view: str = "mixed",
    ):
        if mask_view not in {"mixed", "random", "block", "none"}:
            raise ValueError(f"Unsupported mask_view: {mask_view}")
        if mode == "test" and mask_view != "none":
            raise ValueError("Test data must use mask_view='none'")
        if mode in {"train", "val"} and mask_view == "none":
            raise ValueError("Training/validation requires a masking view")
        heldout = tuple(sorted({int(value) for value in outer_heldout}))
        self.outer_heldout = heldout
        self.mask_view = mask_view
        self.current_epoch = 0

        # The base constructor is used only for loading/auditing data.  Its
        # single-station masking is bypassed because __getitem__ is replaced.
        super().__init__(config, pollutant, -1, mode=mode, variant=variant)
        if any(index < 0 or index >= self.total_stations for index in heldout):
            raise IndexError("outer_heldout contains an invalid station id")
        self.candidate_stations = np.asarray(
            [index for index in range(self.total_stations) if index not in heldout],
            dtype=np.int64,
        )
        if len(self.candidate_stations) < 2:
            raise ValueError("At least two non-held-out stations are required")

        # Re-fit normalization after excluding the complete outer holdout.
        train_raw = np.asarray(self.conc_img_raw[self.train_time_indices])
        if self.conc_scaler is not None:
            train_raw = (
                train_raw * float(self.conc_scaler.scale_[0])
                + float(self.conc_scaler.mean_[0])
            )
        rows, cols = self.station_positions[self.candidate_stations].T
        fit_values = train_raw[:, rows, cols].reshape(-1)
        fit_values = fit_values[np.isfinite(fit_values)]
        if fit_values.size == 0:
            raise ValueError("No finite training labels outside the outer holdout")
        self.fold_mean = float(fit_values.mean())
        self.fold_std = float(fit_values.std())
        if self.fold_std < 1e-8:
            self.fold_std = 1.0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _validation_rng(self, current_t: int, mask_kind: str) -> np.random.Generator:
        seed = deterministic_validation_seed(
            self.config.VALIDATION_MASK_SEED, current_t, self.pollutant,
            self.outer_heldout, mask_kind,
        )
        return np.random.default_rng(seed)

    def _mask_kind(self, sample_index: int) -> str:
        if self.mask_view != "mixed":
            return self.mask_view
        if self.mode != "train":
            raise RuntimeError("Mixed masks are only allowed in the training split")
        # Exact balance (difference at most one sample) without a tunable mixing
        # probability.  Epoch parity lets every time index see both mask types.
        return balanced_mask_kind(sample_index, self.current_epoch)

    def _select_targets(
        self, sample_index: int, current_t: int, mask_kind: str
    ) -> list[int]:
        candidates = self.candidate_stations
        number = max(1, int(round(len(candidates) * self.mask_ratio)))
        number = min(number, len(candidates))
        rng = None if self.mode == "train" else self._validation_rng(current_t, mask_kind)
        return select_masked_stations(
            candidates, self.station_positions, number, mask_kind, rng=rng
        )

    def __getitem__(self, idx: int) -> dict:
        array_idx = int(self.time_indices[idx])
        original_time_idx = int(self.valid_time_indices_original[array_idx])
        sparse = self._to_fold_normalized(
            np.asarray(self.conc_img_raw[array_idx], dtype=np.float32)
        ).copy()
        covariates = np.asarray(self.cov_img_full[array_idx], dtype=np.float32).copy()
        if not np.isfinite(covariates).all():
            raise ValueError(f"{self.pollutant}: non-finite covariates at array index {array_idx}")
        target = np.full(self.config.IMG_SHAPE, np.nan, dtype=np.float32)

        # Outer stations are absent in every split.  They become targets only
        # in the test split and never influence inner validation/normalization.
        for station_idx in self.outer_heldout:
            row, col = self.station_positions[station_idx]
            value = sparse[row, col]
            sparse[row, col] = np.nan
            if self.mode == "test":
                target[row, col] = value

        mask_kind = "none"
        selected: list[int] = []
        if self.mode in {"train", "val"}:
            mask_kind = self._mask_kind(idx)
            selected = self._select_targets(idx, array_idx, mask_kind)
            for station_idx in selected:
                row, col = self.station_positions[station_idx]
                target[row, col] = sparse[row, col]
                sparse[row, col] = np.nan

        observed_mask = np.isfinite(sparse) & (self.valid_grid_mask > 0)
        known_values = sparse.copy()
        observed_values = sparse[observed_mask]
        sample_mean = float(observed_values.mean()) if observed_values.size else 0.0
        sample_std = float(observed_values.std()) if observed_values.size > 1 else 1.0
        if sample_std < 1e-6:
            sample_std = 1.0

        filled = np.where(observed_mask, sparse, sample_mean).astype(np.float32)
        neighbor = core.observed_neighbor_features(
            sparse, observed_mask, self.radius, sample_mean, sample_std
        )
        distance = distance_transform_edt(~observed_mask).astype(np.float32)
        valid_distance = distance[self.valid_grid_mask > 0]
        maximum = float(valid_distance.max()) if valid_distance.size else float(distance.max())
        distance /= max(maximum, 1.0)
        distance *= self.valid_grid_mask

        base_features = np.concatenate(
            [filled[None], neighbor, self.coord_feat, distance[None]], axis=0
        ).astype(np.float32)
        model_input = np.concatenate([base_features, covariates], axis=0)
        if not self.fusion_variant.use_covariates:
            model_input = model_input[: self.base_channels]
        infer_mask = ((self.valid_grid_mask > 0) & (~observed_mask)).astype(np.float32)
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
            "mask_type": self.MASK_CODES[mask_kind],
            "masked_station_count": len(selected),
        }


def resolve_reference_root(path: str | Path) -> Path:
    root = Path(path)
    return root if root.name == "D3" else root / "D3"


def configure(args: argparse.Namespace) -> D3Config:
    config = D3Config()
    config.DATA_ROOT_DIR = args.data_root
    config.EPOCHS = args.epochs
    config.PATIENCE = args.patience
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.LOG_INTERVAL = args.log_interval
    config.REUSE_CHECKPOINTS = True
    core.GLOBAL_SEED = GLOBAL_SEED
    fusion.GLOBAL_SEED = GLOBAL_SEED
    core.Config = D3Config
    fusion.Config = D3Config
    seed_everything(GLOBAL_SEED)
    return config


def load_groups(reference_root: Path, pollutant: str, feature_dim: int, config: D3Config):
    protocol_path = reference_root / f"protocol_{pollutant}.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"缺少D3协议文件: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    observed_ablation = protocol.get("ablation", {})
    checks = {
        "variant": observed_ablation.get("code") == "D3",
        "seed": protocol.get("seed") == GLOBAL_SEED,
        "scb_channel_attention": protocol.get("scb_channel_attention") is False,
        "bottleneck_fusion": protocol.get("bottleneck_fusion") is True,
        "decoder_fusion": protocol.get("decoder_fusion") is True,
        "group_width": protocol.get("group_width") == config.GROUP_WIDTH,
        "group_dropout": np.isclose(protocol.get("group_dropout", np.nan), config.GROUP_DROPOUT),
        "importance_beta": np.isclose(protocol.get("importance_beta", np.nan), 0.0),
        "tv_weights": protocol.get("tv_weights") == config.TV_WEIGHTS,
        "laplacian_weights": protocol.get("laplacian_weights") == config.LAPLACIAN_WEIGHTS,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"既有D3协议与最终主模型不一致: {failed}")
    payload = protocol.get("resolved_groups")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{protocol_path}没有resolved_groups")
    groups = [
        fusion.GroupSpec(str(item["name"]), tuple(int(i) for i in item["indices"]), str(item["kind"]))
        for item in payload
    ]
    flat = [index for group in groups for index in group.indices]
    if sorted(flat) != list(range(feature_dim)) or len(flat) != len(set(flat)):
        raise ValueError(f"{pollutant}: 协变量分组没有恰好覆盖全部{feature_dim}个通道")
    return groups, protocol_path, protocol


def validate_data_files(data_root: Path, pollutant: str) -> None:
    folder = data_root / pollutant
    required = [
        "cov_img_full.npy", "coord_img.npy", "station_positions.npy",
        "valid_time_indices.npy", "time_splits.npy", "valid_grid_mask.npy",
    ]
    if not (folder / "conc_img_raw.npy").exists():
        required.extend(["conc_img_full.npy", "conc_scaler.pkl"])
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{pollutant}缺少数据文件: {missing}")
    valid = np.load(folder / "valid_grid_mask.npy")
    if valid.shape != (32, 32) or int(np.count_nonzero(valid)) != 999:
        raise ValueError(f"{pollutant}: valid_grid_mask必须对应32×32中的999个有效网格")


def raw_concentrations(folder: Path) -> np.ndarray:
    raw_path = folder / "conc_img_raw.npy"
    if raw_path.exists():
        return np.load(raw_path, mmap_mode="r")
    values = np.load(folder / "conc_img_full.npy", mmap_mode="r")
    scaler = joblib.load(folder / "conc_scaler.pkl")
    return np.asarray(values) * float(scaler.scale_[0]) + float(scaler.mean_[0])


def validate_reference_predictions(args, pollutant: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data_folder = Path(args.data_root) / pollutant
    positions = np.load(data_folder / "station_positions.npy").astype(int)
    valid_mask = np.load(data_folder / "valid_grid_mask.npy").astype(bool)
    prediction_path = Path(args.output) / "loso" / "D3HM" / "predictions" / f"{pollutant}_test_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"缺少本程序的D3-HM留出站预测: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    required = {"station", "time_idx", "true", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{prediction_path}缺少列: {sorted(required-set(frame.columns))}")
    if "variant" in frame and set(frame["variant"].astype(str)) != {"D3HM"}:
        raise ValueError(f"{prediction_path}并非纯D3-HM结果")
    if frame.duplicated(["station", "time_idx"]).any():
        raise ValueError(f"{pollutant}: 存在重复站点—时刻预测")
    if not np.isfinite(frame[["true", "prediction"]].to_numpy()).all():
        raise ValueError(f"{pollutant}: 预测文件包含非有限值")
    station_ids = set(frame["station"].astype(int))
    if station_ids != set(range(len(positions))):
        raise ValueError(f"{pollutant}: D3-HM预测未覆盖全部{len(positions)}个站点")

    original_times = np.load(data_folder / "valid_time_indices.npy").astype(int)
    splits = np.load(data_folder / "time_splits.npy", allow_pickle=True).item()
    test_array_ids = np.asarray(splits["test"], dtype=int)
    raw = raw_concentrations(data_folder)
    expected = []
    for station, (row, col) in enumerate(positions):
        for array_idx in test_array_ids:
            truth = float(raw[array_idx, row, col])
            if np.isfinite(truth):
                expected.append((station, int(original_times[array_idx]), truth))
    expected_df = pd.DataFrame(expected, columns=["station", "time_idx", "expected_true"])
    merged = expected_df.merge(frame, on=["station", "time_idx"], how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{pollutant}: D3-HM预测与当前测试站点—时刻集合不一致")
    if not np.allclose(merged["expected_true"], merged["true"], rtol=2e-6, atol=2e-5):
        raise ValueError(f"{pollutant}: 预测文件中的真值与当前数据不一致")
    frame = frame.copy()
    frame["station"] = frame["station"].astype(int)
    frame["error"] = frame["prediction"] - frame["true"]
    return frame, positions, valid_mask


def setup_figure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def save_figure(fig, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix("." + fmt), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_diagnostics(args) -> None:
    setup_figure_style()
    figure_dir = Path(args.output) / "figures"
    metric_dir = Path(args.output) / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    formats = [part.strip() for part in args.figure_formats.split(",") if part.strip()]
    payload = {}
    station_frames = []
    aggregate_rows = []
    for pollutant in args.pollutants:
        frame, positions, valid = validate_reference_predictions(args, pollutant)
        station_rows = []
        for station, part in frame.groupby("station", sort=True):
            values = metric_dict(part["true"].to_numpy(), part["prediction"].to_numpy())
            row, col = positions[int(station)]
            station_rows.append({"pollutant": pollutant, "station": int(station), "row": int(row), "col": int(col), "n": len(part), **values})
        station_df = pd.DataFrame(station_rows)
        station_frames.append(station_df)
        values = metric_dict(frame["true"].to_numpy(), frame["prediction"].to_numpy())
        aggregate_rows.append({"pollutant": pollutant, "n_stations": len(station_df), "n_test_observations": len(frame), **values})
        payload[pollutant] = {"samples": frame, "stations": station_df, "positions": positions, "valid": valid}
    pd.concat(station_frames, ignore_index=True).to_csv(metric_dir / "D3HM_station_metrics_for_figures.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(metric_dir / "D3HM_aggregate_metrics_recomputed.csv", index=False)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.65 * len(args.pollutants), 3.35), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        frame = payload[pollutant]["samples"]
        true = frame["true"].to_numpy(); prediction = frame["prediction"].to_numpy()
        limits = [float(min(true.min(), prediction.min())), float(max(true.max(), prediction.max()))]
        margin = 0.03 * max(limits[1] - limits[0], 1.0)
        limits = [limits[0] - margin, limits[1] + margin]
        ax.scatter(true, prediction, s=7, alpha=0.22, color="#315f8c", edgecolors="none", rasterized=True)
        ax.plot(limits, limits, color="#b04a3f", lw=1.1, ls="--")
        ax.set(xlim=limits, ylim=limits, xlabel=r"Observed concentration ($\mu$g m$^{-3}$)")
        if label_index == 0:
            ax.set_ylabel(r"Predicted concentration ($\mu$g m$^{-3}$)")
        metric = metric_dict(true, prediction)
        ax.text(0.04, 0.96, f"MAE = {metric['MAE']:.2f}\nRMSE = {metric['RMSE']:.2f}\n$R^2$ = {metric['R2']:.3f}",
                transform=ax.transAxes, va="top", ha="left", linespacing=1.35)
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.grid(color="#d9d9d9", lw=0.45, alpha=0.65)
    save_figure(fig, figure_dir / "Fig_D3HM_observed_vs_predicted", formats, args.dpi)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.25), constrained_layout=True)
    axes = np.atleast_1d(axes)
    background = ListedColormap(["#f1f1ed"])
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        item = payload[pollutant]; station_df = item["stations"]
        ax.imshow(np.ma.masked_where(~item["valid"], item["valid"].astype(float)), cmap=background, vmin=0, vmax=1, origin="upper")
        scatter = ax.scatter(station_df["col"], station_df["row"], c=station_df["RMSE"], cmap="magma", s=38,
                             edgecolors="black", linewidths=0.35)
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.set_xlabel("Grid column")
        if label_index == 0:
            ax.set_ylabel("Grid row")
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_aspect("equal")
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
        colorbar.set_label(r"Station RMSE ($\mu$g m$^{-3}$)")
    save_figure(fig, figure_dir / "Fig_D3HM_station_error_maps", formats, args.dpi)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.05), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        station_df = payload[pollutant]["stations"].sort_values("RMSE").reset_index(drop=True)
        ax.plot(np.arange(1, len(station_df)+1), station_df["RMSE"], color="#315f8c", lw=1.2)
        ax.scatter(np.arange(1, len(station_df)+1), station_df["RMSE"], s=10, color="#315f8c")
        ax.axhline(station_df["RMSE"].median(), color="#b04a3f", lw=0.9, ls="--", label="Median")
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.set_xlabel("Stations ranked by RMSE")
        if label_index == 0:
            ax.set_ylabel(r"RMSE ($\mu$g m$^{-3}$)")
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
    save_figure(fig, figure_dir / "Fig_D3HM_stationwise_RMSE", formats, args.dpi)
    print(f"D3-HM定量图已生成: {figure_dir}", flush=True)


def model_fingerprint(
    config,
    pollutant,
    station_idx,
    groups,
    protocol_path,
    outer_heldout: list[int] | tuple[int, ...] | None = None,
) -> str:
    heldout = tuple(sorted(outer_heldout if outer_heldout is not None else ([station_idx] if station_idx >= 0 else [])))
    data_folder = Path(config.DATA_ROOT_DIR) / pollutant
    input_files = [
        data_folder / name for name in (
            "conc_img_raw.npy", "conc_img_full.npy", "conc_scaler.pkl",
            "cov_img_full.npy", "coord_img.npy", "station_positions.npy",
            "valid_time_indices.npy", "time_splits.npy", "valid_grid_mask.npy",
        ) if (data_folder / name).exists()
    ]
    return stable_hash({
        "model_version": config.MODEL_VERSION,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name, "hybrid_masking.py",
                "rh4_final_strict_light.py", "fusion_final_strict.py",
            )
        },
        "seed": GLOBAL_SEED,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "outer_heldout": heldout,
        "groups": [asdict(group) for group in groups],
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "data_files": {item.name: {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns} for item in input_files},
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "masking": [config.START_MASK_RATIO, config.FINAL_MASK_RATIO, config.CURRICULUM_EPOCH],
        "masking_protocol": config.MASKING_PROTOCOL_VERSION,
        "training_mask_views": ["random", "nearest-station block"],
        "training_balance": "deterministic alternating 1:1 within each epoch",
        "validation_views": ["fixed random", "fixed nearest-station block"],
        "validation_score": "equal mean of random and block losses",
        "validation_mask_seed": config.VALIDATION_MASK_SEED,
        "tv": config.TV_WEIGHTS,
        "laplacian": config.LAPLACIAN_WEIGHTS,
    })


def train_or_load(
    config,
    pollutant,
    station_idx,
    groups,
    protocol_path,
    checkpoint,
    outer_heldout: list[int] | tuple[int, ...] | None = None,
):
    variant = d3_variant(config)
    heldout = tuple(sorted(outer_heldout if outer_heldout is not None else ([station_idx] if station_idx >= 0 else [])))
    train_dataset = HybridMaskFusionDataset(
        config, pollutant, heldout, mode="train", variant=variant, mask_view="mixed"
    )
    val_random_dataset = HybridMaskFusionDataset(
        config, pollutant, heldout, mode="val", variant=variant, mask_view="random"
    )
    val_block_dataset = HybridMaskFusionDataset(
        config, pollutant, heldout, mode="val", variant=variant, mask_view="block"
    )
    fingerprint = model_fingerprint(
        config, pollutant, station_idx, groups, protocol_path, heldout
    )
    marker_path = Path(str(checkpoint) + ".complete.json")
    seed = fusion.fold_seed(pollutant, station_idx)
    seed_everything(seed)
    model = GFSCUNetD3(train_dataset.base_channels, train_dataset.F_dim, groups, variant, config).to(DEVICE)
    if checkpoint.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") == "completed" and marker.get("fingerprint") == fingerprint:
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            print(f"复用D3-HM检查点: {pollutant}, fold={station_idx}", flush=True)
            return model.eval(), train_dataset, marker
        raise RuntimeError(f"检查点配置与当前主程序不同，请更换输出目录: {checkpoint}")

    # A lone weight/marker file is an interrupted fold, never a completed one.
    checkpoint.unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_loader = core.make_loader(train_dataset, config, shuffle=True)
    val_random_loader = core.make_loader(val_random_dataset, config, shuffle=False)
    val_block_loader = core.make_loader(val_block_dataset, config, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LR,
                                               steps_per_epoch=max(1, len(train_loader)), epochs=config.EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)
    best_loss = float("inf"); best_random = float("inf"); best_block = float("inf")
    best_epoch = -1; waiting = 0; started = time.time()
    for epoch in range(config.EPOCHS):
        ratio = config.START_MASK_RATIO if epoch < config.CURRICULUM_EPOCH else config.FINAL_MASK_RATIO
        train_dataset.set_epoch(epoch)
        train_dataset.set_mask_ratio(ratio)
        val_random_dataset.set_mask_ratio(config.FINAL_MASK_RATIO)
        val_block_dataset.set_mask_ratio(config.FINAL_MASK_RATIO)
        model.train(); train_losses = []
        train_random_samples = 0; train_block_samples = 0
        for batch in train_loader:
            inputs, observed, known, targets, valid = core.move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                prediction = model(inputs, observed, known)
                loss, _ = fusion.fusion_loss(config, pollutant, prediction, targets, valid)
            if torch.isfinite(loss):
                train_losses.append(float(loss.item()))
                mask_codes = batch["mask_type"].detach().cpu().numpy().reshape(-1)
                train_random_samples += int(np.sum(mask_codes == HybridMaskFusionDataset.MASK_CODES["random"]))
                train_block_samples += int(np.sum(mask_codes == HybridMaskFusionDataset.MASK_CODES["block"]))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            scheduler.step()
        model.eval()
        validation_by_view = {}
        with torch.no_grad():
            for name, loader in (("random", val_random_loader), ("block", val_block_loader)):
                values = []
                for batch in loader:
                    inputs, observed, known, targets, valid = core.move_batch(batch)
                    prediction = model(inputs, observed, known)
                    loss, _ = fusion.fusion_loss(config, pollutant, prediction, targets, valid)
                    if torch.isfinite(loss): values.append(float(loss.item()))
                validation_by_view[name] = float(np.mean(values)) if values else float("inf")
        current = 0.5 * (validation_by_view["random"] + validation_by_view["block"])
        if current < best_loss:
            best_loss = current; best_epoch = epoch + 1; waiting = 0
            best_random = validation_by_view["random"]
            best_block = validation_by_view["block"]
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
        if epoch == 0 or (epoch + 1) % config.LOG_INTERVAL == 0:
            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            print(
                f"  {pollutant.upper()} fold={station_idx} epoch={epoch+1:03d} "
                f"train={train_mean:.6f} n_r={train_random_samples} n_b={train_block_samples} "
                f"val_r={validation_by_view['random']:.6f} val_b={validation_by_view['block']:.6f} "
                f"val={current:.6f} best={best_loss:.6f}", flush=True
            )
        if waiting >= config.PATIENCE: break
    if not checkpoint.exists(): raise RuntimeError("训练未产生有限的验证检查点")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    marker = {"status": "completed", "fingerprint": fingerprint, "model_version": config.MODEL_VERSION,
              "pollutant": pollutant, "station_idx": station_idx, "seed": GLOBAL_SEED, "fold_seed": seed,
              "outer_heldout": list(heldout), "masking_protocol": config.MASKING_PROTOCOL_VERSION,
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "best_validation_random_loss": best_random,
              "best_validation_block_loss": best_block,
              "train_seconds": time.time()-started, "parameter_count": sum(p.numel() for p in model.parameters())}
    atomic_json(marker_path, marker)
    return model.eval(), train_dataset, marker


def evaluate_loso(config, pollutant, station_idx, model):
    dataset = HybridMaskFusionDataset(
        config, pollutant, [station_idx], mode="test", variant=d3_variant(config), mask_view="none"
    )
    loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
    row, col = dataset.station_positions[station_idx]
    records = []
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, target, _ = core.move_batch(batch)
            prediction = model(inputs, observed, known)
            true_z = float(target[0, 0, row, col].cpu()); pred_z = float(prediction[0, 0, row, col].cpu())
            if np.isfinite(true_z) and np.isfinite(pred_z):
                true = true_z*dataset.fold_std+dataset.fold_mean; pred = pred_z*dataset.fold_std+dataset.fold_mean
                records.append({"variant":"D3HM", "pollutant":pollutant, "station":station_idx,
                                "time_idx":int(batch["time_idx"].item()), "true":true, "prediction":pred, "error":pred-true})
    frame = pd.DataFrame(records)
    if frame.empty: raise RuntimeError(f"{pollutant} station {station_idx}: 无有效测试样本")
    return frame, {"variant":"D3HM", "pollutant":pollutant, "station":station_idx, "n_test":len(frame),
                   **metric_dict(frame.true, frame.prediction), "fold_train_mean":dataset.fold_mean, "fold_train_std":dataset.fold_std}


def run_loso(args, config):
    root = Path(args.output) / "loso" / "D3HM"
    for pollutant in args.pollutants:
        probe = HybridMaskFusionDataset(
            config, pollutant, [0], mode="test", variant=d3_variant(config), mask_view="none"
        )
        groups, protocol_path, _ = load_groups(resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config)
        total = probe.total_stations; del probe
        stop = total if args.max_folds <= 0 else min(total, args.fold_start + args.max_folds)
        fold_dir = root / "folds"; fold_dir.mkdir(parents=True, exist_ok=True)
        for station_idx in range(args.fold_start, stop):
            checkpoint = root / "models" / f"{pollutant}_station_{station_idx:02d}_D3HM.pth"
            model, _, info = train_or_load(config, pollutant, station_idx, groups, protocol_path, checkpoint)
            frame, summary = evaluate_loso(config, pollutant, station_idx, model)
            summary.update({k:v for k,v in info.items() if k in {
                "best_epoch","best_validation_loss","best_validation_random_loss",
                "best_validation_block_loss","train_seconds","parameter_count"}})
            frame.to_csv(fold_dir / f"{pollutant}_station_{station_idx:02d}_predictions.csv", index=False)
            atomic_json(fold_dir / f"{pollutant}_station_{station_idx:02d}_metrics.json", summary)
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # 从磁盘重新收集所有与当前配置相容的已完成折，支持分段运行和恢复。
        pollutant_frames = []; pollutant_rows = []
        for station_idx in range(total):
            checkpoint = root / "models" / f"{pollutant}_station_{station_idx:02d}_D3HM.pth"
            marker_path = Path(str(checkpoint) + ".complete.json")
            prediction_path = fold_dir / f"{pollutant}_station_{station_idx:02d}_predictions.csv"
            metric_path = fold_dir / f"{pollutant}_station_{station_idx:02d}_metrics.json"
            if not (marker_path.exists() and prediction_path.exists() and metric_path.exists()): continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected = model_fingerprint(config, pollutant, station_idx, groups, protocol_path, [station_idx])
            if marker.get("status") != "completed" or marker.get("fingerprint") != expected: continue
            frame = pd.read_csv(prediction_path)
            if frame.empty or set(frame["station"].astype(int)) != {station_idx}: continue
            pollutant_frames.append(frame)
            pollutant_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
        if pollutant_frames:
            prediction_df = pd.concat(pollutant_frames, ignore_index=True); station_df = pd.DataFrame(pollutant_rows)
            (root/"predictions").mkdir(parents=True, exist_ok=True); (root/"metrics").mkdir(parents=True, exist_ok=True)
            prediction_df.to_csv(root/"predictions"/f"{pollutant}_test_predictions.csv", index=False)
            station_df.to_csv(root/"metrics"/f"{pollutant}_station_metrics.csv", index=False)
            overall = metric_dict(prediction_df.true, prediction_df.prediction)
            pd.DataFrame([{ "variant":"D3HM", "pollutant":pollutant, "n_stations":station_df.station.nunique(),
                            "n_test_observations":len(prediction_df), **overall,
                            "mean_station_MAE":station_df.MAE.mean(), "mean_station_RMSE":station_df.RMSE.mean(),
                            "mean_station_R2":station_df.R2.mean(), "median_station_R2":station_df.R2.median()}]).to_csv(
                                root/"metrics"/f"{pollutant}_aggregate_metrics.csv", index=False)
    print(f"独立D3-HM LOSO输出: {root}", flush=True)


def load_spatial_blocks(path: str | Path, station_positions: np.ndarray) -> tuple[list[list[int]], dict]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"缺少既有空间划分文件: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    raw_parts = payload.get("heldout_by_fold")
    if not isinstance(raw_parts, list) or len(raw_parts) < 2:
        raise ValueError("spatial_split.json缺少有效的heldout_by_fold")
    parts = validate_spatial_partition(raw_parts, len(station_positions))
    saved_positions = payload.get("station_positions")
    if saved_positions is not None and not np.array_equal(
        np.asarray(saved_positions, dtype=np.int64), station_positions.astype(np.int64)
    ):
        raise ValueError("空间划分中的站点位置与当前数据不一致")
    return parts, payload


def evaluate_outer_holdout(config, pollutant, heldout, fold, model):
    dataset = HybridMaskFusionDataset(
        config, pollutant, heldout, mode="test", variant=d3_variant(config), mask_view="none"
    )
    loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
    records = []
    gate_records: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, target, _ = core.move_batch(batch)
            output = model(inputs, observed, known)
            for station_idx in heldout:
                row, col = dataset.station_positions[station_idx]
                true_z = float(target[0, 0, row, col].cpu())
                pred_z = float(output[0, 0, row, col].cpu())
                if not (np.isfinite(true_z) and np.isfinite(pred_z)):
                    continue
                true = true_z * dataset.fold_std + dataset.fold_mean
                pred = pred_z * dataset.fold_std + dataset.fold_mean
                records.append(
                    {"variant":"D3HM", "pollutant":pollutant, "fold":int(fold),
                     "station":int(station_idx), "time_idx":int(batch["time_idx"].item()),
                     "true":true, "prediction":pred, "error":pred-true}
                )
            for name, value in model.gate_means().items():
                gate_records.setdefault(name, []).append(float(value))
    frame = pd.DataFrame(records)
    if frame.empty or frame.duplicated(["station", "time_idx"]).any():
        raise RuntimeError(f"{pollutant} block {fold}: empty or duplicated test predictions")

    # Exact key/truth audit against the current raw dataset.
    expected = []
    for array_idx in dataset.time_indices:
        normalized = dataset._to_fold_normalized(
            np.asarray(dataset.conc_img_raw[int(array_idx)], dtype=np.float32)
        )
        for station_idx in heldout:
            row, col = dataset.station_positions[station_idx]
            value = float(normalized[row, col])
            if np.isfinite(value):
                expected.append(
                    {"station":int(station_idx),
                     "time_idx":int(dataset.valid_time_indices_original[int(array_idx)]),
                     "true_expected":value * dataset.fold_std + dataset.fold_mean}
                )
    audit = pd.DataFrame(expected).merge(
        frame, on=["station", "time_idx"], how="outer", indicator=True
    )
    if not (audit["_merge"] == "both").all():
        raise RuntimeError(f"{pollutant} block {fold}: missing or extra test predictions")
    if not np.allclose(audit["true_expected"], audit["true"], rtol=2e-6, atol=2e-5):
        raise RuntimeError(f"{pollutant} block {fold}: stored truths disagree with raw data")
    summary = {
        "variant":"D3HM", "pollutant":pollutant, "fold":int(fold),
        "heldout_stations":list(map(int, heldout)), "n_stations":frame.station.nunique(),
        "n_test":len(frame), **metric_dict(frame.true, frame.prediction),
        "fold_train_mean":dataset.fold_mean, "fold_train_std":dataset.fold_std,
    }
    for name, values in gate_records.items():
        summary[f"gate_{name}_mean"] = float(np.mean(values))
    return frame, summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_spatial_block(args, config):
    root = Path(args.output) / "spatial_block" / "D3HM"
    (root / "folds").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)

    first = HybridMaskFusionDataset(
        config, args.pollutants[0], [], mode="test", variant=d3_variant(config), mask_view="none"
    )
    parts, split_payload = load_spatial_blocks(args.spatial_split_json, first.station_positions)
    split_hash = stable_hash(split_payload)
    atomic_json(root / "spatial_split_used.json", {**split_payload, "source":str(Path(args.spatial_split_json).resolve())})
    print(f"Spatial block sizes: {[len(part) for part in parts]}", flush=True)
    stop = len(parts) if args.max_folds <= 0 else min(len(parts), args.fold_start + args.max_folds)
    selected_folds = set(range(args.fold_start, stop))
    if not selected_folds:
        raise ValueError("fold-start/max-folds did not select any spatial block")

    for pollutant in args.pollutants:
        probe = HybridMaskFusionDataset(
            config, pollutant, parts[0], mode="test", variant=d3_variant(config), mask_view="none"
        )
        groups, protocol_path, _ = load_groups(
            resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config
        )
        del probe
        for fold, heldout in enumerate(parts):
            prediction_path = root / "folds" / f"{pollutant}_block_{fold:02d}_predictions.csv"
            metric_path = root / "folds" / f"{pollutant}_block_{fold:02d}_metrics.json"
            result_marker = root / "folds" / f"{pollutant}_block_{fold:02d}.complete.json"
            checkpoint = root / "models" / f"{pollutant}_block_{fold:02d}_D3HM.pth"
            fingerprint = model_fingerprint(
                config, pollutant, fold, groups, protocol_path, heldout
            )
            reusable = False
            if prediction_path.exists() and metric_path.exists() and result_marker.exists():
                marker = json.loads(result_marker.read_text(encoding="utf-8"))
                reusable = (
                    marker.get("status") == "completed"
                    and marker.get("fingerprint") == fingerprint
                    and marker.get("split_hash") == split_hash
                    and marker.get("prediction_sha256") == file_sha256(prediction_path)
                )
            if reusable:
                print(f"复用D3-HM空间块结果: {pollutant} block={fold}", flush=True)
                continue
            if fold not in selected_folds:
                continue
            print(f"\nD3-HM {pollutant} block {fold}: held out {heldout}", flush=True)
            model, _, training_info = train_or_load(
                config, pollutant, fold, groups, protocol_path, checkpoint,
                outer_heldout=heldout,
            )
            frame, summary = evaluate_outer_holdout(
                config, pollutant, heldout, fold, model
            )
            summary.update({
                key:value for key, value in training_info.items()
                if key in {"best_epoch", "best_validation_loss",
                           "best_validation_random_loss", "best_validation_block_loss",
                           "train_seconds", "parameter_count"}
            })
            temporary = prediction_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False); os.replace(temporary, prediction_path)
            atomic_json(metric_path, summary)
            atomic_json(result_marker, {
                "status":"completed", "fingerprint":fingerprint,
                "split_hash":split_hash, "heldout_stations":heldout,
                "n_predictions":len(frame),
                "prediction_sha256":file_sha256(prediction_path),
            })
            print(metric_dict(frame.true, frame.prediction), flush=True)
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_predictions = []; fold_rows = []; completion_rows = []
    for pollutant in args.pollutants:
        probe = HybridMaskFusionDataset(
            config, pollutant, parts[0], mode="test", variant=d3_variant(config), mask_view="none"
        )
        groups, protocol_path, _ = load_groups(
            resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config
        )
        del probe
        for fold, heldout in enumerate(parts):
            prediction_path = root / "folds" / f"{pollutant}_block_{fold:02d}_predictions.csv"
            metric_path = root / "folds" / f"{pollutant}_block_{fold:02d}_metrics.json"
            marker_path = root / "folds" / f"{pollutant}_block_{fold:02d}.complete.json"
            fingerprint = model_fingerprint(config, pollutant, fold, groups, protocol_path, heldout)
            complete = False
            if prediction_path.exists() and metric_path.exists() and marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                complete = (
                    marker.get("status") == "completed"
                    and marker.get("fingerprint") == fingerprint
                    and marker.get("split_hash") == split_hash
                    and marker.get("prediction_sha256") == file_sha256(prediction_path)
                )
            completion_rows.append({"variant":"D3HM", "pollutant":pollutant,
                                    "fold":fold, "complete":complete})
            if complete:
                all_predictions.append(pd.read_csv(prediction_path))
                fold_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))

    completion = pd.DataFrame(completion_rows)
    completion.to_csv(root / "metrics" / "completion_status.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "metrics" / "fold_metrics.csv", index=False)
    if all_predictions:
        predictions = pd.concat(all_predictions, ignore_index=True)
        if predictions.duplicated(["pollutant", "station", "time_idx"]).any():
            raise RuntimeError("Merged spatial-block predictions contain duplicate keys")
        predictions.to_csv(root / "predictions" / "test_predictions.csv", index=False)
        station_rows = []
        aggregate_rows = []
        for (pollutant, station), part in predictions.groupby(["pollutant", "station"], sort=True):
            station_rows.append({"variant":"D3HM", "pollutant":pollutant,
                                 "station":station, "n_test":len(part),
                                 **metric_dict(part.true, part.prediction)})
        station_metrics = pd.DataFrame(station_rows)
        station_metrics.to_csv(root / "metrics" / "station_metrics.csv", index=False)
        for pollutant, part in predictions.groupby("pollutant", sort=True):
            station_part = station_metrics[station_metrics.pollutant == pollutant]
            state = completion[completion.pollutant == pollutant]
            aggregate_rows.append({
                "variant":"D3HM", "pollutant":pollutant,
                "n_stations":part.station.nunique(), "n_test_observations":len(part),
                "completed_folds":int(state.complete.sum()), "expected_folds":len(state),
                "complete":bool(state.complete.all()),
                **{f"pooled_{key}":value for key, value in metric_dict(part.true, part.prediction).items()},
                "mean_station_MAE":float(station_part.MAE.mean()),
                "mean_station_RMSE":float(station_part.RMSE.mean()),
                "mean_station_R2":float(station_part.R2.mean()),
                "median_station_R2":float(station_part.R2.median()),
            })
        aggregate = pd.DataFrame(aggregate_rows)
        aggregate.to_csv(root / "metrics" / "aggregate_metrics.csv", index=False)
        print(aggregate.to_string(index=False), flush=True)
    print(f"D3-HM空间分块输出: {root}", flush=True)


def reconstruct_full_fields(args, config):
    root = Path(args.output); result_dir = root / "full_field"; model_dir = root / "models"
    result_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pollutant in args.pollutants:
        probe = HybridMaskFusionDataset(
            config, pollutant, [], mode="test", variant=d3_variant(config), mask_view="none"
        )
        groups, protocol_path, protocol = load_groups(resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config)
        del probe
        checkpoint = model_dir / f"{pollutant}_full_region_D3HM.pth"
        model, _, training_info = train_or_load(
            config, pollutant, -1, groups, protocol_path, checkpoint, outer_heldout=[]
        )
        dataset = HybridMaskFusionDataset(
            config, pollutant, [], mode="test", variant=d3_variant(config), mask_view="none"
        )
        loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
        predictions=[]; observations=[]; times=[]; preservation=[]
        with torch.no_grad():
            for batch in loader:
                inputs, observed, known, _, _ = core.move_batch(batch)
                output = model(inputs, observed, known)
                difference = torch.abs(output[observed.bool()] - known[observed.bool()])
                preservation.append(float(difference.max().cpu()) if difference.numel() else 0.0)
                field = output[0,0].cpu().numpy()*dataset.fold_std+dataset.fold_mean
                known_raw = known[0,0].cpu().numpy()*dataset.fold_std+dataset.fold_mean
                visible = observed[0,0].cpu().numpy().astype(bool)
                field[dataset.valid_grid_mask<=0]=np.nan; known_raw[~visible]=np.nan; known_raw[dataset.valid_grid_mask<=0]=np.nan
                predictions.append(field.astype(np.float32)); observations.append(known_raw.astype(np.float32)); times.append(int(batch["time_idx"].item()))
        if max(preservation, default=0.0) > 1e-5: raise RuntimeError(f"{pollutant}: 可见站点硬保持检查失败")
        predictions=np.stack(predictions); observations=np.stack(observations); times=np.asarray(times,dtype=np.int64)
        archive=result_dir/f"{pollutant}_D3HM_full_region_test_fields.npz"
        np.savez_compressed(archive, predictions=predictions, known_observations=observations,
                            original_time_indices=times, station_positions=dataset.station_positions,
                            valid_grid_mask=dataset.valid_grid_mask, train_mean=dataset.fold_mean, train_std=dataset.fold_std)
        station_means=np.nanmean(observations,axis=(1,2)); target=float(np.nanquantile(station_means,args.representative_quantile))
        selected=int(np.nanargmin(np.abs(station_means-target)))
        records.append({"pollutant":pollutant,"array_index_in_test_output":selected,"original_time_idx":int(times[selected]),
                        "selection_rule":f"visible-station mean nearest quantile {args.representative_quantile}",
                        "visible_station_mean":float(station_means[selected]),"archive":str(archive),
                        "checkpoint":str(checkpoint),"best_epoch":training_info.get("best_epoch"),
                        "best_validation_loss":training_info.get("best_validation_loss")})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(result_dir/"representative_times.csv",index=False)
    plot_representative_fields(args, records)


def plot_representative_fields(args, records):
    setup_figure_style(); formats=[x.strip() for x in args.figure_formats.split(",") if x.strip()]
    fig,axes=plt.subplots(1,len(records),figsize=(3.7*len(records),3.35),constrained_layout=True);axes=np.atleast_1d(axes)
    for index,(ax,record) in enumerate(zip(axes,records)):
        data=np.load(record["archive"]);selected=int(record["array_index_in_test_output"])
        field=data["predictions"][selected];known=data["known_observations"][selected];positions=data["station_positions"]
        finite=field[np.isfinite(field)];vmin=float(finite.min());vmax=float(finite.max())
        image=ax.imshow(field,cmap="viridis",vmin=vmin,vmax=vmax,origin="upper")
        rr,cc=positions.T;station_values=known[rr,cc]
        ax.scatter(cc,rr,c=station_values,cmap="viridis",vmin=vmin,vmax=vmax,s=20,edgecolors="white",linewidths=.35)
        pollutant=record["pollutant"]
        ax.set_title(f"({chr(97+index)}) {POLLUTANT_LABELS[pollutant]}, time {record['original_time_idx']}",loc="left")
        ax.set_xlabel("Grid column")
        if index==0:ax.set_ylabel("Grid row")
        ax.set_xlim(-.5,31.5);ax.set_ylim(31.5,-.5);ax.set_aspect("equal")
        bar=fig.colorbar(image,ax=ax,fraction=.047,pad=.03);bar.set_label(r"Concentration ($\mu$g m$^{-3}$)")
    save_figure(fig,Path(args.output)/"figures"/"Fig_D3HM_representative_fields",formats,args.dpi)
    print(f"完整浓度场与代表性场图已生成: {Path(args.output)/'full_field'}",flush=True)


def write_run_protocol(args, config):
    payload={"program":"GFSC-UNet D3-HM independent main","model_version":config.MODEL_VERSION,"mode":args.mode,
             "seed":GLOBAL_SEED,"device":str(DEVICE),"data_root":str(Path(args.data_root).resolve()),
             "D3_reference_root":str(resolve_reference_root(args.d3_reference_root).resolve()),"output":str(Path(args.output).resolve()),
             "pollutants":args.pollutants,"architecture":{"SCB_channel_attention":False,"group_width":8,
             "group_dropout":.25,"bottleneck_gated_adapter":True,"decoder_gated_adapters":True},
             "training":{"epochs":config.EPOCHS,"patience":config.PATIENCE,"batch_size":config.BATCH_SIZE,
             "learning_rate":config.LR,"weight_decay":config.WEIGHT_DECAY,"mask_ratio":[.05,.25],"switch_epoch":50,
             "masking_protocol":config.MASKING_PROTOCOL_VERSION,
             "training_views":"balanced alternating random and nearest-station block masks",
             "validation_views":"fixed random plus fixed nearest-station block masks",
             "checkpoint_score":"equal mean of the two validation losses",
             "validation_mask_seed":config.VALIDATION_MASK_SEED},
             "spatial_weights":{"TV":config.TV_WEIGHTS,"Laplacian":config.LAPLACIAN_WEIGHTS},
             "interpretation":{"LOSO_diagnostics":"quantitative held-out-station evidence",
             "spatial_block":"quantitative contiguous held-out-region evidence",
             "full_field":"qualitative spatial reconstruction; unmonitored grids have no reference truth"}}
    atomic_json(Path(args.output)/"run_protocol.json",payload)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["spatial-block","diagnostics","full-field","all","loso"],default="spatial-block")
    parser.add_argument("--data-root",default="/etc/cj/boundary/duoyuan/data")
    parser.add_argument("--d3-reference-root",default="/etc/cj/boundary/f3_direct_ablation_final_strict_v1/D3")
    parser.add_argument("--output",default="/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1")
    parser.add_argument("--spatial-split-json",default="/etc/cj/boundary/d3_spatial_block_validation_v1/spatial_split.json")
    parser.add_argument("--pollutants",default="no2,o3,pm25")
    parser.add_argument("--epochs",type=int,default=150);parser.add_argument("--patience",type=int,default=15)
    parser.add_argument("--batch-size",type=int,default=32);parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--log-interval",type=int,default=10);parser.add_argument("--fold-start",type=int,default=0)
    parser.add_argument("--max-folds",type=int,default=0)
    parser.add_argument("--representative-quantile",type=float,default=.5)
    parser.add_argument("--figure-formats",default="png,pdf,svg");parser.add_argument("--dpi",type=int,default=400)
    parser.add_argument("--preflight",action="store_true",help="只检查数据、分组和空间划分，不训练")
    args=parser.parse_args();args.pollutants=[x.strip().lower() for x in args.pollutants.split(",") if x.strip()]
    if not args.pollutants or set(args.pollutants)-set(POLLUTANT_LABELS):raise ValueError("pollutants只能从no2,o3,pm25选择")
    if not 0<=args.representative_quantile<=1:raise ValueError("representative-quantile必须在0到1之间")
    if min(args.epochs,args.patience,args.batch_size,args.dpi)<=0 or args.num_workers<0:raise ValueError("训练/输出参数无效")
    if args.fold_start < 0 or args.max_folds < 0:raise ValueError("fold-start/max-folds不能为负数")
    return args


def main():
    args=parse_args();config=configure(args);Path(args.output).mkdir(parents=True,exist_ok=True)
    for pollutant in args.pollutants:
        validate_data_files(Path(args.data_root),pollutant)
        probe=HybridMaskFusionDataset(
            config,pollutant,[0],mode="test",variant=d3_variant(config),mask_view="none"
        )
        load_groups(resolve_reference_root(args.d3_reference_root),pollutant,probe.F_dim,config);del probe
    write_run_protocol(args,config)
    print(f"device={DEVICE} | mode={args.mode} | pollutants={args.pollutants}",flush=True)
    if args.preflight:
        if args.mode in {"spatial-block","all"}:
            probe=HybridMaskFusionDataset(
                config,args.pollutants[0],[],mode="test",variant=d3_variant(config),mask_view="none"
            )
            parts,_=load_spatial_blocks(args.spatial_split_json,probe.station_positions)
            print(f"preflight spatial blocks: {[len(part) for part in parts]}",flush=True)
        print("D3-HM preflight passed; no training was started.",flush=True)
        return
    if args.mode in {"spatial-block","all"}:run_spatial_block(args,config)
    if args.mode in {"loso","all"}:run_loso(args,config)
    if args.mode in {"diagnostics","all"}:build_diagnostics(args)
    if args.mode in {"full-field","all"}:reconstruct_full_fields(args,config)


if __name__=="__main__":
    main()
