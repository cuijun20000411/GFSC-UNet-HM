"""F3-centred strict LOSO ablations for the final grouped-fusion SC-UNet.

This runner complements ``fusion_final_strict.py``.  It does not retrain or
overwrite the completed F0--F3 experiments.  Instead, it trains only direct
ablations of the final F3 model and then combines their outputs with the
existing F0--F3 station metrics for paired analysis against F3.

Required neighbouring files
----------------------------
* ``rh4_final_strict_light.py``
* ``fusion_final_strict.py``

Direct F3 ablations
-------------------
D1  F3 without the distance-to-nearest-observation channel.
D2  F3 without local mean, local std, and sample/global observed mean; missing
    concentration pixels are initialized to the fold-normalized mean (zero).
D3  F3 without SCB channel attention.
D4  F3 with a fixed 25% training mask (no masking curriculum).
D5  F3 without the total-variation penalty.
D6  F3 without the Laplacian penalty.
D7  F3 with supervised loss only (no TV or Laplacian penalties).
D8  F3 with one ungrouped covariate encoder.  Its condition width is identical
    to F3 and its encoder parameter count is matched as closely as possible.
D9  F3 without the bottleneck adapter; decoder adapters and group dropout stay.
D10 F3 without decoder adapters; bottleneck adapter and group dropout stay.
D11 F3 without the dynamic-spatial covariate group.
D12 F3 without the static-spatial covariate group.
D13 F3 without the global-temporal covariate group.

Existing references automatically included when available
----------------------------------------------------------
F0  Seven observation-derived channels; no covariates.
F1  Grouped bottleneck gated residual fusion.
F2  F1 plus decoder fusion; no whole-group dropout.
F3  Final full model and the reference for statistical tests.

The defaults run the five highest-priority direct ablations D1,D2,D3,D4,D8.
The loss and adapter ablations can be launched in a second phase.  Every fold
uses the same pollutant/station seed convention as F0--F3.  Checkpoints and
fold outputs are fingerprinted so that changed hyperparameters are never
silently reused.

Examples
--------
One-fold smoke test::

    SCUNET_POLLUTANTS=no2 F3_ABLATION_VARIANTS=D1,D3 \
      F3_ABLATION_MAX_STATIONS=1 python -u f3_direct_ablation_final_strict.py

Run the recommended structural set::

    F3_ABLATION_VARIANTS=D1,D2,D3,D4,D8 \
      python -u f3_direct_ablation_final_strict.py

Run the loss and direct-path set::

    F3_ABLATION_VARIANTS=D5,D6,D7,D9,D10 \
      python -u f3_direct_ablation_final_strict.py

Run the leave-one-covariate-group-out set::

    F3_ABLATION_VARIANTS=D11,D12,D13 \
      python -u f3_direct_ablation_final_strict.py

Run every direct ablation (very expensive: 13 x pollutants x stations)::

    F3_ABLATION_VARIANTS=ALL python -u f3_direct_ablation_final_strict.py

Rebuild only the cross-variant summaries from completed files::

    F3_ABLATION_SUMMARY_ONLY=1 python -u f3_direct_ablation_final_strict.py

Important environment variables
-------------------------------
``F3_ABLATION_RESULT_ROOT``
    New output root.  Default:
    ``/etc/cj/boundary/f3_direct_ablation_final_strict_v1``.
``F3_REFERENCE_RESULT_ROOT``
    Existing F0--F3 root.  Default:
    ``/etc/cj/boundary/fusion_final_strict_v1``.
``F3_ABLATION_VARIANTS``
    Comma-separated subset of D1--D13, or ``ALL``.
``F3_ABLATION_SEED``
    Must match the existing F3 seed for paired use; default 42.
``F3_ABLATION_FOLD_START`` / ``F3_ABLATION_MAX_STATIONS``
    Resume/range controls.  Completed folds are reused by default.

The outer held-out station is never used in training, validation, scaling, or
checkpoint selection.  Test truth is used only after model selection, and all
reported metrics are calculated after inverse transformation in raw units.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast

try:
    import rh4_final_strict_light as core
    import fusion_final_strict as fusion
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Place f3_direct_ablation_final_strict.py beside "
        "rh4_final_strict_light.py and fusion_final_strict.py."
    ) from exc


GLOBAL_SEED = int(
    os.environ.get("F3_ABLATION_SEED", os.environ.get("FUSION_SEED", "42"))
)
DEVICE = core.DEVICE
USE_AMP = core.USE_AMP
core.GLOBAL_SEED = GLOBAL_SEED
fusion.GLOBAL_SEED = GLOBAL_SEED
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def fold_seed(pollutant: str, station_idx: int) -> int:
    """Exactly match the deterministic fold seed used by F0--F3."""

    digest = hashlib.sha256(
        f"{GLOBAL_SEED}|{pollutant}|{station_idx}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((GLOBAL_SEED + offset) % (2**31 - 1))


seed_everything(GLOBAL_SEED)


@dataclass(frozen=True)
class DirectAblation:
    code: str
    description: str
    use_distance: bool = True
    use_statistics: bool = True
    use_scb_attention: bool = True
    fixed_mask_ratio: bool = False
    tv_multiplier: float = 1.0
    laplacian_multiplier: float = 1.0
    grouping: str = "grouped"
    bottleneck_fusion: bool = True
    decoder_fusion: bool = True
    ablate_group: str | None = None


ABLATIONS: dict[str, DirectAblation] = {
    "D1": DirectAblation(
        "D1", "F3 without distance-to-nearest-observation channel", use_distance=False
    ),
    "D2": DirectAblation(
        "D2",
        "F3 without local mean, local std, or sample/global observed mean",
        use_statistics=False,
    ),
    "D3": DirectAblation(
        "D3", "F3 without SCB channel attention", use_scb_attention=False
    ),
    "D4": DirectAblation(
        "D4", "F3 with fixed 25 percent training mask", fixed_mask_ratio=True
    ),
    "D5": DirectAblation(
        "D5", "F3 without total-variation penalty", tv_multiplier=0.0
    ),
    "D6": DirectAblation(
        "D6", "F3 without Laplacian penalty", laplacian_multiplier=0.0
    ),
    "D7": DirectAblation(
        "D7",
        "F3 with supervised loss only",
        tv_multiplier=0.0,
        laplacian_multiplier=0.0,
    ),
    "D8": DirectAblation(
        "D8",
        "F3 with a parameter-matched single ungrouped covariate encoder",
        grouping="ungrouped",
    ),
    "D9": DirectAblation(
        "D9",
        "F3 without bottleneck adapter",
        bottleneck_fusion=False,
        decoder_fusion=True,
    ),
    "D10": DirectAblation(
        "D10",
        "F3 without decoder adapters",
        bottleneck_fusion=True,
        decoder_fusion=False,
    ),
    "D11": DirectAblation(
        "D11",
        "F3 without the dynamic-spatial covariate group",
        ablate_group="dynamic_spatial",
    ),
    "D12": DirectAblation(
        "D12",
        "F3 without the static-spatial covariate group",
        ablate_group="static_spatial",
    ),
    "D13": DirectAblation(
        "D13",
        "F3 without the global-temporal covariate group",
        ablate_group="global_temporal",
    ),
}


class Config(fusion.Config):
    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    ABLATION_ROOT = os.environ.get(
        "F3_ABLATION_RESULT_ROOT",
        "/etc/cj/boundary/f3_direct_ablation_final_strict_v1",
    )
    REFERENCE_ROOT = os.environ.get(
        "F3_REFERENCE_RESULT_ROOT",
        "/etc/cj/boundary/fusion_final_strict_v1",
    )
    POLLUTANTS = [
        item.strip().lower()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    SELECTED_ABLATIONS = [
        item.strip().upper()
        for item in os.environ.get(
            "F3_ABLATION_VARIANTS", "D1,D2,D3,D4,D8"
        ).split(",")
        if item.strip()
    ]
    FOLD_START = int(os.environ.get("F3_ABLATION_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("F3_ABLATION_MAX_STATIONS", "0"))
    REUSE_CHECKPOINTS = os.environ.get("F3_ABLATION_REUSE_COMPLETED", "1") == "1"
    SUMMARY_ONLY = os.environ.get("F3_ABLATION_SUMMARY_ONLY", "0") == "1"
    ALLOW_REFERENCE_MISMATCH = (
        os.environ.get("F3_ABLATION_ALLOW_REFERENCE_MISMATCH", "0") == "1"
    )
    NUM_WORKERS = int(
        os.environ.get("F3_ABLATION_NUM_WORKERS", str(fusion.Config.NUM_WORKERS))
    )
    LOG_INTERVAL = int(
        os.environ.get("F3_ABLATION_LOG_INTERVAL", str(fusion.Config.LOG_INTERVAL))
    )
    EPOCHS = int(
        os.environ.get("F3_ABLATION_EPOCHS", str(fusion.Config.EPOCHS))
    )
    BATCH_SIZE = int(
        os.environ.get("F3_ABLATION_BATCH_SIZE", str(fusion.Config.BATCH_SIZE))
    )
    PATIENCE = int(
        os.environ.get("F3_ABLATION_PATIENCE", str(fusion.Config.PATIENCE))
    )
    BOOTSTRAP_REPEATS = int(
        os.environ.get("F3_ABLATION_BOOTSTRAP_REPEATS", "2000")
    )
    PAIR_BOOTSTRAP_REPEATS = int(
        os.environ.get("F3_ABLATION_PAIR_BOOTSTRAP_REPEATS", "5000")
    )
    MODEL_VERSION = "strict_f3_direct_ablation_v1"


def selected_ablations(config: Config) -> list[DirectAblation]:
    requested = list(config.SELECTED_ABLATIONS)
    if "ALL" in requested:
        requested = list(ABLATIONS)
    unknown = sorted(set(requested) - set(ABLATIONS))
    if unknown:
        raise ValueError(
            f"Unknown F3_ABLATION_VARIANTS={unknown}; available={list(ABLATIONS)}"
        )
    if not requested and not config.SUMMARY_ONLY:
        raise ValueError("F3_ABLATION_VARIANTS did not select any experiment")
    seen: set[str] = set()
    result = []
    for code in requested:
        if code not in seen:
            result.append(ABLATIONS[code])
            seen.add(code)
    return result


def fusion_variant(ablation: DirectAblation, config: Config) -> fusion.FusionVariant:
    return fusion.FusionVariant(
        code=ablation.code,
        description=ablation.description,
        use_covariates=True,
        bottleneck_fusion=ablation.bottleneck_fusion,
        decoder_fusion=ablation.decoder_fusion,
        group_dropout=config.GROUP_DROPOUT,
        ablate_group=ablation.ablate_group,
    )


class DirectAblationDataset(fusion.FusionDataset):
    """Apply only an F3-centred input ablation after strict sample creation."""

    def __init__(self, *args, ablation: DirectAblation, **kwargs):
        self.direct_ablation = ablation
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        model_input = sample["input"].clone()

        # Final strict base-channel order:
        # 0 filled concentration
        # 1 local observed-only mean
        # 2 local observed-only std
        # 3 sample/global observed mean
        # 4:6 normalized coordinates
        # 6 normalized distance to nearest observed station
        # 7: dense covariates
        if not self.direct_ablation.use_statistics:
            observed = sample["channel_mask"][0] > 0
            model_input[0] = torch.where(
                observed,
                model_input[0],
                torch.zeros_like(model_input[0]),
            )
            model_input[1:4] = 0.0
            known_values = sample["known_values"].clone()
            known_values[0] = torch.where(
                observed,
                known_values[0],
                torch.zeros_like(known_values[0]),
            )
            sample["known_values"] = known_values
        if not self.direct_ablation.use_distance:
            model_input[6] = 0.0
        sample["input"] = model_input
        return sample


def effective_groups_and_width(
    config: Config,
    ablation: DirectAblation,
    resolved_groups: list[fusion.GroupSpec],
    feature_dim: int,
) -> tuple[list[fusion.GroupSpec], int]:
    if ablation.grouping == "grouped":
        if ablation.ablate_group is not None:
            available = {group.name for group in resolved_groups}
            if ablation.ablate_group not in available:
                raise KeyError(
                    f"{ablation.code} requests group {ablation.ablate_group!r}; "
                    f"available groups are {sorted(available)}"
                )
            if len(resolved_groups) <= 1:
                raise ValueError(
                    f"{ablation.code} would remove the only covariate group"
                )
        return resolved_groups, config.GROUP_WIDTH
    if ablation.grouping == "ungrouped":
        if ablation.ablate_group is not None:
            raise ValueError("An ungrouped ablation cannot remove a named group")
        # Match the total encoded condition width: G grouped branches * W each.
        matched_width = config.GROUP_WIDTH * len(resolved_groups)
        return [
            fusion.GroupSpec(
                "all_covariates", tuple(range(feature_dim)), kind="spatial"
            )
        ], matched_width
    raise ValueError(f"Unknown grouping mode: {ablation.grouping}")


def grouped_encoder_parameter_budget(
    groups: list[fusion.GroupSpec], width: int
) -> int:
    """Exact affine-parameter count of Fusion's grouped encoder branches."""

    hidden = max(width, min(32, 2 * width))
    total = 0
    for group in groups:
        kernel_elements = 1 if group.kind == "global" else 9
        # Conv1 weight/bias + GN1 affine + Conv2 weight/bias + GN2 affine.
        total += hidden * (
            len(group.indices) + 3 + width * kernel_elements
        ) + 3 * width
    return int(total)


def matched_ungrouped_hidden(
    feature_dim: int,
    condition_channels: int,
    target_parameters: int,
) -> tuple[int, int]:
    """Choose a hidden width minimizing encoder parameter-count mismatch."""

    candidates = range(1, 129)

    def parameter_count(hidden: int) -> int:
        # The ungrouped branch uses two convolutions: 1x1 then spatial 3x3.
        return int(
            hidden * (feature_dim + 3 + 9 * condition_channels)
            + 3 * condition_channels
        )

    hidden = min(
        candidates,
        key=lambda value: (
            abs(parameter_count(value) - target_parameters),
            value,
        ),
    )
    return hidden, parameter_count(hidden)


class ParameterMatchedUngroupedEncoder(nn.Module):
    """One all-covariate branch with F3-matched output and parameter budgets."""

    def __init__(
        self,
        feature_dim: int,
        condition_channels: int,
        hidden_channels: int,
        dropout_probability: float,
    ):
        super().__init__()
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Group dropout must be in [0, 1)")
        self.out_channels = int(condition_channels)
        self.dropout_probability = float(dropout_probability)
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_channels, kernel_size=1),
            nn.GroupNorm(fusion.norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                condition_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(
                fusion.norm_groups(condition_channels), condition_channels
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        features = self.net(covariates)
        if self.training and self.dropout_probability > 0:
            keep = (
                torch.rand(
                    features.shape[0], 1, 1, 1, device=features.device
                )
                >= self.dropout_probability
            ).to(features.dtype)
            features = features * keep / (1.0 - self.dropout_probability)
        return features


class DirectFusionSCUNet(fusion.FusionSCUNet):
    """FusionSCUNet with a controlled parameter-matched D8 encoder swap."""

    def __init__(
        self,
        base_channels: int,
        cov_channels: int,
        effective_groups: list[fusion.GroupSpec],
        variant: fusion.FusionVariant,
        model_config: Config,
        ablation: DirectAblation,
        reference_groups: list[fusion.GroupSpec],
        reference_group_width: int,
    ):
        super().__init__(
            base_channels,
            cov_channels,
            effective_groups,
            variant,
            model_config,
        )
        self.ungrouped_hidden_channels: int | None = None
        self.encoder_parameter_target: int | None = None
        self.encoder_parameter_actual: int | None = None
        if ablation.grouping != "ungrouped":
            return
        if self.covariate_encoder is None:
            raise RuntimeError("D8 requires an initialized covariate encoder")

        condition_channels = reference_group_width * len(reference_groups)
        target_parameters = grouped_encoder_parameter_budget(
            reference_groups, reference_group_width
        )
        hidden_channels, actual_parameters = matched_ungrouped_hidden(
            cov_channels,
            condition_channels,
            target_parameters,
        )
        replacement = ParameterMatchedUngroupedEncoder(
            cov_channels,
            condition_channels,
            hidden_channels,
            variant.group_dropout,
        )
        if replacement.out_channels != self.covariate_encoder.out_channels:
            raise RuntimeError(
                "D8 condition width mismatch after parameter-matched encoder swap"
            )
        self.covariate_encoder = replacement
        self.ungrouped_hidden_channels = hidden_channels
        self.encoder_parameter_target = target_parameters
        self.encoder_parameter_actual = actual_parameters


def model_config_for(
    config: Config, ablation: DirectAblation, effective_width: int
) -> Config:
    model_config = copy.copy(config)
    model_config.GROUP_WIDTH = effective_width
    model_config.USE_SCB_ATTENTION = ablation.use_scb_attention
    return model_config


def direct_ablation_loss(
    config: Config,
    pollutant: str,
    ablation: DirectAblation,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_grid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supervised_mask = torch.isfinite(target) & (valid_grid_mask > 0)
    if supervised_mask.any():
        huber = F.smooth_l1_loss(
            prediction[supervised_mask],
            target[supervised_mask],
            beta=1.0,
            reduction="none",
        )
        mse = F.mse_loss(
            prediction[supervised_mask], target[supervised_mask], reduction="none"
        )
        supervised = (
            config.HUBER_WEIGHT * huber + config.MSE_WEIGHT * mse
        ).mean()
    else:
        supervised = prediction.sum() * 0.0

    if ablation.tv_multiplier > 0:
        tv = core.masked_total_variation(prediction, valid_grid_mask)
    else:
        tv = prediction.sum() * 0.0
    if ablation.laplacian_multiplier > 0:
        laplacian = core.masked_laplacian_penalty(
            prediction, valid_grid_mask, config.DIFFUSION_COEFF
        )
    else:
        laplacian = prediction.sum() * 0.0

    total = (
        supervised
        + ablation.tv_multiplier * config.TV_WEIGHTS[pollutant] * tv
        + ablation.laplacian_multiplier
        * config.LAPLACIAN_WEIGHTS[pollutant]
        * laplacian
    )
    return total, {"supervised": supervised, "tv": tv, "laplacian": laplacian}


def stable_hash(payload: dict | list) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_payload(
    config: Config,
    ablation: DirectAblation,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
) -> dict:
    active_effective_groups = [
        group
        for group in effective_groups
        if group.name != ablation.ablate_group
    ]
    condition_channels = effective_width * len(active_effective_groups)
    encoder_target = grouped_encoder_parameter_budget(
        resolved_groups, config.GROUP_WIDTH
    )
    if ablation.grouping == "ungrouped":
        ungrouped_hidden, encoder_actual = matched_ungrouped_hidden(
            sum(len(group.indices) for group in resolved_groups),
            condition_channels,
            encoder_target,
        )
    else:
        ungrouped_hidden = None
        encoder_actual = grouped_encoder_parameter_budget(
            active_effective_groups, effective_width
        )
    return {
        "model_version": config.MODEL_VERSION,
        "source_sha256": SOURCE_SHA256,
        "seed": GLOBAL_SEED,
        "ablation": asdict(ablation),
        "pollutant": pollutant,
        "outer_protocol": "station-wise strict LOSO",
        "inner_validation": "mask non-held-out training-fold stations only",
        "fold_scaler": "training times and non-held-out stations only",
        "time_split": "chronological train/validation/test from preprocessing",
        "prediction_clipping": False,
        "importance_beta": 0.0,
        "curriculum_masking": not ablation.fixed_mask_ratio,
        "fixed_training_mask_ratio": (
            config.FINAL_MASK_RATIO if ablation.fixed_mask_ratio else None
        ),
        "tv_multiplier": ablation.tv_multiplier,
        "laplacian_multiplier": ablation.laplacian_multiplier,
        "tv_weights": config.TV_WEIGHTS,
        "laplacian_weights": config.LAPLACIAN_WEIGHTS,
        "scb_channel_attention": ablation.use_scb_attention,
        "feature_group_source": group_source,
        "resolved_groups": [asdict(group) for group in resolved_groups],
        "effective_groups": [asdict(group) for group in effective_groups],
        "active_effective_groups": [
            asdict(group) for group in active_effective_groups
        ],
        "group_width": effective_width,
        "condition_channels": condition_channels,
        "encoder_parameter_target_F3": encoder_target,
        "encoder_parameter_actual": encoder_actual,
        "ungrouped_hidden_channels": ungrouped_hidden,
        "group_dropout": config.GROUP_DROPOUT,
        "gate_initial_bias": config.GATE_BIAS,
        "bottleneck_fusion": ablation.bottleneck_fusion,
        "decoder_fusion": ablation.decoder_fusion,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "start_mask_ratio": config.START_MASK_RATIO,
        "final_mask_ratio": config.FINAL_MASK_RATIO,
        "curriculum_epoch": config.CURRICULUM_EPOCH,
    }


def experiment_fingerprint(
    config: Config,
    ablation: DirectAblation,
    pollutant: str,
    station_idx: int,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
) -> str:
    payload = protocol_payload(
        config,
        ablation,
        pollutant,
        resolved_groups,
        effective_groups,
        effective_width,
        group_source,
    )
    payload["station_idx"] = station_idx
    payload["fold_seed"] = fold_seed(pollutant, station_idx)
    return stable_hash(payload)


def variant_root(config: Config, ablation: DirectAblation) -> Path:
    return Path(config.ABLATION_ROOT, ablation.code)


def prepare_variant_root(
    config: Config,
    ablation: DirectAblation,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
    audit: pd.DataFrame,
) -> Path:
    root = variant_root(config, ablation)
    for name in ("models", "folds", "metrics", "predictions"):
        (root / name).mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(
        config,
        ablation,
        pollutant,
        resolved_groups,
        effective_groups,
        effective_width,
        group_source,
    )
    protocol["protocol_signature"] = stable_hash(protocol)
    (root / f"protocol_{pollutant}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit.to_csv(
        root / "metrics" / f"{pollutant}_feature_channel_audit.csv", index=False
    )
    return root


def completion_marker(checkpoint: Path) -> Path:
    return Path(str(checkpoint) + ".complete.json")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def marker_matches(
    marker: dict,
    ablation: DirectAblation,
    pollutant: str,
    station_idx: int,
    fingerprint: str,
) -> bool:
    return (
        marker.get("status") == "completed"
        and marker.get("model_version") == Config.MODEL_VERSION
        and marker.get("seed") == GLOBAL_SEED
        and marker.get("variant") == ablation.code
        and marker.get("pollutant") == pollutant
        and marker.get("station_idx") == station_idx
        and marker.get("experiment_fingerprint") == fingerprint
    )


def make_dataset(
    config: Config,
    pollutant: str,
    station_idx: int,
    mode: str,
    ablation: DirectAblation,
    variant: fusion.FusionVariant,
) -> DirectAblationDataset:
    return DirectAblationDataset(
        config,
        pollutant,
        station_idx,
        mode=mode,
        variant=variant,
        ablation=ablation,
    )


def train_fold_model(
    config: Config,
    pollutant: str,
    station_idx: int,
    ablation: DirectAblation,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
    checkpoint: Path,
) -> tuple[DirectFusionSCUNet, dict[str, float | int | str]]:
    seed = fold_seed(pollutant, station_idx)
    seed_everything(seed)
    variant = fusion_variant(ablation, config)
    train_dataset = make_dataset(
        config, pollutant, station_idx, "train", ablation, variant
    )
    val_dataset = make_dataset(
        config, pollutant, station_idx, "val", ablation, variant
    )
    model_config = model_config_for(config, ablation, effective_width)
    model = DirectFusionSCUNet(
        train_dataset.base_channels,
        train_dataset.F_dim,
        effective_groups,
        variant,
        model_config,
        ablation,
        resolved_groups,
        config.GROUP_WIDTH,
    ).to(DEVICE)
    if model.covariate_encoder is None:
        raise RuntimeError(f"{ablation.code} unexpectedly has no covariate encoder")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    encoder_parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.covariate_encoder.parameters()
        )
    )
    encoder_parameter_target_f3 = grouped_encoder_parameter_budget(
        resolved_groups, config.GROUP_WIDTH
    )
    condition_channels = int(model.covariate_encoder.out_channels)
    fingerprint = experiment_fingerprint(
        config,
        ablation,
        pollutant,
        station_idx,
        resolved_groups,
        effective_groups,
        effective_width,
        group_source,
    )

    marker_path = completion_marker(checkpoint)
    if config.REUSE_CHECKPOINTS and checkpoint.exists() and marker_path.exists():
        marker = load_json(marker_path)
        if marker_matches(marker, ablation, pollutant, station_idx, fingerprint):
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            print(
                f"    reuse {ablation.code} {pollutant.upper()} "
                f"station {station_idx:02d}"
            )
            return model.eval(), {
                "train_seconds": float(marker.get("train_seconds", 0.0)),
                "best_epoch": int(marker.get("best_epoch", -1)),
                "best_validation_loss": float(
                    marker.get("best_validation_loss", np.nan)
                ),
                "parameter_count": parameter_count,
                "encoder_parameter_count": encoder_parameter_count,
                "encoder_parameter_target_F3": encoder_parameter_target_f3,
                "condition_channels": condition_channels,
                "experiment_fingerprint": fingerprint,
            }

    checkpoint.unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)
    train_loader = core.make_loader(train_dataset, config, shuffle=True)
    val_loader = core.make_loader(val_dataset, config, shuffle=False)
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
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    started = time.time()

    for epoch in range(config.EPOCHS):
        if ablation.fixed_mask_ratio:
            mask_ratio = config.FINAL_MASK_RATIO
        else:
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
            inputs, observed, known, targets, valid_domain = core.move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                predictions = model(inputs, observed, known)
                loss, _ = direct_ablation_loss(
                    config,
                    pollutant,
                    ablation,
                    predictions,
                    targets,
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

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, observed, known, targets, valid_domain = core.move_batch(batch)
                predictions = model(inputs, observed, known)
                loss, _ = direct_ablation_loss(
                    config,
                    pollutant,
                    ablation,
                    predictions,
                    targets,
                    valid_domain,
                )
                if torch.isfinite(loss):
                    validation_losses.append(float(loss.item()))
        mean_val = (
            float(np.mean(validation_losses))
            if validation_losses
            else float("inf")
        )
        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint)
            patience = 0
        else:
            patience += 1

        if epoch == 0 or (epoch + 1) % config.LOG_INTERVAL == 0:
            mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
            print(
                f"    {ablation.code} {pollutant.upper()} station {station_idx:02d} "
                f"epoch {epoch + 1:03d} | train={mean_train:.6f} "
                f"val={mean_val:.6f} best={best_val:.6f}"
            )
        if patience >= config.PATIENCE:
            break

    if not checkpoint.exists():
        raise RuntimeError("Training ended without a finite validation checkpoint")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    elapsed = time.time() - started
    marker = {
        "status": "completed",
        "model_version": config.MODEL_VERSION,
        "source_sha256": SOURCE_SHA256,
        "seed": GLOBAL_SEED,
        "fold_seed": seed,
        "variant": ablation.code,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "feature_dim": train_dataset.F_dim,
        "experiment_fingerprint": fingerprint,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "train_seconds": elapsed,
        "parameter_count": parameter_count,
        "encoder_parameter_count": encoder_parameter_count,
        "encoder_parameter_target_F3": encoder_parameter_target_f3,
        "condition_channels": condition_channels,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model.eval(), {
        "train_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "parameter_count": parameter_count,
        "encoder_parameter_count": encoder_parameter_count,
        "encoder_parameter_target_F3": encoder_parameter_target_f3,
        "condition_channels": condition_channels,
        "experiment_fingerprint": fingerprint,
    }


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def evaluate_fold(
    config: Config,
    pollutant: str,
    station_idx: int,
    ablation: DirectAblation,
    model: DirectFusionSCUNet,
) -> tuple[dict, pd.DataFrame]:
    variant = fusion_variant(ablation, config)
    dataset = make_dataset(
        config, pollutant, station_idx, "test", ablation, variant
    )
    loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
    row, col = dataset.station_positions[station_idx]
    records = []
    gate_records: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, targets, _ = core.move_batch(batch)
            output = model(inputs, observed, known)
            true_normalized = float(targets[0, 0, row, col].cpu())
            pred_normalized = float(output[0, 0, row, col].cpu())
            if not (np.isfinite(true_normalized) and np.isfinite(pred_normalized)):
                continue
            true_value = true_normalized * dataset.fold_std + dataset.fold_mean
            pred_value = pred_normalized * dataset.fold_std + dataset.fold_mean
            records.append(
                {
                    "variant": ablation.code,
                    "pollutant": pollutant,
                    "station": station_idx,
                    "time_idx": int(batch["time_idx"].item()),
                    "true": true_value,
                    "prediction": pred_value,
                    "error": pred_value - true_value,
                }
            )
            for name, value in model.gate_means().items():
                gate_records.setdefault(name, []).append(value)

    if not records:
        raise RuntimeError(f"No test predictions for station {station_idx}")
    samples = pd.DataFrame(records)
    y_true = samples["true"].to_numpy(dtype=np.float64)
    y_pred = samples["prediction"].to_numpy(dtype=np.float64)
    summary = {
        "variant": ablation.code,
        "description": ablation.description,
        "pollutant": pollutant,
        "station": station_idx,
        "n_test": len(samples),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
        "fold_train_mean": dataset.fold_mean,
        "fold_train_std": dataset.fold_std,
    }
    for name, values in gate_records.items():
        summary[f"gate_{name}_mean"] = float(np.mean(values))
    return summary, samples


def fold_paths(
    root: Path, pollutant: str, station_idx: int
) -> tuple[Path, Path, Path]:
    stem = f"{pollutant}_station_{station_idx:02d}"
    return (
        root / "folds" / f"{stem}_metrics.json",
        root / "folds" / f"{stem}_predictions.csv",
        root / "folds" / f"{stem}.complete.json",
    )


def reuse_completed_fold(
    config: Config,
    root: Path,
    ablation: DirectAblation,
    pollutant: str,
    station_idx: int,
    fingerprint: str,
) -> tuple[dict, pd.DataFrame] | None:
    if not config.REUSE_CHECKPOINTS:
        return None
    return load_completed_fold(
        root,
        ablation,
        pollutant,
        station_idx,
        fingerprint,
    )


def load_completed_fold(
    root: Path,
    ablation: DirectAblation,
    pollutant: str,
    station_idx: int,
    fingerprint: str,
) -> tuple[dict, pd.DataFrame] | None:
    """Read a valid fold independent of whether training reuse is enabled."""

    metric_path, prediction_path, marker_path = fold_paths(
        root, pollutant, station_idx
    )
    if not metric_path.exists() or not prediction_path.exists() or not marker_path.exists():
        return None
    marker = load_json(marker_path)
    if not marker_matches(marker, ablation, pollutant, station_idx, fingerprint):
        return None
    try:
        return load_json(metric_path), pd.read_csv(prediction_path)
    except (OSError, ValueError, KeyError):
        return None


def save_completed_fold(
    config: Config,
    root: Path,
    ablation: DirectAblation,
    pollutant: str,
    station_idx: int,
    fingerprint: str,
    summary: dict,
    samples: pd.DataFrame,
) -> None:
    metric_path, prediction_path, marker_path = fold_paths(
        root, pollutant, station_idx
    )
    metric_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    samples.to_csv(prediction_path, index=False)
    marker = {
        "status": "completed",
        "model_version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "variant": ablation.code,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "n_test": len(samples),
        "experiment_fingerprint": fingerprint,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def aggregate_results(
    config: Config,
    ablation: DirectAblation,
    pollutant: str,
    station_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    y_true = sample_df["true"].to_numpy(dtype=np.float64)
    y_pred = sample_df["prediction"].to_numpy(dtype=np.float64)
    row = {
        "variant": ablation.code,
        "description": ablation.description,
        "pollutant": pollutant,
        "n_stations": station_df["station"].nunique(),
        "n_test_observations": len(sample_df),
        "pooled_MAE": mean_absolute_error(y_true, y_pred),
        "pooled_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pooled_R2": safe_r2(y_true, y_pred),
        "mean_station_MAE": station_df["MAE"].mean(),
        "mean_station_RMSE": station_df["RMSE"].mean(),
        "mean_station_R2": station_df["R2"].mean(),
        "median_station_R2": station_df["R2"].median(),
        "mean_train_seconds": station_df.get("train_seconds", pd.Series(dtype=float)).mean(),
        "parameter_count": station_df.get("parameter_count", pd.Series(dtype=float)).max(),
        "encoder_parameter_count": station_df.get(
            "encoder_parameter_count", pd.Series(dtype=float)
        ).max(),
        "encoder_parameter_target_F3": station_df.get(
            "encoder_parameter_target_F3", pd.Series(dtype=float)
        ).max(),
        "condition_channels": station_df.get(
            "condition_channels", pd.Series(dtype=float)
        ).max(),
    }
    for metric in ("MAE", "RMSE", "R2"):
        low, high = bootstrap_mean_ci(
            station_df[metric].to_numpy(dtype=np.float64),
            config.BOOTSTRAP_REPEATS,
            fold_seed(pollutant, ord(metric[0])),
        )
        row[f"mean_station_{metric}_CI_low"] = low
        row[f"mean_station_{metric}_CI_high"] = high
    for column in station_df.columns:
        if column.startswith("gate_") and column.endswith("_mean"):
            row[column] = station_df[column].mean()
    return pd.DataFrame([row])


def run_variant_pollutant(
    config: Config,
    ablation: DirectAblation,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
    audit: pd.DataFrame,
    total_stations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = prepare_variant_root(
        config,
        ablation,
        pollutant,
        resolved_groups,
        effective_groups,
        effective_width,
        group_source,
        audit,
    )
    stop = total_stations
    if config.MAX_STATIONS > 0:
        stop = min(total_stations, config.FOLD_START + config.MAX_STATIONS)
    if not 0 <= config.FOLD_START < stop:
        raise ValueError(
            f"Invalid fold range: start={config.FOLD_START}, stop={stop}"
        )

    print(
        f"\n{ablation.code} | {pollutant.upper()} strict LOSO folds "
        f"{config.FOLD_START}..{stop - 1}"
    )
    for station_idx in range(config.FOLD_START, stop):
        fingerprint = experiment_fingerprint(
            config,
            ablation,
            pollutant,
            station_idx,
            resolved_groups,
            effective_groups,
            effective_width,
            group_source,
        )
        reused = reuse_completed_fold(
            config,
            root,
            ablation,
            pollutant,
            station_idx,
            fingerprint,
        )
        if reused is not None:
            print(f"  reuse completed fold station {station_idx:02d}")
            continue

        checkpoint = (
            root
            / "models"
            / f"{pollutant}_station_{station_idx:02d}_{ablation.code}.pth"
        )
        model, training_info = train_fold_model(
            config,
            pollutant,
            station_idx,
            ablation,
            resolved_groups,
            effective_groups,
            effective_width,
            group_source,
            checkpoint,
        )
        summary, samples = evaluate_fold(
            config, pollutant, station_idx, ablation, model
        )
        summary.update(training_info)
        save_completed_fold(
            config,
            root,
            ablation,
            pollutant,
            station_idx,
            fingerprint,
            summary,
            samples,
        )
        print(
            f"  station {station_idx:02d}: MAE={summary['MAE']:.4f}, "
            f"RMSE={summary['RMSE']:.4f}, R2={summary['R2']:.4f}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rebuild from every compatible fold on disk.  This makes FOLD_START and
    # MAX_STATIONS safe for resume/chunked runs without dropping earlier folds
    # from the aggregate CSVs.
    station_summaries = []
    sample_frames = []
    for station_idx in range(total_stations):
        fingerprint = experiment_fingerprint(
            config,
            ablation,
            pollutant,
            station_idx,
            resolved_groups,
            effective_groups,
            effective_width,
            group_source,
        )
        completed = load_completed_fold(
            root,
            ablation,
            pollutant,
            station_idx,
            fingerprint,
        )
        if completed is not None:
            summary, samples = completed
            station_summaries.append(summary)
            sample_frames.append(samples)
    if not station_summaries:
        raise RuntimeError(
            f"No compatible completed folds for {ablation.code} {pollutant}"
        )
    if len(station_summaries) < total_stations:
        print(
            f"  [partial] aggregate uses {len(station_summaries)}/"
            f"{total_stations} completed stations"
        )

    station_df = pd.DataFrame(station_summaries)
    sample_df = pd.concat(sample_frames, ignore_index=True)
    aggregate_df = aggregate_results(
        config, ablation, pollutant, station_df, sample_df
    )
    station_df.to_csv(
        root / "metrics" / f"{pollutant}_station_metrics.csv", index=False
    )
    sample_df.to_csv(
        root / "predictions" / f"{pollutant}_test_predictions.csv", index=False
    )
    aggregate_df.to_csv(
        root / "metrics" / f"{pollutant}_aggregate_metrics.csv", index=False
    )
    efficiency_columns = [
        column
        for column in (
            "variant",
            "pollutant",
            "station",
            "train_seconds",
            "best_epoch",
            "best_validation_loss",
            "parameter_count",
            "encoder_parameter_count",
            "encoder_parameter_target_F3",
            "condition_channels",
            "experiment_fingerprint",
        )
        if column in station_df.columns
    ]
    station_df[efficiency_columns].to_csv(
        root / "metrics" / f"{pollutant}_efficiency.csv", index=False
    )
    print(
        aggregate_df[
            ["variant", "pollutant", "pooled_MAE", "pooled_RMSE", "pooled_R2"]
        ].to_string(index=False)
    )
    return aggregate_df, station_df


def validate_reference_protocol(
    config: Config,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
) -> None:
    path = Path(config.REFERENCE_ROOT, "F3", f"protocol_{pollutant}.json")
    if not path.exists():
        message = f"F3 reference protocol is missing: {path}"
        if config.ALLOW_REFERENCE_MISMATCH:
            print(f"[warning] {message}")
            return
        raise FileNotFoundError(message)
    protocol = load_json(path)
    expected_signature = fusion.group_signature(resolved_groups)
    observed_variant = protocol.get("variant", {})
    checks = {
        "model_version": (
            protocol.get("model_version"),
            fusion.Config.MODEL_VERSION,
        ),
        "seed": (protocol.get("seed"), GLOBAL_SEED),
        "group_signature": (
            protocol.get("group_signature"),
            expected_signature,
        ),
        "group_dropout": (
            observed_variant.get("group_dropout"),
            config.GROUP_DROPOUT,
        ),
        "group_width": (protocol.get("group_width"), config.GROUP_WIDTH),
        "gate_initial_bias": (
            protocol.get("gate_initial_bias"),
            config.GATE_BIAS,
        ),
        "importance_beta": (protocol.get("importance_beta"), 0.0),
        "curriculum_masking": (
            protocol.get("curriculum_masking"),
            True,
        ),
        "tv_weights": (protocol.get("tv_weights"), config.TV_WEIGHTS),
        "laplacian_weights": (
            protocol.get("laplacian_weights"),
            config.LAPLACIAN_WEIGHTS,
        ),
        "bottleneck_fusion": (
            observed_variant.get("bottleneck_fusion"),
            True,
        ),
        "decoder_fusion": (
            observed_variant.get("decoder_fusion"),
            True,
        ),
        "scb_channel_attention": (
            protocol.get("scb_channel_attention"),
            True,
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        message = (
            f"F3 reference protocol mismatch for {pollutant}: "
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )
        if config.ALLOW_REFERENCE_MISMATCH:
            print(f"[warning] {message}")
        else:
            raise RuntimeError(message)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_bootstrap_ci(
    differences: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(repeats, len(differences))
    )
    estimates = differences[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def paired_rank_biserial(oriented_differences: np.ndarray) -> float:
    """Paired rank-biserial effect; positive values favor F3."""

    values = oriented_differences[np.isfinite(oriented_differences)]
    values = values[~np.isclose(values, 0.0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    denominator = float(ranks.sum())
    if denominator <= 0:
        return float("nan")
    return float(
        (ranks[values > 0].sum() - ranks[values < 0].sum())
        / denominator
    )


def read_available_results(
    config: Config,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    aggregates: list[pd.DataFrame] = []
    stations: list[pd.DataFrame] = []

    for code in ("F0", "F1", "F2", "F3"):
        for pollutant in config.POLLUTANTS:
            aggregate_path = (
                Path(config.REFERENCE_ROOT)
                / code
                / "metrics"
                / f"{pollutant}_aggregate_metrics.csv"
            )
            station_path = (
                Path(config.REFERENCE_ROOT)
                / code
                / "metrics"
                / f"{pollutant}_station_metrics.csv"
            )
            if aggregate_path.exists():
                frame = pd.read_csv(aggregate_path)
                frame["result_source"] = "existing_F0_F3"
                aggregates.append(frame)
            if station_path.exists():
                frame = pd.read_csv(station_path)
                frame["result_source"] = "existing_F0_F3"
                stations.append(frame)

    for code in ABLATIONS:
        for pollutant in config.POLLUTANTS:
            aggregate_path = (
                Path(config.ABLATION_ROOT)
                / code
                / "metrics"
                / f"{pollutant}_aggregate_metrics.csv"
            )
            station_path = (
                Path(config.ABLATION_ROOT)
                / code
                / "metrics"
                / f"{pollutant}_station_metrics.csv"
            )
            if aggregate_path.exists():
                frame = pd.read_csv(aggregate_path)
                frame["result_source"] = "F3_direct_ablation"
                aggregates.append(frame)
            if station_path.exists():
                frame = pd.read_csv(station_path)
                frame["result_source"] = "F3_direct_ablation"
                stations.append(frame)
    return aggregates, stations


def build_paired_tests(
    config: Config, station_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    if not station_frames:
        return pd.DataFrame()
    stations = pd.concat(station_frames, ignore_index=True)
    stations = stations.drop_duplicates(
        subset=["variant", "pollutant", "station"], keep="last"
    )
    rows = []
    for pollutant in sorted(stations["pollutant"].unique()):
        pollutant_df = stations[stations["pollutant"] == pollutant]
        reference = pollutant_df[pollutant_df["variant"] == "F3"]
        if reference.empty:
            continue
        for candidate_code in sorted(
            set(pollutant_df["variant"].astype(str)) - {"F3"}
        ):
            candidate = pollutant_df[pollutant_df["variant"] == candidate_code]
            for metric in ("MAE", "RMSE", "R2"):
                paired = reference[["station", metric]].merge(
                    candidate[["station", metric]],
                    on="station",
                    suffixes=("_F3", "_candidate"),
                ).dropna()
                if len(paired) < 2:
                    continue
                delta = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_F3"]
                ).to_numpy(dtype=np.float64)
                if np.allclose(delta, 0.0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            wilcoxon(
                                delta,
                                alternative="two-sided",
                                zero_method="wilcox",
                            ).pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                ci_low, ci_high = paired_bootstrap_ci(
                    delta,
                    config.PAIR_BOOTSTRAP_REPEATS,
                    fold_seed(pollutant, sum(ord(char) for char in candidate_code + metric)),
                )
                if metric in {"MAE", "RMSE"}:
                    full_better_rate = float(np.mean(delta > 0))
                    oriented_delta = delta
                    relative_change = float(
                        100.0
                        * paired[f"{metric}_candidate"].mean()
                        / max(abs(float(paired[f"{metric}_F3"].mean())), 1e-12)
                        - 100.0
                    )
                else:
                    full_better_rate = float(np.mean(delta < 0))
                    oriented_delta = -delta
                    relative_change = float("nan")
                rows.append(
                    {
                        "pollutant": pollutant,
                        "reference": "F3",
                        "candidate": candidate_code,
                        "family": (
                            "existing_fusion_progression"
                            if candidate_code in {"F0", "F1", "F2"}
                            else "F3_direct_ablation"
                        ),
                        "metric": metric,
                        "n_stations": len(paired),
                        "mean_F3": float(paired[f"{metric}_F3"].mean()),
                        "mean_candidate": float(
                            paired[f"{metric}_candidate"].mean()
                        ),
                        "mean_candidate_minus_F3": float(np.mean(delta)),
                        "mean_delta_CI_low": ci_low,
                        "mean_delta_CI_high": ci_high,
                        "median_candidate_minus_F3": float(np.median(delta)),
                        "relative_candidate_change_vs_F3_percent": relative_change,
                        "F3_better_station_rate": full_better_rate,
                        "rank_biserial_F3_advantage": paired_rank_biserial(
                            oriented_delta
                        ),
                        "wilcoxon_p": p_value,
                    }
                )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests

    tests["holm_global_p"] = holm_adjust(tests["wilcoxon_p"].tolist())
    tests["holm_within_pollutant_metric_p"] = np.nan
    grouped_indices = tests.groupby(["pollutant", "metric"]).groups
    for indices in grouped_indices.values():
        index_list = list(indices)
        adjusted = holm_adjust(
            [float(tests.loc[index, "wilcoxon_p"]) for index in index_list]
        )
        for index, value in zip(index_list, adjusted):
            tests.loc[index, "holm_within_pollutant_metric_p"] = value
    tests["holm_within_test_family_p"] = np.nan
    family_indices = tests.groupby(
        ["pollutant", "metric", "family"]
    ).groups
    for indices in family_indices.values():
        index_list = list(indices)
        adjusted = holm_adjust(
            [float(tests.loc[index, "wilcoxon_p"]) for index in index_list]
        )
        for index, value in zip(index_list, adjusted):
            tests.loc[index, "holm_within_test_family_p"] = value
    tests["significant_global_0_05"] = tests["holm_global_p"] < 0.05
    tests["significant_within_pollutant_metric_0_05"] = (
        tests["holm_within_pollutant_metric_p"] < 0.05
    )
    tests["significant_within_test_family_0_05"] = (
        tests["holm_within_test_family_p"] < 0.05
    )
    return tests


def write_cross_variant_outputs(config: Config) -> None:
    root = Path(config.ABLATION_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    aggregate_frames, station_frames = read_available_results(config)
    if not aggregate_frames:
        print("[warning] no aggregate result files were found")
        return
    aggregate = pd.concat(aggregate_frames, ignore_index=True)
    aggregate = aggregate.drop_duplicates(
        subset=["variant", "pollutant"], keep="last"
    )
    order = {code: index for index, code in enumerate(("F0", "F1", "F2", "F3", *ABLATIONS))}
    aggregate["_variant_order"] = aggregate["variant"].map(order).fillna(999)
    aggregate = aggregate.sort_values(
        ["pollutant", "_variant_order", "variant"]
    ).drop(columns="_variant_order")
    aggregate.to_csv(
        root / "all_f3_centered_ablation_aggregate_metrics.csv", index=False
    )
    for pollutant, frame in aggregate.groupby("pollutant", sort=True):
        frame.to_csv(
            root / f"{pollutant}_f3_centered_ablation_comparison.csv", index=False
        )

    expected_by_pollutant = (
        aggregate[aggregate["variant"] == "F3"]
        .set_index("pollutant")["n_stations"]
        .to_dict()
    )
    completion = aggregate[["variant", "pollutant", "n_stations"]].copy()
    completion["expected_stations"] = completion.apply(
        lambda row: expected_by_pollutant.get(
            row["pollutant"],
            aggregate.loc[
                aggregate["pollutant"] == row["pollutant"], "n_stations"
            ].max(),
        ),
        axis=1,
    )
    completion["complete"] = (
        completion["n_stations"] == completion["expected_stations"]
    )
    completion.to_csv(root / "f3_ablation_completion_status.csv", index=False)

    paper_rows = aggregate[
        aggregate["variant"].isin({"F3", *ABLATIONS})
    ].copy()
    reference_columns = [
        "pollutant",
        "pooled_MAE",
        "pooled_RMSE",
        "pooled_R2",
    ]
    reference = aggregate[aggregate["variant"] == "F3"][
        reference_columns
    ].rename(
        columns={
            "pooled_MAE": "F3_pooled_MAE",
            "pooled_RMSE": "F3_pooled_RMSE",
            "pooled_R2": "F3_pooled_R2",
        }
    )
    if not reference.empty and not paper_rows.empty:
        paper_rows = paper_rows.merge(reference, on="pollutant", how="inner")
        for metric in ("MAE", "RMSE", "R2"):
            paper_rows[f"delta_pooled_{metric}_vs_F3"] = (
                paper_rows[f"pooled_{metric}"]
                - paper_rows[f"F3_pooled_{metric}"]
            )
        for metric in ("MAE", "RMSE"):
            denominator = paper_rows[f"F3_pooled_{metric}"].abs().clip(
                lower=1e-12
            )
            paper_rows[f"relative_pooled_{metric}_change_vs_F3_percent"] = (
                100.0
                * paper_rows[f"delta_pooled_{metric}_vs_F3"]
                / denominator
            )
        paper_rows.to_csv(
            root / "paper_table_f3_direct_ablation.csv", index=False
        )

    tests = build_paired_tests(config, station_frames)
    if not tests.empty:
        tests.to_csv(root / "paired_tests_vs_F3.csv", index=False)


def write_run_manifest(config: Config, selected: list[DirectAblation]) -> None:
    root = Path(config.ABLATION_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": config.MODEL_VERSION,
        "source_sha256": SOURCE_SHA256,
        "seed": GLOBAL_SEED,
        "available_ablations": {
            code: asdict(item) for code, item in ABLATIONS.items()
        },
        "selected_ablations": [asdict(item) for item in selected],
        "pollutants": config.POLLUTANTS,
        "fold_start": config.FOLD_START,
        "max_stations": config.MAX_STATIONS,
        "reference_root": str(Path(config.REFERENCE_ROOT).resolve()),
        "result_root": str(root.resolve()),
        "summary_only": config.SUMMARY_ONLY,
        "reuse_completed": config.REUSE_CHECKPOINTS,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    config = Config()
    selected = selected_ablations(config)
    write_run_manifest(config, selected)
    print(
        f"device={DEVICE} | seed={GLOBAL_SEED} | "
        f"result_root={config.ABLATION_ROOT}"
    )
    print(
        "selected direct ablations="
        + (",".join(item.code for item in selected) if selected else "none")
    )
    print(f"existing F0--F3 root={config.REFERENCE_ROOT}")

    if config.SUMMARY_ONLY:
        write_cross_variant_outputs(config)
        return

    for pollutant in config.POLLUTANTS:
        probe = core.SpatialInpaintDatasetLOOCV(
            config, pollutant, mask_station_idx=0, mode="test"
        )
        resolved_groups, group_source, audit = fusion.resolve_groups(
            config, probe, pollutant
        )
        total_stations = probe.total_stations
        feature_dim = probe.F_dim
        print(
            f"\n{pollutant.upper()} resolved groups: "
            + ", ".join(
                f"{group.name}[{len(group.indices)}]" for group in resolved_groups
            )
        )
        validate_reference_protocol(config, pollutant, resolved_groups)
        del probe

        for ablation in selected:
            effective_groups, effective_width = effective_groups_and_width(
                config, ablation, resolved_groups, feature_dim
            )
            run_variant_pollutant(
                config,
                ablation,
                pollutant,
                resolved_groups,
                effective_groups,
                effective_width,
                group_source,
                audit,
                total_stations,
            )

    write_cross_variant_outputs(config)


if __name__ == "__main__":
    main()
