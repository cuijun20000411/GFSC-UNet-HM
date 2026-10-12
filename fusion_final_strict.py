"""Strict F0--F4 multi-source fusion experiments for SC-UNet.

This runner extends ``rh4_final_strict_light.py`` without changing its strict
outer-LOSO data protocol, fold-only concentration scaling, curriculum masking,
TV penalty, Laplacian penalty, hard preservation of observed stations, or raw
test-space metrics.

Variants
--------
F0
    Exact A1-style strong baseline: seven observation-derived channels only.
F1
    F0 plus grouped covariate encoders and a zero-initialized gated residual
    adapter at the 4x4 bottleneck.
F2
    F1 plus gated residual adapters at the final two decoder resolutions
    (16x16 and 32x32).
F3
    F2 plus sample-wise whole-group covariate dropout during training.
F4
    F3 with one covariate group removed at a time.  Bare ``F4`` expands to one
    run per resolved group; ``F4_<group>`` selects a single removal.

The original ASF input concatenation is intentionally replaced by residual
conditioning.  The SCB convolution/residual block and its channel attention
are retained by default across every F variant, so this experiment isolates
fusion rather than re-testing A5.  Set ``FUSION_USE_SCB_ATTENTION=0`` only for
a separate follow-up sensitivity analysis.

Feature groups
--------------
The original ``*_features.npy`` and ``cov_img_full.npy`` files do not contain
feature names.  Therefore the runner never guesses semantic labels such as
"meteorology" or "traffic".

* ``FUSION_GROUP_MODE=auto`` (default) forms reproducible structural groups
  from training-time covariate variation: ``dynamic_spatial``,
  ``static_spatial``, ``global_temporal`` and, if present, ``constant``.
* ``FUSION_GROUP_MODE=json`` reads the semantic groups supplied by
  ``FUSION_GROUPS_JSON``.  This is the required mode for source-wise F4 claims
  such as meteorology/land-use/traffic removal.

JSON example::

    {
      "no2": {
        "meteorology": {"indices": [0, 1, 2], "kind": "spatial"},
        "traffic": {"indices": [3, 4], "kind": "spatial"},
        "land_use": {"indices": [5, 6, 7], "kind": "spatial"},
        "temporal": {"indices": [37, 38], "kind": "global"}
      },
      "o3": {},
      "pm25": {}
    }

Every channel index must be assigned exactly once.  Indices are zero-based and
refer to the last axis of the original ``*_features.npy`` arrays, equivalently
the channel axis of ``cov_img_full.npy``.

Examples
--------
One-fold structural-group smoke test::

    SCUNET_POLLUTANTS=no2 FUSION_VARIANTS=F0,F1 \
      FUSION_MAX_STATIONS=1 python -u fusion_final_strict.py

Full F0--F3 experiment::

    FUSION_VARIANTS=F0,F1,F2,F3 python -u fusion_final_strict.py

Semantic F4 removals::

    FUSION_GROUP_MODE=json \
      FUSION_GROUPS_JSON=/path/to/feature_groups.json \
      FUSION_VARIANTS=F4 \
      FUSION_F4_GROUPS=meteorology,land_use,traffic \
      python -u fusion_final_strict.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast

try:
    import rh4_final_strict_light as core
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Place fusion_final_strict.py in the same directory as "
        "rh4_final_strict_light.py before running it."
    ) from exc


GLOBAL_SEED = int(os.environ.get("FUSION_SEED", "42"))
DEVICE = core.DEVICE
USE_AMP = core.USE_AMP
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


def fold_seed(pollutant: str, station_idx: int) -> int:
    """Match the deterministic fold pairing used by the A0--A8 suite."""

    digest = hashlib.sha256(
        f"{GLOBAL_SEED}|{pollutant}|{station_idx}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((GLOBAL_SEED + offset) % (2**31 - 1))


seed_everything(GLOBAL_SEED)


class Config(core.Config):
    DATA_ROOT_DIR = os.environ.get(
        "SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"
    )
    FUSION_ROOT = os.environ.get(
        "FUSION_RESULT_ROOT", "/etc/cj/boundary/fusion_final_strict_v1"
    )
    POLLUTANTS = [
        item.strip().lower()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    SELECTED_VARIANTS = [
        item.strip()
        for item in os.environ.get("FUSION_VARIANTS", "F0").split(",")
        if item.strip()
    ]
    FOLD_START = int(os.environ.get("FUSION_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("FUSION_MAX_STATIONS", "0"))
    REUSE_CHECKPOINTS = os.environ.get("FUSION_REUSE_COMPLETED", "1") == "1"
    NUM_WORKERS = int(
        os.environ.get("FUSION_NUM_WORKERS", str(core.Config.NUM_WORKERS))
    )
    LOG_INTERVAL = int(
        os.environ.get("FUSION_LOG_INTERVAL", str(core.Config.LOG_INTERVAL))
    )
    EPOCHS = int(os.environ.get("FUSION_EPOCHS", str(core.Config.EPOCHS)))
    BATCH_SIZE = int(
        os.environ.get("FUSION_BATCH_SIZE", str(core.Config.BATCH_SIZE))
    )
    PATIENCE = int(os.environ.get("FUSION_PATIENCE", str(core.Config.PATIENCE)))

    GROUP_MODE = os.environ.get("FUSION_GROUP_MODE", "auto").strip().lower()
    GROUPS_JSON = os.environ.get("FUSION_GROUPS_JSON", "").strip()
    ALLOW_UNGROUPED = os.environ.get("FUSION_ALLOW_UNGROUPED", "0") == "1"
    F4_GROUPS = [
        item.strip()
        for item in os.environ.get("FUSION_F4_GROUPS", "").split(",")
        if item.strip()
    ]
    AUTO_GROUP_MAX_TIMES = int(os.environ.get("FUSION_AUTO_GROUP_TIMES", "64"))
    AUTO_VARIATION_TOL = float(os.environ.get("FUSION_AUTO_VARIATION_TOL", "1e-6"))

    GROUP_WIDTH = int(os.environ.get("FUSION_GROUP_WIDTH", "8"))
    GROUP_DROPOUT = float(os.environ.get("FUSION_GROUP_DROPOUT", "0.25"))
    GATE_BIAS = float(os.environ.get("FUSION_GATE_BIAS", "-3.0"))
    USE_SCB_ATTENTION = os.environ.get("FUSION_USE_SCB_ATTENTION", "1") == "1"
    BOOTSTRAP_REPEATS = int(os.environ.get("FUSION_BOOTSTRAP_REPEATS", "2000"))
    MODEL_VERSION = "strict_grouped_gated_fusion_v1"


@dataclass(frozen=True)
class GroupSpec:
    name: str
    indices: tuple[int, ...]
    kind: str = "spatial"


@dataclass(frozen=True)
class FusionVariant:
    code: str
    description: str
    use_covariates: bool
    bottleneck_fusion: bool
    decoder_fusion: bool
    group_dropout: float
    ablate_group: str | None = None


BASE_VARIANTS = {
    "F0": FusionVariant(
        "F0",
        "A1 strong baseline: seven observation-derived channels",
        use_covariates=False,
        bottleneck_fusion=False,
        decoder_fusion=False,
        group_dropout=0.0,
    ),
    "F1": FusionVariant(
        "F1",
        "F0 plus grouped bottleneck gated residual fusion",
        use_covariates=True,
        bottleneck_fusion=True,
        decoder_fusion=False,
        group_dropout=0.0,
    ),
    "F2": FusionVariant(
        "F2",
        "F1 plus gated residual fusion at the final two decoder scales",
        use_covariates=True,
        bottleneck_fusion=True,
        decoder_fusion=True,
        group_dropout=0.0,
    ),
    "F3": FusionVariant(
        "F3",
        "F2 plus whole-group covariate dropout",
        use_covariates=True,
        bottleneck_fusion=True,
        decoder_fusion=True,
        group_dropout=Config.GROUP_DROPOUT,
    ),
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Group name cannot be converted to a safe id: {value!r}")
    return slug


def group_signature(groups: Iterable[GroupSpec]) -> str:
    payload = [asdict(group) for group in groups]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_groups(
    groups: list[GroupSpec], feature_dim: int, allow_ungrouped: bool
) -> list[GroupSpec]:
    if not groups:
        raise ValueError("At least one non-empty covariate group is required")
    seen: dict[int, str] = {}
    clean: list[GroupSpec] = []
    used_names: set[str] = set()
    for group in groups:
        if group.kind not in {"spatial", "global"}:
            raise ValueError(
                f"Group {group.name!r} kind must be 'spatial' or 'global'"
            )
        if not group.indices:
            raise ValueError(f"Group {group.name!r} has no channel indices")
        safe_name = slugify(group.name)
        if safe_name in used_names:
            raise ValueError(f"Duplicate normalized group name: {safe_name}")
        used_names.add(safe_name)
        indices = tuple(sorted({int(index) for index in group.indices}))
        for index in indices:
            if not 0 <= index < feature_dim:
                raise IndexError(
                    f"Group {group.name!r} index {index} is outside 0..{feature_dim - 1}"
                )
            if index in seen:
                raise ValueError(
                    f"Covariate channel {index} appears in both {seen[index]!r} "
                    f"and {group.name!r}"
                )
            seen[index] = group.name
        clean.append(GroupSpec(group.name, indices, group.kind))

    missing = sorted(set(range(feature_dim)) - set(seen))
    if missing and not allow_ungrouped:
        raise ValueError(
            "Feature-group configuration is incomplete; unassigned zero-based "
            f"channels: {missing}. Set FUSION_ALLOW_UNGROUPED=1 only if these "
            "channels should form an explicit 'ungrouped' control group."
        )
    if missing:
        clean.append(GroupSpec("ungrouped", tuple(missing), "spatial"))
    return clean


def channel_audit(
    dataset: core.SpatialInpaintDatasetLOOCV,
    max_times: int,
    variation_tol: float,
) -> tuple[pd.DataFrame, list[GroupSpec]]:
    """Classify channels by observed training-time/spatial variation only."""

    train_ids = np.asarray(dataset.train_time_indices, dtype=np.int64)
    if train_ids.size == 0:
        raise ValueError("Cannot audit covariates without training time indices")
    count = min(max(2, max_times), len(train_ids))
    positions = np.linspace(0, len(train_ids) - 1, count, dtype=np.int64)
    selected = train_ids[np.unique(positions)]
    values = np.asarray(dataset.cov_img_full[selected], dtype=np.float32)
    valid = dataset.valid_grid_mask.astype(bool).reshape(-1)
    values = values.reshape(values.shape[0], values.shape[1], -1)[:, :, valid]

    temporal_std = np.nanstd(values, axis=0).mean(axis=1)
    spatial_std = np.nanstd(values, axis=2).mean(axis=0)
    total_std = np.nanstd(values, axis=(0, 2))
    finite_fraction = np.isfinite(values).mean(axis=(0, 2))

    labels = []
    grouped: dict[str, list[int]] = {
        "dynamic_spatial": [],
        "static_spatial": [],
        "global_temporal": [],
        "constant": [],
    }
    for index, (time_sd, space_sd) in enumerate(zip(temporal_std, spatial_std)):
        varies_time = bool(time_sd > variation_tol)
        varies_space = bool(space_sd > variation_tol)
        if varies_time and varies_space:
            label = "dynamic_spatial"
        elif varies_space:
            label = "static_spatial"
        elif varies_time:
            label = "global_temporal"
        else:
            label = "constant"
        labels.append(label)
        grouped[label].append(index)

    audit = pd.DataFrame(
        {
            "channel_index": np.arange(dataset.F_dim, dtype=np.int64),
            "temporal_std": temporal_std,
            "spatial_std": spatial_std,
            "total_std": total_std,
            "finite_fraction": finite_fraction,
            "auto_group": labels,
        }
    )
    groups = [
        GroupSpec(
            name,
            tuple(indices),
            "global" if name in {"global_temporal", "constant"} else "spatial",
        )
        for name, indices in grouped.items()
        if indices
    ]
    return audit, groups


def load_json_groups(path: Path, pollutant: str) -> list[GroupSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Feature-group JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if pollutant not in payload or not isinstance(payload[pollutant], dict):
        raise KeyError(f"JSON does not define an object for pollutant {pollutant!r}")

    groups = []
    for name, value in payload[pollutant].items():
        if str(name).startswith("_"):
            continue
        if isinstance(value, list):
            indices = value
            kind = "spatial"
        elif isinstance(value, dict):
            indices = value.get("indices", [])
            kind = str(value.get("kind", "spatial")).lower()
        else:
            raise TypeError(
                f"Group {name!r} must be a list or an object with indices/kind"
            )
        groups.append(GroupSpec(str(name), tuple(int(x) for x in indices), kind))
    return groups


def resolve_groups(
    config: Config,
    dataset: core.SpatialInpaintDatasetLOOCV,
    pollutant: str,
) -> tuple[list[GroupSpec], str, pd.DataFrame]:
    audit, auto_groups = channel_audit(
        dataset, config.AUTO_GROUP_MAX_TIMES, config.AUTO_VARIATION_TOL
    )
    if config.GROUP_MODE == "auto":
        groups = auto_groups
        source = "auto structural grouping from training-time variation"
    elif config.GROUP_MODE == "json":
        if not config.GROUPS_JSON:
            raise ValueError(
                "FUSION_GROUP_MODE=json requires FUSION_GROUPS_JSON=/path/file.json"
            )
        groups = load_json_groups(Path(config.GROUPS_JSON), pollutant)
        source = f"semantic JSON: {Path(config.GROUPS_JSON).resolve()}"
    else:
        raise ValueError("FUSION_GROUP_MODE must be 'auto' or 'json'")
    groups = validate_groups(groups, dataset.F_dim, config.ALLOW_UNGROUPED)
    return groups, source, audit


def find_group(requested: str, groups: list[GroupSpec]) -> str:
    target = slugify(requested)
    matches = [group.name for group in groups if slugify(group.name) == target]
    if len(matches) != 1:
        available = [group.name for group in groups]
        raise KeyError(f"Unknown F4 group {requested!r}; available groups: {available}")
    return matches[0]


def variants_for_groups(config: Config, groups: list[GroupSpec]) -> list[FusionVariant]:
    resolved: list[FusionVariant] = []
    for token_raw in config.SELECTED_VARIANTS:
        token = token_raw.strip()
        upper = token.upper()
        if upper in BASE_VARIANTS:
            variant = BASE_VARIANTS[upper]
            if upper == "F3":
                variant = FusionVariant(
                    **{**asdict(variant), "group_dropout": config.GROUP_DROPOUT}
                )
            resolved.append(variant)
            continue
        if upper == "F4":
            requested_groups = config.F4_GROUPS or [group.name for group in groups]
        elif upper.startswith("F4_") or upper.startswith("F4-"):
            requested_groups = [token[3:]]
        else:
            raise KeyError(
                f"Unknown fusion variant {token!r}; use F0,F1,F2,F3,F4 or F4_<group>"
            )

        for requested in requested_groups:
            group_name = find_group(requested, groups)
            code = f"F4_{slugify(group_name)}"
            resolved.append(
                FusionVariant(
                    code=code,
                    description=f"F3 with covariate group {group_name!r} removed",
                    use_covariates=True,
                    bottleneck_fusion=True,
                    decoder_fusion=True,
                    group_dropout=config.GROUP_DROPOUT,
                    ablate_group=group_name,
                )
            )

    unique: dict[str, FusionVariant] = {}
    for variant in resolved:
        unique.setdefault(variant.code, variant)
    return list(unique.values())


class FusionDataset(core.SpatialInpaintDatasetLOOCV):
    """Return only seven base channels for F0; retain covariates otherwise."""

    def __init__(self, *args, variant: FusionVariant, **kwargs):
        self.fusion_variant = variant
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        if not self.fusion_variant.use_covariates:
            sample["input"] = sample["input"][: self.base_channels]
        return sample


def norm_groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ChannelAttention(nn.Module):
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
    """The original SCB; only its channel-attention submodule is optional."""

    def __init__(
        self, in_channels: int, out_channels: int, use_channel_attention: bool
    ):
        super().__init__()
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
        self.norm1 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.norm2 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.attention = (
            ChannelAttention(out_channels, reduction=8)
            if use_channel_attention
            else nn.Identity()
        )
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


class CovariateGroupEncoder(nn.Module):
    def __init__(self, spec: GroupSpec, width: int):
        super().__init__()
        self.spec = spec
        self.register_buffer(
            "channel_indices", torch.tensor(spec.indices, dtype=torch.long)
        )
        hidden = max(width, min(32, 2 * width))
        kernel_size = 1 if spec.kind == "global" else 3
        padding = kernel_size // 2
        second_conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": padding,
        }
        if kernel_size == 3:
            second_conv_kwargs["padding_mode"] = "replicate"
        self.net = nn.Sequential(
            nn.Conv2d(len(spec.indices), hidden, kernel_size=1),
            nn.GroupNorm(norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, width, **second_conv_kwargs),
            nn.GroupNorm(norm_groups(width), width),
            nn.SiLU(inplace=True),
        )

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        selected = torch.index_select(covariates, 1, self.channel_indices)
        return self.net(selected)


class GroupedCovariateEncoder(nn.Module):
    def __init__(
        self,
        groups: list[GroupSpec],
        width: int,
        dropout_probability: float,
    ):
        super().__init__()
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Group dropout must be in [0, 1)")
        self.groups = tuple(groups)
        self.width = width
        self.dropout_probability = float(dropout_probability)
        self.encoders = nn.ModuleList(
            [CovariateGroupEncoder(group, width) for group in groups]
        )
        self.out_channels = width * len(groups)

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        encoded = []
        for encoder in self.encoders:
            features = encoder(covariates)
            if self.training and self.dropout_probability > 0:
                keep = (
                    torch.rand(
                        features.shape[0], 1, 1, 1, device=features.device
                    )
                    >= self.dropout_probability
                ).to(features.dtype)
                features = features * keep / (1.0 - self.dropout_probability)
            encoded.append(features)
        return torch.cat(encoded, dim=1)


class GatedResidualAdapter(nn.Module):
    """Zero-initialized covariate correction controlled by mask/distance gates."""

    def __init__(self, main_channels: int, cov_channels: int, gate_bias: float):
        super().__init__()
        hidden = max(8, min(64, main_channels // 2))
        self.condition_projection = nn.Sequential(
            nn.Conv2d(cov_channels, main_channels, kernel_size=1),
            nn.GroupNorm(norm_groups(main_channels), main_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(cov_channels, hidden, kernel_size=1),
            nn.GroupNorm(norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, main_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * main_channels + 2, hidden, kernel_size=1),
            nn.GroupNorm(norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, main_channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        main: torch.Tensor,
        covariates: torch.Tensor,
        observed_mask: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        size = main.shape[-2:]
        if covariates.shape[-2:] != size:
            covariates = F.adaptive_avg_pool2d(covariates, size)
        observed = F.interpolate(observed_mask, size=size, mode="nearest")
        distance_scaled = F.interpolate(
            distance, size=size, mode="bilinear", align_corners=False
        )
        condition = self.condition_projection(covariates)
        gate = torch.sigmoid(
            self.gate(torch.cat([main, condition, observed, distance_scaled], dim=1))
        )
        self.last_gate = gate.detach()
        return main + gate * self.residual(covariates)


class FusionSCUNet(nn.Module):
    """A1 backbone with optional grouped, gated residual conditioning."""

    def __init__(
        self,
        base_channels: int,
        cov_channels: int,
        groups: list[GroupSpec],
        variant: FusionVariant,
        config: Config,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.cov_channels = cov_channels
        self.variant = variant

        # Define the backbone first and identically for every variant.  Together
        # with equal fold seeds, this preserves paired backbone initialization.
        attention = config.USE_SCB_ATTENTION
        self.enc1 = SpatialConstraintBlock(base_channels, 32, attention)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(32, 64, attention)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(64, 128, attention)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2), SpatialConstraintBlock(128, 256, attention)
        )
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = SpatialConstraintBlock(256, 128, attention)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = SpatialConstraintBlock(128, 64, attention)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = SpatialConstraintBlock(64, 32, attention)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

        self.covariate_encoder: GroupedCovariateEncoder | None = None
        self.bottleneck_adapter: GatedResidualAdapter | None = None
        self.decoder2_adapter: GatedResidualAdapter | None = None
        self.decoder1_adapter: GatedResidualAdapter | None = None
        if variant.use_covariates:
            active_groups = [
                group for group in groups if group.name != variant.ablate_group
            ]
            if not active_groups:
                raise ValueError(
                    f"Variant {variant.code} removes every available covariate group"
                )
            self.covariate_encoder = GroupedCovariateEncoder(
                active_groups, config.GROUP_WIDTH, variant.group_dropout
            )
            condition_channels = self.covariate_encoder.out_channels
            if variant.bottleneck_fusion:
                self.bottleneck_adapter = GatedResidualAdapter(
                    256, condition_channels, config.GATE_BIAS
                )
            if variant.decoder_fusion:
                self.decoder2_adapter = GatedResidualAdapter(
                    64, condition_channels, config.GATE_BIAS
                )
                self.decoder1_adapter = GatedResidualAdapter(
                    32, condition_channels, config.GATE_BIAS
                )

    def forward(
        self,
        x: torch.Tensor,
        observed_mask: torch.Tensor,
        known_values: torch.Tensor,
    ) -> torch.Tensor:
        base = x[:, : self.base_channels]
        distance = base[:, 6:7]
        cov_encoded = None
        if self.covariate_encoder is not None:
            covariates = x[:, self.base_channels :]
            if covariates.shape[1] != self.cov_channels:
                raise ValueError(
                    f"Expected {self.cov_channels} covariate channels, "
                    f"received {covariates.shape[1]}"
                )
            cov_encoded = self.covariate_encoder(covariates)

        enc1 = self.enc1(base)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        if self.bottleneck_adapter is not None and cov_encoded is not None:
            enc4 = self.bottleneck_adapter(
                enc4, cov_encoded, observed_mask, distance
            )

        dec3 = self.dec3(torch.cat([self.up4(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up3(dec3), enc2], dim=1))
        if self.decoder2_adapter is not None and cov_encoded is not None:
            dec2 = self.decoder2_adapter(
                dec2, cov_encoded, observed_mask, distance
            )
        dec1 = self.dec1(torch.cat([self.up2(dec2), enc1], dim=1))
        if self.decoder1_adapter is not None and cov_encoded is not None:
            dec1 = self.decoder1_adapter(
                dec1, cov_encoded, observed_mask, distance
            )

        raw_prediction = self.out_conv(dec1)
        return observed_mask * known_values + (1.0 - observed_mask) * raw_prediction

    def gate_means(self) -> dict[str, float]:
        result = {}
        for name, adapter in (
            ("bottleneck", self.bottleneck_adapter),
            ("decoder_16", self.decoder2_adapter),
            ("decoder_32", self.decoder1_adapter),
        ):
            if adapter is not None and adapter.last_gate is not None:
                result[name] = float(adapter.last_gate.mean().cpu())
        return result


def fusion_loss(
    config: Config,
    pollutant: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_grid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A1 loss: no covariate-derived beta weighting; retain TV/Laplacian."""

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
    tv = core.masked_total_variation(prediction, valid_grid_mask)
    laplacian = core.masked_laplacian_penalty(
        prediction, valid_grid_mask, config.DIFFUSION_COEFF
    )
    total = (
        supervised
        + config.TV_WEIGHTS[pollutant] * tv
        + config.LAPLACIAN_WEIGHTS[pollutant] * laplacian
    )
    return total, {"supervised": supervised, "tv": tv, "laplacian": laplacian}


def completion_marker(checkpoint: Path) -> Path:
    return Path(str(checkpoint) + ".complete.json")


def marker_matches(
    marker: dict,
    config: Config,
    variant: FusionVariant,
    pollutant: str,
    station_idx: int,
    feature_dim: int,
    signature: str,
) -> bool:
    return (
        marker.get("status") == "completed"
        and marker.get("model_version") == config.MODEL_VERSION
        and marker.get("seed") == GLOBAL_SEED
        and marker.get("variant") == variant.code
        and marker.get("pollutant") == pollutant
        and marker.get("station_idx") == station_idx
        and marker.get("feature_dim") == feature_dim
        and marker.get("group_signature") == signature
        and marker.get("use_scb_attention") == config.USE_SCB_ATTENTION
    )


def train_fold_model(
    config: Config,
    pollutant: str,
    station_idx: int,
    groups: list[GroupSpec],
    signature: str,
    variant: FusionVariant,
    checkpoint: Path,
) -> tuple[FusionSCUNet, dict[str, float | int]]:
    seed = fold_seed(pollutant, station_idx)
    seed_everything(seed)
    train_dataset = FusionDataset(
        config, pollutant, station_idx, mode="train", variant=variant
    )
    val_dataset = FusionDataset(
        config, pollutant, station_idx, mode="val", variant=variant
    )
    model = FusionSCUNet(
        train_dataset.base_channels,
        train_dataset.F_dim,
        groups,
        variant,
        config,
    ).to(DEVICE)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    marker_path = completion_marker(checkpoint)
    if config.REUSE_CHECKPOINTS and checkpoint.exists() and marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = {}
        if marker_matches(
            marker,
            config,
            variant,
            pollutant,
            station_idx,
            train_dataset.F_dim,
            signature,
        ):
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            print(f"    reuse {variant.code} {pollutant.upper()} station {station_idx:02d}")
            return model.eval(), {
                "train_seconds": float(marker.get("train_seconds", 0.0)),
                "best_epoch": int(marker.get("best_epoch", -1)),
                "best_validation_loss": float(
                    marker.get("best_validation_loss", np.nan)
                ),
                "parameter_count": parameter_count,
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
                loss, _ = fusion_loss(
                    config, pollutant, predictions, targets, valid_domain
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
                loss, _ = fusion_loss(
                    config, pollutant, predictions, targets, valid_domain
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
            mean_train = (
                float(np.mean(train_losses)) if train_losses else float("nan")
            )
            print(
                f"    {variant.code} {pollutant.upper()} station {station_idx:02d} "
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
        "seed": GLOBAL_SEED,
        "fold_seed": seed,
        "variant": variant.code,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "feature_dim": train_dataset.F_dim,
        "group_signature": signature,
        "use_scb_attention": config.USE_SCB_ATTENTION,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "train_seconds": elapsed,
        "parameter_count": parameter_count,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model.eval(), {
        "train_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "parameter_count": parameter_count,
    }


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def evaluate_fold(
    config: Config,
    pollutant: str,
    station_idx: int,
    variant: FusionVariant,
    model: FusionSCUNet,
) -> tuple[dict, pd.DataFrame]:
    dataset = FusionDataset(
        config, pollutant, station_idx, mode="test", variant=variant
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
                    "variant": variant.code,
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
        "variant": variant.code,
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
    variant: FusionVariant,
    pollutant: str,
    station_idx: int,
    feature_dim: int,
    signature: str,
) -> tuple[dict, pd.DataFrame] | None:
    metric_path, prediction_path, marker_path = fold_paths(
        root, pollutant, station_idx
    )
    if not config.REUSE_CHECKPOINTS:
        return None
    if not metric_path.exists() or not prediction_path.exists() or not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not marker_matches(
            marker,
            config,
            variant,
            pollutant,
            station_idx,
            feature_dim,
            signature,
        ):
            return None
        return json.loads(metric_path.read_text(encoding="utf-8")), pd.read_csv(
            prediction_path
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_completed_fold(
    config: Config,
    root: Path,
    variant: FusionVariant,
    pollutant: str,
    station_idx: int,
    feature_dim: int,
    signature: str,
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
        "variant": variant.code,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "feature_dim": feature_dim,
        "group_signature": signature,
        "use_scb_attention": config.USE_SCB_ATTENTION,
        "n_test": len(samples),
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
    variant: FusionVariant,
    pollutant: str,
    station_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    y_true = sample_df["true"].to_numpy(dtype=np.float64)
    y_pred = sample_df["prediction"].to_numpy(dtype=np.float64)
    row = {
        "variant": variant.code,
        "description": variant.description,
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


def prepare_variant_root(
    config: Config,
    variant: FusionVariant,
    pollutant: str,
    groups: list[GroupSpec],
    source: str,
    signature: str,
    audit: pd.DataFrame,
) -> Path:
    root = Path(config.FUSION_ROOT, variant.code)
    for name in ("models", "folds", "metrics", "predictions"):
        (root / name).mkdir(parents=True, exist_ok=True)
    active_groups = [group for group in groups if group.name != variant.ablate_group]
    protocol = {
        "model_version": config.MODEL_VERSION,
        "seed": GLOBAL_SEED,
        "variant": asdict(variant),
        "pollutant": pollutant,
        "outer_protocol": "station-wise strict LOSO",
        "inner_validation": "mask non-held-out training-fold stations only",
        "fold_scaler": "training times and non-held-out stations only",
        "time_split": "chronological train/validation/test from preprocessing",
        "prediction_clipping": False,
        "importance_beta": 0.0,
        "curriculum_masking": True,
        "tv_weights": config.TV_WEIGHTS,
        "laplacian_weights": config.LAPLACIAN_WEIGHTS,
        "asf": "replaced by grouped zero-initialized gated residual conditioning",
        "scb_channel_attention": config.USE_SCB_ATTENTION,
        "feature_group_source": source,
        "feature_groups_all": [asdict(group) for group in groups],
        "feature_groups_active": [asdict(group) for group in active_groups],
        "group_signature": signature,
        "group_width": config.GROUP_WIDTH,
        "gate_initial_bias": config.GATE_BIAS,
    }
    (root / f"protocol_{pollutant}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit.to_csv(root / "metrics" / f"{pollutant}_feature_channel_audit.csv", index=False)
    return root


def run_variant_pollutant(
    config: Config,
    variant: FusionVariant,
    pollutant: str,
    groups: list[GroupSpec],
    source: str,
    signature: str,
    audit: pd.DataFrame,
    total_stations: int,
    feature_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = prepare_variant_root(
        config, variant, pollutant, groups, source, signature, audit
    )
    stop = total_stations
    if config.MAX_STATIONS > 0:
        stop = min(total_stations, config.FOLD_START + config.MAX_STATIONS)
    if not 0 <= config.FOLD_START < stop:
        raise ValueError(
            f"Invalid fold range: start={config.FOLD_START}, stop={stop}"
        )

    station_summaries = []
    sample_frames = []
    print(
        f"\n{variant.code} | {pollutant.upper()} strict LOSO folds "
        f"{config.FOLD_START}..{stop - 1}"
    )
    for station_idx in range(config.FOLD_START, stop):
        reused = reuse_completed_fold(
            config,
            root,
            variant,
            pollutant,
            station_idx,
            feature_dim,
            signature,
        )
        if reused is not None:
            summary, samples = reused
            print(f"  reuse completed fold station {station_idx:02d}")
            station_summaries.append(summary)
            sample_frames.append(samples)
            continue

        checkpoint = (
            root / "models" / f"{pollutant}_station_{station_idx:02d}_{variant.code}.pth"
        )
        model, training_info = train_fold_model(
            config,
            pollutant,
            station_idx,
            groups,
            signature,
            variant,
            checkpoint,
        )
        summary, samples = evaluate_fold(
            config, pollutant, station_idx, variant, model
        )
        summary.update(training_info)
        station_summaries.append(summary)
        sample_frames.append(samples)
        save_completed_fold(
            config,
            root,
            variant,
            pollutant,
            station_idx,
            feature_dim,
            signature,
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

    station_df = pd.DataFrame(station_summaries)
    sample_df = pd.concat(sample_frames, ignore_index=True)
    aggregate_df = aggregate_results(
        config, variant, pollutant, station_df, sample_df
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
    print(
        aggregate_df[
            ["variant", "pollutant", "pooled_MAE", "pooled_RMSE", "pooled_R2"]
        ].to_string(index=False)
    )
    return aggregate_df, station_df


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


def paired_tests_vs_f0(station_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not station_frames:
        return pd.DataFrame()
    stations = pd.concat(station_frames, ignore_index=True)
    rows = []
    for pollutant in sorted(stations["pollutant"].unique()):
        pollutant_df = stations[stations["pollutant"] == pollutant]
        reference = pollutant_df[pollutant_df["variant"] == "F0"]
        if reference.empty:
            continue
        for variant in sorted(pollutant_df["variant"].unique()):
            if variant == "F0":
                continue
            candidate = pollutant_df[pollutant_df["variant"] == variant]
            for metric in ("MAE", "RMSE", "R2"):
                paired = reference[["station", metric]].merge(
                    candidate[["station", metric]],
                    on="station",
                    suffixes=("_F0", "_candidate"),
                ).dropna()
                if len(paired) < 2:
                    continue
                delta = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_F0"]
                ).to_numpy(dtype=np.float64)
                try:
                    p_value = float(wilcoxon(delta, zero_method="wilcox").pvalue)
                except ValueError:
                    p_value = 1.0
                rows.append(
                    {
                        "pollutant": pollutant,
                        "reference": "F0",
                        "candidate": variant,
                        "metric": metric,
                        "n_stations": len(paired),
                        "mean_candidate_minus_F0": float(np.mean(delta)),
                        "median_candidate_minus_F0": float(np.median(delta)),
                        "wilcoxon_p": p_value,
                    }
                )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["holm_p"] = holm_adjust(tests["wilcoxon_p"].tolist())
    return tests


def main() -> None:
    config = Config()
    if config.GROUP_WIDTH <= 0:
        raise ValueError("FUSION_GROUP_WIDTH must be positive")
    if not 0.0 <= config.GROUP_DROPOUT < 1.0:
        raise ValueError("FUSION_GROUP_DROPOUT must be in [0, 1)")
    Path(config.FUSION_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Selected variants: {config.SELECTED_VARIANTS}")
    print(f"Feature grouping mode: {config.GROUP_MODE}")

    aggregate_frames = []
    station_frames = []
    resolved_group_payload = {}
    for pollutant in config.POLLUTANTS:
        probe = core.SpatialInpaintDatasetLOOCV(
            config, pollutant, mask_station_idx=0, mode="test"
        )
        groups, source, audit = resolve_groups(config, probe, pollutant)
        signature = group_signature(groups)
        variants = variants_for_groups(config, groups)
        resolved_group_payload[pollutant] = {
            "source": source,
            "signature": signature,
            "groups": [asdict(group) for group in groups],
        }
        total_stations = probe.total_stations
        feature_dim = probe.F_dim
        print(
            f"\n{pollutant.upper()} groups: "
            + ", ".join(f"{g.name}[{len(g.indices)}]" for g in groups)
        )
        del probe

        for variant in variants:
            aggregate, stations = run_variant_pollutant(
                config,
                variant,
                pollutant,
                groups,
                source,
                signature,
                audit,
                total_stations,
                feature_dim,
            )
            aggregate_frames.append(aggregate)
            station_frames.append(stations)

    Path(config.FUSION_ROOT, "resolved_feature_groups.json").write_text(
        json.dumps(resolved_group_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if aggregate_frames:
        pd.concat(aggregate_frames, ignore_index=True).to_csv(
            Path(config.FUSION_ROOT, "all_aggregate_metrics.csv"), index=False
        )
    tests = paired_tests_vs_f0(station_frames)
    if not tests.empty:
        tests.to_csv(
            Path(config.FUSION_ROOT, "paired_tests_vs_F0.csv"), index=False
        )


if __name__ == "__main__":
    main()
