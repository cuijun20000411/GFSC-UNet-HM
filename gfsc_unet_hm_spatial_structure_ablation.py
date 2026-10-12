"""Five-fold spatial structural ablations for GFSC-UNet-HM.

This runner keeps the published hybrid masking, dual-view validation, fold-only
normalization, loss, optimizer, random seed, and fixed five-block spatial split
unchanged.  Only the covariate-conditioning structure is varied:

FULL          complete GFSC-UNet-HM;
NO_COV        remove all auxiliary-variable paths;
UNGROUPED     replace grouped encoders with a parameter-matched single encoder;
DIRECT_CONCAT concatenate grouped covariate features with the observation input;
NO_GATE       retain three fusion scales but remove the sigmoid gates.

The script reuses the audited training and evaluation functions supplied with
the released GFSC-UNet-HM code. It also builds pooled metrics, station-paired
Wilcoxon tests, and Holm-adjusted p-values from completed prediction files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import fusion_final_strict as fusion
import gfsc_unet_d3_hybrid_mask_main as base


SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
KEYS = ["pollutant", "station", "time_idx"]
VARIANT_ORDER = ["FULL", "NO_COV", "UNGROUPED", "DIRECT_CONCAT", "NO_GATE"]


@dataclass(frozen=True)
class VariantSpec:
    code: str
    description: str
    model_kind: str
    use_covariates: bool = True
    bottleneck_fusion: bool = True
    decoder_fusion: bool = True


VARIANTS = {
    "FULL": VariantSpec(
        "FULL", "Complete GFSC-UNet-HM", "full"
    ),
    "NO_COV": VariantSpec(
        "NO_COV", "Remove every auxiliary-variable path", "no_cov",
        use_covariates=False, bottleneck_fusion=False, decoder_fusion=False,
    ),
    "UNGROUPED": VariantSpec(
        "UNGROUPED", "Parameter-matched single covariate encoder", "ungrouped"
    ),
    "DIRECT_CONCAT": VariantSpec(
        "DIRECT_CONCAT",
        "Grouped covariate encoding concatenated at the observation input",
        "direct_concat", bottleneck_fusion=False, decoder_fusion=False,
    ),
    "NO_GATE": VariantSpec(
        "NO_GATE", "Three-scale residual conditioning without sigmoid gates", "no_gate"
    ),
}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-12 else float("nan"),
    }


def grouped_encoder_parameter_budget(groups: list[fusion.GroupSpec], width: int) -> int:
    hidden = max(width, min(32, 2 * width))
    total = 0
    for group in groups:
        kernel_elements = 1 if group.kind == "global" else 9
        total += hidden * (len(group.indices) + 3 + width * kernel_elements) + 3 * width
    return int(total)


def matched_ungrouped_hidden(
    feature_dim: int, condition_channels: int, target_parameters: int
) -> tuple[int, int]:
    def count(hidden: int) -> int:
        return int(hidden * (feature_dim + 3 + 9 * condition_channels) + 3 * condition_channels)

    hidden = min(range(1, 129), key=lambda value: (abs(count(value) - target_parameters), value))
    return hidden, count(hidden)


class ParameterMatchedUngroupedEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        condition_channels: int,
        hidden_channels: int,
        dropout_probability: float,
    ):
        super().__init__()
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
            nn.GroupNorm(fusion.norm_groups(condition_channels), condition_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        features = self.net(covariates)
        if self.training and self.dropout_probability > 0:
            keep = (
                torch.rand(features.shape[0], 1, 1, 1, device=features.device)
                >= self.dropout_probability
            ).to(features.dtype)
            features = features * keep / (1.0 - self.dropout_probability)
        return features


class UngroupedFusionSCUNet(fusion.FusionSCUNet):
    def __init__(self, base_channels, cov_channels, groups, variant, config):
        super().__init__(base_channels, cov_channels, groups, variant, config)
        if self.covariate_encoder is None:
            raise RuntimeError("UNGROUPED requires an active covariate encoder")
        condition_channels = config.GROUP_WIDTH * len(groups)
        target = grouped_encoder_parameter_budget(groups, config.GROUP_WIDTH)
        hidden, actual = matched_ungrouped_hidden(cov_channels, condition_channels, target)
        replacement = ParameterMatchedUngroupedEncoder(
            cov_channels, condition_channels, hidden, variant.group_dropout
        )
        if replacement.out_channels != self.covariate_encoder.out_channels:
            raise RuntimeError("Condition-width mismatch in UNGROUPED")
        self.covariate_encoder = replacement
        self.encoder_parameter_target = target
        self.encoder_parameter_actual = actual
        self.ungrouped_hidden_channels = hidden


class UngatedResidualAdapter(nn.Module):
    def __init__(self, main_channels: int, cov_channels: int):
        super().__init__()
        hidden = max(8, min(64, main_channels // 2))
        self.residual = nn.Sequential(
            nn.Conv2d(cov_channels, hidden, kernel_size=1),
            nn.GroupNorm(fusion.norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, main_channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.last_gate = None

    def forward(self, main, covariates, observed_mask, distance):
        del observed_mask, distance
        if covariates.shape[-2:] != main.shape[-2:]:
            covariates = F.adaptive_avg_pool2d(covariates, main.shape[-2:])
        return main + self.residual(covariates)


class NoGateFusionSCUNet(fusion.FusionSCUNet):
    def __init__(self, base_channels, cov_channels, groups, variant, config):
        super().__init__(base_channels, cov_channels, groups, variant, config)
        if self.covariate_encoder is None:
            raise RuntimeError("NO_GATE requires an active covariate encoder")
        condition_channels = self.covariate_encoder.out_channels
        self.bottleneck_adapter = UngatedResidualAdapter(256, condition_channels)
        self.decoder2_adapter = UngatedResidualAdapter(64, condition_channels)
        self.decoder1_adapter = UngatedResidualAdapter(32, condition_channels)


class DirectConcatSCUNet(nn.Module):
    """Encode the three groups, then concatenate them once at the input."""

    def __init__(self, base_channels, cov_channels, groups, variant, config):
        super().__init__()
        self.base_channels = int(base_channels)
        self.cov_channels = int(cov_channels)
        self.variant = variant
        self.covariate_encoder = fusion.GroupedCovariateEncoder(
            groups, config.GROUP_WIDTH, variant.group_dropout
        )
        input_channels = self.base_channels + self.covariate_encoder.out_channels
        attention = config.USE_SCB_ATTENTION
        self.enc1 = fusion.SpatialConstraintBlock(input_channels, 32, attention)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), fusion.SpatialConstraintBlock(32, 64, attention))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), fusion.SpatialConstraintBlock(64, 128, attention))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), fusion.SpatialConstraintBlock(128, 256, attention))
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = fusion.SpatialConstraintBlock(256, 128, attention)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = fusion.SpatialConstraintBlock(128, 64, attention)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = fusion.SpatialConstraintBlock(64, 32, attention)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, observed_mask, known_values):
        base_features = x[:, : self.base_channels]
        covariates = x[:, self.base_channels :]
        if covariates.shape[1] != self.cov_channels:
            raise ValueError(
                f"Expected {self.cov_channels} covariate channels, got {covariates.shape[1]}"
            )
        encoded = self.covariate_encoder(covariates)
        enc1 = self.enc1(torch.cat([base_features, encoded], dim=1))
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec3 = self.dec3(torch.cat([self.up4(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up3(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up2(dec2), enc1], dim=1))
        raw = self.out_conv(dec1)
        return observed_mask * known_values + (1.0 - observed_mask) * raw

    def gate_means(self) -> dict[str, float]:
        return {}


def fusion_variant(spec: VariantSpec, config) -> fusion.FusionVariant:
    return fusion.FusionVariant(
        code=spec.code,
        description=spec.description,
        use_covariates=spec.use_covariates,
        bottleneck_fusion=spec.bottleneck_fusion,
        decoder_fusion=spec.decoder_fusion,
        group_dropout=float(config.GROUP_DROPOUT) if spec.use_covariates else 0.0,
        ablate_group=None,
    )


def model_class(spec: VariantSpec):
    return {
        "full": fusion.FusionSCUNet,
        "no_cov": fusion.FusionSCUNet,
        "ungrouped": UngroupedFusionSCUNet,
        "direct_concat": DirectConcatSCUNet,
        "no_gate": NoGateFusionSCUNet,
    }[spec.model_kind]


def patch_variant(spec: VariantSpec, config) -> None:
    original_fingerprint = base.model_fingerprint
    config.MODEL_VERSION = f"gfsc_unet_hm_spatial_structure_{spec.code.lower()}_v1"
    base.d3_variant = lambda current_config: fusion_variant(spec, current_config)
    base.GFSCUNetD3 = model_class(spec)

    def fingerprint(*args, **kwargs):
        base_hash = original_fingerprint(*args, **kwargs)
        payload = {
            "base_fingerprint": base_hash,
            "runner_sha256": SOURCE_SHA256,
            "variant": asdict(spec),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    base.model_fingerprint = fingerprint


def restore_base(originals: dict[str, object]) -> None:
    for name, value in originals.items():
        setattr(base, name, value)


def prediction_path(result_root: Path) -> Path:
    direct = result_root / "predictions" / "test_predictions.csv"
    nested = result_root / "spatial_block" / "D3HM" / "predictions" / "test_predictions.csv"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"No completed spatial-block prediction file under {result_root}")


def parse_existing(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--existing must use VARIANT=/absolute/or/relative/path")
        code, path = value.split("=", 1)
        code = code.strip().upper()
        if code not in VARIANTS:
            raise ValueError(f"Unknown existing variant: {code}")
        result[code] = Path(path).expanduser().resolve()
    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(output_root: Path, existing: dict[str, Path]) -> None:
    frames = {}
    for code in VARIANT_ORDER:
        root = existing.get(code, output_root / code)
        path = prediction_path(root)
        frame = pd.read_csv(path)
        required = set(KEYS + ["true", "prediction"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        frame = frame[KEYS + ["true", "prediction"]].copy()
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{code} contains duplicate prediction keys")
        if not np.isfinite(frame[["true", "prediction"]].to_numpy()).all():
            raise ValueError(f"{code} contains non-finite truth or predictions")
        frames[code] = frame.sort_values(KEYS).reset_index(drop=True)

    full = frames["FULL"]
    full_keys = full[KEYS]
    for code, frame in frames.items():
        if not frame[KEYS].equals(full_keys):
            raise ValueError(f"Prediction keys for {code} do not match FULL")
        if not np.allclose(frame["true"], full["true"], rtol=2e-6, atol=2e-5):
            raise ValueError(f"Test truth for {code} does not match FULL")

    output = output_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    station_rows = []
    for code, frame in frames.items():
        for pollutant, part in frame.groupby("pollutant", sort=True):
            metric_rows.append({
                "variant": code,
                "pollutant": pollutant,
                "n_stations": int(part.station.nunique()),
                "n_predictions": int(len(part)),
                **metric_dict(part.true, part.prediction),
            })
        for (pollutant, station), part in frame.groupby(["pollutant", "station"], sort=True):
            station_rows.append({
                "variant": code,
                "pollutant": pollutant,
                "station": int(station),
                "n_test": int(len(part)),
                **metric_dict(part.true, part.prediction),
            })
    metrics = pd.DataFrame(metric_rows)
    stations = pd.DataFrame(station_rows)
    metrics.to_csv(output / "aggregate_metrics.csv", index=False)
    stations.to_csv(output / "station_metrics.csv", index=False)

    tests = []
    for code in VARIANT_ORDER[1:]:
        for pollutant in sorted(full.pollutant.unique()):
            reference = stations[(stations.variant == "FULL") & (stations.pollutant == pollutant)]
            candidate = stations[(stations.variant == code) & (stations.pollutant == pollutant)]
            paired = reference[["station", "RMSE"]].merge(
                candidate[["station", "RMSE"]], on="station", suffixes=("_full", "_variant"), validate="one_to_one"
            )
            difference = paired.RMSE_variant.to_numpy() - paired.RMSE_full.to_numpy()
            p_value = 1.0 if np.allclose(difference, 0.0) else float(
                wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue
            )
            tests.append({
                "reference": "FULL",
                "variant": code,
                "pollutant": pollutant,
                "n_stations": len(paired),
                "median_RMSE_difference_variant_minus_full": float(np.median(difference)),
                "variant_better_stations": int(np.sum(difference < 0)),
                "full_better_stations": int(np.sum(difference > 0)),
                "p_raw": p_value,
            })
    tests_frame = pd.DataFrame(tests)
    tests_frame["p_holm_12"] = holm_adjust(tests_frame.p_raw.tolist())
    tests_frame["significant_0_05"] = tests_frame.p_holm_12 < 0.05
    tests_frame.to_csv(output / "paired_station_tests_vs_full.csv", index=False)

    table = metrics.pivot(index="variant", columns="pollutant", values="RMSE").reindex(VARIANT_ORDER)
    full_rmse = table.loc["FULL"]
    relative = 100.0 * (table - full_rmse) / full_rmse
    table.columns = [f"RMSE_{name}" for name in table.columns]
    relative.columns = [f"RMSE_change_vs_FULL_percent_{name}" for name in relative.columns]
    paper = table.join(relative).reset_index()
    paper.to_csv(output / "paper_table_spatial_structure_ablation.csv", index=False)
    protocol = {
        "variants": {code: asdict(VARIANTS[code]) for code in VARIANT_ORDER},
        "paired_unit": "held-out station RMSE over all test times",
        "test": "two-sided Wilcoxon signed-rank",
        "multiplicity": "Holm adjustment across 4 ablations x 3 pollutants = 12 tests",
        "prediction_paths": {
            code: str(prediction_path(existing.get(code, output_root / code)))
            for code in VARIANT_ORDER
        },
    }
    (output / "summary_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(paper.to_string(index=False), flush=True)
    print(f"Summary written to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/etc/cj/boundary/duoyuan/data")
    parser.add_argument(
        "--d3-reference-root",
        default="/etc/cj/boundary/f3_direct_ablation_final_strict_v1/D3",
    )
    parser.add_argument(
        "--spatial-split-json",
        default=str(Path(__file__).with_name("spatial_split.json")),
    )
    parser.add_argument(
        "--output",
        default="/etc/cj/boundary/gfsc_unet_hm_spatial_structure_ablation_v1",
    )
    parser.add_argument("--pollutants", default="no2,o3,pm25")
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--existing",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Use completed predictions outside --output; repeat as needed.",
    )
    args = parser.parse_args()
    args.pollutants = [item.strip().lower() for item in args.pollutants.split(",") if item.strip()]
    args.variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if not args.pollutants or set(args.pollutants) - {"no2", "o3", "pm25"}:
        raise ValueError("--pollutants must be selected from no2,o3,pm25")
    if args.fold_start < 0 or args.max_folds < 0:
        raise ValueError("fold range cannot be negative")
    return args


def preflight(args: argparse.Namespace, config) -> None:
    for pollutant in args.pollutants:
        base.validate_data_files(Path(args.data_root), pollutant)
    probe = base.HybridMaskFusionDataset(
        config,
        args.pollutants[0],
        [],
        mode="test",
        variant=fusion_variant(VARIANTS["FULL"], config),
        mask_view="none",
    )
    parts, _ = base.load_spatial_blocks(args.spatial_split_json, probe.station_positions)
    groups, _, _ = base.load_groups(
        base.resolve_reference_root(args.d3_reference_root),
        args.pollutants[0],
        probe.F_dim,
        config,
    )
    sample = probe[0]
    for code in args.variants:
        spec = VARIANTS[code]
        variant = fusion_variant(spec, config)
        model = model_class(spec)(probe.base_channels, probe.F_dim, groups, variant, config).eval()
        model_input = sample["input"].unsqueeze(0)
        if not spec.use_covariates:
            model_input = model_input[:, : probe.base_channels]
        with torch.no_grad():
            output = model(
                model_input,
                sample["channel_mask"].unsqueeze(0),
                sample["known_values"].unsqueeze(0),
            )
        if output.shape != (1, 1, 32, 32):
            raise RuntimeError(f"{code} returned unexpected shape {tuple(output.shape)}")
        print(f"{code}: parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"spatial blocks: {[len(part) for part in parts]}")
    print("Preflight passed; no training was started.")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing = parse_existing(args.existing)
    if args.summary_only:
        summarize(output_root, existing)
        return

    run_args = SimpleNamespace(
        data_root=args.data_root,
        d3_reference_root=args.d3_reference_root,
        spatial_split_json=args.spatial_split_json,
        pollutants=args.pollutants,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        fold_start=args.fold_start,
        max_folds=args.max_folds,
        output="",
    )
    config = base.configure(run_args)
    if args.preflight:
        preflight(args, config)
        return

    originals = {
        "d3_variant": base.d3_variant,
        "GFSCUNetD3": base.GFSCUNetD3,
        "model_fingerprint": base.model_fingerprint,
    }
    for code in args.variants:
        if code in existing:
            print(f"{code}: using existing results at {existing[code]}", flush=True)
            continue
        restore_base(originals)
        spec = VARIANTS[code]
        patch_variant(spec, config)
        run_args.output = str(output_root / code)
        print(f"\n=== {code}: {spec.description} ===", flush=True)
        base.run_spatial_block(run_args, config)
    restore_base(originals)

    available = set(existing)
    available.update(
        code for code in VARIANT_ORDER if (output_root / code).exists()
    )
    if set(VARIANT_ORDER).issubset(available):
        summarize(output_root, existing)
    else:
        missing = sorted(set(VARIANT_ORDER) - available)
        print(f"Training completed for the selected variants. Summary awaits: {missing}")


if __name__ == "__main__":
    main()
