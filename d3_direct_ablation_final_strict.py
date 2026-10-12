"""D3-centred confirmatory ablations under the strict 55-station LOSO protocol.

D3 is the completed F3 direct ablation that removes SCB channel attention.  This
runner keeps that choice fixed and removes one remaining component at a time.
It reuses the tested training/evaluation engine in
``f3_direct_ablation_final_strict.py`` and never overwrites F0--F3 or D1--D13.

Required neighbouring files
----------------------------
* ``rh4_final_strict_light.py``
* ``fusion_final_strict.py``
* ``f3_direct_ablation_final_strict.py``

D3-centred variants
--------------------
D3A1   no distance-to-nearest-observation channel
D3A2   no local mean/std or sample observed mean channels
D3A3   fixed 25% training mask (no masking curriculum)
D3A4   no total-variation penalty
D3A5   no Laplacian penalty
D3A6   supervised loss only
D3A7   parameter-matched ungrouped covariate encoder
D3A8   no bottleneck gated adapter
D3A9   no decoder gated adapters
D3A10  no covariate conditioning path (D3 backbone only)
D3A11  no dynamic-spatial covariate group
D3A12  no static-spatial covariate group
D3A13  no global-temporal covariate group
D3A14  no whole-group dropout

The already-completed F3 result is the inverse ablation that restores channel
attention, so it is included in summaries but is not retrained here.

Examples
--------
Smoke test::

    SCUNET_POLLUTANTS=no2 D3_ABLATION_VARIANTS=D3A1 \
      D3_ABLATION_MAX_STATIONS=1 python -u d3_direct_ablation_final_strict.py

Run the primary structural set::

    D3_ABLATION_VARIANTS=PRIMARY python -u d3_direct_ablation_final_strict.py

Run loss/curriculum controls::

    D3_ABLATION_VARIANTS=LOSS python -u d3_direct_ablation_final_strict.py

Run every D3-centred ablation::

    D3_ABLATION_VARIANTS=ALL python -u d3_direct_ablation_final_strict.py

Rebuild summaries only::

    D3_ABLATION_SUMMARY_ONLY=1 python -u d3_direct_ablation_final_strict.py

Default paths
-------------
``D3_ABLATION_RESULT_ROOT``
    /etc/cj/boundary/d3_direct_ablation_final_strict_v1
``D3_REFERENCE_RESULT_ROOT``
    /etc/cj/boundary/f3_direct_ablation_final_strict_v1
``FUSION_REFERENCE_RESULT_ROOT``
    /etc/cj/boundary/fusion_final_strict_v1

All model-selection, scaling, masking, and test rules remain identical to the
parent strict runner.  Outer-station truth is used only after checkpoint
selection.  Predictions are reported in raw concentration units without test
clipping.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

try:
    import f3_direct_ablation_final_strict as engine
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Place d3_direct_ablation_final_strict.py beside "
        "f3_direct_ablation_final_strict.py, fusion_final_strict.py, and "
        "rh4_final_strict_light.py."
    ) from exc


fusion = engine.fusion
core = engine.core
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GLOBAL_SEED = int(
    os.environ.get("D3_ABLATION_SEED", os.environ.get("FUSION_SEED", "42"))
)


@dataclass(frozen=True)
class D3Ablation:
    code: str
    description: str
    use_distance: bool = True
    use_statistics: bool = True
    use_scb_attention: bool = False
    fixed_mask_ratio: bool = False
    tv_multiplier: float = 1.0
    laplacian_multiplier: float = 1.0
    grouping: str = "grouped"
    bottleneck_fusion: bool = True
    decoder_fusion: bool = True
    ablate_group: str | None = None
    group_dropout_override: float | None = None
    disconnect_covariates: bool = False


ABLATIONS: dict[str, D3Ablation] = {
    "D3A1": D3Ablation(
        "D3A1", "D3 without distance-to-nearest-observation channel",
        use_distance=False,
    ),
    "D3A2": D3Ablation(
        "D3A2", "D3 without local mean, local std, or sample observed mean",
        use_statistics=False,
    ),
    "D3A3": D3Ablation(
        "D3A3", "D3 with fixed 25 percent training mask",
        fixed_mask_ratio=True,
    ),
    "D3A4": D3Ablation(
        "D3A4", "D3 without total-variation penalty", tv_multiplier=0.0,
    ),
    "D3A5": D3Ablation(
        "D3A5", "D3 without Laplacian penalty", laplacian_multiplier=0.0,
    ),
    "D3A6": D3Ablation(
        "D3A6", "D3 with supervised loss only",
        tv_multiplier=0.0, laplacian_multiplier=0.0,
    ),
    "D3A7": D3Ablation(
        "D3A7", "D3 with a parameter-matched ungrouped covariate encoder",
        grouping="ungrouped",
    ),
    "D3A8": D3Ablation(
        "D3A8", "D3 without the bottleneck gated adapter",
        bottleneck_fusion=False,
    ),
    "D3A9": D3Ablation(
        "D3A9", "D3 without the decoder gated adapters",
        decoder_fusion=False,
    ),
    "D3A10": D3Ablation(
        "D3A10", "D3 backbone without covariate conditioning",
        bottleneck_fusion=False,
        decoder_fusion=False,
        disconnect_covariates=True,
    ),
    "D3A11": D3Ablation(
        "D3A11", "D3 without the dynamic-spatial covariate group",
        ablate_group="dynamic_spatial",
    ),
    "D3A12": D3Ablation(
        "D3A12", "D3 without the static-spatial covariate group",
        ablate_group="static_spatial",
    ),
    "D3A13": D3Ablation(
        "D3A13", "D3 without the global-temporal covariate group",
        ablate_group="global_temporal",
    ),
    "D3A14": D3Ablation(
        "D3A14", "D3 without whole-group dropout",
        group_dropout_override=0.0,
    ),
}

PRESETS = {
    "PRIMARY": [
        "D3A1", "D3A2", "D3A7", "D3A8", "D3A9", "D3A10",
        "D3A11", "D3A12", "D3A13", "D3A14",
    ],
    "LOSS": ["D3A3", "D3A4", "D3A5", "D3A6"],
    "GROUPS": ["D3A11", "D3A12", "D3A13", "D3A14"],
}


class Config(engine.Config):
    ABLATION_ROOT = os.environ.get(
        "D3_ABLATION_RESULT_ROOT",
        "/etc/cj/boundary/d3_direct_ablation_final_strict_v1",
    )
    D3_REFERENCE_ROOT = os.environ.get(
        "D3_REFERENCE_RESULT_ROOT",
        "/etc/cj/boundary/f3_direct_ablation_final_strict_v1",
    )
    REFERENCE_ROOT = os.environ.get(
        "FUSION_REFERENCE_RESULT_ROOT",
        os.environ.get(
            "F3_REFERENCE_RESULT_ROOT",
            "/etc/cj/boundary/fusion_final_strict_v1",
        ),
    )
    POLLUTANTS = [
        item.strip().lower()
        for item in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if item.strip()
    ]
    SELECTED_ABLATIONS = [
        item.strip().upper()
        for item in os.environ.get("D3_ABLATION_VARIANTS", "PRIMARY").split(",")
        if item.strip()
    ]
    FOLD_START = int(os.environ.get("D3_ABLATION_FOLD_START", "0"))
    MAX_STATIONS = int(os.environ.get("D3_ABLATION_MAX_STATIONS", "0"))
    REUSE_CHECKPOINTS = os.environ.get("D3_ABLATION_REUSE_COMPLETED", "1") == "1"
    SUMMARY_ONLY = os.environ.get("D3_ABLATION_SUMMARY_ONLY", "0") == "1"
    DEFER_SUMMARY = os.environ.get("D3_ABLATION_DEFER_SUMMARY", "0") == "1"
    ALLOW_REFERENCE_MISMATCH = (
        os.environ.get("D3_ABLATION_ALLOW_REFERENCE_MISMATCH", "0") == "1"
    )
    NUM_WORKERS = int(
        os.environ.get("D3_ABLATION_NUM_WORKERS", str(fusion.Config.NUM_WORKERS))
    )
    LOG_INTERVAL = int(
        os.environ.get("D3_ABLATION_LOG_INTERVAL", str(fusion.Config.LOG_INTERVAL))
    )
    EPOCHS = int(os.environ.get("D3_ABLATION_EPOCHS", str(fusion.Config.EPOCHS)))
    BATCH_SIZE = int(
        os.environ.get("D3_ABLATION_BATCH_SIZE", str(fusion.Config.BATCH_SIZE))
    )
    PATIENCE = int(
        os.environ.get("D3_ABLATION_PATIENCE", str(fusion.Config.PATIENCE))
    )
    BOOTSTRAP_REPEATS = int(os.environ.get("D3_ABLATION_BOOTSTRAP_REPEATS", "2000"))
    PAIR_BOOTSTRAP_REPEATS = int(
        os.environ.get("D3_ABLATION_PAIR_BOOTSTRAP_REPEATS", "5000")
    )
    MODEL_VERSION = "strict_d3_centered_ablation_v1"


def selected_ablations(config: Config) -> list[D3Ablation]:
    requested: list[str] = []
    for token in config.SELECTED_ABLATIONS:
        if token == "ALL":
            requested.extend(ABLATIONS)
        elif token in PRESETS:
            requested.extend(PRESETS[token])
        else:
            requested.append(token)
    unknown = sorted(set(requested) - set(ABLATIONS))
    if unknown:
        raise ValueError(
            f"Unknown D3_ABLATION_VARIANTS={unknown}; available="
            f"{list(ABLATIONS)}, presets={list(PRESETS)}"
        )
    if not requested and not config.SUMMARY_ONLY:
        raise ValueError("D3_ABLATION_VARIANTS did not select an experiment")
    result: list[D3Ablation] = []
    seen: set[str] = set()
    for code in requested:
        if code not in seen:
            result.append(ABLATIONS[code])
            seen.add(code)
    return result


def group_dropout_for(ablation: D3Ablation, config: Config) -> float:
    if ablation.group_dropout_override is None:
        return float(config.GROUP_DROPOUT)
    return float(ablation.group_dropout_override)


def fusion_variant(
    ablation: D3Ablation, config: Config
) -> fusion.FusionVariant:
    return fusion.FusionVariant(
        code=ablation.code,
        description=ablation.description,
        use_covariates=True,
        bottleneck_fusion=ablation.bottleneck_fusion,
        decoder_fusion=ablation.decoder_fusion,
        group_dropout=group_dropout_for(ablation, config),
        ablate_group=ablation.ablate_group,
    )


class DisconnectedCovariateEncoder(nn.Module):
    """Zero-parameter placeholder used only by the full no-conditioning control."""

    out_channels = 0

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        return covariates[:, :0]


_OriginalDirectFusionSCUNet = engine.DirectFusionSCUNet


class D3CenteredFusionSCUNet(_OriginalDirectFusionSCUNet):
    def __init__(
        self,
        base_channels: int,
        cov_channels: int,
        effective_groups: list[fusion.GroupSpec],
        variant: fusion.FusionVariant,
        model_config: Config,
        ablation: D3Ablation,
        reference_groups: list[fusion.GroupSpec],
        reference_group_width: int,
    ):
        super().__init__(
            base_channels,
            cov_channels,
            effective_groups,
            variant,
            model_config,
            ablation,
            reference_groups,
            reference_group_width,
        )
        if ablation.disconnect_covariates:
            if self.bottleneck_adapter is not None or self.decoder2_adapter is not None:
                raise RuntimeError("D3A10 must not retain any covariate adapter")
            self.covariate_encoder = DisconnectedCovariateEncoder()


def protocol_payload(
    config: Config,
    ablation: D3Ablation,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
    effective_groups: list[fusion.GroupSpec],
    effective_width: int,
    group_source: str,
) -> dict:
    active_groups = [
        group for group in effective_groups if group.name != ablation.ablate_group
    ]
    dropout = group_dropout_for(ablation, config)
    if ablation.disconnect_covariates:
        condition_channels = 0
        encoder_actual = 0
        ungrouped_hidden = None
    else:
        condition_channels = effective_width * len(active_groups)
        if ablation.grouping == "ungrouped":
            ungrouped_hidden, encoder_actual = engine.matched_ungrouped_hidden(
                sum(len(group.indices) for group in resolved_groups),
                condition_channels,
                engine.grouped_encoder_parameter_budget(
                    resolved_groups, config.GROUP_WIDTH
                ),
            )
        else:
            ungrouped_hidden = None
            encoder_actual = engine.grouped_encoder_parameter_budget(
                active_groups, effective_width
            )
    return {
        "model_version": config.MODEL_VERSION,
        "source_sha256": SOURCE_SHA256,
        "seed": GLOBAL_SEED,
        "reference_model": "D3",
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
        "scb_channel_attention": False,
        "feature_group_source": group_source,
        "resolved_groups": [asdict(group) for group in resolved_groups],
        "effective_groups": [asdict(group) for group in effective_groups],
        "active_effective_groups": [asdict(group) for group in active_groups],
        "group_width": effective_width,
        "condition_channels": condition_channels,
        "encoder_parameter_target_D3": engine.grouped_encoder_parameter_budget(
            resolved_groups, config.GROUP_WIDTH
        ),
        "encoder_parameter_actual": encoder_actual,
        "ungrouped_hidden_channels": ungrouped_hidden,
        "group_dropout": dropout,
        "gate_initial_bias": config.GATE_BIAS,
        "bottleneck_fusion": ablation.bottleneck_fusion,
        "decoder_fusion": ablation.decoder_fusion,
        "covariate_conditioning_connected": not ablation.disconnect_covariates,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "start_mask_ratio": config.START_MASK_RATIO,
        "final_mask_ratio": config.FINAL_MASK_RATIO,
        "curriculum_epoch": config.CURRICULUM_EPOCH,
    }


def _json_normalize(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def validate_reference_protocol(
    config: Config,
    pollutant: str,
    resolved_groups: list[fusion.GroupSpec],
) -> None:
    path = Path(config.D3_REFERENCE_ROOT, "D3", f"protocol_{pollutant}.json")
    if not path.exists():
        message = f"D3 reference protocol is missing: {path}"
        if config.ALLOW_REFERENCE_MISMATCH:
            print(f"[warning] {message}")
            return
        raise FileNotFoundError(message)
    protocol = engine.load_json(path)
    observed_ablation = protocol.get("ablation", {})
    observed_groups = _json_normalize(protocol.get("resolved_groups", []))
    expected_groups = _json_normalize([asdict(group) for group in resolved_groups])
    checks = {
        "variant": (observed_ablation.get("code"), "D3"),
        "seed": (protocol.get("seed"), GLOBAL_SEED),
        "resolved_groups": (observed_groups, expected_groups),
        "group_width": (protocol.get("group_width"), config.GROUP_WIDTH),
        "group_dropout": (protocol.get("group_dropout"), config.GROUP_DROPOUT),
        "importance_beta": (protocol.get("importance_beta"), 0.0),
        "curriculum_masking": (protocol.get("curriculum_masking"), True),
        "tv_multiplier": (protocol.get("tv_multiplier"), 1.0),
        "laplacian_multiplier": (protocol.get("laplacian_multiplier"), 1.0),
        "scb_channel_attention": (
            protocol.get("scb_channel_attention"), False
        ),
        "bottleneck_fusion": (protocol.get("bottleneck_fusion"), True),
        "decoder_fusion": (protocol.get("decoder_fusion"), True),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        message = (
            f"D3 reference protocol mismatch for {pollutant}: "
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )
        if config.ALLOW_REFERENCE_MISMATCH:
            print(f"[warning] {message}")
        else:
            raise RuntimeError(message)


def read_available_results(
    config: Config,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    aggregates: list[pd.DataFrame] = []
    stations: list[pd.DataFrame] = []

    def add(code: str, root: Path, source: str) -> None:
        for pollutant in config.POLLUTANTS:
            aggregate_path = root / code / "metrics" / f"{pollutant}_aggregate_metrics.csv"
            station_path = root / code / "metrics" / f"{pollutant}_station_metrics.csv"
            if aggregate_path.exists():
                frame = pd.read_csv(aggregate_path)
                frame["result_source"] = source
                aggregates.append(frame)
            if station_path.exists():
                frame = pd.read_csv(station_path)
                frame["result_source"] = source
                stations.append(frame)

    fusion_root = Path(config.REFERENCE_ROOT)
    for code in ("F0", "F1", "F2", "F3"):
        add(code, fusion_root, "existing_F0_F3")
    add("D3", Path(config.D3_REFERENCE_ROOT), "existing_D3")
    result_root = Path(config.ABLATION_ROOT)
    for code in ABLATIONS:
        add(code, result_root, "D3_centered_ablation")
    return aggregates, stations


def build_paired_tests(
    config: Config, station_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    if not station_frames:
        return pd.DataFrame()
    stations = pd.concat(station_frames, ignore_index=True).drop_duplicates(
        subset=["variant", "pollutant", "station"], keep="last"
    )
    rows: list[dict] = []
    for pollutant in sorted(stations["pollutant"].unique()):
        part = stations[stations["pollutant"] == pollutant]
        reference = part[part["variant"] == "D3"]
        if reference.empty:
            continue
        for candidate_code in sorted(set(part["variant"].astype(str)) - {"D3"}):
            candidate = part[part["variant"] == candidate_code]
            for metric in ("MAE", "RMSE", "R2"):
                paired = reference[["station", metric]].merge(
                    candidate[["station", metric]],
                    on="station",
                    suffixes=("_D3", "_candidate"),
                ).dropna()
                if len(paired) < 2:
                    continue
                delta = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_D3"]
                ).to_numpy(dtype=np.float64)
                if np.allclose(delta, 0.0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                ci_low, ci_high = engine.paired_bootstrap_ci(
                    delta,
                    config.PAIR_BOOTSTRAP_REPEATS,
                    engine.fold_seed(
                        pollutant,
                        sum(ord(char) for char in candidate_code + metric),
                    ),
                )
                if metric in {"MAE", "RMSE"}:
                    d3_better_rate = float(np.mean(delta > 0))
                    oriented = delta
                    relative_change = float(
                        100.0
                        * delta.mean()
                        / max(abs(float(paired[f"{metric}_D3"].mean())), 1e-12)
                    )
                else:
                    d3_better_rate = float(np.mean(delta < 0))
                    oriented = -delta
                    relative_change = float("nan")
                rows.append(
                    {
                        "pollutant": pollutant,
                        "reference": "D3",
                        "candidate": candidate_code,
                        "family": (
                            "D3_confirmatory_ablation"
                            if candidate_code in ABLATIONS
                            else "existing_model"
                        ),
                        "metric": metric,
                        "n_stations": len(paired),
                        "mean_D3": float(paired[f"{metric}_D3"].mean()),
                        "mean_candidate": float(paired[f"{metric}_candidate"].mean()),
                        "mean_candidate_minus_D3": float(delta.mean()),
                        "mean_delta_CI_low": ci_low,
                        "mean_delta_CI_high": ci_high,
                        "median_candidate_minus_D3": float(np.median(delta)),
                        "relative_candidate_change_vs_D3_percent": relative_change,
                        "D3_better_station_rate": d3_better_rate,
                        "rank_biserial_D3_advantage": engine.paired_rank_biserial(oriented),
                        "wilcoxon_p": p_value,
                    }
                )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["holm_global_p"] = engine.holm_adjust(tests["wilcoxon_p"].tolist())
    tests["holm_within_pollutant_metric_p"] = np.nan
    for indices in tests.groupby(["pollutant", "metric"]).groups.values():
        index_list = list(indices)
        adjusted = engine.holm_adjust(
            [float(tests.loc[index, "wilcoxon_p"]) for index in index_list]
        )
        tests.loc[index_list, "holm_within_pollutant_metric_p"] = adjusted
    tests["holm_within_test_family_p"] = np.nan
    for indices in tests.groupby(["pollutant", "metric", "family"]).groups.values():
        index_list = list(indices)
        adjusted = engine.holm_adjust(
            [float(tests.loc[index, "wilcoxon_p"]) for index in index_list]
        )
        tests.loc[index_list, "holm_within_test_family_p"] = adjusted
    tests["significant_global_0_05"] = tests["holm_global_p"] < 0.05
    tests["significant_within_pollutant_metric_0_05"] = (
        tests["holm_within_pollutant_metric_p"] < 0.05
    )
    tests["significant_within_test_family_0_05"] = (
        tests["holm_within_test_family_p"] < 0.05
    )
    return tests


def write_cross_variant_outputs(config: Config) -> None:
    if config.DEFER_SUMMARY and not config.SUMMARY_ONLY:
        return
    root = Path(config.ABLATION_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    aggregate_frames, station_frames = read_available_results(config)
    if not aggregate_frames:
        print("[warning] no aggregate result files were found")
        return
    aggregate = pd.concat(aggregate_frames, ignore_index=True).drop_duplicates(
        subset=["variant", "pollutant"], keep="last"
    )
    order = {
        code: index
        for index, code in enumerate(("F0", "F1", "F2", "F3", "D3", *ABLATIONS))
    }
    aggregate["_variant_order"] = aggregate["variant"].map(order).fillna(999)
    aggregate = aggregate.sort_values(
        ["pollutant", "_variant_order", "variant"]
    ).drop(columns="_variant_order")
    aggregate.to_csv(root / "all_d3_centered_ablation_aggregate_metrics.csv", index=False)
    for pollutant, frame in aggregate.groupby("pollutant", sort=True):
        frame.to_csv(root / f"{pollutant}_d3_centered_ablation_comparison.csv", index=False)

    expected = (
        aggregate[aggregate["variant"] == "D3"]
        .set_index("pollutant")["n_stations"]
        .to_dict()
    )
    completion = aggregate[["variant", "pollutant", "n_stations"]].copy()
    completion["expected_stations"] = completion.apply(
        lambda row: expected.get(row["pollutant"], np.nan), axis=1
    )
    completion["complete"] = completion["n_stations"] == completion["expected_stations"]
    completion.to_csv(root / "d3_ablation_completion_status.csv", index=False)

    paper_rows = aggregate[aggregate["variant"].isin({"F3", "D3", *ABLATIONS})].copy()
    reference = aggregate[aggregate["variant"] == "D3"][[
        "pollutant", "pooled_MAE", "pooled_RMSE", "pooled_R2"
    ]].rename(
        columns={
            "pooled_MAE": "D3_pooled_MAE",
            "pooled_RMSE": "D3_pooled_RMSE",
            "pooled_R2": "D3_pooled_R2",
        }
    )
    if not reference.empty and not paper_rows.empty:
        paper_rows = paper_rows.merge(reference, on="pollutant", how="inner")
        for metric in ("MAE", "RMSE", "R2"):
            paper_rows[f"delta_pooled_{metric}_vs_D3"] = (
                paper_rows[f"pooled_{metric}"] - paper_rows[f"D3_pooled_{metric}"]
            )
        for metric in ("MAE", "RMSE"):
            denominator = paper_rows[f"D3_pooled_{metric}"].abs().clip(lower=1e-12)
            paper_rows[f"relative_pooled_{metric}_change_vs_D3_percent"] = (
                100.0 * paper_rows[f"delta_pooled_{metric}_vs_D3"] / denominator
            )
        paper_rows.to_csv(root / "paper_table_d3_direct_ablation.csv", index=False)

    tests = build_paired_tests(config, station_frames)
    if not tests.empty:
        tests.to_csv(root / "paired_tests_vs_D3.csv", index=False)


def write_run_manifest(config: Config, selected: list[D3Ablation]) -> None:
    root = Path(config.ABLATION_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": config.MODEL_VERSION,
        "source_sha256": SOURCE_SHA256,
        "seed": GLOBAL_SEED,
        "reference_model": "D3",
        "available_ablations": {code: asdict(item) for code, item in ABLATIONS.items()},
        "presets": PRESETS,
        "selected_ablations": [asdict(item) for item in selected],
        "pollutants": config.POLLUTANTS,
        "fold_start": config.FOLD_START,
        "max_stations": config.MAX_STATIONS,
        "d3_reference_root": str(Path(config.D3_REFERENCE_ROOT).resolve()),
        "fusion_reference_root": str(Path(config.REFERENCE_ROOT).resolve()),
        "result_root": str(root.resolve()),
        "summary_only": config.SUMMARY_ONLY,
        "defer_summary": config.DEFER_SUMMARY,
        "reuse_completed": config.REUSE_CHECKPOINTS,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def install_engine_overrides() -> None:
    engine.GLOBAL_SEED = GLOBAL_SEED
    engine.SOURCE_SHA256 = SOURCE_SHA256
    core.GLOBAL_SEED = GLOBAL_SEED
    fusion.GLOBAL_SEED = GLOBAL_SEED
    engine.Config = Config
    engine.DirectAblation = D3Ablation
    engine.ABLATIONS = ABLATIONS
    engine.selected_ablations = selected_ablations
    engine.fusion_variant = fusion_variant
    engine.DirectFusionSCUNet = D3CenteredFusionSCUNet
    engine.protocol_payload = protocol_payload
    engine.validate_reference_protocol = validate_reference_protocol
    engine.read_available_results = read_available_results
    engine.build_paired_tests = build_paired_tests
    engine.write_cross_variant_outputs = write_cross_variant_outputs
    engine.write_run_manifest = write_run_manifest


def main() -> None:
    install_engine_overrides()
    engine.seed_everything(GLOBAL_SEED)
    engine.main()


if __name__ == "__main__":
    main()
