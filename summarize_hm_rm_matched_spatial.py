#!/usr/bin/env python3
"""Audit and summarize matched-seed HM versus RM spatial-block results.

The script uses the existing seed-42 runs and the new seed-43/44 runs.  It
checks that all runs contain the same pollutants and five held-out blocks,
verifies the actual station membership of every block, and then reports
paired pooled-RMSE differences.  Confidence intervals use a paired
hierarchical bootstrap over seeds and spatial folds; they are descriptive and
are not added to the manuscript's Holm families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLLUTANTS = ("no2", "o3", "pm25")
SEEDS = (42, 43, 44)


def first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {names} exists under {root}")


def read_split(root: Path) -> tuple[list[list[int]], str]:
    path = first_existing(root, ["spatial_split.json", "spatial_split_used.json"])
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    blocks = [[int(value) for value in block] for block in payload["heldout_by_fold"]]
    signature = str(payload.get("split_signature", ""))
    return blocks, signature


def canonical_model_column(frame: pd.DataFrame) -> pd.Series:
    if "model" in frame:
        return frame["model"].astype(str)
    if "variant" in frame:
        return frame["variant"].astype(str)
    raise ValueError("Metrics file has neither 'model' nor 'variant' column")


def load_run(root: Path, seed: int, kind: str) -> tuple[pd.DataFrame, pd.DataFrame, list[list[int]], str]:
    aggregate_path = first_existing(root, ["metrics/aggregate_metrics.csv"])
    fold_path = first_existing(root, ["metrics/fold_metrics.csv"])
    aggregate = pd.read_csv(aggregate_path)
    folds = pd.read_csv(fold_path)
    target = "D3" if kind == "RM" else "D3HM"
    aggregate = aggregate.loc[canonical_model_column(aggregate) == target].copy()
    folds = folds.loc[canonical_model_column(folds) == target].copy()
    if aggregate.empty or folds.empty:
        raise ValueError(f"{root} does not contain {target} metrics")
    aggregate["seed"] = seed
    folds["seed"] = seed
    aggregate["masking"] = kind
    folds["masking"] = kind
    if set(aggregate["pollutant"]) != set(POLLUTANTS):
        raise ValueError(f"Unexpected pollutants in {aggregate_path}")
    for pollutant in POLLUTANTS:
        part = folds.loc[folds["pollutant"] == pollutant]
        if set(part["fold"].astype(int)) != set(range(5)) or len(part) != 5:
            raise ValueError(f"{fold_path}: {pollutant} does not contain exactly five folds")
    blocks, signature = read_split(root)
    if len(blocks) != 5 or sorted(value for block in blocks for value in block) != list(range(55)):
        raise ValueError(f"Invalid five-block station partition in {root}")
    return aggregate, folds, blocks, signature


def run_roots(args: argparse.Namespace, kind: str) -> dict[int, Path]:
    if kind == "RM":
        return {
            42: args.rm_seed42_root,
            43: args.rm_multiseed_root / "seed_43",
            44: args.rm_multiseed_root / "seed_44",
        }
    return {
        42: args.hm_seed42_root,
        43: args.hm_multiseed_root / "seed_43" / "spatial_block" / "D3HM",
        44: args.hm_multiseed_root / "seed_44" / "spatial_block" / "D3HM",
    }


def paired_hierarchical_bootstrap(
    paired_folds: pd.DataFrame, repeats: int, random_seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | str]] = []
    for pollutant in POLLUTANTS:
        part = paired_folds.loc[paired_folds["pollutant"] == pollutant].copy()
        seed_values = np.asarray(SEEDS)
        differences = np.empty(repeats, dtype=float)
        relative = np.empty(repeats, dtype=float)
        for repeat in range(repeats):
            selected = []
            for seed in rng.choice(seed_values, size=len(seed_values), replace=True):
                seed_part = part.loc[part["seed"] == int(seed)]
                indices = rng.integers(0, len(seed_part), size=len(seed_part))
                selected.append(seed_part.iloc[indices])
            sample = pd.concat(selected, ignore_index=True)
            weights = sample["n_test"].to_numpy(float)
            rm = np.sqrt(np.average(np.square(sample["RM_RMSE"]), weights=weights))
            hm = np.sqrt(np.average(np.square(sample["HM_RMSE"]), weights=weights))
            differences[repeat] = hm - rm
            relative[repeat] = 100.0 * (hm - rm) / rm
        rows.append(
            {
                "pollutant": pollutant,
                "difference_ci_low": float(np.quantile(differences, 0.025)),
                "difference_ci_high": float(np.quantile(differences, 0.975)),
                "relative_change_ci_low_percent": float(np.quantile(relative, 0.025)),
                "relative_change_ci_high_percent": float(np.quantile(relative, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def pollutant_tex(name: str) -> str:
    return {"no2": r"NO$_2$", "o3": r"O$_3$", "pm25": r"PM$_{2.5}$"}[name]


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-seed comparison of random-only and hybrid masking under five-fold contiguous spatial-block holdout. Values are mean $\pm$ sample standard deviation across seeds 42, 43, and 44. Negative differences favour hybrid masking; confidence intervals are obtained by paired hierarchical bootstrap over seeds and spatial folds.}",
        r"\label{tab:matched_seed_masking}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Pollutant & GFSC-UNet-RM RMSE & GFSC-UNet-HM RMSE & HM$-$RM & Relative change (\%) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{pollutant_tex(row.pollutant)} & "
            f"${row.RM_RMSE_mean:.4f}\\pm{row.RM_RMSE_sd:.4f}$ & "
            f"${row.HM_RMSE_mean:.4f}\\pm{row.HM_RMSE_sd:.4f}$ & "
            f"${row.paired_difference_mean:+.4f}$ "
            f"$[{row.difference_ci_low:+.4f},\\,{row.difference_ci_high:+.4f}]$ & "
            f"${row.relative_change_mean_percent:+.2f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rm-seed42-root",
        type=Path,
        default=Path("/etc/cj/boundary/d3_spatial_block_validation_v1"),
    )
    parser.add_argument(
        "--rm-multiseed-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_rm_multiseed_spatial_block_v1"),
    )
    parser.add_argument(
        "--hm-seed42-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/spatial_block/D3HM"),
    )
    parser.add_argument(
        "--hm-multiseed-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3hm_multiseed_spatial_block_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/etc/cj/boundary/hm_rm_matched_seed_spatial_results_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()

    aggregate_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    reference_blocks: list[list[int]] | None = None
    audit_rows = []
    for kind in ("RM", "HM"):
        for seed, root in run_roots(args, kind).items():
            aggregate, folds, blocks, signature = load_run(root, seed, kind)
            if reference_blocks is None:
                reference_blocks = blocks
            elif blocks != reference_blocks:
                raise RuntimeError(f"Spatial blocks in {root} differ from the reference partition")
            aggregate_frames.append(aggregate)
            fold_frames.append(folds)
            audit_rows.append(
                {
                    "masking": kind,
                    "seed": seed,
                    "root": str(root),
                    "split_signature": signature,
                    "partition_matches_reference": True,
                }
            )

    aggregates = pd.concat(aggregate_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    keep_agg = ["masking", "seed", "pollutant", "pooled_MAE", "pooled_RMSE", "pooled_R2"]
    per_seed_long = aggregates[keep_agg].sort_values(["pollutant", "seed", "masking"])
    per_seed = per_seed_long.pivot(
        index=["seed", "pollutant"], columns="masking", values="pooled_RMSE"
    ).reset_index()
    if per_seed[["HM", "RM"]].isna().any().any():
        raise RuntimeError("HM and RM do not contain the same seed-pollutant combinations")
    per_seed["HM_minus_RM_RMSE"] = per_seed["HM"] - per_seed["RM"]
    per_seed["HM_relative_change_percent"] = (
        100.0 * per_seed["HM_minus_RM_RMSE"] / per_seed["RM"]
    )

    fold_keep = ["masking", "seed", "pollutant", "fold", "n_test", "RMSE"]
    paired_folds = folds[fold_keep].pivot(
        index=["seed", "pollutant", "fold", "n_test"], columns="masking", values="RMSE"
    ).reset_index()
    paired_folds = paired_folds.rename(columns={"HM": "HM_RMSE", "RM": "RM_RMSE"})
    if paired_folds[["HM_RMSE", "RM_RMSE"]].isna().any().any() or len(paired_folds) != 45:
        raise RuntimeError("Expected 45 paired seed-pollutant-fold rows")
    paired_folds["HM_minus_RM_RMSE"] = paired_folds["HM_RMSE"] - paired_folds["RM_RMSE"]

    summary = per_seed.groupby("pollutant").agg(
        n_seeds=("seed", "nunique"),
        RM_RMSE_mean=("RM", "mean"),
        RM_RMSE_sd=("RM", "std"),
        HM_RMSE_mean=("HM", "mean"),
        HM_RMSE_sd=("HM", "std"),
        paired_difference_mean=("HM_minus_RM_RMSE", "mean"),
        paired_difference_sd=("HM_minus_RM_RMSE", "std"),
        relative_change_mean_percent=("HM_relative_change_percent", "mean"),
    ).reset_index()
    intervals = paired_hierarchical_bootstrap(
        paired_folds, args.bootstrap_repeats, args.bootstrap_seed
    )
    summary = summary.merge(intervals, on="pollutant", validate="one_to_one")
    summary["pollutant"] = pd.Categorical(summary["pollutant"], POLLUTANTS, ordered=True)
    summary = summary.sort_values("pollutant")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_long.to_csv(args.output_dir / "hm_rm_per_seed_long.csv", index=False)
    per_seed.to_csv(args.output_dir / "hm_rm_paired_seed_metrics.csv", index=False)
    paired_folds.to_csv(args.output_dir / "hm_rm_paired_seed_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "hm_rm_matched_seed_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "spatial_partition_audit.csv", index=False)
    write_latex(summary, args.output_dir / "hm_rm_matched_seed_table.tex")
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
