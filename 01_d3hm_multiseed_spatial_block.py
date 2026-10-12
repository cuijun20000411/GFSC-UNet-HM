#!/usr/bin/env python3
"""Repeat the final D3-HM five-block experiment for additional random seeds.

The original D3-HM training program remains the single source of truth.  This
runner starts it in isolated Python subprocesses after setting its GLOBAL_SEED,
always reuses the same spatial split, and finally reports per-seed metrics and
the mean +/- sample standard deviation across seeds 42, 43 and 44.

No prediction ensembling is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd


POLLUTANTS = ("no2", "o3", "pm25")
KEYS = ["pollutant", "station", "time_idx"]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_dict(true: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    y = np.asarray(true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(np.c_[y, p]).all():
        raise ValueError("Metrics require equal, non-empty, finite arrays")
    residual = p - y
    denominator = float(np.square(y - y.mean()).sum())
    return {
        "MAE": float(np.abs(residual).mean()),
        "RMSE": float(np.sqrt(np.square(residual).mean())),
        "R2": float(1.0 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
    }


def parse_csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers")
    return values


def parse_pollutants(text: str) -> list[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not values or set(values) - set(POLLUTANTS) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Pollutants must be unique entries from no2,o3,pm25")
    return values


def prediction_file(root: Path) -> Path:
    candidates = [
        root / "predictions" / "test_predictions.csv",
        root / "test_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No combined test prediction file under {root}")


def normalize_predictions(root: Path, expected_pollutants: list[str]) -> pd.DataFrame:
    path = prediction_file(root)
    frame = pd.read_csv(path)
    required = set(KEYS + ["true", "prediction"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["pollutant"] = frame["pollutant"].astype(str).str.lower()
    frame["station"] = pd.to_numeric(frame["station"], errors="raise").astype(int)
    frame["time_idx"] = pd.to_numeric(frame["time_idx"], errors="raise").astype(int)
    if set(frame["pollutant"]) != set(expected_pollutants):
        raise ValueError(
            f"{path}: pollutants={sorted(frame.pollutant.unique())}, "
            f"expected={sorted(expected_pollutants)}"
        )
    if frame.duplicated(KEYS).any():
        examples = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head().to_dict("records")
        raise ValueError(f"{path}: duplicate prediction keys, e.g. {examples}")
    if not np.isfinite(frame[["true", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite truth or prediction")
    for pollutant, part in frame.groupby("pollutant"):
        stations = set(part.station)
        if stations != set(range(55)):
            raise ValueError(f"{path}: {pollutant} does not cover exactly stations 0..54")
    return frame.sort_values(KEYS).reset_index(drop=True)


def validate_seed_protocol(root: Path, expected_seed: int) -> None:
    candidates = [root / "run_protocol.json", root.parent.parent / "run_protocol.json"]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"Missing run_protocol.json for seed {expected_seed}: {root}")
    payload = json.loads(existing[0].read_text(encoding="utf-8"))
    observed = payload.get("seed", payload.get("global_seed"))
    if observed is not None and int(observed) != expected_seed:
        raise ValueError(f"{existing[0]} records seed={observed}, expected {expected_seed}")


def run_training_seed(args: argparse.Namespace, seed: int) -> list[str]:
    seed_output = args.output / f"seed_{seed}"
    # The D3 protocol is an immutable architecture/grouping reference produced
    # with seed 42.  Its recorded seed must therefore be checked against 42,
    # while the new D3-HM training itself uses 43/44.  Temporarily switching only
    # the main module's GLOBAL_SEED inside load_groups preserves every original
    # protocol check without contaminating core/fusion training randomness.
    bootstrap = """
import importlib.util
import pathlib
import sys

p = pathlib.Path(sys.argv[1]).resolve()
training_seed = int(sys.argv[2])
reference_protocol_seed = int(sys.argv[3])
sys.path.insert(0, str(p.parent))
spec = importlib.util.spec_from_file_location("d3hm_seed_runner", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.GLOBAL_SEED = training_seed
original_load_groups = module.load_groups

def load_groups_with_fixed_reference_seed(*args, **kwargs):
    previous_seed = module.GLOBAL_SEED
    try:
        module.GLOBAL_SEED = reference_protocol_seed
        return original_load_groups(*args, **kwargs)
    finally:
        module.GLOBAL_SEED = previous_seed

module.load_groups = load_groups_with_fixed_reference_seed
sys.argv = [str(p)] + sys.argv[4:]
module.main()
"""
    command = [
        sys.executable,
        "-c",
        bootstrap,
        str(args.main_script),
        str(seed),
        str(args.reference_seed),
        "--mode",
        "spatial-block",
        "--data-root",
        str(args.data_root),
        "--d3-reference-root",
        str(args.d3_reference_root),
        "--output",
        str(seed_output),
        "--spatial-split-json",
        str(args.spatial_split_json),
        "--pollutants",
        ",".join(args.pollutants),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--fold-start",
        "0",
        "--max-folds",
        "0",
    ]
    print("\n[launch] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=args.main_script.parent)
    return command


def summarize(args: argparse.Namespace, seed_roots: dict[int, Path]) -> None:
    frames: dict[int, pd.DataFrame] = {}
    sources = []
    for seed, root in sorted(seed_roots.items()):
        validate_seed_protocol(root, seed)
        frame = normalize_predictions(root, args.pollutants)
        frames[seed] = frame
        file = prediction_file(root)
        sources.append({"seed": seed, "root": str(root.resolve()), "prediction_sha256": sha256(file)})

    reference_seed = min(frames)
    reference = frames[reference_seed][KEYS + ["true"]].rename(columns={"true": "reference_true"})
    for seed, frame in frames.items():
        aligned = reference.merge(frame[KEYS + ["true"]], on=KEYS, how="outer", indicator=True)
        if not (aligned["_merge"] == "both").all():
            raise ValueError(f"Seed {seed} prediction key set differs from seed {reference_seed}")
        if not np.allclose(aligned.reference_true, aligned.true, rtol=2e-6, atol=2e-5):
            raise ValueError(f"Seed {seed} truth values differ from seed {reference_seed}")

    per_seed_rows = []
    for seed, frame in frames.items():
        for pollutant, part in frame.groupby("pollutant", sort=True):
            pooled = metric_dict(part.true, part.prediction)
            station = []
            for _, station_part in part.groupby("station", sort=True):
                station.append(metric_dict(station_part.true, station_part.prediction))
            station_frame = pd.DataFrame(station)
            per_seed_rows.append(
                {
                    "model": "D3HM",
                    "protocol": "five-fold spatial block",
                    "seed": seed,
                    "pollutant": pollutant,
                    "n_stations": int(part.station.nunique()),
                    "n_test_observations": len(part),
                    **{f"pooled_{key}": value for key, value in pooled.items()},
                    **{f"mean_station_{key}": float(station_frame[key].mean()) for key in ("MAE", "RMSE", "R2")},
                }
            )
    per_seed = pd.DataFrame(per_seed_rows).sort_values(["pollutant", "seed"])
    aggregate_rows = []
    metric_columns = [
        "pooled_MAE",
        "pooled_RMSE",
        "pooled_R2",
        "mean_station_MAE",
        "mean_station_RMSE",
        "mean_station_R2",
    ]
    requested_seeds = sorted(seed_roots)
    for pollutant, part in per_seed.groupby("pollutant", sort=True):
        observed_seeds = sorted(part.seed.astype(int).tolist())
        row = {
            "model": "D3HM",
            "protocol": "five-fold spatial block",
            "pollutant": pollutant,
            "seeds": ",".join(map(str, observed_seeds)),
            "n_seeds": len(observed_seeds),
            "all_requested_seeds_complete": observed_seeds == requested_seeds,
        }
        for column in metric_columns:
            row[column + "_mean"] = float(part[column].mean())
            row[column + "_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else np.nan
        aggregate_rows.append(row)
    mean_std = pd.DataFrame(aggregate_rows)
    metrics_dir = args.output / "metrics"
    atomic_csv(metrics_dir / "d3hm_spatial_block_per_seed_metrics.csv", per_seed)
    atomic_csv(metrics_dir / "d3hm_spatial_block_multiseed_mean_std.csv", mean_std)
    atomic_json(
        args.output / "multiseed_protocol.json",
        {
            "model": "D3HM",
            "evaluation": "five-fold spatial-block holdout",
            "seeds": requested_seeds,
            "reference_seed_reused_read_only": args.reference_seed,
            "reference_protocol_seed": args.reference_seed,
            "seed_separation": (
                "D3 protocol provenance/grouping is validated with the fixed reference seed; "
                "D3-HM initialization, loaders, hybrid masks and group dropout use each training seed"
            ),
            "spatial_split_json": str(args.spatial_split_json.resolve()),
            "spatial_split_sha256": sha256(args.spatial_split_json),
            "aggregation": "metrics are computed independently per seed; reported std uses ddof=1; no ensembling",
            "sources": sources,
        },
    )
    print("\n" + mean_std.to_string(index=False), flush=True)
    print(f"\nSaved: {metrics_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-script", type=Path, default=Path("gfsc_unet_d3_hybrid_mask_main.py"))
    parser.add_argument("--data-root", type=Path, default=Path("/etc/cj/boundary/duoyuan/data"))
    parser.add_argument(
        "--d3-reference-root",
        type=Path,
        default=Path("/etc/cj/boundary/f3_direct_ablation_final_strict_v1/D3"),
    )
    parser.add_argument(
        "--reference-seed-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/spatial_block/D3HM"),
        help="Completed seed-42 D3-HM spatial-block result root (read-only).",
    )
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument(
        "--spatial-split-json",
        type=Path,
        default=Path(
            "/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/"
            "spatial_block/D3HM/spatial_split_used.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3hm_multiseed_spatial_block_v1"),
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("43,44"))
    parser.add_argument("--pollutants", type=parse_pollutants, default=parse_pollutants("no2,o3,pm25"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.main_script = args.main_script.resolve()
    args.output = args.output.resolve()
    args.reference_seed_root = args.reference_seed_root.resolve()
    args.spatial_split_json = args.spatial_split_json.resolve()
    if args.reference_seed < 0 or args.reference_seed in args.seeds:
        parser.error("reference seed must be non-negative and absent from --seeds")
    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.num_workers < 0:
        parser.error("training settings are invalid")
    return args


def main() -> None:
    args = parse_args()
    required = [args.main_script, args.spatial_split_json, prediction_file(args.reference_seed_root)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    if importlib.util.spec_from_file_location("d3hm_preflight", args.main_script) is None:
        raise ImportError(f"Cannot import {args.main_script}")
    validate_seed_protocol(args.reference_seed_root, args.reference_seed)
    normalize_predictions(args.reference_seed_root, args.pollutants)
    if args.preflight:
        print("PASS: main script, reference seed, spatial split and prediction schema are valid.")
        return

    commands = []
    if not args.summary_only:
        for seed in args.seeds:
            commands.append({"seed": seed, "command": run_training_seed(args, seed)})
    seed_roots = {args.reference_seed: args.reference_seed_root}
    seed_roots.update(
        {seed: args.output / f"seed_{seed}" / "spatial_block" / "D3HM" for seed in args.seeds}
    )
    summarize(args, seed_roots)
    if commands:
        atomic_json(args.output / "training_commands.json", {"commands": commands})


if __name__ == "__main__":
    main()
