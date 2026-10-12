#!/usr/bin/env python3
"""Strict station-level paired tests between D3-HM and saved baselines.

Prediction rows are first aligned on pollutant, held-out station and time index.
The inferential unit is the held-out station (not individual hourly rows), which
avoids temporal pseudoreplication.  Positive RMSE differences and positive
rank-biserial correlations mean that D3-HM is better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


POLLUTANTS = ("no2", "o3", "pm25")
KEYS = ["pollutant", "station", "time_idx"]
COLUMN_ALIASES = {
    "pollutant": ("pollutant", "pollutant_name", "target"),
    "station": ("station", "station_idx", "site", "site_id"),
    "time_idx": ("time_idx", "time", "timestamp_idx", "original_time_idx"),
    "true": ("true", "y_true", "observed", "observation"),
    "prediction": ("prediction", "pred", "y_pred", "predicted"),
}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use MODEL=/absolute/or/relative/path")
    name, raw_path = text.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both MODEL and path are required")
    return name.strip(), Path(raw_path).expanduser().resolve()


def infer_pollutant(path: Path) -> str | None:
    name = path.name.lower().replace("pm2_5", "pm25").replace("pm2.5", "pm25")
    for pollutant in POLLUTANTS:
        if name.startswith(pollutant + "_") or f"_{pollutant}_" in name:
            return pollutant
    return None


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    preferred = [path / "predictions" / "test_predictions.csv", path / "test_predictions.csv"]
    existing_combined = [item for item in preferred if item.exists()]
    if existing_combined:
        return [existing_combined[0]]
    per_pollutant = []
    for pollutant in POLLUTANTS:
        candidates = [
            path / "predictions" / f"{pollutant}_test_predictions.csv",
            path / f"{pollutant}_test_predictions.csv",
        ]
        matches = [item for item in candidates if item.exists()]
        if matches:
            per_pollutant.append(matches[0])
    if per_pollutant:
        return per_pollutant
    raise FileNotFoundError(
        f"No test_predictions.csv or {{pollutant}}_test_predictions.csv under {path}"
    )


def rename_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    rename = {}
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        hits = [lower_to_original[alias] for alias in aliases if alias in lower_to_original]
        if len(hits) > 1:
            raise ValueError(f"{path}: ambiguous columns for {canonical}: {hits}")
        if hits:
            rename[hits[0]] = canonical
    return frame.rename(columns=rename)


def load_predictions(model_name: str, path: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    sources = []
    for file in prediction_files(path):
        raw = pd.read_csv(file)
        frame = rename_columns(raw, file)
        if "model" in frame.columns:
            available = sorted(frame["model"].dropna().astype(str).unique())
            exact = frame["model"].astype(str) == model_name
            if exact.any():
                frame = frame.loc[exact].copy()
            elif len(available) == 1:
                frame = frame.copy()
            else:
                raise ValueError(
                    f"{file}: contains models {available}; none exactly matches '{model_name}'"
                )
        if "pollutant" not in frame:
            inferred = infer_pollutant(file)
            if inferred is None:
                raise ValueError(f"{file}: no pollutant column and filename is not informative")
            frame["pollutant"] = inferred
        required = set(KEYS + ["true", "prediction"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{file}: missing columns {sorted(missing)}")
        frame = frame[KEYS + ["true", "prediction"]].copy()
        frame["pollutant"] = frame.pollutant.astype(str).str.lower().str.replace("pm2.5", "pm25")
        frame["station"] = pd.to_numeric(frame.station, errors="raise").astype(int)
        frame["time_idx"] = pd.to_numeric(frame.time_idx, errors="raise").astype(int)
        frame["true"] = pd.to_numeric(frame.true, errors="raise")
        frame["prediction"] = pd.to_numeric(frame.prediction, errors="raise")
        frames.append(frame)
        sources.append({"file": str(file), "sha256": sha256(file), "rows_loaded": len(frame)})
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(KEYS).any():
        duplicate = result.loc[result.duplicated(KEYS, keep=False), KEYS].head().to_dict("records")
        raise ValueError(f"{model_name}: duplicate prediction keys, e.g. {duplicate}")
    if set(result.pollutant) - set(POLLUTANTS):
        raise ValueError(f"{model_name}: unsupported pollutants {sorted(set(result.pollutant)-set(POLLUTANTS))}")
    if not np.isfinite(result[["true", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_name}: non-finite truth or prediction")
    result["model"] = model_name
    return result.sort_values(KEYS).reset_index(drop=True), sources


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, pollutant, station), part in frame.groupby(["model", "pollutant", "station"], sort=True):
        error = part.prediction.to_numpy(float) - part.true.to_numpy(float)
        denominator = float(np.square(part.true - part.true.mean()).sum())
        rows.append(
            {
                "model": model,
                "pollutant": pollutant,
                "station": int(station),
                "n_test": len(part),
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "R2": float(1.0 - np.square(error).sum() / denominator) if denominator > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-15]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def bootstrap_median_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = rng.integers(0, n, size=(repeats, n))
    medians = np.median(values[draws], axis=1)
    return tuple(float(item) for item in np.quantile(medians, [0.025, 0.975]))


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def align_strict(main: pd.DataFrame, competitor: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    left = main[KEYS + ["true", "prediction"]].rename(
        columns={"true": "true_main", "prediction": "prediction_main"}
    )
    right = competitor[KEYS + ["true", "prediction"]].rename(
        columns={"true": "true_competitor", "prediction": "prediction_competitor"}
    )
    aligned = left.merge(right, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    counts = aligned._merge.value_counts().to_dict()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(
            f"{name}: key sets differ; left_only={counts.get('left_only',0)}, "
            f"right_only={counts.get('right_only',0)}"
        )
    if not np.allclose(aligned.true_main, aligned.true_competitor, rtol=2e-6, atol=2e-5):
        maximum = float(np.max(np.abs(aligned.true_main - aligned.true_competitor)))
        raise ValueError(f"{name}: truth mismatch after key alignment; max abs diff={maximum}")
    audit = {
        "competitor": name,
        "n_aligned_rows": len(aligned),
        "n_pollutants": int(aligned.pollutant.nunique()),
        "n_stations": int(aligned.station.nunique()),
        "max_truth_abs_diff": float(np.max(np.abs(aligned.true_main - aligned.true_competitor))),
        "key_match": True,
    }
    return aligned.drop(columns="_merge"), audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        type=parse_model_path,
        default=parse_model_path(
            "D3HM=/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/loso/D3HM"
        ),
        help="Main model as MODEL=prediction-file-or-result-root.",
    )
    parser.add_argument(
        "--competitor",
        type=parse_model_path,
        action="append",
        required=True,
        help="Repeat for IDW, best ML, best deep baseline and D3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/etc/cj/boundary/d3hm_final_statistical_tests_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1000 or args.tie_tolerance < 0:
        parser.error("bootstrap-repeats must be >=1000 and tie-tolerance >=0")
    names = [name for name, _ in args.competitor]
    if len(names) != len(set(names)):
        parser.error("competitor names must be unique")
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    main_name, main_path = args.main
    main_frame, main_sources = load_predictions(main_name, main_path)
    all_frames = [main_frame]
    audits = []
    sources = {main_name: main_sources}
    aligned_by_competitor = {}
    for competitor_name, competitor_path in args.competitor:
        competitor, competitor_sources = load_predictions(competitor_name, competitor_path)
        aligned, audit = align_strict(main_frame, competitor, competitor_name)
        aligned_by_competitor[competitor_name] = aligned
        all_frames.append(competitor)
        audits.append(audit)
        sources[competitor_name] = competitor_sources

    station = station_metrics(pd.concat(all_frames, ignore_index=True))
    rows = []
    for competitor_name, aligned in aligned_by_competitor.items():
        paired_station = station[station.model.isin([main_name, competitor_name])].pivot(
            index=["pollutant", "station"], columns="model", values="RMSE"
        )
        if paired_station.isna().any().any():
            raise ValueError(f"{competitor_name}: missing station-level pair after row alignment")
        for pollutant, part in paired_station.groupby(level="pollutant", sort=True):
            main_rmse = part[main_name].to_numpy(float)
            other_rmse = part[competitor_name].to_numpy(float)
            difference = other_rmse - main_rmse  # positive => D3-HM is better
            if np.all(np.abs(difference) <= args.tie_tolerance):
                statistic, pvalue = 0.0, 1.0
            else:
                test = wilcoxon(difference, alternative="two-sided", zero_method="pratt", method="auto")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            ci_low, ci_high = bootstrap_median_ci(
                difference,
                args.bootstrap_repeats,
                args.bootstrap_seed + 1009 * len(rows),
            )
            rows.append(
                {
                    "main_model": main_name,
                    "competitor": competitor_name,
                    "pollutant": pollutant,
                    "statistical_unit": "held-out station",
                    "n_paired_stations": len(difference),
                    "main_better_stations": int(np.sum(difference > args.tie_tolerance)),
                    "competitor_better_stations": int(np.sum(difference < -args.tie_tolerance)),
                    "tied_stations": int(np.sum(np.abs(difference) <= args.tie_tolerance)),
                    "median_RMSE_competitor_minus_main": float(np.median(difference)),
                    "median_difference_bootstrap_CI95_low": ci_low,
                    "median_difference_bootstrap_CI95_high": ci_high,
                    "mean_RMSE_competitor_minus_main": float(np.mean(difference)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_raw": pvalue,
                    "rank_biserial_correlation": rank_biserial(difference),
                }
            )
    tests = pd.DataFrame(rows)
    tests["wilcoxon_p_holm_all_tests"] = holm_adjust(tests.wilcoxon_p_raw)
    tests["significant_raw_0.05"] = tests.wilcoxon_p_raw < 0.05
    tests["significant_holm_0.05"] = tests.wilcoxon_p_holm_all_tests < 0.05

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "paired_tests_vs_D3HM.csv", tests)
    atomic_csv(args.output / "stationwise_metrics_all_models.csv", station)
    atomic_csv(args.output / "prediction_alignment_audit.csv", pd.DataFrame(audits))
    atomic_json(
        args.output / "paired_test_protocol.json",
        {
            "main_model": main_name,
            "competitors": [name for name, _ in args.competitor],
            "keys": KEYS,
            "strict_key_set_equality": True,
            "truth_tolerance": {"rtol": 2e-6, "atol": 2e-5},
            "statistical_unit": "held-out station",
            "primary_metric": "station RMSE",
            "difference_direction": "competitor minus D3HM; positive favors D3HM",
            "test": "two-sided Wilcoxon signed-rank, Pratt zero handling",
            "effect_size": "matched-pairs rank-biserial correlation; positive favors D3HM",
            "multiplicity": "Holm family-wise adjustment across all reported comparisons",
            "bootstrap": {"repeats": args.bootstrap_repeats, "seed": args.bootstrap_seed},
            "sources": sources,
        },
    )
    print(tests.to_string(index=False), flush=True)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
