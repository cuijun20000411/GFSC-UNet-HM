#!/usr/bin/env python3
"""Regenerate all final paper figures from D3-HM outputs.

Quantitative figures use strict 55-fold LOSO predictions.  The representative
full-field map is explicitly qualitative because unmonitored grid cells have no
reference concentration.  No model is trained by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


POLLUTANTS = ("no2", "o3", "pm25")
LABELS = {"no2": r"NO$_2$", "o3": r"O$_3$", "pm25": r"PM$_{2.5}$"}


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


def metrics(true, prediction) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - true
    denominator = float(np.square(true - true.mean()).sum())
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "R2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else np.nan,
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(stem.with_suffix("." + extension), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def raw_concentration(folder: Path) -> np.ndarray:
    if (folder / "conc_img_raw.npy").exists():
        return np.load(folder / "conc_img_raw.npy", mmap_mode="r")
    values = np.load(folder / "conc_img_full.npy", mmap_mode="r")
    scaler = joblib.load(folder / "conc_scaler.pkl")
    return np.asarray(values) * float(scaler.scale_[0]) + float(scaler.mean_[0])


def prediction_path(root: Path, pollutant: str) -> Path:
    candidates = [
        root / "predictions" / f"{pollutant}_test_predictions.csv",
        root / f"{pollutant}_test_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {pollutant}_test_predictions.csv under {root}")


def load_and_validate_predictions(args, pollutant: str):
    folder = args.data_root / pollutant
    path = prediction_path(args.d3hm_root, pollutant)
    frame = pd.read_csv(path)
    required = {"station", "time_idx", "true", "prediction"}
    if not required.issubset(frame):
        raise ValueError(f"{path}: missing columns {sorted(required-set(frame.columns))}")
    if "variant" in frame and set(frame.variant.astype(str)) not in ({"D3HM"}, {"D3-HM"}):
        raise ValueError(f"{path}: variant column is not pure D3-HM")
    frame = frame.copy()
    frame["station"] = pd.to_numeric(frame.station, errors="raise").astype(int)
    frame["time_idx"] = pd.to_numeric(frame.time_idx, errors="raise").astype(int)
    frame["true"] = pd.to_numeric(frame.true, errors="raise")
    frame["prediction"] = pd.to_numeric(frame.prediction, errors="raise")
    if frame.duplicated(["station", "time_idx"]).any():
        raise ValueError(f"{path}: duplicate station-time predictions")
    if not np.isfinite(frame[["true", "prediction"]].to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite prediction values")

    positions = np.load(folder / "station_positions.npy").astype(int)
    valid = np.load(folder / "valid_grid_mask.npy").astype(bool)
    if set(frame.station) != set(range(len(positions))) or len(positions) != 55:
        raise ValueError(f"{path}: predictions must cover all 55 stations")
    if valid.shape != (32, 32) or int(valid.sum()) != 999:
        raise ValueError(f"{folder}: expected a 32x32 mask with 999 valid grids")

    original_times = np.load(folder / "valid_time_indices.npy").astype(int)
    split = np.load(folder / "time_splits.npy", allow_pickle=True).item()
    test_indices = np.asarray(split["test"], dtype=int)
    raw = raw_concentration(folder)
    expected = []
    for station, (row, column) in enumerate(positions):
        for array_index in test_indices:
            value = float(raw[array_index, row, column])
            if np.isfinite(value):
                expected.append((station, int(original_times[array_index]), value))
    expected_frame = pd.DataFrame(expected, columns=["station", "time_idx", "expected_true"])
    aligned = expected_frame.merge(frame, on=["station", "time_idx"], how="outer", indicator=True)
    if not (aligned._merge == "both").all():
        raise ValueError(f"{path}: prediction keys do not equal the current strict test set")
    if not np.allclose(aligned.expected_true, aligned.true, rtol=2e-6, atol=2e-5):
        raise ValueError(f"{path}: stored truth differs from the source data")
    frame["error"] = frame.prediction - frame.true
    return frame, positions, valid, str(path)


def full_field_path(root: Path, pollutant: str) -> Path:
    candidates = [
        root / f"{pollutant}_D3HM_full_region_test_fields.npz",
        root / "full_field" / f"{pollutant}_D3HM_full_region_test_fields.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def npz_key(data, candidates: tuple[str, ...], path: Path) -> str:
    hits = [key for key in candidates if key in data.files]
    if len(hits) != 1:
        raise KeyError(f"{path}: expected exactly one of {candidates}; found {hits}")
    return hits[0]


def load_full_field(root: Path, pollutant: str, quantile: float):
    path = full_field_path(root, pollutant)
    data = np.load(path)
    prediction_key = npz_key(data, ("predictions", "predicted_fields", "fields"), path)
    known_key = npz_key(data, ("known_observations", "observations", "known"), path)
    time_key = npz_key(data, ("original_time_indices", "time_indices", "times"), path)
    position_key = npz_key(data, ("station_positions", "positions"), path)
    valid_key = npz_key(data, ("valid_grid_mask", "valid_mask"), path)
    fields = np.asarray(data[prediction_key], dtype=float)
    known = np.asarray(data[known_key], dtype=float)
    times = np.asarray(data[time_key], dtype=int)
    positions = np.asarray(data[position_key], dtype=int)
    valid = np.asarray(data[valid_key], dtype=bool)
    if fields.shape != known.shape or fields.ndim != 3 or len(fields) != len(times):
        raise ValueError(f"{path}: inconsistent full-field array shapes")
    if fields.shape[1:] != (32, 32) or valid.shape != (32, 32):
        raise ValueError(f"{path}: expected 32x32 fields")
    preservation = np.abs(fields - known)
    preservation = preservation[np.isfinite(known)]
    maximum_preservation_error = float(preservation.max()) if preservation.size else np.nan
    if preservation.size and maximum_preservation_error > 1e-4:
        raise ValueError(f"{path}: visible-station hard preservation failed ({maximum_preservation_error})")
    visible_mean = np.nanmean(known, axis=(1, 2))
    target = float(np.nanquantile(visible_mean, quantile))
    selected = int(np.nanargmin(np.abs(visible_mean - target)))
    return {
        "path": path,
        "field": fields[selected],
        "known": known[selected],
        "time": int(times[selected]),
        "positions": positions,
        "valid": valid,
        "selected_index": selected,
        "visible_mean": float(visible_mean[selected]),
        "max_preservation_error": maximum_preservation_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/etc/cj/boundary/duoyuan/data"))
    parser.add_argument(
        "--d3hm-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/loso/D3HM"),
    )
    parser.add_argument(
        "--full-field-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/full_field"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/final_D3HM_figures"),
    )
    parser.add_argument("--pollutants", default="no2,o3,pm25")
    parser.add_argument("--representative-quantile", type=float, default=0.5)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--skip-field", action="store_true")
    args = parser.parse_args()
    args.pollutants = [item.strip().lower() for item in args.pollutants.split(",") if item.strip()]
    args.formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not args.pollutants or set(args.pollutants) - set(POLLUTANTS):
        parser.error("pollutants must come from no2,o3,pm25")
    if not 0 <= args.representative_quantile <= 1 or args.dpi <= 0:
        parser.error("invalid quantile or dpi")
    if set(args.formats) - {"png", "pdf", "svg"} or not args.formats:
        parser.error("formats must come from png,pdf,svg")
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.d3hm_root = args.d3hm_root.resolve()
    args.full_field_root = args.full_field_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    setup_style()
    figures = args.output / "figures"
    metrics_dir = args.output / "metrics"
    payload = {}
    station_frames = []
    aggregate_rows = []
    source_paths = {}
    field_payload = {}
    if not args.skip_field:
        for pollutant in args.pollutants:
            field_payload[pollutant] = load_full_field(
                args.full_field_root, pollutant, args.representative_quantile
            )
    for pollutant in args.pollutants:
        frame, positions, valid, path = load_and_validate_predictions(args, pollutant)
        station_rows = []
        for station, part in frame.groupby("station", sort=True):
            row, column = positions[int(station)]
            station_rows.append(
                {
                    "model": "D3HM",
                    "pollutant": pollutant,
                    "station": int(station),
                    "row": int(row),
                    "col": int(column),
                    "n_test": len(part),
                    **metrics(part.true, part.prediction),
                }
            )
        station_frame = pd.DataFrame(station_rows)
        aggregate_rows.append(
            {
                "model": "D3HM",
                "pollutant": pollutant,
                "n_stations": len(station_frame),
                "n_test_observations": len(frame),
                **metrics(frame.true, frame.prediction),
            }
        )
        station_frames.append(station_frame)
        payload[pollutant] = {"samples": frame, "stations": station_frame, "valid": valid}
        source_paths[pollutant] = path
    station_all = pd.concat(station_frames, ignore_index=True)
    aggregate = pd.DataFrame(aggregate_rows)
    atomic_csv(metrics_dir / "D3HM_station_metrics_for_figures.csv", station_all)
    atomic_csv(metrics_dir / "D3HM_aggregate_metrics_recomputed.csv", aggregate)

    # Observed versus predicted
    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.65 * len(args.pollutants), 3.35), constrained_layout=True)
    for index, (axis, pollutant) in enumerate(zip(np.atleast_1d(axes), args.pollutants)):
        frame = payload[pollutant]["samples"]
        true, prediction = frame.true.to_numpy(), frame.prediction.to_numpy()
        limits = [float(min(true.min(), prediction.min())), float(max(true.max(), prediction.max()))]
        margin = 0.03 * max(limits[1] - limits[0], 1.0)
        limits = [limits[0] - margin, limits[1] + margin]
        axis.scatter(true, prediction, s=7, alpha=0.20, color="#315f8c", edgecolors="none", rasterized=True)
        axis.plot(limits, limits, color="#b04a3f", lw=1.1, ls="--")
        axis.set(xlim=limits, ylim=limits, xlabel=r"Observed concentration ($\mu$g m$^{-3}$)")
        if index == 0:
            axis.set_ylabel(r"Predicted concentration ($\mu$g m$^{-3}$)")
        score = metrics(true, prediction)
        axis.text(
            0.04,
            0.96,
            f"MAE = {score['MAE']:.2f}\nRMSE = {score['RMSE']:.2f}\n$R^2$ = {score['R2']:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            linespacing=1.35,
        )
        axis.set_title(f"({chr(97+index)}) {LABELS[pollutant]}", loc="left")
        axis.grid(color="#d9d9d9", lw=0.45, alpha=0.65)
    save_figure(fig, figures / "Fig_D3HM_observed_vs_predicted", args.formats, args.dpi)

    # Station error map
    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.25), constrained_layout=True)
    background = ListedColormap(["#f1f1ed"])
    for index, (axis, pollutant) in enumerate(zip(np.atleast_1d(axes), args.pollutants)):
        item = payload[pollutant]
        axis.imshow(np.ma.masked_where(~item["valid"], item["valid"].astype(float)), cmap=background, vmin=0, vmax=1, origin="upper")
        scatter = axis.scatter(
            item["stations"].col,
            item["stations"].row,
            c=item["stations"].RMSE,
            cmap="magma",
            s=38,
            edgecolors="black",
            linewidths=0.35,
        )
        axis.set_title(f"({chr(97+index)}) {LABELS[pollutant]}", loc="left")
        axis.set_xlabel("Grid column")
        if index == 0:
            axis.set_ylabel("Grid row")
        axis.set(xlim=(-0.5, 31.5), ylim=(31.5, -0.5), aspect="equal")
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.047, pad=0.03)
        colorbar.set_label(r"Station RMSE ($\mu$g m$^{-3}$)")
    save_figure(fig, figures / "Fig_D3HM_station_error_maps", args.formats, args.dpi)

    # Ranked station RMSE
    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.05), constrained_layout=True)
    for index, (axis, pollutant) in enumerate(zip(np.atleast_1d(axes), args.pollutants)):
        station = payload[pollutant]["stations"].sort_values("RMSE").reset_index(drop=True)
        rank = np.arange(1, len(station) + 1)
        axis.plot(rank, station.RMSE, color="#315f8c", lw=1.2)
        axis.scatter(rank, station.RMSE, s=10, color="#315f8c")
        axis.axhline(station.RMSE.median(), color="#b04a3f", lw=0.9, ls="--")
        axis.set_title(f"({chr(97+index)}) {LABELS[pollutant]}", loc="left")
        axis.set_xlabel("Stations ranked by RMSE")
        if index == 0:
            axis.set_ylabel(r"RMSE ($\mu$g m$^{-3}$)")
        axis.grid(axis="y", color="#d9d9d9", lw=0.45)
    save_figure(fig, figures / "Fig_D3HM_stationwise_RMSE", args.formats, args.dpi)

    field_audit = []
    if not args.skip_field:
        fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.7 * len(args.pollutants), 3.35), constrained_layout=True)
        for index, (axis, pollutant) in enumerate(zip(np.atleast_1d(axes), args.pollutants)):
            item = field_payload[pollutant]
            field = np.where(item["valid"], item["field"], np.nan)
            finite = field[np.isfinite(field)]
            image = axis.imshow(field, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()), origin="upper")
            rows, columns = item["positions"].T
            station_values = item["known"][rows, columns]
            axis.scatter(
                columns,
                rows,
                c=station_values,
                cmap="viridis",
                vmin=float(finite.min()),
                vmax=float(finite.max()),
                s=20,
                edgecolors="white",
                linewidths=0.35,
            )
            axis.set_title(f"({chr(97+index)}) {LABELS[pollutant]}, time {item['time']}", loc="left")
            axis.set_xlabel("Grid column")
            if index == 0:
                axis.set_ylabel("Grid row")
            axis.set(xlim=(-0.5, 31.5), ylim=(31.5, -0.5), aspect="equal")
            bar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
            bar.set_label(r"Concentration ($\mu$g m$^{-3}$)")
            field_audit.append(
                {
                    "pollutant": pollutant,
                    "source": str(item["path"]),
                    "selected_array_index": item["selected_index"],
                    "original_time_idx": item["time"],
                    "selection_rule": f"visible-station mean nearest quantile {args.representative_quantile}",
                    "visible_station_mean": item["visible_mean"],
                    "max_visible_station_preservation_error": item["max_preservation_error"],
                }
            )
        save_figure(fig, figures / "Fig_D3HM_representative_fields", args.formats, args.dpi)
        atomic_csv(metrics_dir / "D3HM_representative_field_audit.csv", pd.DataFrame(field_audit))

    atomic_json(
        args.output / "figure_protocol.json",
        {
            "model": "D3HM",
            "quantitative_evidence": "strict 55-fold LOSO at held-out stations",
            "representative_field_interpretation": "qualitative spatial pattern only; unmonitored grids have no reference truth",
            "prediction_sources": source_paths,
            "full_field_sources": {key: str(value["path"]) for key, value in field_payload.items()},
            "figure_formats": args.formats,
            "dpi": args.dpi,
        },
    )
    print(aggregate.to_string(index=False))
    print(f"\nSaved figures: {figures}")


if __name__ == "__main__":
    main()
