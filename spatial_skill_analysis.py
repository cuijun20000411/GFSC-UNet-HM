#!/usr/bin/env python3
"""
严格 LOSO 空间技能分析与空间分块 SpatialMean 补充脚本。

本脚本不训练模型、不加载模型权重，只读取已经保存的测试预测。

主要输出：
1. table_s5_spatial_skill.csv / .tex
2. timestamp_spatial_diagnostics.csv
3. bootstrap_replicates.csv.gz
4. block_rmse_with_spatialmean.csv
5. fig08_spatial_blocks_with_spatialmean.pdf / .png
6. main_text_spatial_skill.tex

推荐调用示例见脚本末尾打印的提示，或运行：
    python spatial_skill_analysis.py --help
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


POLLUTANT_ORDER = ["no2", "o3", "pm25"]
POLLUTANT_TEX = {
    "no2": r"$\mathrm{NO}_2$",
    "o3": r"$\mathrm{O}_3$",
    "pm25": r"$\mathrm{PM}_{2.5}$",
}
MODEL_ORDER = ["GFSC-UNet-HM", "GFSC-UNet-RM", "IDW", "SpatialMean"]


def _normalise_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def canonical_pollutant(value: object) -> str:
    token = _normalise_token(value)
    mapping = {
        "no2": "no2",
        "o3": "o3",
        "pm25": "pm25",
        "pm25ugm3": "pm25",
    }
    if token not in mapping:
        raise ValueError(f"无法识别污染物名称：{value!r}")
    return mapping[token]


def canonical_model(value: object) -> str:
    token = _normalise_token(value)
    mapping = {
        "d3hm": "GFSC-UNet-HM",
        "gfscunethm": "GFSC-UNet-HM",
        "hybridmask": "GFSC-UNet-HM",
        "d3": "GFSC-UNet-RM",
        "gfscunetrm": "GFSC-UNet-RM",
        "randommask": "GFSC-UNet-RM",
        "idw": "IDW",
        "spatialmean": "SpatialMean",
    }
    return mapping.get(token, str(value))


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"缺少必要字段；候选字段为 {list(candidates)}，实际字段为 {list(columns)}")


def standardise_predictions(
    frame: pd.DataFrame,
    source: Path,
    *,
    forced_model: str | None = None,
    raw_model_filter: str | None = None,
) -> pd.DataFrame:
    """把不同实验脚本产生的预测文件统一成固定字段。"""
    df = frame.copy()

    pollutant_col = first_existing(df.columns, ["pollutant", "pollutant_name"])
    station_col = first_existing(df.columns, ["station", "station_id", "site", "site_id"])
    time_col = first_existing(df.columns, ["time_idx", "timestamp", "datetime", "time"])
    true_col = first_existing(df.columns, ["true", "y_true", "observed", "target"])
    pred_col = first_existing(df.columns, ["prediction", "y_pred", "predicted", "pred"])

    model_col = None
    for candidate in ["model", "variant", "method"]:
        if candidate in df.columns:
            model_col = candidate
            break

    if raw_model_filter is not None:
        if model_col is None:
            raise ValueError(f"{source} 没有 model/variant 字段，不能使用 --block-rm-model")
        keep = df[model_col].astype(str).map(_normalise_token) == _normalise_token(raw_model_filter)
        df = df.loc[keep].copy()
        if df.empty:
            raise ValueError(f"{source} 中找不到模型 {raw_model_filter!r}")

    if "hide_ratio" in df.columns:
        df = df.loc[np.isclose(pd.to_numeric(df["hide_ratio"]), 0.0)].copy()
    if "repeat" in df.columns:
        # 0%隐藏率通常只有 repeat=0；显式筛选避免重复计数。
        repeat_values = pd.to_numeric(df["repeat"], errors="coerce")
        if (repeat_values == 0).any():
            df = df.loc[repeat_values == 0].copy()

    if forced_model is not None:
        model = pd.Series(forced_model, index=df.index)
    elif model_col is not None:
        model = df[model_col].map(canonical_model)
    else:
        raise ValueError(f"{source} 没有 model/variant/method 字段，也没有指定 forced_model")

    result = pd.DataFrame(
        {
            "pollutant": df[pollutant_col].map(canonical_pollutant),
            "station": pd.to_numeric(df[station_col], errors="raise").astype(int),
            "time": df[time_col],
            "true": pd.to_numeric(df[true_col], errors="raise").astype(float),
            "prediction": pd.to_numeric(df[pred_col], errors="raise").astype(float),
            "model": model,
            "source": str(source),
        }
    )

    if "fold" in df.columns:
        result["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    elif "block" in df.columns:
        result["fold"] = pd.to_numeric(df["block"], errors="raise").astype(int)

    # 时间字段尽量保留原始整数；若为时间戳，则转换为统一字符串。
    numeric_time = pd.to_numeric(result["time"], errors="coerce")
    if numeric_time.notna().all():
        result["time"] = numeric_time.astype(int)
    else:
        parsed = pd.to_datetime(result["time"], errors="raise")
        result["time"] = parsed

    if result[["true", "prediction"]].isna().any().any():
        raise ValueError(f"{source} 含有缺失的 true 或 prediction")
    return result


def read_prediction_file(
    path: Path,
    *,
    forced_model: str | None = None,
    raw_model_filter: str | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return standardise_predictions(
        pd.read_csv(path),
        path,
        forced_model=forced_model,
        raw_model_filter=raw_model_filter,
    )


def load_loso_predictions(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"{path} 下没有CSV文件")

    frames = [read_prediction_file(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df["model"].isin(["GFSC-UNet-HM", "IDW", "SpatialMean"])].copy()

    required_models = {"GFSC-UNet-HM", "IDW", "SpatialMean"}
    found = set(df["model"].unique())
    if not required_models.issubset(found):
        raise ValueError(f"LOSO预测缺少模型：{sorted(required_models - found)}；现有模型：{sorted(found)}")

    keys = ["pollutant", "model", "station", "time"]
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        # 完全相同的重复记录可以安全去重；冲突记录直接终止。
        check = df.loc[duplicated].groupby(keys).agg(
            n_true=("true", "nunique"), n_pred=("prediction", "nunique")
        )
        if (check[["n_true", "n_pred"]] > 1).any().any():
            raise ValueError("LOSO预测中存在键相同但数值冲突的重复记录")
        df = df.drop_duplicates(keys)

    return df.sort_values(["pollutant", "model", "time", "station"]).reset_index(drop=True)


def validate_loso_keys(df: pd.DataFrame) -> None:
    """确认三个模型使用完全相同的站点、时刻和真实值。"""
    for pollutant, part in df.groupby("pollutant"):
        reference = None
        reference_model = None
        for model, model_part in part.groupby("model"):
            current = model_part.set_index(["station", "time"])["true"].sort_index()
            if reference is None:
                reference = current
                reference_model = model
                continue
            if not current.index.equals(reference.index):
                raise ValueError(f"{pollutant}: {model} 与 {reference_model} 的测试键不一致")
            if not np.allclose(current.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-6):
                raise ValueError(f"{pollutant}: {model} 与 {reference_model} 的真实值不一致")


def validate_loso_spatial_mean(df: pd.DataFrame, tolerance: float = 2e-5) -> float:
    """核验LOSO SpatialMean只使用同一时刻其余站点的观测值。"""
    part = df.loc[df["model"] == "SpatialMean"].copy()
    group = part.groupby(["pollutant", "time"])["true"]
    total = group.transform("sum")
    count = group.transform("count")
    if (count <= 1).any():
        raise ValueError("SpatialMean核验失败：某些时刻少于两个站点")
    expected = (total - part["true"]) / (count - 1)
    max_difference = float(np.max(np.abs(expected - part["prediction"])))
    if max_difference > tolerance:
        raise ValueError(
            "LOSO SpatialMean并非同一时刻其余站点的严格均值："
            f"最大绝对差={max_difference:.6g}"
        )
    return max_difference


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def circular_moving_block_indices(
    n_times: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成长度为 n_times 的24小时循环移动块bootstrap索引。"""
    n_blocks = math.ceil(n_times / block_length)
    starts = rng.integers(0, n_times, size=n_blocks)
    offsets = np.arange(block_length)
    sampled = np.concatenate([(start + offsets) % n_times for start in starts])
    return sampled[:n_times]


def percentile_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def analyse_loso(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    block_hours: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    validate_loso_keys(df)
    summary_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    rng = np.random.default_rng(seed)

    for pollutant in POLLUTANT_ORDER:
        part = df.loc[df["pollutant"] == pollutant].copy()
        if part.empty:
            continue

        # 真实值在模型之间相同，任取一个模型构造逐时刻空间离散度。
        truth = (
            part.loc[part["model"] == "SpatialMean"]
            .pivot(index="time", columns="station", values="true")
            .sort_index()
        )
        if truth.isna().any().any():
            raise ValueError(f"{pollutant}: 逐时刻站点矩阵不完整，不能计算空间相关")
        ordered_times = truth.index.to_list()
        n_times = len(ordered_times)
        spatial_sd = truth.std(axis=1, ddof=0)
        threshold = float(spatial_sd.quantile(0.75))
        thresholds[pollutant] = threshold
        high_flag = (spatial_sd >= threshold).to_numpy()

        by_model: dict[str, dict[str, object]] = {}
        for model in ["SpatialMean", "IDW", "GFSC-UNet-HM"]:
            model_part = part.loc[part["model"] == model].copy()
            model_part["sq_error"] = (model_part["prediction"] - model_part["true"]) ** 2

            # 中心化误差只评价同一时刻的空间形态，不评价城市整体水平偏差。
            model_part["true_centered"] = (
                model_part["true"] - model_part.groupby("time")["true"].transform("mean")
            )
            model_part["pred_centered"] = (
                model_part["prediction"]
                - model_part.groupby("time")["prediction"].transform("mean")
            )
            model_part["centered_sq_error"] = (
                model_part["pred_centered"] - model_part["true_centered"]
            ) ** 2

            per_time = model_part.groupby("time", sort=True).agg(
                sse=("sq_error", "sum"),
                n=("sq_error", "size"),
                centered_sse=("centered_sq_error", "sum"),
            ).reindex(ordered_times)
            if per_time.isna().any().any():
                raise ValueError(f"{pollutant}/{model}: 缺少某些测试时刻")

            pearsons: list[float] = []
            spearmans: list[float] = []
            for time_value, time_part in model_part.groupby("time", sort=True):
                # SpatialMean在逐站LOSO中会因排除目标站而产生机械负相关，故不解释其r。
                if model == "SpatialMean":
                    pearson_value = np.nan
                    spearman_value = np.nan
                else:
                    pearson_value = safe_pearson(
                        time_part["true"].to_numpy(), time_part["prediction"].to_numpy()
                    )
                    spearman_value = safe_spearman(
                        time_part["true"].to_numpy(), time_part["prediction"].to_numpy()
                    )
                pearsons.append(pearson_value)
                spearmans.append(spearman_value)
                time_rows.append(
                    {
                        "pollutant": pollutant,
                        "model": model,
                        "time": time_value,
                        "spatial_sd_observed": float(spatial_sd.loc[time_value]),
                        "high_dispersion": bool(spatial_sd.loc[time_value] >= threshold),
                        "spatial_pearson_r": pearson_value,
                        "spatial_spearman_rho": spearman_value,
                        "rmse": float(np.sqrt(np.mean((time_part["prediction"] - time_part["true"]) ** 2))),
                    }
                )

            by_model[model] = {
                "part": model_part,
                "per_time": per_time,
                "pearsons": np.asarray(pearsons, dtype=float),
                "spearmans": np.asarray(spearmans, dtype=float),
            }

        ref_time = by_model["SpatialMean"]["per_time"]
        ref_mse = float(ref_time["sse"].sum() / ref_time["n"].sum())

        # 同一组24小时移动块同时用于所有模型，保证配对比较。
        bootstrap_indices = [
            circular_moving_block_indices(n_times, block_hours, rng)
            for _ in range(n_bootstrap)
        ]

        for model in ["SpatialMean", "IDW", "GFSC-UNet-HM"]:
            info = by_model[model]
            model_part = info["part"]
            per_time = info["per_time"]
            pearsons = info["pearsons"]
            spearmans = info["spearmans"]

            mse = float(per_time["sse"].sum() / per_time["n"].sum())
            pooled_rmse = math.sqrt(mse)
            centered_rmse = math.sqrt(
                float(per_time["centered_sse"].sum() / per_time["n"].sum())
            )
            skill = 0.0 if model == "SpatialMean" else 1.0 - mse / ref_mse

            high_part = model_part.loc[model_part["time"].isin(spatial_sd.index[high_flag])]
            high_rmse = float(
                np.sqrt(np.mean((high_part["prediction"] - high_part["true"]) ** 2))
            )

            skill_reps: list[float] = []
            high_rmse_reps: list[float] = []
            sse_values = per_time["sse"].to_numpy(float)
            n_values = per_time["n"].to_numpy(float)
            ref_sse_values = ref_time["sse"].to_numpy(float)
            ref_n_values = ref_time["n"].to_numpy(float)

            for replicate, indices in enumerate(bootstrap_indices):
                mse_boot = float(sse_values[indices].sum() / n_values[indices].sum())
                ref_mse_boot = float(
                    ref_sse_values[indices].sum() / ref_n_values[indices].sum()
                )
                skill_boot = 0.0 if model == "SpatialMean" else 1.0 - mse_boot / ref_mse_boot

                high_indices = indices[high_flag[indices]]
                if len(high_indices) > 0:
                    high_mse_boot = float(
                        sse_values[high_indices].sum() / n_values[high_indices].sum()
                    )
                    high_rmse_boot = math.sqrt(high_mse_boot)
                else:
                    high_rmse_boot = np.nan

                skill_reps.append(skill_boot)
                high_rmse_reps.append(high_rmse_boot)
                bootstrap_rows.append(
                    {
                        "replicate": replicate,
                        "pollutant": pollutant,
                        "model": model,
                        "mse_skill_vs_spatialmean": skill_boot,
                        "high_dispersion_rmse": high_rmse_boot,
                    }
                )

            skill_low, skill_high = percentile_ci(skill_reps)
            high_low, high_high = percentile_ci(high_rmse_reps)
            finite_r = pearsons[np.isfinite(pearsons)]
            finite_rho = spearmans[np.isfinite(spearmans)]

            summary_rows.append(
                {
                    "pollutant": pollutant,
                    "model": model,
                    "n_stations": int(model_part["station"].nunique()),
                    "n_timestamps": n_times,
                    "pooled_rmse": pooled_rmse,
                    "centered_spatial_rmse": centered_rmse,
                    "mse_skill_vs_spatialmean": skill,
                    "mse_skill_ci_low": skill_low,
                    "mse_skill_ci_high": skill_high,
                    "median_spatial_pearson_r": float(np.median(finite_r)) if len(finite_r) else np.nan,
                    "spatial_pearson_q1": float(np.quantile(finite_r, 0.25)) if len(finite_r) else np.nan,
                    "spatial_pearson_q3": float(np.quantile(finite_r, 0.75)) if len(finite_r) else np.nan,
                    "positive_spatial_r_fraction": float(np.mean(finite_r > 0)) if len(finite_r) else np.nan,
                    "median_spatial_spearman_rho": float(np.median(finite_rho)) if len(finite_rho) else np.nan,
                    "spatial_sd_q75_threshold": threshold,
                    "n_high_dispersion_timestamps": int(high_flag.sum()),
                    "high_dispersion_rmse": high_rmse,
                    "high_dispersion_rmse_ci_low": high_low,
                    "high_dispersion_rmse_ci_high": high_high,
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(time_rows),
        pd.DataFrame(bootstrap_rows),
        thresholds,
    )


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_table_s5_tex(
    summary: pd.DataFrame,
    path: Path,
    n_bootstrap: int,
    block_hours: int,
    fold_metrics: pd.DataFrame | None = None,
    block_pooled: pd.DataFrame | None = None,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Spatial-pattern reconstruction diagnostics and spatial-block results. In Panel A, MSE skill is calculated relative to SpatialMean; positive values indicate lower mean squared error. High-dispersion timestamps are the upper quartile of the observed cross-station standard deviation for each pollutant. Panel B reports fold-wise and pooled RMSE after adding SpatialMean to the fixed five-fold contiguous spatial hold-out.}",
        r"\label{tab:s5_spatial_skill}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\textit{Panel A: strict LOSO-CV spatial-pattern diagnostics}\par\smallskip",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Pollutant & Method & RMSE & MSE skill (\%, 95\% CI) & Median spatial $r$ (IQR) & High-dispersion RMSE (95\% CI) \\",
        r"\midrule",
    ]

    for pollutant in POLLUTANT_ORDER:
        block = summary.loc[summary["pollutant"] == pollutant]
        if block.empty:
            continue
        for model in ["SpatialMean", "IDW", "GFSC-UNet-HM"]:
            row = block.loc[block["model"] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            if model == "SpatialMean":
                skill_text = "Reference"
                r_text = "--"
            else:
                skill_text = (
                    f"{100*row['mse_skill_vs_spatialmean']:.2f} "
                    f"({100*row['mse_skill_ci_low']:.2f}, {100*row['mse_skill_ci_high']:.2f})"
                )
                r_text = (
                    f"{row['median_spatial_pearson_r']:.3f} "
                    f"({row['spatial_pearson_q1']:.3f}, {row['spatial_pearson_q3']:.3f})"
                )
            high_text = (
                f"{row['high_dispersion_rmse']:.3f} "
                f"({row['high_dispersion_rmse_ci_low']:.3f}, "
                f"{row['high_dispersion_rmse_ci_high']:.3f})"
            )
            lines.append(
                f"{POLLUTANT_TEX[pollutant]} & {tex_escape(model)} & "
                f"{row['pooled_rmse']:.3f} & {skill_text} & {r_text} & {high_text} \\\\"
            )
        lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    if fold_metrics is not None and block_pooled is not None:
        lines.extend(
            [
                r"\par\medskip",
                r"\textit{Panel B: fixed five-fold contiguous spatial hold-out}\par\smallskip",
                r"\begin{tabular}{llrrrrrr}",
                r"\toprule",
                r"Pollutant & Method & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Pooled \\",
                r"\midrule",
            ]
        )
        for pollutant in POLLUTANT_ORDER:
            for model in MODEL_ORDER:
                rows = fold_metrics.loc[
                    (fold_metrics["pollutant"] == pollutant)
                    & (fold_metrics["model"] == model)
                ].set_index("fold")
                pooled_row = block_pooled.loc[
                    (block_pooled["pollutant"] == pollutant)
                    & (block_pooled["model"] == model),
                    "rmse",
                ]
                if rows.empty or pooled_row.empty:
                    continue
                fold_text = " & ".join(
                    f"{rows.loc[fold, 'rmse']:.3f}" if fold in rows.index else "--"
                    for fold in range(1, 6)
                )
                lines.append(
                    f"{POLLUTANT_TEX[pollutant]} & {tex_escape(model)} & "
                    f"{fold_text} & {pooled_row.iloc[0]:.3f} \\\\"
                )
            lines.append(r"\addlinespace")
        lines.extend([r"\bottomrule", r"\end{tabular}"])

    lines.extend(
        [
            r"\vspace{2pt}",
            (
                r"\parbox{0.98\textwidth}{\footnotesize "
                f"Confidence intervals were obtained from {n_bootstrap:,} paired circular moving-block "
                f"bootstrap replicates with a block length of {block_hours} h. "
                r"Spatial Pearson correlations were calculated across the 55 out-of-fold station predictions at each test timestamp. "
                r"The correlation of SpatialMean is omitted because target-specific leave-one-station-out means induce a deterministic inverse perturbation across assembled folds. "
                r"For Panel B, SpatialMean at each test timestamp was calculated exclusively from stations outside the held-out fold.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_text(
    summary: pd.DataFrame,
    path: Path,
    block_pooled: pd.DataFrame | None = None,
) -> None:
    values: dict[str, pd.Series] = {}
    for pollutant in POLLUTANT_ORDER:
        row = summary.loc[
            (summary["pollutant"] == pollutant) & (summary["model"] == "GFSC-UNet-HM")
        ]
        if not row.empty:
            values[pollutant] = row.iloc[0]
    if len(values) != 3:
        return

    r_values = ", ".join(f"{values[p]['median_spatial_pearson_r']:.3f}" for p in POLLUTANT_ORDER)
    positive_values = ", ".join(
        f"{100*values[p]['positive_spatial_r_fraction']:.1f}\\%" for p in POLLUTANT_ORDER
    )
    skill_values = ", ".join(
        (
            f"{100*values[p]['mse_skill_vs_spatialmean']:.2f}\\% "
            f"(95\\% CI: {100*values[p]['mse_skill_ci_low']:.2f}\\% to "
            f"{100*values[p]['mse_skill_ci_high']:.2f}\\%)"
        )
        for p in POLLUTANT_ORDER
    )
    high_reductions = []
    for pollutant in POLLUTANT_ORDER:
        reference = summary.loc[
            (summary["pollutant"] == pollutant) & (summary["model"] == "SpatialMean"),
            "high_dispersion_rmse",
        ].iloc[0]
        model = values[pollutant]["high_dispersion_rmse"]
        high_reductions.append(100 * (reference - model) / reference)
    high_values = ", ".join(f"{value:.2f}\\%" for value in high_reductions)

    text = (
        "Although SpatialMean closely approaches GFSC-UNet-HM in pooled RMSE, this baseline "
        "primarily captures the contemporaneous city-wide concentration level and does not explicitly "
        "model interstation spatial contrasts. We therefore evaluated the cross-validated predictions "
        "across the 55 stations at each test timestamp. The median timestamp-wise spatial Pearson "
        f"correlations of GFSC-UNet-HM were {r_values} for $\\mathrm{{NO}}_2$, $\\mathrm{{O}}_3$, "
        "and $\\mathrm{PM}_{2.5}$, respectively, with positive correlations observed at "
        f"{positive_values} of the test timestamps. Its full-sample MSE skills relative to SpatialMean "
        f"were {skill_values}, respectively; only the $\\mathrm{{O}}_3$ interval excluded zero. "
        "During the quartile of timestamps with the greatest cross-station concentration dispersion, "
        "the GFSC-UNet-HM point estimates of RMSE were lower than those of SpatialMean by "
        f"{high_values}, respectively. Complete spatial-skill statistics and 95\\% confidence intervals "
        "from the 24-h moving-block bootstrap are reported in Table~S5 of the Supplementary Material.\n"
    )
    if block_pooled is not None:
        model_values = []
        reference_values = []
        for pollutant in POLLUTANT_ORDER:
            model_values.append(
                block_pooled.loc[
                    (block_pooled["pollutant"] == pollutant)
                    & (block_pooled["model"] == "GFSC-UNet-HM"),
                    "rmse",
                ].iloc[0]
            )
            reference_values.append(
                block_pooled.loc[
                    (block_pooled["pollutant"] == pollutant)
                    & (block_pooled["model"] == "SpatialMean"),
                    "rmse",
                ].iloc[0]
            )
        model_text = ", ".join(f"{value:.3f}" for value in model_values)
        reference_text = ", ".join(f"{value:.3f}" for value in reference_values)
        text += (
            "Under the fixed five-fold contiguous spatial hold-out, pooled GFSC-UNet-HM RMSE values "
            f"were {model_text}, compared with {reference_text} for SpatialMean for "
            "$\\mathrm{NO}_2$, $\\mathrm{O}_3$, and $\\mathrm{PM}_{2.5}$, respectively. "
            "The model therefore improved on the contemporaneous-mean baseline for "
            "$\\mathrm{O}_3$ but not for $\\mathrm{NO}_2$ or $\\mathrm{PM}_{2.5}$ under block extrapolation; "
            "fold-wise values are reported in Table~S5 and Fig.~8.\n"
        )
    path.write_text(text, encoding="utf-8")


def load_block_predictions(args: argparse.Namespace) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    if args.block_hm:
        frames.append(read_prediction_file(args.block_hm, forced_model="GFSC-UNet-HM"))
    if args.block_rm:
        frames.append(
            read_prediction_file(
                args.block_rm,
                forced_model="GFSC-UNet-RM",
                raw_model_filter=args.block_rm_model,
            )
        )
    if args.block_baselines:
        baseline = read_prediction_file(args.block_baselines)
        baseline = baseline.loc[baseline["model"].isin(["IDW", "SpatialMean"])].copy()
        frames.append(baseline)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    if "fold" not in df.columns:
        raise ValueError("空间分块预测必须包含 fold 或 block 字段")

    keys = ["pollutant", "model", "fold", "station", "time"]
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        check = df.loc[duplicated].groupby(keys).agg(
            n_true=("true", "nunique"), n_pred=("prediction", "nunique")
        )
        if (check[["n_true", "n_pred"]] > 1).any().any():
            raise ValueError("空间分块预测存在冲突的重复记录")
        df = df.drop_duplicates(keys)

    expected = {"GFSC-UNet-HM", "GFSC-UNet-RM", "IDW", "SpatialMean"}
    missing = expected - set(df["model"].unique())
    if missing:
        raise ValueError(f"空间分块预测缺少模型：{sorted(missing)}")
    return df


def validate_block_spatial_mean(df: pd.DataFrame, tolerance: float = 2e-5) -> float:
    """核验空间分块SpatialMean只使用当前折之外的可见站点。"""
    reference = (
        df[["pollutant", "station", "time", "fold", "true"]]
        .drop_duplicates(["pollutant", "station", "time", "fold"])
        .copy()
    )
    station_fold = reference[["pollutant", "station", "fold"]].drop_duplicates()
    conflicts = station_fold.groupby(["pollutant", "station"])["fold"].nunique()
    if (conflicts != 1).any():
        raise ValueError("空间分块核验失败：同一站点被分配到多个测试折")

    spatial_mean = df.loc[df["model"] == "SpatialMean"].copy()
    differences: list[np.ndarray] = []
    for (pollutant, fold), rows in spatial_mean.groupby(["pollutant", "fold"]):
        visible_stations = station_fold.loc[
            (station_fold["pollutant"] == pollutant) & (station_fold["fold"] != fold),
            "station",
        ]
        visible = reference.loc[
            (reference["pollutant"] == pollutant)
            & (reference["station"].isin(visible_stations))
        ]
        expected = visible.groupby("time", as_index=False)["true"].mean().rename(
            columns={"true": "expected"}
        )
        checked = rows.merge(expected, on="time", how="left", validate="many_to_one")
        if checked["expected"].isna().any():
            raise ValueError(f"{pollutant} fold={fold}: 缺少可见站点同期观测")
        differences.append(np.abs(checked["prediction"] - checked["expected"]).to_numpy())

    if not differences:
        raise ValueError("空间分块预测中没有SpatialMean")
    max_difference = float(np.max(np.concatenate(differences)))
    if max_difference > tolerance:
        raise ValueError(
            "空间分块SpatialMean并非当前折之外可见站点的严格均值："
            f"最大绝对差={max_difference:.6g}"
        )
    return max_difference


def summarise_block_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pollutant, model, fold), part in df.groupby(["pollutant", "model", "fold"]):
        error = part["prediction"] - part["true"]
        rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "fold": int(fold) + 1,
                "n_stations": int(part["station"].nunique()),
                "n_predictions": int(len(part)),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["pollutant", "model", "fold"]).reset_index(drop=True)


def summarise_block_pooled(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pollutant, model), part in df.groupby(["pollutant", "model"]):
        error = part["prediction"] - part["true"]
        rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "n_predictions": int(len(part)),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return pd.DataFrame(rows).sort_values(["pollutant", "model"]).reset_index(drop=True)


def plot_block_results(
    fold_metrics: pd.DataFrame,
    station_blocks_path: Path | None,
    output_pdf: Path,
    output_png: Path,
) -> None:
    if station_blocks_path is not None:
        stations = pd.read_csv(station_blocks_path)
        station_col = first_existing(stations.columns, ["station", "station_id"])
        block_col = first_existing(stations.columns, ["block", "fold"])
        x_col = first_existing(stations.columns, ["x", "normalized_x", "grid_col"])
        y_col = first_existing(stations.columns, ["y", "normalized_y", "grid_row"])

        fig = plt.figure(figsize=(12.0, 6.8))
        grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.65], hspace=0.38, wspace=0.34)
        map_ax = fig.add_subplot(grid[:, 0])
        result_axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]

        colours = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
        block_offset = 1 if stations[block_col].min() == 0 else 0
        for block in sorted(stations[block_col].unique()):
            current = stations.loc[stations[block_col] == block]
            fold_label = int(block) + block_offset
            map_ax.scatter(
                current[x_col],
                current[y_col],
                s=28,
                color=colours[(fold_label - 1) % len(colours)],
                edgecolor="white",
                linewidth=0.35,
                label=f"Fold {fold_label} ($n={current[station_col].nunique()}$)",
            )
        map_ax.set_title("(a) Fixed five-fold spatial partition", loc="left")
        map_ax.set_xlabel("Normalized easting")
        map_ax.set_ylabel("Normalized northing")
        map_ax.grid(True, color="0.88", linewidth=0.6)
        map_ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    else:
        fig, result_axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)

    styles = {
        "GFSC-UNet-HM": dict(color="0.05", linestyle="-", marker="o"),
        "GFSC-UNet-RM": dict(color="0.35", linestyle="--", marker="s"),
        "IDW": dict(color="0.60", linestyle=":", marker="^"),
        "SpatialMean": dict(color="0.20", linestyle="-.", marker="D"),
    }
    panel_letters = ["b", "c", "d"]
    for axis, pollutant, letter in zip(result_axes, POLLUTANT_ORDER, panel_letters):
        part = fold_metrics.loc[fold_metrics["pollutant"] == pollutant]
        for model in MODEL_ORDER:
            values = part.loc[part["model"] == model].sort_values("fold")
            if values.empty:
                continue
            axis.plot(
                values["fold"],
                values["rmse"],
                label=model,
                linewidth=1.5,
                markersize=4.5,
                markerfacecolor=styles[model]["color"],
                **styles[model],
            )
        axis.set_title(f"({letter}) {POLLUTANT_TEX[pollutant]}", loc="left")
        axis.set_ylabel(r"RMSE ($\mu$g m$^{-3}$)")
        axis.set_xticks(sorted(part["fold"].unique()))
        axis.grid(axis="y", color="0.86", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    result_axes[-1].set_xlabel("Held-out spatial fold")
    result_axes[0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从现有预测文件计算LOSO空间技能、24小时块bootstrap及空间分块SpatialMean。"
    )
    parser.add_argument(
        "--loso",
        type=Path,
        required=True,
        help="LOSO预测CSV，或包含no2/o3/pm25预测CSV的目录。",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出目录。")
    parser.add_argument("--bootstrap", type=int, default=5000, help="bootstrap次数，默认5000。")
    parser.add_argument("--block-hours", type=int, default=24, help="移动块长度，默认24小时。")
    parser.add_argument("--seed", type=int, default=20260923, help="bootstrap随机种子。")

    parser.add_argument("--block-hm", type=Path, help="seed-42 GFSC-UNet-HM空间分块预测CSV。")
    parser.add_argument("--block-rm", type=Path, help="seed-42 GFSC-UNet-RM空间分块预测CSV。")
    parser.add_argument(
        "--block-rm-model",
        default="D3",
        help="--block-rm中RM对应的原始模型名称，默认D3。",
    )
    parser.add_argument(
        "--block-baselines",
        type=Path,
        help="包含IDW和SpatialMean的空间分块预测CSV。",
    )
    parser.add_argument(
        "--station-blocks",
        type=Path,
        help="可选：含station/block/x/y的站点空间折CSV，用于重绘完整图8。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap < 1000:
        raise ValueError("正式结果建议至少使用1000次bootstrap")
    if args.block_hours < 2:
        raise ValueError("块长度至少为2小时")

    args.output.mkdir(parents=True, exist_ok=True)
    loso = load_loso_predictions(args.loso)
    loso_spatialmean_difference = validate_loso_spatial_mean(loso)
    summary, time_diagnostics, bootstrap, thresholds = analyse_loso(
        loso,
        n_bootstrap=args.bootstrap,
        block_hours=args.block_hours,
        seed=args.seed,
    )

    summary.to_csv(args.output / "table_s5_spatial_skill.csv", index=False)
    time_diagnostics.to_csv(args.output / "timestamp_spatial_diagnostics.csv", index=False)
    bootstrap.to_csv(args.output / "bootstrap_replicates.csv.gz", index=False, compression="gzip")

    block = load_block_predictions(args)
    block_spatialmean_difference = None
    fold_metrics = None
    block_pooled = None
    if block is not None:
        block_spatialmean_difference = validate_block_spatial_mean(block)
        fold_metrics = summarise_block_predictions(block)
        block_pooled = summarise_block_pooled(block)
        fold_metrics.to_csv(args.output / "block_rmse_with_spatialmean.csv", index=False)
        block_pooled.to_csv(args.output / "block_pooled_rmse_with_spatialmean.csv", index=False)
        plot_block_results(
            fold_metrics,
            args.station_blocks,
            args.output / "fig08_spatial_blocks_with_spatialmean.pdf",
            args.output / "fig08_spatial_blocks_with_spatialmean.png",
        )

    write_table_s5_tex(
        summary,
        args.output / "table_s5_spatial_skill.tex",
        args.bootstrap,
        args.block_hours,
        fold_metrics,
        block_pooled,
    )
    write_main_text(
        summary,
        args.output / "main_text_spatial_skill.tex",
        block_pooled,
    )

    manifest = [
        "Spatial-skill analysis completed.",
        f"LOSO input: {args.loso}",
        f"Bootstrap replicates: {args.bootstrap}",
        f"Circular moving-block length: {args.block_hours} h",
        f"Random seed: {args.seed}",
        f"LOSO SpatialMean maximum reconstruction difference: {loso_spatialmean_difference:.8g}",
    ]
    if block_spatialmean_difference is not None:
        manifest.append(
            "Spatial-block SpatialMean maximum reconstruction difference: "
            f"{block_spatialmean_difference:.8g}"
        )
    for pollutant, threshold in thresholds.items():
        manifest.append(f"{pollutant} upper-quartile spatial-SD threshold: {threshold:.6f}")
    (args.output / "analysis_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"完成。结果保存在：{args.output.resolve()}")
    print("主结果：table_s5_spatial_skill.csv / .tex")
    if block is not None:
        print("空间分块结果：block_rmse_with_spatialmean.csv 和更新后的图8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
