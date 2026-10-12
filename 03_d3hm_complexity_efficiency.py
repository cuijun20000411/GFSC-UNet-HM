#!/usr/bin/env python3
"""Profile D3-HM and selected deep baselines under one controlled protocol.

All neural models receive the same single-field input on the same device.  The
script reports parameters, raw checkpoint size, median/P95 latency, throughput,
and (on CUDA) total and incremental peak allocated memory.  Accuracy is not
computed here; use the strict LOSO results for accuracy comparisons.

Baseline syntax (one checkpoint per pollutant):
  --baseline 'ModelName@no2=/path/to/model_script.py|/path/to/checkpoint.pth'
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


POLLUTANTS = ("no2", "o3", "pm25")


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot construct import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)


def parse_mapping(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("NAME and PATH are required")
    return name.strip().lower(), Path(path).expanduser().resolve()


def parse_baseline(text: str) -> tuple[str, str, Path, Path]:
    if "=" not in text or "|" not in text or "@" not in text.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Use ModelName@pollutant=script.py|checkpoint.pth")
    identity, remainder = text.split("=", 1)
    name, pollutant = identity.rsplit("@", 1)
    script, checkpoint = remainder.split("|", 1)
    pollutant = pollutant.strip().lower()
    if not all(item.strip() for item in (name, pollutant, script, checkpoint)):
        raise argparse.ArgumentTypeError("ModelName, pollutant, script and checkpoint are required")
    if pollutant not in POLLUTANTS:
        raise argparse.ArgumentTypeError("Baseline pollutant must be no2, o3 or pm25")
    return (
        name.strip(),
        pollutant,
        Path(script).expanduser().resolve(),
        Path(checkpoint).expanduser().resolve(),
    )


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} is not a state_dict checkpoint")
    if all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def make_d3hm_bundle(args: argparse.Namespace, pollutant: str, checkpoint: Path, device: torch.device):
    module = load_module(args.d3hm_main, f"d3hm_profile_{pollutant}")
    if hasattr(module, "GLOBAL_SEED"):
        module.GLOBAL_SEED = args.seed
    config_args = SimpleNamespace(
        data_root=str(args.data_root),
        d3_reference_root=str(args.d3_reference_root),
        output=str(args.output),
        epochs=150,
        patience=15,
        batch_size=1,
        num_workers=0,
        log_interval=1000,
    )
    if not hasattr(module, "configure"):
        raise AttributeError(f"{args.d3hm_main} does not expose configure()")
    config = module.configure(config_args)
    config.BATCH_SIZE = 1
    config.NUM_WORKERS = 0
    if not hasattr(module, "d3_variant"):
        raise AttributeError(f"{args.d3hm_main} does not expose d3_variant()")
    variant = module.d3_variant(config)
    dataset_class = getattr(module, "HybridMaskFusionDataset", None)
    if dataset_class is None:
        fusion = getattr(module, "fusion", None)
        dataset_class = getattr(fusion, "FusionDataset", None)
    if dataset_class is None:
        raise AttributeError("Cannot find HybridMaskFusionDataset/FusionDataset")
    dataset_signature = inspect.signature(dataset_class)
    dataset_kwargs = {"mode": "test", "variant": variant}
    if "mask_view" in dataset_signature.parameters:
        # Random/block masks are used only during training and validation.
        # Efficiency profiling reproduces ordinary test-time inference.
        dataset_kwargs["mask_view"] = "none"
    if "outer_heldout" in dataset_signature.parameters:
        # HybridMaskFusionDataset expects an iterable because one fold may hold
        # out either one LOSO station or a complete spatial block.  This script
        # profiles the station-00 LOSO checkpoint, hence the one-element tuple.
        dataset = dataset_class(
            config, pollutant, outer_heldout=(0,), **dataset_kwargs
        )
    else:
        # Compatibility with the original single-station FusionDataset.
        dataset = dataset_class(config, pollutant, 0, **dataset_kwargs)

    reference_root = Path(args.d3_reference_root)
    if hasattr(module, "resolve_reference_root"):
        reference_root = module.resolve_reference_root(reference_root)
    groups_result = module.load_groups(reference_root, pollutant, dataset.F_dim, config)
    groups = groups_result[0] if isinstance(groups_result, tuple) else groups_result
    model_class = getattr(module, "GFSCUNetD3", None)
    if model_class is None:
        raise AttributeError(f"{args.d3hm_main} does not expose GFSCUNetD3")
    model = model_class(dataset.base_channels, dataset.F_dim, groups, variant, config).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    sample = dataset[0]

    def sample_tensor(*keys: str) -> torch.Tensor:
        for key in keys:
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise KeyError(
            f"D3-HM dataset sample lacks all aliases {keys}; "
            f"available keys={sorted(sample)}"
        )

    # HybridMaskFusionDataset uses the explicit names on the left; aliases on
    # the right retain compatibility with older dataset implementations.
    tensors = [
        sample_tensor("input"),
        sample_tensor("channel_mask", "observed"),
        sample_tensor("known_values", "known"),
    ]
    tensors = [tensor.unsqueeze(0).to(device) for tensor in tensors]
    valid = sample.get("valid_grid_mask", sample.get("valid"))
    if valid is not None:
        valid = valid.unsqueeze(0).to(device)
    return model.eval(), tuple(tensors), valid


def run_functional_preflight(
    args: argparse.Namespace, device: torch.device
) -> None:
    """Load every requested checkpoint and execute one real forward pass."""
    checked = 0
    for pollutant, checkpoint in args.checkpoint_map.items():
        model, inputs, valid = make_d3hm_bundle(
            args, pollutant, checkpoint, device
        )
        with torch.inference_mode():
            output = call_model(model, inputs, valid)
        if not torch.isfinite(output).all():
            raise FloatingPointError(
                f"D3HM/{pollutant} produced non-finite preflight output"
            )
        print(
            f"PASS forward: D3HM/{pollutant}, "
            f"input={tuple(inputs[0].shape)}, output={tuple(output.shape)}",
            flush=True,
        )
        checked += 1
        del output, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for baseline_name, baseline_pollutant, script, baseline_checkpoint in args.baseline:
            if baseline_pollutant != pollutant:
                continue
            module = load_module(script, f"preflight_{baseline_name}_{pollutant}")
            baseline_model = construct_baseline_model(
                module, baseline_name, inputs[0].shape[1]
            ).to(device)
            baseline_model.load_state_dict(
                load_state_dict(baseline_checkpoint, device), strict=True
            )
            baseline_model.eval()
            with torch.inference_mode():
                output = call_model(baseline_model, inputs, valid)
            if not torch.isfinite(output).all():
                raise FloatingPointError(
                    f"{baseline_name}/{pollutant} produced non-finite "
                    "preflight output"
                )
            print(
                f"PASS forward: {baseline_name}/{pollutant}, "
                f"input={tuple(inputs[0].shape)}, output={tuple(output.shape)}",
                flush=True,
            )
            checked += 1
            del output, baseline_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del inputs, valid
    print(
        f"PASS: {checked} model/pollutant forward checks on {device}.",
        flush=True,
    )


def construct_baseline_model(module, model_name: str, input_channels: int):
    aliases = {
        "PConvUNetCovariate": "PConvUNetCov",
        "PartialConvUNetCov": "PConvUNetCov",
    }
    class_name = aliases.get(model_name, model_name)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise AttributeError(f"Model class '{class_name}' is absent from {module.__file__}")
    try:
        return model_class(input_channels)
    except TypeError as error:
        raise TypeError(
            f"{class_name} must be constructible with input_channels. Original error: {error}"
        ) from error


def call_model(model, inputs: tuple[torch.Tensor, ...], valid: torch.Tensor | None):
    signature = inspect.signature(model.forward)
    positional = [
        item
        for item in signature.parameters.values()
        if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 3:
        return model(*inputs)
    if len(positional) == 4 and valid is not None:
        return model(*inputs, valid)
    raise TypeError(
        f"Unsupported forward signature {signature}; expected (input, observed, known[, valid])"
    )


def profile_forward(
    model,
    inputs: tuple[torch.Tensor, ...],
    valid: torch.Tensor | None,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = call_model(model, inputs, valid)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Model produced non-finite output during warm-up")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_memory = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = call_model(model, inputs, valid)
                end.record()
                end.synchronize()
                durations.append(float(start.elapsed_time(end)))
            peak = int(torch.cuda.max_memory_allocated(device))
            total_peak_mb = peak / (1024**2)
            incremental_peak_mb = max(0, peak - baseline_memory) / (1024**2)
        else:
            durations = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                output = call_model(model, inputs, valid)
                durations.append((time.perf_counter_ns() - start) / 1e6)
            total_peak_mb = np.nan
            incremental_peak_mb = np.nan
    durations_array = np.asarray(durations, dtype=float)
    return {
        "latency_ms_mean": float(durations_array.mean()),
        "latency_ms_std": float(durations_array.std(ddof=1)),
        "latency_ms_median": float(np.median(durations_array)),
        "latency_ms_p95": float(np.quantile(durations_array, 0.95)),
        "fields_per_second_from_median": float(1000.0 / np.median(durations_array)),
        "peak_gpu_memory_mb_total": total_peak_mb,
        "peak_gpu_memory_mb_incremental_forward": incremental_peak_mb,
    }


def model_row(
    model_name: str,
    pollutant: str,
    model,
    checkpoint: Path,
    profile: dict[str, float],
    inputs: tuple[torch.Tensor, ...],
) -> dict:
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return {
        "model": model_name,
        "pollutant": pollutant,
        "batch_size": int(inputs[0].shape[0]),
        "input_shape": "x".join(map(str, inputs[0].shape)),
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "parameter_memory_fp32_mb": parameters * 4 / (1024**2),
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024**2),
        **profile,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3hm-main", type=Path, default=Path("gfsc_unet_d3_hybrid_mask_main.py"))
    parser.add_argument("--data-root", type=Path, default=Path("/etc/cj/boundary/duoyuan/data"))
    parser.add_argument(
        "--d3-reference-root",
        type=Path,
        default=Path("/etc/cj/boundary/f3_direct_ablation_final_strict_v1/D3"),
    )
    parser.add_argument(
        "--d3hm-checkpoint",
        type=parse_mapping,
        action="append",
        required=True,
        help="Repeat: no2=/path/to/a completed D3-HM checkpoint.pth",
    )
    parser.add_argument(
        "--baseline",
        type=parse_baseline,
        action="append",
        default=[],
        help="Repeat: 'ModelName@no2=model_script.py|checkpoint.pth'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/etc/cj/boundary/d3hm_final_efficiency_v1"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    checkpoint_map = dict(args.d3hm_checkpoint)
    if len(checkpoint_map) != len(args.d3hm_checkpoint) or set(checkpoint_map) - set(POLLUTANTS):
        parser.error("D3-HM checkpoint keys must be unique pollutants from no2,o3,pm25")
    if min(args.warmup, args.repeats, args.threads) <= 0:
        parser.error("warmup, repeats and threads must be positive")
    baseline_keys = [(name, pollutant) for name, pollutant, _, _ in args.baseline]
    if len(baseline_keys) != len(set(baseline_keys)):
        parser.error("Each baseline model/pollutant pair must be unique")
    args.checkpoint_map = checkpoint_map
    args.d3hm_main = args.d3hm_main.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    required = [args.d3hm_main, *args.checkpoint_map.values()]
    required.extend(
        path for _, _, script, checkpoint in args.baseline for path in (script, checkpoint)
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    if args.preflight:
        print(
            f"PASS: {len(required)} required files exist; "
            f"selected device={device}.",
            flush=True,
        )
        run_functional_preflight(args, device)
        return

    rows = []
    for pollutant, checkpoint in args.checkpoint_map.items():
        print(f"Profiling D3HM/{pollutant} ...", flush=True)
        model, inputs, valid = make_d3hm_bundle(args, pollutant, checkpoint, device)
        profile = profile_forward(model, inputs, valid, device, args.warmup, args.repeats)
        rows.append(model_row("D3HM", pollutant, model, checkpoint, profile, inputs))
        # Do not keep D3-HM resident while measuring a baseline.  Otherwise the
        # baseline's total peak GPU memory would incorrectly include D3-HM.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for baseline_name, baseline_pollutant, script, baseline_checkpoint in args.baseline:
            if baseline_pollutant != pollutant:
                continue
            print(f"Profiling {baseline_name}/{pollutant} ...", flush=True)
            module = load_module(script, f"eff_{baseline_name}_{pollutant}")
            baseline_model = construct_baseline_model(module, baseline_name, inputs[0].shape[1]).to(device)
            baseline_model.load_state_dict(load_state_dict(baseline_checkpoint, device), strict=True)
            baseline_model.eval()
            baseline_profile = profile_forward(
                baseline_model, inputs, valid, device, args.warmup, args.repeats
            )
            rows.append(
                model_row(
                    baseline_name,
                    pollutant,
                    baseline_model,
                    baseline_checkpoint,
                    baseline_profile,
                    inputs,
                )
            )
            del baseline_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del inputs, valid

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "model_complexity_and_inference_efficiency.csv", frame)
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": args.threads,
        "warmup_runs": args.warmup,
        "timed_runs": args.repeats,
        "batch_size": 1,
        "timing_scope": "forward pass only; data loading and host-device transfer excluded",
        "memory_scope": "PyTorch allocated memory with model resident; incremental forward peak also reported",
        "seed": args.seed,
    }
    atomic_json(args.output / "efficiency_protocol_and_hardware.json", hardware)
    print("\n" + frame.to_string(index=False), flush=True)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
