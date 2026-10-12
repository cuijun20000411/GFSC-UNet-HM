#!/usr/bin/env python3
"""Run GFSC-UNet-RM for seeds 43 and 44 under the fixed five spatial folds.

This launcher deliberately calls the original, audited
``d3_spatial_block_final_strict.py`` instead of reimplementing training.  It
therefore changes only the training seed and output directory.  Existing
completed folds are reused by the original program after fingerprint checks.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


EXPECTED_HASHES = {
    "d3_spatial_block_final_strict.py": "e9dd95a4ecdfab2cca5fe0fe6570c2f010ef77b354e8c8d4fecfde9596a08be8",
    "d3_validation_common.py": "5010f2885abe201e1f2a9a934f256dff314298db7be1ed13b5364b723f29a7fd",
    "fusion_final_strict.py": "24c25be66e808f59bd2b3e059b22a88f6bf3387ef425fba55ecc316e2aed89ee",
    "rh4_final_strict_light.py": "bd6ee9fb129db6d0535ed4966cee0c15d0e27bfaaeef1d4a30517de120d3fc05",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique comma-separated list")
    return values


def validate_sources(script: Path) -> None:
    source_dir = script.parent
    required = [script, source_dir / "d3_validation_common.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source file(s): " + ", ".join(missing))

    expected_script_hash = EXPECTED_HASHES["d3_spatial_block_final_strict.py"]
    actual_script_hash = sha256(script)
    if actual_script_hash != expected_script_hash:
        raise RuntimeError(
            f"Unexpected SHA-256 for {script}: {actual_script_hash}; "
            f"expected {expected_script_hash}. Use the audited source file."
        )

    for name in (
        "d3_validation_common.py",
        "fusion_final_strict.py",
        "rh4_final_strict_light.py",
    ):
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise RuntimeError(
                f"Unexpected SHA-256 for {path}: {actual}; expected {EXPECTED_HASHES[name]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("/etc/cj/boundary/d3_spatial_block_final_strict.py"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/etc/cj/boundary/duoyuan/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/etc/cj/boundary/gfsc_unet_rm_multiseed_spatial_block_v1"),
    )
    parser.add_argument("--seeds", type=csv_ints, default=csv_ints("43,44"))
    parser.add_argument("--pollutants", default="no2,o3,pm25")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    args.script = args.script.resolve()
    validate_sources(args.script)
    if any(seed == 42 for seed in args.seeds):
        raise ValueError("Seed 42 is the existing reference run and must not be rerun here")

    for seed in args.seeds:
        output = args.output_root / f"seed_{seed}"
        command = [
            sys.executable,
            "-u",
            str(args.script),
            "--data-root",
            str(args.data_root),
            "--output",
            str(output),
            "--models",
            "D3",
            "--pollutants",
            args.pollutants,
            "--blocks",
            "5",
            "--split-seed",
            "42",
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--log-interval",
            str(args.log_interval),
        ]
        if args.preflight_only:
            command.append("--preflight")
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, cwd=args.script.parent, check=True)

    print("All requested RM spatial-block runs completed.", flush=True)


if __name__ == "__main__":
    main()
