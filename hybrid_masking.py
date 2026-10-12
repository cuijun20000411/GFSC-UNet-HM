"""Pure-NumPy masking utilities shared by D3-HM training and tests."""
from __future__ import annotations

import random
import numpy as np


def balanced_mask_kind(sample_index: int, epoch: int) -> str:
    """Alternate mask views exactly; each time index sees both across epochs."""
    return "random" if (int(sample_index) + int(epoch)) % 2 == 0 else "block"


def deterministic_validation_seed(
    base_seed: int,
    current_t: int,
    pollutant: str,
    outer_heldout: tuple[int, ...],
    mask_kind: str,
) -> int:
    if mask_kind not in {"random", "block"}:
        raise ValueError(mask_kind)
    pollutant_code = sum((i + 1) * ord(ch) for i, ch in enumerate(pollutant))
    heldout_code = sum((i + 1) * value for i, value in enumerate(outer_heldout))
    kind_code = 104729 if mask_kind == "block" else 0
    return int(
        (int(base_seed) + 1009 * int(current_t) + 9176 * pollutant_code
         + 37 * heldout_code + kind_code) % (2**32)
    )


def select_masked_stations(
    candidates: np.ndarray,
    station_positions: np.ndarray,
    number: int,
    mask_kind: str,
    *,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Select scattered random stations or a center and its nearest neighbors."""
    candidates = np.asarray(candidates, dtype=np.int64)
    positions = np.asarray(station_positions, dtype=np.float64)
    number = min(max(1, int(number)), len(candidates))
    if not len(candidates) or len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidates must be non-empty and unique")
    if candidates.min() < 0 or candidates.max() >= len(positions):
        raise IndexError("candidate station outside station_positions")
    if mask_kind == "random":
        if rng is None:
            return random.sample(candidates.tolist(), number)
        return rng.choice(candidates, size=number, replace=False).astype(int).tolist()
    if mask_kind != "block":
        raise ValueError(mask_kind)
    center = int(random.choice(candidates.tolist()) if rng is None else rng.choice(candidates))
    coordinates = positions[candidates]
    center_coordinate = positions[center]
    squared_distance = np.square(coordinates - center_coordinate).sum(axis=1)
    order = np.lexsort((candidates, squared_distance))
    selected = candidates[order[:number]].astype(int).tolist()
    if center not in selected:
        raise RuntimeError("contiguous mask selection lost its center")
    return selected


def validate_spatial_partition(parts: list[list[int]], station_count: int) -> list[list[int]]:
    clean = [sorted({int(value) for value in part}) for part in parts]
    if len(clean) < 2 or any(not part for part in clean):
        raise ValueError("at least two non-empty spatial blocks are required")
    flat = [value for part in clean for value in part]
    if sorted(flat) != list(range(int(station_count))) or len(flat) != len(set(flat)):
        raise ValueError("spatial blocks must cover every station exactly once")
    return clean
