"""CPU-only logic tests; no PyTorch installation is required."""
from __future__ import annotations

import random
import numpy as np

from hybrid_masking import (
    balanced_mask_kind,
    deterministic_validation_seed,
    select_masked_stations,
    validate_spatial_partition,
)


def main() -> None:
    positions = np.asarray(
        [(row, col) for row in range(5) for col in range(5)], dtype=np.int64
    )
    heldout = (0, 6, 12)
    candidates = np.asarray(
        [index for index in range(len(positions)) if index not in heldout], dtype=np.int64
    )

    kinds0 = [balanced_mask_kind(index, 0) for index in range(11)]
    kinds1 = [balanced_mask_kind(index, 1) for index in range(11)]
    assert abs(kinds0.count("random") - kinds0.count("block")) <= 1
    assert all(left != right for left, right in zip(kinds0, kinds1))

    seed = deterministic_validation_seed(42, 19, "no2", heldout, "block")
    selected1 = select_masked_stations(
        candidates, positions, 6, "block", rng=np.random.default_rng(seed)
    )
    selected2 = select_masked_stations(
        candidates, positions, 6, "block", rng=np.random.default_rng(seed)
    )
    assert selected1 == selected2
    assert len(selected1) == 6 and not set(selected1).intersection(heldout)

    # The selected block must equal the nearest candidates around its center.
    possible_centers = []
    for center in selected1:
        distance = np.square(positions[candidates] - positions[center]).sum(axis=1)
        order = np.lexsort((candidates, distance))
        if candidates[order[:6]].astype(int).tolist() == selected1:
            possible_centers.append(center)
    assert possible_centers

    random.seed(7)
    random_selection = select_masked_stations(candidates, positions, 5, "random")
    assert len(random_selection) == len(set(random_selection)) == 5
    assert not set(random_selection).intersection(heldout)

    parts = validate_spatial_partition([[0, 2], [1, 3, 4]], 5)
    assert parts == [[0, 2], [1, 3, 4]]
    try:
        validate_spatial_partition([[0, 1], [1, 2]], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping partition was accepted")
    print("PASS: balanced, deterministic, contiguous and leakage-safe masking logic")


if __name__ == "__main__":
    main()
