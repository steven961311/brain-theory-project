from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Connectivity:
    row_offsets: np.ndarray
    indices: np.ndarray
    n_inputs: int
    n_outputs: int

    def row(self, index: int) -> np.ndarray:
        start = int(self.row_offsets[index])
        end = int(self.row_offsets[index + 1])
        return self.indices[start:end]

    @property
    def n_weights(self) -> int:
        return int(self.indices.size)

    def digest(self) -> str:
        value = hashlib.sha256()
        value.update(np.asarray(self.row_offsets, dtype=np.int64).tobytes())
        value.update(np.asarray(self.indices, dtype=np.int32).tobytes())
        value.update(f"{self.n_inputs}:{self.n_outputs}".encode("ascii"))
        return value.hexdigest()


def part1_connectivity() -> Connectivity:
    rows: list[np.ndarray] = []
    for y in range(28):
        for x in range(28):
            local: list[int] = []
            for yy in range(max(0, y - 2), min(28, y + 3)):
                for xx in range(max(0, x - 2), min(28, x + 3)):
                    local.append(yy * 28 + xx)
            rows.append(np.asarray(local, dtype=np.int32))
    offsets = np.zeros(785, dtype=np.int64)
    offsets[1:] = np.cumsum([row.size for row in rows])
    return Connectivity(
        row_offsets=offsets,
        indices=np.concatenate(rows),
        n_inputs=784,
        n_outputs=784,
    )


def part2_connectivity() -> Connectivity:
    base = part1_connectivity()
    rows = [base.row(i).copy() for i in range(784)]
    all_inputs = np.arange(880, dtype=np.int32)
    rows.extend(all_inputs.copy() for _ in range(96))
    offsets = np.zeros(881, dtype=np.int64)
    offsets[1:] = np.cumsum([row.size for row in rows])
    return Connectivity(
        row_offsets=offsets,
        indices=np.concatenate(rows),
        n_inputs=880,
        n_outputs=880,
    )


def class_readout_connectivity() -> Connectivity:
    """Ten dense image-to-class rows used by the leakage-free Part II."""
    rows = 10
    columns = 784
    return Connectivity(
        row_offsets=np.arange(0, (rows + 1) * columns, columns, dtype=np.int64),
        indices=np.tile(np.arange(columns, dtype=np.int32), rows),
        n_inputs=columns,
        n_outputs=rows,
    )


def template_cleanup_connectivity() -> Connectivity:
    """Dense 96-neuron associative memory for valid digit templates."""
    size = 96
    return Connectivity(
        row_offsets=np.arange(0, (size + 1) * size, size, dtype=np.int64),
        indices=np.tile(np.arange(size, dtype=np.int32), size),
        n_inputs=size,
        n_outputs=size,
    )


def nominal_weight_counts() -> dict[str, int]:
    """Return counts stated in the assignment before clipping boundary windows."""
    return {
        "part1_weights": 784 * 25,
        "part1_thresholds": 784,
        "part2_cross_weights": 784 * 96,
        "part2_template_weights": 96 * 96,
        "part2_template_thresholds": 96,
    }
