from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .connectivity import Connectivity, part2_connectivity


@dataclass(frozen=True)
class RowTrainingResult:
    row: int
    iterations: int
    margin: float
    stable: bool
    constant_target: bool = False


@dataclass(frozen=True)
class RecallResult:
    state: np.ndarray
    status: str
    steps: int
    cycle_length: int = 0


class SparseETAM:
    """A row-sparse ETAM network implementing equations (7)-(12)."""

    def __init__(self, connectivity: Connectivity):
        self.connectivity = connectivity
        self.weights = np.zeros(connectivity.n_weights, dtype=np.float32)
        self.thresholds = np.zeros(connectivity.n_outputs, dtype=np.float32)
        self.trained = np.zeros(connectivity.n_outputs, dtype=bool)
        self.training_info: list[dict[str, Any] | None] = [None] * connectivity.n_outputs
        self.metadata: dict[str, Any] = {}

    def row_weights(self, row: int) -> np.ndarray:
        start = int(self.connectivity.row_offsets[row])
        end = int(self.connectivity.row_offsets[row + 1])
        return self.weights[start:end]

    def set_row_weights(self, row: int, values: np.ndarray) -> None:
        target = self.row_weights(row)
        source = np.asarray(values, dtype=np.float32)
        if target.shape != source.shape:
            raise ValueError(f"row {row} expects {target.size} weights, got {source.size}")
        target[:] = source

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("ETAM weight vector has zero or non-finite norm")
        return vector / norm

    @staticmethod
    def _hard_patterns(
        distances: np.ndarray, positive: np.ndarray, negative: np.ndarray
    ) -> tuple[float, int, float, int]:
        positive_indices = np.flatnonzero(positive)
        negative_indices = np.flatnonzero(negative)
        p_local = int(np.argmin(distances[positive_indices]))
        n_local = int(np.argmax(distances[negative_indices]))
        p_index = int(positive_indices[p_local])
        n_index = int(negative_indices[n_local])
        return (
            float(distances[p_index]),
            p_index,
            float(distances[n_index]),
            n_index,
        )

    @classmethod
    def _center_threshold(
        cls,
        x: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, int, int, np.ndarray]:
        distances = x @ weights - threshold
        dp, p_index, dn, n_index = cls._hard_patterns(
            distances, target == 1, target == -1
        )
        threshold += (dp + dn) / 2.0
        centered = distances - (dp + dn) / 2.0
        margin = (dp - dn) / 2.0
        return threshold, margin, p_index, n_index, centered

    def train_row(
        self,
        row: int,
        patterns: np.ndarray,
        targets: np.ndarray,
        alpha: float = 0.005,
        max_iterations: int = 10000,
        tolerance: float = 1.0e-7,
    ) -> RowTrainingResult:
        indices = self.connectivity.row(row)
        x = np.asarray(patterns[:, indices], dtype=np.float32)
        target = np.asarray(targets[:, row], dtype=np.int8)
        if not np.all(np.isin(target, (-1, 1))):
            raise ValueError(f"row {row} targets must be bipolar")

        existing = self.row_weights(row)
        continuing = (
            self.training_info[row] is not None
            and not self.trained[row]
            and np.linalg.norm(existing) > 0.0
        )
        if continuing:
            weights = self._normalize(existing.copy())
            threshold = float(self.thresholds[row])
        else:
            initial = target.astype(np.float32) @ x
            weights = self._normalize(np.asarray(initial, dtype=np.float32))
            threshold = 0.0
        positive = target == 1
        negative = target == -1

        if not positive.any() or not negative.any():
            limit = float(np.sqrt(indices.size) + 1.0)
            threshold = -limit if positive.any() else limit
            result = RowTrainingResult(row, 0, limit, True, constant_target=True)
            self.set_row_weights(row, weights)
            self.thresholds[row] = threshold
            self.trained[row] = True
            self.training_info[row] = asdict(result)
            return result

        threshold, margin, p_index, n_index, centered = self._center_threshold(
            x, target, weights, threshold
        )
        stable = bool(np.all(centered[positive] >= 0.0) and np.all(centered[negative] < 0.0))
        iterations = 0

        while iterations < max_iterations:
            update = (
                target[p_index] * x[p_index] + target[n_index] * x[n_index]
            )
            candidate_weights = self._normalize(weights + alpha * update)
            try:
                (
                    candidate_threshold,
                    candidate_margin,
                    candidate_p,
                    candidate_n,
                    candidate_distances,
                ) = self._center_threshold(
                    x, target, candidate_weights, threshold
                )
            except ValueError:
                break
            candidate_stable = bool(
                np.all(candidate_distances[positive] >= 0.0)
                and np.all(candidate_distances[negative] < 0.0)
            )

            # Before all patterns are stored, follow the ECR-like updates. Once
            # stable, Eq. (12) is accepted only while the minimal margin grows.
            if stable and (
                not candidate_stable or candidate_margin <= margin + tolerance
            ):
                break
            if np.array_equal(candidate_weights, weights):
                break

            weights = candidate_weights
            threshold = candidate_threshold
            margin = candidate_margin
            p_index = candidate_p
            n_index = candidate_n
            centered = candidate_distances
            stable = candidate_stable
            iterations += 1

        previous_iterations = 0
        if continuing and isinstance(self.training_info[row], dict):
            previous_iterations = int(self.training_info[row].get("iterations", 0))
        result = RowTrainingResult(
            row, previous_iterations + iterations, float(margin), stable
        )
        self.set_row_weights(row, weights)
        self.thresholds[row] = threshold
        # An unstable row remains resumable. Its latest weights are still saved
        # for diagnostics and non-strict quick experiments.
        self.trained[row] = stable
        self.training_info[row] = asdict(result)
        return result

    def train(
        self,
        patterns: np.ndarray,
        targets: np.ndarray | None = None,
        rows: Iterable[int] | None = None,
        alpha: float = 0.005,
        max_iterations: int = 10000,
        tolerance: float = 1.0e-7,
        checkpoint_every: int = 0,
        checkpoint: Callable[["SparseETAM", int], None] | None = None,
        progress: Callable[[RowTrainingResult], None] | None = None,
    ) -> list[RowTrainingResult]:
        values = np.asarray(patterns, dtype=np.int8)
        expected = (None, self.connectivity.n_inputs)
        if values.ndim != 2 or values.shape[1] != expected[1]:
            raise ValueError(
                f"patterns must have shape (samples, {self.connectivity.n_inputs})"
            )
        desired = values if targets is None else np.asarray(targets, dtype=np.int8)
        if desired.shape != (values.shape[0], self.connectivity.n_outputs):
            raise ValueError(
                f"targets must have shape ({values.shape[0]}, "
                f"{self.connectivity.n_outputs})"
            )
        selected = range(self.connectivity.n_outputs) if rows is None else rows
        results: list[RowTrainingResult] = []
        completed_since_checkpoint = 0
        for row in selected:
            row = int(row)
            if self.trained[row]:
                continue
            result = self.train_row(
                row,
                values,
                desired,
                alpha=alpha,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            results.append(result)
            completed_since_checkpoint += 1
            if progress is not None:
                progress(result)
            if (
                checkpoint is not None
                and checkpoint_every > 0
                and completed_since_checkpoint >= checkpoint_every
            ):
                checkpoint(self, row)
                completed_since_checkpoint = 0
        if checkpoint is not None and results:
            checkpoint(self, results[-1].row)
        return results

    def initialize_hebbian_readout(
        self, patterns: np.ndarray, targets: np.ndarray
    ) -> None:
        """Apply normalized Eq. (7) without per-row thresholds.

        This is intended for multiclass competition, where independently
        centered one-vs-all thresholds are not comparable across classes.
        """
        values = np.asarray(patterns, dtype=np.float32)
        desired = np.asarray(targets, dtype=np.int8)
        if values.shape != (values.shape[0], self.connectivity.n_inputs):
            raise ValueError("invalid readout pattern shape")
        if desired.shape != (values.shape[0], self.connectivity.n_outputs):
            raise ValueError("invalid readout target shape")
        for row in range(self.connectivity.n_outputs):
            indices = self.connectivity.row(row)
            weights = desired[:, row].astype(np.float32) @ values[:, indices]
            self.set_row_weights(row, self._normalize(weights))
            self.thresholds[row] = 0.0
            self.trained[row] = True
            self.training_info[row] = {
                "row": row,
                "iterations": 0,
                "margin": None,
                "stable": False,
                "constant_target": False,
                "method": "normalized_hebbian_eq7_multiclass",
            }

    def local_fields(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.float32).reshape(self.connectivity.n_inputs)
        result = np.empty(self.connectivity.n_outputs, dtype=np.float32)
        for row in range(self.connectivity.n_outputs):
            result[row] = (
                self.row_weights(row) @ vector[self.connectivity.row(row)]
                - self.thresholds[row]
            )
        return result

    def predict_once(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.float32).reshape(self.connectivity.n_inputs)
        fields = self.local_fields(vector)
        result = np.empty(self.connectivity.n_outputs, dtype=np.int8)
        for row in range(self.connectivity.n_outputs):
            total = float(fields[row])
            if total > 0.0:
                result[row] = 1
            elif total < 0.0:
                result[row] = -1
            else:
                current = vector[row] if row < vector.size else 0
                result[row] = int(current) if current in (-1, 1) else 1
        return result

    def recall(self, initial: np.ndarray, max_steps: int = 100) -> RecallResult:
        state = np.asarray(initial, dtype=np.int8).reshape(self.connectivity.n_inputs).copy()
        seen = {state.tobytes(): 0}
        for step in range(1, max_steps + 1):
            next_state = self.predict_once(state)
            if np.array_equal(next_state, state):
                return RecallResult(next_state, "stable", step)
            key = next_state.tobytes()
            if key in seen:
                return RecallResult(next_state, "cycle", step, step - seen[key])
            seen[key] = step
            state = next_state
        return RecallResult(state, "max_steps", max_steps)

    def stability_violations(
        self,
        patterns: np.ndarray,
        targets: np.ndarray | None = None,
        limit: int = 100,
    ) -> list[tuple[int, int, float]]:
        values = np.asarray(patterns, dtype=np.int8)
        desired = values if targets is None else np.asarray(targets, dtype=np.int8)
        violations: list[tuple[int, int, float]] = []
        for row in range(self.connectivity.n_outputs):
            if not self.trained[row]:
                violations.append((row, -1, float("nan")))
                if len(violations) >= limit:
                    return violations
                continue
            distances = (
                values[:, self.connectivity.row(row)] @ self.row_weights(row)
                - self.thresholds[row]
            )
            bad = np.flatnonzero(
                ((desired[:, row] == 1) & (distances < 0.0))
                | ((desired[:, row] == -1) & (distances >= 0.0))
            )
            for pattern in bad:
                violations.append((row, int(pattern), float(distances[pattern])))
                if len(violations) >= limit:
                    return violations
        return violations

    def expand_to_part2(self) -> "SparseETAM":
        if self.connectivity.n_inputs != 784 or self.connectivity.n_outputs != 784:
            raise ValueError("only a Part I model can be expanded to Part II")
        expanded = SparseETAM(part2_connectivity())
        for row in range(784):
            if not np.array_equal(
                self.connectivity.row(row), expanded.connectivity.row(row)
            ):
                raise ValueError("Part I connectivity does not match Part II image rows")
            expanded.set_row_weights(row, self.row_weights(row))
        expanded.thresholds[:784] = self.thresholds
        expanded.trained[:784] = self.trained
        expanded.training_info[:784] = self.training_info
        expanded.metadata.update(self.metadata)
        return expanded

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined = dict(self.metadata)
        if metadata:
            combined.update(metadata)
        combined["connectivity_hash"] = self.connectivity.digest()
        combined["training_info"] = self.training_info
        encoded = json.dumps(combined, ensure_ascii=True, sort_keys=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            weights=self.weights,
            thresholds=self.thresholds,
            trained=self.trained,
            row_offsets=self.connectivity.row_offsets,
            indices=self.connectivity.indices,
            dimensions=np.asarray(
                [self.connectivity.n_inputs, self.connectivity.n_outputs], dtype=np.int64
            ),
            metadata=np.asarray(encoded),
        )
        temporary.replace(destination)
        self.metadata = combined

    @classmethod
    def load(cls, path: str | Path) -> "SparseETAM":
        with np.load(Path(path), allow_pickle=False) as archive:
            dimensions = archive["dimensions"].astype(np.int64)
            connectivity = Connectivity(
                row_offsets=archive["row_offsets"].astype(np.int64),
                indices=archive["indices"].astype(np.int32),
                n_inputs=int(dimensions[0]),
                n_outputs=int(dimensions[1]),
            )
            model = cls(connectivity)
            model.weights[:] = archive["weights"]
            model.thresholds[:] = archive["thresholds"]
            model.trained[:] = archive["trained"]
            model.metadata = json.loads(str(archive["metadata"]))
        stored_hash = model.metadata.get("connectivity_hash")
        if stored_hash and stored_hash != connectivity.digest():
            raise ValueError("checkpoint connectivity hash mismatch")
        info = model.metadata.get("training_info")
        if isinstance(info, list) and len(info) == connectivity.n_outputs:
            model.training_info = info
        return model
