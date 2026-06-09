from pathlib import Path

import numpy as np

from etam_mnist.connectivity import Connectivity
from etam_mnist.model import SparseETAM


def full_connectivity(size: int) -> Connectivity:
    rows = np.tile(np.arange(size, dtype=np.int32), size)
    offsets = np.arange(0, size * size + 1, size, dtype=np.int64)
    return Connectivity(offsets, rows, size, size)


def test_etam_stores_small_bipolar_patterns():
    patterns = np.array(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=np.int8,
    )
    model = SparseETAM(full_connectivity(2))
    results = model.train(patterns, alpha=0.005, max_iterations=20)
    assert all(result.stable for result in results)
    assert model.stability_violations(patterns) == []
    for row in range(2):
        assert np.isclose(np.linalg.norm(model.row_weights(row)), 1.0)


def test_last_non_improving_update_is_not_accepted():
    patterns = np.array(
        [[1, 1], [1, -1], [-1, 1], [-1, -1]],
        dtype=np.int8,
    )
    model = SparseETAM(full_connectivity(2))
    result = model.train_row(0, patterns, patterns, alpha=0.005, max_iterations=20)
    assert result.stable
    # Hebbian initialization is already optimal for this symmetric pattern set.
    np.testing.assert_allclose(model.row_weights(0), [1.0, 0.0], atol=1e-6)


def test_recall_detects_two_cycle():
    model = SparseETAM(full_connectivity(2))
    model.set_row_weights(0, [0.0, 1.0])
    model.set_row_weights(1, [1.0, 0.0])
    model.trained[:] = True
    result = model.recall(np.array([1, -1], dtype=np.int8), max_steps=5)
    assert result.status == "cycle"
    assert result.cycle_length == 2


def test_checkpoint_round_trip(tmp_path: Path):
    model = SparseETAM(full_connectivity(2))
    model.weights[:] = [1.0, 2.0, 3.0, 4.0]
    model.thresholds[:] = [0.25, -0.5]
    model.trained[:] = [True, False]
    path = tmp_path / "model.npz"
    model.save(path, {"split_hash": "abc", "part": 1})
    loaded = SparseETAM.load(path)
    np.testing.assert_array_equal(loaded.weights, model.weights)
    np.testing.assert_array_equal(loaded.thresholds, model.thresholds)
    np.testing.assert_array_equal(loaded.trained, model.trained)
    assert loaded.metadata["split_hash"] == "abc"


def test_unstable_row_remains_resumable():
    patterns = np.array(
        [[1, 1], [1, -1], [-1, 1], [-1, -1]],
        dtype=np.int8,
    )
    targets = patterns.copy()
    model = SparseETAM(full_connectivity(2))
    result = model.train_row(0, patterns, targets, alpha=0.0, max_iterations=0)
    assert result.stable
    assert model.trained[0]

    contradictory = np.array([[1], [1], [1]], dtype=np.int8)
    contradictory_targets = np.array([[1], [1], [-1]], dtype=np.int8)
    one = SparseETAM(full_connectivity(1))
    result = one.train_row(
        0, contradictory, contradictory_targets, alpha=0.005, max_iterations=0
    )
    assert not result.stable
    assert not one.trained[0]
    first_iterations = result.iterations
    resumed = one.train_row(
        0, contradictory, contradictory_targets, alpha=0.005, max_iterations=1
    )
    assert resumed.iterations >= first_iterations


def test_multiclass_hebbian_readout_has_comparable_zero_thresholds():
    patterns = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    targets = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1]], dtype=np.int8)
    model = SparseETAM(full_connectivity(2))
    model.initialize_hebbian_readout(patterns, targets)
    np.testing.assert_array_equal(model.thresholds, [0.0, 0.0])
    assert model.local_fields([1, 1]).shape == (2,)
