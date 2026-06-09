import numpy as np

from etam_mnist.connectivity import (
    class_readout_connectivity,
    nominal_weight_counts,
    part1_connectivity,
    part2_connectivity,
    template_cleanup_connectivity,
)


def test_part1_has_clipped_5_by_5_windows_and_self_feedback():
    connectivity = part1_connectivity()
    sizes = np.diff(connectivity.row_offsets)
    assert connectivity.n_inputs == connectivity.n_outputs == 784
    assert sizes.max() == 25
    assert sizes.min() == 9
    assert connectivity.row(0).tolist() == [0, 1, 2, 28, 29, 30, 56, 57, 58]
    assert all(row in connectivity.row(row) for row in range(784))


def test_part2_is_one_way_image_to_template():
    part1 = part1_connectivity()
    part2 = part2_connectivity()
    assert part2.n_inputs == part2.n_outputs == 880
    assert part2.n_weights == part1.n_weights + 96 * 880
    for row in (0, 400, 783):
        np.testing.assert_array_equal(part2.row(row), part1.row(row))
        assert np.all(part2.row(row) < 784)
    np.testing.assert_array_equal(part2.row(784), np.arange(880))
    counts = nominal_weight_counts()
    assert counts["part2_cross_weights"] == 784 * 96
    assert counts["part2_template_weights"] == 96 * 96


def test_leakage_free_part2_components():
    classifier = class_readout_connectivity()
    cleanup = template_cleanup_connectivity()
    assert (classifier.n_inputs, classifier.n_outputs) == (784, 10)
    assert classifier.n_weights == 784 * 10
    assert (cleanup.n_inputs, cleanup.n_outputs) == (96, 96)
    assert cleanup.n_weights == 96 * 96
