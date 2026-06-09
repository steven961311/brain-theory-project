import numpy as np

from etam_mnist.config import AugmentationConfig
from etam_mnist.data import _stratified_indices, augment_images, bipolarize


def test_bipolarize_uses_requested_threshold():
    images = np.array([[[0, 127], [128, 255]]], dtype=np.uint8)
    actual = bipolarize(images, threshold=128)
    np.testing.assert_array_equal(actual, [[-1, -1, 1, 1]])


def test_stratified_indices_are_reproducible_and_disjoint():
    labels = np.repeat(np.arange(10), 20)
    test = _stratified_indices(labels, 50, seed=7)
    train = _stratified_indices(labels, 100, seed=8, excluded=test)
    np.testing.assert_array_equal(test, _stratified_indices(labels, 50, seed=7))
    assert len(np.intersect1d(train, test)) == 0
    assert np.bincount(labels[test], minlength=10).tolist() == [5] * 10


def test_augmentation_cartesian_product_shape():
    image = -np.ones((1, 784), dtype=np.int8)
    image[0, 14 * 28 + 14] = 1
    labels = np.array([3], dtype=np.int8)
    config = AugmentationConfig(enabled=True, angles=(0.0, 5.0), scales=(1.0, 0.9))
    augmented, augmented_labels = augment_images(image, labels, config)
    assert augmented.shape == (4, 784)
    assert augmented_labels.tolist() == [3, 3, 3, 3]
    assert set(np.unique(augmented)) <= {-1, 1}

