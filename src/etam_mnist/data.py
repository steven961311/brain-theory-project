from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np
from scipy import ndimage

from .config import AugmentationConfig


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}


@dataclass(frozen=True)
class Dataset:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    split_hash: str


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_mnist(data_dir: str | Path) -> list[Path]:
    target = Path(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename, expected_md5 in MNIST_FILES.items():
        path = target / filename
        if path.exists() and _md5(path) == expected_md5:
            downloaded.append(path)
            continue
        with urlopen(f"{MNIST_BASE_URL}/{filename}", timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
        actual = _md5(path)
        if actual != expected_md5:
            path.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {filename}: {actual}")
        downloaded.append(path)
    return downloaded


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic = struct.unpack(">I", handle.read(4))[0]
        dtype_code = magic >> 8
        dimensions = magic & 0xFF
        if dtype_code != 0x08:
            raise ValueError(f"unsupported IDX dtype in {path}: {dtype_code}")
        shape = tuple(struct.unpack(">I", handle.read(4))[0] for _ in range(dimensions))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"truncated IDX file {path}: expected {expected}, got {data.size}")
    return data.reshape(shape)


def load_raw_mnist(data_dir: str | Path, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_dir)
    if download:
        download_mnist(root)
    images = _read_idx(root / "train-images-idx3-ubyte.gz")
    labels = _read_idx(root / "train-labels-idx1-ubyte.gz")
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise ValueError(f"unexpected MNIST training shapes: {images.shape}, {labels.shape}")
    return images, labels


def bipolarize(images: np.ndarray, threshold: int = 128) -> np.ndarray:
    flat = np.asarray(images).reshape(len(images), -1)
    return np.where(flat >= threshold, 1, -1).astype(np.int8)


def _stratified_indices(
    labels: np.ndarray, size: int, seed: int, excluded: np.ndarray | None = None
) -> np.ndarray:
    labels = np.asarray(labels)
    available = np.ones(labels.shape[0], dtype=bool)
    if excluded is not None:
        available[np.asarray(excluded, dtype=np.int64)] = False
    counts = np.array([np.count_nonzero((labels == c) & available) for c in range(10)])
    if size > counts.sum():
        raise ValueError(f"requested {size} samples, only {counts.sum()} available")

    quotas = counts / counts.sum() * size
    selected_counts = np.floor(quotas).astype(int)
    remainder = size - int(selected_counts.sum())
    order = np.argsort(-(quotas - selected_counts), kind="stable")
    selected_counts[order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for label, count in enumerate(selected_counts):
        candidates = np.flatnonzero((labels == label) & available)
        result.append(rng.choice(candidates, size=int(count), replace=False))
    indices = np.concatenate(result)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def prepare_dataset(
    data_dir: str | Path,
    train_size: int,
    test_size: int,
    seed: int,
    threshold: int = 128,
    download: bool = True,
) -> Dataset:
    images, labels = load_raw_mnist(data_dir, download=download)
    test_idx = _stratified_indices(labels, test_size, seed)
    train_idx = _stratified_indices(labels, train_size, seed + 1, excluded=test_idx)
    digest = hashlib.sha256()
    digest.update(np.sort(train_idx).tobytes())
    digest.update(np.sort(test_idx).tobytes())
    digest.update(str(seed).encode("ascii"))
    return Dataset(
        train_images=bipolarize(images[train_idx], threshold),
        train_labels=labels[train_idx].astype(np.int8),
        test_images=bipolarize(images[test_idx], threshold),
        test_labels=labels[test_idx].astype(np.int8),
        split_hash=digest.hexdigest(),
    )


def _center_fit(image: np.ndarray, shape: tuple[int, int] = (28, 28)) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    src_y0 = max(0, (image.shape[0] - shape[0]) // 2)
    src_x0 = max(0, (image.shape[1] - shape[1]) // 2)
    cropped = image[src_y0 : src_y0 + shape[0], src_x0 : src_x0 + shape[1]]
    dst_y0 = max(0, (shape[0] - cropped.shape[0]) // 2)
    dst_x0 = max(0, (shape[1] - cropped.shape[1]) // 2)
    result[dst_y0 : dst_y0 + cropped.shape[0], dst_x0 : dst_x0 + cropped.shape[1]] = cropped
    return result


def augment_images(
    bipolar_images: np.ndarray, labels: np.ndarray, config: AugmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Create the Cartesian product of configured rotations and scales."""
    if not config.enabled:
        return np.asarray(bipolar_images, dtype=np.int8), np.asarray(labels, dtype=np.int8)
    source = ((np.asarray(bipolar_images).reshape(-1, 28, 28) + 1) / 2).astype(np.float32)
    variants: list[np.ndarray] = []
    variant_labels: list[np.ndarray] = []
    for angle in config.angles:
        for scale in config.scales:
            transformed = np.empty_like(source)
            for index, image in enumerate(source):
                rotated = ndimage.rotate(
                    image, angle, reshape=False, order=1, mode="constant", cval=0.0
                )
                zoomed = ndimage.zoom(rotated, scale, order=1, mode="constant", cval=0.0)
                transformed[index] = _center_fit(zoomed)
            variants.append(np.where(transformed >= 0.5, 1, -1).reshape(len(source), -1))
            variant_labels.append(np.asarray(labels))
    return (
        np.concatenate(variants).astype(np.int8),
        np.concatenate(variant_labels).astype(np.int8),
    )

