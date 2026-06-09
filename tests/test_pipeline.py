from pathlib import Path

import numpy as np

from etam_mnist.config import Config
from etam_mnist.data import Dataset
from etam_mnist.evaluation import evaluate_artifacts
from etam_mnist.pipeline import train_part1, train_part2


def synthetic_dataset(samples: int = 20) -> Dataset:
    rng = np.random.default_rng(42)
    images = rng.choice((-1, 1), size=(samples, 784)).astype(np.int8)
    labels = (np.arange(samples) % 10).astype(np.int8)
    return Dataset(
        train_images=images,
        train_labels=labels,
        test_images=images[:10],
        test_labels=labels[:10],
        split_hash="synthetic",
    )


def test_synthetic_end_to_end(tmp_path: Path):
    config = Config(
        artifact_dir=str(tmp_path),
        train_size=20,
        test_size=10,
        max_iterations=50,
        max_recall_steps=3,
        checkpoint_every=200,
        evaluation_limit=2,
    )
    dataset = synthetic_dataset()
    part1 = train_part1(config, dataset=dataset)
    assert part1.trained.all()
    part2 = train_part2(config, dataset=dataset)
    assert part2.trained.all()
    assert (tmp_path / "part2_classifier.npz").exists()
    assert (tmp_path / "part2_cleanup.npz").exists()
    metrics = evaluate_artifacts(config, dataset)
    assert "part1" in metrics
    assert "part2" in metrics
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "part2_confusion.png").exists()
