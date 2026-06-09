from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config
from .connectivity import (
    class_readout_connectivity,
    part1_connectivity,
    template_cleanup_connectivity,
)
from .data import Dataset, augment_images, prepare_dataset
from .model import RowTrainingResult, SparseETAM
from .templates import digit_templates


def load_dataset(config: Config) -> Dataset:
    return prepare_dataset(
        config.data_path,
        train_size=config.train_size,
        test_size=config.test_size,
        seed=config.seed,
        threshold=config.threshold,
    )


def augmented_patterns(
    images: np.ndarray, labels: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    return augment_images(images, labels, config.augmentation)


def part2_patterns(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    templates = digit_templates()
    return np.concatenate((images, templates[np.asarray(labels, dtype=np.int64)]), axis=1)


def _checkpoint_metadata(config: Config, dataset: Dataset, part: int) -> dict:
    return {
        "version": 1,
        "part": part,
        "split_hash": dataset.split_hash,
        "config": config.to_dict(),
    }


def _verify_checkpoint(model: SparseETAM, dataset: Dataset, part: int) -> None:
    stored_hash = model.metadata.get("split_hash")
    if stored_hash and stored_hash != dataset.split_hash:
        raise ValueError(
            "checkpoint uses a different dataset split; remove it or use matching config"
        )
    stored_part = model.metadata.get("part")
    if stored_part and int(stored_part) != part:
        raise ValueError(f"expected Part {part} checkpoint, found Part {stored_part}")


def _write_training_results(path: Path, rows: Iterable[RowTrainingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row", "iterations", "margin", "stable", "constant_target"],
        )
        if not existing:
            writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "row": result.row,
                    "iterations": result.iterations,
                    "margin": result.margin,
                    "stable": result.stable,
                    "constant_target": result.constant_target,
                }
            )


def _progress(result: RowTrainingResult) -> None:
    if result.row % 25 == 0 or not result.stable:
        print(
            f"row={result.row:3d} iterations={result.iterations:5d} "
            f"margin={result.margin:.6f} stable={result.stable}",
            flush=True,
        )


def _validate_or_raise(
    model: SparseETAM,
    patterns: np.ndarray,
    config: Config,
    checkpoint_path: Path,
    metadata: dict,
) -> None:
    if not config.strict_stability:
        return
    violations = model.stability_violations(patterns, limit=100)
    if violations:
        model.save(checkpoint_path, metadata)
        preview = ", ".join(
            f"(neuron={row}, pattern={pattern}, distance={distance:.6g})"
            for row, pattern, distance in violations[:10]
        )
        raise RuntimeError(
            f"training patterns are not all stable; first violations: {preview}"
        )


def train_part1(
    config: Config,
    dataset: Dataset | None = None,
    resume: bool = False,
) -> SparseETAM:
    dataset = dataset or load_dataset(config)
    artifact_dir = config.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "part1.npz"
    metadata = _checkpoint_metadata(config, dataset, 1)
    if resume and checkpoint_path.exists():
        model = SparseETAM.load(checkpoint_path)
        _verify_checkpoint(model, dataset, 1)
    else:
        model = SparseETAM(part1_connectivity())

    train_images, _ = augmented_patterns(
        dataset.train_images, dataset.train_labels, config
    )

    def save_checkpoint(current: SparseETAM, _: int) -> None:
        current.save(checkpoint_path, metadata)

    results = model.train(
        train_images,
        alpha=config.alpha,
        max_iterations=config.max_iterations,
        tolerance=config.margin_tolerance,
        checkpoint_every=config.checkpoint_every,
        checkpoint=save_checkpoint,
        progress=_progress,
    )
    _write_training_results(artifact_dir / "part1_training.csv", results)
    model.save(checkpoint_path, metadata)
    _validate_or_raise(model, train_images, config, checkpoint_path, metadata)
    return model


def train_part2(
    config: Config,
    dataset: Dataset | None = None,
    resume: bool = False,
) -> SparseETAM:
    dataset = dataset or load_dataset(config)
    artifact_dir = config.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = artifact_dir / "part2_classifier.npz"
    cleanup_path = artifact_dir / "part2_cleanup.npz"
    classifier_metadata = _checkpoint_metadata(config, dataset, 2)
    classifier_metadata["component"] = "leakage_free_class_readout"
    cleanup_metadata = _checkpoint_metadata(config, dataset, 2)
    cleanup_metadata["component"] = "template_cleanup"

    train_images, train_labels = augmented_patterns(
        dataset.train_images, dataset.train_labels, config
    )
    class_targets = np.eye(10, dtype=np.int8)[train_labels] * 2 - 1

    if resume and classifier_path.exists():
        classifier = SparseETAM.load(classifier_path)
        _verify_checkpoint(classifier, dataset, 2)
    else:
        classifier = SparseETAM(class_readout_connectivity())
        classifier.initialize_hebbian_readout(train_images, class_targets)
        classifier.save(classifier_path, classifier_metadata)

    templates = digit_templates()
    if resume and cleanup_path.exists():
        cleanup = SparseETAM.load(cleanup_path)
        _verify_checkpoint(cleanup, dataset, 2)
    else:
        cleanup = SparseETAM(template_cleanup_connectivity())
        results = cleanup.train(
            templates,
            alpha=config.alpha,
            max_iterations=config.max_iterations,
            tolerance=config.margin_tolerance,
            progress=_progress,
        )
        _write_training_results(artifact_dir / "part2_cleanup_training.csv", results)
        cleanup.save(cleanup_path, cleanup_metadata)

    violations = cleanup.stability_violations(templates)
    if violations:
        raise RuntimeError(f"template cleanup memory is unstable: {violations[:10]}")
    return classifier


def train_requested(config: Config, part: str, resume: bool = False) -> None:
    dataset = load_dataset(config)
    if part in {"1", "all"}:
        train_part1(config, dataset=dataset, resume=resume)
    if part in {"2", "all"}:
        train_part2(config, dataset=dataset, resume=resume)


def experiment_configs(config: Config, include_augmentation: bool) -> list[Config]:
    baseline = replace(
        config,
        artifact_dir=str(config.artifact_path / "baseline"),
        augmentation=replace(config.augmentation, enabled=False),
        strict_stability=False,
    )
    variants = [baseline]
    if include_augmentation:
        variants.append(
            replace(
                config,
                artifact_dir=str(config.artifact_path / "augmented"),
                augmentation=replace(config.augmentation, enabled=True),
                strict_stability=False,
            )
        )
    return variants


def write_run_manifest(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
