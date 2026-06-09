from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "etam-mnist-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "etam-mnist-cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .data import Dataset
from .model import SparseETAM
from .templates import decode_template, digit_templates


NOISE_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)


def _status_rates(statuses: list[str]) -> dict[str, float]:
    total = max(len(statuses), 1)
    return {
        "convergence_rate": statuses.count("stable") / total,
        "limit_cycle_rate": statuses.count("cycle") / total,
        "max_steps_rate": statuses.count("max_steps") / total,
    }


def evaluate_part1(
    model: SparseETAM,
    images: np.ndarray,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, Any]] = []
    for rate in NOISE_RATES:
        total_pixels = 0
        correct_pixels = 0
        exact = 0
        statuses: list[str] = []
        steps: list[int] = []
        flips = int(round(784 * rate))
        for original in images:
            noisy = original.copy()
            if flips:
                changed = rng.choice(784, size=flips, replace=False)
                noisy[changed] *= -1
            recalled = model.recall(noisy, max_steps=max_steps)
            correct_pixels += int(np.count_nonzero(recalled.state == original))
            total_pixels += 784
            exact += int(np.array_equal(recalled.state, original))
            statuses.append(recalled.status)
            steps.append(recalled.steps)
        row: dict[str, Any] = {
            "noise_rate": rate,
            "pixel_accuracy": correct_pixels / max(total_pixels, 1),
            "exact_recall_rate": exact / max(len(images), 1),
            "mean_steps": float(np.mean(steps)) if steps else 0.0,
        }
        row.update(_status_rates(statuses))
        metrics.append(row)
    return metrics


def evaluate_part2(
    classifier: SparseETAM,
    cleanup: SparseETAM,
    images: np.ndarray,
    labels: np.ndarray,
    max_steps: int,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions: list[int] = []
    statuses: list[str] = []
    steps: list[int] = []
    confusion = np.zeros((10, 10), dtype=np.int64)
    for image, label in zip(images, labels, strict=True):
        class_scores = classifier.local_fields(image)
        raw_prediction = int(np.argmax(class_scores))
        initial_template = digit_templates()[raw_prediction]
        recalled = cleanup.recall(initial_template, max_steps=max_steps)
        prediction = decode_template(recalled.state)
        predictions.append(prediction)
        statuses.append(recalled.status)
        steps.append(recalled.steps)
        confusion[int(label), prediction] += 1
    predicted = np.asarray(predictions)
    labels = np.asarray(labels)
    per_class = {}
    for label in range(10):
        mask = labels == label
        per_class[str(label)] = float(np.mean(predicted[mask] == label)) if mask.any() else 0.0
    metrics: dict[str, Any] = {
        "accuracy": float(np.mean(predicted == labels)) if labels.size else 0.0,
        "architecture": "784->10 class readout -> 96-bit template -> 96 cleanup",
        "per_class_accuracy": per_class,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
    }
    metrics.update(_status_rates(statuses))
    return metrics, confusion


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_noise(rows: list[dict[str, Any]], path: Path) -> None:
    rates = [row["noise_rate"] * 100 for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(rates, [row["pixel_accuracy"] for row in rows], "o-", label="Pixel accuracy")
    axis.plot(
        rates,
        [row["exact_recall_rate"] for row in rows],
        "s-",
        label="Exact recall",
    )
    axis.set(xlabel="Bit-flip noise (%)", ylabel="Rate", ylim=(0, 1.02))
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion(confusion: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set(xlabel="Predicted", ylabel="True", xticks=range(10), yticks=range(10))
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _report_text(metrics: dict[str, Any], artifact_dir: Path) -> str:
    lines = [
        "# MNIST ETAM 實驗報告",
        "",
        "本報告由 `etam-mnist evaluate` 自動產生。模型依 ETAM Eq.(7–12) 訓練，"
        "採 bipolar 像素與同步 recurrent recall。",
        "",
    ]
    if "part1" in metrics:
        lines.extend(
            [
                "## Part I：影像關聯記憶",
                "",
                "| Bit-flip | Pixel accuracy | Exact recall | Convergence | Limit cycle |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics["part1"]:
            lines.append(
                f"| {row['noise_rate']:.0%} | {row['pixel_accuracy']:.4f} | "
                f"{row['exact_recall_rate']:.4f} | {row['convergence_rate']:.4f} | "
                f"{row['limit_cycle_rate']:.4f} |"
            )
        lines.extend(["", "![Part I noise curve](part1_noise.png)", ""])
    if "part2" in metrics:
        part2 = metrics["part2"]
        lines.extend(
            [
                "## Part II：數字分類",
                "",
                f"- Architecture：`{part2.get('architecture', 'legacy')}`",
                f"- Accuracy：{part2['accuracy']:.4f}",
                f"- Convergence rate：{part2['convergence_rate']:.4f}",
                f"- Limit-cycle rate：{part2['limit_cycle_rate']:.4f}",
                f"- Mean recall steps：{part2['mean_steps']:.2f}",
                "",
                "![Confusion matrix](part2_confusion.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 產物",
            "",
            f"- 原始指標：`{artifact_dir.name}/metrics.json`",
            "- Part I 表格：`part1_metrics.csv`",
            "- Part II confusion matrix：`part2_confusion.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_artifacts(config: Config, dataset: Dataset) -> dict[str, Any]:
    artifact_dir = config.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    count = min(config.evaluation_limit, len(dataset.test_images))
    images = dataset.test_images[:count]
    labels = dataset.test_labels[:count]
    metrics: dict[str, Any] = {
        "evaluation_samples": count,
        "split_hash": dataset.split_hash,
    }

    part1_path = artifact_dir / "part1.npz"
    if part1_path.exists():
        part1 = SparseETAM.load(part1_path)
        part1_rows = evaluate_part1(part1, images, config.max_recall_steps, config.seed)
        metrics["part1"] = part1_rows
        _write_csv(artifact_dir / "part1_metrics.csv", part1_rows)
        _plot_noise(part1_rows, artifact_dir / "part1_noise.png")

    classifier_path = artifact_dir / "part2_classifier.npz"
    cleanup_path = artifact_dir / "part2_cleanup.npz"
    if classifier_path.exists() and cleanup_path.exists():
        classifier = SparseETAM.load(classifier_path)
        cleanup = SparseETAM.load(cleanup_path)
        part2_metrics, confusion = evaluate_part2(
            classifier, cleanup, images, labels, config.max_recall_steps
        )
        metrics["part2"] = part2_metrics
        np.savetxt(
            artifact_dir / "part2_confusion.csv",
            confusion,
            fmt="%d",
            delimiter=",",
        )
        _plot_confusion(confusion, artifact_dir / "part2_confusion.png")

    if "part1" not in metrics and "part2" not in metrics:
        raise FileNotFoundError(f"no model checkpoints found in {artifact_dir}")

    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "report.md").write_text(
        _report_text(metrics, artifact_dir), encoding="utf-8"
    )
    return metrics
