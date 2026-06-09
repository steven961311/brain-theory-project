from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    angles: tuple[float, ...] = (-15.0, -5.0, 5.0, 15.0)
    scales: tuple[float, ...] = (0.9, 1.1)


@dataclass(frozen=True)
class Config:
    seed: int = 20260609
    data_dir: str = "data"
    artifact_dir: str = "artifacts/quick"
    train_size: int = 500
    test_size: int = 100
    threshold: int = 128
    alpha: float = 0.005
    max_iterations: int = 30
    margin_tolerance: float = 1.0e-7
    max_recall_steps: int = 30
    checkpoint_every: int = 50
    strict_stability: bool = False
    evaluation_limit: int = 100
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        aug = raw.pop("augmentation", {})
        return cls(
            **raw,
            augmentation=AugmentationConfig(
                enabled=bool(aug.get("enabled", False)),
                angles=tuple(float(v) for v in aug.get("angles", (-15, -5, 5, 15))),
                scales=tuple(float(v) for v in aug.get("scales", (0.9, 1.1))),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)

