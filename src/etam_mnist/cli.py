from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .data import download_mnist
from .pipeline import (
    experiment_configs,
    load_dataset,
    train_requested,
    write_run_manifest,
)


def _config(path: str) -> Config:
    return Config.load(path)


def command_download(args: argparse.Namespace) -> int:
    config = _config(args.config)
    paths = download_mnist(config.data_path)
    for path in paths:
        print(path)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _config(args.config)
    write_run_manifest(config, config.artifact_path / "run_config.json")
    train_requested(config, args.part, resume=args.resume)
    print(f"models written to {config.artifact_path}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_artifacts

    config = _config(args.config)
    dataset = load_dataset(config)
    metrics = evaluate_artifacts(config, dataset)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def command_experiment(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_artifacts

    source = _config(args.config)
    summaries = {}
    for config in experiment_configs(source, args.augmentation):
        name = Path(config.artifact_dir).name
        print(f"running {name} experiment", flush=True)
        write_run_manifest(config, config.artifact_path / "run_config.json")
        train_requested(config, "all", resume=args.resume)
        dataset = load_dataset(config)
        summaries[name] = evaluate_artifacts(config, dataset)
    output = source.artifact_path / "experiment_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"comparison written to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etam-mnist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and verify MNIST")
    download.add_argument("--config", default="configs/quick.toml")
    download.set_defaults(func=command_download)

    train = subparsers.add_parser("train", help="train Part I and/or Part II")
    train.add_argument("--config", default="configs/quick.toml")
    train.add_argument("--part", choices=("1", "2", "all"), default="all")
    train.add_argument("--resume", action="store_true")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate saved checkpoints")
    evaluate.add_argument("--config", default="configs/quick.toml")
    evaluate.set_defaults(func=command_evaluate)

    experiment = subparsers.add_parser(
        "experiment", help="run baseline and optional augmentation experiments"
    )
    experiment.add_argument("--config", default="configs/quick.toml")
    experiment.add_argument("--augmentation", action="store_true")
    experiment.add_argument("--resume", action="store_true")
    experiment.set_defaults(func=command_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
