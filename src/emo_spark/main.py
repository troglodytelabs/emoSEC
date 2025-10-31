"""CLI entrypoint for running the emoSpark pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from typing import List

from .config import RuntimeConfig
from .pipeline import run_pipeline

LOGGER = logging.getLogger("emo_spark.cli")


def parse_model_list(raw: str) -> List[str]:
    """Parse comma-separated model list from CLI argument.

    Args:
        raw: Comma-separated string of model names.

    Returns:
        List of trimmed model name strings.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for emoSpark training.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Run the emoSpark training pipeline")
    parser.add_argument(
        "--models",
        type=parse_model_list,
        default=parse_model_list(
            "logistic_regression,linear_svm,naive_bayes,random_forest"
        ),
        help="Comma-separated list of model families to train",
    )
    parser.add_argument(
        "--no-train-metrics",
        action="store_true",
        help="Skip computing train-set predictions and metrics",
    )
    parser.add_argument(
        "--no-feature-save",
        action="store_true",
        help="Skip saving intermediate feature datasets to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Main CLI entrypoint for emoSpark training pipeline.

    Parses command line arguments, initializes configuration,
    runs the complete pipeline, and prints summary results.

    Args:
        argv: Command line arguments, defaults to sys.argv if None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = RuntimeConfig()
    LOGGER.info("Using configuration: %s", config)

    artifacts = run_pipeline(
        config,
        models_to_train=args.models,
        include_train_metrics=not args.no_train_metrics,
        save_feature_datasets=not args.no_feature_save,
    )

    summary = {
        "test_set_path": artifacts.test_set_path,
        "demo_samples_path": artifacts.demo_samples_path,
        "feature_pipeline_path": artifacts.feature_pipeline_path,
        "trained_models": list(artifacts.model_results.keys()),
        "tuned_thresholds": artifacts.tuned_thresholds,
        "threshold_paths": artifacts.threshold_paths,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
