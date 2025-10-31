"""Configuration utilities for emoSpark pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RuntimeConfig:
    """Runtime configuration supporting local and EMR execution."""

    environment: str = field(
        default_factory=lambda: os.getenv("EMO_SPARK_ENV", "local")
    )
    input_path: str = field(
        default_factory=lambda: os.getenv("EMO_SPARK_INPUT_PATH", "data")
    )
    output_path: str = field(
        default_factory=lambda: os.getenv("EMO_SPARK_OUTPUT_PATH", "output")
    )
    sample_fraction: Optional[float] = field(
        default_factory=lambda: _get_float_env("EMO_SPARK_SAMPLE_FRACTION")
    )
    seed: int = field(default_factory=lambda: int(os.getenv("EMO_SPARK_SEED", "42")))
    cache_intermediate: bool = field(
        default_factory=lambda: os.getenv("EMO_SPARK_CACHE", "1") == "1"
    )
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = field(
        default_factory=lambda: os.getenv("EMO_SPARK_DISABLE_STRATIFY", "0") != "1"
    )
    repartition_target: Optional[int] = field(
        default_factory=lambda: _get_int_env("EMO_SPARK_REPARTITIONS")
    )

    # Feature settings
    max_vocab: int = field(
        default_factory=lambda: int(os.getenv("EMO_SPARK_MAX_VOCAB", "20000"))
    )
    min_df: int = field(default_factory=lambda: int(os.getenv("EMO_SPARK_MIN_DF", "5")))
    ngram_orders: List[int] = field(
        default_factory=lambda: _get_ngram_orders(os.getenv("EMO_SPARK_NGRAMS"))
    )
    idf_min_doc_freq: int = field(
        default_factory=lambda: int(os.getenv("EMO_SPARK_IDF_MIN_DOC_FREQ", "1"))
    )

    # Logistic regression hyperparameters (single setting to avoid grid search)
    lr_reg_param: float = field(
        default_factory=lambda: _get_first_float(os.getenv("EMO_SPARK_LR_REG"), 0.1)
    )
    lr_elastic_net_param: float = field(
        default_factory=lambda: _get_first_float(os.getenv("EMO_SPARK_LR_ELASTIC"), 0.0)
    )
    lr_max_iter: int = field(
        default_factory=lambda: _get_first_int(os.getenv("EMO_SPARK_LR_MAX_ITER"), 50)
    )

    # Execution options
    master: Optional[str] = field(default_factory=lambda: os.getenv("EMO_SPARK_MASTER"))
    app_name: str = field(
        default_factory=lambda: os.getenv("EMO_SPARK_APP_NAME", "emoSparkPipeline")
    )

    # Demo options
    demo_sample_size: int = field(
        default_factory=lambda: int(os.getenv("EMO_SPARK_DEMO_SAMPLES", "15"))
    )
    probability_thresholds: Dict[str, float] = field(
        default_factory=lambda: _get_threshold_map(os.getenv("EMO_SPARK_THRESHOLDS"))
    )
    auto_tune_thresholds: bool = field(
        default_factory=lambda: os.getenv("EMO_SPARK_AUTO_TUNE_THRESHOLDS", "1") == "1"
    )
    threshold_candidates: List[float] = field(
        default_factory=lambda: _get_threshold_candidates(
            os.getenv("EMO_SPARK_THRESHOLD_GRID")
        )
    )

    def threshold_for(self, label: str) -> float:
        """Get probability threshold for a specific emotion label.

        Args:
            label: Emotion label name.

        Returns:
            Probability threshold for the label, defaults to 0.5 if not configured.
        """
        return float(self.probability_thresholds.get(label, 0.5))


def _get_float_env(name: str) -> Optional[float]:
    """Parse environment variable as float, returning None on missing/invalid value.

    Args:
        name: Environment variable name to parse.

    Returns:
        Float value if valid, None otherwise.
    """
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value


def _get_int_env(name: str) -> Optional[int]:
    """Parse environment variable as integer, returning None on missing/invalid value.

    Args:
        name: Environment variable name to parse.

    Returns:
        Integer value if valid, None otherwise.
    """
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value


def _get_float_list(raw: Optional[str]) -> List[float]:
    """Parse comma-separated string into list of floats.

    Args:
        raw: Comma-separated string of float values.

    Returns:
        List of float values, empty list if input is None or empty.
    """
    if not raw:
        return []
    return [float(item) for item in raw.split(",") if item]


def _get_int_list(raw: Optional[str]) -> List[int]:
    """Parse comma-separated string into list of integers.

    Args:
        raw: Comma-separated string of integer values.

    Returns:
        List of integer values, empty list if input is None or empty.
    """
    if not raw:
        return []
    return [int(item) for item in raw.split(",") if item]


def _get_ngram_orders(raw: Optional[str]) -> List[int]:
    """Parse n-gram order configuration from comma-separated string.

    Args:
        raw: Comma-separated string of n-gram orders.

    Returns:
        Sorted list of unique n-gram orders, defaults to [1, 2, 3] if None.
    """
    if not raw:
        return [1, 2, 3]
    values = sorted({int(item) for item in raw.split(",") if item})
    return values or [1, 2, 3]


def _get_threshold_map(raw: Optional[str]) -> Dict[str, float]:
    """Parse probability thresholds from JSON or key=value pairs.

    Supports two formats:
    1. JSON object: '{"joy": 0.6, "fear": 0.4}'
    2. Key=value pairs: 'joy=0.6,fear=0.4'

    Args:
        raw: Threshold configuration string in JSON or key=value format.

    Returns:
        Dictionary mapping emotion labels to probability thresholds,
        defaults to DEFAULT_PROBABILITY_THRESHOLDS if parsing fails.
    """
    from .constants import DEFAULT_PROBABILITY_THRESHOLDS  # local import to avoid cycle

    if not raw:
        return dict(DEFAULT_PROBABILITY_THRESHOLDS)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
        for item in raw.split(","):
            if "=" not in item:
                continue
            label, value = item.split("=", 1)
            try:
                parsed[label.strip()] = float(value.strip())
            except ValueError:
                continue

    if isinstance(parsed, dict):
        thresholds: Dict[str, float] = {}
        for key, value in parsed.items():
            try:
                thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if thresholds:
            return thresholds

    return dict(DEFAULT_PROBABILITY_THRESHOLDS)


def _get_first_float(raw: Optional[str], default: float) -> float:
    """Return the first float from a comma-separated string or a default."""

    values = _get_float_list(raw)
    return values[0] if values else default


def _get_first_int(raw: Optional[str], default: int) -> int:
    """Return the first int from a comma-separated string or a default."""

    values = _get_int_list(raw)
    return values[0] if values else default


def _get_threshold_candidates(raw: Optional[str]) -> List[float]:
    """Parse threshold search grid ensuring a reasonable default."""

    values = _get_float_list(raw) if raw else [0.3, 0.4, 0.5, 0.6, 0.7]
    unique = sorted({round(val, 4) for val in values if 0.0 <= val <= 1.0})
    return unique or [0.5]
