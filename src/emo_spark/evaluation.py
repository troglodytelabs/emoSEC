"""Evaluation orchestration for emoSpark."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT

from .constants import TARGET_LABELS
from .metrics import EvaluationResult, compute_multilabel_metrics
from .models import ModelSet

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetrics:
    """Metrics for a specific dataset split.

    Attributes:
        dataset: Dataset name (train, validation, test).
        metrics: Computed evaluation metrics.
    """

    dataset: str
    metrics: EvaluationResult

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary with dataset name included.

        Returns:
            Dictionary with dataset name and all metrics.
        """
        payload = self.metrics.as_dict()
        payload["dataset"] = self.dataset
        return payload


class EvaluationManager:
    """Orchestrate evaluation across multiple datasets and models.

    Handles probability threshold application, metric computation,
    and persistence of evaluation results.

    Attributes:
        label_cols: List of emotion labels to evaluate.
        thresholds: Probability thresholds per label.
    """

    def __init__(
        self,
        label_cols: Iterable[str] = TARGET_LABELS,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize evaluation manager.

        Args:
            label_cols: Emotion labels to evaluate.
            thresholds: Probability thresholds for each label.
        """
        self.label_cols = list(label_cols)
        self.thresholds = dict(thresholds or {})

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Merge a new threshold map into the manager."""

        self.thresholds.update(
            {label: float(value) for label, value in new_thresholds.items()}
        )

    def get_thresholds(self) -> Dict[str, float]:
        """Return a copy of the active threshold configuration."""

        return dict(self.thresholds)

    def tune_thresholds(
        self,
        validation_predictions: DataFrame,
        model_type: str,
        candidates: Sequence[float],
    ) -> Dict[str, float]:
        """Optimize per-emotion probability thresholds on validation data."""

        tuned = self.get_thresholds()
        valid_candidates = [float(val) for val in candidates if 0.0 <= val <= 1.0]
        if not valid_candidates:
            logger.warning("No valid threshold candidates provided; skipping tuning")
            return tuned

        # For each emotion, search over threshold candidates to find optimal F1
        # Different emotions have different class imbalances, so optimal thresholds vary
        # Example: rare emotions (fear) may need lower thresholds to avoid missing them
        for label in self.label_cols:
            prob_col = f"prob_{model_type}_{label}"

            # Skip labels where model doesn't provide probability scores
            # (e.g., LinearSVC only provides raw decision margins)
            if prob_col not in validation_predictions.columns:
                logger.debug(
                    "Model '%s' lacks probability column for '%s'; retaining threshold %.3f",
                    model_type,
                    label,
                    tuned.get(label, 0.5),
                )
                continue

            logger.debug(
                "Tuning threshold for model '%s', label '%s' over %d candidates",
                model_type,
                label,
                len(valid_candidates),
            )

            # Extract probability scores as scalar column (handle vector format)
            probability_expr = _ensure_scalar_probability(
                validation_predictions, prob_col
            ).alias("probability")

            # Evaluate each threshold candidate on validation data
            # For each threshold, compute TP, FP, FN, and F1 score
            # Grid search: try all candidates and pick the one with best F1
            candidate_stats = _evaluate_threshold_grid(
                validation_predictions.select(
                    F.col(label).cast("double").alias("target"),
                    probability_expr,
                ),
                valid_candidates,
            )

            # Select the threshold with highest F1 score
            # Ties are broken by recall (prefer higher recall), then lower threshold
            # Rationale: Better to detect more emotions (high recall) than miss them
            best_threshold, best_score = _select_best_threshold(candidate_stats)
            logger.debug(
                "Selected threshold %.3f for '%s' (F1=%.4f)",
                best_threshold,
                label,
                best_score,
            )
            tuned[label] = best_threshold

        # Update internal state with tuned thresholds
        self.thresholds = tuned
        return tuned

    def project_predictions(self, df: DataFrame, model_type: str) -> DataFrame:
        """Project model-specific predictions to standard format with thresholds.

        Applies probability thresholds to convert probabilities to binary
        predictions. Falls back to raw predictions if probabilities unavailable.

        Args:
            df: DataFrame with model-specific prediction columns.
            model_type: Model algorithm name for column naming.

        Returns:
            DataFrame with standard pred_{label} and prob_{label} columns.
        """
        select_cols = [col for col in ("example_id", "text") if col in df.columns]
        label_cols_available = [
            label for label in self.label_cols if label in df.columns
        ]
        select_cols.extend(label_cols_available)
        for label in self.label_cols:
            select_cols.append(f"pred_{model_type}_{label}")
            prob_name = f"prob_{model_type}_{label}"
            if prob_name in df.columns:
                select_cols.append(prob_name)

        subset = df.select(*select_cols)

        projected = subset
        for label in self.label_cols:
            model_pred_col = f"pred_{model_type}_{label}"
            prob_col_name = f"prob_{model_type}_{label}"
            threshold = float(self.thresholds.get(label, 0.5))

            if prob_col_name in subset.columns:
                probability = _ensure_scalar_probability(subset, prob_col_name)
                projected = projected.withColumn(
                    f"prob_{label}", probability.cast("double")
                )
                projected = projected.withColumn(
                    f"pred_{label}",
                    F.when(probability >= threshold, F.lit(1.0)).otherwise(F.lit(0.0)),
                )
            elif model_pred_col in subset.columns:
                projected = projected.withColumn(
                    f"pred_{label}", F.col(model_pred_col).cast("double")
                )
        return projected

    def evaluate(
        self, predictions: DataFrame, model_type: str, dataset_name: str
    ) -> DatasetMetrics:
        """Evaluate predictions and compute all metrics.

        Args:
            predictions: DataFrame with model predictions.
            model_type: Model algorithm name.
            dataset_name: Dataset split name (train/validation/test).

        Returns:
            DatasetMetrics with computed evaluation results.
        """
        projected = self.project_predictions(predictions, model_type)
        metrics = compute_multilabel_metrics(projected, self.label_cols)
        return DatasetMetrics(dataset=dataset_name, metrics=metrics)

    def persist_metrics(
        self, results: List[DatasetMetrics], output_dir: str, model_type: str
    ) -> None:
        """Save evaluation metrics to JSON file.

        Args:
            results: List of DatasetMetrics to save.
            output_dir: Directory for output files.
            model_type: Model algorithm name for filename.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"metrics_{model_type}.json")
        payload = [result.to_dict() for result in results]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def run_full_evaluation(
        self,
        model_set: ModelSet,
        output_dir: str,
        include_train: bool = False,
        train_predictions: Optional[DataFrame] = None,
    ) -> List[DatasetMetrics]:
        """Run evaluation on all dataset splits and persist results.

        Args:
            model_set: ModelSet with predictions for all splits.
            output_dir: Directory for saving metrics.
            include_train: Whether to include training set evaluation.
            train_predictions: Training predictions if include_train is True.

        Returns:
            List of DatasetMetrics for all evaluated splits.
        """
        results: List[DatasetMetrics] = []
        if include_train and train_predictions is not None:
            results.append(
                self.evaluate(train_predictions, model_set.model_type, "train")
            )
        results.append(
            self.evaluate(
                model_set.validation_predictions, model_set.model_type, "validation"
            )
        )
        results.append(
            self.evaluate(model_set.test_predictions, model_set.model_type, "test")
        )
        self.persist_metrics(results, output_dir, model_set.model_type)
        return results


def _ensure_scalar_probability(df: DataFrame, column_name: str):
    """Return a column expression yielding scalar probabilities for evaluation."""

    field = next(
        (field for field in df.schema.fields if field.name == column_name), None
    )
    if field is None:
        raise ValueError(f"Column '{column_name}' not found for probability extraction")

    if isinstance(field.dataType, VectorUDT):
        return vector_to_array(F.col(column_name)).getItem(1)

    return F.col(column_name).cast("double")


def save_predictions(
    df: DataFrame, output_dir: str, model_type: str, dataset_name: str
) -> None:
    """Save predictions to Parquet format.

    Args:
        df: DataFrame with predictions to save.
        output_dir: Base directory for predictions.
        model_type: Model algorithm name for subdirectory.
        dataset_name: Dataset split name for subdirectory.
    """
    path = os.path.join(output_dir, model_type, dataset_name)
    df.write.mode("overwrite").parquet(path)


def _evaluate_threshold_grid(
    df: DataFrame, candidates: Sequence[float]
) -> Dict[float, Dict[str, float]]:
    """Compute confusion-matrix statistics for each threshold candidate."""

    # Generate unique aliases for each threshold to avoid Spark column name conflicts
    # Example: threshold 0.45 → "thr_0_45"
    candidate_alias = {
        candidate: _threshold_alias(candidate) for candidate in candidates
    }

    agg_exprs = []
    probability = F.col("probability")
    target = F.col("target")

    # For each threshold, compute TP, FP, FN in a single aggregation pass
    # This is much more efficient than looping and computing separately
    for candidate, alias in candidate_alias.items():
        # prediction = 1 if probability >= threshold, else 0
        prediction = probability >= F.lit(candidate)

        agg_exprs.extend(
            [
                # True Positive: predicted positive AND actually positive
                F.sum(F.when(prediction & (target == 1.0), 1).otherwise(0)).alias(
                    f"tp_{alias}"
                ),
                # False Positive: predicted positive BUT actually negative
                F.sum(F.when(prediction & (target == 0.0), 1).otherwise(0)).alias(
                    f"fp_{alias}"
                ),
                # False Negative: predicted negative BUT actually positive
                F.sum(F.when(~prediction & (target == 1.0), 1).otherwise(0)).alias(
                    f"fn_{alias}"
                ),
            ]
        )

    # Execute all threshold evaluations in one pass (efficient!)
    stats_row = df.agg(*agg_exprs).collect()[0].asDict()

    # Calculate precision, recall, F1 for each threshold
    results: Dict[float, Dict[str, float]] = {}
    for candidate, alias in candidate_alias.items():
        tp = float(stats_row.get(f"tp_{alias}", 0.0))
        fp = float(stats_row.get(f"fp_{alias}", 0.0))
        fn = float(stats_row.get(f"fn_{alias}", 0.0))

        # Precision: correct positives / all predicted positives
        precision = _safe_divide(tp, tp + fp)
        # Recall: correct positives / all actual positives
        recall = _safe_divide(tp, tp + fn)
        # F1: harmonic mean of precision and recall
        f1 = (
            _safe_divide(2 * precision * recall, precision + recall)
            if precision or recall
            else 0.0
        )

        results[candidate] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return results


def _select_best_threshold(stats: Dict[float, Dict[str, float]]) -> tuple[float, float]:
    """Pick the threshold with the highest F1 (ties resolved by recall, then lower threshold).

    Tie-breaking rationale:
    1. Prefer higher F1 (primary metric)
    2. If F1 is tied, prefer higher recall (better to detect emotion than miss it)
    3. If still tied, prefer lower threshold (more inclusive, less likely to miss)
    """

    best_threshold = 0.5  # Default fallback
    best_f1 = -1.0
    best_recall = -1.0

    # Sort by threshold value to ensure consistent tie-breaking
    for candidate, metrics in sorted(stats.items()):
        f1 = metrics.get("f1", 0.0)
        recall = metrics.get("recall", 0.0)

        # Update best if: F1 is better, OR F1 is tied but recall is better
        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            best_threshold = candidate
            best_f1 = f1
            best_recall = recall

    return best_threshold, best_f1


def _threshold_alias(threshold: float) -> str:
    """Generate a safe string alias for Spark aggregation columns."""

    formatted = f"{threshold:.4f}".rstrip("0").rstrip(".")
    sanitized = formatted.replace("-", "neg").replace(".", "_")
    return f"thr_{sanitized or '0'}"


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two floats, returning 0 when the denominator is 0."""

    return numerator / denominator if denominator else 0.0
