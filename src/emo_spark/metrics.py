"""Evaluation helpers for multi-label emotion classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .constants import TARGET_LABELS


@dataclass
class LabelMetrics:
    """Per-label classification metrics.

    Attributes:
        label: Emotion label name.
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        support: Number of positive examples.
    """

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvaluationResult:
    """Complete multi-label evaluation metrics.

    Attributes:
        hamming_loss: Average per-label disagreement.
        subset_accuracy: Exact match accuracy.
        micro_precision: Micro-averaged precision.
        micro_recall: Micro-averaged recall.
        micro_f1: Micro-averaged F1 score.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1 score.
        per_label: List of per-label metrics.
    """

    hamming_loss: float
    subset_accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: List[LabelMetrics]

    def as_dict(self) -> Dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of all metrics.
        """
        return {
            "hamming_loss": self.hamming_loss,
            "subset_accuracy": self.subset_accuracy,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_label": [metric.__dict__ for metric in self.per_label],
        }


def _safe_divide(num: float, denom: float) -> float:
    """Safely divide two numbers, returning 0 if denominator is 0.

    Args:
        num: Numerator.
        denom: Denominator.

    Returns:
        Division result or 0.0 if denominator is 0.
    """
    return num / denom if denom else 0.0


def compute_multilabel_metrics(
    df: DataFrame, label_cols: Iterable[str]
) -> EvaluationResult:
    """Compute comprehensive multi-label classification metrics.

    Calculates:
    - Hamming loss (per-label error rate)
    - Subset accuracy (exact match)
    - Micro/macro precision, recall, F1
    - Per-label precision, recall, F1, support

    Args:
        df: DataFrame with label columns and pred_{label} columns.
        label_cols: List of emotion label names.

    Returns:
        EvaluationResult with all computed metrics.
    """
    label_cols = list(label_cols)
    prediction_cols = [f"pred_{label}" for label in label_cols]

    diff_sum = None
    for label, pred in zip(label_cols, prediction_cols):
        diff = F.abs(F.col(label) - F.col(pred))
        diff_sum = diff if diff_sum is None else diff_sum + diff

    row_hamming = (diff_sum / float(len(label_cols))).alias("row_hamming")
    hamming_loss = (
        df.select(row_hamming).agg(F.avg("row_hamming").alias("hamming")).first()[0]
    )

    match_product = None
    for label, pred in zip(label_cols, prediction_cols):
        match = F.when(F.col(label) == F.col(pred), F.lit(1.0)).otherwise(F.lit(0.0))
        match_product = match if match_product is None else match_product * match
    subset_accuracy = (
        df.select(match_product.alias("subset_match"))
        .agg(F.avg("subset_match"))
        .first()[0]
    )

    agg_exprs = []
    for label, pred in zip(label_cols, prediction_cols):
        agg_exprs.extend(
            [
                F.sum(
                    F.when((F.col(label) == 1.0) & (F.col(pred) == 1.0), 1).otherwise(0)
                ).alias(f"{label}_tp"),
                F.sum(
                    F.when((F.col(label) == 0.0) & (F.col(pred) == 1.0), 1).otherwise(0)
                ).alias(f"{label}_fp"),
                F.sum(
                    F.when((F.col(label) == 1.0) & (F.col(pred) == 0.0), 1).otherwise(0)
                ).alias(f"{label}_fn"),
                F.sum(F.when(F.col(label) == 1.0, 1).otherwise(0)).alias(
                    f"{label}_support"
                ),
            ]
        )

    agg_row = df.agg(*agg_exprs).collect()[0].asDict()

    per_label: List[LabelMetrics] = []
    micro_tp = micro_fp = micro_fn = 0.0

    for label in label_cols:
        tp = float(agg_row.get(f"{label}_tp", 0.0))
        fp = float(agg_row.get(f"{label}_fp", 0.0))
        fn = float(agg_row.get(f"{label}_fn", 0.0))
        support = int(agg_row.get(f"{label}_support", 0))

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = (
            _safe_divide(2 * precision * recall, precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_label.append(
            LabelMetrics(
                label=label, precision=precision, recall=recall, f1=f1, support=support
            )
        )

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_precision = sum(metric.precision for metric in per_label) / len(per_label)
    macro_recall = sum(metric.recall for metric in per_label) / len(per_label)
    macro_f1 = sum(metric.f1 for metric in per_label) / len(per_label)

    return EvaluationResult(
        hamming_loss=float(hamming_loss or 0.0),
        subset_accuracy=float(subset_accuracy or 0.0),
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_label=per_label,
    )


def attach_prediction_columns(df: DataFrame) -> DataFrame:
    """Select and rename prediction columns for evaluation.

    Args:
        df: DataFrame with label and prediction columns.

    Returns:
        DataFrame with text, labels, and predictions only.
    """
    cols = [F.col(label).alias(label) for label in TARGET_LABELS]
    pred_cols = [
        F.col(f"pred_{label}").alias(f"pred_{label}") for label in TARGET_LABELS
    ]
    return df.select("text", *cols, *pred_cols)
