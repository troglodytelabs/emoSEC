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

    # ========================================================
    # HAMMING LOSS: Average per-label error rate
    # ========================================================
    # Measures the fraction of labels that are incorrectly predicted
    # Example: If 2 out of 8 labels are wrong, hamming loss = 2/8 = 0.25
    # Lower is better (0 = perfect)
    diff_sum = None
    for label, pred in zip(label_cols, prediction_cols):
        # abs(true - pred) = 1 if mismatch, 0 if match
        diff = F.abs(F.col(label) - F.col(pred))
        diff_sum = diff if diff_sum is None else diff_sum + diff

    # Average the error count across all labels (row-wise)
    row_hamming = (diff_sum / float(len(label_cols))).alias("row_hamming")
    # Then average across all examples (dataset-wise)
    hamming_loss = (
        df.select(row_hamming).agg(F.avg("row_hamming").alias("hamming")).first()[0]
    )

    # ========================================================
    # SUBSET ACCURACY: Exact match rate
    # ========================================================
    # Percentage of examples where ALL labels are predicted correctly
    # Very strict metric - even one wrong label counts as failure
    # Example: True=[1,0,1], Pred=[1,0,1] → match=1
    #          True=[1,0,1], Pred=[1,1,1] → match=0 (one error)
    match_product = None
    for label, pred in zip(label_cols, prediction_cols):
        # Check if this specific label matches (1.0 = match, 0.0 = mismatch)
        match = F.when(F.col(label) == F.col(pred), F.lit(1.0)).otherwise(F.lit(0.0))
        # Multiply all matches together (product = 1 only if ALL match)
        match_product = match if match_product is None else match_product * match

    # Average the product across all examples
    # This gives the proportion of perfectly predicted examples
    subset_accuracy = (
        df.select(match_product.alias("subset_match"))
        .agg(F.avg("subset_match"))
        .first()[0]
    )

    # ========================================================
    # CONFUSION MATRIX COMPONENTS (TP, FP, FN, Support)
    # ========================================================
    # Compute these for each label to calculate precision/recall/F1
    # TP (True Positive): Predicted=1, Actual=1 (correctly detected)
    # FP (False Positive): Predicted=1, Actual=0 (false alarm)
    # FN (False Negative): Predicted=0, Actual=1 (missed detection)
    # Support: Total actual positive examples for this label
    agg_exprs = []
    for label, pred in zip(label_cols, prediction_cols):
        agg_exprs.extend(
            [
                # Count true positives: model said yes AND label is yes
                F.sum(
                    F.when((F.col(label) == 1.0) & (F.col(pred) == 1.0), 1).otherwise(0)
                ).alias(f"{label}_tp"),
                # Count false positives: model said yes BUT label is no
                F.sum(
                    F.when((F.col(label) == 0.0) & (F.col(pred) == 1.0), 1).otherwise(0)
                ).alias(f"{label}_fp"),
                # Count false negatives: model said no BUT label is yes
                F.sum(
                    F.when((F.col(label) == 1.0) & (F.col(pred) == 0.0), 1).otherwise(0)
                ).alias(f"{label}_fn"),
                # Count total actual positives (ground truth)
                F.sum(F.when(F.col(label) == 1.0, 1).otherwise(0)).alias(
                    f"{label}_support"
                ),
            ]
        )

    # Execute all aggregations in a single pass for efficiency
    agg_row = df.agg(*agg_exprs).collect()[0].asDict()

    # ========================================================
    # PER-LABEL METRICS: Precision, Recall, F1, Support
    # ========================================================
    per_label: List[LabelMetrics] = []
    micro_tp = micro_fp = micro_fn = 0.0

    for label in label_cols:
        # Extract confusion matrix values for this label, defaulting None to 0
        tp_val = agg_row.get(f"{label}_tp")
        fp_val = agg_row.get(f"{label}_fp")
        fn_val = agg_row.get(f"{label}_fn")
        support_val = agg_row.get(f"{label}_support")

        tp = float(tp_val) if tp_val is not None else 0.0
        fp = float(fp_val) if fp_val is not None else 0.0
        fn = float(fn_val) if fn_val is not None else 0.0
        support = int(support_val) if support_val is not None else 0

        # Precision: Of all examples we predicted positive, how many were correct?
        # precision = TP / (TP + FP) = correct positives / all predicted positives
        # High precision = low false alarm rate
        precision = _safe_divide(tp, tp + fp)

        # Recall: Of all actual positive examples, how many did we detect?
        # recall = TP / (TP + FN) = detected positives / all actual positives
        # High recall = low miss rate
        recall = _safe_divide(tp, tp + fn)

        # F1: Harmonic mean of precision and recall
        # f1 = 2 × (precision × recall) / (precision + recall)
        # Balances precision and recall into a single score
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

        # Accumulate for micro-averaging (global counts across all labels)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    # ========================================================
    # MICRO-AVERAGING: Global metrics treating all labels equally
    # ========================================================
    # Pool all (label, prediction) pairs and compute metrics globally
    # This gives more weight to common labels (e.g., joy appears more than trust)
    # Good for overall system performance assessment
    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    # ========================================================
    # MACRO-AVERAGING: Mean of per-label metrics
    # ========================================================
    # Compute metric for each label independently, then average
    # Treats all labels equally regardless of frequency
    # Good for assessing performance on rare labels (e.g., trust, surprise)
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
