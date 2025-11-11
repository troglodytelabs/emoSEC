"""Utility helpers for training and evaluating the Spark emotion ensemble."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql.functions import array, col, lit, sum as spark_sum, when
from pyspark.sql.types import DoubleType


def compute_class_weights(
    train_df: DataFrame, train_count: int, emotions: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """Return positive/negative weights per emotion to offset class imbalance."""
    label_sum_columns = [
        spark_sum(col("labels")[idx]).alias(emotion)
        for idx, emotion in enumerate(emotions)
    ]
    label_sums_row = train_df.select(*label_sum_columns).collect()[0]

    class_weights: Dict[str, Dict[str, float]] = {}
    for emotion in emotions:
        positives = float(label_sums_row[emotion] or 0.0)
        negatives = float(train_count - positives)
        if positives == 0.0 or negatives == 0.0:
            class_weights[emotion] = {"positive": 1.0, "negative": 1.0}
        else:
            class_weights[emotion] = {
                "positive": negatives / positives,
                "negative": 1.0,
            }
    return class_weights


def generate_ensemble_outputs(
    dataset_df: DataFrame,
    dataset_name: str,
    algorithms: Iterable[str],
    emotions: Iterable[str],
    all_models: Dict[str, Dict[str, object]],
    decision_threshold: float,
    thresholds: Optional[Dict[str, float]] = None,
    algo_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> DataFrame:
    """Score an arbitrary split and assemble the per-emotion ensemble outputs."""
    algorithms = list(algorithms)
    emotions = list(emotions)
    dataset_predictions = {algo: [] for algo in algorithms}

    print(f"\nscoring {dataset_name} data with ensemble...")
    for algorithm in algorithms:
        print(f"  predicting with {algorithm} on {dataset_name}...", flush=True)
        for idx, emotion in enumerate(emotions):
            emotion_df = dataset_df.withColumn(
                "label", col("labels").getItem(idx).cast(DoubleType())
            )
            model = all_models[algorithm][emotion]
            predictions = model.transform(emotion_df)
            if algorithm == "svm":
                prob_col = col("prediction")
            else:
                prob_col = vector_to_array(col("probability"))[1]
            predictions = predictions.select(
                "row_id",
                prob_col.alias(f"prob_{algorithm}_{emotion}"),
            )
            dataset_predictions[algorithm].append(predictions)

    combined = dataset_df.select("row_id", "text", "labels")
    for algorithm in algorithms:
        for pred_df in dataset_predictions[algorithm]:
            combined = combined.join(pred_df, on="row_id", how="left")

    for emotion in emotions:
        weight_map = (algo_weights or {}).get(emotion)
        if weight_map:
            entries = [(algo, float(weight_map.get(algo, 0.0))) for algo in algorithms]
            entries = [(algo, weight) for algo, weight in entries if weight > 0.0]
        else:
            entries = []
        if entries:
            denom = sum(weight for _, weight in entries)
            weighted_sum = None
            for algo, weight in entries:
                expr = col(f"prob_{algo}_{emotion}") * lit(weight)
                weighted_sum = expr if weighted_sum is None else weighted_sum + expr
            ensemble_prob = weighted_sum / lit(denom)
        else:
            prob_cols = [col(f"prob_{algo}_{emotion}") for algo in algorithms]
            ensemble_prob = sum(prob_cols) / len(prob_cols)
        combined = combined.withColumn(f"ensemble_prob_{emotion}", ensemble_prob)

    ensemble_prob_cols = [col(f"ensemble_prob_{emotion}") for emotion in emotions]
    combined = combined.withColumn("probabilities", array(*ensemble_prob_cols))

    threshold_lookup = {
        emotion: (thresholds or {}).get(emotion, decision_threshold)
        for emotion in emotions
    }
    prediction_cols = [
        when(
            col(f"ensemble_prob_{emotion}") >= lit(threshold_lookup[emotion]),
            1.0,
        ).otherwise(0.0)
        for emotion in emotions
    ]
    combined = combined.withColumn("predictions", array(*prediction_cols))

    combined = combined.cache()
    combined_count = combined.count()
    print(f"  {dataset_name} combined dataframe ready with {combined_count} examples")
    return combined


def tune_thresholds(
    validation_df: DataFrame,
    emotions: Iterable[str],
    base_threshold: float,
) -> Dict[str, float]:
    """Grid-search per-emotion thresholds that maximise F1 on a validation split."""
    emotions = list(emotions)
    candidate_values = {round(value, 2) for value in [i / 100 for i in range(5, 91, 5)]}
    candidate_values.add(round(base_threshold, 2))

    tuned: Dict[str, float] = {}
    for idx, emotion in enumerate(emotions):
        rows = validation_df.select(
            col("labels")[idx].alias("label"),
            col("probabilities")[idx].alias("prob"),
        ).collect()
        best_threshold = base_threshold
        best_f1 = -1.0
        for threshold in sorted(candidate_values):
            tp = fp = fn = 0
            for row in rows:
                label = row["label"]
                prob = row["prob"]
                predicted = 1 if prob >= threshold else 0
                if label == 1.0 and predicted == 1:
                    tp += 1
                elif label == 0.0 and predicted == 1:
                    fp += 1
                elif label == 1.0 and predicted == 0:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            if f1 > best_f1 or (f1 == best_f1 and threshold < best_threshold):
                best_f1 = f1
                best_threshold = threshold
        tuned[emotion] = round(float(best_threshold), 3)
        print(f"  {emotion}: best_threshold={best_threshold:.3f} (f1={best_f1:.3f})")
    return tuned
