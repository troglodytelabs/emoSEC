"""Interactive demo helpers for the emoSpark project."""

from __future__ import annotations

import argparse
import json
import os
import numbers
from functools import reduce
from typing import Dict, Iterable, List, Optional

from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LinearSVCModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestClassificationModel,
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import SparseVector, Vector, VectorUDT, Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import RuntimeConfig
from .constants import EMOTION_DESCRIPTORS, TARGET_LABELS
from .spark import build_spark_session

MODEL_CLASS_MAP = {
    "logistic_regression": LogisticRegressionModel,
    "linear_svm": LinearSVCModel,
    "naive_bayes": NaiveBayesModel,
    "random_forest": RandomForestClassificationModel,
}

ENSEMBLE_MODEL = "majority_vote"

_NB_FEATURE_COL = "_nb_features"


def _clip_vector_to_non_negative(vector: Vector | None) -> Vector:
    """Clamp vector values to the non-negative domain for Naive Bayes."""

    if vector is None:
        return Vectors.sparse(0, [], [])

    if isinstance(vector, SparseVector):
        indices = []
        values = []
        for idx, val in zip(vector.indices, vector.values):
            if val > 0.0:
                indices.append(int(idx))
                values.append(float(val))
        return Vectors.sparse(vector.size, indices, values)

    dense_values = [float(val) if val > 0.0 else 0.0 for val in vector]
    return Vectors.dense(dense_values)


_NB_FEATURE_UDF = F.udf(_clip_vector_to_non_negative, VectorUDT())


class DemoEngine:
    """Load saved models and run predictions for interactive exploration.

    Provides end-to-end prediction pipeline:
    - Loads feature pipeline and trained models
    - Applies feature transformations
    - Runs predictions with threshold awareness
    - Generates narrative storytelling

    Attributes:
        spark: Active SparkSession.
        model_type: Model algorithm name.
        labels: List of emotion labels.
        model_base_path: Base directory containing models.
        thresholds: Probability thresholds per label.
        feature_pipeline: Loaded PipelineModel for features.
    models_by_type: Loaded models grouped by model family.
    ensemble_model_types: Base model names used for majority vote.
    """

    def __init__(
        self,
        spark: SparkSession,
        model_base_path: str,
        model_type: str,
        labels: Iterable[str] = TARGET_LABELS,
        thresholds: Optional[Dict[str, float]] = None,
        feature_pipeline_path: Optional[str] = None,
    ) -> None:
        """Initialize demo engine with saved artifacts.

        Args:
            spark: Active SparkSession.
            model_base_path: Base directory containing models subdirectory.
            model_type: Model algorithm to use for predictions.
            labels: Emotion labels to predict.
            thresholds: Probability thresholds per label.
            feature_pipeline_path: Path to feature pipeline, defaults to
                model_base_path/feature_pipeline.
        """
        self.spark = spark
        self.model_type = model_type
        self.labels = list(labels)
        self.model_base_path = model_base_path
        self.thresholds = thresholds or {}
        self.feature_pipeline = self._load_feature_pipeline(
            feature_pipeline_path or os.path.join(model_base_path, "feature_pipeline")
        )
        self.models_by_type: Dict[str, Dict[str, object]] = {}
        self.ensemble_model_types: List[str] = []

        if model_type == ENSEMBLE_MODEL:
            self.ensemble_model_types = self._discover_available_models()
            if len(self.ensemble_model_types) < 2:
                raise ValueError(
                    "Majority vote ensemble requires at least two trained base models"
                )
            for base_type in self.ensemble_model_types:
                self.models_by_type[base_type] = self._load_model_family(base_type)
        else:
            self.models_by_type[model_type] = self._load_model_family(model_type)

    def _load_feature_pipeline(self, path: str) -> PipelineModel:
        """Load saved feature pipeline from disk.

        Args:
            path: Path to saved PipelineModel.

        Returns:
            Loaded PipelineModel.
        """
        return PipelineModel.load(path)

    def _load_model_family(self, model_type: str) -> Dict[str, object]:
        """Load all per-emotion models from disk.

        Args:
            model_type: Model algorithm name.

        Returns:
            Dictionary mapping labels to loaded models.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in MODEL_CLASS_MAP:
            raise ValueError(f"Unsupported model_type for demo: {model_type}")
        model_cls = MODEL_CLASS_MAP[model_type]
        models: Dict[str, object] = {}
        for label in self.labels:
            path = os.path.join(self.model_base_path, model_type, label)
            try:
                models[label] = model_cls.load(path)
            except Exception as exc:  # pragma: no cover - Spark loader errors vary
                raise FileNotFoundError(
                    f"Unable to load {model_type} model for '{label}' at {path}"
                ) from exc
        return models

    def _discover_available_models(self) -> List[str]:
        """Discover trained model families available on disk."""

        try:
            entries = os.listdir(self.model_base_path)
        except FileNotFoundError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"Cannot inspect model directory {self.model_base_path}: {exc}"
            ) from exc

        discovered: List[str] = []
        for entry in sorted(entries):
            if entry == "feature_pipeline" or entry not in MODEL_CLASS_MAP:
                continue
            path = os.path.join(self.model_base_path, entry)
            if os.path.isdir(path):
                discovered.append(entry)
        return discovered

    def predict_texts(
        self,
        texts: Iterable[str],
        *,
        base_threshold: Optional[float] = None,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, object]]:
        """Generate predictions and narratives for input texts.

        For each text:
        - Applies feature pipeline
        - Runs all emotion-specific models
        - Applies probability thresholds
        - Generates narrative story

        Args:
            texts: Input texts to predict on.
            base_threshold: Override all thresholds with this value.
            threshold_overrides: Override specific label thresholds.

        Returns:
            List of prediction dictionaries with text, predictions,
            positive labels, top labels, and narrative story.
        """
        df = self.spark.createDataFrame([(text,) for text in texts], ["text"])
        feature_df = self.feature_pipeline.transform(df)
        feature_df = feature_df.select("text", "features")
        if self.model_type == ENSEMBLE_MODEL:
            predictions = self._run_majority_vote(feature_df)
        else:
            predictions = self._apply_models(
                feature_df,
                self.models_by_type[self.model_type],
                model_type=self.model_type,
            )
        rows = predictions.collect()

        results: List[Dict[str, object]] = []
        thresholds = self._resolve_thresholds(base_threshold, threshold_overrides)
        for row in rows:
            row_dict = row.asDict()
            entry: Dict[str, object] = {
                "text": row_dict["text"],
                "predictions": [],
                "thresholds": dict(thresholds),
            }
            positive_labels: List[str] = []
            scored_labels: List[Dict[str, object]] = []
            for label in self.labels:
                threshold = thresholds.get(label, 0.5)
                pred_col = self._col_name("pred", label)
                prob_col = self._col_name("prob", label)
                raw_col = self._col_name("raw", label)
                vote_share_col = self._col_name("vote_share", label)

                pred_value = float(row_dict[pred_col])
                probability = None
                if prob_col in row_dict:
                    vector = row_dict[prob_col]
                    if isinstance(vector, numbers.Number):
                        probability = float(vector)
                    elif hasattr(vector, "__getitem__") and len(vector) > 1:
                        probability = float(vector[1])
                    elif hasattr(vector, "__getitem__"):
                        probability = float(vector[0])
                raw_value = None
                if raw_col in row_dict:
                    raw_data = row_dict[raw_col]
                    if hasattr(raw_data, "__getitem__") and len(raw_data) > 1:
                        raw_value = float(raw_data[1])
                    elif hasattr(raw_data, "__getitem__"):
                        raw_value = float(raw_data[0])
                    else:
                        raw_value = float(raw_data)
                if probability is None and vote_share_col in row_dict:
                    probability = float(row_dict[vote_share_col])

                is_positive = False
                scored_labels.append(
                    {
                        "label": label,
                        "predicted": False,
                        "score": probability,
                        "raw": raw_value,
                        "threshold": threshold,
                    }
                )
                if probability is not None:
                    if probability >= threshold:
                        positive_labels.append(label)
                        is_positive = True
                elif pred_value >= 0.5:
                    positive_labels.append(label)
                    is_positive = True

                scored_labels[-1]["predicted"] = is_positive

            entry["predictions"] = scored_labels
            entry["positive_labels"] = positive_labels
            entry["top_labels"] = _top_labels(scored_labels, top_n=5)
            entry["story"] = _compose_story(scored_labels, thresholds)
            results.append(entry)
        return results

    def _run_majority_vote(self, df: DataFrame) -> DataFrame:
        """Apply base models and aggregate predictions via majority vote."""

        result = df
        for model_type in self.ensemble_model_types:
            models = self.models_by_type.get(model_type)
            if not models:
                raise ValueError(f"Base models not loaded for '{model_type}'")
            result = self._apply_models(result, models, model_type=model_type)

        result = self._append_vote_columns(result)
        return self._select_vote_columns(result)

    def _apply_models(
        self,
        df: DataFrame,
        models: Dict[str, object],
        *,
        model_type: str,
    ) -> DataFrame:
        """Apply all loaded models of a specific family to a feature DataFrame."""

        result = df
        nb_features_added = False
        if model_type == "naive_bayes":
            result, nb_features_added = self._ensure_nb_features(result)

        for label in self.labels:
            model = models[label]
            result = model.transform(result)

        if nb_features_added:
            result = result.drop(_NB_FEATURE_COL)
        return result

    def _append_vote_columns(self, df: DataFrame) -> DataFrame:
        """Append vote share and ensemble predictions for each label."""

        result = df
        if not self.ensemble_model_types:
            return result

        for label in self.labels:
            vote_inputs = []
            for model_type in self.ensemble_model_types:
                prob_col = self._col_name("prob", label, model_type)
                pred_col = self._col_name("pred", label, model_type)
                if prob_col in result.columns:
                    vote_inputs.append(self._extract_numeric_column(result, prob_col))
                elif pred_col in result.columns:
                    vote_inputs.append(F.col(pred_col).cast("double"))

            if not vote_inputs:
                continue

            sum_expr = reduce(
                lambda acc, col: acc + col, vote_inputs[1:], vote_inputs[0]
            )
            average_expr = sum_expr / float(len(vote_inputs))

            result = result.withColumn(
                self._col_name("vote_share", label), average_expr.cast("double")
            )
            result = result.withColumn(
                self._col_name("pred", label),
                F.when(average_expr >= 0.5, F.lit(1.0)).otherwise(F.lit(0.0)),
            )

        return result

    def _select_vote_columns(self, df: DataFrame) -> DataFrame:
        """Select relevant columns for majority vote outputs."""

        keep_cols = ["text"] + list(self.labels)
        for label in self.labels:
            keep_cols.extend(
                [
                    self._col_name("pred", label),
                    self._col_name("vote_share", label),
                ]
            )

        existing = [col for col in keep_cols if col in df.columns]
        return df.select(*existing)

    def _ensure_nb_features(self, df: DataFrame) -> tuple[DataFrame, bool]:
        """Ensure helper column for Naive Bayes predictions is available."""

        if _NB_FEATURE_COL in df.columns:
            return df, False
        return (
            df.withColumn(_NB_FEATURE_COL, _NB_FEATURE_UDF(F.col("features"))),
            True,
        )

    def _extract_numeric_column(self, df: DataFrame, column_name: str):
        """Project probability/raw columns to double scalars."""

        if self._is_vector_column(df, column_name):
            return F.element_at(vector_to_array(F.col(column_name)), -1).cast("double")
        return F.col(column_name).cast("double")

    def _is_vector_column(self, df: DataFrame, column_name: str) -> bool:
        """Check whether a DataFrame column stores a vector."""

        field = next(
            (field for field in df.schema.fields if field.name == column_name), None
        )
        if field is None:
            return False
        return isinstance(field.dataType, VectorUDT)

    def _col_name(
        self, prefix: str, label: str, model_type: Optional[str] = None
    ) -> str:
        """Generate column name for model outputs.

        Args:
            prefix: Column prefix (pred, prob, raw).
            label: Emotion label.

        Returns:
            Formatted column name.
        """
        model = model_type or self.model_type
        return f"{prefix}_{model}_{label}"

    def _resolve_thresholds(
        self,
        base_threshold: Optional[float],
        overrides: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Resolve effective thresholds from config and overrides.

        Precedence: overrides > base_threshold > config thresholds > 0.5

        Args:
            base_threshold: Override all thresholds.
            overrides: Override specific label thresholds.

        Returns:
            Dictionary of resolved thresholds per label.
        """
        thresholds = {label: self.thresholds.get(label, 0.5) for label in self.labels}
        if base_threshold is not None:
            thresholds = {label: float(base_threshold) for label in self.labels}
        if overrides:
            for label, value in overrides.items():
                if label in thresholds:
                    thresholds[label] = float(value)
        return thresholds


def _top_labels(predictions: List[Dict[str, object]], top_n: int) -> List[str]:
    """Extract top N predicted labels by score.

    Args:
        predictions: List of prediction dictionaries with scores.
        top_n: Number of top labels to return.

    Returns:
        List of top N emotion labels by score.
    """
    scored = [p for p in predictions if p.get("score") is not None]
    if not scored:
        return [p["label"] for p in predictions if p.get("predicted")][:top_n]
    ordered = sorted(scored, key=lambda item: item.get("score", 0.0), reverse=True)
    return [item["label"] for item in ordered[:top_n]]


def _compose_story(
    scored_labels: List[Dict[str, object]], thresholds: Dict[str, float]
) -> str:
    """Compose narrative story from prediction scores.

    Generates human-readable emotional narrative using EMOTION_DESCRIPTORS
    and probability scores above threshold.

    Args:
        scored_labels: List of prediction dictionaries with scores.
        thresholds: Probability thresholds per label.

    Returns:
        Narrative string describing emotional content.
    """
    confident = [
        item
        for item in scored_labels
        if item.get("score") is not None
        and item.get("score", 0.0) >= thresholds.get(item["label"], 0.5)
    ]

    if not confident:
        positives = [item for item in scored_labels if item.get("predicted")]
        if not positives:
            return "Emotional signals stay below all configured thresholds."
        primary = positives[0]
        descriptor = EMOTION_DESCRIPTORS.get(primary["label"], {})
        headline = descriptor.get("headline", primary["label"]).capitalize()
        return f"{headline} emerge despite muted probability scores."

    ordered = sorted(confident, key=lambda item: item.get("score", 0.0), reverse=True)
    leader = ordered[0]
    leader_descriptor = EMOTION_DESCRIPTORS.get(leader["label"], {})
    leader_headline = leader_descriptor.get("headline", leader["label"]).capitalize()
    leader_tone = leader_descriptor.get("tone", leader["label"])
    leader_score = leader.get("score") or 0.0
    leader_percent = f"{leader_score * 100:.0f}%"

    supporting = ordered[1] if len(ordered) > 1 else None
    if supporting:
        support_descriptor = EMOTION_DESCRIPTORS.get(supporting["label"], {})
        support_tone = support_descriptor.get("tone", supporting["label"])
        support_score = supporting.get("score") or 0.0
        support_percent = f"{support_score * 100:.0f}%"
        return (
            f"{leader_headline} lead the narrative ({leader_percent}), "
            f"with a {support_tone} undertone from {supporting['label']} ({support_percent})."
        )

    return f"{leader_headline} set a {leader_tone} tone at {leader_percent}."


def _load_demo_samples(base_path: str) -> List[str]:
    """Load pre-saved demo samples from JSON file.

    Args:
        base_path: Base output directory containing demo subdirectory.

    Returns:
        List of sample text strings.

    Raises:
        FileNotFoundError: If demo samples file doesn't exist.
    """
    demo_path = os.path.join(base_path, "demo", "demo_samples.json")
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demo samples not found at {demo_path}")
    with open(demo_path, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    return [sample["text"] for sample in samples]


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for demo interface.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Run interactive emoSpark predictions")
    parser.add_argument(
        "--model",
        default=ENSEMBLE_MODEL,
        choices=list(MODEL_CLASS_MAP.keys()) + [ENSEMBLE_MODEL],
        help="Model family to use for predictions (majority_vote blends trained models)",
    )
    parser.add_argument(
        "--output-path", help="Base output directory containing models", default=None
    )
    parser.add_argument(
        "--text", action="append", dest="texts", help="Custom text to score"
    )
    parser.add_argument(
        "--use-demo-samples", action="store_true", help="Use saved demo samples"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override probability threshold for all emotions",
    )
    parser.add_argument(
        "--thresholds-json",
        help="JSON object mapping emotions to probability thresholds",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Main CLI entrypoint for interactive demo.

    Loads trained models and runs predictions on provided texts
    or demo samples, outputting results as JSON.

    Args:
        argv: Command line arguments, defaults to sys.argv if None.
    """
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    config = RuntimeConfig()
    base_output = args.output_path or config.output_path

    texts: List[str] = args.texts or []
    if args.use_demo_samples:
        texts.extend(_load_demo_samples(base_output))

    if not texts:
        parser.error("Provide --text or --use-demo-samples to generate predictions")

    threshold_overrides: Optional[Dict[str, float]] = None
    if args.thresholds_json:
        try:
            raw_overrides = json.loads(args.thresholds_json)
            if isinstance(raw_overrides, dict):
                threshold_overrides = {
                    str(key): float(value) for key, value in raw_overrides.items()
                }
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid --thresholds-json payload: {exc}")

    tuned_thresholds = _load_tuned_thresholds(base_output, args.model)
    threshold_config = dict(config.probability_thresholds)
    if tuned_thresholds:
        threshold_config.update(tuned_thresholds)

    spark = build_spark_session(config)
    try:
        engine = DemoEngine(
            spark,
            model_base_path=os.path.join(base_output, "models"),
            model_type=args.model,
            thresholds=threshold_config,
            feature_pipeline_path=os.path.join(
                base_output, "models", "feature_pipeline"
            ),
        )
        results = engine.predict_texts(
            texts,
            base_threshold=args.threshold,
            threshold_overrides=threshold_overrides,
        )
        print(json.dumps(results, indent=2))
    finally:
        spark.stop()


def _load_tuned_thresholds(
    base_output: str, model_type: str
) -> Optional[Dict[str, float]]:
    """Load tuned thresholds emitted by the training pipeline, if available."""

    path = os.path.join(
        base_output,
        "evaluation",
        "thresholds",
        f"thresholds_{model_type}.json",
    )
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {str(key): float(value) for key, value in payload.items()}
    except (OSError, ValueError):
        return None
    return None


if __name__ == "__main__":  # pragma: no cover
    main()
