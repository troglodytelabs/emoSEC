"""End-to-end orchestration for the emoSpark project."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Iterable, List, Optional, Tuple

from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from .config import RuntimeConfig
from .constants import TARGET_LABELS
from .data import (
    attach_label_array,
    ensure_repartition,
    load_goemotions,
    load_lexicon,
    load_vad_lexicon,
    resolve_paths,
    save_split,
    stratified_split,
)
from .evaluation import EvaluationManager, save_predictions
from .features import FeatureBuilder
from .metrics import EvaluationResult
from .models import ModelSet, ModelTrainer, save_model_set
from .spark import build_spark_session

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Container for all outputs from the complete pipeline run."""

    config: RuntimeConfig
    model_results: Dict[str, ModelSet]
    evaluation_results: Dict[str, Dict[str, EvaluationResult]]
    feature_pipeline_path: Optional[str]
    test_set_path: str
    demo_samples_path: Optional[str]
    tuned_thresholds: Dict[str, Dict[str, float]]
    threshold_paths: Dict[str, Optional[str]]


@dataclass
class FeatureStageArtifacts:
    """Artifacts produced (or reused) from the feature engineering stage."""

    train_features: DataFrame
    validation_features: DataFrame
    test_features: DataFrame
    raw_test_df: DataFrame
    feature_pipeline_path: Optional[str]
    test_set_path: str
    resumed: bool


@dataclass
class _ResumedFeatureData:
    """Helper container for cached feature datasets."""

    train: DataFrame
    validation: DataFrame
    test: DataFrame
    pipeline_path: Optional[str]
    holdout_test: Optional[DataFrame]


def run_pipeline(
    config: RuntimeConfig,
    *,
    models_to_train: Optional[Iterable[str]] = None,
    include_train_metrics: bool = True,
    save_feature_datasets: bool = True,
    spark: Optional[SparkSession] = None,
    stop_spark: bool = True,
) -> PipelineArtifacts:
    """Run the complete training, evaluation, and export workflow."""

    logging.basicConfig(level=logging.INFO)

    spark_provided = spark is not None
    spark_session = spark or build_spark_session(config)

    try:
        feature_stage = _prepare_or_resume_features(
            spark_session,
            config,
            save_feature_datasets=save_feature_datasets,
        )

        train_features = feature_stage.train_features
        val_features = feature_stage.validation_features
        test_features = feature_stage.test_features
        test_df = feature_stage.raw_test_df
        feature_pipeline_path = feature_stage.feature_pipeline_path
        test_set_path = feature_stage.test_set_path

        if feature_stage.resumed:
            logger.info("Resumed feature datasets from previous run")

        # ============================================================
        # STEP 5: MODEL TRAINING (ONE-VS-REST STRATEGY)
        # ============================================================
        trainer = ModelTrainer(config, TARGET_LABELS)
        models_requested = (
            list(models_to_train)
            if models_to_train
            else [
                "logistic_regression",
                "linear_svm",
                "naive_bayes",
                "random_forest",
            ]
        )

        model_results: Dict[str, ModelSet] = {}
        evaluation_results: Dict[str, Dict[str, EvaluationResult]] = {}
        tuned_thresholds: Dict[str, Dict[str, float]] = {}
        threshold_paths: Dict[str, Optional[str]] = {}

        for model_type in models_requested:
            logger.info("Processing model family: %s", model_type)

            resumed_model_set = _try_resume_model_set(
                spark_session,
                config,
                model_type,
                include_train_metrics,
                trainer.label_cols,
            )

            newly_trained = False
            if resumed_model_set is not None:
                model_set = resumed_model_set
                logger.info(
                    "Reusing existing predictions for %s; skipping retraining",
                    model_type,
                )
            else:
                newly_trained = True
                logger.info("Training model family: %s", model_type)

                if model_type == "logistic_regression":
                    model_set = trainer.train_logistic_regression(
                        train_features,
                        val_features,
                        test_features,
                        generate_train_predictions=include_train_metrics,
                    )
                elif model_type == "linear_svm":
                    model_set = trainer.train_linear_svm(
                        train_features,
                        val_features,
                        test_features,
                        generate_train_predictions=include_train_metrics,
                    )
                elif model_type == "naive_bayes":
                    model_set = trainer.train_naive_bayes(
                        train_features,
                        val_features,
                        test_features,
                        generate_train_predictions=include_train_metrics,
                    )
                elif model_type == "random_forest":
                    model_set = trainer.train_random_forest(
                        train_features,
                        val_features,
                        test_features,
                        generate_train_predictions=include_train_metrics,
                    )
                else:
                    raise ValueError(f"Unsupported model type requested: {model_type}")

            model_results[model_type] = model_set

            # ============================================================
            # STEP 6: SAVE MODELS AND PREDICTIONS
            # ============================================================
            if newly_trained:
                model_dir = _build_output_path(config.output_path, "models", model_type)
                logger.info("Saving trained %s models under %s", model_type, model_dir)
                save_model_set(model_set, model_dir)

                prediction_dir = _build_output_path(config.output_path, "predictions")
                save_predictions(
                    model_set.validation_predictions,
                    prediction_dir,
                    model_type,
                    "validation",
                )
                save_predictions(
                    model_set.test_predictions, prediction_dir, model_type, "test"
                )
                if include_train_metrics and model_set.train_predictions is not None:
                    save_predictions(
                        model_set.train_predictions, prediction_dir, model_type, "train"
                    )

            # ============================================================
            # STEP 7: EVALUATE MODEL PERFORMANCE
            # ============================================================
            thresholds_used, metrics_map, thresholds_path = _evaluate_and_record(
                model_set,
                config,
                include_train_metrics,
            )
            evaluation_results[model_type] = metrics_map
            tuned_thresholds[model_type] = thresholds_used
            threshold_paths[model_type] = thresholds_path

        base_model_types = list(model_results.keys())
        if len(base_model_types) > 1:
            logger.info(
                "Combining base models via majority vote: %s",
                ", ".join(base_model_types),
            )
            try:
                voting_model_set = _build_majority_vote_ensemble(
                    {name: model_results[name] for name in base_model_types},
                    trainer.label_cols,
                    include_train_metrics,
                )
            except ValueError as exc:
                logger.warning("Skipping majority vote ensemble creation: %s", exc)
            else:
                model_results["majority_vote"] = voting_model_set

                prediction_dir = _build_output_path(config.output_path, "predictions")
                save_predictions(
                    voting_model_set.validation_predictions,
                    prediction_dir,
                    "majority_vote",
                    "validation",
                )
                save_predictions(
                    voting_model_set.test_predictions,
                    prediction_dir,
                    "majority_vote",
                    "test",
                )
                if (
                    include_train_metrics
                    and voting_model_set.train_predictions is not None
                ):
                    save_predictions(
                        voting_model_set.train_predictions,
                        prediction_dir,
                        "majority_vote",
                        "train",
                    )

                thresholds_used, metrics_map, thresholds_path = _evaluate_and_record(
                    voting_model_set,
                    config,
                    include_train_metrics,
                )
                evaluation_results["majority_vote"] = metrics_map
                tuned_thresholds["majority_vote"] = thresholds_used
                threshold_paths["majority_vote"] = thresholds_path

        demo_samples_path = _export_demo_samples(test_df, config)

        if config.cache_intermediate:
            train_features.unpersist()
            val_features.unpersist()
            test_features.unpersist()

        return PipelineArtifacts(
            config=config,
            model_results=model_results,
            evaluation_results=evaluation_results,
            feature_pipeline_path=feature_pipeline_path,
            test_set_path=test_set_path,
            demo_samples_path=demo_samples_path,
            tuned_thresholds=tuned_thresholds,
            threshold_paths=threshold_paths,
        )
    finally:
        if not spark_provided and stop_spark:
            spark_session.stop()


def _prepare_feature_dataset(df: DataFrame) -> DataFrame:
    """Select only essential columns for model training.

    Args:
        df: Transformed DataFrame with many columns.

    Returns:
        DataFrame with text, features, and label columns only.
    """
    base = df
    if "example_id" not in base.columns:
        base = base.withColumn("example_id", F.monotonically_increasing_id())

    keep_cols = ["example_id", "text", "features"] + TARGET_LABELS
    available = [col for col in keep_cols if col in base.columns]
    return base.select(*available)


def _prepare_or_resume_features(
    spark: SparkSession,
    config: RuntimeConfig,
    *,
    save_feature_datasets: bool,
) -> FeatureStageArtifacts:
    """Load cached feature datasets or rebuild them from raw data."""

    resumed = _resume_feature_datasets(spark, config)
    test_set_path = _build_output_path(config.output_path, "holdout", "test_set")

    if resumed is not None:
        logger.info("Reusing cached feature datasets from previous run")

        train_features = resumed.train
        val_features = resumed.validation
        test_features = resumed.test

        if "label_array" not in train_features.columns:
            train_features = attach_label_array(train_features)
        if "label_array" not in val_features.columns:
            val_features = attach_label_array(val_features)
        if "label_array" not in test_features.columns:
            test_features = attach_label_array(test_features)

        if config.cache_intermediate:
            train_features = train_features.persist(StorageLevel.MEMORY_AND_DISK)
            val_features = val_features.persist(StorageLevel.MEMORY_AND_DISK)
            test_features = test_features.persist(StorageLevel.MEMORY_AND_DISK)

        raw_test_df = resumed.holdout_test
        if raw_test_df is None:
            available = [
                col for col in ["text", *TARGET_LABELS] if col in test_features.columns
            ]
            raw_test_df = (
                test_features.select(*available) if available else test_features
            )

        logger.info(
            "Feature set row counts -> train: %d, validation: %d, test: %d",
            train_features.count(),
            val_features.count(),
            test_features.count(),
        )

        return FeatureStageArtifacts(
            train_features=train_features,
            validation_features=val_features,
            test_features=test_features,
            raw_test_df=raw_test_df,
            feature_pipeline_path=resumed.pipeline_path,
            test_set_path=test_set_path,
            resumed=True,
        )

    # No cached features were available; rebuild from raw inputs
    paths = resolve_paths(config)
    logger.info("Loading GoEmotions datasets from %s", paths.goemotions_files)
    raw_df = load_goemotions(spark, config, paths.goemotions_files)
    raw_df = ensure_repartition(raw_df, config)

    logger.info(
        "Splitting dataset into train/val/test with stratify=%s", config.stratify
    )
    train_df, val_df, test_df = stratified_split(raw_df, config)

    logger.info(
        "Split sizes -> train: %d, validation: %d, test: %d",
        train_df.count(),
        val_df.count(),
        test_df.count(),
    )

    _ensure_local_dir(test_set_path)
    save_split(test_df, test_set_path)

    logger.info("Loading NRC lexicon from %s", paths.nrc_lexicon)
    lexicon = load_lexicon(spark, paths.nrc_lexicon)

    try:
        logger.info("Loading NRC VAD lexicon from %s", paths.nrc_vad_lexicon)
        vad_lexicon = load_vad_lexicon(spark, paths.nrc_vad_lexicon)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Unable to load VAD lexicon (%s); proceeding with empty", exc)
        vad_lexicon: Dict[str, Tuple[float, float, float]] = {}

    feature_builder = FeatureBuilder(config, lexicon, vad_lexicon)
    logger.info("Fitting feature pipeline on training data")
    feature_builder.fit(train_df)

    logger.info("Generating train/validation/test feature sets")
    train_features = _prepare_feature_dataset(feature_builder.transform(train_df))
    val_features = _prepare_feature_dataset(feature_builder.transform(val_df))
    test_features = _prepare_feature_dataset(feature_builder.transform(test_df))

    feature_dim = _infer_feature_dim(train_features)
    if feature_dim:
        logger.info(
            "Assembled feature vector dimensionality: %d (%s)",
            feature_dim,
            ", ".join(feature_builder.vector_columns),
        )

    train_features = attach_label_array(train_features)
    val_features = attach_label_array(val_features)
    test_features = attach_label_array(test_features)

    if config.cache_intermediate:
        logger.info("Caching feature datasets for reuse")
        train_features = train_features.persist(StorageLevel.MEMORY_AND_DISK)
        val_features = val_features.persist(StorageLevel.MEMORY_AND_DISK)
        test_features = test_features.persist(StorageLevel.MEMORY_AND_DISK)

    logger.info(
        "Feature set row counts -> train: %d, validation: %d, test: %d",
        train_features.count(),
        val_features.count(),
        test_features.count(),
    )

    feature_pipeline_path = None
    if feature_builder.pipeline_model is not None:
        feature_pipeline_path = _build_output_path(
            config.output_path, "models", "feature_pipeline"
        )
        logger.info("Saving fitted feature pipeline to %s", feature_pipeline_path)
        if not feature_pipeline_path.startswith("s3://"):
            _ensure_local_dir(feature_pipeline_path)
        feature_builder.pipeline_model.write().overwrite().save(feature_pipeline_path)

    if save_feature_datasets:
        _save_feature_datasets(train_features, val_features, test_features, config)

    raw_test_df = test_df.select("text", *TARGET_LABELS)

    return FeatureStageArtifacts(
        train_features=train_features,
        validation_features=val_features,
        test_features=test_features,
        raw_test_df=raw_test_df,
        feature_pipeline_path=feature_pipeline_path,
        test_set_path=test_set_path,
        resumed=False,
    )


def _resume_feature_datasets(
    spark: SparkSession, config: RuntimeConfig
) -> Optional[_ResumedFeatureData]:
    """Load previously saved feature datasets if they are all available."""

    feature_dir = _build_output_path(config.output_path, "features")
    train_df = _load_feature_split(spark, feature_dir, "train")
    val_df = _load_feature_split(spark, feature_dir, "validation")
    test_df = _load_feature_split(spark, feature_dir, "test")

    if train_df is None or val_df is None or test_df is None:
        return None

    pipeline_path = _build_output_path(config.output_path, "models", "feature_pipeline")
    if not pipeline_path.startswith("s3://") and not os.path.exists(pipeline_path):
        pipeline_path = None

    holdout_path = _build_output_path(config.output_path, "holdout", "test_set")
    holdout_df = _load_holdout_split(spark, holdout_path)

    return _ResumedFeatureData(
        train=train_df,
        validation=val_df,
        test=test_df,
        pipeline_path=pipeline_path,
        holdout_test=holdout_df,
    )


def _load_feature_split(
    spark: SparkSession, feature_dir: str, split: str
) -> Optional[DataFrame]:
    """Load a cached feature split if it exists."""

    path = os.path.join(feature_dir, split)
    if not feature_dir.startswith("s3://") and not os.path.exists(path):
        return None

    try:
        return spark.read.parquet(path)
    except (AnalysisException, FileNotFoundError):
        logger.debug("Feature parquet missing for split '%s'", split)
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to load feature split %s due to %s", split, exc)
        return None


def _load_holdout_split(spark: SparkSession, path: str) -> Optional[DataFrame]:
    """Load the cached holdout test set if available."""

    if not path.startswith("s3://") and not os.path.exists(path):
        return None

    try:
        return spark.read.parquet(path)
    except (AnalysisException, FileNotFoundError):
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to load holdout test set due to %s", exc)
        return None


def _save_feature_datasets(
    train_features: DataFrame,
    val_features: DataFrame,
    test_features: DataFrame,
    config: RuntimeConfig,
) -> None:
    """Persist intermediate feature datasets for reuse."""

    feature_dir = _build_output_path(config.output_path, "features")
    logger.info("Saving feature datasets to %s", feature_dir)
    _ensure_local_dir(feature_dir)

    train_features.write.mode("overwrite").parquet(os.path.join(feature_dir, "train"))
    val_features.write.mode("overwrite").parquet(
        os.path.join(feature_dir, "validation")
    )
    test_features.write.mode("overwrite").parquet(os.path.join(feature_dir, "test"))


def _try_resume_model_set(
    spark: SparkSession,
    config: RuntimeConfig,
    model_type: str,
    include_train_predictions: bool,
    label_cols: Iterable[str],
) -> Optional[ModelSet]:
    """Attempt to rebuild a ModelSet from previously persisted predictions."""

    prediction_root = _build_output_path(config.output_path, "predictions")
    validation_df = _load_prediction_split(
        spark, prediction_root, model_type, "validation"
    )
    test_df = _load_prediction_split(spark, prediction_root, model_type, "test")

    if validation_df is None or test_df is None:
        return None

    train_df = None
    if include_train_predictions:
        train_df = _load_prediction_split(spark, prediction_root, model_type, "train")

    label_summaries: Dict[str, Dict[str, float]] = {
        str(label): {} for label in label_cols
    }

    return ModelSet(
        model_type=model_type,
        models={},
        validation_predictions=validation_df,
        test_predictions=test_df,
        training_summaries=label_summaries,
        train_predictions=train_df,
    )


def _load_prediction_split(
    spark: SparkSession, prediction_root: str, model_type: str, split: str
) -> Optional[DataFrame]:
    """Load a persisted prediction split if available."""

    path = os.path.join(prediction_root, model_type, split)

    if not prediction_root.startswith("s3://") and not os.path.exists(path):
        return None

    try:
        return spark.read.parquet(path)
    except (AnalysisException, FileNotFoundError):
        logger.debug("Prediction parquet missing for %s/%s", model_type, split)
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Unable to load predictions for %s/%s due to %s", model_type, split, exc
        )
        return None


def _assemble_majority_vote_dataset(
    model_sets: Dict[str, ModelSet],
    dataset_attr: str,
    label_cols: Iterable[str],
) -> Optional[DataFrame]:
    """Combine stored predictions from all base models for a specific split."""

    frames: List[tuple[str, DataFrame]] = []
    for model_type, model_set in model_sets.items():
        df = getattr(model_set, dataset_attr, None)
        if df is None:
            continue
        frames.append((model_type, df))

    if len(frames) < 2:
        return None

    combined = _join_prediction_frames(frames, label_cols)
    model_types = [name for name, _ in frames]
    combined = _append_majority_vote_columns(combined, label_cols, model_types)
    return _select_majority_vote_columns(combined, label_cols)


def _join_prediction_frames(
    frames: List[tuple[str, DataFrame]], label_cols: Iterable[str]
) -> DataFrame:
    """Join per-model prediction frames on shared example identifiers."""

    base_type, base_df = frames[0]
    combined = _trim_prediction_columns(
        base_df, base_type, label_cols, include_labels=True
    )
    join_keys = _resolve_join_keys(combined)

    for model_type, df in frames[1:]:
        trimmed = _trim_prediction_columns(
            df, model_type, label_cols, include_labels=False
        )
        missing_keys = [key for key in join_keys if key not in trimmed.columns]
        if missing_keys:
            raise ValueError(
                f"Predictions for model '{model_type}' are missing join keys: {missing_keys}"
            )
        combined = combined.join(trimmed, on=join_keys, how="inner")

    return combined


def _trim_prediction_columns(
    df: DataFrame,
    model_type: str,
    label_cols: Iterable[str],
    *,
    include_labels: bool,
) -> DataFrame:
    """Select the minimum set of columns required from a prediction DataFrame."""

    join_cols: List[str] = []
    if "example_id" in df.columns:
        join_cols.append("example_id")
    if not join_cols and "text" in df.columns:
        join_cols.append("text")

    if not join_cols:
        raise ValueError("Prediction DataFrame lacks 'example_id' or 'text' columns")

    selected: List[str] = list(dict.fromkeys(join_cols))

    if include_labels:
        selected.extend([col for col in label_cols if col in df.columns])

    for label in label_cols:
        for candidate in (
            f"prob_{model_type}_{label}",
            f"raw_{model_type}_{label}",
            f"pred_{model_type}_{label}",
        ):
            if candidate in df.columns:
                selected.append(candidate)

    return df.select(*selected)


def _resolve_join_keys(df: DataFrame) -> List[str]:
    """Resolve stable join keys present in a prediction DataFrame."""

    if "example_id" in df.columns:
        return ["example_id"]
    if "text" in df.columns:
        return ["text"]
    raise ValueError("Unable to determine join keys for majority vote assembly")


def _append_majority_vote_columns(
    df: DataFrame, label_cols: Iterable[str], model_types: Iterable[str]
) -> DataFrame:
    """Append majority vote predictions for each label."""

    result = df
    model_list = [name for name in model_types if name]
    if not model_list:
        return result

    for label in label_cols:
        vote_inputs = []
        for model_type in model_list:
            prob_col = f"prob_{model_type}_{label}"
            pred_col = f"pred_{model_type}_{label}"
            if prob_col in result.columns:
                vote_inputs.append(_extract_numeric_column(result, prob_col))
            elif pred_col in result.columns:
                vote_inputs.append(F.col(pred_col).cast("double"))

        if not vote_inputs:
            continue

        sum_expr = reduce(lambda acc, col: acc + col, vote_inputs[1:], vote_inputs[0])
        average_expr = sum_expr / float(len(vote_inputs))

        result = result.withColumn(
            f"prob_majority_vote_{label}", average_expr.cast("double")
        )
        result = result.withColumn(
            f"vote_share_majority_vote_{label}", average_expr.cast("double")
        )
        result = result.withColumn(
            f"pred_majority_vote_{label}",
            F.when(average_expr >= 0.5, F.lit(1.0)).otherwise(F.lit(0.0)),
        )

    return result


def _select_majority_vote_columns(
    df: DataFrame, label_cols: Iterable[str]
) -> DataFrame:
    """Select relevant columns for majority vote outputs."""

    base_cols = [col for col in ["example_id", "text"] if col in df.columns]
    base_cols.extend([col for col in label_cols if col in df.columns])
    vote_cols = []
    for label in label_cols:
        vote_cols.extend(
            [
                f"pred_majority_vote_{label}",
                f"prob_majority_vote_{label}",
                f"vote_share_majority_vote_{label}",
            ]
        )

    keep_cols = [col for col in base_cols + vote_cols if col in df.columns]
    return df.select(*keep_cols)


def _build_majority_vote_ensemble(
    model_sets: Dict[str, ModelSet],
    label_cols: Iterable[str],
    include_train_predictions: bool,
) -> ModelSet:
    """Construct a ModelSet representing a majority vote ensemble."""

    label_list = list(label_cols)
    base_model_types = list(model_sets.keys())

    train_predictions = (
        _assemble_majority_vote_dataset(model_sets, "train_predictions", label_list)
        if include_train_predictions
        else None
    )
    val_predictions = _assemble_majority_vote_dataset(
        model_sets, "validation_predictions", label_list
    )
    test_predictions = _assemble_majority_vote_dataset(
        model_sets, "test_predictions", label_list
    )

    if val_predictions is None or test_predictions is None:
        raise ValueError(
            "Majority vote ensemble requires predictions from at least two base models"
        )

    summaries = {
        label: {"model_votes": float(len(base_model_types))} for label in label_list
    }

    return ModelSet(
        model_type="majority_vote",
        models={},
        validation_predictions=val_predictions,
        test_predictions=test_predictions,
        training_summaries=summaries,
        train_predictions=train_predictions,
    )


def _extract_numeric_column(df: DataFrame, column_name: str):
    """Normalize vector/scalar prediction outputs to a double column."""

    if _is_vector_column(df, column_name):
        return F.element_at(vector_to_array(F.col(column_name)), -1).cast("double")
    return F.col(column_name).cast("double")


def _is_vector_column(df: DataFrame, column_name: str) -> bool:
    """Check whether a DataFrame column stores a vector."""

    field = next((item for item in df.schema.fields if item.name == column_name), None)
    if field is None:
        return False
    return isinstance(field.dataType, VectorUDT)


def _evaluate_and_record(
    model_set: ModelSet,
    config: RuntimeConfig,
    include_train_metrics: bool,
) -> Tuple[Dict[str, float], Dict[str, EvaluationResult], Optional[str]]:
    """Run evaluation with optional threshold tuning and persist results."""

    manager = EvaluationManager(
        TARGET_LABELS, thresholds=dict(config.probability_thresholds)
    )

    if config.auto_tune_thresholds:
        manager.tune_thresholds(
            model_set.validation_predictions,
            model_set.model_type,
            config.threshold_candidates,
        )

    metrics = manager.run_full_evaluation(
        model_set,
        _build_output_path(config.output_path, "evaluation"),
        include_train=include_train_metrics,
        train_predictions=model_set.train_predictions,
    )

    thresholds = manager.get_thresholds()
    thresholds_path = _persist_thresholds(
        thresholds,
        _build_output_path(config.output_path, "evaluation"),
        model_set.model_type,
    )

    metric_map = {item.dataset: item.metrics for item in metrics}
    return thresholds, metric_map, thresholds_path


def _persist_thresholds(
    thresholds: Dict[str, float], evaluation_dir: str, model_type: str
) -> Optional[str]:
    """Persist tuned thresholds to disk when running on local storage."""

    if not thresholds:
        return None

    threshold_dir = os.path.join(evaluation_dir, "thresholds")
    if threshold_dir.startswith("s3://"):
        logger.warning(
            "Skipping threshold persistence for remote path %s", threshold_dir
        )
        return None

    _ensure_local_dir(threshold_dir)
    path = os.path.join(threshold_dir, f"thresholds_{model_type}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    return path


def _infer_feature_dim(df: DataFrame) -> Optional[int]:
    """Extract feature vector dimensionality from DataFrame metadata.

    Args:
        df: DataFrame with features column.

    Returns:
        Number of features if metadata available, None otherwise.
    """
    if "features" not in df.columns:
        return None
    metadata = df.schema["features"].metadata or {}
    ml_attr = metadata.get("ml_attr", {}) if metadata else {}
    if not ml_attr:
        return None
    if "num_attrs" in ml_attr:
        return int(ml_attr["num_attrs"])
    if "numAttrs" in ml_attr:
        return int(ml_attr["numAttrs"])
    return None


def _export_demo_samples(test_df: DataFrame, config: RuntimeConfig) -> Optional[str]:
    """Export sample texts from test set for demo purposes.

    Args:
        test_df: Test DataFrame with text and labels.
        config: Runtime configuration with demo_sample_size.

    Returns:
        Path to saved demo samples JSON, or None if size is 0.
    """
    sample_size = config.demo_sample_size
    if sample_size <= 0:
        return None

    sample_df = test_df.select("text", *TARGET_LABELS).orderBy("text")
    sampled = sample_df.orderBy(F.rand(config.seed)).limit(sample_size)

    samples = [row.asDict(recursive=True) for row in sampled.collect()]

    demo_dir = _build_output_path(config.output_path, "demo")
    _ensure_local_dir(demo_dir)
    path = os.path.join(demo_dir, "demo_samples.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)
    return path


def _ensure_local_dir(path: str) -> None:
    """Create local directory if path is not S3.

    Args:
        path: Directory path to create.
    """
    if path.startswith("s3://"):
        return
    os.makedirs(path, exist_ok=True)


def _build_output_path(base: str, *parts: str) -> str:
    """Construct output path supporting both local and S3.

    Args:
        base: Base path (local directory or S3 prefix).
        parts: Path components to join.

    Returns:
        Complete output path with proper separators.
    """
    if base.startswith("s3://"):
        relative = "/".join(parts)
        return f"{base.rstrip('/')}/{relative}" if relative else base
    return os.path.abspath(os.path.join(base, *parts))
