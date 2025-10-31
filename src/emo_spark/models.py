"""Model training utilities for multi-label emotion classification."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from pyspark.ml import Transformer
from pyspark.ml.classification import (
    LinearSVC,
    LinearSVCModel,
    LogisticRegression,
    LogisticRegressionModel,
    NaiveBayes,
    NaiveBayesModel,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.linalg import SparseVector, Vector, VectorUDT, Vectors
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import RuntimeConfig
from .constants import TARGET_LABELS

logger = logging.getLogger(__name__)


def _clip_vector_to_non_negative(vector: Vector | None) -> Vector:
    """Clamp vector values to non-negative domain while preserving sparsity."""

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


_CLIP_TO_NON_NEGATIVE_UDF = F.udf(_clip_vector_to_non_negative, VectorUDT())


@dataclass
class ModelSet:
    """Group of per-emotion binary models for a given algorithm.

    Attributes:
        model_type: Name of model algorithm (e.g., 'logistic_regression').
        models: Dictionary mapping emotion labels to trained models.
        validation_predictions: DataFrame with validation set predictions.
        test_predictions: DataFrame with test set predictions.
        training_summaries: Dictionary of training metrics per emotion.
        train_predictions: Optional DataFrame with train set predictions.
    """

    model_type: str
    models: Dict[str, Transformer]
    validation_predictions: DataFrame
    test_predictions: DataFrame
    training_summaries: Dict[str, Dict[str, float]]
    train_predictions: Optional[DataFrame] = None


class ModelTrainer:
    """Train collections of binary classifiers for each target emotion label.

    Implements one-vs-rest strategy for multi-label classification.
    Supports multiple algorithms with per-emotion model training.

    Attributes:
        config: Runtime configuration with hyperparameters.
        label_cols: List of emotion labels to train models for.
    """

    def __init__(
        self, config: RuntimeConfig, label_cols: Iterable[str] = TARGET_LABELS
    ):
        """Initialize model trainer.

        Args:
            config: Runtime configuration.
            label_cols: Emotion labels to train models for.
        """
        self.config = config
        self.label_cols = list(label_cols)

    # ------------------------------------------------------------------
    # Logistic Regression without cross-validation
    # ------------------------------------------------------------------
    def train_logistic_regression(
        self,
        train_df: DataFrame,
        val_df: DataFrame,
        test_df: DataFrame,
        *,
        generate_train_predictions: bool = False,
    ) -> ModelSet:
        """Train logistic regression models with fixed hyperparameters.

        Uses a single configuration (no grid search) for faster, simpler
        training while still producing per-emotion classifiers.

        Args:
            train_df: Training DataFrame with features and labels.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            generate_train_predictions: Whether to generate train predictions.

        Returns:
            ModelSet with trained models and predictions.
        """

        model_type = "logistic_regression"
        cached_train = train_df.cache()

        models: Dict[str, LogisticRegressionModel] = {}
        summaries: Dict[str, Dict[str, float]] = {}
        total_labels = len(self.label_cols)

        for index, label in enumerate(self.label_cols, start=1):
            logger.info(
                "[LR] Fitting classifier for '%s' (%d/%d)", label, index, total_labels
            )
            estimator = LogisticRegression(
                featuresCol="features",
                labelCol=label,
                predictionCol=self._col_name(model_type, label, "pred"),
                probabilityCol=self._col_name(model_type, label, "prob"),
                rawPredictionCol=self._col_name(model_type, label, "raw"),
                regParam=self.config.lr_reg_param,
                elasticNetParam=self.config.lr_elastic_net_param,
                maxIter=self.config.lr_max_iter,
            )

            model = estimator.fit(cached_train)
            models[label] = model
            summaries[label] = {
                "regParam": float(model.getRegParam()),
                "elasticNetParam": float(model.getElasticNetParam()),
                "maxIter": float(model.getMaxIter()),
            }
            try:
                training_summary = model.summary
                objective_history = getattr(training_summary, "objectiveHistory", [])
                if objective_history:
                    summaries[label]["objective"] = float(objective_history[-1])
            except AttributeError:
                pass

            logger.info(
                "[LR] Completed '%s' (reg=%.4g, elastic=%.2f, maxIter=%d)",
                label,
                float(model.getRegParam()),
                float(model.getElasticNetParam()),
                int(model.getMaxIter()),
            )

        train_predictions = (
            self._apply_models(cached_train, models)
            if generate_train_predictions
            else None
        )

        cached_train.unpersist()

        val_predictions = self._apply_models(val_df, models)
        test_predictions = self._apply_models(test_df, models)

        return ModelSet(
            model_type=model_type,
            models=models,
            validation_predictions=val_predictions,
            test_predictions=test_predictions,
            training_summaries=summaries,
            train_predictions=train_predictions,
        )

    # ------------------------------------------------------------------
    # Linear SVM (hinge loss)
    # ------------------------------------------------------------------
    def train_linear_svm(
        self,
        train_df: DataFrame,
        val_df: DataFrame,
        test_df: DataFrame,
        *,
        generate_train_predictions: bool = False,
    ) -> ModelSet:
        """Train linear SVM models with hinge loss.

        Uses fixed hyperparameters (maxIter=50, regParam=0.1).

        Args:
            train_df: Training DataFrame with features and labels.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            generate_train_predictions: Whether to generate train predictions.

        Returns:
            ModelSet with trained models and predictions.
        """
        model_type = "linear_svm"
        cached_train = train_df.cache() if generate_train_predictions else train_df

        models: Dict[str, LinearSVCModel] = {}
        summaries: Dict[str, Dict[str, float]] = {}
        total_labels = len(self.label_cols)

        for index, label in enumerate(self.label_cols, start=1):
            logger.info(
                "[SVM] Training classifier for '%s' (%d/%d)", label, index, total_labels
            )
            estimator = LinearSVC(
                featuresCol="features",
                labelCol=label,
                predictionCol=self._col_name(model_type, label, "pred"),
                rawPredictionCol=self._col_name(model_type, label, "raw"),
                maxIter=50,
                regParam=0.1,
            )
            model = estimator.fit(cached_train)
            models[label] = model
            summaries[label] = {
                "maxIter": float(model.getMaxIter()),
                "regParam": float(model.getRegParam()),
            }
            logger.info(
                "[SVM] Completed '%s' (maxIter=%d, reg=%.4g)",
                label,
                int(model.getMaxIter()),
                float(model.getRegParam()),
            )

        train_predictions = (
            self._apply_models(cached_train, models)
            if generate_train_predictions
            else None
        )

        if generate_train_predictions:
            cached_train.unpersist()

        val_predictions = self._apply_models(val_df, models)
        test_predictions = self._apply_models(test_df, models)

        return ModelSet(
            model_type=model_type,
            models=models,
            validation_predictions=val_predictions,
            test_predictions=test_predictions,
            training_summaries=summaries,
            train_predictions=train_predictions,
        )

    # ------------------------------------------------------------------
    # Naive Bayes (multinomial variant)
    # ------------------------------------------------------------------
    def train_naive_bayes(
        self,
        train_df: DataFrame,
        val_df: DataFrame,
        test_df: DataFrame,
        *,
        generate_train_predictions: bool = False,
    ) -> ModelSet:
        """Train Naive Bayes models with non-negative multinomial likelihood.

        Clips negative feature values and swaps to the multinomial variant so
        that the estimator can ingest the mixed TF-IDF feature space safely.

        Args:
            train_df: Training DataFrame with features and labels.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            generate_train_predictions: Whether to generate train predictions.

        Returns:
            ModelSet with trained models and predictions.
        """
        model_type = "naive_bayes"

        train_nb = train_df.withColumn(
            "_nb_features", _CLIP_TO_NON_NEGATIVE_UDF(F.col("features"))
        ).cache()
        val_nb = val_df.withColumn(
            "_nb_features", _CLIP_TO_NON_NEGATIVE_UDF(F.col("features"))
        )
        test_nb = test_df.withColumn(
            "_nb_features", _CLIP_TO_NON_NEGATIVE_UDF(F.col("features"))
        )

        models: Dict[str, NaiveBayesModel] = {}
        summaries: Dict[str, Dict[str, float]] = {}
        total_labels = len(self.label_cols)

        for index, label in enumerate(self.label_cols, start=1):
            logger.info(
                "[NB] Training classifier for '%s' (%d/%d)", label, index, total_labels
            )
            estimator = NaiveBayes(
                featuresCol="_nb_features",
                labelCol=label,
                predictionCol=self._col_name(model_type, label, "pred"),
                probabilityCol=self._col_name(model_type, label, "prob"),
                rawPredictionCol=self._col_name(model_type, label, "raw"),
                modelType="multinomial",
            )
            model = estimator.fit(train_nb)
            models[label] = model
            summaries[label] = {"smoothing": float(model.getSmoothing())}
            logger.info(
                "[NB] Completed '%s' (smoothing=%.3f)",
                label,
                float(model.getSmoothing()),
            )

        train_predictions = (
            self._drop_column(self._apply_models(train_nb, models), "_nb_features")
            if generate_train_predictions
            else None
        )

        train_nb.unpersist()

        val_predictions = self._drop_column(
            self._apply_models(val_nb, models), "_nb_features"
        )
        test_predictions = self._drop_column(
            self._apply_models(test_nb, models), "_nb_features"
        )

        return ModelSet(
            model_type=model_type,
            models=models,
            validation_predictions=val_predictions,
            test_predictions=test_predictions,
            training_summaries=summaries,
            train_predictions=train_predictions,
        )

    # ------------------------------------------------------------------
    # Random Forest ensemble
    # ------------------------------------------------------------------
    def train_random_forest(
        self,
        train_df: DataFrame,
        val_df: DataFrame,
        test_df: DataFrame,
        *,
        generate_train_predictions: bool = False,
    ) -> ModelSet:
        """Train Random Forest ensemble models.

        Uses 100 trees with maxDepth=12 and auto feature subset selection.

        Args:
            train_df: Training DataFrame with features and labels.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            generate_train_predictions: Whether to generate train predictions.

        Returns:
            ModelSet with trained models and predictions.
        """
        model_type = "random_forest"
        cached_train = train_df.cache() if generate_train_predictions else train_df

        models: Dict[str, RandomForestClassificationModel] = {}
        summaries: Dict[str, Dict[str, float]] = {}
        total_labels = len(self.label_cols)

        for index, label in enumerate(self.label_cols, start=1):
            logger.info(
                "[RF] Training classifier for '%s' (%d/%d)", label, index, total_labels
            )
            estimator = RandomForestClassifier(
                featuresCol="features",
                labelCol=label,
                predictionCol=self._col_name(model_type, label, "pred"),
                probabilityCol=self._col_name(model_type, label, "prob"),
                rawPredictionCol=self._col_name(model_type, label, "raw"),
                numTrees=100,
                maxDepth=12,
                subsamplingRate=0.8,
                featureSubsetStrategy="auto",
            )
            model = estimator.fit(cached_train)
            models[label] = model
            num_trees_attr = getattr(model, "getNumTrees", None)
            num_trees_value = (
                num_trees_attr() if callable(num_trees_attr) else num_trees_attr
            )
            if num_trees_value is None:
                num_trees_value = getattr(model, "numTrees", 0)
            max_depth_attr = getattr(model, "getMaxDepth", None)
            max_depth_value = (
                max_depth_attr() if callable(max_depth_attr) else max_depth_attr
            )
            if max_depth_value is None:
                max_depth_value = getattr(model, "maxDepth", 0)
            summaries[label] = {
                "numTrees": float(num_trees_value),
                "maxDepth": float(max_depth_value),
            }
            logger.info(
                "[RF] Completed '%s' (trees=%d, maxDepth=%d)",
                label,
                int(num_trees_value),
                int(max_depth_value),
            )

        train_predictions = (
            self._apply_models(cached_train, models)
            if generate_train_predictions
            else None
        )

        if generate_train_predictions:
            cached_train.unpersist()

        val_predictions = self._apply_models(val_df, models)
        test_predictions = self._apply_models(test_df, models)

        return ModelSet(
            model_type=model_type,
            models=models,
            validation_predictions=val_predictions,
            test_predictions=test_predictions,
            training_summaries=summaries,
            train_predictions=train_predictions,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_models(self, df: DataFrame, models: Dict[str, Transformer]) -> DataFrame:
        """Apply all emotion-specific models to generate predictions.

        Args:
            df: Input DataFrame with features.
            models: Dictionary of models keyed by emotion label.

        Returns:
            DataFrame with all prediction columns added.
        """
        logger.info(
            "Running inference for %d emotions with model type '%s'",
            len(models),
            next(iter(models.values())).__class__.__name__ if models else "unknown",
        )
        result = df
        for label in self.label_cols:
            model = models[label]
            result = model.transform(result)
        return result

    @staticmethod
    def _col_name(model_type: str, label: str, prefix: str) -> str:
        """Generate consistent column names for predictions.

        Args:
            model_type: Model algorithm name.
            label: Emotion label.
            prefix: Column prefix (pred, prob, raw).

        Returns:
            Formatted column name.
        """
        return f"{prefix}_{model_type}_{label}"

    @staticmethod
    def _drop_column(df: DataFrame, column_name: str) -> DataFrame:
        """Drop temporary helper columns when present."""

        return df.drop(column_name) if column_name in df.columns else df


def save_model_set(model_set: ModelSet, output_dir: str) -> None:
    """Save all models in a ModelSet to disk.

    Creates subdirectories for each emotion label and saves the
    corresponding model using Spark's model persistence.

    Args:
        model_set: ModelSet containing trained models.
        output_dir: Base directory for saving models.
    """
    os.makedirs(output_dir, exist_ok=True)
    for label, model in model_set.models.items():
        path = os.path.join(output_dir, label)
        model.write().overwrite().save(path)
