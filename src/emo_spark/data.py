"""Data ingestion and preparation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import RuntimeConfig
from .constants import (
    GOEMOTIONS_LABELS,
    PLUTCHIK_EMOTIONS,
    PLUTCHIK_TO_GOEMOTIONS,
    TARGET_LABELS,
)


@dataclass
class DataPaths:
    """Container for file paths to all required data assets.

    Attributes:
        goemotions_files: List of paths to GoEmotions CSV shards.
        nrc_lexicon: Path to NRC Emotion Lexicon file.
        nrc_vad_lexicon: Path to NRC VAD (Valence-Arousal-Dominance) Lexicon file.
    """

    goemotions_files: List[str]
    nrc_lexicon: str
    nrc_vad_lexicon: str


def resolve_paths(config: RuntimeConfig) -> DataPaths:
    """Construct absolute file paths from configuration, supporting local and S3.

    Args:
        config: Runtime configuration containing input_path.

    Returns:
        DataPaths object with resolved file paths.
    """
    base_path = config.input_path
    if not base_path.startswith("s3://"):
        base_path = os.path.abspath(base_path)

    go_paths: List[str] = []
    for suffix in ("goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"):
        go_paths.append(os.path.join(base_path, suffix))

    nrc_path = os.path.join(base_path, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

    vad_path = os.path.join(base_path, "NRC-VAD-Lexicon-v2.1.txt")

    return DataPaths(
        goemotions_files=go_paths,
        nrc_lexicon=nrc_path,
        nrc_vad_lexicon=vad_path,
    )


def load_goemotions(
    spark: SparkSession, config: RuntimeConfig, paths: Iterable[str]
) -> DataFrame:
    """Load and unify all GoEmotions CSV files.

    Performs:
    - Loads multiple CSV shards with multiline support
    - Filters unclear examples
    - Converts labels to double type
    - Aggregates rater annotations
    - Projects to Plutchik emotions
    - Optional sampling

    Args:
        spark: Active SparkSession.
        config: Runtime configuration with sampling parameters.
        paths: Iterable of file paths to GoEmotions CSV files.

    Returns:
        Unified DataFrame with 'text' column and Plutchik emotion labels.
    """

    frames: list[DataFrame] = []
    for path in paths:
        frames.append(
            spark.read.option("header", True)
            .option("multiLine", True)
            .option("escape", '"')
            .csv(path)
        )

    df = frames[0]
    for other in frames[1:]:
        df = df.unionByName(other)

    df = filter_unclear_examples(df)

    columns = ["text"] + GOEMOTIONS_LABELS
    if "id" in df.columns:
        columns.insert(0, "id")
    df = df.select(*columns)

    for label in GOEMOTIONS_LABELS:
        df = df.withColumn(label, F.col(label).cast("double"))

    df = aggregate_rater_annotations(df)

    df = project_to_plutchik(df)

    if config.sample_fraction is not None and 0 < config.sample_fraction < 1:
        df = df.sample(
            withReplacement=False, fraction=config.sample_fraction, seed=config.seed
        )

    return df.dropna(subset=["text"]).withColumn("text", F.col("text").cast("string"))


def compute_primary_label(df: DataFrame) -> DataFrame:
    """Derive a primary label used for stratified splitting.

    Selects the first positive Plutchik emotion as the primary label,
    or 'neutral' if none are positive.

    Args:
        df: DataFrame with Plutchik emotion columns.

    Returns:
        DataFrame with added 'primary_label' column.
    """

    def _first_positive(*cols: float) -> str:
        for idx, value in enumerate(cols):
            if value and value > 0:
                return TARGET_LABELS[idx]
        return "neutral"

    udf = F.udf(_first_positive, T.StringType())

    label_cols = [F.col(label) for label in TARGET_LABELS]
    df = df.withColumn("primary_label", udf(*label_cols))
    return df


def stratified_split(
    df: DataFrame, config: RuntimeConfig
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Perform a stratified train/val/test split on primary label.

    Uses primary_label for stratification to ensure rare emotion classes
    are represented across all splits. Falls back to random split if
    stratification is disabled.

    Args:
        df: Input DataFrame with emotion labels.
        config: Runtime configuration with split ratios and stratify flag.

    Returns:
        Tuple of (train_df, validation_df, test_df).
    """

    if not config.stratify:
        splits = df.randomSplit(
            [config.train_ratio, config.val_ratio, config.test_ratio], seed=config.seed
        )
        return tuple(splits)  # type: ignore[return-value]

    df = compute_primary_label(df)

    window = Window.partitionBy("primary_label").orderBy(F.rand(seed=config.seed))
    ranked = df.withColumn("rank", F.percent_rank().over(window))

    train = ranked.filter(F.col("rank") <= config.train_ratio).drop("rank")
    val = ranked.filter(
        (F.col("rank") > config.train_ratio)
        & (F.col("rank") <= config.train_ratio + config.val_ratio)
    ).drop("rank")
    test = ranked.filter(F.col("rank") > config.train_ratio + config.val_ratio).drop(
        "rank"
    )

    return (
        train.drop("primary_label"),
        val.drop("primary_label"),
        test.drop("primary_label"),
    )


def save_split(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    """Save a DataFrame split to Parquet format.

    Args:
        df: DataFrame to save.
        path: Output path (local or S3).
        mode: Write mode, defaults to 'overwrite'.
    """
    df.write.mode(mode).parquet(path)


def load_lexicon(spark: SparkSession, path: str) -> Dict[str, Dict[str, int]]:
    """Load NRC lexicon into a broadcastable dictionary.

    Reads tab-separated NRC Emotion Lexicon and converts to a nested
    dictionary mapping terms to their associated emotions.

    Args:
        spark: Active SparkSession.
        path: Path to NRC lexicon file.

    Returns:
        Dictionary mapping lowercase terms to emotion associations.
        Format: {term: {emotion: 1, ...}}
    """

    schema = T.StructType(
        [
            T.StructField("term", T.StringType(), False),
            T.StructField("emotion", T.StringType(), False),
            T.StructField("association", T.IntegerType(), False),
        ]
    )

    df = spark.read.csv(path, sep="\t", schema=schema)

    grouped = (
        df.filter(F.col("association") == 1)
        .groupBy("term")
        .agg(F.collect_set("emotion").alias("emotions"))
        .collect()
    )

    lexicon: Dict[str, Dict[str, int]] = {}
    for row in grouped:
        lexicon[row["term"].lower()] = {emotion: 1 for emotion in row["emotions"]}

    return lexicon


def load_vad_lexicon(
    spark: SparkSession, path: str
) -> Dict[str, Tuple[float, float, float]]:
    """Load NRC Valence-Arousal-Dominance lexicon.

    Reads the NRC VAD lexicon with automatic delimiter detection
    and normalizes column names to lowercase.
    Args:
        spark: Active SparkSession.
        path: Path to NRC VAD lexicon file.

    Returns:
        Dictionary mapping lowercase terms to (valence, arousal, dominance) tuples.

    Raises:
        FileNotFoundError: If file doesn't exist at local path.
        ValueError: If required columns are missing.
    """

    if not path.startswith("s3://") and not os.path.exists(path):
        raise FileNotFoundError(
            f"NRC VAD lexicon not found at {path}. See prepare_data.sh for instructions."
        )

    reader = spark.read.option("header", True).option("inferSchema", True)
    try:
        df = reader.option("sep", "\t").csv(path)
        if len(df.columns) < 4:
            df = reader.csv(path)
    except Exception:  # pragma: no cover - defensive fallback
        df = reader.csv(path)

    renamed = df
    for column in df.columns:
        lower = column.strip().lower()
        if lower != column:
            renamed = renamed.withColumnRenamed(column, lower)

    required = {"term", "valence", "arousal", "dominance"}
    if not required.issubset({col.lower() for col in renamed.columns}):
        raise ValueError(
            "Expected columns term, valence, arousal, dominance in VAD lexicon"
        )

    normalized = renamed.selectExpr(
        "lower(term) as term",
        "cast(valence as double) as valence",
        "cast(arousal as double) as arousal",
        "cast(dominance as double) as dominance",
    ).dropna(subset=["term"])

    entries = normalized.collect()

    lexicon: Dict[str, Tuple[float, float, float]] = {}
    for row in entries:
        lexicon[row["term"]] = (
            float(row["valence"] or 0.0),
            float(row["arousal"] or 0.0),
            float(row["dominance"] or 0.0),
        )

    return lexicon


def attach_label_array(df: DataFrame) -> DataFrame:
    """Add an array column with ordered label values.

    Creates 'label_array' column containing all TARGET_LABELS values
    in a single array for convenience in some operations.

    Args:
        df: DataFrame with individual label columns.

    Returns:
        DataFrame with added 'label_array' column.
    """
    return df.withColumn(
        "label_array", F.array([F.col(label) for label in TARGET_LABELS])
    )


def ensure_repartition(df: DataFrame, config: RuntimeConfig) -> DataFrame:
    """Optionally repartition DataFrame based on configuration.

    Args:
        df: Input DataFrame.
        config: Runtime configuration with repartition_target setting.

    Returns:
        Repartitioned DataFrame if target specified, otherwise original.
    """
    if config.repartition_target:
        return df.repartition(config.repartition_target)
    return df


def project_to_plutchik(df: DataFrame) -> DataFrame:
    """Aggregate GoEmotions labels into Plutchik primary emotions.

    Maps fine-grained GoEmotions labels to 8 Plutchik emotions using
    the PLUTCHIK_TO_GOEMOTIONS mapping. Takes maximum value when
    multiple source labels map to one target.

    Args:
        df: DataFrame with GoEmotions label columns.

    Returns:
        DataFrame with Plutchik emotion columns and 'text'.
    """

    projected = df
    for target, source_labels in PLUTCHIK_TO_GOEMOTIONS.items():
        available = [F.col(label) for label in source_labels if label in df.columns]
        if not available:
            projected = projected.withColumn(target, F.lit(0.0))
            continue
        if len(available) == 1:
            combined = available[0]
        else:
            combined = F.greatest(*available)
        projected = projected.withColumn(
            target,
            F.coalesce(combined.cast("double"), F.lit(0.0)),
        )

    select_cols = ["text"] + PLUTCHIK_EMOTIONS
    return projected.select(*select_cols)


def filter_unclear_examples(df: DataFrame) -> DataFrame:
    """Remove rows flagged as very unclear by annotators.

    Filters out examples where 'example_very_unclear' is true/1/yes.

    Args:
        df: Input DataFrame, potentially with 'example_very_unclear' column.

    Returns:
        Filtered DataFrame with unclear examples removed.
    """
    if "example_very_unclear" not in df.columns:
        return df

    normalized_value = F.lower(F.trim(F.col("example_very_unclear").cast("string")))
    flagged = normalized_value.isin("true", "1", "yes")
    cleaned = df.filter(~flagged)
    if "example_very_unclear" in cleaned.columns:
        cleaned = cleaned.drop("example_very_unclear")
    return cleaned


def aggregate_rater_annotations(df: DataFrame) -> DataFrame:
    """Collapse per-rater annotations into a single multi-label row per example.

    When multiple raters annotate the same text, this aggregates their
    labels using max(), producing one row per unique text.

    Args:
        df: DataFrame with multiple annotations from multiple raters.

    Returns:
        Deduplicated DataFrame with aggregated labels.
    """

    group_cols = ["id", "text"]
    aggregations = [F.max(F.col(label)).alias(label) for label in GOEMOTIONS_LABELS]
    aggregated = df.groupBy(*group_cols).agg(*aggregations)
    aggregated = aggregated.drop("id")

    return aggregated
