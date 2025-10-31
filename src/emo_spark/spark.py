"""Spark session utilities."""

from __future__ import annotations

from pyspark.sql import SparkSession

from .config import RuntimeConfig


def build_spark_session(config: RuntimeConfig) -> SparkSession:
    """Create a SparkSession configured for local testing or EMR.

    Configures Spark with appropriate settings based on environment:
    - Local: Uses local[*] master
    - EMR: Uses cluster settings with optimized shuffle partitions

    Args:
        config: Runtime configuration with app name, master, and environment.

    Returns:
        Configured SparkSession ready for use.
    """

    builder = SparkSession.builder.appName(config.app_name)
    if config.master:
        builder = builder.master(config.master)
    elif config.environment == "local":
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()

    if config.environment != "local":
        spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    return spark
