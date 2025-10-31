"""emoSpark: Scalable multi-label emotion classification with PySpark."""

from importlib.metadata import PackageNotFoundError, version

from .config import RuntimeConfig
from .pipeline import run_pipeline

try:  # pragma: no cover
    __version__ = version("emo-spark")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "RuntimeConfig", "run_pipeline"]
