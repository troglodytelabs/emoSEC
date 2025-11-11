"""Helpers for loading and broadcasting NRC lexicons."""

from __future__ import annotations

from typing import Dict, List, Tuple

from pyspark.sql import SparkSession


def load_nrc_emotion_lexicon(
    path: str, target_emotions: List[str]
) -> Dict[str, List[str]]:
    """Load the NRC emotion lexicon filtered to target emotions."""
    nrc_emotion_lex: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, association = parts
            if emotion in {"positive", "negative"}:
                continue
            if int(association) != 1:
                continue
            if emotion not in target_emotions:
                continue
            nrc_emotion_lex.setdefault(word, []).append(emotion)
    return nrc_emotion_lex


def load_nrc_vad_lexicon(path: str) -> Dict[str, Tuple[float, float, float]]:
    """Load the NRC VAD lexicon as a mapping word -> (valence, arousal, dominance)."""
    nrc_vad_lex: Dict[str, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            word = parts[0]
            try:
                raw_valence = float(parts[1])
                raw_arousal = float(parts[2])
                raw_dominance = float(parts[3])
            except ValueError:
                continue
            # the original lexicon stores scores in [-1, 1]; shift to [0, 1] so that downstream feature vectors remain non-negative for algorithms such as multinomial naive bayes.
            valence = (raw_valence + 1.0) / 2.0
            arousal = (raw_arousal + 1.0) / 2.0
            dominance = (raw_dominance + 1.0) / 2.0
            nrc_vad_lex[word] = (valence, arousal, dominance)
    return nrc_vad_lex


def broadcast_lexicons(
    spark: SparkSession,
    nrc_emotion_lex: Dict[str, List[str]],
    nrc_vad_lex: Dict[str, Tuple[float, float, float]],
):
    """Broadcast lexicons to the Spark cluster."""
    nrc_lexicon_bc = spark.sparkContext.broadcast(nrc_emotion_lex)
    nrc_vad_bc = spark.sparkContext.broadcast(nrc_vad_lex)
    return nrc_lexicon_bc, nrc_vad_bc
