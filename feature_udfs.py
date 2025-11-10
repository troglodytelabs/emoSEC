"""Factory helpers for Spark UDFs used in the feature pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

from config import EMOTION_MAPPING, GOEMOTIONS_LABELS, PLUTCHIK_EMOTIONS


# Feature extractors
def get_extract_nrc_udf(nrc_lexicon_bc: Any):
    """Return a UDF that computes NRC emotion features."""

    def extract_nrc_features(text: str):
        words = re.findall(r"\b[a-z]+\b", text.lower())
        word_count = len(words) if words else 1
        emotion_counts = {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

        for word in words:
            if word in nrc_lexicon_bc.value:
                for emotion in nrc_lexicon_bc.value[word]:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1.0

        raw_counts = [emotion_counts[e] for e in PLUTCHIK_EMOTIONS]
        normalized_ratios = [emotion_counts[e] / word_count for e in PLUTCHIK_EMOTIONS]
        return raw_counts + normalized_ratios

    return udf(extract_nrc_features, ArrayType(DoubleType()))


def get_extract_vad_udf(nrc_vad_bc: Any):
    """Return a UDF that computes NRC VAD statistics."""

    def extract_vad_features(text: str):
        words = re.findall(r"\b[a-z]+\b", text.lower())
        valence_scores: List[float] = []
        arousal_scores: List[float] = []
        dominance_scores: List[float] = []

        for word in words:
            if word in nrc_vad_bc.value:
                v, a, d = nrc_vad_bc.value[word]
                valence_scores.append(v)
                arousal_scores.append(a)
                dominance_scores.append(d)

        features = []
        for scores in [valence_scores, arousal_scores, dominance_scores]:
            if scores:
                mean_val = sum(scores) / len(scores)
                features.append(mean_val)
                if len(scores) > 1:
                    variance = sum((x - mean_val) ** 2 for x in scores) / len(scores)
                    std_val = variance**0.5
                else:
                    std_val = 0.0
                features.append(std_val)
                range_val = max(scores) - min(scores)
                features.append(range_val)
            else:
                features.extend([0.5, 0.0, 0.0])

        return features

    return udf(extract_vad_features, ArrayType(DoubleType()))


def get_extract_linguistic_udf():
    """Return a UDF that computes simple linguistic cue features."""

    def extract_linguistic_features(text: str, words: List[str]):
        features: List[float] = []
        word_count = len(words) if words else 1
        char_count = len(text) if text else 1
        features.append(float(word_count))
        features.append(float(char_count))
        features.append(text.count("!") / char_count)
        features.append(text.count("?") / char_count)
        features.append(text.count("...") / char_count)
        caps_count = sum(1 for char in text if char.isupper())
        features.append(caps_count / char_count)
        repeated = len(re.findall(r"(.)\1{2,}", text))
        features.append(repeated / char_count)
        return features

    return udf(extract_linguistic_features, ArrayType(DoubleType()))


# Label conversion helpers
def get_labels_udf():
    """Return a UDF that maps GoEmotions labels to Plutchik labels."""

    def get_plutchik_labels(goemotions_dict: Dict[str, int]):
        plutchik_set = set()
        for emotion in GOEMOTIONS_LABELS:
            if goemotions_dict.get(emotion, 0) != 1:
                continue
            mapped = EMOTION_MAPPING.get(emotion)
            if mapped:
                plutchik_set.add(mapped)
        return [
            1.0 if emotion in plutchik_set else 0.0 for emotion in PLUTCHIK_EMOTIONS
        ]

    return udf(get_plutchik_labels, ArrayType(DoubleType()))


# Feature combination helper
def get_combine_udf():
    """Return a UDF that concatenates all feature groups into a single vector."""

    def combine_features(tfidf, nrc, vad, linguistic):
        tfidf_dense = (
            tfidf.toArray().tolist() if hasattr(tfidf, "toArray") else list(tfidf)
        )
        return Vectors.dense(tfidf_dense + nrc + vad + linguistic)

    return udf(combine_features, VectorUDT())
