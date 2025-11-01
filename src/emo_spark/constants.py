"""Project-wide constants for emoSpark.

This module defines all emotion taxonomies, label mappings, and configuration
constants used throughout the emoSpark pipeline.
"""

from __future__ import annotations

from typing import Dict, List

"""List of all 27 fine-grained emotion labels plus neutral from GoEmotions dataset."""
GOEMOTIONS_LABELS: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

"""Eight primary emotions from Plutchik's wheel of emotions."""
PLUTCHIK_EMOTIONS: List[str] = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

"""Mapping from Plutchik emotions to GoEmotions labels.
    
Used to project fine-grained GoEmotions annotations onto the
coarser Plutchik taxonomy for modeling.
"""
PLUTCHIK_TO_GOEMOTIONS: Dict[str, List[str]] = {
    "joy": [
        "amusement",
        "excitement",
        "joy",
        "optimism",
        "pride",
        "relief",
    ],
    "trust": [
        "admiration",
        "approval",
        "caring",
        "gratitude",
        "love",
    ],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "confusion"],
    "sadness": ["grief", "remorse", "sadness", "disappointment"],
    "disgust": ["disgust", "embarrassment"],
    "anger": ["anger", "annoyance", "disapproval"],
    "anticipation": ["desire", "curiosity"],
}


"""Ten emotion categories in the NRC Emotion Lexicon."""
NRC_EMOTIONS: List[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "positive",
    "negative",
]

"""Tuned probability thresholds for converting model probabilities to binary predictions.

These values are optimized to balance precision and recall for each emotion.
"""
DEFAULT_PROBABILITY_THRESHOLDS: Dict[str, float] = {
    "joy": 0.55,
    "trust": 0.5,
    "fear": 0.45,
    "surprise": 0.5,
    "sadness": 0.5,
    "disgust": 0.45,
    "anger": 0.5,
    "anticipation": 0.5,
}

"""Narrative descriptors for emotion storytelling in demo output.
    
Each emotion has a 'tone' adjective and 'headline' phrase for
generating human-readable emotional narratives.
"""
EMOTION_DESCRIPTORS: Dict[str, Dict[str, str]] = {
    "joy": {"tone": "uplifting", "headline": "bright bursts of joy"},
    "trust": {"tone": "reassuring", "headline": "steady notes of trust"},
    "fear": {"tone": "tense", "headline": "undercurrents of fear"},
    "surprise": {"tone": "curious", "headline": "sparks of surprise"},
    "sadness": {"tone": "reflective", "headline": "shadows of sadness"},
    "disgust": {"tone": "sharp", "headline": "flashes of disgust"},
    "anger": {"tone": "fiery", "headline": "streaks of anger"},
    "anticipation": {"tone": "eager", "headline": "stirrings of anticipation"},
}

# Primary target labels for multi-label classification
TARGET_LABELS: List[str] = PLUTCHIK_EMOTIONS

# Column name constants
DEFAULT_TEXT_COL = "text"  # Raw input text column
TOKEN_COL = "tokens"  # Tokenized text
STOPWORD_FREE_COL = "filtered_tokens"  # Tokens after stopword removal
