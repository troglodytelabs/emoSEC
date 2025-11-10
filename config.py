"""Configuration constants for the emotion classification pipeline."""

PLUTCHIK_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

GOEMOTIONS_LABELS = [
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

EMOTION_MAPPING = {
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "gratitude": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "disgust": "disgust",
    "embarrassment": "disgust",
    "approval": "trust",
    "admiration": "trust",
    "caring": "trust",
    "curiosity": "anticipation",
    "desire": "anticipation",
    "neutral": None,
}

NRC_EMOTION_PATH = "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
NRC_VAD_PATH = "data/NRC-VAD-Lexicon-v2.1.txt"

SAMPLE_SIZE = 0.1
TRAIN_SPLIT = 0.8
TFIDF_FEATURES = 500
NGRAM_RANGE = 3
MAX_ITERATIONS = 10
REGULARIZATION = 0.01
DECISION_THRESHOLD = 0.25

USE_LOGISTIC_REGRESSION = True
USE_SVM = True
USE_NAIVE_BAYES = True
USE_RANDOM_FOREST = True
