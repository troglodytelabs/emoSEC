# Scalable Emotion Classification for Affective Computing
### Multi-Label Text Classification with Hybrid Lexical–Dimensional Features

**Authors:** Devin Dyson, Madhur Deep Jain
**Date:** November 2025

This project implements a distributed emotion classification pipeline using Apache Spark, combining lexicon-based and statistical NLP features for affective computing research.
It classifies text (from the GoEmotions dataset) into Plutchik’s 8 primary emotions: *joy, sadness, anger, fear, surprise, disgust, trust,* and *anticipation.*

---

## Overview

The system performs large-scale, multi-label emotion classification by integrating both **lexical (NRC Emotion and VAD lexicons)** and **statistical (TF–IDF, n-grams)** features in a unified Spark ML pipeline.
It trains multiple classifiers—Logistic Regression, SVM, Naive Bayes, and Random Forest—and combines them via ensemble probability averaging and majority voting.

---

## Features

- Distributed, multi-label classification using Apache Spark MLlib
- Hybrid feature set:
  - TF–IDF with unigrams, bigrams, and trigrams
  - NRC Emotion Lexicon (raw counts and normalized ratios)
  - NRC Valence–Arousal–Dominance (VAD) Lexicon (mean, std, range)
  - Linguistic features (punctuation density, emphasis markers, word/char length)
- Multiple classifiers: Logistic Regression, SVM, Naive Bayes, Random Forest
- Ensemble aggregation via probability averaging and majority voting
- Per-emotion, micro- and macro-averaged metrics

---

## Requirements

### Python Environment
Recommended Python versions: **3.11 or 3.12**
Python 3.14 has known compatibility issues with PySpark.

```bash
# create virtual environment
python3 -m venv emoSpark-env
source emoSpark-env/bin/activate

# install dependencies
pip install -r requirements.txt
