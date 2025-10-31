## emoSpark

Scalable multi-label emotion classification on the GoEmotions corpus using Apache Spark. The project supports both local experimentation (single laptop) and distributed training on AWS EMR. It combines classical feature engineering (TF-IDF, NRC lexicon signals, NRC VAD affective intensities, linguistic cues) with multiple model architectures and ships with an interactive storytelling demo.

## Contents

- [Overview](#overview)
- [Data Flow and Model Training Pipeline](#data-flow-and-model-training-pipeline)
- [Data Assets](#data-assets)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Running on AWS EMR](#running-on-aws-emr)
- [Inspecting Results](#inspecting-results)
- [Interactive Demo](#interactive-demo)
- [Repository Layout](#repository-layout)

## Overview

- **Feature engineering pipeline** combining n-gram TF-IDF, NRC emotion lexicon counts, NRC Valence/Arousal/Dominance statistics, and custom linguistic indicators (`src/emo_spark/features.py`).
- **Model zoo** with per-emotion one-vs-rest training for logistic regression, linear SVM, naive Bayes, random forest, plus a majority-vote ensemble that blends available base families using their stored predictions—no extra model training required (`src/emo_spark/pipeline.py`).
- **Evaluation suite** computing Hamming loss, subset accuracy, micro/macro F1, and per-emotion metrics with configurable probability thresholds stored as JSON/Parquet (`src/emo_spark/evaluation.py`).
- **Interactive storytelling demo** (`python -m emo_spark.demo`) that loads trained artifacts, respects per-emotion probability thresholds, and narrates emotional storylines for custom text.
- **Cloud friendly**: all I/O paths resolve for local storage and S3 buckets, toggled via environment variables.

## Data Flow and Model Training Pipeline

This section provides a detailed, step-by-step explanation of how raw text data flows through the emoSpark pipeline to produce trained emotion classification models.

### Table of Contents

1. [Overview](#pipeline-overview)
2. [Data Loading and Preparation](#data-loading-and-preparation)
3. [Feature Engineering](#feature-engineering)
4. [Model Training Strategy](#model-training-strategy)
5. [Evaluation and Metrics](#evaluation-and-metrics)
6. [Complete Data Flow Diagram](#complete-data-flow-diagram)

---

### Pipeline Overview

**Goal**: Train multi-label classifiers to predict 8 Plutchik emotions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation) from text.

**Strategy**: One-vs-rest binary classification – train 8 separate binary classifiers, one per emotion.

**Key Insight**: Each text can have multiple emotions simultaneously (multi-label), so we don't use traditional multi-class classification.

---

### Data Loading and Preparation

#### Step 1: Load GoEmotions Dataset

**Input**: 3 CSV files (`goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`)

**Each Row Contains**:

- `text`: The input text (e.g., Reddit comment)
- 27 GoEmotions labels: Binary columns (0 or 1) for fine-grained emotions
  - Examples: `amusement`, `anger`, `annoyance`, `approval`, `caring`, etc.
- Optional: `id`, `example_very_unclear` (quality flags)

**Processing**:

```python
# Load all 3 CSV shards
raw_df = spark.read.csv(["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"])

# Filter out unclear examples (if flagged by annotators)
raw_df = filter_unclear_examples(raw_df)

# Aggregate rater annotations (some texts have multiple annotators)
# For each unique text, take max() across all annotators per label
raw_df = aggregate_rater_annotations(raw_df)
```

**Result**: Single DataFrame with unique texts and consolidated emotion labels.

---

#### Step 2: Project to Plutchik Emotions

GoEmotions has 27 fine-grained labels. We map them to 8 coarser Plutchik emotions.

**Mapping** (defined in `constants.py`):

```python
PLUTCHIK_TO_GOEMOTIONS = {
    "joy": ["amusement", "excitement", "joy", "optimism", "pride", "relief"],
    "trust": ["admiration", "approval", "caring", "gratitude", "love"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "confusion"],
    "sadness": ["grief", "remorse", "sadness", "disappointment"],
    "disgust": ["disgust", "embarrassment"],
    "anger": ["anger", "annoyance", "disapproval"],
    "anticipation": ["desire", "curiosity"],
}
```

**Projection Logic**:
For each Plutchik emotion, take the **maximum** of its constituent GoEmotions labels.

```python
# Example: joy = max(amusement, excitement, joy, optimism, pride, relief)
df = df.withColumn("joy", F.greatest(F.col("amusement"), F.col("excitement"), ...))
```

**Why Maximum?** If a text expresses ANY of the fine-grained emotions, we consider the broader emotion present.

**Result**: DataFrame with 8 Plutchik emotion columns (each 0.0 or 1.0) plus the text.

---

#### Step 3: Stratified Train/Validation/Test Split

**Goal**: Split data 70% train / 15% validation / 15% test, while ensuring rare emotions are represented in all splits.

**Challenge**: Some emotions are rare (e.g., trust, surprise appear in <10% of examples). Random splitting could exclude them from validation/test sets.

**Solution**: Stratified splitting on "primary label"

```python
# 1. Compute primary label (first positive emotion, or "neutral")
def primary_label(row):
    for emotion in PLUTCHIK_EMOTIONS:
        if row[emotion] == 1.0:
            return emotion
    return "neutral"

df = df.withColumn("primary_label", udf(primary_label))

# 2. Use window function to split by primary_label distribution
window = Window.partitionBy("primary_label").orderBy(F.rand(seed=42))
df = df.withColumn("rank", F.percent_rank().over(window))

# 3. Split based on cumulative rank
train = df.filter(F.col("rank") <= 0.70)
val   = df.filter((F.col("rank") > 0.70) & (F.col("rank") <= 0.85))
test  = df.filter(F.col("rank") > 0.85)
```

**Result**: Three DataFrames (train, val, test) with balanced emotion distributions.

**Example Split Sizes** (for full dataset ~58K examples):

- Train: ~40,600 examples
- Validation: ~8,700 examples
- Test: ~8,700 examples

---

### Feature Engineering

#### Overview: From Text to Feature Vectors

Each text is transformed into a high-dimensional feature vector (~60,000 features) through a multi-stage Spark ML Pipeline.

**Pipeline Stages**:

1. Tokenization → 2. Stopword Removal → 3-5. TF-IDF (1-grams, 2-grams, 3-grams) → 6. Lexicon Features → 7. VAD Features → 8. Linguistic Features → 9. Vector Assembly

---

#### Stage 1-2: Tokenization and Stopword Removal

**Input**: Raw text string

```python
text = "I'm so happy today! This is amazing!"
```

**Stage 1 - Tokenization**:

```python
RegexTokenizer(pattern="\\w+", toLowercase=True)
```

**Output**: Array of lowercase word tokens

```python
tokens = ["i", "m", "so", "happy", "today", "this", "is", "amazing"]
```

**Stage 2 - Stopword Removal**:

```python
StopWordsRemover()  # Removes: ["i", "m", "so", "this", "is"]
```

**Output**: Filtered tokens

```python
filtered_tokens = ["happy", "today", "amazing"]
```

**Why Remove Stopwords?**

- Common words ("the", "is", "a") appear in all texts regardless of emotion
- Removing them reduces noise and dimensionality
- Keeps emotionally meaningful words

---

#### Stage 3-5: TF-IDF Features for N-grams

**Goal**: Capture word importance using TF-IDF (Term Frequency × Inverse Document Frequency)

**TF-IDF Intuition**:

- **TF**: How often a term appears in THIS document
- **IDF**: How rare the term is across ALL documents
- **TF-IDF = TF × IDF**: Rare terms that appear frequently get high scores

**Process for Each N-gram Order** (1, 2, 3):

**1-gram (Unigrams)**: Individual words

```python
vocabulary = ["happy", "sad", "angry", "amazing", ...] # 20,000 most frequent words
tf_vector = count_occurrences(filtered_tokens, vocabulary)
idf_weights = learn_from_training_data()  # Log(total_docs / docs_with_term)
tfidf_1gram = tf_vector * idf_weights
```

**2-gram (Bigrams)**: Consecutive word pairs

```python
NGram(n=2) → ["happy today", "today amazing"]
tfidf_2gram = CountVectorizer + IDF (same process, 20K vocab)
```

**3-gram (Trigrams)**: Three consecutive words

```python
NGram(n=3) → ["happy today amazing"]
tfidf_3gram = CountVectorizer + IDF (same process, 20K vocab)
```

**Why Multiple N-grams?**

- Unigrams: Capture individual emotion words ("happy", "angry")
- Bigrams: Capture phrases ("not happy", "very sad")
- Trigrams: Capture longer context ("can't wait to see")

**Feature Counts**:

- TF-IDF 1-gram: ~20,000 features
- TF-IDF 2-gram: ~20,000 features
- TF-IDF 3-gram: ~20,000 features
- **Total**: ~60,000 TF-IDF features

---

#### Stage 6: NRC Emotion Lexicon Features (33 features)

**Goal**: Count explicit emotion words using the NRC Emotion Lexicon

**NRC Lexicon**: Dictionary mapping ~14,000 words to 10 emotions

```python
{
    "happy": ["joy", "positive"],
    "angry": ["anger", "negative"],
    "fearful": ["fear", "negative"],
    ...
}
```

**Feature Extraction**:
For each of 10 NRC emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust, positive, negative):

1. **Raw Count**: How many tokens match this emotion

   ```python
   joy_count = count_tokens_in_lexicon(tokens, "joy")  # e.g., 2
   ```

2. **Ratio**: Count normalized by total tokens

   ```python
   joy_ratio = joy_count / len(tokens)  # e.g., 2/3 = 0.67
   ```

3. **Binary Flag**: 1 if any match, 0 otherwise

   ```python
   joy_flag = 1 if joy_count > 0 else 0  # e.g., 1
   ```

4. **Aggregates**:
   - Total lexicon matches
   - Total match ratio (lexicon coverage)
   - Dominant emotion index (which emotion has most matches)

**Total**: 10×3 + 3 = **33 lexicon features**

**Why Useful?**

- Captures explicit emotion vocabulary
- Complements TF-IDF (which is emotion-agnostic)
- Provides interpretable emotion signals

---

#### Stage 7: NRC VAD Features (10 features)

**Goal**: Capture affective dimensions using Valence-Arousal-Dominance scores

**NRC VAD Lexicon**: Maps ~20,000 words to 3 continuous scores (0-1 scale)

```python
{
    "happy": (0.95, 0.71, 0.68),  # (valence, arousal, dominance)
    "calm": (0.74, 0.29, 0.59),
    "terrified": (0.12, 0.85, 0.22),
    ...
}
```

**Dimensions**:

- **Valence**: Positive (1.0) vs Negative (0.0) emotion
- **Arousal**: Excited/Activated (1.0) vs Calm (0.0)
- **Dominance**: In-control/Dominant (1.0) vs Submissive (0.0)

**Feature Extraction**:
For each dimension (valence, arousal, dominance):

1. **Mean**: Average score across all tokens
2. **Std Dev**: Variability of scores
3. **Range**: Max - Min scores

Plus: 4. **Coverage**: Proportion of tokens found in VAD lexicon

**Total**: 3 dimensions × 3 stats + 1 coverage = **10 VAD features**

**Example**:

```python
tokens = ["happy", "excited", "joy"]
valence_scores = [0.95, 0.88, 0.96]
valence_mean = 0.93  # Very positive
valence_std = 0.04   # Low variability (consistently positive)
valence_range = 0.08
```

**Why Useful?**

- Captures emotion intensity, not just presence/absence
- Dimensional representation complements categorical (lexicon)
- Research-backed affective computing features

---

#### Stage 8: Linguistic Features (12 features)

**Goal**: Capture writing style and emotional expression patterns

**Features**:

1. **Length Features**:

   - Word count: Number of tokens
   - Character count: Total characters
   - Average word length: Characters per word

2. **Punctuation Features**:

   - Exclamation count: "!" (excitement, surprise)
   - Question count: "?" (confusion, curiosity)
   - Multi-punctuation count: "!!", "?!", "???" (strong emotion)
   - Punctuation density: Punctuation per word

3. **Capitalization Features**:

   - ALL CAPS token count: "AMAZING" (shouting, emphasis)
   - Title case ratio: Proportion of Title Case Words
   - Uppercase character ratio: UPPERCASE letters / total

4. **Other**:
   - Special character count: Non-alphanumeric symbols
   - Digit count: Numbers in text

**Total**: **12 linguistic features**

**Example**:

```python
text = "OMG this is AMAZING!! I'm so excited!!!"
linguistic_features = {
    "exclamation_count": 3,
    "all_caps_token_count": 2,  # "OMG", "AMAZING"
    "multi_punct_count": 2,      # "!!", "!!!"
    "punctuation_density": 3/7,  # High emotional intensity
    ...
}
```

**Why Useful?**

- Captures emotion expression style (e.g., ALL CAPS = strong emotion)
- Punctuation indicates emotional intensity
- Complements word-based features with structural patterns

---

#### Stage 9: Vector Assembly

**Goal**: Combine all feature types into a single feature vector

```python
VectorAssembler(
    inputCols=["tfidf_1gram", "tfidf_2gram", "tfidf_3gram",
               "lexicon_vector", "vad_vector", "linguistic_vector"],
    outputCol="features"
)
```

**Final Feature Vector Dimensionality**:

```
TF-IDF 1-grams:      ~20,000
TF-IDF 2-grams:      ~20,000
TF-IDF 3-grams:      ~20,000
Lexicon features:         33
VAD features:             10
Linguistic features:      12
─────────────────────────────
TOTAL:               ~60,055 features
```

**Result**: Each text is represented as a sparse vector of 60,055 features.

**Sparse Vector Example**:

```python
# Most values are 0 (only relevant terms have non-zero TF-IDF scores)
features = SparseVector(60055, {
    145: 2.34,   # "happy" TF-IDF score
    2678: 1.89,  # "amazing" TF-IDF score
    ...
    60045: 0.67, # joy_ratio (lexicon)
    60052: 2.0,  # exclamation_count
})
```

---

### Model Training Strategy

#### One-vs-Rest Multi-Label Classification

**Problem**: Each text can have multiple emotions simultaneously.

**Example**:

```
Text: "I'm so excited but also a bit nervous!"
Labels: joy=1, anticipation=1, fear=1, (all others=0)
```

**Solution**: Train 8 independent binary classifiers, one per emotion.

**Training Process**:

```python
for emotion in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
    # Train a binary classifier: does this text have THIS emotion?
    model = LogisticRegression(
        featuresCol="features",  # 60K feature vector
        labelCol=emotion,         # 0 or 1 for this emotion
    )
    model.fit(train_data)
    models[emotion] = model
```

**Key Points**:

- Each model is independent (doesn't know about other emotions)
- Models can all predict 1 (multiple emotions)
- Models can all predict 0 (neutral text)
- No mutual exclusivity constraint

---

#### Base Model Families

The pipeline fits the same one-vs-rest formulation with four different learning algorithms. Each family learns eight independent binary classifiers (one per emotion) using the shared feature vector.

- **Logistic Regression** – L-BFGS solver with configurable `regParam`, `elasticNetParam`, and `maxIter` values supplied by `RuntimeConfig`. Regularisation is fixed per run for reproducibility rather than re-tuned for every label.
- **Linear SVM** – LinearSVC with hinge loss, `maxIter=50`, and `regParam=0.1`. Produces decision margins instead of calibrated probabilities.
- **Naive Bayes** – Bernoulli variant that assumes feature independence; provides a fast lexical baseline.
- **Random Forest** – 100-tree ensemble with `maxDepth=12`, `subsamplingRate=0.8`, and automatic feature sub-sampling to capture non-linear relationships.

Each fitted model emits three columns per emotion when available:

- `raw_<model>_<emotion>` – raw decision values or margins.
- `prob_<model>_<emotion>` – class probability vector (when the algorithm supports it).
- `pred_<model>_<emotion>` – binary prediction prior to thresholding.

All base families write predictions for the train/validation/test splits so downstream components can reuse them without recomputation.

#### Majority Vote Ensemble

When two or more base families are trained, their predictions can be blended without fitting another model. For every emotion we collect the most informative signal available from each family (probability when exposed, raw margin otherwise, and binary prediction as a fallback), convert it to a double, and average the values. The resulting vote share (`vote_share_majority_vote_<emotion>`) represents the fraction of models agreeing that the emotion is present. A label is considered positive when at least half of the participating families vote for it. This simple consensus often improves the overall micro/macro F1 while keeping the implementation lightweight and transparent.

#### Prediction Phase

**Input**: Test example with engineered feature vector.

**Process**:

1. Apply the feature pipeline to obtain `features`.
2. Run every requested base family to append their `pred_*`, `prob_*`, and `raw_*` columns.
3. If multiple families are available, compute `vote_share_majority_vote_*` columns and corresponding ensemble predictions.
4. Execute the target model family (base or ensemble) to generate final per-emotion outputs.

The resulting row contains the meta-model probability (if available), raw margin, and binary decision for each Plutchik emotion. Threshold tuning (described below) converts these scores into the final multi-label prediction set, e.g. `[joy, fear, anticipation]`.

---

#### Probability Thresholds

**Why Not Use 0.5 for All Emotions?**

Different emotions have different base rates and class imbalances:

- Joy: Common (~25% of examples) → Higher threshold (0.55)
- Fear: Rare (~7% of examples) → Lower threshold (0.45)
- Neutral examples: Should not trigger any emotions

**Tuned Thresholds** (from `constants.py`):

```python
DEFAULT_PROBABILITY_THRESHOLDS = {
    "joy": 0.55,         # Require high confidence (avoid false positives)
    "trust": 0.50,       # Balanced
    "fear": 0.45,        # Lower threshold (rare, don't want to miss)
    "surprise": 0.50,    # Balanced
    "sadness": 0.50,     # Balanced
    "disgust": 0.45,     # Lower threshold (rare)
    "anger": 0.50,       # Balanced
    "anticipation": 0.50,# Balanced
}
```

**How Thresholds are Tuned**:

1. Compute validation set probabilities
2. For each threshold in [0.3, 0.35, 0.40, ..., 0.70]:
   - Apply threshold to validation predictions
   - Compute F1 score
3. Select threshold maximizing F1 per emotion

---

### Evaluation and Metrics

#### Multi-Label Evaluation Metrics

**Challenge**: Traditional accuracy is misleading for multi-label problems.

**Example**:

```
True labels:     [joy=1, fear=1, all others=0]
Predicted:       [joy=1, anger=1, all others=0]
Accuracy: 6/8 = 75%, but we missed fear and false-alarmed anger!
```

**Multi-Label Metrics Used**:

#### 1. Hamming Loss

**Definition**: Average per-label error rate

```python
hamming_loss = (incorrect_labels) / (total_examples × num_labels)
```

**Example**:

```
Example 1: True=[1,1,0,0,0,0,0,0], Pred=[1,0,0,0,0,0,0,0] → 1 error / 8 labels
Example 2: True=[0,0,1,0,0,0,0,0], Pred=[0,0,1,0,0,0,0,0] → 0 errors / 8 labels
Average: (1+0)/(2×8) = 0.0625 (6.25% average label error)
```

**Lower is better** (0 = perfect)

#### 2. Subset Accuracy (Exact Match)

**Definition**: Percentage of examples with ALL labels correct

```python
subset_acc = count(true_labels == pred_labels) / total_examples
```

**Strict metric**: Even one wrong label counts as wrong
**Example**: 0.35 = 35% of examples have perfect predictions

#### 3. Micro-F1

**Definition**: Global F1 across all labels (favors common emotions)

**Computation**:

```python
# Pool all (label, prediction) pairs across all emotions and examples
TP = count(true=1 AND pred=1)  # True positives across all
FP = count(true=0 AND pred=1)  # False positives across all
FN = count(true=1 AND pred=0)  # False negatives across all

micro_precision = TP / (TP + FP)
micro_recall = TP / (TP + FN)
micro_f1 = 2 × (micro_precision × micro_recall) / (micro_precision + micro_recall)
```

**Interpretation**: Overall system performance, weighted by emotion frequency

#### 4. Macro-F1

**Definition**: Average F1 per emotion (treats all emotions equally)

**Computation**:

```python
# Compute F1 for each emotion separately
for emotion in EMOTIONS:
    f1[emotion] = compute_f1_score(emotion_predictions)

# Average across emotions (unweighted)
macro_f1 = mean(f1.values())
```

**Interpretation**: Performance on rare emotions counts as much as common ones

**Example**:

```
joy (common):    F1=0.65
fear (rare):     F1=0.35
macro_f1 = (0.65 + 0.35 + ... ) / 8 = 0.48
```

#### 5. Per-Emotion Metrics

For each emotion: Precision, Recall, F1, Support

**Example for "joy"**:

```python
True Positives (TP):  200  # Correctly predicted joy
False Positives (FP):  50  # Predicted joy, but wasn't
False Negatives (FN):  30  # Missed joy that was there
Support:              230  # Total examples with joy=1

Precision = TP/(TP+FP) = 200/250 = 0.80 (80% of joy predictions are correct)
Recall = TP/(TP+FN) = 200/230 = 0.87 (87% of joy examples are caught)
F1 = 2 × (P×R)/(P+R) = 0.83
```

---

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA INPUT                              │
│  GoEmotions CSVs: text + 27 fine-grained emotion labels         │
│  Example: "I'm so happy!", amusement=1, joy=1, excitement=1     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: DATA CLEANING & PROJECTION                 │
│  - Filter unclear examples                                      │
│  - Aggregate rater annotations (max per label)                  │
│  - Project 27 labels → 8 Plutchik emotions (via mapping)        │
│  Output: text + [joy, trust, fear, surprise, sadness,           │
│                   disgust, anger, anticipation] (0 or 1 each)   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: STRATIFIED TRAIN/VAL/TEST SPLIT                 │
│  - Compute primary_label for stratification                     │
│  - Split: 70% train / 15% val / 15% test                        │
│  - Ensures rare emotions in all splits                          │
│  Output: train_df, val_df, test_df                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: FEATURE ENGINEERING PIPELINE                  │
│  (Fit on training data only!)                                   │
│                                                                 │
│  3.1: Tokenization                                              │
│       "I'm happy!" → ["i", "m", "happy"]                        │
│                                                                 │
│  3.2: Stopword Removal                                          │
│       ["i", "m", "happy"] → ["happy"]                           │
│                                                                 │
│  3.3-3.5: TF-IDF (1-gram, 2-gram, 3-gram)                       │
│       - Learn vocabulary (20K most frequent per n-gram)         │
│       - Learn IDF weights from training data                    │
│       - Transform text → sparse TF-IDF vectors (~60K features)  │
│                                                                 │
│  3.6: NRC Lexicon Features (33 features)                        │
│       - Count emotion words per NRC category                    │
│       - Ratios, flags, dominant emotion                         │
│                                                                 │
│  3.7: NRC VAD Features (10 features)                            │
│       - Valence/Arousal/Dominance statistics                    │
│       - Mean, std, range per dimension                          │
│                                                                 │
│  3.8: Linguistic Features (12 features)                         │
│       - Punctuation counts, capitalization, lengths             │
│                                                                 │
│  3.9: Vector Assembly                                           │
│       - Combine all feature types into single vector            │
│                                                                 │
│  Output: train_features, val_features, test_features            │
│          Each row: (text, features, 8 emotion labels)           │
│          features = sparse vector of ~60,055 dimensions         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│      STEP 4: MODEL TRAINING (ONE-VS-REST)                        │
│  For each requested model family:                                │
│                                                                  │
│  4.1: Train eight binary classifiers (one per Plutchik emotion)  │
│       sharing the engineered feature vector.                     │
│       - Logistic Regression: regularised L-BFGS fit              │
│       - Linear SVM: hinge-loss margins                           │
│       - Naive Bayes: Bernoulli distributions                     │
│       - Random Forest: 100-tree ensemble                         │
│                                                                  │
│  4.2: Persist artefacts                                          │
│       - Save per-emotion Spark MLlib models                      │
│       - Emit predictions for train/validation/test splits        │
│                                                                  │
│  4.3: Optional majority-vote ensemble                            │
│       - If ≥2 base families exist, reuse their stored predictions│
│         for each split                                           │
│       - Combine per-emotion votes; positive if ≥50% agree        │
│       - Persist ensemble predictions for every data split        │
│                                                                  │
│  Output: Model directories for each family plus an optional      │
│          majority-vote ensemble that blends their signals        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│           STEP 5: PREDICTION & EVALUATION                        │
│                                                                  │
│  5.1: Generate Predictions on Val/Test Sets                      │
│       For each model family (base + ensemble):                   │
│         - Apply all 8 per-emotion classifiers                    │
│         - Capture probabilities or margins                       │
│         - Apply tuned thresholds: pred = 1 if score ≥ threshold  │
│                                                                  │
│  5.2: Compute Multi-Label Metrics                                │
│       - Hamming Loss: avg per-label error                        │
│       - Subset Accuracy: exact match rate                        │
│       - Micro-F1: global F1 (favors common emotions)             │
│       - Macro-F1: average F1 per emotion (equal weight)          │
│       - Per-emotion: precision, recall, F1, support              │
│                                                                  │
│  5.3: Save Results                                               │
│       - Models: models/<model_type>/<emotion>/                   │
│       - Predictions: predictions/<model_type>/<split>/           │
│       - Metrics: evaluation/metrics_<model_type>.json            │
│                                                                  │
│  Output: Comprehensive evaluation JSON files                     │
└──────────────────────────────────────────────────────────────────┘
```

---

### Key Takeaways

1. **Multi-Label Nature**: Each text can have 0, 1, or multiple emotions simultaneously

2. **One-vs-Rest Strategy**: Train 8 independent binary classifiers per algorithm, with an optional majority-vote ensemble to capture consensus gains

3. **Rich Feature Engineering**: Combine multiple feature types for robust representations:

   - Sparse TF-IDF (word importance)
   - Dense lexicon features (explicit emotion words)
   - VAD features (affective dimensions)
   - Linguistic features (writing style)

4. **Rigorous Evaluation**: Use multi-label metrics (Hamming loss, subset accuracy, micro/macro F1) not simple accuracy

5. **Threshold Tuning**: Emotion-specific thresholds account for class imbalance and base rates

6. **Scalability**: Spark-based pipeline handles large datasets and distributed training

7. **End-to-End Pipeline**: From raw CSV to trained models with comprehensive evaluation

---

### References

- **GoEmotions Dataset**: Demszky et al. (2020) - Fine-grained emotion classification
- **Plutchik's Wheel**: Plutchik (1980) - 8 primary emotions framework
- **NRC Emotion Lexicon**: Mohammad & Turney (2013) - Word-emotion associations
- **NRC VAD Lexicon**: Mohammad (2018) - Valence-arousal-dominance norms

---

## Data Assets

- **GoEmotions**: 58k+ English Reddit comments annotated with 27 fine-grained emotion labels plus neutrality. The raw CSVs are split into three shards (`data/goemotions_1.csv` through `_3.csv`).
- **Plutchik projection**: Raw labels are projected onto Plutchik’s eight primary emotions (`joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`) using the mappings in `src/emo_spark/constants.py` (`project_to_plutchik`).
- **NRC Emotion Lexicon**: Word-level associations between tokens and 10 discrete emotions. Used to derive count ratios, binary flags, and coverage diagnostics in `LexiconFeatureTransformer`.
- **NRC VAD Lexicon**: Continuous valence–arousal–dominance scores per token. Summaries feed `VADFeatureTransformer`.
- **Stratified splits**: During ingestion we stratify train/validation/test on the dominant Plutchik label to keep rare classes (e.g., `trust`, `surprise`) represented.

## Environment Setup

**Requirements**: Python 3.13 or higher

1. **Create a virtual environment and install dependencies**

   ```bash
   # Option 1: Using uv (recommended for faster dependency resolution)
   # Install uv first if not already installed:
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows:
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   uv sync
   source .venv/bin/activate  # On macOS/Linux
   # Or: .venv\Scripts\activate  # On Windows

   # Option 2: Using standard Python venv and pip
   python3.13 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # Or: .venv\Scripts\activate  # On Windows
   pip install -e .
   ```

## Data Preparation

1. **Ensure the GoEmotions CSVs exist** or run the helper script:

   ```bash
   chmod +x prepare_data.sh  # Make script executable (first time only)
   ./prepare_data.sh
   ```

   The script will check for the three GoEmotions CSV shards in the `data/` directory.

2. **NRC Lexicons**: The script also verifies the presence of the NRC lexica. If they are missing, you'll need to download them manually from the official sources and place them at:

   - `data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` - Download from: [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
   - `data/NRC-VAD-Lexicon-v2.1.txt` - Download from: [NRC VAD Lexicon](http://saifmohammad.com/WebPages/nrc-vad.html)

3. **(Optional) For AWS EMR**: Upload the entire `data/` directory to an S3 prefix for distributed runs, keeping the filenames identical. For example:
   ```bash
   aws s3 sync data/ s3://your-bucket/emoSpark/data/
   ```

**Note**: The loader in `src/emo_spark/data.py` automatically detects whether paths are local or S3-based on the `EMO_SPARK_INPUT_PATH` environment variable.

## Configuration

Environment variables control most runtime behaviour. Defaults are tuned for balanced throughput and accuracy; override only what you need.

| Variable                    | Purpose                                                                           | Default        |
| --------------------------- | --------------------------------------------------------------------------------- | -------------- |
| `EMO_SPARK_ENV`             | Execution context. Use `local` or `emr` to switch logging/storage conventions.    | `local`        |
| `EMO_SPARK_INPUT_PATH`      | Folder (local) or S3 prefix containing GoEmotions CSVs and NRC lexica.            | `data`         |
| `EMO_SPARK_OUTPUT_PATH`     | Output folder or S3 prefix for features, models, metrics, and demos.              | `output`       |
| `EMO_SPARK_SAMPLE_FRACTION` | Optional float (0–1) to subsample before splitting. Handy for laptop experiments. | unset          |
| `EMO_SPARK_SEED`            | Seed used for sampling, splitting, and Spark randomness.                          | `42`           |
| `EMO_SPARK_CACHE`           | `1` to persist intermediate DataFrames, `0` to disable caching.                   | `1`            |
| `EMO_SPARK_THRESHOLDS`      | JSON string or `label=value` pairs overriding Plutchik probability thresholds.    | tuned defaults |
| `EMO_SPARK_REPARTITION`     | Target partition count for feature tables; improves shuffle balance.              | unset          |

Refer to `src/emo_spark/config.py` for the exhaustive list, including advanced knobs for n-gram ranges, vocab sizes, regularization grids, and I/O shuffles.

## Running Locally

1. **Activate the environment** and export any configuration overrides:

   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # Or: .venv\Scripts\activate  # On Windows

   # Optional: Sample a smaller fraction for faster iteration during development
   export EMO_SPARK_SAMPLE_FRACTION=0.2
   export EMO_SPARK_OUTPUT_PATH=output
   ```

2. **Launch training**. Select one or more base models (comma-separated). When two or more families are trained, the pipeline automatically derives a majority-vote ensemble across their predictions:

   ```bash
   # Train all models (recommended for best performance)
   python -m emo_spark.main \
     --models logistic_regression,linear_svm,naive_bayes,random_forest \
     --verbose

   # Or train just one model for quick testing
   python -m emo_spark.main \
     --models logistic_regression \
     --verbose
   ```

3. **Outputs** are written to the configured output path:

   - `output/features/{train,validation,test}/` – Engineered features in Parquet format, ready for reuse.
   - `output/models/feature_pipeline/` – Fitted `PipelineModel` capturing tokenization, TF-IDF, and lexicon transforms.
   - `output/models/<model_type>/<emotion>/` – Per-label Spark MLlib models for each algorithm.
   - `output/predictions/<model_type>/<split>/` – Scored predictions with probability columns when available.
   - `output/predictions/majority_vote/<split>/` – Ensemble predictions produced when multiple families are trained.
   - `output/evaluation/metrics_<model_type>.json` – Micro/macro F1, Hamming loss, subset accuracy, and per-emotion metrics (includes `metrics_majority_vote.json` when the ensemble is produced).
   - `output/evaluation/thresholds/thresholds_<model_type>.json` – Auto-tuned probability thresholds per emotion.
   - `output/holdout/test_set/` – Untouched holdout split for future comparisons and demos.
   - `output/demo/demo_samples.json` – Sample texts from test set for interactive demonstration.

4. **Speed up exploratory runs**: Reduce the model set (`--models logistic_regression`) and use sampling (`EMO_SPARK_SAMPLE_FRACTION=0.1`). Restore full data before producing final results.

## Running on AWS EMR

1. **Provision infrastructure**

   - Create an S3 bucket (or reuse one) and upload the entire `data/` directory plus any configuration files:
     ```bash
     aws s3 sync data/ s3://your-bucket/emoSpark/data/
     ```
   - Launch an EMR cluster with Spark 3.5+ and Python 3.13+. Use at least 2-4 worker nodes (m5.xlarge or larger) for reasonable performance on the full dataset.

2. **Bootstrap dependencies**

   - Either `git clone` this repository on the master node or stage a tarball to S3 and extract.
   - On the master node, run `pip install -e .` inside a Python 3.10+ environment (EMR’s default) or create a virtual environment mirroring the local setup.
   - Set `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to the interpreter that has the project installed if you deviate from the system Python.

3. **Configure environment variables** (per session or via `spark-submit --conf spark.yarn.appMasterEnv...`):

   ```bash
   export EMO_SPARK_ENV=emr
   export EMO_SPARK_INPUT_PATH=s3://<bucket>/data
   export EMO_SPARK_OUTPUT_PATH=s3://<bucket>/emoSpark-output
   ```

4. **Submit the job**

   - **Option A (spark-submit)** – recommended for YARN-managed runs:

     ```bash
     spark-submit \
       --master yarn \
       --deploy-mode cluster \
       --conf spark.executor.instances=8 \
       --conf spark.executor.memory=6g \
       --conf spark.executor.cores=2 \
       src/emo_spark/main.py \
       --models logistic_regression --verbose
     ```

     Supply additional `--conf` flags for shuffle tuning (`spark.sql.shuffle.partitions`, etc.) as needed.

   - **Option B (python -m)** – quick interactive runs from the master node shell:

     ```bash
     python -m emo_spark.main --models logistic_regression --verbose
     ```

5. **Result collection**
   - Metrics, predictions, and persisted models land under the configured S3 output prefix.
   - EMR step logs (stdout/stderr) provide progress and per-stage summaries.

## Inspecting Results

- **Metrics JSON**: Each trained model writes `output/evaluation/metrics_<model>.json`. Inspect with `jq`:

  ```bash
  jq '.[] | {dataset, micro_f1: .micro_f1, macro_f1: .macro_f1, hamming_loss: .hamming_loss}' \
    output/evaluation/metrics_logistic_regression.json
  ```

- **Per-emotion breakdown**: The JSON entries contain nested `per_label` stats. Example of extracting the `joy` F1 score:

  ```bash
  jq '.[] | select(.dataset == "validation") | .per_label.joy.f1' \
    output/evaluation/metrics_logistic_regression.json
  ```

- **Predictions**: View Parquet predictions in Spark SQL, pandas, or DuckDB:

  ```bash
  python - <<'PY'
  import pandas as pd
  df = pd.read_parquet('output/predictions/logistic_regression/validation')
  print(df.head())
  PY
  ```

- **Reference performance**: On the full GoEmotions training set with tuned thresholds, logistic regression typically achieves micro-F1 in the high 0.4s and macro-F1 around the low 0.4s. Exact numbers vary with sampling, CV folds, and threshold overrides; use the metrics JSON as the source of truth for your run.

## Interactive Demo

After training, launch the storytelling demo to inspect predictions interactively. The majority-vote ensemble provides the strongest scores when multiple base models are available, but you can still target individual model families:

```bash
python -m emo_spark.demo \
  --model majority_vote \
  --use-demo-samples \
  --thresholds-json '{"joy":0.6,"fear":0.4}'
```

Add custom examples using repeated `--text "Some input"` arguments. The demo reports threshold-aware predictions, probability scores where available, and narrates the leading emotional storyline.

If your training run produced only a single family, specify it explicitly (for example `--model logistic_regression`) so the demo can load the correct artifacts.

## Repository Layout

```
data/                     # GoEmotions CSVs and NRC emotion/VAD lexica
docs/                     # Architecture notes, diagrams, and research context
output/                   # Generated artifacts (features, models, metrics)
src/emo_spark/            # PySpark source code
  config.py               # Runtime configuration dataclass and defaults
  data.py                 # Data loading, projection, and stratified splitting
  features.py             # Feature engineering pipeline components
  models.py               # Model orchestration and cross-validation logic
  metrics.py              # Multi-label metric computations and utilities
  evaluation.py           # Evaluation manager for metrics + persistence
  pipeline.py             # End-to-end pipeline wiring helpers
  main.py                 # CLI entrypoint for training
  demo.py                 # Interactive demo CLI
```
