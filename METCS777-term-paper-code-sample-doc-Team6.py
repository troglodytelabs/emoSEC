from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf, col, array
from pyspark.sql.types import ArrayType, DoubleType
from datasets import load_dataset
import re

# create spark session for distributed computing across multiple cores
# spark enables parallel processing of large datasets
spark = SparkSession.builder \
    .appName("GoEmotions-NRC-TFIDF") \
    .getOrCreate()

print(f"spark version: {spark.version}")

# load the goemotions dataset from huggingface
# goemotions contains reddit comments labeled with 28 emotion categories
print("loading goemotions dataset...")
dataset = load_dataset("go_emotions", "raw")
all_data = dataset["train"]

# use 50% of the full dataset (105,612 records) to balance performance and memory
# using more data improves model quality, especially for rare emotions
sample_size = int(len(all_data) * 0.5)
sampled_data = all_data.select(range(sample_size))

print(f"sampled {len(sampled_data)} records (50% of full dataset)")

# split the sampled data into 80% training and 20% testing
# this ensures we have separate data to evaluate model performance
split_idx = int(len(sampled_data) * 0.8)
train_data = sampled_data.select(range(split_idx))
test_data = sampled_data.select(range(split_idx, len(sampled_data)))

print(f"split into {len(train_data)} training and {len(test_data)} test records")

# define all 28 emotion labels from the goemotions dataset
# these are the original fine-grained emotion categories
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# map the 28 goemotions labels to plutchik's 8 basic emotions
# this reduces complexity and aligns with established emotion theory
# multiple goemotions can map to the same plutchik emotion (many-to-one mapping)
GOEMOTIONS_TO_PLUTCHIK = {
    # positive emotions that indicate joy or happiness
    "joy": "joy", "amusement": "joy", "excitement": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "relief": "joy", "gratitude": "joy",
    # negative emotions indicating sadness or loss
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
    # negative emotions involving hostility or frustration
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    # emotions related to anxiety or threat
    "fear": "fear", "nervousness": "fear",
    # emotions involving unexpectedness or confusion
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    # emotions involving aversion or repulsion
    "disgust": "disgust", "embarrassment": "disgust",
    # emotions involving acceptance or approval
    "approval": "trust", "admiration": "trust", "caring": "trust",
    # emotions involving expectation or interest
    "curiosity": "anticipation", "desire": "anticipation",
    # neutral is not an emotion in plutchik's model
    "neutral": None
}

# these are our 8 target emotion labels for classification
PLUTCHIK_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]

# load the nrc emotion lexicon which maps words to emotions
# this provides domain knowledge about which words are associated with which emotions
print("loading nrc emotion lexicon...")
nrc_lexicon = {}
emotion_lexicon_path = "/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# read the tab-delimited lexicon file
with open(emotion_lexicon_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, association = parts
            # only keep words with association=1 (word has this emotion)
            # exclude positive/negative as they're sentiment not specific emotions
            if int(association) == 1 and emotion not in ['positive', 'negative']:
                if word not in nrc_lexicon:
                    nrc_lexicon[word] = set()
                # add this emotion to the set of emotions for this word
                nrc_lexicon[word].add(emotion)

# convert sets to lists for serialization to spark workers
nrc_lexicon = {word: list(emotions) for word, emotions in nrc_lexicon.items()}
print(f"loaded {len(nrc_lexicon)} words from nrc emotion lexicon")

# load the nrc vad (valence-arousal-dominance) lexicon
# valence: how positive/negative (0=negative, 1=positive)
# arousal: how calm/excited (0=calm, 1=excited)
# dominance: how controlled/in-control (0=controlled, 1=in-control)
print("loading nrc vad lexicon...")
nrc_vad = {}
vad_path = "/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-VAD-Lexicon.txt"

with open(vad_path, 'r') as f:
    next(f)  # skip the header line
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            word, valence, arousal, dominance = parts
            try:
                # store all three vad dimensions for each word
                nrc_vad[word] = {
                    'valence': float(valence),
                    'arousal': float(arousal),
                    'dominance': float(dominance)
                }
            except ValueError:
                # skip any lines with malformed numeric values
                continue

print(f"loaded {len(nrc_vad)} words from nrc vad lexicon")

# broadcast the lexicons to all spark worker nodes
# broadcasting sends the data once to each worker instead of with every task
# this is much more efficient for read-only data that all tasks need
nrc_lexicon_bc = spark.sparkContext.broadcast(nrc_lexicon)
nrc_vad_bc = spark.sparkContext.broadcast(nrc_vad)

# define function to extract nrc emotion features from text
# counts how many words in the text are associated with each plutchik emotion
def extract_nrc_features(text):
    """count occurrences of emotion-bearing words for each plutchik emotion"""
    # tokenize text into lowercase alphabetic words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # initialize counter for each of the 8 emotions
    emotion_counts = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
    # count words associated with each emotion
    for word in words:
        if word in nrc_lexicon_bc.value:
            for emotion in nrc_lexicon_bc.value[word]:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1.0
    # return as array in consistent order (important for machine learning)
    return [emotion_counts[e] for e in PLUTCHIK_EMOTIONS]

# define function to extract vad features from text
# computes average valence, arousal, and dominance across all words
def extract_vad_features(text):
    """compute average vad (valence-arousal-dominance) scores across all words in text"""
    # tokenize text into lowercase words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    valence_sum = arousal_sum = dominance_sum = count = 0
    # accumulate vad scores for all words that have them
    for word in words:
        if word in nrc_vad_bc.value:
            vad = nrc_vad_bc.value[word]
            valence_sum += vad['valence']
            arousal_sum += vad['arousal']
            dominance_sum += vad['dominance']
            count += 1
    # return averages if any words were found, otherwise return neutral (0.5)
    if count > 0:
        return [valence_sum / count, arousal_sum / count, dominance_sum / count]
    return [0.5, 0.5, 0.5]

# define function to convert goemotions labels to plutchik labels
# maps the fine-grained goemotions to coarser plutchik emotions
def get_plutchik_labels(row):
    """convert multi-label goemotions annotations to plutchik emotion labels"""
    plutchik_set = set()
    # check each goemotions label in the row
    for emotion in GOEMOTIONS:
        if row.get(emotion, 0) == 1:
            # map to corresponding plutchik emotion
            plutchik = GOEMOTIONS_TO_PLUTCHIK.get(emotion)
            if plutchik:
                plutchik_set.add(plutchik)
    # return binary array (1 if emotion present, 0 if not) in consistent order
    return [1.0 if e in plutchik_set else 0.0 for e in PLUTCHIK_EMOTIONS]

# register python functions as spark user-defined functions (udfs)
# this allows us to apply these functions to spark dataframe columns
extract_nrc_udf = udf(extract_nrc_features, ArrayType(DoubleType()))
extract_vad_udf = udf(extract_vad_features, ArrayType(DoubleType()))
get_labels_udf = udf(get_plutchik_labels, ArrayType(DoubleType()))

# convert huggingface dataset to spark dataframe
# each row contains text and a dictionary of all goemotions labels
print("\npreparing training data...")
train_df = spark.createDataFrame([
    (row['text'], {e: row.get(e, 0) for e in GOEMOTIONS})
    for row in train_data
], ["text", "goemotions"])

# extract lexicon-based features using our custom functions
# add nrc emotion counts as a new column (8 dimensions)
train_df = train_df.withColumn("nrc_features", extract_nrc_udf(col("text")))
# add vad scores as a new column (3 dimensions)
train_df = train_df.withColumn("vad_features", extract_vad_udf(col("text")))
# add plutchik labels as a new column (8 binary values)
train_df = train_df.withColumn("labels", get_labels_udf(col("goemotions")))

# tokenize text into individual words for tf-idf processing
tokenizer = Tokenizer(inputCol="text", outputCol="words")
train_df = tokenizer.transform(train_df)

# compute term frequency using hashing trick
# hashing maps words to a fixed-size feature space (500 dimensions)
# this is more memory efficient than tracking all unique vocabulary
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=500)
train_df = hashingTF.transform(train_df)

# compute inverse document frequency to weight terms
# idf downweights common words (like "the", "is") and emphasizes distinctive words
# this helps the model focus on words that are actually informative
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(train_df)  # fit idf on training data
train_df = idf_model.transform(train_df)  # apply idf transformation

# cache the training dataframe in memory since we'll iterate over it 8 times
# (once for each emotion classifier)
train_df.cache()
print(f"prepared {train_df.count()} training examples")

# prepare test data using the same transformations as training
# important: use the same tokenizer and idf_model fitted on training data
print("\npreparing test data...")
test_df = spark.createDataFrame([
    (row['text'], {e: row.get(e, 0) for e in GOEMOTIONS})
    for row in test_data
], ["text", "goemotions"])

# apply same feature extraction functions
test_df = test_df.withColumn("nrc_features", extract_nrc_udf(col("text")))
test_df = test_df.withColumn("vad_features", extract_vad_udf(col("text")))
test_df = test_df.withColumn("labels", get_labels_udf(col("goemotions")))

# apply same tokenization and tf-idf transformations
test_df = tokenizer.transform(test_df)
test_df = hashingTF.transform(test_df)
test_df = idf_model.transform(test_df)  # use idf model fitted on training data

# cache test data in memory for faster evaluation
test_df.cache()
print(f"prepared {test_df.count()} test examples")

# import libraries for feature combination
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.functions import vector_to_array

# define function to combine all features into a single feature vector
# concatenates tf-idf (500d) + nrc counts (8d) + vad scores (3d) = 511 total features
def combine_features(tfidf, nrc, vad):
    """merge tf-idf, nrc emotion counts, and vad scores into single feature vector"""
    # convert sparse tf-idf vector to dense array
    tfidf_dense = tfidf.toArray().tolist() if hasattr(tfidf, 'toArray') else list(tfidf)
    # concatenate all feature types
    return Vectors.dense(tfidf_dense + nrc + vad)

# register the combination function as a udf
combine_features_udf = udf(combine_features, VectorUDT())

print("\ncombining features...")
# add combined feature column to training data
train_df = train_df.withColumn("features",
    combine_features_udf(col("tfidf_features"), col("nrc_features"), col("vad_features")))

# add combined feature column to test data
test_df = test_df.withColumn("features",
    combine_features_udf(col("tfidf_features"), col("nrc_features"), col("vad_features")))

# train separate binary logistic regression classifiers for each emotion
# this is a one-vs-rest approach for multi-label classification
# each classifier learns to predict whether its specific emotion is present
print("\ntraining logistic regression models for each emotion...")
models = {}  # store trained models
predictions_dfs = []  # store prediction dataframes for each emotion

for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    print(f"  training model for {emotion}...")

    # create a function to extract binary label for this specific emotion
    # extracts the idx-th element from the labels array (1.0 or 0.0)
    get_label_udf = udf(lambda labels: float(labels[idx]), DoubleType())

    # create training and test sets with single binary label column for this emotion
    emotion_train = train_df.withColumn("label", get_label_udf(col("labels")))
    emotion_test = test_df.withColumn("label", get_label_udf(col("labels")))

    # initialize and train logistic regression classifier
    # maxIter=10: run optimization for 10 iterations
    # regParam=0.01: regularization strength to prevent overfitting
    # elasticNetParam=0.0: use L2 regularization (ridge regression)
    lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.0)
    model = lr.fit(emotion_train)
    models[emotion] = model

    # make predictions on test set
    predictions = model.transform(emotion_test)
    # extract binary prediction (0 or 1)
    predictions = predictions.withColumn(f"pred_{emotion}", col("prediction"))
    # extract probability of positive class (emotion is present)
    # probability vector is [prob_negative, prob_positive], we want the second element
    predictions = predictions.withColumn(f"prob_{emotion}",
        vector_to_array(col("probability"))[1])

    # store predictions for this emotion (just text, prediction, and probability)
    predictions_dfs.append(predictions.select("text", f"pred_{emotion}", f"prob_{emotion}"))

# combine predictions from all 8 emotion models into a single dataframe
print("\ncombining predictions...")
# start with just text and true labels
combined = test_df.select("text", "labels")

# join predictions from each emotion model
# this adds pred_joy, prob_joy, pred_sadness, prob_sadness, etc. columns
for pred_df in predictions_dfs:
    combined = combined.join(pred_df, on="text", how="left")

# define custom decision threshold for converting probabilities to predictions
# default is 0.5 but we use 0.25 for better recall (catch more true positives)
DECISION_THRESHOLD = 0.25

# define function to apply custom threshold to probability array
def apply_threshold(probs, threshold=DECISION_THRESHOLD):
    """convert probabilities to binary predictions using custom threshold"""
    return [1.0 if p >= threshold else 0.0 for p in probs]

# register as udf
apply_threshold_udf = udf(apply_threshold, ArrayType(DoubleType()))

# collect all probabilities into a single array column
prob_cols = [col(f"prob_{e}") for e in PLUTCHIK_EMOTIONS]
combined = combined.withColumn("probabilities", array(*prob_cols))

# apply custom threshold to get final predictions
# this replaces spark's default 0.5 threshold with our custom 0.25
combined = combined.withColumn("predictions", apply_threshold_udf(col("probabilities")))

# evaluate model performance by computing precision, recall, and f1 scores
print(f"\nevaluating model performance (threshold={DECISION_THRESHOLD})...")

# compute per-emotion metrics
print("\nper-emotion performance:")
for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    # extract true labels and predictions for this emotion
    emotion_results = combined.select(
        (col("labels")[idx]).alias("label"),
        (col("predictions")[idx]).alias("prediction")
    )

    # compute confusion matrix elements
    # true positive: correctly predicted emotion is present
    tp = emotion_results.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
    # false positive: incorrectly predicted emotion is present
    fp = emotion_results.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
    # false negative: failed to predict emotion that is actually present
    fn = emotion_results.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()

    # compute standard classification metrics
    # precision: what fraction of predictions were correct
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall: what fraction of true labels were found
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1: harmonic mean of precision and recall
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  {emotion:12s}: p={prec:.3f} r={rec:.3f} f1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

# compute overall multi-label metrics using micro-averaging
# micro-averaging treats each label instance equally across all emotions
def calculate_matches(labels, predictions):
    """count matching predictions across all emotions for a single example"""
    # count how many emotions were correctly predicted
    matches = sum(1 for i in range(len(labels)) if labels[i] == 1.0 and predictions[i] == 1.0)
    # count total true emotions
    true_count = sum(1 for x in labels if x == 1.0)
    # count total predicted emotions
    pred_count = sum(1 for x in predictions if x == 1.0)
    return float(matches), float(true_count), float(pred_count)

# register as udf
calculate_matches_udf = udf(calculate_matches, ArrayType(DoubleType()))

# apply matching calculation to each example
metrics_df = combined.withColumn("metrics",
    calculate_matches_udf(col("labels"), col("predictions")))

# aggregate metrics across all examples
# sum up matches, true labels, and predictions across entire test set
total_matched = metrics_df.select(col("metrics")[0]).rdd.map(lambda x: x[0]).sum()
total_true = metrics_df.select(col("metrics")[1]).rdd.map(lambda x: x[0]).sum()
total_predicted = metrics_df.select(col("metrics")[2]).rdd.map(lambda x: x[0]).sum()

# compute overall metrics
precision = total_matched / total_predicted if total_predicted > 0 else 0
recall = total_matched / total_true if total_true > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\noverall multi-label metrics (micro-averaged):")
print(f"  precision: {precision:.3f} ({int(total_matched)}/{int(total_predicted)})")
print(f"  recall: {recall:.3f} ({int(total_matched)}/{int(total_true)})")
print(f"  f1-score: {f1:.3f}")

# show sample predictions to illustrate model behavior
print("\nsample predictions:")
# collect all results to driver (this brings data from distributed workers)
all_samples = combined.select("text", "labels", "predictions", "probabilities").collect()

# select diverse samples by taking every nth example
# this gives us variety rather than just the first 10 examples
sample_indices = range(0, len(all_samples), len(all_samples) // 10)

for i, sample_idx in enumerate(sample_indices[:10], 1):
    sample = all_samples[sample_idx]
    text = sample['text']
    # extract emotion names for true labels (where label array has 1.0)
    true_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['labels']) if val == 1.0]
    # extract emotion names for predictions (where prediction array has 1.0)
    pred_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['predictions']) if val == 1.0]
    # show all probabilities above 0.2 to see near-misses
    probs = {PLUTCHIK_EMOTIONS[idx]: sample['probabilities'][idx]
             for idx in range(len(PLUTCHIK_EMOTIONS)) if sample['probabilities'][idx] > 0.2}

    # display sample with truncated text
    print(f"\n{i}. {text[:80]}...")
    print(f"   true: {true_labels}")
    print(f"   pred: {pred_labels}")
    print(f"   probs: {dict(sorted(probs.items(), key=lambda x: -x[1]))}")

# stop spark session and release all resources
spark.stop()
