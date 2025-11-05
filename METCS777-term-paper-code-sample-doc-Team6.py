"""
scalable emotion classification for affective computing using apache spark
multi-label text classification with hybrid lexical-dimensional features

author: group 6 - devin dyson, madhur deep jain
date: november 5, 2025
"""

import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import udf, col, array
from pyspark.sql.types import ArrayType, DoubleType
from datasets import load_dataset


# configuration section
# these are all the settings you can adjust to customize the model

# plutchik's 8 basic emotions - our target labels for classification
PLUTCHIK_EMOTIONS = [
    'joy', 'sadness', 'anger', 'fear',
    'surprise', 'disgust', 'trust', 'anticipation'
]

# goemotions has 28 fine-grained emotion labels from reddit comments
# we'll map these to the 8 plutchik emotions for simpler classification
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# mapping dictionary: converts goemotions labels to plutchik emotions
# multiple goemotions can map to the same plutchik emotion (many-to-one)
# neutral maps to none since it's not an emotion in plutchik's model
EMOTION_MAPPING = {
    'joy': 'joy', 'amusement': 'joy', 'excitement': 'joy', 'love': 'joy',
    'optimism': 'joy', 'pride': 'joy', 'relief': 'joy', 'gratitude': 'joy',
    'sadness': 'sadness', 'grief': 'sadness', 'disappointment': 'sadness', 'remorse': 'sadness',
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    'fear': 'fear', 'nervousness': 'fear',
    'surprise': 'surprise', 'realization': 'surprise', 'confusion': 'surprise',
    'disgust': 'disgust', 'embarrassment': 'disgust',
    'approval': 'trust', 'admiration': 'trust', 'caring': 'trust',
    'curiosity': 'anticipation', 'desire': 'anticipation',
    'neutral': None
}

# paths to nrc lexicon files on your local machine
# update these paths to match your file locations
NRC_EMOTION_PATH = '/Users/devindyson/Desktop/paper/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
NRC_VAD_PATH = '/Users/devindyson/Desktop/paper/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Intensity-Lexicon-v1/NRC-Emotion-Intensity-Lexicon-v1.txt'

# model hyperparameters - adjust these to tune performance
SAMPLE_SIZE = 0.5  # fraction of dataset to use (0.5 = 50%, reduces memory usage)
TRAIN_SPLIT = 0.8  # fraction for training (0.8 = 80% train, 20% test)
TFIDF_FEATURES = 500  # number of tf-idf hash features (higher = more vocab coverage)
MAX_ITERATIONS = 10  # number of optimization iterations for logistic regression
REGULARIZATION = 0.01  # l2 regularization strength (prevents overfitting)
DECISION_THRESHOLD = 0.25  # probability threshold for positive prediction (lower = more recall)


# start of main script
print('emotion classification pipeline')

# initialize spark session for distributed computing
# this creates the spark context that manages parallel processing
spark = SparkSession.builder \
    .appName('emotion-classification') \
    .getOrCreate()

# reduce logging noise - only show warnings and errors
spark.sparkContext.setLogLevel('WARN')
print(f'spark version: {spark.version}')


# load nrc emotion lexicon from file
# this lexicon maps words to discrete emotion categories
# format: word \t emotion \t association (1 or 0)
print(f'loading nrc emotion lexicon from {NRC_EMOTION_PATH}...')
nrc_emotion_lex = {}

with open(NRC_EMOTION_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        # split tab-delimited line into components
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, association = parts
            # only keep words with positive association (association=1)
            # exclude 'positive' and 'negative' since they're sentiment not emotion
            if int(association) == 1 and emotion not in ['positive', 'negative']:
                # create entry for word if it doesn't exist yet
                if word not in nrc_emotion_lex:
                    nrc_emotion_lex[word] = []
                # add this emotion to the word's emotion list
                nrc_emotion_lex[word].append(emotion)

print(f'  loaded {len(nrc_emotion_lex)} words')


# load nrc vad (valence-arousal-dominance) lexicon from file
# this lexicon provides continuous emotional dimensions for words
# valence: how positive/negative (0=very negative, 1=very positive)
# arousal: how calm/excited (0=very calm, 1=very excited)
# dominance: how powerless/powerful (0=powerless, 1=powerful)
print(f'loading nrc vad lexicon from {NRC_VAD_PATH}...')
nrc_vad_lex = {}

with open(NRC_VAD_PATH, 'r', encoding='utf-8') as f:
    # skip the header line (column names)
    next(f)
    for line in f:
        # split tab-delimited line
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            word = parts[0]
            try:
                # parse floating point values for each dimension
                valence = float(parts[1])
                arousal = float(parts[2])
                dominance = float(parts[3])
                # store as tuple of three values
                nrc_vad_lex[word] = (valence, arousal, dominance)
            except ValueError:
                # skip lines with malformed numbers
                continue

print(f'  loaded {len(nrc_vad_lex)} words')


# broadcast lexicons to all spark worker nodes
# broadcasting sends the data once to each worker instead of with every task
# this is much more efficient for read-only reference data
# all workers will now have local copies of both lexicons in memory
nrc_lexicon_bc = spark.sparkContext.broadcast(nrc_emotion_lex)
nrc_vad_bc = spark.sparkContext.broadcast(nrc_vad_lex)
print('lexicons broadcasted to spark workers')


# load goemotions dataset from huggingface
# this dataset contains 58k reddit comments labeled with emotions
# using 'raw' config to get original multi-label format
print('loading goemotions dataset from huggingface...')
dataset = load_dataset('go_emotions', 'raw')
all_data = dataset['train']

# sample a fraction of the data to reduce memory usage and training time
# using 50% by default, but you can adjust SAMPLE_SIZE constant above
num_samples = int(len(all_data) * SAMPLE_SIZE)
sampled_data = all_data.select(range(num_samples))
print(f'  sampled {num_samples} records ({SAMPLE_SIZE*100}% of full dataset)')

# split sampled data into training and test sets
# training data is used to fit the model
# test data is held out to evaluate how well the model generalizes
split_idx = int(len(sampled_data) * TRAIN_SPLIT)
train_data = sampled_data.select(range(split_idx))
test_data = sampled_data.select(range(split_idx, len(sampled_data)))
print(f'  train: {len(train_data)} examples')
print(f'  test: {len(test_data)} examples')


# define user-defined functions (udfs) for feature extraction
# these will be applied to dataframe columns in parallel across spark workers

# function to extract nrc emotion features from text
# counts how many words in the text match each of the 8 plutchik emotions
def extract_nrc_features(text):
    # tokenize text into lowercase words using regex
    # \b ensures word boundaries, [a-z]+ matches alphabetic words
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # initialize counter for each emotion
    emotion_counts = {e: 0.0 for e in PLUTCHIK_EMOTIONS}

    # for each word, check if it's in the emotion lexicon
    for word in words:
        if word in nrc_lexicon_bc.value:
            # get list of emotions associated with this word
            for emotion in nrc_lexicon_bc.value[word]:
                # increment counter if emotion is one of our 8 target emotions
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1.0

    # return counts as a list in consistent order (important for ml)
    return [emotion_counts[e] for e in PLUTCHIK_EMOTIONS]

# function to extract vad (valence-arousal-dominance) features from text
# computes average vad scores across all words in the text
def extract_vad_features(text):
    # tokenize text into lowercase words
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # accumulate vad scores for all words that have them
    valence_sum = arousal_sum = dominance_sum = count = 0
    for word in words:
        if word in nrc_vad_bc.value:
            # unpack the three vad dimensions for this word
            v, a, d = nrc_vad_bc.value[word]
            valence_sum += v
            arousal_sum += a
            dominance_sum += d
            count += 1

    # return averages if any words were found in lexicon
    # otherwise return neutral values (0.5 for each dimension)
    if count > 0:
        return [valence_sum / count, arousal_sum / count, dominance_sum / count]
    return [0.5, 0.5, 0.5]

# function to convert goemotions multi-labels to plutchik labels
# goemotions has 28 emotions, we map them to 8 plutchik emotions
def get_plutchik_labels(goemotions_dict):
    # use set to avoid duplicates (multiple goemotions can map to same plutchik)
    plutchik_set = set()

    # check each of the 28 goemotions labels
    for emotion in GOEMOTIONS_LABELS:
        # if this emotion is present in the example (value=1)
        if goemotions_dict.get(emotion, 0) == 1:
            # map to corresponding plutchik emotion
            plutchik = EMOTION_MAPPING.get(emotion)
            # add to set if it's a valid plutchik emotion (not none/neutral)
            if plutchik:
                plutchik_set.add(plutchik)

    # return binary array: 1.0 if emotion present, 0.0 if absent
    # order must match PLUTCHIK_EMOTIONS list
    return [1.0 if e in plutchik_set else 0.0 for e in PLUTCHIK_EMOTIONS]

# function to combine all feature types into single vector
# concatenates tf-idf features + nrc emotion counts + vad scores
def combine_features(tfidf, nrc, vad):
    # convert sparse tf-idf vector to dense array
    # sparse vectors save memory but we need dense for concatenation
    tfidf_dense = tfidf.toArray().tolist() if hasattr(tfidf, 'toArray') else list(tfidf)

    # concatenate: tf-idf (500d) + nrc (8d) + vad (3d) = 511 total dimensions
    return Vectors.dense(tfidf_dense + nrc + vad)

# register python functions as spark udfs so they can be applied to columns
# ArrayType(DoubleType()) means function returns array of floating point numbers
extract_nrc_udf = udf(extract_nrc_features, ArrayType(DoubleType()))
extract_vad_udf = udf(extract_vad_features, ArrayType(DoubleType()))
get_labels_udf = udf(get_plutchik_labels, ArrayType(DoubleType()))
combine_udf = udf(combine_features, VectorUDT())


# prepare training data
print('preparing training data...')

# convert huggingface dataset to spark dataframe
# each row has text and a dictionary of all goemotions labels
train_df = spark.createDataFrame([
    (row['text'], {e: row.get(e, 0) for e in GOEMOTIONS_LABELS})
    for row in train_data
], ['text', 'goemotions'])

# apply feature extraction udfs to create new columns
# nrc_features: 8-element array of emotion word counts
train_df = train_df.withColumn('nrc_features', extract_nrc_udf(col('text')))
# vad_features: 3-element array of average valence, arousal, dominance
train_df = train_df.withColumn('vad_features', extract_vad_udf(col('text')))
# labels: 8-element binary array of plutchik emotion labels
train_df = train_df.withColumn('labels', get_labels_udf(col('goemotions')))

# tokenize text into individual words for tf-idf processing
# this creates a new 'words' column with array of tokens
tokenizer = Tokenizer(inputCol='text', outputCol='words')
train_df = tokenizer.transform(train_df)

# compute term frequency using hashing trick
# hashing maps words to fixed-size feature space (500 buckets)
# this avoids having to maintain a vocabulary dictionary
hashing_tf = HashingTF(inputCol='words', outputCol='raw_features', numFeatures=TFIDF_FEATURES)
train_df = hashing_tf.transform(train_df)

# compute inverse document frequency (idf)
# idf downweights common words (like "the", "is") that appear in many documents
# this emphasizes words that are distinctive to specific documents
idf = IDF(inputCol='raw_features', outputCol='tfidf_features')
idf_model = idf.fit(train_df)  # fit idf on training corpus
train_df = idf_model.transform(train_df)  # apply transformation

# combine all feature types into single feature vector
# this creates 'features' column with concatenated tf-idf + nrc + vad
train_df = train_df.withColumn('features',
    combine_udf(col('tfidf_features'), col('nrc_features'), col('vad_features')))

# cache training dataframe in memory for faster access
# we'll iterate over it 8 times (once per emotion), so caching improves speed
train_df.cache()
print(f'prepared {train_df.count()} training examples')


# prepare test data using same transformations
print('preparing test data...')

# convert test data to spark dataframe
test_df = spark.createDataFrame([
    (row['text'], {e: row.get(e, 0) for e in GOEMOTIONS_LABELS})
    for row in test_data
], ['text', 'goemotions'])

# apply same feature extraction pipeline as training data
# important: use same tokenizer and idf_model fitted on training data
test_df = test_df.withColumn('nrc_features', extract_nrc_udf(col('text')))
test_df = test_df.withColumn('vad_features', extract_vad_udf(col('text')))
test_df = test_df.withColumn('labels', get_labels_udf(col('goemotions')))

# apply same tokenization and tf-idf transformations
test_df = tokenizer.transform(test_df)
test_df = hashing_tf.transform(test_df)
test_df = idf_model.transform(test_df)  # use idf fitted on training data, not refit

# combine features into single vector
test_df = test_df.withColumn('features',
    combine_udf(col('tfidf_features'), col('nrc_features'), col('vad_features')))

# cache test data in memory for evaluation
test_df.cache()
print(f'prepared {test_df.count()} test examples')


# train one-vs-rest binary classifiers for each emotion
# each classifier learns to predict whether its specific emotion is present
print('training logistic regression models...')
models = {}  # dictionary to store trained models

# iterate over each of the 8 plutchik emotions
for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    print(f'  training {emotion} classifier...')

    # create udf to extract binary label for this specific emotion
    # extracts the idx-th element from the labels array
    get_label_udf = udf(lambda labels: float(labels[idx]), DoubleType())

    # add 'label' column with binary target for this emotion
    # 1.0 if emotion is present, 0.0 if absent
    emotion_train = train_df.withColumn('label', get_label_udf(col('labels')))

    # initialize logistic regression classifier with hyperparameters
    # maxIter: number of optimization iterations (more = better fit but slower)
    # regParam: regularization strength (prevents overfitting)
    # elasticNetParam=0.0: use L2 regularization only (ridge regression)
    lr = LogisticRegression(
        maxIter=MAX_ITERATIONS,
        regParam=REGULARIZATION,
        elasticNetParam=0.0
    )

    # train the model on this emotion's binary classification task
    model = lr.fit(emotion_train)

    # store trained model in dictionary
    models[emotion] = model

print('  training complete')


# make predictions on test set using all trained models
print('making predictions...')

# collect predictions from each emotion classifier
predictions_dfs = []

for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    # extract binary label for this emotion in test set
    get_label_udf = udf(lambda labels: float(labels[idx]), DoubleType())
    emotion_test = test_df.withColumn('label', get_label_udf(col('labels')))

    # apply trained model to make predictions
    predictions = models[emotion].transform(emotion_test)

    # extract binary prediction (0 or 1)
    predictions = predictions.withColumn(f'pred_{emotion}', col('prediction'))

    # extract probability of positive class (emotion present)
    # probability column is vector [prob_absent, prob_present]
    # we want the second element (index 1)
    predictions = predictions.withColumn(f'prob_{emotion}',
        vector_to_array(col('probability'))[1])

    # keep only text, prediction, and probability columns
    predictions_dfs.append(predictions.select('text', f'pred_{emotion}', f'prob_{emotion}'))

# combine predictions from all 8 models into single dataframe
# start with text and true labels from test set
combined = test_df.select('text', 'labels')

# join predictions from each emotion model
# this adds pred_joy, prob_joy, pred_sadness, prob_sadness, etc.
for pred_df in predictions_dfs:
    combined = combined.join(pred_df, on='text', how='left')

# collect all probability columns into single array
prob_cols = [col(f'prob_{e}') for e in PLUTCHIK_EMOTIONS]
combined = combined.withColumn('probabilities', array(*prob_cols))

# apply custom decision threshold to probabilities
# default spark threshold is 0.5, but we use 0.25 for better recall
# lower threshold means we predict positive more often (catch more true positives)
apply_threshold_udf = udf(
    lambda probs: [1.0 if p >= DECISION_THRESHOLD else 0.0 for p in probs],
    ArrayType(DoubleType())
)
combined = combined.withColumn('predictions', apply_threshold_udf(col('probabilities')))


# evaluate model performance on test set
print(f'evaluating performance (threshold={DECISION_THRESHOLD})...')
print('per-emotion metrics:')

# compute precision, recall, and f1 score for each emotion
for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    # extract true labels and predictions for this emotion only
    emotion_results = combined.select(
        (col('labels')[idx]).alias('label'),
        (col('predictions')[idx]).alias('prediction')
    )

    # compute confusion matrix elements
    # true positive: correctly predicted emotion is present
    tp = emotion_results.filter((col('label') == 1.0) & (col('prediction') == 1.0)).count()
    # false positive: incorrectly predicted emotion is present (it's not)
    fp = emotion_results.filter((col('label') == 0.0) & (col('prediction') == 1.0)).count()
    # false negative: failed to predict emotion that actually is present
    fn = emotion_results.filter((col('label') == 1.0) & (col('prediction') == 0.0)).count()

    # compute standard classification metrics
    # precision: of all positive predictions, how many were correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall: of all true positives, how many did we find?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1: harmonic mean of precision and recall (balanced metric)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # print metrics for this emotion
    print(f'  {emotion:12s}: p={precision:.3f} r={recall:.3f} f1={f1:.3f}')


# compute overall multi-label metrics using micro-averaging
# micro-averaging treats each label instance equally across all emotions
print('overall multi-label metrics (micro-averaged):')

# define function to count matches across all emotions for a single example
def calculate_matches(labels, predictions):
    # count correctly predicted emotions (both label and prediction are 1)
    matches = sum(1 for i in range(len(labels)) if labels[i] == 1.0 and predictions[i] == 1.0)
    # count total number of true emotion labels
    true_count = sum(1 for x in labels if x == 1.0)
    # count total number of predicted emotions
    pred_count = sum(1 for x in predictions if x == 1.0)
    # return as floats for aggregation
    return [float(matches), float(true_count), float(pred_count)]

# register as udf
calculate_matches_udf = udf(calculate_matches, ArrayType(DoubleType()))

# apply to each row to get per-example match counts
metrics_df = combined.withColumn('metrics',
    calculate_matches_udf(col('labels'), col('predictions')))

# aggregate across all test examples
# sum up total matches, true labels, and predictions
total_matched = metrics_df.select(col('metrics')[0]).rdd.map(lambda x: x[0]).sum()
total_true = metrics_df.select(col('metrics')[1]).rdd.map(lambda x: x[0]).sum()
total_predicted = metrics_df.select(col('metrics')[2]).rdd.map(lambda x: x[0]).sum()

# compute overall precision, recall, and f1
overall_precision = total_matched / total_predicted if total_predicted > 0 else 0
overall_recall = total_matched / total_true if total_true > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

print(f'  precision: {overall_precision:.3f}')
print(f'  recall: {overall_recall:.3f}')
print(f'  f1-score: {overall_f1:.3f}')


# display sample predictions to illustrate model behavior
print('sample predictions:')

# collect all results from distributed workers to driver
all_samples = combined.select('text', 'labels', 'predictions', 'probabilities').collect()

# select diverse samples by taking every nth example
# this gives variety rather than just the first 10 consecutive examples
sample_indices = range(0, len(all_samples), len(all_samples) // 10)

# show first 10 diverse samples
for i, sample_idx in enumerate(sample_indices[:10], 1):
    sample = all_samples[sample_idx]

    # extract emotion names where true label is 1
    true_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['labels']) if val == 1.0]

    # extract emotion names where prediction is 1
    pred_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['predictions']) if val == 1.0]

    # collect all probabilities above 0.2 to see near-misses
    probs = {PLUTCHIK_EMOTIONS[idx]: sample['probabilities'][idx]
             for idx in range(len(PLUTCHIK_EMOTIONS)) if sample['probabilities'][idx] > 0.2}

    # display sample with truncated text (first 80 characters)
    print(f'\n{i}. {sample["text"][:80]}...')
    print(f'   true: {true_labels}')
    print(f'   pred: {pred_labels}')
    # show probabilities sorted from highest to lowest
    print(f'   probs: {dict(sorted(probs.items(), key=lambda x: -x[1]))}')


print('pipeline complete')

# stop spark session and release all resources
spark.stop()
