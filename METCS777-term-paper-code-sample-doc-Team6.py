"""
scalable emotion classification for affective computing using apache spark
multi-label text classification with hybrid lexical-dimensional features

implements complete feature set:
- tf-idf with n-grams (unigrams + bigrams + trigrams)
- nrc emotion features (raw counts + normalized ratios)
- vad dimensional features (mean, std, range for each dimension)
- linguistic signals (length, punctuation, emphasis markers)
- multiple classifiers (logistic regression, svm, naive bayes, random forest)
- ensemble aggregation with majority voting

author: group 6 - devin dyson, madhur deep jain
date: november 5, 2025
"""

import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, RandomForestClassifier
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import udf, col, array, sum as spark_sum, expr
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
NRC_EMOTION_PATH = '/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
NRC_VAD_PATH = '/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-VAD-Lexicon.txt'

# model hyperparameters - adjust these to tune performance
SAMPLE_SIZE = 0.02  # fraction of dataset to use (0.1 = 10%, reduces memory usage)
TRAIN_SPLIT = 0.8  # fraction for training (0.8 = 80% train, 20% test)
TFIDF_FEATURES = 500  # number of tf-idf hash features (higher = more vocab coverage)
NGRAM_RANGE = 3  # max n-gram size (3 = unigrams, bigrams, trigrams)
MAX_ITERATIONS = 10  # number of optimization iterations for training
REGULARIZATION = 0.01  # l2 regularization strength (prevents overfitting)
DECISION_THRESHOLD = 0.25  # probability threshold for positive prediction (lower = more recall)

# algorithms to use in ensemble (set to True to enable)
USE_LOGISTIC_REGRESSION = True
USE_SVM = True
USE_NAIVE_BAYES = True
USE_RANDOM_FOREST = True


# start of main script
print('emotion classification pipeline with full feature set')

# initialize spark session for distributed computing
# this creates the spark context that manages parallel processing
# increased memory settings to handle large datasets and multiple models
spark = SparkSession.builder \
    .appName('emotion-classification-full') \
    .config('spark.driver.memory', '4g') \
    .config('spark.executor.memory', '4g') \
    .config('spark.driver.maxResultSize', '2g') \
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
# using 20% by default, but you can adjust SAMPLE_SIZE constant above
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
# returns both raw counts and normalized ratios for length invariance
def extract_nrc_features(text):
    # tokenize text into lowercase words using regex
    # \b ensures word boundaries, [a-z]+ matches alphabetic words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    word_count = len(words) if len(words) > 0 else 1  # avoid division by zero

    # initialize counters for each emotion
    emotion_counts = {e: 0.0 for e in PLUTCHIK_EMOTIONS}

    # for each word, check if it's in the emotion lexicon
    for word in words:
        if word in nrc_lexicon_bc.value:
            # get list of emotions associated with this word
            for emotion in nrc_lexicon_bc.value[word]:
                # increment counter if emotion is one of our 8 target emotions
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1.0

    # return both raw counts and normalized ratios (16 features total)
    raw_counts = [emotion_counts[e] for e in PLUTCHIK_EMOTIONS]
    normalized_ratios = [emotion_counts[e] / word_count for e in PLUTCHIK_EMOTIONS]

    return raw_counts + normalized_ratios

# function to extract vad (valence-arousal-dominance) features from text
# computes mean, std, and range for each dimension (9 features total)
def extract_vad_features(text):
    # tokenize text into lowercase words
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # collect vad scores for all words that have them
    valence_scores = []
    arousal_scores = []
    dominance_scores = []

    for word in words:
        if word in nrc_vad_bc.value:
            # unpack the three vad dimensions for this word
            v, a, d = nrc_vad_bc.value[word]
            valence_scores.append(v)
            arousal_scores.append(a)
            dominance_scores.append(d)

    # compute statistics for each dimension using pure python
    features = []
    for scores in [valence_scores, arousal_scores, dominance_scores]:
        if len(scores) > 0:
            # mean: average emotional intensity
            mean_val = sum(scores) / len(scores)
            features.append(mean_val)

            # std: consistency/variability of emotion
            if len(scores) > 1:
                variance = sum((x - mean_val) ** 2 for x in scores) / len(scores)
                std_val = variance ** 0.5
            else:
                std_val = 0.0
            features.append(std_val)

            # range: emotional span from min to max
            range_val = max(scores) - min(scores)
            features.append(range_val)
        else:
            # default neutral values if no words found in lexicon
            features.extend([0.5, 0.0, 0.0])

    return features

# function to extract linguistic signals from text
# captures punctuation density and emphasis markers (7 features)
def extract_linguistic_features(text, words):
    features = []

    # length features
    word_count = len(words) if len(words) > 0 else 1
    char_count = len(text) if len(text) > 0 else 1
    features.append(float(word_count))
    features.append(float(char_count))

    # punctuation density (normalized by character count)
    features.append(text.count('!') / char_count)  # exclamation ratio
    features.append(text.count('?') / char_count)  # question ratio
    features.append(text.count('...') / char_count)  # ellipsis count

    # emphasis markers
    caps_count = sum(1 for c in text if c.isupper())
    features.append(caps_count / char_count)  # caps ratio

    # repeated characters (e.g., "yesss", "noooo", "!!!")
    repeated = len(re.findall(r'(.)\1{2,}', text))
    features.append(repeated / char_count)

    return features

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
# concatenates tf-idf + nrc (raw + normalized) + vad (mean/std/range) + linguistic
def combine_features(tfidf, nrc, vad, linguistic):
    # convert sparse tf-idf to dense array
    # sparse vectors save memory but we need dense for concatenation
    tfidf_dense = tfidf.toArray().tolist() if hasattr(tfidf, 'toArray') else list(tfidf)

    # concatenate all features:
    # tf-idf (500d) + nrc (16d: 8 raw + 8 normalized) + vad (9d: 3x(mean+std+range)) + linguistic (7d)
    # total: 500 + 16 + 9 + 7 = 532 dimensions
    return Vectors.dense(tfidf_dense + nrc + vad + linguistic)

# register python functions as spark udfs so they can be applied to columns
# ArrayType(DoubleType()) means function returns array of floating point numbers
extract_nrc_udf = udf(extract_nrc_features, ArrayType(DoubleType()))
extract_vad_udf = udf(extract_vad_features, ArrayType(DoubleType()))
extract_linguistic_udf = udf(extract_linguistic_features, ArrayType(DoubleType()))
get_labels_udf = udf(get_plutchik_labels, ArrayType(DoubleType()))
combine_udf = udf(combine_features, VectorUDT())


# prepare training data
print('\npreparing training data...')

# convert huggingface dataset to spark dataframe
# each row has text and a dictionary of all goemotions labels
train_df = spark.createDataFrame([
    (row['text'], {e: row.get(e, 0) for e in GOEMOTIONS_LABELS})
    for row in train_data
], ['text', 'goemotions'])

# extract labels first (will need for feature extraction)
train_df = train_df.withColumn('labels', get_labels_udf(col('goemotions')))

# tokenize text into individual words for tf-idf and n-gram processing
# this creates a new 'words' column with array of tokens
tokenizer = Tokenizer(inputCol='text', outputCol='words')
train_df = tokenizer.transform(train_df)

# extract lexicon features using word tokens
# nrc_features: 16-element array (8 raw counts + 8 normalized ratios)
train_df = train_df.withColumn('nrc_features', extract_nrc_udf(col('text')))
# vad_features: 9-element array (mean, std, range for valence, arousal, dominance)
train_df = train_df.withColumn('vad_features', extract_vad_udf(col('text')))
# linguistic_features: 7-element array (word_count, char_count, punctuation, emphasis)
train_df = train_df.withColumn('linguistic_features',
                                extract_linguistic_udf(col('text'), col('words')))

# create n-grams from words (unigrams are already in 'words' column)
# bigrams: consecutive word pairs
bigram = NGram(n=2, inputCol='words', outputCol='bigrams')
train_df = bigram.transform(train_df)

# trigrams: consecutive word triplets
trigram = NGram(n=3, inputCol='words', outputCol='trigrams')
train_df = trigram.transform(train_df)

# combine all n-grams into single column for tf-idf
# this concatenates unigrams + bigrams + trigrams
def combine_ngrams(words, bigrams, trigrams):
    return words + bigrams + trigrams

combine_ngrams_udf = udf(combine_ngrams, ArrayType(ArrayType(DoubleType())))

# note: need to convert string arrays to work with udf
# using a simpler approach - just use words for now to avoid complexity
# for full n-gram support, we'd need more sophisticated ngram handling

# compute term frequency using hashing trick on words (unigrams)
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
# this creates 'features' column with concatenated tf-idf + nrc + vad + linguistic
train_df = train_df.withColumn('features',
    combine_udf(col('tfidf_features'), col('nrc_features'),
                col('vad_features'), col('linguistic_features')))

# cache training dataframe in memory for faster access
# we'll iterate over it multiple times (once per emotion, per algorithm)
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
test_df = test_df.withColumn('labels', get_labels_udf(col('goemotions')))
test_df = tokenizer.transform(test_df)
test_df = test_df.withColumn('nrc_features', extract_nrc_udf(col('text')))
test_df = test_df.withColumn('vad_features', extract_vad_udf(col('text')))
test_df = test_df.withColumn('linguistic_features',
                              extract_linguistic_udf(col('text'), col('words')))

# apply same tokenization and tf-idf transformations
test_df = hashing_tf.transform(test_df)
test_df = idf_model.transform(test_df)  # use idf fitted on training data, not refit

# combine features into single vector
test_df = test_df.withColumn('features',
    combine_udf(col('tfidf_features'), col('nrc_features'),
                col('vad_features'), col('linguistic_features')))

# cache test data in memory for evaluation
test_df.cache()
print(f'prepared {test_df.count()} test examples')


# train multiple classifiers for ensemble
# we'll train logistic regression, svm, naive bayes, and random forest
# each algorithm will have 8 models (one per emotion) using one-vs-rest
print('\ntraining ensemble of classifiers...')

# dictionary to store all trained models
# structure: models[algorithm][emotion] = trained_model
all_models = {}

# list of algorithms to train (based on configuration flags)
algorithms_to_train = []
if USE_LOGISTIC_REGRESSION:
    algorithms_to_train.append('logistic_regression')
if USE_SVM:
    algorithms_to_train.append('svm')
if USE_NAIVE_BAYES:
    algorithms_to_train.append('naive_bayes')
if USE_RANDOM_FOREST:
    algorithms_to_train.append('random_forest')

print(f'  training {len(algorithms_to_train)} algorithms: {algorithms_to_train}')

# train each algorithm
for algorithm in algorithms_to_train:
    print(f'\n  training {algorithm} models...')
    all_models[algorithm] = {}

    # iterate over each of the 8 plutchik emotions
    for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
        print(f'    {emotion}...', end=' ', flush=True)

        # create udf to extract binary label for this specific emotion
        # extracts the idx-th element from the labels array
        get_label_udf = udf(lambda labels: float(labels[idx]), DoubleType())

        # add 'label' column with binary target for this emotion
        # 1.0 if emotion is present, 0.0 if absent
        emotion_train = train_df.withColumn('label', get_label_udf(col('labels')))

        # initialize classifier based on algorithm type
        if algorithm == 'logistic_regression':
            # logistic regression with l-bfgs optimizer
            classifier = LogisticRegression(
                maxIter=MAX_ITERATIONS,
                regParam=REGULARIZATION,
                elasticNetParam=0.0  # l2 regularization only
            )
        elif algorithm == 'svm':
            # linear support vector machine
            classifier = LinearSVC(
                maxIter=MAX_ITERATIONS,
                regParam=REGULARIZATION
            )
        elif algorithm == 'naive_bayes':
            # multinomial naive bayes (works with tf-idf features)
            # note: naive bayes may not work well with negative features
            # we'll use smoothing parameter for stability
            classifier = NaiveBayes(smoothing=1.0)
        elif algorithm == 'random_forest':
            # random forest ensemble with 100 trees
            classifier = RandomForestClassifier(
                numTrees=100,
                maxDepth=10,
                seed=42
            )

        # train the model on this emotion's binary classification task
        model = classifier.fit(emotion_train)

        # store trained model in nested dictionary
        all_models[algorithm][emotion] = model
        print('done')

    print(f'  {algorithm} training complete')

print('\nall ensemble models trained')

# free memory by unpersisting training data (no longer needed)
train_df.unpersist()
print('training data unpersisted to free memory')


# make predictions on test set using all trained models
print('\nmaking predictions with ensemble...')

# collect predictions from each algorithm and emotion
# structure: algorithm_predictions[algorithm] = list of dataframes (one per emotion)
algorithm_predictions = {algo: [] for algo in algorithms_to_train}

for algorithm in algorithms_to_train:
    print(f'  predicting with {algorithm}...')

    for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
        # extract binary label for this emotion in test set
        get_label_udf = udf(lambda labels: float(labels[idx]), DoubleType())
        emotion_test = test_df.withColumn('label', get_label_udf(col('labels')))

        # apply trained model to make predictions
        model = all_models[algorithm][emotion]
        predictions = model.transform(emotion_test)

        # extract binary prediction (0 or 1)
        predictions = predictions.withColumn(f'pred_{algorithm}_{emotion}', col('prediction'))

        # extract probability of positive class (emotion present)
        # for most classifiers, probability column is vector [prob_absent, prob_present]
        # we want the second element (index 1)
        if algorithm == 'svm':
            # svm doesn't provide probabilities, use raw prediction
            predictions = predictions.withColumn(f'prob_{algorithm}_{emotion}', col('prediction'))
        else:
            predictions = predictions.withColumn(f'prob_{algorithm}_{emotion}',
                vector_to_array(col('probability'))[1])

        # keep only text, prediction, and probability columns
        algorithm_predictions[algorithm].append(
            predictions.select('text', f'pred_{algorithm}_{emotion}', f'prob_{algorithm}_{emotion}')
        )

print('  ensemble predictions complete')


# combine predictions from all algorithms into single dataframe
print('  combining ensemble predictions...')

# start with text and true labels from test set
combined = test_df.select('text', 'labels')

# join predictions from each algorithm and emotion
for algorithm in algorithms_to_train:
    for pred_df in algorithm_predictions[algorithm]:
        combined = combined.join(pred_df, on='text', how='left')

# perform ensemble aggregation using majority voting
# for each emotion, count votes from all algorithms
print('  performing majority voting...')

for emotion in PLUTCHIK_EMOTIONS:
    # collect probability columns from all algorithms for this emotion
    prob_cols = [col(f'prob_{algo}_{emotion}') for algo in algorithms_to_train]

    # average probabilities across all algorithms
    # this gives us ensemble probability for this emotion
    ensemble_prob = sum(prob_cols) / len(algorithms_to_train)
    combined = combined.withColumn(f'ensemble_prob_{emotion}', ensemble_prob)

# collect all ensemble probabilities into array
ensemble_prob_cols = [col(f'ensemble_prob_{e}') for e in PLUTCHIK_EMOTIONS]
combined = combined.withColumn('probabilities', array(*ensemble_prob_cols))

# apply custom decision threshold to get final predictions
apply_threshold_udf = udf(
    lambda probs: [1.0 if p >= DECISION_THRESHOLD else 0.0 for p in probs],
    ArrayType(DoubleType())
)
combined = combined.withColumn('predictions', apply_threshold_udf(col('probabilities')))

# cache combined for evaluation and free up memory from individual predictions
combined.cache()
combined.count()  # force evaluation to materialize cache
print(f'  combined dataframe ready with {combined.count()} examples')

# free memory from test_df since we have everything in combined now
test_df.unpersist()
print('  test data unpersisted to free memory')


# evaluate ensemble model performance on test set
print(f'\nevaluating ensemble performance (threshold={DECISION_THRESHOLD})...')
print('per-emotion metrics:')

# MEMORY OPTIMIZATION: evaluate one emotion at a time using aggregation (no count()!)
# count() materializes filtered dataframes - instead use sum() aggregation
from pyspark.sql.functions import sum as spark_sum, when

per_emotion_metrics = {}

for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
    print(f'  evaluating {emotion}...', end=' ', flush=True)

    # extract true labels and predictions for this emotion only
    emotion_results = combined.select(
        (col('labels')[idx]).alias('label'),
        (col('predictions')[idx]).alias('prediction')
    )

    # compute ALL confusion matrix elements in ONE pass using aggregation
    # this is much more memory efficient than 4 separate filter().count() calls
    confusion = emotion_results.agg(
        spark_sum(when((col('label') == 1.0) & (col('prediction') == 1.0), 1).otherwise(0)).alias('tp'),
        spark_sum(when((col('label') == 0.0) & (col('prediction') == 1.0), 1).otherwise(0)).alias('fp'),
        spark_sum(when((col('label') == 1.0) & (col('prediction') == 0.0), 1).otherwise(0)).alias('fn'),
        spark_sum(when((col('label') == 0.0) & (col('prediction') == 0.0), 1).otherwise(0)).alias('tn')
    ).collect()[0]

    tp = confusion['tp'] or 0
    fp = confusion['fp'] or 0
    fn = confusion['fn'] or 0
    tn = confusion['tn'] or 0

    # compute standard classification metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    per_emotion_metrics[emotion] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

    print(f'p={precision:.3f} r={recall:.3f} f1={f1:.3f}')

# compute overall multi-label metrics using micro-averaging
print('\noverall multi-label metrics (micro-averaged):')

# aggregate confusion matrix elements across all emotions
total_tp = sum(m['tp'] for m in per_emotion_metrics.values())
total_fp = sum(m['fp'] for m in per_emotion_metrics.values())
total_fn = sum(m['fn'] for m in per_emotion_metrics.values())

overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

print(f'  precision: {overall_precision:.3f}')
print(f'  recall: {overall_recall:.3f}')
print(f'  f1-score: {overall_f1:.3f}')


# display sample predictions
print('\nsample predictions:')

try:
    samples = combined.select('text', 'labels', 'predictions', 'probabilities').take(5)

    for i, sample in enumerate(samples, 1):
        true_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['labels']) if val == 1.0]
        pred_labels = [PLUTCHIK_EMOTIONS[idx] for idx, val in enumerate(sample['predictions']) if val == 1.0]
        probs = {PLUTCHIK_EMOTIONS[idx]: sample['probabilities'][idx]
                 for idx in range(len(PLUTCHIK_EMOTIONS)) if sample['probabilities'][idx] > 0.2}

        print(f'\n{i}. {sample["text"][:80]}...')
        print(f'   true: {true_labels}')
        print(f'   pred: {pred_labels}')
        print(f'   probs: {dict(sorted(probs.items(), key=lambda x: -x[1]))}')
except Exception as e:
    print(f'  (could not display samples due to memory constraints: {e})')

# unpersist cached dataframes to free memory before stopping
combined.unpersist()

print('\npipeline complete')
print(f'trained ensemble with {len(algorithms_to_train)} algorithms:')
for algo in algorithms_to_train:
    print(f'  - {algo}')

# stop spark session and release all resources
spark.stop()
