from pyspark.sql import SparkSession
from datasets import load_dataset
from collections import Counter
import re
import time

# create spark session and context for distributed processing
spark = SparkSession.builder.appName("GoEmotions-NRC-VAD").getOrCreate()
sc = spark.sparkContext

# step 1: load goemotions dataset
# goemotions is a reddit comments dataset with 28 emotion labels
dataset = load_dataset("go_emotions", "raw")
train_data = dataset["train"]
# parallelize converts the data to an rdd with 100 partitions for distributed processing
rdd = sc.parallelize(train_data, numSlices=100)

print(f"loaded {len(train_data)} records from goemotions")

# step 2: define emotion labels and mappings
# these are all 28 emotion categories from the goemotions dataset
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# map the 28 goemotions labels to plutchik's 8 basic emotions
# this simplifies the output space and makes visualization cleaner
# each goemotions label maps to exactly one plutchik emotion (many-to-one mapping)
GOEMOTIONS_TO_PLUTCHIK = {
    # joy - includes happiness, amusement, excitement, love
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "gratitude": "joy",

    # sadness - includes grief, disappointment, remorse
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",

    # anger - includes annoyance and disapproval
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",

    # fear - includes nervousness
    "fear": "fear",
    "nervousness": "fear",

    # surprise - includes realization and confusion
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",

    # disgust - includes embarrassment
    "disgust": "disgust",
    "embarrassment": "disgust",

    # trust - includes approval, admiration, caring
    "approval": "trust",
    "admiration": "trust",
    "caring": "trust",

    # anticipation - includes curiosity and desire
    "curiosity": "anticipation",
    "desire": "anticipation",

    # neutral doesn't map to any plutchik emotion
    "neutral": None
}

# step 3: load nrc emotion lexicon (tells us which emotions are associated with each word)
print("loading nrc emotion lexicon...")
nrc_lexicon = {}
emotion_lexicon_path = "/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# read the tab-delimited file
with open(emotion_lexicon_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, association = parts
            # only keep entries where association = 1 (word has this emotion)
            # exclude positive/negative as they're sentiment not emotion
            if int(association) == 1 and emotion not in ['positive', 'negative']:
                if word not in nrc_lexicon:
                    nrc_lexicon[word] = set()
                nrc_lexicon[word].add(emotion)

# convert sets to lists for serialization to spark workers
nrc_lexicon = {word: list(emotions) for word, emotions in nrc_lexicon.items()}
print(f"loaded {len(nrc_lexicon)} words from nrc emotion lexicon")

# step 4: load nrc vad lexicon (provides valence, arousal, dominance scores)
# this is the key for intensity scoring!
print("loading nrc vad lexicon...")
nrc_vad = {}
vad_path = "/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-VAD-Lexicon.txt"

with open(vad_path, 'r') as f:
    next(f)  # skip header line (Word Valence Arousal Dominance)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            word, valence, arousal, dominance = parts
            try:
                nrc_vad[word] = {
                    'valence': float(valence),      # 0-1: how negative/positive (pleasantness)
                    'arousal': float(arousal),       # 0-1: how calm/excited (intensity/activation)
                    'dominance': float(dominance)    # 0-1: how submissive/dominant (control/power)
                }
            except ValueError:
                # skip lines with invalid numbers
                continue

print(f"loaded {len(nrc_vad)} words from nrc vad lexicon")

# step 5: define intensity modifiers that appear before emotion words
# these words amplify, diminish, or negate the emotional intensity
INTENSIFIERS = {
    # amplifiers - make emotions stronger (multiply by 1.3-2.0)
    "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "really": 1.4,
    "absolutely": 1.8, "totally": 1.6, "completely": 1.7, "utterly": 1.9,
    "so": 1.3, "too": 1.3, "highly": 1.5, "deeply": 1.6, "truly": 1.5,
    "super": 1.4, "exceptionally": 1.8, "extraordinarily": 2.0,

    # diminishers - make emotions weaker (multiply by 0.3-0.8)
    "slightly": 0.5, "somewhat": 0.6, "rather": 0.7, "fairly": 0.7,
    "pretty": 0.7, "quite": 0.8, "relatively": 0.7, "moderately": 0.6,
    "a bit": 0.5, "kind of": 0.6, "sort of": 0.6, "barely": 0.3,
    "hardly": 0.3, "scarcely": 0.3, "almost": 0.8, "nearly": 0.8,

    # negators - flip or nullify emotions (negative value means negate)
    "not": -1.0, "no": -1.0, "never": -1.0, "neither": -1.0,
    "nobody": -1.0, "nothing": -1.0, "nowhere": -1.0, "none": -1.0,
    "without": -0.5, "lack": -0.5, "lacking": -0.5,
}

# opposite emotions for handling negation
# e.g., "not happy" should increase sadness, not just decrease joy
OPPOSITE_EMOTIONS = {
    "joy": "sadness",
    "sadness": "joy",
    "anger": "trust",
    "trust": "anger",
    "fear": "anticipation",
    "anticipation": "fear",
    "surprise": "anticipation",
    "disgust": "trust",
}

# categorize emotions by their valence for alignment with vad scores
# this helps us know if high valence should amplify or diminish an emotion
POSITIVE_EMOTIONS = {'joy', 'trust', 'anticipation'}
NEGATIVE_EMOTIONS = {'sadness', 'anger', 'fear', 'disgust'}
NEUTRAL_EMOTIONS = {'surprise'}

# step 6: broadcast all lexicons to spark workers
# broadcasting sends a read-only copy to each worker node efficiently
# this avoids sending the data with every task
nrc_broadcast = sc.broadcast(nrc_lexicon)
vad_broadcast = sc.broadcast(nrc_vad)
modifiers_broadcast = sc.broadcast(INTENSIFIERS)
opposites_broadcast = sc.broadcast(OPPOSITE_EMOTIONS)

# step 7: define function to calculate emotion scores using vad
def calculate_emotion_scores_vad(words):
    """
    calculate weighted emotion scores by combining:
    1. nrc emotion lexicon - which emotions are present in each word
    2. nrc vad lexicon - arousal (intensity), valence (positive/negative), dominance (control)
    3. contextual modifiers - words that amplify, diminish, or negate emotions

    returns a dictionary of emotion: score pairs (0-1 range)
    """
    emotion_scores = {}

    # iterate through each word in the text
    for i, word in enumerate(words):
        # check if this word has associated emotions in nrc lexicon
        if word in nrc_broadcast.value:
            # get vad scores for this word (arousal = intensity!)
            # if word not in vad lexicon, use neutral defaults (0.5)
            vad = vad_broadcast.value.get(word, {
                'valence': 0.5,    # neutral valence
                'arousal': 0.5,    # medium arousal/intensity
                'dominance': 0.5   # neutral dominance
            })

            # arousal is our base intensity - how emotionally activated this word is
            # high arousal (near 1.0) = intense emotion like "terrified" or "ecstatic"
            # low arousal (near 0.0) = calm emotion like "content" or "peaceful"
            base_intensity = vad['arousal']

            # valence tells us if the word is positive or negative
            # high valence (near 1.0) = positive like "wonderful"
            # low valence (near 0.0) = negative like "horrible"
            valence = vad['valence']

            # dominance tells us about control/power
            # high dominance (near 1.0) = controlling like "commanding"
            # low dominance (near 0.0) = submissive like "helpless"
            dominance = vad['dominance']

            # check for contextual modifiers in the previous 2 words
            # e.g., "very" or "not" before our emotion word
            modifier_strength = 1.0  # default: no modification
            negated = False          # flag for negation words

            # look back up to 2 words
            for j in range(max(0, i-2), i):
                prev_word = words[j]
                if prev_word in modifiers_broadcast.value:
                    mod_value = modifiers_broadcast.value[prev_word]
                    if mod_value < 0:  # this is a negation word
                        negated = True
                    else:
                        # this is an amplifier or diminisher
                        # multiply the modifier strength (e.g., "very extremely" would be 1.5 * 2.0)
                        modifier_strength *= mod_value

            # get all emotions associated with this word
            emotions = nrc_broadcast.value[word]

            # calculate score for each emotion
            for emotion in emotions:
                # start with base intensity from arousal
                intensity = base_intensity * modifier_strength

                # adjust intensity based on valence alignment
                # if emotion and word valence match, amplify; if mismatch, reduce
                if emotion in POSITIVE_EMOTIONS:
                    # positive emotions should be amplified by positive valence
                    # range: 0.5 (negative word) to 1.0 (positive word)
                    valence_adjustment = 0.5 + (valence * 0.5)
                    intensity *= valence_adjustment
                elif emotion in NEGATIVE_EMOTIONS:
                    # negative emotions should be amplified by negative valence
                    # range: 0.5 (positive word) to 1.0 (negative word)
                    valence_adjustment = 0.5 + ((1 - valence) * 0.5)
                    intensity *= valence_adjustment

                # adjust certain emotions based on dominance
                # emotions like anger involve more control/dominance
                # emotions like fear involve less control/dominance
                if emotion == 'anger' or emotion == 'disgust':
                    # high dominance amplifies anger/disgust
                    # range: 0.7 (low dominance) to 1.0 (high dominance)
                    intensity *= (0.7 + dominance * 0.3)
                elif emotion == 'fear' or emotion == 'sadness':
                    # low dominance amplifies fear/sadness
                    # range: 0.7 (high dominance) to 1.0 (low dominance)
                    intensity *= (0.7 + (1 - dominance) * 0.3)

                # handle negation by flipping to opposite emotion
                # e.g., "not happy" becomes sadness instead of joy
                if negated:
                    if emotion in opposites_broadcast.value:
                        # flip to opposite emotion
                        emotion = opposites_broadcast.value[emotion]
                        # slightly reduce intensity for negation (0.8x)
                        intensity *= 0.8
                    else:
                        # no clear opposite, so greatly reduce this emotion
                        intensity *= 0.2

                # accumulate scores (words can trigger same emotion multiple times)
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0
                emotion_scores[emotion] += intensity

    # normalize scores - improved approach
    if emotion_scores:
        # don't penalize short texts as much - just do basic normalization
        # keep top 1-3 emotions strong, suppress weaker ones
        max_score = max(emotion_scores.values())

        if max_score > 0:
            # simple normalization: divide by max to get 0-1 range
            # this preserves relative differences better
            emotion_scores = {
                e: score / max_score
                for e, score in emotion_scores.items()
            }

            # only keep emotions that are at least 40% of the max
            # this reduces over-prediction of weak emotions
            emotion_scores = {
                e: score
                for e, score in emotion_scores.items()
                if score >= 0.4
            }

    return emotion_scores

# step 8: process dataset with rdd operations
start = time.time()

# step 8a: extract true labels from goemotions and tokenize text
# map transforms each record by extracting relevant fields
results = rdd.map(lambda r: {
    'text': r['text'],  # original reddit comment
    # tokenize: extract all lowercase alphabetic words
    'words': re.findall(r'\b[a-z]+\b', r['text'].lower()),
    # get all goemotions labels where the value is 1 (emotion is present)
    'true_goemotions': [e for e in GOEMOTIONS if r.get(e, 0) == 1],
})

# step 8b: map true goemotions labels to plutchik's 8 emotions
# this creates our ground truth in the plutchik space for fair comparison
results = results.map(lambda r: {
    **r,  # keep all existing fields
    # convert each goemotions label to its plutchik equivalent
    # filter out None (neutral has no plutchik mapping)
    'true_plutchik': list(set([
        GOEMOTIONS_TO_PLUTCHIK[label]
        for label in r['true_goemotions']
        if GOEMOTIONS_TO_PLUTCHIK.get(label) is not None
    ]))
})

# step 8c: calculate vad-based emotion scores for each text
# this is where the magic happens - combining emotion lexicon with vad intensity
results = results.map(lambda r: {
    **r,
    # call our vad scoring function on the tokenized words
    # returns dict of emotion: score (0-1) pairs
    'emotion_scores': calculate_emotion_scores_vad(r['words'])
})

# step 8d: threshold scores to get binary predictions
# only predict emotions that exceed the threshold
# this converts continuous scores to discrete predictions for evaluation
# LOWERED threshold since we're now filtering out weak emotions in the scoring function
PREDICTION_THRESHOLD = 0.5  # only predict strong emotions (top 40% get through, then must exceed 0.5 of those)

results = results.map(lambda r: {
    **r,
    # filter emotion scores to only include those above threshold
    'predicted_plutchik': [
        emotion for emotion, score in r['emotion_scores'].items()
        if score >= PREDICTION_THRESHOLD
    ]
})

# step 8e: calculate matches between predictions and ground truth
# for multi-label evaluation, we need to know which predictions were correct
results = results.map(lambda r: {
    **r,
    # matched emotions = intersection of predicted and true sets
    'matched': list(set(r['predicted_plutchik']) & set(r['true_plutchik'])),
    # count metrics for later aggregation
    'num_matched': len(set(r['predicted_plutchik']) & set(r['true_plutchik'])),
    'num_true': len(r['true_plutchik']),
    'num_predicted': len(r['predicted_plutchik'])
}).cache()  # cache keeps results in memory for multiple operations

# trigger computation by calling count (rdds are lazy evaluated)
total = results.count()
print(f"processed {total} records in {time.time() - start:.1f}s")

# step 9: show sample results
print("\nsample results:")
for r in results.take(10):
    print(f"\ntext: {r['text'][:80]}...")
    print(f"  true emotions: {r['true_plutchik']}")
    print(f"  emotion scores: {dict(sorted(r['emotion_scores'].items(), key=lambda x: -x[1]))}")
    print(f"  predicted (threshold={PREDICTION_THRESHOLD}): {r['predicted_plutchik']}")
    print(f"  matched: {r['matched']}")

# step 10: calculate overall performance metrics
# count records with predictions and matches
records_with_predictions = results.filter(lambda r: len(r['predicted_plutchik']) > 0).count()
records_with_matches = results.filter(lambda r: r['num_matched'] > 0).count()

# aggregate counts for micro-averaged metrics
# micro-averaging treats each label instance equally (good for imbalanced datasets)
total_true = results.map(lambda r: r['num_true']).reduce(lambda a, b: a + b)
total_predicted = results.map(lambda r: r['num_predicted']).reduce(lambda a, b: a + b)
total_matched = results.map(lambda r: r['num_matched']).reduce(lambda a, b: a + b)

# calculate precision, recall, f1
precision = total_matched / total_predicted if total_predicted > 0 else 0
recall = total_matched / total_true if total_true > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nmulti-label metrics (micro-averaged):")
print(f"  precision: {precision:.3f} ({total_matched}/{total_predicted})")
print(f"  recall: {recall:.3f} ({total_matched}/{total_true})")
print(f"  f1-score: {f1:.3f}")
print(f"  records with any match: {records_with_matches}/{total} = {records_with_matches/total*100:.1f}%")
print(f"  prediction threshold: {PREDICTION_THRESHOLD}")

# step 11: calculate per-emotion performance
# this shows which emotions are easier/harder to detect
print("\nper-emotion performance:")
all_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]

for emotion in all_emotions:
    # for each record, check if emotion is in true and predicted sets
    emotion_results = results.map(lambda r: {
        'true_has': emotion in r['true_plutchik'],
        'pred_has': emotion in r['predicted_plutchik']
    })

    # calculate confusion matrix elements
    tp = emotion_results.filter(lambda r: r['true_has'] and r['pred_has']).count()      # true positive
    fp = emotion_results.filter(lambda r: not r['true_has'] and r['pred_has']).count()  # false positive
    fn = emotion_results.filter(lambda r: r['true_has'] and not r['pred_has']).count()  # false negative
    tn = emotion_results.filter(lambda r: not r['true_has'] and not r['pred_has']).count()  # true negative

    # calculate metrics for this emotion
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_emotion = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # calculate average intensity when emotion is present
    avg_intensity = results.map(
        lambda r: r['emotion_scores'].get(emotion, 0.0)
    ).filter(lambda score: score > 0).mean() if results.filter(
        lambda r: emotion in r['emotion_scores']
    ).count() > 0 else 0.0

    print(f"  {emotion:12s}: p={prec:.3f} r={rec:.3f} f1={f1_emotion:.3f} | avg_intensity={avg_intensity:.3f}")

# step 12: emotion frequency analysis
print("\nemotion frequency (ground truth):")
true_counts = results.flatMap(lambda r: r['true_plutchik']).map(lambda e: (e, 1)).reduceByKey(lambda a, b: a + b)
for e, c in true_counts.sortBy(lambda x: -x[1]).collect():
    print(f"  {e:12s}: {c:6d} ({c/total*100:.1f}%)")

print("\nemotion frequency (predicted):")
pred_counts = results.flatMap(lambda r: r['predicted_plutchik']).map(lambda e: (e, 1)).reduceByKey(lambda a, b: a + b)
for e, c in pred_counts.sortBy(lambda x: -x[1]).collect():
    print(f"  {e:12s}: {c:6d} ({c/total*100:.1f}%)")

# stop spark context and clean up resources
sc.stop()
