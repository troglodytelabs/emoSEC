from pyspark.sql import SparkSession
from datasets import load_dataset
from collections import Counter
import re
import time

spark = SparkSession.builder.appName("GoEmotions-NRC-RDD").getOrCreate()
sc = spark.sparkContext

# step 1: load goemotions dataset
dataset = load_dataset("go_emotions", "raw")
train_data = dataset["train"]
rdd = sc.parallelize(train_data)

print(f"loaded {len(train_data)} records")

# step 2: define emotion labels and mappings
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

NRC_MAP = {
    "joy": ["joy", "amusement", "excitement", "optimism"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization"],
    "disgust": ["disgust", "embarrassment"],
    "trust": ["approval", "admiration", "gratitude", "caring"],
    "anticipation": ["curiosity", "desire", "optimism"]
}

# step 3: load nrc emotion lexicon from file
print("loading nrc lexicon...")
nrc_lexicon = {}
lexicon_path = "/Users/devindyson/Desktop/troglodytelabs/emoSpark/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

with open(lexicon_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, association = parts
            if int(association) == 1:
                if word not in nrc_lexicon:
                    nrc_lexicon[word] = []
                nrc_lexicon[word].append(emotion)

print(f"loaded {len(nrc_lexicon)} words from nrc lexicon")

# step 4: broadcast lexicon to all workers
nrc_broadcast = sc.broadcast(nrc_lexicon)

# step 5: process with rdd operations
start = time.time()

# step 5a: extract true labels and tokenize text
results = rdd.map(lambda r: {
    'text': r['text'],
    'true_labels': [e for e in GOEMOTIONS if r.get(e, 0) == 1],
    'words': re.findall(r'\b[a-z]+\b', r['text'].lower()),
})

# step 5b: extract nrc emotions from words
results = results.map(lambda r: {
    **r,
    'nrc_emotions': [emotion for word in r['words']
                     for emotion in nrc_broadcast.value.get(word, [])
                     if emotion not in ['positive', 'negative']]
})

# step 5c: count nrc emotions
results = results.map(lambda r: {
    **r,
    'nrc_counts': dict(Counter(r['nrc_emotions']))
})

# step 5d: map nrc to goemotions predictions
results = results.map(lambda r: {
    **r,
    'predicted': list(set([label for nrc in r['nrc_counts'].keys()
                          for label in NRC_MAP.get(nrc, [])]))
})

# step 5e: calculate matches
results = results.map(lambda r: {
    **r,
    'matched': list(set(r['predicted']) & set(r['true_labels'])),
    'has_match': len(set(r['predicted']) & set(r['true_labels'])) > 0
}).cache()

total = results.count()
print(f"processed {total} records in {time.time() - start:.1f}s")

# step 6: show sample results
print("\nsample results:")
for r in results.take(3):
    print(f"text: {r['text'][:60]}...")
    print(f"  true: {r['true_labels']}")
    print(f"  nrc emotions: {list(r['nrc_counts'].keys())}")
    print(f"  predicted: {r['predicted']}")
    print(f"  matched: {r['matched']}\n")

# step 7: calculate metrics
correct = results.filter(lambda r: r['has_match']).count()
print(f"accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# step 8: analyze emotion distributions
true_counts = results.flatMap(lambda r: r['true_labels']).map(lambda e: (e, 1)).reduceByKey(lambda a, b: a + b)
print("\ntop 5 true emotions:")
for e, c in true_counts.sortBy(lambda x: -x[1]).take(5):
    print(f"  {e}: {c}")

pred_counts = results.flatMap(lambda r: r['predicted']).map(lambda e: (e, 1)).reduceByKey(lambda a, b: a + b)
print("\ntop 5 predicted emotions:")
for e, c in pred_counts.sortBy(lambda x: -x[1]).take(5):
    print(f"  {e}: {c}")

nrc_counts = results.flatMap(lambda r: list(r['nrc_counts'].keys())).map(lambda e: (e, 1)).reduceByKey(lambda a, b: a + b)
print("\nnrc emotions detected:")
for e, c in nrc_counts.sortBy(lambda x: -x[1]).collect():
    print(f"  {e}: {c}")

sc.stop()
