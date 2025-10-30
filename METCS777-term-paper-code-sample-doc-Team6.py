from pyspark.sql import SparkSession
from datasets import load_dataset
from nrclex import NRCLex
import time

spark = SparkSession.builder.appName("GoEmotions-NRC-RDD").getOrCreate()
sc = spark.sparkContext

# load goemotions dataset into rdd
dataset = load_dataset("go_emotions", "raw")
data = dataset["train"]
rdd = sc.parallelize(data)

# GoEmotions labels (27 emotions + neutral)
GOEMOTIONS = [
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

# NRC lexicon emotions to GoEmotions map
NRC_MAP = {
    "joy": ["joy", "amusement", "excitement", "optimism"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization"],
    "disgust": ["disgust", "embarrassment"],
    "trust": ["approval", "admiration", "gratitude", "caring"],
    "anticipation": ["curiosity", "desire", "optimism"],
}

# process with rdd operations
start = time.time()

# step 1: analyze with NRCLex and extract true labels
results = rdd.map(
    lambda r: {
        "text": r["text"],
        "true_labels": [e for e in GOEMOTIONS if r.get(e, 0) == 1],
        "nrc_emotions": {
            e: f
            for e, f in NRCLex(r["text"]).affect_frequencies.items()
            if e not in ["positive", "negative"] and f > 0
        },
    }
)

# step 2: map NRC to GoEmotions predictions
results = results.map(
    lambda r: {
        **r,
        "predicted": list(
            set(
                [
                    label
                    for nrc in r["nrc_emotions"].keys()
                    for label in NRC_MAP.get(nrc, [])
                ]
            )
        ),
    }
)

# step 3: calculate matches
results = results.map(
    lambda r: {
        **r,
        "matched": list(set(r["predicted"]) & set(r["true_labels"])),
        "has_match": len(set(r["predicted"]) & set(r["true_labels"])) > 0,
    }
).cache()

total = results.count()
print(f"Processed {total} records in {time.time() - start:.1f}s")

# sample results
for r in results.take(3):
    print(f"\nText: {r['text'][:60]}...")
    print(f"  True: {r['true_labels']}")
    print(f"  Predicted: {r['predicted']}")
    print(f"  Matched: {r['matched']}")

# metrics
correct = results.filter(lambda r: r["has_match"]).count()
print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.1f}%")

# emotion counts
true_counts = (
    results.flatMap(lambda r: r["true_labels"])
    .map(lambda e: (e, 1))
    .reduceByKey(lambda a, b: a + b)
)
print("\nTop true emotions:")
for e, c in true_counts.sortBy(lambda x: -x[1]).take(5):
    print(f"  {e}: {c}")

pred_counts = (
    results.flatMap(lambda r: r["predicted"])
    .map(lambda e: (e, 1))
    .reduceByKey(lambda a, b: a + b)
)
print("\nTop predicted emotions:")
for e, c in pred_counts.sortBy(lambda x: -x[1]).take(5):
    print(f"  {e}: {c}")

sc.stop()
