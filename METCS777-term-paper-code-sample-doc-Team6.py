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

import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.ml.classification import (
    LogisticRegression,
    LinearSVC,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.sql.functions import col, sum as spark_sum, when, lit
from pyspark.sql.types import DoubleType
from datasets import load_dataset

from config import (
    DECISION_THRESHOLD,
    GOEMOTIONS_LABELS,
    MODEL_BASE_PATH,
    MAX_ITERATIONS,
    NRC_EMOTION_PATH,
    NRC_VAD_PATH,
    PLUTCHIK_EMOTIONS,
    REGULARIZATION,
    SAMPLE_SIZE,
    TFIDF_FEATURES,
    TRAIN_SPLIT,
    USE_LOGISTIC_REGRESSION,
    USE_NAIVE_BAYES,
    USE_RANDOM_FOREST,
    USE_SVM,
)
from feature_udfs import (
    get_combine_udf,
    get_extract_linguistic_udf,
    get_extract_nrc_udf,
    get_extract_vad_udf,
    get_labels_udf,
)
from lexicon_utils import (
    broadcast_lexicons,
    load_nrc_emotion_lexicon,
    load_nrc_vad_lexicon,
)
from training_utils import (
    compute_class_weights,
    generate_ensemble_outputs,
    tune_thresholds,
)


def run_pipeline():
    spark = (
        SparkSession.builder.appName("emotion-classification-full")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"spark version: {spark.version}")

    print(f"loading nrc emotion lexicon from {NRC_EMOTION_PATH}...")
    nrc_emotion_lex = load_nrc_emotion_lexicon(NRC_EMOTION_PATH, PLUTCHIK_EMOTIONS)
    print(f"  loaded {len(nrc_emotion_lex)} words")

    print(f"loading nrc vad lexicon from {NRC_VAD_PATH}...")
    nrc_vad_lex = load_nrc_vad_lexicon(NRC_VAD_PATH)
    print(f"  loaded {len(nrc_vad_lex)} words")

    nrc_lexicon_bc, nrc_vad_bc = broadcast_lexicons(spark, nrc_emotion_lex, nrc_vad_lex)
    print("lexicons broadcasted to spark workers")

    print("loading goemotions dataset from huggingface...")
    dataset = load_dataset("go_emotions", "raw")
    all_data = dataset["train"]

    num_samples = int(len(all_data) * SAMPLE_SIZE)
    sampled_data = all_data.select(range(num_samples))
    sampled_rows = [
        (
            idx,
            row["text"],
            {emotion: row.get(emotion, 0) for emotion in GOEMOTIONS_LABELS},
        )
        for idx, row in enumerate(sampled_data)
    ]
    print(f"  sampled {num_samples} records ({SAMPLE_SIZE * 100}% of full dataset)")

    split_idx = int(len(sampled_rows) * TRAIN_SPLIT)
    train_rows = sampled_rows[:split_idx]
    test_rows = sampled_rows[split_idx:]
    print(f"  train: {len(train_rows)} examples")
    print(f"  test: {len(test_rows)} examples")

    model_dir = Path(MODEL_BASE_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)

    labels_udf = get_labels_udf()
    extract_nrc_udf = get_extract_nrc_udf(nrc_lexicon_bc)
    extract_vad_udf = get_extract_vad_udf(nrc_vad_bc)
    extract_linguistic_udf = get_extract_linguistic_udf()
    combine_udf = get_combine_udf()

    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    print("\npreparing training data...")
    full_train_df = spark.createDataFrame(train_rows, ["row_id", "text", "goemotions"])
    full_train_df = full_train_df.withColumn("labels", labels_udf(col("goemotions")))
    full_train_df = tokenizer.transform(full_train_df)
    full_train_df = full_train_df.withColumn(
        "nrc_features", extract_nrc_udf(col("text"))
    )
    full_train_df = full_train_df.withColumn(
        "vad_features", extract_vad_udf(col("text"))
    )
    full_train_df = full_train_df.withColumn(
        "linguistic_features", extract_linguistic_udf(col("text"), col("words"))
    )

    bigram = NGram(n=2, inputCol="words", outputCol="bigrams")
    full_train_df = bigram.transform(full_train_df)
    trigram = NGram(n=3, inputCol="words", outputCol="trigrams")
    full_train_df = trigram.transform(full_train_df)

    hashing_tf = HashingTF(
        inputCol="words", outputCol="raw_features", numFeatures=TFIDF_FEATURES
    )
    full_train_df = hashing_tf.transform(full_train_df)

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(full_train_df)
    idf_output_path = model_dir / "idf_model"
    idf_model.write().overwrite().save(str(idf_output_path))
    full_train_df = idf_model.transform(full_train_df)

    full_train_df = full_train_df.withColumn(
        "features",
        combine_udf(
            col("tfidf_features"),
            col("nrc_features"),
            col("vad_features"),
            col("linguistic_features"),
        ),
    )

    train_df, val_df = full_train_df.randomSplit([0.9, 0.1], seed=42)
    train_df = train_df.cache()
    val_df = val_df.cache()
    train_count = train_df.count()
    val_count = val_df.count()
    print(f"prepared {train_count} training examples")
    print(f"prepared {val_count} validation examples")

    class_weights = compute_class_weights(train_df, train_count, PLUTCHIK_EMOTIONS)
    print("computed per-emotion class weights for imbalance handling")

    print("preparing test data...")
    test_df = spark.createDataFrame(test_rows, ["row_id", "text", "goemotions"])
    test_df = test_df.withColumn("labels", labels_udf(col("goemotions")))
    test_df = tokenizer.transform(test_df)
    test_df = test_df.withColumn("nrc_features", extract_nrc_udf(col("text")))
    test_df = test_df.withColumn("vad_features", extract_vad_udf(col("text")))
    test_df = test_df.withColumn(
        "linguistic_features", extract_linguistic_udf(col("text"), col("words"))
    )

    test_df = hashing_tf.transform(test_df)
    test_df = idf_model.transform(test_df)
    test_df = test_df.withColumn(
        "features",
        combine_udf(
            col("tfidf_features"),
            col("nrc_features"),
            col("vad_features"),
            col("linguistic_features"),
        ),
    )

    test_df.cache()
    print(f"prepared {test_df.count()} test examples")

    print("\ntraining ensemble of classifiers...")
    all_models = {}
    algorithms_to_train = []
    if USE_LOGISTIC_REGRESSION:
        algorithms_to_train.append("logistic_regression")
    if USE_SVM:
        algorithms_to_train.append("svm")
    if USE_NAIVE_BAYES:
        algorithms_to_train.append("naive_bayes")
    if USE_RANDOM_FOREST:
        algorithms_to_train.append("random_forest")

    print(f"  training {len(algorithms_to_train)} algorithms: {algorithms_to_train}")

    for algorithm in algorithms_to_train:
        print(f"\n  training {algorithm} models...")
        all_models[algorithm] = {}
        for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
            print(f"    {emotion}...", end=" ", flush=True)
            emotion_train = train_df.withColumn(
                "label", col("labels").getItem(idx).cast(DoubleType())
            )
            weights = class_weights[emotion]
            emotion_train = emotion_train.withColumn(
                "weight",
                when(col("label") == 1.0, lit(weights["positive"])).otherwise(
                    lit(weights["negative"])
                ),
            )

            if algorithm == "logistic_regression":
                classifier = LogisticRegression(
                    maxIter=MAX_ITERATIONS,
                    regParam=REGULARIZATION,
                    elasticNetParam=0.0,
                )
            elif algorithm == "svm":
                classifier = LinearSVC(maxIter=MAX_ITERATIONS, regParam=REGULARIZATION)
            elif algorithm == "naive_bayes":
                classifier = NaiveBayes(smoothing=1.0)
            elif algorithm == "random_forest":
                classifier = RandomForestClassifier(numTrees=100, maxDepth=10, seed=42)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            classifier = classifier.setWeightCol("weight")
            model = classifier.fit(emotion_train)
            all_models[algorithm][emotion] = model
            save_path = model_dir / algorithm / emotion
            model.write().overwrite().save(str(save_path))
            print("saved")
        print(f"  {algorithm} training complete")

    print("\nall ensemble models trained")
    train_df.unpersist()
    print("training data unpersisted to free memory")

    val_combined = generate_ensemble_outputs(
        val_df,
        "validation",
        algorithms_to_train,
        PLUTCHIK_EMOTIONS,
        all_models,
        DECISION_THRESHOLD,
    )
    tuned_thresholds = tune_thresholds(
        val_combined, PLUTCHIK_EMOTIONS, DECISION_THRESHOLD
    )
    val_combined.unpersist()
    val_df.unpersist()

    print("\nusing tuned thresholds per emotion:")
    for emotion in PLUTCHIK_EMOTIONS:
        print(f"  {emotion}: {tuned_thresholds[emotion]:.3f}")

    metadata = {
        "algorithms": algorithms_to_train,
        "emotions": PLUTCHIK_EMOTIONS,
        "decision_threshold": DECISION_THRESHOLD,
        "thresholds": tuned_thresholds,
        "tfidf_features": TFIDF_FEATURES,
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"model artifacts stored in {model_dir}")

    combined = generate_ensemble_outputs(
        test_df,
        "test",
        algorithms_to_train,
        PLUTCHIK_EMOTIONS,
        all_models,
        DECISION_THRESHOLD,
        thresholds=tuned_thresholds,
    )
    test_df.unpersist()
    print("test data unpersisted to free memory")

    print("\nevaluating ensemble performance with tuned thresholds...")
    print("per-emotion metrics:")

    per_emotion_metrics = {}
    for idx, emotion in enumerate(PLUTCHIK_EMOTIONS):
        threshold_value = tuned_thresholds.get(emotion, DECISION_THRESHOLD)
        print(
            f"  evaluating {emotion} (threshold={threshold_value:.3f})...",
            end=" ",
            flush=True,
        )
        emotion_results = combined.select(
            (col("labels")[idx]).alias("label"),
            (col("predictions")[idx]).alias("prediction"),
        )
        confusion = emotion_results.agg(
            spark_sum(
                when((col("label") == 1.0) & (col("prediction") == 1.0), 1).otherwise(0)
            ).alias("tp"),
            spark_sum(
                when((col("label") == 0.0) & (col("prediction") == 1.0), 1).otherwise(0)
            ).alias("fp"),
            spark_sum(
                when((col("label") == 1.0) & (col("prediction") == 0.0), 1).otherwise(0)
            ).alias("fn"),
            spark_sum(
                when((col("label") == 0.0) & (col("prediction") == 0.0), 1).otherwise(0)
            ).alias("tn"),
        ).collect()[0]

        tp = confusion["tp"] or 0
        fp = confusion["fp"] or 0
        fn = confusion["fn"] or 0
        tn = confusion["tn"] or 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        per_emotion_metrics[emotion] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        print(f"p={precision:.3f} r={recall:.3f} f1={f1:.3f}")

    print("\noverall multi-label metrics (micro-averaged):")
    total_tp = sum(metrics["tp"] for metrics in per_emotion_metrics.values())
    total_fp = sum(metrics["fp"] for metrics in per_emotion_metrics.values())
    total_fn = sum(metrics["fn"] for metrics in per_emotion_metrics.values())

    overall_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    overall_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    )
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    print(f"  precision: {overall_precision:.3f}")
    print(f"  recall: {overall_recall:.3f}")
    print(f"  f1-score: {overall_f1:.3f}")

    print("\nsample predictions:")
    try:
        samples = combined.select(
            "text", "labels", "predictions", "probabilities"
        ).take(5)
        for idx, sample in enumerate(samples, 1):
            true_labels = [
                PLUTCHIK_EMOTIONS[label_idx]
                for label_idx, value in enumerate(sample["labels"])
                if value == 1.0
            ]
            pred_labels = [
                PLUTCHIK_EMOTIONS[label_idx]
                for label_idx, value in enumerate(sample["predictions"])
                if value == 1.0
            ]
            probs = {
                PLUTCHIK_EMOTIONS[label_idx]: sample["probabilities"][label_idx]
                for label_idx in range(len(PLUTCHIK_EMOTIONS))
                if sample["probabilities"][label_idx] > 0.2
            }
            print(f"\n{idx}. {sample['text'][:80]}...")
            print(f"   true: {true_labels}")
            print(f"   pred: {pred_labels}")
            print(f"   probs: {dict(sorted(probs.items(), key=lambda item: -item[1]))}")
    except Exception as exc:
        print(f"  (could not display samples due to memory constraints: {exc})")

    combined.unpersist()

    print("\npipeline complete")
    print(f"trained ensemble with {len(algorithms_to_train)} algorithms:")
    for algorithm in algorithms_to_train:
        print(f"  - {algorithm}")

    spark.stop()


if __name__ == "__main__":
    run_pipeline()
