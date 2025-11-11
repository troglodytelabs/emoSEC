"""Demo for the trained emotion ensemble."""

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDFModel, Tokenizer
from pyspark.ml.classification import (
    LinearSVCModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestClassificationModel,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import array, col, when, lit

from config import (
    DECISION_THRESHOLD,
    MODEL_BASE_PATH,
    NRC_EMOTION_PATH,
    NRC_VAD_PATH,
    PLUTCHIK_EMOTIONS,
    TFIDF_FEATURES,
)
from feature_udfs import (
    get_combine_udf,
    get_extract_linguistic_udf,
    get_extract_nrc_udf,
    get_extract_vad_udf,
)
from lexicon_utils import (
    broadcast_lexicons,
    load_nrc_emotion_lexicon,
    load_nrc_vad_lexicon,
)

MODEL_LOADERS = {
    "logistic_regression": LogisticRegressionModel,
    "svm": LinearSVCModel,
    "naive_bayes": NaiveBayesModel,
    "random_forest": RandomForestClassificationModel,
}


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    raise FileNotFoundError(
        f"metadata.json not found in {model_dir}. Run the training script first."
    )


def load_algorithms(metadata: dict) -> list[str]:
    algorithms = metadata.get("algorithms", [])
    if not algorithms:
        raise RuntimeError("No algorithms listed in metadata; cannot run inference.")
    return algorithms


def build_feature_frame(spark: SparkSession, text: str):
    print("loading lexicons...")
    nrc_emotion = load_nrc_emotion_lexicon(NRC_EMOTION_PATH, PLUTCHIK_EMOTIONS)
    nrc_vad = load_nrc_vad_lexicon(NRC_VAD_PATH)
    nrc_bc, vad_bc = broadcast_lexicons(spark, nrc_emotion, nrc_vad)

    extract_nrc_udf = get_extract_nrc_udf(nrc_bc)
    extract_vad_udf = get_extract_vad_udf(vad_bc)
    extract_linguistic_udf = get_extract_linguistic_udf()
    combine_udf = get_combine_udf()

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing_tf = HashingTF(
        inputCol="words", outputCol="raw_features", numFeatures=TFIDF_FEATURES
    )

    df = spark.createDataFrame([(0, text)], ["row_id", "text"])
    df = tokenizer.transform(df)
    df = df.withColumn("nrc_features", extract_nrc_udf(col("text")))
    df = df.withColumn("vad_features", extract_vad_udf(col("text")))
    df = df.withColumn(
        "linguistic_features", extract_linguistic_udf(col("text"), col("words"))
    )
    df = hashing_tf.transform(df)
    return df, combine_udf


def apply_feature_vector(df, combine_udf, idf_model: IDFModel):
    df = idf_model.transform(df)
    df = df.withColumn(
        "features",
        combine_udf(
            col("tfidf_features"),
            col("nrc_features"),
            col("vad_features"),
            col("linguistic_features"),
        ),
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Plutchik emotion predictions for input text."
    )
    parser.add_argument("--text", nargs="+", help="Text to classify.")
    args = parser.parse_args()

    input_text = " ".join(args.text).strip()
    if not input_text:
        parser.error("Input text must not be empty.")

    model_dir = Path(MODEL_BASE_PATH)
    metadata = load_metadata(model_dir)
    algorithms = load_algorithms(metadata)
    emotions = metadata.get("emotions", PLUTCHIK_EMOTIONS)
    threshold = metadata.get("decision_threshold", DECISION_THRESHOLD)
    trained_tfidf = metadata.get("tfidf_features", TFIDF_FEATURES)
    if trained_tfidf != TFIDF_FEATURES:
        raise RuntimeError(
            "Configured TFIDF feature count does not match saved model metadata."
        )
    thresholds = metadata.get("thresholds")
    if isinstance(thresholds, dict):
        threshold_lookup = {
            emotion: float(thresholds.get(emotion, threshold)) for emotion in emotions
        }
    else:
        threshold_lookup = {emotion: float(threshold) for emotion in emotions}

    idf_path = model_dir / "idf_model"
    if not idf_path.exists():
        raise FileNotFoundError(f"IDF model missing at {idf_path}.")

    spark = (
        SparkSession.builder.appName("emotion-classification-demo")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        idf_model = IDFModel.load(str(idf_path))
        df, combine_udf = build_feature_frame(spark, input_text)
        df = apply_feature_vector(df, combine_udf, idf_model)

        algorithm_predictions: dict[str, list] = {algo: [] for algo in algorithms}
        for algorithm in algorithms:
            loader = MODEL_LOADERS.get(algorithm)
            if loader is None:
                raise KeyError(
                    f"No model loader registered for algorithm '{algorithm}'."
                )
            for idx, emotion in enumerate(emotions):
                model_path = model_dir / algorithm / emotion
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Missing model checkpoint for {algorithm}/{emotion}."
                    )
                model = loader.load(str(model_path))
                predictions = model.transform(df)
                predictions = predictions.withColumn(
                    f"pred_{algorithm}_{emotion}", col("prediction")
                )
                if algorithm == "svm":
                    prob_col = col(f"pred_{algorithm}_{emotion}")
                else:
                    prob_col = vector_to_array(col("probability"))[1]
                predictions = predictions.withColumn(
                    f"prob_{algorithm}_{emotion}", prob_col
                )
                algorithm_predictions[algorithm].append(
                    predictions.select(
                        "row_id",
                        f"pred_{algorithm}_{emotion}",
                        f"prob_{algorithm}_{emotion}",
                    )
                )

        combined = df.select("row_id", "text")
        for algorithm in algorithms:
            for pred_df in algorithm_predictions[algorithm]:
                combined = combined.join(pred_df, on="row_id", how="left")

        for emotion in emotions:
            prob_cols = [col(f"prob_{algo}_{emotion}") for algo in algorithms]
            ensemble_prob = sum(prob_cols) / len(algorithms)
            combined = combined.withColumn(f"ensemble_prob_{emotion}", ensemble_prob)

        ensemble_prob_cols = [col(f"ensemble_prob_{emotion}") for emotion in emotions]
        combined = combined.withColumn("probabilities", array(*ensemble_prob_cols))

        prediction_cols = [
            when(
                col(f"ensemble_prob_{emotion}") >= lit(threshold_lookup[emotion]),
                1.0,
            ).otherwise(0.0)
            for emotion in emotions
        ]
        combined = combined.withColumn("predictions", array(*prediction_cols))

        result = combined.select("text", "probabilities", "predictions").first()
        probs = list(result["probabilities"])
        preds = list(result["predictions"])

        scored = sorted(zip(emotions, probs), key=lambda item: item[1], reverse=True)
        top = [f"{emotion}: {score:.3f}" for emotion, score in scored if score > 0]
        active = [emotions[idx] for idx, value in enumerate(preds) if value == 1.0]

        print("\nInput:")
        print(f"  {result['text']}")
        print("\nPredicted emotions:")
        print(f"  {active if active else 'None above threshold'}")
        if top:
            print("\nEnsemble probabilities:")
            for line in top:
                print(f"  {line}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
