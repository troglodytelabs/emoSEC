"""Feature engineering components."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    NGram,
    RegexTokenizer,
    StopWordsRemover,
    VectorAssembler,
)
from pyspark.ml.linalg import Vector, VectorUDT, Vectors
from pyspark.ml.param import Param
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import RuntimeConfig
from .constants import DEFAULT_TEXT_COL, NRC_EMOTIONS, STOPWORD_FREE_COL, TOKEN_COL


class LexiconFeatureTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Create lexicon-based dense feature vectors from NRC Emotion Lexicon.

    Generates 33 features per document:
    - 10 raw emotion counts
    - 10 emotion ratios (count/token_count)
    - 10 binary flags (presence/absence)
    - Total matches
    - Total match ratio
    - Dominant emotion index

    Attributes:
        feature_names: List of generated feature names for reference.
    """

    def __init__(
        self,
        lexicon: Dict[str, Dict[str, int]] | None = None,
        inputCol: str | None = None,
        outputCol: str | None = None,
    ) -> None:
        """Initialize lexicon feature transformer.

        Args:
            lexicon: NRC emotion lexicon as nested dict {term: {emotion: 1}}.
            inputCol: Input column name containing token arrays.
            outputCol: Output column name for feature vectors.
        """
        super().__init__()
        self.lexicon_json = Param(
            self, "lexicon_json", "Serialized NRC emotion lexicon as JSON"
        )
        self.inputCol = Param(self, "inputCol", "Input column containing tokens")
        self.outputCol = Param(self, "outputCol", "Output column for features")
        self._lexicon_cache: Dict[str, Dict[str, int]] | None = None
        self._setDefault(
            lexicon_json="{}",
            inputCol=STOPWORD_FREE_COL,
            outputCol="lexicon_vector",
        )
        if inputCol is not None:
            self.setInputCol(inputCol)
        if outputCol is not None:
            self.setOutputCol(outputCol)
        if lexicon is not None:
            self.setLexicon(lexicon)
        self.feature_names: List[str] = []
        for emotion in NRC_EMOTIONS:
            self.feature_names.append(f"lex_count_{emotion}")
        for emotion in NRC_EMOTIONS:
            self.feature_names.append(f"lex_ratio_{emotion}")
        for emotion in NRC_EMOTIONS:
            self.feature_names.append(f"lex_flag_{emotion}")
        self.feature_names.extend(
            [
                "lex_total_matches",
                "lex_total_ratio",
                "lex_dominant_index",
            ]
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        lexicon = self.getLexicon()
        # Broadcast lexicon to all Spark executors for efficient lookup
        # This avoids shipping the dictionary with every UDF call
        broadcast = dataset.sparkSession.sparkContext.broadcast(lexicon)

        def compute_features(tokens: Sequence[str]) -> Vector:
            """Extract 33 lexicon-based features from token sequence.

            Features computed:
            - 10 raw counts: how many tokens match each NRC emotion
            - 10 ratios: counts normalized by total token count
            - 10 binary flags: 1 if any token matches emotion, 0 otherwise
            - 1 total matches: total tokens found in lexicon
            - 1 total ratio: proportion of tokens found in lexicon
            - 1 dominant index: which emotion has most matches (-1 if none)
            """
            token_count = len(tokens) if tokens else 0
            # Initialize emotion counters for 10 NRC emotions
            counts = {emotion: 0 for emotion in NRC_EMOTIONS}
            total_matches = 0

            # Count emotion associations for each token in the document
            for token in tokens or []:
                entry = broadcast.value.get(token.lower())
                if not entry:
                    continue  # Token not in lexicon, skip
                total_matches += 1
                # A token can be associated with multiple emotions
                for emotion in entry:
                    if emotion in counts:
                        counts[emotion] += 1

            # Avoid division by zero for empty documents
            denominator = float(token_count) if token_count else 1.0
            data: List[float] = []

            # Features 1-10: Raw counts per emotion
            for emotion in NRC_EMOTIONS:
                data.append(float(counts.get(emotion, 0)))

            # Features 11-20: Ratio of emotion words to total words
            for emotion in NRC_EMOTIONS:
                data.append(float(counts.get(emotion, 0)) / denominator)

            # Features 21-30: Binary presence/absence flags
            for emotion in NRC_EMOTIONS:
                data.append(1.0 if counts.get(emotion, 0) > 0 else 0.0)

            # Feature 31: Total lexicon matches
            data.append(float(total_matches))
            # Feature 32: Lexicon coverage (proportion of tokens in lexicon)
            data.append(float(total_matches) / denominator)

            # Feature 33: Dominant emotion index (which emotion is most frequent)
            if total_matches:
                dominant = max(NRC_EMOTIONS, key=lambda e: counts.get(e, 0))
                data.append(float(NRC_EMOTIONS.index(dominant)))
            else:
                data.append(-1.0)  # No emotions found

            return Vectors.dense(data)

        udf = F.udf(compute_features, VectorUDT())
        return dataset.withColumn(self.getOutputCol(), udf(F.col(self.getInputCol())))

    def setLexicon(
        self, lexicon: Dict[str, Dict[str, int]]
    ) -> "LexiconFeatureTransformer":
        """Set the NRC emotion lexicon and serialize for persistence.

        Args:
            lexicon: Emotion lexicon dictionary.

        Returns:
            Self for method chaining.
        """
        self._lexicon_cache = lexicon
        self._set(lexicon_json=json.dumps(lexicon))
        return self

    def getLexicon(self) -> Dict[str, Dict[str, int]]:
        """Get the NRC emotion lexicon, deserializing if needed.

        Returns:
            Emotion lexicon dictionary.
        """
        if self._lexicon_cache is not None:
            return self._lexicon_cache
        raw = self.getOrDefault(self.lexicon_json)
        if isinstance(raw, dict):
            self._lexicon_cache = {str(k): dict(v) for k, v in raw.items()}
        else:
            self._lexicon_cache = json.loads(raw) if raw else {}
        return self._lexicon_cache

    def setInputCol(self, value: str) -> "LexiconFeatureTransformer":
        self._set(inputCol=value)
        return self

    def getInputCol(self) -> str:
        return self.getOrDefault(self.inputCol)

    def setOutputCol(self, value: str) -> "LexiconFeatureTransformer":
        self._set(outputCol=value)
        return self

    def getOutputCol(self) -> str:
        return self.getOrDefault(self.outputCol)


class LinguisticFeatureTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Create custom linguistic features from text and tokens.

    Generates 12 linguistic features:
    - word_count, char_count, avg_word_length
    - exclamation_count, question_count, multi_punct_count
    - all_caps_token_count, title_case_ratio, uppercase_char_ratio
    - special_char_count, punctuation_density, digit_count

    Attributes:
        feature_names: List of generated feature names.
    """

    def __init__(
        self,
        textCol: str | None = None,
        tokensCol: str | None = None,
        outputCol: str | None = None,
    ) -> None:
        """Initialize linguistic feature transformer.

        Args:
            textCol: Column name containing raw text.
            tokensCol: Column name containing tokenized text.
            outputCol: Output column name for feature vectors.
        """
        super().__init__()
        self.textCol = Param(self, "textCol", "Column containing raw text")
        self.tokensCol = Param(self, "tokensCol", "Column containing tokenized text")
        self.outputCol = Param(
            self, "outputCol", "Output column for linguistic features"
        )
        self._setDefault(
            textCol=DEFAULT_TEXT_COL,
            tokensCol=STOPWORD_FREE_COL,
            outputCol="linguistic_vector",
        )
        if textCol is not None:
            self.setTextCol(textCol)
        if tokensCol is not None:
            self.setTokensCol(tokensCol)
        if outputCol is not None:
            self.setOutputCol(outputCol)
        self.feature_names = [
            "word_count",
            "char_count",
            "avg_word_length",
            "exclamation_count",
            "question_count",
            "multi_punct_count",
            "all_caps_token_count",
            "title_case_ratio",
            "uppercase_char_ratio",
            "special_char_count",
            "punctuation_density",
            "digit_count",
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Transform dataset by computing linguistic features.

        Args:
            dataset: Input DataFrame with text and token columns.

        Returns:
            DataFrame with added linguistic feature vector column.
        """

        def compute_features(text: str, tokens: Sequence[str]) -> Vector:
            text = text or ""
            tokens = tokens or []
            word_count = float(len(tokens))
            char_count = float(len(text))
            avg_word_length = (
                sum(len(tok) for tok in tokens) / word_count if word_count else 0.0
            )
            exclamation_count = text.count("!")
            question_count = text.count("?")
            multi_punct_count = sum(
                1
                for seg in text.split()
                if any(p in seg for p in ("!!", "??", "?!", "!?"))
            )
            all_caps_token_count = sum(
                1 for tok in tokens if len(tok) > 1 and tok.isupper()
            )
            title_case_tokens = sum(1 for tok in tokens if tok.istitle())
            title_case_ratio = title_case_tokens / word_count if word_count else 0.0
            uppercase_chars = sum(1 for ch in text if ch.isupper())
            uppercase_char_ratio = uppercase_chars / char_count if char_count else 0.0
            special_char_count = sum(
                1 for ch in text if not ch.isalnum() and not ch.isspace()
            )
            punctuation_total = exclamation_count + question_count
            punctuation_density = punctuation_total / word_count if word_count else 0.0
            digit_count = sum(1 for ch in text if ch.isdigit())

            values = [
                word_count,
                char_count,
                avg_word_length,
                float(exclamation_count),
                float(question_count),
                float(multi_punct_count),
                float(all_caps_token_count),
                title_case_ratio,
                uppercase_char_ratio,
                float(special_char_count),
                punctuation_density,
                float(digit_count),
            ]
            return Vectors.dense(values)

        udf = F.udf(compute_features, VectorUDT())
        return dataset.withColumn(
            self.getOutputCol(),
            udf(F.col(self.getTextCol()), F.col(self.getTokensCol())),
        )

    def setTextCol(self, value: str) -> "LinguisticFeatureTransformer":
        self._set(textCol=value)
        return self

    def getTextCol(self) -> str:
        return self.getOrDefault(self.textCol)

    def setTokensCol(self, value: str) -> "LinguisticFeatureTransformer":
        self._set(tokensCol=value)
        return self

    def getTokensCol(self) -> str:
        return self.getOrDefault(self.tokensCol)

    def setOutputCol(self, value: str) -> "LinguisticFeatureTransformer":
        self._set(outputCol=value)
        return self

    def getOutputCol(self) -> str:
        return self.getOrDefault(self.outputCol)


class VADFeatureTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Derive valence/arousal/dominance statistics per document.

    Computes statistics from NRC VAD lexicon:
    - Mean, std, range for valence, arousal, and dominance
    - Token coverage ratio

    Total: 10 features per document.

    Attributes:
        feature_names: List of generated feature names.
    """

    def __init__(
        self,
        vad_lexicon: Dict[str, Tuple[float, float, float]] | None = None,
        inputCol: str | None = None,
        outputCol: str | None = None,
    ) -> None:
        """Initialize VAD feature transformer.

        Args:
            vad_lexicon: NRC VAD lexicon as {term: (valence, arousal, dominance)}.
            inputCol: Input column name containing token arrays.
            outputCol: Output column name for feature vectors.
        """
        super().__init__()
        self.vad_json = Param(self, "vad_json", "Serialized NRC VAD lexicon as JSON")
        self.inputCol = Param(self, "inputCol", "Input column containing tokens")
        self.outputCol = Param(self, "outputCol", "Output column for VAD features")
        self._vad_cache: Dict[str, Tuple[float, float, float]] | None = None
        self._setDefault(
            vad_json="{}",
            inputCol=STOPWORD_FREE_COL,
            outputCol="vad_vector",
        )
        if inputCol is not None:
            self.setInputCol(inputCol)
        if outputCol is not None:
            self.setOutputCol(outputCol)
        if vad_lexicon is not None:
            self.setVadLexicon(vad_lexicon)
        self.feature_names = [
            "vad_valence_mean",
            "vad_valence_std",
            "vad_valence_range",
            "vad_arousal_mean",
            "vad_arousal_std",
            "vad_arousal_range",
            "vad_dominance_mean",
            "vad_dominance_std",
            "vad_dominance_range",
            "vad_token_coverage",
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Transform dataset by computing VAD statistics.

        Args:
            dataset: Input DataFrame with token column.

        Returns:
            DataFrame with added VAD feature vector column.
        """
        vad_map = self.getVadLexicon()
        broadcast = dataset.sparkSession.sparkContext.broadcast(vad_map)

        def compute(tokens: Sequence[str]) -> Vector:
            tokens = tokens or []
            matches = [
                broadcast.value[token.lower()]
                for token in tokens
                if token and token.lower() in broadcast.value
            ]
            total_tokens = float(len(tokens)) if tokens else 0.0
            coverage = len(matches) / total_tokens if total_tokens else 0.0

            def stats(index: int) -> Tuple[float, float, float]:
                if not matches:
                    return 0.0, 0.0, 0.0
                values = [triple[index] for triple in matches]
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((val - mean) ** 2 for val in values) / (
                        len(values) - 1
                    )
                else:
                    variance = 0.0
                spread = max(values) - min(values)
                return mean, variance**0.5, spread

            valence = stats(0)
            arousal = stats(1)
            dominance = stats(2)

            values = [
                valence[0],
                valence[1],
                valence[2],
                arousal[0],
                arousal[1],
                arousal[2],
                dominance[0],
                dominance[1],
                dominance[2],
                coverage,
            ]
            return Vectors.dense(values)

        udf = F.udf(compute, VectorUDT())
        return dataset.withColumn(self.getOutputCol(), udf(F.col(self.getInputCol())))

    def setVadLexicon(
        self, lexicon: Dict[str, Tuple[float, float, float]]
    ) -> "VADFeatureTransformer":
        """Set the NRC VAD lexicon and serialize for persistence.

        Args:
            lexicon: VAD lexicon dictionary.

        Returns:
            Self for method chaining.
        """
        self._vad_cache = lexicon
        serializable = {term: list(values) for term, values in lexicon.items()}
        self._set(vad_json=json.dumps(serializable))
        return self

    def getVadLexicon(self) -> Dict[str, Tuple[float, float, float]]:
        """Get the NRC VAD lexicon, deserializing if needed.

        Returns:
            VAD lexicon dictionary.
        """
        if self._vad_cache is not None:
            return self._vad_cache
        raw = self.getOrDefault(self.vad_json)
        if isinstance(raw, dict):
            data = {str(term): tuple(values) for term, values in raw.items()}
        else:
            loaded = json.loads(raw) if raw else {}
            data = {str(term): tuple(values) for term, values in loaded.items()}
        self._vad_cache = data
        return self._vad_cache

    def setInputCol(self, value: str) -> "VADFeatureTransformer":
        self._set(inputCol=value)
        return self

    def getInputCol(self) -> str:
        return self.getOrDefault(self.inputCol)

    def setOutputCol(self, value: str) -> "VADFeatureTransformer":
        self._set(outputCol=value)
        return self

    def getOutputCol(self) -> str:
        return self.getOrDefault(self.outputCol)


class FeatureBuilder:
    """Build feature pipeline combining TF-IDF, lexicon, and linguistic cues.

    Constructs a complete Spark ML Pipeline with:
    - Text tokenization and stopword removal
    - N-gram TF-IDF features for configured orders (1, 2, 3)
    - NRC emotion lexicon features
    - NRC VAD affective features
    - Custom linguistic features
    - Vector assembly combining all feature types

    Attributes:
        config: Runtime configuration.
        lexicon: NRC emotion lexicon.
        vad_lexicon: NRC VAD lexicon.
        pipeline_model: Fitted PipelineModel after fit() is called.
        vector_columns: List of vector column names being assembled.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        lexicon: Dict[str, Dict[str, int]],
        vad_lexicon: Dict[str, Tuple[float, float, float]],
    ) -> None:
        """Initialize feature builder with configuration and lexicons.

        Args:
            config: Runtime configuration with feature parameters.
            lexicon: NRC emotion lexicon.
            vad_lexicon: NRC VAD lexicon.
        """
        self.config = config
        self.lexicon = lexicon
        self.vad_lexicon = vad_lexicon
        self.pipeline_model = None
        self.vector_columns: List[str] = []

    def build(self) -> Pipeline:
        """Build the complete feature engineering pipeline.

        Returns:
            Spark ML Pipeline with all feature extraction stages.
        """
        # ============================================================
        # STAGE 1: TOKENIZATION
        # ============================================================
        # Convert raw text to array of lowercase word tokens
        # Pattern "\\w+" matches word characters (letters, digits, underscore)
        tokenizer = RegexTokenizer(
            inputCol=DEFAULT_TEXT_COL,
            outputCol=TOKEN_COL,
            pattern="\\w+",
            gaps=False,  # Pattern defines tokens, not delimiters
            toLowercase=True,
        )

        # ============================================================
        # STAGE 2: STOPWORD REMOVAL
        # ============================================================
        # Remove common words that don't carry emotion signal
        # Examples: "the", "is", "at", "which", "on"
        stopwords = StopWordsRemover(inputCol=TOKEN_COL, outputCol=STOPWORD_FREE_COL)

        stages = [tokenizer, stopwords]
        vector_columns: List[str] = []

        # ============================================================
        # STAGES 3-N: TF-IDF FOR MULTIPLE N-GRAM ORDERS
        # ============================================================
        # Generate TF-IDF features for unigrams, bigrams, trigrams
        # TF-IDF = Term Frequency × Inverse Document Frequency
        # - TF: How often a term appears in THIS document
        # - IDF: How rare the term is ACROSS ALL documents
        # Rare terms that appear frequently in a doc get high scores
        for n in self.config.ngram_orders:
            if n == 1:
                # For unigrams, use the stopword-filtered tokens directly
                input_col = STOPWORD_FREE_COL
            else:
                # For n>1, generate n-grams from stopword-filtered tokens
                # Example: ["happy", "birthday", "to"] → ["happy birthday", "birthday to"]
                input_col = f"{n}gram"
                stages.append(
                    NGram(n=n, inputCol=STOPWORD_FREE_COL, outputCol=input_col)
                )

            # CountVectorizer: Learn vocabulary and count term frequencies
            # - vocabSize: maximum vocabulary size (most frequent terms)
            # - minDF: ignore terms appearing in fewer than minDF documents
            tf_col = f"tf_{n}gram"
            idf_col = f"tfidf_{n}gram"
            vectorizer = CountVectorizer(
                inputCol=input_col,
                outputCol=tf_col,
                vocabSize=self.config.max_vocab,
                minDF=self.config.min_df,
            )

            # IDF: Learn inverse document frequency weights
            # Terms appearing in many documents get downweighted
            idf = IDF(
                inputCol=tf_col,
                outputCol=idf_col,
                minDocFreq=self.config.idf_min_doc_freq,
            )
            stages.extend([vectorizer, idf])
            vector_columns.append(idf_col)

        # ============================================================
        # STAGE N+1: NRC EMOTION LEXICON FEATURES
        # ============================================================
        # Add 33 features based on emotion word counts from NRC lexicon
        # Captures explicit emotion words in the text
        lexicon_stage = LexiconFeatureTransformer(
            lexicon=self.lexicon,
            inputCol=STOPWORD_FREE_COL,
            outputCol="lexicon_vector",
        )
        vector_columns.append("lexicon_vector")

        # ============================================================
        # STAGE N+2: NRC VAD (VALENCE-AROUSAL-DOMINANCE) FEATURES
        # ============================================================
        # Add 10 features based on affective dimensions:
        # - Valence: positive vs negative emotion
        # - Arousal: calm vs excited state
        # - Dominance: submissive vs dominant feeling
        # Provides continuous emotion intensity information
        vad_stage = VADFeatureTransformer(
            vad_lexicon=self.vad_lexicon,
            inputCol=STOPWORD_FREE_COL,
            outputCol="vad_vector",
        )
        vector_columns.append("vad_vector")

        # ============================================================
        # STAGE N+3: CUSTOM LINGUISTIC FEATURES
        # ============================================================
        # Add 12 features based on writing style and punctuation:
        # - Word/character counts, average word length
        # - Exclamation marks, question marks (emotional intensity)
        # - ALL CAPS words (shouting, emphasis)
        # - Title case, uppercase ratios (formality indicators)
        # - Special characters, digits
        linguistic_stage = LinguisticFeatureTransformer(
            textCol=DEFAULT_TEXT_COL,
            tokensCol=STOPWORD_FREE_COL,
            outputCol="linguistic_vector",
        )
        vector_columns.append("linguistic_vector")

        # ============================================================
        # STAGE N+4: VECTOR ASSEMBLY
        # ============================================================
        # Combine all feature vectors into a single feature vector
        # Final dimensionality: ~20K+ features depending on vocab size
        # - TF-IDF unigrams: ~20K features
        # - TF-IDF bigrams: ~20K features
        # - TF-IDF trigrams: ~20K features
        # - Lexicon features: 33 features
        # - VAD features: 10 features
        # - Linguistic features: 12 features
        assembler = VectorAssembler(
            inputCols=vector_columns,
            outputCol="features",
        )

        stages.extend([lexicon_stage, vad_stage, linguistic_stage, assembler])
        self.vector_columns = vector_columns
        return Pipeline(stages=stages)

    def fit(self, df: DataFrame) -> FeatureBuilder:
        """Fit the feature pipeline on training data.

        Args:
            df: Training DataFrame with text column.

        Returns:
            Self for method chaining.
        """
        pipeline = self.build()
        self.pipeline_model = pipeline.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform dataset using fitted pipeline.

        Args:
            df: Input DataFrame with text column.

        Returns:
            Transformed DataFrame with features column.

        Raises:
            RuntimeError: If transform called before fit.
        """
        if self.pipeline_model is None:
            raise RuntimeError("FeatureBuilder must be fit before calling transform().")
        transformed = self.pipeline_model.transform(df)
        return transformed
