# emotion classification with spark

simple script-based implementation of multi-label emotion classification.

## quick start

```bash
# install dependencies
pip install -r requirements.txt

# update lexicon paths in emotion_classification.py (lines 58-59)
NRC_EMOTION_PATH = '/path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
NRC_VAD_PATH = '/path/to/NRC-Emotion-Intensity-Lexicon-v1.txt'

# run script
python emotion_classification.py
```

## what it does

classifies text into 8 plutchik emotions:
- joy, sadness, anger, fear, surprise, disgust, trust, anticipation

features used:
- **tf-idf** (500 dimensions) - captures important words
- **nrc emotion counts** (8 dimensions) - counts emotion words
- **nrc vad scores** (3 dimensions) - valence, arousal, dominance

dataset:
- **goemotions** from huggingface (58k reddit comments, auto-downloaded)
- maps 28 fine-grained emotions → 8 plutchik emotions

model:
- **logistic regression** (one-vs-rest for multi-label classification)

## script structure

the code is now pure scripting with extensive inline comments:

1. **configuration** (lines 18-67) - all settings in one place
2. **initialize spark** (lines 70-77)
3. **load lexicons** (lines 80-132) - nrc emotion and vad lexicons
4. **broadcast lexicons** (lines 135-141) - send to spark workers
5. **load goemotions** (lines 144-161) - download and split data
6. **define udfs** (lines 164-249) - functions for feature extraction
7. **prepare training data** (lines 252-293) - extract features
8. **prepare test data** (lines 296-323) - apply same transformations
9. **train models** (lines 326-358) - one classifier per emotion
10. **make predictions** (lines 361-406) - predict on test set
11. **evaluate per-emotion** (lines 409-443) - metrics for each emotion
12. **evaluate overall** (lines 446-483) - micro-averaged metrics
13. **show samples** (lines 486-515) - display example predictions
14. **cleanup** (lines 518-520) - stop spark

## no classes, no functions

everything is inline scripting with extensive comments explaining each line.

## configuration options

edit these constants at the top:

```python
SAMPLE_SIZE = 0.5  # use 50% of dataset
TRAIN_SPLIT = 0.8  # 80% train, 20% test
TFIDF_FEATURES = 500  # tf-idf feature dimensions
MAX_ITERATIONS = 10  # training iterations
REGULARIZATION = 0.01  # l2 regularization
DECISION_THRESHOLD = 0.25  # prediction threshold
```

## data sources

### goemotions
- auto-downloaded from huggingface: `go_emotions` dataset
- 58k reddit comments with 28 emotion labels
- mapped to 8 plutchik emotions

### nrc lexicons
download from: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

files needed:
1. `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` - word→emotion mappings
2. `NRC-Emotion-Intensity-Lexicon-v1.txt` - word→vad scores

update paths in script (lines 58-59).

## dependencies

just 2 packages:
- `pyspark==3.5.0` - spark for distributed computing
- `datasets==2.14.6` - huggingface datasets library

no nltk needed - we use simple regex for tokenization.

## requirements

- python 3.8+
- java 8+ (for spark)
- ~2gb ram minimum
- internet connection (first run only, to download goemotions)

## expected performance

with default settings (50% data, threshold=0.25):

**per-emotion f1 scores:**
- positive emotions (joy, trust): 0.60-0.75
- negative emotions (anger, sadness, fear): 0.45-0.60
- neutral emotions (surprise, anticipation): 0.40-0.55

**overall metrics:**
- precision: ~0.55
- recall: ~0.50
- f1-score: ~0.52

## authors

devin dyson, madhur deep jain  
met cs 777 - big data analytics  
boston university  
november 5, 2025
