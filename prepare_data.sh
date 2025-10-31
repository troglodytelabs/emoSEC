#!/bin/bash

# Create data directory
mkdir -p data

# Check if GoEmotions data exists
if [ ! -f "data/goemotions_1.csv" ]; then
    echo "Downloading GoEmotions dataset..."
    # Download from Google Research GitHub
    wget -O data/goemotions_1.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
    wget -O data/goemotions_2.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
    wget -O data/goemotions_3.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
fi

# Check if NRC Lexicon exists
if [ ! -f "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt" ]; then
    echo "Please download NRC Emotion Lexicon from:"
    echo "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
    echo "And place it in data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
fi

# Check if NRC VAD Lexicon exists
if [ ! -f "data/NRC-VAD-Lexicon-v2.1.txt" ]; then
    echo "Please download NRC Valence-Arousal-Dominance (VAD) Lexicon from:"
    echo "https://saifmohammad.com/WebPages/nrc-vad.html"
    echo "Extract the tab-separated file and save it as data/NRC-VAD-Lexicon-v2.1.txt"
fi

echo "Data preparation complete!"
