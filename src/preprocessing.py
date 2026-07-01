# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for financial sentiment analysis.
Loads CSV, cleans text, lemmatizes, and performs stratified train/val/test split.
"""
import os
import re
import pickle
import json
import logging

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from config import DATA_PATH, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download NLTK resources
NLTK_RESOURCES = ["punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger",
                  "averaged_perceptron_tagger_eng", "omw-1.4", "stopwords"]
for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))

# Keep direction/magnitude words important for finance
financial_keeps = {"up", "down", "above", "below", "over", "under", "more", "most", "less", "few", "too", "very", "not", "no"}
stop_words = stop_words - financial_keeps


def clean_text(text):
    """Clean text: remove special chars, lowercase, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)  # keep alphanumeric + apostrophe + hyphen
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words)


def load_data(path):
    """Load data from CSV."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Sentiment distribution:\n{df['Sentiment'].value_counts()}")
    return df


def preprocess(df):
    """Apply text cleaning and lemmatization."""
    logger.info("Cleaning text...")
    df = df.copy()
    df["cleaned"] = df["Sentence"].apply(clean_text)
    logger.info("Lemmatizing...")
    df["lemmatized"] = df["cleaned"].apply(lemmatize)
    return df


def split_data(df):
    """Perform stratified train/val/test split (70/15/15)."""
    logger.info("Splitting data into train/val/test (70/15/15)...")
    label = "Sentiment"
    X = df[["Sentence", "cleaned", "lemmatized"]]
    y = df[label]

    # 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # 50% of temp = 15% each for val and test
    val_fraction = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_fraction,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Assemble splits
    splits = {
        "train": {"text": X_train["lemmatized"].tolist(), "label": y_train.tolist()},
        "val":   {"text": X_val["lemmatized"].tolist(),   "label": y_val.tolist()},
        "test":  {"text": X_test["lemmatized"].tolist(),  "label": y_test.tolist()},
    }

    # Also keep original (non-lemmatized) cleaned text for reference
    splits_meta = {
        "train": {"text_cleaned": X_train["cleaned"].tolist()},
        "val":   {"text_cleaned": X_val["cleaned"].tolist()},
        "test":  {"text_cleaned": X_test["cleaned"].tolist()},
    }

    return splits, splits_meta, y_train, y_val, y_test


def save_splits(splits, splits_meta, output_dir):
    """Save splits as pickle and JSON."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Pickle
    with open(os.path.join(data_dir, "splits.pkl"), "wb") as f:
        pickle.dump(splits, f)
    with open(os.path.join(data_dir, "splits_meta.pkl"), "wb") as f:
        pickle.dump(splits_meta, f)

    # JSON
    def json_serializable(obj):
        return {k: {kk: vv for kk, vv in v.items()} for k, v in obj.items()}

    with open(os.path.join(data_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(json_serializable(splits), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved splits to {data_dir}/")


def run():
    """Main preprocessing entry point."""
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df = load_data(DATA_PATH)
    df = preprocess(df)
    splits, splits_meta, y_train, y_val, y_test = split_data(df)
    save_splits(splits, splits_meta, OUTPUT_DIR)

    logger.info("Preprocessing complete.")
    return splits, splits_meta, y_train, y_val, y_test


if __name__ == "__main__":
    run()
