# -*- coding: utf-8 -*-
"""
Traditional ML pipeline for financial sentiment classification.
TF-IDF vectorization + LogisticRegression / SVM / MultinomialNB with GridSearchCV.
"""
import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from config import OUTPUT_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_splits():
    """Load preprocessed splits."""
    data_dir = os.path.join(OUTPUT_DIR, "data")
    with open(os.path.join(data_dir, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    return splits


def build_tfidf_vectorizer():
    """TF-IDF with unigram + bigram."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


def get_models():
    """Define models and their hyperparameter grids."""
    models = {
        "LogisticRegression": {
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
            "params": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "LinearSVC": {
            "clf": LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual="auto"),
            "params": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "MultinomialNB": {
            "clf": MultinomialNB(),
            "params": {
                "clf__alpha": [0.01, 0.1, 0.5, 1.0],
            },
        },
    }
    return models


def evaluate(y_true, y_pred):
    """Compute evaluation metrics."""
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":   recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":       f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class":   {str(i): f for i, f in enumerate(f1_per_class)},
    }


def train_and_evaluate():
    """Run full ML pipeline and return results."""
    logger.info("=" * 60)
    logger.info("TRADITIONAL ML PIPELINE")
    logger.info("=" * 60)

    splits = load_splits()
    X_train = splits["train"]["text"]
    y_train = splits["train"]["label"]
    X_val   = splits["val"]["text"]
    y_val   = splits["val"]["label"]
    X_test  = splits["test"]["text"]
    y_test  = splits["test"]["label"]

    # All labels for label encoder
    all_labels = list(set(y_train))
    label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
    idx2label = {i: l for l, i in label2idx.items()}

    y_train_idx = [label2idx[l] for l in y_train]
    y_val_idx   = [label2idx[l] for l in y_val]
    y_test_idx  = [label2idx[l] for l in y_test]

    models = get_models()
    results = {}
    best_model = None
    best_f1 = -1
    best_model_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, cfg in models.items():
        logger.info(f"\n--- Training {name} ---")
        start_time = time.time()

        vectorizer = build_tfidf_vectorizer()
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", cfg["clf"])])

        grid = GridSearchCV(
            pipeline,
            cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train_idx)

        train_time = time.time() - start_time
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV F1:  {grid.best_score_:.4f}")
        logger.info(f"  Train time:   {train_time:.2f}s")

        start_inf = time.time()
        y_pred = grid.predict(X_test)
        inf_time = time.time() - start_inf

        metrics = evaluate(y_test_idx, y_pred)
        metrics["train_time_s"] = train_time
        metrics["inference_latency_ms"] = (inf_time / len(X_test)) * 1000
        metrics["memory_usage_mb"] = len(pickle.dumps(grid.best_estimator_)) / (1024 * 1024)
        metrics["_y_true"] = [int(x) for x in y_test_idx]
        metrics["_y_pred"] = [int(x) for x in y_pred]
        results[name] = metrics

        print(f"\n{classification_report(y_test_idx, y_pred, target_names=sorted(idx2label.values()), zero_division=0)}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model = grid
            best_model_name = name

    logger.info(f"\nBest model: {best_model_name} (macro-F1={best_f1:.4f})")

    # Save best model + vectorizer
    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    # Save the pipeline (best_estimator_), not GridSearchCV
    best_pipeline = best_model.best_estimator_
    with open(os.path.join(models_dir, "ml_best_model.pkl"), "wb") as f:
        pickle.dump(best_pipeline, f)
    with open(os.path.join(models_dir, "ml_vectorizer.pkl"), "wb") as f:
        pickle.dump(best_pipeline.named_steps["tfidf"], f)
    with open(os.path.join(models_dir, "label_mapping.pkl"), "wb") as f:
        pickle.dump({"label2idx": label2idx, "idx2label": idx2label}, f)

    logger.info(f"Saved best model ({best_model_name}) to models/")

    # -------------------------------------------------------
    # SMOTE section: oversample training data and re-train
    # -------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("TRADITIONAL ML PIPELINE — WITH SMOTE")
    logger.info("=" * 60)

    # Build and fit TF-IDF on original training data first
    tfidf_sm = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_sm.fit_transform(X_train)
    X_test_tfidf  = tfidf_sm.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train_idx)
    logger.info(f"  SMOTE resampled: {X_train_sm.shape[0]} samples (original: {X_train_tfidf.shape[0]})")

    smote_results = {}

    # LogisticRegression + SMOTE
    for sm_name, sm_clf in [
        ("LogisticRegression+SMOTE", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")),
        ("LinearSVC+SMOTE",          LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual="auto")),
        ("MultinomialNB+SMOTE",      MultinomialNB()),
    ]:
        logger.info(f"\n--- Training {sm_name} ---")
        start_time = time.time()
        sm_clf.fit(X_train_sm, y_train_sm)
        train_time = time.time() - start_time

        start_inf = time.time()
        y_pred_sm = sm_clf.predict(X_test_tfidf)
        inf_time = time.time() - start_inf
        
        metrics_sm = evaluate(y_test_idx, y_pred_sm)
        metrics_sm["train_time_s"] = train_time
        metrics_sm["inference_latency_ms"] = (inf_time / len(X_test)) * 1000
        metrics_sm["memory_usage_mb"] = len(pickle.dumps(sm_clf)) / (1024 * 1024)
        metrics_sm["_y_true"] = [int(x) for x in y_test_idx]
        metrics_sm["_y_pred"] = [int(x) for x in y_pred_sm]
        smote_results[sm_name] = metrics_sm

        print(f"\n{classification_report(y_test_idx, y_pred_sm, target_names=sorted(idx2label.values()), zero_division=0)}")
        logger.info(f"  F1 (macro): {metrics_sm['f1_macro']:.4f}  |  Train time: {train_time:.2f}s")

    return results, smote_results, best_model, best_model_name


if __name__ == "__main__":
    results, smote_results, _, _ = train_and_evaluate()
    for name, m in results.items():
        print(f"\n{name}: {m}")
    for name, m in smote_results.items():
        print(f"\n{name}: {m}")
