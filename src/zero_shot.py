# -*- coding: utf-8 -*-
import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import pipeline

from config import OUTPUT_DIR, RANDOM_STATE
from transformer_pipeline import load_splits, compute_metrics, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_zero_shot():
    logger.info("=" * 60)
    logger.info("ZERO-SHOT EVALUATION (BART-LARGE-MNLI)")
    logger.info("=" * 60)

    model_name = "facebook/bart-large-mnli"

    splits_meta_path = os.path.join(OUTPUT_DIR, "data", "splits_meta.pkl")
    splits_path = os.path.join(OUTPUT_DIR, "data", "splits.pkl")
    
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)
        
    with open(splits_meta_path, "rb") as f:
        splits_meta = pickle.load(f)

    # We only need to evaluate on test set
    X_test  = splits_meta["test"]["text_cleaned"] # Use original cleaned text, not lemmatized for zero-shot
    y_test  = splits["test"]["label"]

    all_labels = sorted(set(y_test))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    y_test_idx = [label2idx[l] for l in y_test]

    logger.info(f"Loading zero-shot pipeline for {model_name}...")
    # Device parsing for pipeline
    device_id = 0 if DEVICE.type == "cuda" else -1
    classifier = pipeline("zero-shot-classification", model=model_name, device=device_id)

    candidate_labels = ["positive", "negative", "neutral"]

    start_inf = time.time()
    
    # Run in batches for speed
    batch_size = 16
    test_preds_str = []
    
    logger.info(f"Running inference on {len(X_test)} samples...")
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i+batch_size]
        results = classifier(batch_texts, candidate_labels)
        for res in results:
            # First label is the highest scoring one
            test_preds_str.append(res["labels"][0])
            
        if (i + batch_size) % 160 == 0:
            logger.info(f"  Processed {i + batch_size}/{len(X_test)}")

    inf_time = time.time() - start_inf
    
    test_preds_idx = [label2idx[l] for l in test_preds_str]

    metrics = compute_metrics(y_test_idx, test_preds_idx)
    metrics["train_time_s"] = 0.0
    metrics["inference_latency_ms"] = (inf_time / len(X_test)) * 1000

    if torch.cuda.is_available():
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        metrics["peak_gpu_memory_mb"] = 0.0

    logger.info(f"\nZero-Shot Test metrics: {metrics}")

    return metrics, test_preds_idx, y_test_idx

if __name__ == "__main__":
    run_zero_shot()
