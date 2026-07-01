# -*- coding: utf-8 -*-
import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight

from config import OUTPUT_DIR, RANDOM_STATE
from transformer_pipeline import (
    DEVICE, BATCH_SIZE, MAX_LENGTH, EPOCHS, LR, WARMUP_RATIO, FREEZE_LAYERS,
    load_splits, freeze_bottom_layers, compute_metrics,
    _build_dataloaders, _training_loop, evaluate, predict
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ADDITIONAL_MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-small",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
]

def run_model_pipeline(model_name):
    logger.info("=" * 60)
    logger.info(f"TRAINING ADDITIONAL MODEL: {model_name}")
    logger.info("=" * 60)

    splits = load_splits()
    X_train = splits["train"]["text"]
    y_train = splits["train"]["label"]
    X_val   = splits["val"]["text"]
    y_val   = splits["val"]["label"]
    X_test  = splits["test"]["text"]
    y_test  = splits["test"]["label"]

    all_labels = sorted(set(y_train))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    num_labels = len(all_labels)

    y_train_idx = [label2idx[l] for l in y_train]
    y_val_idx   = [label2idx[l] for l in y_val]
    y_test_idx  = [label2idx[l] for l in y_test]

    logger.info(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ignore mismatched sizes for pre-trained heads like twitter-roberta
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    freeze_bottom_layers(model, FREEZE_LAYERS)

    train_loader, val_loader, test_loader = _build_dataloaders(
        tokenizer, X_train, y_train_idx, X_val, y_val_idx, X_test, y_test_idx
    )

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    best_state, train_time = _training_loop(
        model, train_loader, val_loader, DEVICE,
        epochs=EPOCHS, lr=LR, warmup_ratio=WARMUP_RATIO, class_weights=class_weights
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # Test evaluation
    test_preds, test_labels, _ = evaluate(model, test_loader, DEVICE, class_weights)
    metrics = compute_metrics(test_labels, test_preds)
    metrics["train_time_s"] = train_time

    start_inf = time.time()
    _ = predict(model, test_loader, DEVICE)
    inf_time = time.time() - start_inf
    metrics["inference_latency_ms"] = (inf_time / len(X_test)) * 1000

    if torch.cuda.is_available():
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        metrics["peak_gpu_memory_mb"] = 0.0

    logger.info(f"\n{model_name} Test metrics: {metrics}")

    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    clean_name = model_name.replace("/", "_")
    torch.save(model.state_dict(), os.path.join(models_dir, f"{clean_name}_model.pt"))
    tokenizer.save_pretrained(os.path.join(models_dir, f"{clean_name}_tokenizer"))
    with open(os.path.join(models_dir, f"{clean_name}_label_mapping.pkl"), "wb") as f:
        pickle.dump({"label2idx": label2idx, "idx2label": idx2label, "num_labels": num_labels}, f)

    return metrics, model_name, test_preds, test_labels

def run_all_additional_models():
    all_results = {}
    preds_dict = {}
    labels_dict = {}
    for model_name in ADDITIONAL_MODELS:
        try:
            metrics, _, preds, labels = run_model_pipeline(model_name)
            clean_name = model_name.split("/")[-1]
            all_results[clean_name] = metrics
            preds_dict[clean_name] = preds
            labels_dict[clean_name] = labels
        except Exception as e:
            logger.error(f"Failed pipeline for {model_name}: {e}")
    return all_results, preds_dict, labels_dict

if __name__ == "__main__":
    results, _, _ = run_all_additional_models()
    print("Additional models results:", results)
