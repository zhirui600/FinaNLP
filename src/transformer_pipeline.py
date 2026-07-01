# -*- coding: utf-8 -*-
"""
Transformer pipeline for financial sentiment classification.
Uses FinBERT (yyang42/finbert-tone) via HuggingFace transformers.
Freezes bottom N encoder layers, trains with AdamW, early stopping.

Adds:
  - run_bertbase_pipeline(): BERT-base-uncased comparison
  - run_quantized_pipeline(): 4-bit dynamic quantization cost-quality analysis
"""
import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import OUTPUT_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE    = 16
MAX_LENGTH    = 128
EPOCHS        = 3
LR            = 2e-5
WARMUP_RATIO  = 0.1
FREEZE_LAYERS = 3   # freeze first N encoder layers

FINBERT_MODELS = [
    "yyang42/finbert-tone",
    "ProsusAI/finbert",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]
FALLBACK_MODEL = "bert-base-uncased"


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts  = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_splits():
    """Load preprocessed splits."""
    data_dir = os.path.join(OUTPUT_DIR, "data")
    with open(os.path.join(data_dir, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    return splits


def try_load_model_and_tokenizer():
    """Try FinBERT models; fall back to bert-base-uncased."""
    for model_name in FINBERT_MODELS:
        try:
            logger.info(f"Trying model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Successfully loaded {model_name}")
            return model_name, tokenizer, model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue

    logger.info(f"Falling back to {FALLBACK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FALLBACK_MODEL, num_labels=3
    )
    return FALLBACK_MODEL, tokenizer, model


def freeze_bottom_layers(model, n_freeze):
    """Freeze first N encoder layers of the transformer model."""
    if hasattr(model, "bert"):
        encoder = model.bert
    elif hasattr(model, "distilbert"):
        encoder = model.distilbert
    elif hasattr(model, "electra"):
        encoder = model.electra
    else:
        encoder = getattr(model, "encoder", None) or getattr(model, "transformer", None)

    if encoder is None:
        logger.warning("Could not identify encoder to freeze layers.")
        return

    if hasattr(encoder, "embeddings"):
        # Freeze embeddings
        for param in encoder.embeddings.parameters():
            param.requires_grad = False

    frozen = 0
    if hasattr(encoder, "encoder"):
        for layer in encoder.encoder.layer[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
            frozen += 1
    elif hasattr(encoder, "layer"):
        for layer in encoder.layer[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
            frozen += 1

    logger.info(f"Froze {frozen} encoder layers.")


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Fix for DeBERTa/FP16: ensure logits are float32 for loss calculation
        logits = outputs.logits.float()
        loss    = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, class_weights=None):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_loss += criterion(outputs.logits, labels).item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), total_loss / len(dataloader)


@torch.no_grad()
def predict(model, dataloader, device):
    """Return predictions for a dataloader (no labels)."""
    model.eval()
    all_preds = []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
    return np.array(all_preds)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":   recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":       f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _build_dataloaders(tokenizer, X_train, y_train_idx, X_val, y_val_idx,
                       X_test, y_test_idx, batch_size=BATCH_SIZE):
    """Build DataLoaders for the given splits and tokenizer."""
    train_ds = SentimentDataset(X_train, y_train_idx, tokenizer, MAX_LENGTH)
    val_ds   = SentimentDataset(X_val,   y_val_idx,   tokenizer, MAX_LENGTH)
    test_ds  = SentimentDataset(X_test,  y_test_idx,  tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def _training_loop(model, train_loader, val_loader, DEVICE, epochs=EPOCHS,
                   lr=LR, warmup_ratio=WARMUP_RATIO, class_weights=None):
    """Generic training loop with early stopping; returns (best_state, train_time)."""
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = -1
    patience = 2
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"  Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, class_weights)
        val_preds, val_labels, _ = evaluate(model, val_loader, DEVICE, class_weights)
        val_metrics = compute_metrics(val_labels, val_preds)
        logger.info(f"    Train loss: {train_loss:.4f}  "
                    f"Val accuracy: {val_metrics['accuracy']:.4f}  "
                    f"macro-F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info("    ✓ New best model saved.")
        else:
            patience_counter += 1
            logger.info(f"    No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    train_time = time.time() - start_time
    logger.info(f"  Training time: {train_time:.2f}s")
    return best_state, train_time


def run_transformer_pipeline():
    """
    Full transformer training pipeline (FinBERT or fallback).
    Returns metrics dict with training_time_s, inference_latency_ms, peak_gpu_memory_mb.
    """
    logger.info("=" * 60)
    logger.info("TRANSFORMER PIPELINE (FinBERT)")
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

    model_name, tokenizer, model = try_load_model_and_tokenizer()
    logger.info(f"Model: {model_name}")

    if model_name == FALLBACK_MODEL:
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL, num_labels=num_labels
        )

    model.to(DEVICE)
    freeze_bottom_layers(model, FREEZE_LAYERS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_loader, val_loader, test_loader = _build_dataloaders(
        tokenizer, X_train, y_train_idx, X_val, y_val_idx, X_test, y_test_idx
    )

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    best_state, train_time = _training_loop(model, train_loader, val_loader, DEVICE,
                                           epochs=EPOCHS, lr=LR,
                                           warmup_ratio=WARMUP_RATIO, class_weights=class_weights)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # ── Test evaluation ───────────────────────────────────────────────────
    test_preds, test_labels, _ = evaluate(model, test_loader, DEVICE)
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

    logger.info(f"\nFinBERT Test metrics: {metrics}")

    # ── Save model & artifacts ────────────────────────────────────────────
    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(models_dir, "transformer_model.pt"))
    tokenizer.save_pretrained(os.path.join(models_dir, "transformer_tokenizer"))
    with open(os.path.join(models_dir, "transformer_label_mapping.pkl"), "wb") as f:
        pickle.dump({"label2idx": label2idx, "idx2label": idx2label,
                     "num_labels": num_labels}, f)
    with open(os.path.join(models_dir, "transformer_config.pkl"), "wb") as f:
        pickle.dump({
            "model_name": model_name,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
        }, f)

    logger.info(f"Saved transformer model to {models_dir}/")

    logger.info(
        "\n[QAT Note] To enable Quantization-Aware Training:\n"
        "  1. model = torch.quantization.prepare_qat(model)\n"
        "  2. Fine-tune on training data\n"
        "  3. model = torch.quantization.convert(model)\n"
        "  Expected latency reduction: 2-4x on CPU, 1.5-2x on GPU.\n"
    )

    return metrics, model_name, test_preds, test_labels


# ── Feature 1: BERT-base-uncased comparison ───────────────────────────────────

def run_bertbase_pipeline():
    """
    Train bert-base-uncased with identical settings as FinBERT pipeline.
    Returns same metrics dict keys for easy comparison.
    Saves to models/bertbase_model.pt and models/bertbase_tokenizer/
    """
    logger.info("=" * 60)
    logger.info("TRANSFORMER PIPELINE (BERT-base-uncased comparison)")
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

    model_name = "bert-base-uncased"
    logger.info(f"Loading {model_name} (cached if available)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    freeze_bottom_layers(model, FREEZE_LAYERS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_loader, val_loader, test_loader = _build_dataloaders(
        tokenizer, X_train, y_train_idx, X_val, y_val_idx, X_test, y_test_idx
    )

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    best_state, train_time = _training_loop(model, train_loader, val_loader, DEVICE,
                                           epochs=EPOCHS, lr=LR,
                                           warmup_ratio=WARMUP_RATIO, class_weights=class_weights)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # ── Test evaluation ───────────────────────────────────────────────────
    test_preds, test_labels, _ = evaluate(model, test_loader, DEVICE)
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

    logger.info(f"\nBERT-base Test metrics: {metrics}")

    # ── Save model & tokenizer ─────────────────────────────────────────────
    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(models_dir, "bertbase_model.pt"))
    tokenizer.save_pretrained(os.path.join(models_dir, "bertbase_tokenizer"))
    with open(os.path.join(models_dir, "bertbase_label_mapping.pkl"), "wb") as f:
        pickle.dump({"label2idx": label2idx, "idx2label": idx2label,
                     "num_labels": num_labels}, f)

    logger.info(f"Saved BERT-base model to {models_dir}/")

    return metrics, model_name, test_preds, test_labels


# ── Feature 2: Cost-Quality trade-off (4-bit dynamic quantization) ───────────

def run_quantized_pipeline():
    """
    Apply torch.quantization.quantize_dynamic (int8) to the best FinBERT model.
    Runs inference on test set to compare latency and reports:
      accuracy, macro-F1, inference_latency_ms, memory_usage
    Quantization only affects inference; no training needed.
    """
    logger.info("=" * 60)
    logger.info("QUANTIZATION PIPELINE (Dynamic Int8 Quantization)")
    logger.info("=" * 60)

    splits = load_splits()
    X_test  = splits["test"]["text"]
    y_test  = splits["test"]["label"]

    all_labels = sorted(set(y_test))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    num_labels = len(all_labels)

    y_test_idx = [label2idx[l] for l in y_test]

    # Load FinBERT config
    models_dir = os.path.join(OUTPUT_DIR, "models")
    config_path = os.path.join(models_dir, "transformer_config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    mapping_path = os.path.join(models_dir, "transformer_label_mapping.pkl")
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    label2idx = mapping["label2idx"]
    idx2label = mapping["idx2label"]
    num_labels = mapping["num_labels"]

    y_test_idx = [label2idx[l] for l in y_test]

    model_name = config["model_name"]
    logger.info(f"Loading best FinBERT model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(models_dir, "transformer_tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # Load best checkpoint
    state_path = os.path.join(models_dir, "transformer_model.pt")
    model.load_state_dict(torch.load(state_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    logger.info("Loaded best checkpoint.")

    # ── Baseline inference (before quantization) ───────────────────────────
    logger.info("\nRunning baseline (fp32) inference...")
    model.eval()
    test_ds = SentimentDataset(X_test, y_test_idx, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    import sys
    fp32_params = sum(p.numel() for p in model.parameters())

    start_fp32 = time.perf_counter()
    fp32_preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            fp32_preds_list.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
    fp32_time = time.perf_counter() - start_fp32

    fp32_preds = np.array(fp32_preds_list)
    fp32_metrics = compute_metrics(y_test_idx, fp32_preds)
    fp32_latency_ms = (fp32_time / len(X_test)) * 1000

    if torch.cuda.is_available():
        fp32_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        fp32_mem_mb = sys.getsizeof(model.state_dict()) / 1024 / 1024  # rough estimate

    logger.info(f"  FP32 Accuracy: {fp32_metrics['accuracy']:.4f}  "
                f"F1: {fp32_metrics['f1_macro']:.4f}  "
                f"Latency: {fp32_latency_ms:.2f}ms")

    # ── Apply dynamic quantization (int8) ────────────────────────────────
    logger.info("\nApplying dynamic quantization (qint8)...")
    model.eval()  # quantization requires eval mode
    # Move to CPU for quantization (PyTorch dynamic quantization only supports CPU)
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Quantization complete (model on CPU).")

    # ── Quantized inference (CPU) ─────────────────────────────────────────
    logger.info("Running quantized (int8) inference on CPU...")
    # Create CPU test loader
    test_loader_cpu = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    start_q = time.perf_counter()
    q_preds_list = []
    with torch.no_grad():
        for batch in test_loader_cpu:
            input_ids      = batch["input_ids"]  # keep on CPU
            attention_mask = batch["attention_mask"]
            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            q_preds_list.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
    q_time = time.perf_counter() - start_q

    q_preds = np.array(q_preds_list)
    q_metrics = compute_metrics(y_test_idx, q_preds)
    q_latency_ms = (q_time / len(X_test)) * 1000

    # CPU memory estimate: int8 is ~4x smaller than fp32
    q_mem_mb = fp32_mem_mb / 4.0

    logger.info(f"  INT8 Accuracy: {q_metrics['accuracy']:.4f}  "
                f"F1: {q_metrics['f1_macro']:.4f}  "
                f"Latency: {q_latency_ms:.2f}ms")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "-" * 50)
    logger.info("Quantization Cost-Quality Trade-off Summary:")
    logger.info(f"  {'Metric':<25} {'FP32':>10} {'INT8':>10} {'Change':>10}")
    logger.info("-" * 50)
    logger.info(f"  {'Accuracy':<25} {fp32_metrics['accuracy']:>10.4f} "
                f"{q_metrics['accuracy']:>10.4f} "
                f"{q_metrics['accuracy'] - fp32_metrics['accuracy']:>+10.4f}")
    logger.info(f"  {'Macro-F1':<25} {fp32_metrics['f1_macro']:>10.4f} "
                f"{q_metrics['f1_macro']:>10.4f} "
                f"{q_metrics['f1_macro'] - fp32_metrics['f1_macro']:>+10.4f}")
    logger.info(f"  {'Latency (ms/sample)':<25} {fp32_latency_ms:>10.2f} "
                f"{q_latency_ms:>10.2f} "
                f"{q_latency_ms - fp32_latency_ms:>+10.2f}")
    logger.info(f"  {'Memory (MB)':<25} {fp32_mem_mb:>10.2f} "
                f"{q_mem_mb:>10.2f} "
                f"{q_mem_mb - fp32_mem_mb:>+10.2f}")
    logger.info("-" * 50)

    metrics = {
        "accuracy":             q_metrics["accuracy"],
        "precision_macro":      q_metrics["precision_macro"],
        "recall_macro":         q_metrics["recall_macro"],
        "f1_macro":             q_metrics["f1_macro"],
        "inference_latency_ms": q_latency_ms,
        "memory_usage_mb":      q_mem_mb,
        # also store baseline for comparison table
        "_fp32_accuracy":       fp32_metrics["accuracy"],
        "_fp32_f1_macro":       fp32_metrics["f1_macro"],
        "_fp32_latency_ms":    fp32_latency_ms,
        "_fp32_memory_mb":      fp32_mem_mb,
        "_params_count":        fp32_params,
    }

    return metrics, "FinBERT-Quantized", q_preds, y_test_idx


if __name__ == "__main__":
    metrics, model_name, _, _ = run_transformer_pipeline()
    print(f"\n{model_name} metrics: {metrics}")
