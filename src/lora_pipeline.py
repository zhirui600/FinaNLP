# -*- coding: utf-8 -*-
import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.utils.class_weight import compute_class_weight

from config import OUTPUT_DIR, RANDOM_STATE
from transformer_pipeline import (
    DEVICE, BATCH_SIZE, MAX_LENGTH, EPOCHS, LR, WARMUP_RATIO,
    load_splits, compute_metrics,
    _build_dataloaders, _training_loop, evaluate, predict
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_lora_pipeline():
    logger.info("=" * 60)
    logger.info("TRAINING FINBERT WITH LORA + 4-BIT QUANTIZATION")
    logger.info("=" * 60)

    model_name = "ProsusAI/finbert"

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

    logger.info(f"Loading tokenizer and 4-bit model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4-bit quantization config
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, quantization_config=bnb_config,
            ignore_mismatched_sizes=True
        )
        # CRITICAL: Prepare model for kbit training MUST be called before get_peft_model
        model = prepare_model_for_kbit_training(model)
        logger.info("Successfully loaded model in 4-bit and prepared for kbit training.")
    except Exception as e:
        logger.warning(f"Failed to load 4-bit model: {e}. Falling back to standard precision.")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        model.to(DEVICE)
    
    # Apply LoRA
    try:
        # Determine target modules based on model architecture
        target_modules = ["query", "value"]
        if "roberta" in model_name.lower():
            target_modules = ["query", "value"]
        elif "deberta" in model_name.lower():
            target_modules = ["query_proj", "value_proj"]
        
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with target modules: {target_modules}")
    except Exception as e:
        logger.warning(f"Target modules {target_modules} failed: {e}. Using automatic discovery.")
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, lora_config)

    # CRITICAL: ensure all parameters with gradients are float32
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    
    train_loader, val_loader, test_loader = _build_dataloaders(
        tokenizer, X_train, y_train_idx, X_val, y_val_idx, X_test, y_test_idx
    )

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    best_state, train_time = _training_loop(
        model, train_loader, val_loader, DEVICE,
        epochs=EPOCHS, lr=LR*5, warmup_ratio=WARMUP_RATIO, class_weights=class_weights # Higher LR for LoRA
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        # 4-bit model might not support .to(DEVICE) again if already there
        if not hasattr(model, 'is_loaded_in_4bit') or not model.is_loaded_in_4bit:
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

    logger.info(f"\nFinBERT-LoRA Test metrics: {metrics}")

    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    clean_name = "FinBERT-LoRA"
    model.save_pretrained(os.path.join(models_dir, clean_name))
    tokenizer.save_pretrained(os.path.join(models_dir, f"{clean_name}_tokenizer"))

    return metrics, test_preds, test_labels

if __name__ == "__main__":
    run_lora_pipeline()
