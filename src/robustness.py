# -*- coding: utf-8 -*-
"""
Robustness evaluation module.
Adds multiple noise levels to test data and measures performance degradation.
Includes synonym replacement using WordNet.
"""
import os
import random
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet

from config import OUTPUT_DIR
from transformer_pipeline import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    wordnet.synsets("test")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# ── Noise functions ──────────────────────────────────────────────────────────

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(text, p=0.1):
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    words = text.split()
    if not words: return text
    
    n_replace = max(1, int(len(words) * p))
    positions = random.sample(range(len(words)), min(n_replace, len(words)))
    
    for i in positions:
        synonyms = get_synonyms(words[i])
        if synonyms:
            words[i] = random.choice(synonyms)
            
    return " ".join(words)

def apply_all_noise(text, noise_level=0.1):
    """Apply noise proportional to noise_level."""
    if noise_level == 0:
        return text
        
    text = synonym_replacement(text, p=noise_level)
    
    # char level noise
    chars = list(text)
    n_noise = max(1, int(len(chars) * (noise_level / 2)))
    
    # Random swap
    for _ in range(n_noise):
        if len(chars) > 2:
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            
    # Random deletion
    positions = random.sample(range(len(chars)), min(n_noise, len(chars)))
    for i in sorted(positions, reverse=True):
        del chars[i]
        
    return "".join(chars)

def add_noise_to_texts(texts, noise_level=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    return [apply_all_noise(str(t), noise_level) for t in texts]

# ── ML model inference ───────────────────────────────────────────────────────

def load_ml_model():
    models_dir = os.path.join(OUTPUT_DIR, "models")
    path_model = os.path.join(models_dir, "ml_best_model.pkl")
    path_lbl   = os.path.join(models_dir, "label_mapping.pkl")
    if not os.path.exists(path_model):
        return None, None
    with open(path_model, "rb") as f:
        model = pickle.load(f)
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    with open(path_lbl, "rb") as f:
        mapping = pickle.load(f)
    return model, mapping

def load_splits():
    data_dir = os.path.join(OUTPUT_DIR, "data")
    with open(os.path.join(data_dir, "splits.pkl"), "rb") as f:
        return pickle.load(f)

def predict_ml(model, texts):
    texts_list = [str(t) for t in texts]
    preds = model.predict(texts_list)
    return preds

# ── Transformer inference ────────────────────────────────────────────────────

def load_transformer():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        models_dir = os.path.join(OUTPUT_DIR, "models")
        
        # Look for finbert first
        clean_name = "yiyanghkust_finbert-tone"
        model_pt = os.path.join(models_dir, f"{clean_name}_model.pt")
        
        if not os.path.exists(model_pt):
            # Fallback to older transformer path
            model_pt = os.path.join(models_dir, "transformer_model.pt")
            if not os.path.exists(model_pt):
                return None, None, None
                
            with open(os.path.join(models_dir, "transformer_config.pkl"), "rb") as f:
                cfg = pickle.load(f)
            model_name = cfg["model_name"]
            tokenizer_dir = os.path.join(models_dir, "transformer_tokenizer")
            mapping_path = os.path.join(models_dir, "transformer_label_mapping.pkl")
        else:
            model_name = "yiyanghkust/finbert-tone"
            tokenizer_dir = os.path.join(models_dir, f"{clean_name}_tokenizer")
            mapping_path = os.path.join(models_dir, f"{clean_name}_label_mapping.pkl")

        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=mapping["num_labels"], ignore_mismatched_sizes=True
        )
        state_dict = torch.load(model_pt, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model, tokenizer, mapping
    except Exception as e:
        logger.warning(f"Could not load transformer model: {e}")
        return None, None, None

def predict_transformer(model, tokenizer, texts, batch_size=16, max_length=128):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = next(model.parameters()).device
    enc = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [t.to(DEVICE) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)

# ── Main robustness evaluation ───────────────────────────────────────────────

def run_robustness():
    logger.info("=" * 60)
    logger.info("ROBUSTNESS EVALUATION (MULTI-LEVEL)")
    logger.info("=" * 60)

    splits = load_splits()
    test_texts = splits["test"]["text"]
    test_labels = splits["test"]["label"]

    all_labels = sorted(set(splits["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    y_test_idx = [label2idx[l] for l in test_labels]

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    results = {}
    f1_scores = {"ML (TF-IDF)": [], "Transformer": []}
    
    ml_model, _ = load_ml_model()
    trans_model, tokenizer, _ = load_transformer()

    for nl in noise_levels:
        logger.info(f"\n--- Testing Noise Level: {nl*100:.0f}% ---")
        noisy_texts = add_noise_to_texts(test_texts, noise_level=nl)
        
        # Evaluate ML
        if ml_model is not None:
            preds_ml = predict_ml(ml_model, noisy_texts)
            metrics = compute_metrics(y_test_idx, preds_ml)
            f1_scores["ML (TF-IDF)"].append(metrics["f1_macro"])
            logger.info(f"  ML F1: {metrics['f1_macro']:.4f}")
            
        # Evaluate Transformer
        if trans_model is not None:
            preds_tr = predict_transformer(trans_model, tokenizer, noisy_texts)
            metrics = compute_metrics(y_test_idx, preds_tr)
            f1_scores["Transformer"].append(metrics["f1_macro"])
            logger.info(f"  Transformer F1: {metrics['f1_macro']:.4f}")

    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot Degradation Curve
    plt.figure(figsize=(10, 6))
    if ml_model is not None:
        sns.lineplot(x=[nl*100 for nl in noise_levels], y=f1_scores["ML (TF-IDF)"], 
                     marker="o", label="ML (TF-IDF)")
    if trans_model is not None:
        sns.lineplot(x=[nl*100 for nl in noise_levels], y=f1_scores["Transformer"], 
                     marker="s", label="Transformer (FinBERT)")
                     
    plt.title("Performance Degradation under Noise (Synonyms + Typo)")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Macro F1-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "robustness_degradation_curve.png"), dpi=150)
    plt.close()

    # Save results json
    results = {
        "noise_levels": [nl*100 for nl in noise_levels],
        "f1_scores": f1_scores
    }
    import json
    with open(os.path.join(results_dir, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved robustness results to {results_dir}/")
    return results

if __name__ == "__main__":
    run_robustness()
