# -*- coding: utf-8 -*-
import os
import json
import time
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from newsapi import NewsApiClient

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import NEWSAPI_KEY, OUTPUT_DIR
from transformer_pipeline import DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_newsapi_data(query="finance OR stock OR market", page_size=100):
    logger.info("=" * 60)
    logger.info("FETCHING OOD DATA FROM NEWSAPI")
    logger.info("=" * 60)
    
    if not NEWSAPI_KEY or NEWSAPI_KEY == "your_api_key_here":
        logger.warning("NewsAPI Key not configured. Skipping OOD data fetch.")
        return []
        
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        
        # Get news from the last week
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        all_articles = newsapi.get_everything(
            q=query,
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt',
            page_size=page_size
        )
        
        articles = all_articles.get('articles', [])
        logger.info(f"Fetched {len(articles)} articles.")
        
        texts = []
        for a in articles:
            text = f"{a.get('title', '')}. {a.get('description', '')}".strip()
            if text and len(text) > 10:
                texts.append(text)
                
        return texts
    except Exception as e:
        logger.error(f"Error fetching from NewsAPI: {e}")
        return []

def run_ood_evaluation(texts, model_name="ProsusAI/finbert"):
    if not texts:
        return
        
    logger.info(f"Evaluating {len(texts)} OOD texts using {model_name}...")
    
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Use the fine-tuned FinBERT checkpoint if available, else load from HF
    saved_model_path = os.path.join(OUTPUT_DIR, "models", "transformer_model.pt")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)
    if os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path, map_location=DEVICE, weights_only=True))
        logger.info("Loaded fine-tuned FinBERT checkpoint for OOD evaluation.")
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_probs = []
    
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(np.max(probs, axis=1))
    
    # ProsusAI/finbert labels: mapped via label_mapping saved during training
    import pickle
    label_mapping_path = os.path.join(OUTPUT_DIR, "models", "transformer_label_mapping.pkl")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, "rb") as f:
            raw = pickle.load(f)
        # Support both formats: {0: 'neg', ...} and {'idx2label': {0: 'neg', ...}}
        if isinstance(raw, dict) and "idx2label" in raw:
            label_map = {int(k): v for k, v in raw["idx2label"].items()}
        else:
            label_map = {int(k): v for k, v in raw.items()}
    else:
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
    pred_labels = [label_map.get(int(p), "unknown") for p in all_preds]
    
    ood_data = []
    for text, label, conf in zip(texts, pred_labels, all_probs):
        ood_data.append({
            "text": text,
            "prediction": label,
            "confidence": float(conf)
        })
        
    with open(os.path.join(results_dir, "ood_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(ood_data, f, indent=2, ensure_ascii=False)
        
    # Analyze confidence distribution
    df = pd.DataFrame(ood_data)
    
    # Uncertainty Analysis: Percentage of high-confidence predictions
    high_conf_threshold = 0.8
    high_conf_pct = (df["confidence"] > high_conf_threshold).mean() * 100
    logger.info(f"OOD High Confidence Predictions (> {high_conf_threshold}): {high_conf_pct:.2f}%")
    
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x="confidence", hue="prediction", multiple="stack", bins=25, alpha=0.7)
    plt.axvline(high_conf_threshold, color='red', linestyle='--', label=f'Threshold ({high_conf_threshold})')
    plt.title(f"OOD Prediction Confidence Distribution\n(High Confidence Ratio: {high_conf_pct:.1f}%)")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "ood_confidence_distribution.png"), dpi=150)
    plt.close()
    
    # Prediction Shift Analysis
    plt.figure(figsize=(10, 6))
    val_counts = df["prediction"].value_counts(normalize=True) * 100
    
    # Compare with training distribution
    splits_json_path = os.path.join(OUTPUT_DIR, "data", "splits.json")
    try:
        with open(splits_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        train_labels = splits["train"]["label"]
        id_counts = pd.Series(train_labels).value_counts(normalize=True) * 100
        
        comparison_df = pd.DataFrame({
            "In-Distribution (Train)": id_counts,
            "Out-of-Distribution (NewsAPI)": val_counts
        }).fillna(0).reset_index().melt(id_vars="index", var_name="Dataset", value_name="Percentage")
        
        sns.barplot(data=comparison_df, x="index", y="Percentage", hue="Dataset", palette="viridis")
        plt.title("Domain Shift Analysis: Sentiment Distribution Comparison")
        plt.ylabel("Percentage of Total (%)")
        plt.xlabel("Sentiment Class")
    except Exception as e:
        logger.warning(f"Could not load in-distribution data for comparison: {e}")
        sns.barplot(x=val_counts.index, y=val_counts.values, palette="Set2")
        plt.title("OOD Predicted Label Distribution")
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ood_prediction_shift.png"), dpi=150)
    plt.close()
    
    logger.info(f"OOD Evaluation complete. Results saved to {results_dir}")
    
    return ood_data

if __name__ == "__main__":
    texts = fetch_newsapi_data()
    run_ood_evaluation(texts)
