# -*- coding: utf-8 -*-
import os
import json
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_error_analysis(y_true, preds_dict, texts, label_map=None):
    """
    y_true: list of true labels (int)
    preds_dict: dict of model_name -> list of pred labels (int)
    texts: list of corresponding text strings
    label_map: dict mapping label int to string
    """
    logger.info("=" * 60)
    logger.info("RUNNING ERROR ANALYSIS")
    logger.info("=" * 60)
    
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if label_map:
        y_true_str = [label_map.get(y, str(y)) for y in y_true]
    else:
        y_true_str = [str(y) for y in y_true]
        
    df = pd.DataFrame({"text": texts, "true_label": y_true_str})
    
    for name, preds in preds_dict.items():
        if label_map:
            preds_str = [label_map.get(p, str(p)) for p in preds]
        else:
            preds_str = [str(p) for p in preds]
            
        df[f"pred_{name}"] = preds_str
        df[f"correct_{name}"] = (df["true_label"] == df[f"pred_{name}"])
        
    # Identify hard samples (where all models fail)
    model_correct_cols = [c for c in df.columns if c.startswith("correct_")]
    df["num_correct"] = df[model_correct_cols].sum(axis=1)
    
    hard_samples = df[df["num_correct"] == 0].copy()
    logger.info(f"Identified {len(hard_samples)} hard samples (all models failed).")
    
    # Save hard samples
    hard_samples.to_csv(os.path.join(results_dir, "hard_samples.csv"), index=False)
    
    # Error distribution by class
    error_rates = []
    for name in preds_dict.keys():
        col = f"correct_{name}"
        class_errors = df.groupby("true_label")[col].apply(lambda x: (1 - x.mean()) * 100).reset_index()
        class_errors.rename(columns={col: "error_rate"}, inplace=True)
        class_errors["model"] = name
        error_rates.append(class_errors)
        
    error_df = pd.concat(error_rates)
    
    # Limit number of models in the plot if there are too many, take top 5
    models_to_plot = list(preds_dict.keys())[:5]
    error_df_sub = error_df[error_df["model"].isin(models_to_plot)]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=error_df_sub, x="true_label", y="error_rate", hue="model", palette="Set3")
    plt.title("Error Rate by Class across Models")
    plt.ylabel("Error Rate (%)")
    plt.savefig(os.path.join(results_dir, "error_distribution.png"), dpi=150)
    plt.close()
    
    # Model disagreement (Model A correct, Model B incorrect)
    models = list(preds_dict.keys())
    # If too many models, select a subset to avoid huge heatmaps
    if len(models) > 8:
        models = models[:8]
        
    disagreement_matrix = np.zeros((len(models), len(models)))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j:
                m1_corr = df[f"correct_{m1}"]
                m2_corr = df[f"correct_{m2}"]
                # Percentage where m1 is correct and m2 is incorrect
                disagreement = (m1_corr & ~m2_corr).mean() * 100
                disagreement_matrix[i, j] = disagreement
                
    plt.figure(figsize=(max(8, len(models)*1.2), max(6, len(models)*1)))
    sns.heatmap(disagreement_matrix, annot=True, fmt=".1f", xticklabels=models, yticklabels=models, cmap="YlOrRd")
    plt.title("Model Disagreement (%) - Row correct, Column incorrect")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_disagreement.png"), dpi=150)
    plt.close()
    
    # Word count vs error rate analysis
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["length_bin"] = pd.qcut(df["word_count"], q=4, labels=["Short", "Medium", "Long", "Very Long"])
    
    len_errors = []
    for name in models_to_plot:
        col = f"correct_{name}"
        len_error = df.groupby("length_bin")[col].apply(lambda x: (1 - x.mean()) * 100).reset_index()
        len_error.rename(columns={col: "error_rate"}, inplace=True)
        len_error["model"] = name
        len_errors.append(len_error)
        
    len_df = pd.concat(len_errors)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=len_df, x="length_bin", y="error_rate", hue="model", marker="o")
    plt.title("Error Rate by Sentence Length")
    plt.ylabel("Error Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "error_by_length.png"), dpi=150)
    plt.close()
    
    error_stats = {
        "hard_samples_count": len(hard_samples),
        "total_samples": len(df),
        "hard_samples_ratio": len(hard_samples) / len(df)
    }
    
    with open(os.path.join(results_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(error_stats, f, indent=2)
        
    logger.info("Error analysis complete.")
    return error_stats
