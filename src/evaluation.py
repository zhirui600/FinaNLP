# -*- coding: utf-8 -*-
"""
Comprehensive evaluation module.
Evaluates both ML and Transformer models, prints comparison tables,
saves results to JSON, and generates advanced visualizations.
"""
import os
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#4C72B0", "#DD8452"]

def evaluate_model(y_true, y_pred, train_time=None, inference_latency=None, memory_usage=None):
    metrics = {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    if train_time is not None:
        metrics["train_time_s"] = round(train_time, 4)
    if inference_latency is not None:
        metrics["inference_latency_ms"] = round(inference_latency, 4)
    if memory_usage is not None:
        metrics["peak_gpu_memory_mb"] = round(memory_usage, 2)
    return metrics

def save_metrics_json(all_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in all_results.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metrics to {path}")

def save_comparison_csv(metrics_dict, path):
    models = list(metrics_dict.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "train_time_s", "inference_latency_ms", "peak_gpu_memory_mb"]
    
    rows = []
    for model in models:
        m = metrics_dict[model]
        row = {"Model": model}
        for k in metric_keys:
            row[k] = m.get(k, None)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Saved full comparison table to {path}")

def print_comparison_table(metrics_dict, title="Model Comparison"):
    models = list(metrics_dict.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    optional_keys = ["train_time_s", "inference_latency_ms", "peak_gpu_memory_mb"]
    all_keys = metric_keys + [k for k in optional_keys if any(k in metrics_dict[m] for m in models)]

    header = f"{'Model':<35}" + "".join(f"{k.upper().replace('_',' '):>18}" for k in all_keys)
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"{title:^{len(header)}}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for model in models:
        m = metrics_dict[model]
        row = f"{model:<35}"
        for k in all_keys:
            val = m.get(k, "N/A")
            if isinstance(val, float):
                row += f"{val:>18.4f}"
            else:
                row += f"{str(val):>18}"
        print(row)
    print(f"{'=' * len(header)}\n")

def plot_confusion_matrices(metrics_dict, all_y_true, all_y_pred, label_list):
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    models_to_plot = list(all_y_true.keys())
    if not models_to_plot:
        return
        
    # Plot grid
    n_cols = min(3, len(models_to_plot))
    n_rows = (len(models_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    if len(models_to_plot) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_name in enumerate(models_to_plot):
        ax = axes[idx]
        y_true = all_y_true[model_name]
        y_pred = all_y_pred[model_name]
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_list,
                    yticklabels=label_list, ax=ax, cbar=False, linewidths=0.5,
                    linecolor="gray", annot_kws={"size": 10})
        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        
    for idx in range(len(models_to_plot), len(axes)):
        axes[idx].axis('off')

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "confusion_matrices.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_metrics_bar_chart(metrics_dict):
    results_dir = os.path.join(OUTPUT_DIR, "results")
    
    models = list(metrics_dict.keys())
    score_keys = ["accuracy", "f1_macro"]
    
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))

    for i, key in enumerate(score_keys):
        values = [metrics_dict[m].get(key, 0) for m in models]
        bars = ax.bar(x + i * width, values, width, label=key.replace("_macro", "").replace("_", " ").title(),
                      color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

def plot_radar_chart(metrics_dict):
    results_dir = os.path.join(OUTPUT_DIR, "results")
    
    # Select top 5 models to avoid clutter
    models = sorted(list(metrics_dict.keys()), key=lambda k: metrics_dict[k].get("f1_macro", 0), reverse=True)[:5]
    if not models: return
    
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    
    N = len(metrics_to_plot)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(m, 0) for m in metrics_to_plot]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])
        
    plt.xticks(angles[:-1], [m.replace("_macro", "").title() for m in metrics_to_plot], size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    plt.title("Multi-dimensional Performance Comparison (Top 5 Models)", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "radar_comparison.png"), dpi=150)
    plt.close(fig)

def plot_cost_quality_tradeoff(metrics_dict):
    results_dir = os.path.join(OUTPUT_DIR, "results")
    
    models = []
    f1_scores = []
    latencies = []
    memories = []
    
    for m, vals in metrics_dict.items():
        if "f1_macro" in vals and "inference_latency_ms" in vals and "peak_gpu_memory_mb" in vals:
            if vals["inference_latency_ms"] > 0: # Filter out models without latency recorded
                models.append(m)
                f1_scores.append(vals["f1_macro"])
                latencies.append(vals["inference_latency_ms"])
                memories.append(vals["peak_gpu_memory_mb"])
            
    if not models:
        return
        
    plt.figure(figsize=(10, 8))
    
    # Scale memories for bubble size (add a base size so small ones are visible)
    sizes = [max(m / 5, 50) for m in memories]
    
    scatter = plt.scatter(latencies, f1_scores, s=sizes, alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        plt.annotate(model, (latencies[i], f1_scores[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.title("Cost-Quality Tradeoff (Bubble Size = GPU Memory)")
    plt.xlabel("Inference Latency per Sample (ms)")
    plt.ylabel("Macro F1 Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cost_quality_tradeoff.png"), dpi=150)
    plt.close()

def run_statistical_test(all_y_true, all_y_pred):
    results_dir = os.path.join(OUTPUT_DIR, "results")
    stats_results = {}
    
    # Define which models are ML-based vs Transformer-based
    ml_names = {"LogisticRegression", "LinearSVC", "MultinomialNB",
                "LogisticRegression+SMOTE", "LinearSVC+SMOTE", "MultinomialNB+SMOTE"}
    
    transformer_names = {m for m in all_y_true.keys() if m not in ml_names
                         and "Ensemble" not in m and "ZeroShot" not in m}
    
    try:
        models = list(all_y_true.keys())
        if len(models) >= 2:
            # Find best ML model by accuracy
            best_ml = None
            best_ml_acc = -1
            for m in models:
                if m in ml_names and m in all_y_pred:
                    y_true = np.array(all_y_true[m])
                    y_pred = np.array(all_y_pred[m])
                    acc = np.mean(y_true == y_pred)
                    if acc > best_ml_acc:
                        best_ml_acc = acc
                        best_ml = m
            
            # Find best Transformer model by accuracy
            best_trans = None
            best_trans_acc = -1
            for m in models:
                if m in transformer_names and m in all_y_pred:
                    y_true = np.array(all_y_true[m])
                    y_pred = np.array(all_y_pred[m])
                    acc = np.mean(y_true == y_pred)
                    if acc > best_trans_acc:
                        best_trans_acc = acc
                        best_trans = m
            
            if best_ml and best_trans:
                m1, m2 = best_ml, best_trans
            else:
                m1, m2 = models[0], models[1]
            
            y1 = np.array(all_y_pred[m1])
            y2 = np.array(all_y_pred[m2])
            y_true = np.array(all_y_true[m1])
            
            # Contingency table
            yy = sum((y1 == y_true) & (y2 == y_true))
            yn = sum((y1 == y_true) & (y2 != y_true))
            ny = sum((y1 != y_true) & (y2 == y_true))
            nn = sum((y1 != y_true) & (y2 != y_true))
            
            table = [[yy, yn], [ny, nn]]
            result = mcnemar(table, exact=False, correction=True)
            
            stats_results["mcnemar_test"] = {
                "model_1": m1,
                "model_2": m2,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "significant_diff": bool(result.pvalue < 0.05)
            }
            logger.info(f"McNemar Test ({m1} vs {m2}): p-value = {result.pvalue:.4e} -> {'Significant' if result.pvalue < 0.05 else 'Not significant'}")
            
    except Exception as e:
        logger.warning(f"Failed to run statistical tests: {e}")
        
    with open(os.path.join(results_dir, "statistical_tests.json"), "w") as f:
        json.dump(stats_results, f, indent=2)

def run_evaluation(all_results, preds_dict, labels_dict, label_list=None):
    if label_list is None:
        label_list = ["negative", "neutral", "positive"]

    print_comparison_table(all_results, title="Financial Sentiment — Model Comparison")

    results_path = os.path.join(OUTPUT_DIR, "results", "metrics.json")
    save_metrics_json(all_results, results_path)
    save_comparison_csv(all_results, os.path.join(OUTPUT_DIR, "results", "full_comparison_table.csv"))

    plot_confusion_matrices(all_results, labels_dict, preds_dict, label_list)
    plot_metrics_bar_chart(all_results)
    plot_radar_chart(all_results)
    plot_cost_quality_tradeoff(all_results)
    
    if len(preds_dict) >= 2:
        run_statistical_test(labels_dict, preds_dict)

    return all_results
