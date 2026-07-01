# -*- coding: utf-8 -*-
"""
Main orchestrator for the financial sentiment analysis project.
Runs a 12-step comprehensive pipeline.
"""
import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR
from preprocessing import run as run_preprocessing
from ml_pipeline import train_and_evaluate as run_ml_pipeline
from transformer_pipeline import run_transformer_pipeline, run_bertbase_pipeline, run_quantized_pipeline
from additional_models import run_all_additional_models
from lora_pipeline import run_lora_pipeline
from zero_shot import run_zero_shot
from ensemble import evaluate_ensemble
from eda_analysis import run_eda_and_interpretability
from error_analysis import run_error_analysis
from ood_evaluation import fetch_newsapi_data, run_ood_evaluation
from robustness import run_robustness
from evaluation import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("  FINANCIAL SENTIMENT ANALYSIS — COMPREHENSIVE PIPELINE")
    logger.info("=" * 80)

    # Dictionary to store all model predictions for ensemble/evaluation
    preds_dict = {}
    labels_dict = {}
    all_results = {}

    # Step 1: Preprocessing
    logger.info("\n[STEP 1/12] Preprocessing data...")
    try:
        _, splits_meta, y_train, _, _ = run_preprocessing()
        test_texts = splits_meta["test"]["text_cleaned"]
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    # Step 2: EDA & Interpretability
    logger.info("\n[STEP 2/12] Running EDA & Interpretability...")
    try:
        run_eda_and_interpretability()
    except Exception as e:
        logger.error(f"EDA failed: {e}")

    # Step 3: Traditional ML pipeline
    logger.info("\n[STEP 3/12] Running Traditional ML pipeline...")
    try:
        ml_results, smote_results, best_ml_model, best_ml_name = run_ml_pipeline()
        for name, res in ml_results.items():
            if "_y_true" in res and "_y_pred" in res:
                labels_dict[name] = res["_y_true"]
                preds_dict[name] = res["_y_pred"]
        for name, res in smote_results.items():
            if "_y_true" in res and "_y_pred" in res:
                labels_dict[name] = res["_y_true"]
                preds_dict[name] = res["_y_pred"]
        all_results.update(ml_results)
        all_results.update(smote_results)
    except Exception as e:
        logger.error(f"ML pipeline failed: {e}")

    # Step 4: Core Transformer pipeline
    logger.info("\n[STEP 4/12] Running Core Transformer pipelines (FinBERT, BERT-base, Int8)...")
    try:
        metrics, _, trans_preds, trans_labels = run_transformer_pipeline()
        all_results["FinBERT"] = metrics
        preds_dict["FinBERT"] = trans_preds
        labels_dict["FinBERT"] = trans_labels
        
        b_metrics, _, b_preds, b_labels = run_bertbase_pipeline()
        all_results["BERT-base"] = b_metrics
        preds_dict["BERT-base"] = b_preds
        labels_dict["BERT-base"] = b_labels
        
        q_metrics, _, q_preds, q_labels = run_quantized_pipeline()
        all_results["FinBERT-Int8"] = q_metrics
        preds_dict["FinBERT-Int8"] = q_preds
        labels_dict["FinBERT-Int8"] = q_labels
    except Exception as e:
        logger.error(f"Core Transformer pipeline failed: {e}")

    # Step 5: Additional HuggingFace Models
    logger.info("\n[STEP 5/12] Running Additional HuggingFace Models...")
    try:
        add_results, add_preds, add_labels = run_all_additional_models()
        all_results.update(add_results)
        preds_dict.update(add_preds)
        labels_dict.update(add_labels)
    except Exception as e:
        logger.error(f"Additional models pipeline failed: {e}")

    # Step 6: LoRA Fine-Tuning
    logger.info("\n[STEP 6/12] Running LoRA Fine-Tuning...")
    try:
        l_metrics, l_preds, l_labels = run_lora_pipeline()
        all_results["FinBERT-LoRA"] = l_metrics
        preds_dict["FinBERT-LoRA"] = l_preds
        labels_dict["FinBERT-LoRA"] = l_labels
    except Exception as e:
        logger.error(f"LoRA pipeline failed: {e}")

    # Step 7: Zero-Shot Baseline
    logger.info("\n[STEP 7/12] Running Zero-Shot Baseline...")
    try:
        zs_metrics, zs_preds, zs_labels = run_zero_shot()
        all_results["BART-large-MNLI-ZeroShot"] = zs_metrics
        preds_dict["BART-large-MNLI-ZeroShot"] = zs_preds
        labels_dict["BART-large-MNLI-ZeroShot"] = zs_labels
    except Exception as e:
        logger.error(f"Zero-shot baseline failed: {e}")

    # Step 8: Ensemble Methods
    logger.info("\n[STEP 8/12] Evaluating Ensemble Methods...")
    try:
        y_true = list(labels_dict.values())[0] if labels_dict else []
        ens_results = evaluate_ensemble(preds_dict, y_true)
        all_results.update(ens_results)
        
        for name, res in ens_results.items():
            preds_dict[name] = res["_y_pred"]
            labels_dict[name] = res["_y_true"]
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")

    # Step 9: Error Analysis
    logger.info("\n[STEP 9/12] Running Error Analysis...")
    try:
        import pickle
        with open(os.path.join(OUTPUT_DIR, "data", "splits.pkl"), "rb") as f:
            label_list = sorted(set(pickle.load(f)["train"]["label"]))
        label_map = {i: l for i, l in enumerate(label_list)}
        
        y_true = list(labels_dict.values())[0] if labels_dict else []
        run_error_analysis(y_true, preds_dict, test_texts, label_map=label_map)
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")

    # Step 10: OOD Evaluation (NewsAPI)
    logger.info("\n[STEP 10/12] Running OOD Evaluation (NewsAPI)...")
    try:
        ood_texts = fetch_newsapi_data()
        run_ood_evaluation(ood_texts)
    except Exception as e:
        logger.error(f"OOD evaluation failed: {e}")

    # Step 11: Robustness Evaluation
    logger.info("\n[STEP 11/12] Running Robustness Evaluation...")
    try:
        run_robustness()
    except Exception as e:
        import traceback
        logger.error(f"Robustness evaluation failed: {e}\n{traceback.format_exc()}")

    # Step 12: Comprehensive Evaluation & Visualization
    logger.info("\n[STEP 12/12] Comprehensive Evaluation & Visualization...")
    try:
        run_evaluation(all_results, preds_dict, labels_dict, label_list=label_list)
    except Exception as e:
        import traceback
        logger.error(f"Comprehensive evaluation failed: {e}\n{traceback.format_exc()}")

    # Generate Final Summary
    logger.info("\n" + "=" * 80)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 80)
    
    summary_path = os.path.join(OUTPUT_DIR, "results", "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FINAL PIPELINE SUMMARY\n")
        f.write("=" * 80 + "\n")
        
        for name, m in sorted(all_results.items(), key=lambda x: x[1].get('f1_macro', 0), reverse=True):
            f1 = m.get('f1_macro', 'N/A')
            acc = m.get('accuracy', 'N/A')
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            
            line = f"  {name:<35}  F1={f1_str}  Accuracy={acc_str}"
            logger.info(line)
            f.write(line + "\n")
            
    logger.info(f"\nPipeline complete! Full text summary saved to: {summary_path}")
    logger.info(f"All visualizations and CSVs saved to: {os.path.join(OUTPUT_DIR, 'results')}")

if __name__ == "__main__":
    main()
