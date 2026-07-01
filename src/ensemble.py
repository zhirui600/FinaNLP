# -*- coding: utf-8 -*-
import logging
import numpy as np
from transformer_pipeline import compute_metrics

logger = logging.getLogger(__name__)

def weighted_vote_ensemble(preds_list, weights=None):
    """
    preds_list: list of numpy arrays, each containing integer predictions.
    weights: list of weights for each model's vote.
    """
    if not preds_list:
        return np.array([])
        
    if weights is None:
        weights = [1.0] * len(preds_list)
        
    assert len(preds_list) == len(weights), "Number of weights must match number of prediction arrays"
    
    n_samples = len(preds_list[0])
    ensemble_preds = []
    
    for i in range(n_samples):
        votes = {}
        for j, preds in enumerate(preds_list):
            pred = preds[i]
            votes[pred] = votes.get(pred, 0) + weights[j]
        
        # Get the class with maximum weighted vote
        best_pred = max(votes.items(), key=lambda x: x[1])[0]
        ensemble_preds.append(best_pred)
        
    return np.array(ensemble_preds)

def evaluate_ensemble(preds_dict, y_true):
    """
    Evaluate specific ensemble combinations.
    preds_dict: dict mapping model_name to its predictions array.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING ENSEMBLE MODELS")
    logger.info("=" * 60)
    
    ensemble_results = {}
    
    if not preds_dict:
        logger.warning("No predictions provided for ensemble.")
        return ensemble_results
        
    # Example 1: All models equally weighted
    preds_list = list(preds_dict.values())
    ens_preds = weighted_vote_ensemble(preds_list)
    metrics = compute_metrics(y_true, ens_preds)
    metrics["train_time_s"] = 0.0
    metrics["inference_latency_ms"] = 0.0
    metrics["_y_true"] = y_true
    metrics["_y_pred"] = ens_preds.tolist()
    
    name = f"Ensemble_All ({len(preds_list)} models)"
    ensemble_results[name] = metrics
    logger.info(f"{name}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Example 2: Top 3 models (assuming we can identify them later, for now we just do All)
    # The actual selection of models can be done in main.py before calling this.
    
    return ensemble_results
