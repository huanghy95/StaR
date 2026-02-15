"""
Dynamic Graph Causal Discovery Metrics

This module implements metrics for evaluating causal discovery performance on dynamic graphs,
where the causal structure changes over time. These metrics are based on the paper:
"Dynamic Causal Structure Discovery and Causal Effect Estimation" (arXiv:2501.06534v1)

Metrics implemented:
1. Structural Hamming Distance (SHD) - Counts edge differences (additions, deletions, reversals)
2. False Discovery Rate (FDR) - Proportion of incorrectly identified edges
3. True Positive Rate (TPR) - Proportion of correctly identified edges (Recall/Sensitivity)
4. Mean Squared Error (MSE) - Error in estimating edge weights over time

For dynamic graphs, these metrics are computed at each timestamp and then aggregated.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import f1_score


def compute_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance (SHD) between two adjacency matrices.
    
    SHD counts the number of edge operations needed to transform the predicted graph
    into the true graph:
    - Edge additions (false negatives)
    - Edge deletions (false positives)
    - Edge reversals (incorrect direction)
    
    Args:
        true_adj: True adjacency matrix (binary), shape (n, n)
        pred_adj: Predicted adjacency matrix (binary), shape (n, n)
        
    Returns:
        shd: Structural Hamming Distance (integer)
    """
    # Convert to binary if not already using vectorized operations
    true_binary = (true_adj > 0).astype(np.int8)
    pred_binary = (pred_adj > 0).astype(np.int8)
    
    # Count edge differences using vectorized operations
    # False positives: edges in pred but not in true
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    
    # False negatives: edges in true but not in pred
    fn = np.sum((true_binary == 1) & (pred_binary == 0))
    
    # For directed graphs, count reversals using vectorized operations
    # A reversal is when (i,j) exists in true but (j,i) exists in pred (and vice versa)
    # Get upper triangle indices to avoid double counting
    n = true_adj.shape[0]
    upper_tri_idx = np.triu_indices(n, k=1)
    
    # Check reversals: true[i,j]=1, pred[i,j]=0, true[j,i]=0, pred[j,i]=1
    forward_reversal = (
        (true_binary[upper_tri_idx] == 1) & 
        (pred_binary[upper_tri_idx] == 0) & 
        (true_binary[upper_tri_idx[1], upper_tri_idx[0]] == 0) & 
        (pred_binary[upper_tri_idx[1], upper_tri_idx[0]] == 1)
    )
    
    # Check reversals: true[j,i]=1, pred[j,i]=0, true[i,j]=0, pred[i,j]=1
    backward_reversal = (
        (true_binary[upper_tri_idx[1], upper_tri_idx[0]] == 1) & 
        (pred_binary[upper_tri_idx[1], upper_tri_idx[0]] == 0) & 
        (true_binary[upper_tri_idx] == 0) & 
        (pred_binary[upper_tri_idx] == 1)
    )
    
    reversals = np.sum(forward_reversal) + np.sum(backward_reversal)
    
    # SHD = additions + deletions + reversals
    # Note: reversals are already counted as one FP and one FN, so we subtract them once
    shd = fp + fn - reversals
    
    return int(shd)


def compute_fdr_tpr(true_adj: np.ndarray, pred_adj: np.ndarray) -> Tuple[float, float]:
    """
    Compute False Discovery Rate (FDR) and True Positive Rate (TPR).
    
    FDR = FP / (FP + TP) - proportion of incorrect edges among predicted edges
    TPR = TP / (TP + FN) - proportion of correct edges among true edges (Recall)
    
    Args:
        true_adj: True adjacency matrix (binary), shape (n, n)
        pred_adj: Predicted adjacency matrix (binary), shape (n, n)
        
    Returns:
        fdr: False Discovery Rate [0, 1]
        tpr: True Positive Rate [0, 1]
    """
    # Convert to binary if not already
    true_binary = (true_adj > 0).astype(int)
    pred_binary = (pred_adj > 0).astype(int)
    
    # Flatten matrices
    true_flat = true_binary.flatten()
    pred_flat = pred_binary.flatten()
    
    # Compute confusion matrix elements
    tp = np.sum((true_flat == 1) & (pred_flat == 1))  # True positives
    fp = np.sum((true_flat == 0) & (pred_flat == 1))  # False positives
    fn = np.sum((true_flat == 1) & (pred_flat == 0))  # False negatives
    
    # Compute FDR
    if tp + fp == 0:
        fdr = 0.0  # No edges predicted, so no false discoveries
    else:
        fdr = fp / (tp + fp)
    
    # Compute TPR (Recall)
    if tp + fn == 0:
        tpr = 0.0  # No true edges, undefined but set to 0
    else:
        tpr = tp / (tp + fn)
    
    return float(fdr), float(tpr)


def compute_mse_weights(true_weights: np.ndarray, pred_weights: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Mean Squared Error (MSE) for edge weights.
    
    This measures how well the predicted edge weights match the true weights,
    which is important for dynamic graphs where edge strengths vary over time.
    
    Args:
        true_weights: True edge weights, shape (n, n)
        pred_weights: Predicted edge weights, shape (n, n)
        mask: Optional binary mask to compute MSE only on certain edges (e.g., existing edges)
        
    Returns:
        mse: Mean Squared Error
    """
    if mask is not None:
        # Compute MSE only on masked elements
        masked_true = true_weights[mask > 0]
        masked_pred = pred_weights[mask > 0]
        if len(masked_true) == 0:
            return 0.0
        mse = np.mean((masked_true - masked_pred) ** 2)
    else:
        # Compute MSE on all elements
        mse = np.mean((true_weights - pred_weights) ** 2)
    
    return float(mse)


def evaluate_dynamic_causal_discovery(
    true_graphs: List[np.ndarray],
    pred_graphs: List[np.ndarray],
    quantile: float = 0.8,
    diagonal: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate causal discovery performance on dynamic graphs.
    
    This function computes SHD, FDR, TPR, and MSE at each timestamp and aggregates them.
    Following the approach from the dynamic causal discovery paper (arXiv:2501.06534v1).
    
    Args:
        true_graphs: List of true adjacency matrices (one per timestamp), each shape (n, n)
        pred_graphs: List of predicted adjacency matrices (continuous values), each shape (n, n)
        quantile: Quantile for thresholding predicted graphs to binary (default: 0.8)
        diagonal: Whether to include diagonal elements in evaluation (default: False)
        
    Returns:
        results: Dictionary containing:
            - 'per_timestamp': List of dicts with metrics for each timestamp
            - 'aggregated': Dict with mean and std of each metric across timestamps
            - 'final': Dict with metrics computed on the last timestamp only
    """
    if len(true_graphs) != len(pred_graphs):
        raise ValueError(f"Number of true graphs ({len(true_graphs)}) must match "
                        f"number of predicted graphs ({len(pred_graphs)})")
    
    if len(true_graphs) == 0:
        raise ValueError("No graphs provided for evaluation")
    
    n_timestamps = len(true_graphs)
    
    # Pre-allocate arrays for vectorized operations
    shd_values = np.zeros(n_timestamps, dtype=np.int32)
    fdr_values = np.zeros(n_timestamps, dtype=np.float32)
    tpr_values = np.zeros(n_timestamps, dtype=np.float32)
    mse_values = np.zeros(n_timestamps, dtype=np.float32)
    f1_values = np.zeros(n_timestamps, dtype=np.float32)
    
    # Process all graphs
    for t in range(n_timestamps):
        true_adj = true_graphs[t].copy()
        pred_continuous = pred_graphs[t].copy()
        
        # Remove diagonal if needed
        if not diagonal:
            np.fill_diagonal(true_adj, 0)
            np.fill_diagonal(pred_continuous, 0)
        
        # Threshold predicted graph to binary using quantile
        threshold = np.quantile(np.abs(pred_continuous), quantile)
        pred_binary = (np.abs(pred_continuous) >= threshold).astype(np.float32)
        
        # Compute metrics for this timestamp
        shd_values[t] = compute_shd(true_adj, pred_binary)
        fdr_values[t], tpr_values[t] = compute_fdr_tpr(true_adj, pred_binary)
        
        # Compute MSE on edge weights (only on existing edges in true graph)
        true_mask = (true_adj > 0).astype(np.float32)
        mse_values[t] = compute_mse_weights(true_adj, np.abs(pred_continuous), mask=true_mask)
        
        # Also compute F1 score for comparison with static methods
        f1_values[t] = f1_score(true_adj.flatten(), pred_binary.flatten(), zero_division=0)
    
    # Build per-timestamp metrics (only if needed for debugging)
    per_timestamp_metrics = [
        {
            'timestamp': t,
            'SHD': int(shd_values[t]),
            'FDR': float(fdr_values[t]),
            'TPR': float(tpr_values[t]),
            'MSE': float(mse_values[t]),
            'F1': float(f1_values[t])
        }
        for t in range(n_timestamps)
    ]
    
    # Aggregate metrics across timestamps using vectorized operations
    aggregated_metrics = {
        'SHD_mean': float(np.mean(shd_values)),
        'SHD_std': float(np.std(shd_values)),
        'FDR_mean': float(np.mean(fdr_values)),
        'FDR_std': float(np.std(fdr_values)),
        'TPR_mean': float(np.mean(tpr_values)),
        'TPR_std': float(np.std(tpr_values)),
        'MSE_mean': float(np.mean(mse_values)),
        'MSE_std': float(np.std(mse_values)),
        'F1_mean': float(np.mean(f1_values)),
        'F1_std': float(np.std(f1_values))
    }
    
    # Also return metrics for the final timestamp (most recent)
    final_metrics = {
        'SHD': int(shd_values[-1]),
        'FDR': float(fdr_values[-1]),
        'TPR': float(tpr_values[-1]),
        'MSE': float(mse_values[-1]),
        'F1': float(f1_values[-1])
    }
    
    return {
        'per_timestamp': per_timestamp_metrics,
        'aggregated': aggregated_metrics,
        'final': final_metrics
    }


def format_dynamic_metrics_report(results: Dict) -> str:
    """
    Format dynamic graph metrics into a readable report string.
    
    Args:
        results: Results dictionary from evaluate_dynamic_causal_discovery()
        
    Returns:
        report: Formatted string report
    """
    agg = results['aggregated']
    final = results['final']
    
    report = "\n" + "="*80 + "\n"
    report += "DYNAMIC GRAPH CAUSAL DISCOVERY METRICS\n"
    report += "="*80 + "\n\n"
    
    report += "Aggregated Metrics (Mean ± Std across all timestamps):\n"
    report += "-" * 80 + "\n"
    report += f"  Structural Hamming Distance (SHD): {agg['SHD_mean']:.4f} ± {agg['SHD_std']:.4f}\n"
    report += f"  False Discovery Rate (FDR):       {agg['FDR_mean']:.4f} ± {agg['FDR_std']:.4f}\n"
    report += f"  True Positive Rate (TPR):          {agg['TPR_mean']:.4f} ± {agg['TPR_std']:.4f}\n"
    report += f"  Mean Squared Error (MSE):          {agg['MSE_mean']:.4f} ± {agg['MSE_std']:.4f}\n"
    report += f"  F1 Score:                          {agg['F1_mean']:.4f} ± {agg['F1_std']:.4f}\n"
    report += "\n"
    
    report += "Final Timestamp Metrics:\n"
    report += "-" * 80 + "\n"
    report += f"  Structural Hamming Distance (SHD): {final['SHD']:.4f}\n"
    report += f"  False Discovery Rate (FDR):       {final['FDR']:.4f}\n"
    report += f"  True Positive Rate (TPR):          {final['TPR']:.4f}\n"
    report += f"  Mean Squared Error (MSE):          {final['MSE']:.4f}\n"
    report += f"  F1 Score:                          {final['F1']:.4f}\n"
    report += "\n"
    
    report += "Interpretation:\n"
    report += "-" * 80 + "\n"
    report += "  - Lower SHD, FDR, MSE are better\n"
    report += "  - Higher TPR, F1 are better\n"
    report += "  - These metrics account for time-varying causal structure\n"
    report += "="*80 + "\n"
    
    return report

