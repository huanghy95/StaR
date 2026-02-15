"""
Temporal Decoupling Score (TDS) Metric

This metric measures a model's ability to maintain correct anomaly scores on victim
nodes even when the causal edge feeding it temporarily disappears.

Key Concept:
- Good models (TGN with memory): Scores stay high even when edge is OFF → Low correlation → TDS ≈ 0.0
- Bad models (no memory): Scores drop when edge is OFF → High correlation → TDS ≈ 1.0
"""

import numpy as np
from scipy.stats import spearmanr
from typing import List, Dict, Optional


def compute_temporal_decoupling_score(
    scores: np.ndarray,
    graphs: np.ndarray,
    labels: np.ndarray,
    timing_info: List[Dict],
    causal_paths: Optional[List[List[tuple]]] = None
) -> Dict[str, float]:
    """
    Compute Temporal Decoupling Score (TDS).
    
    For every known causal path (e.g., Svc3 → Svc1), TDS measures the decoupling
    of a victim node's score from the instantaneous state of the incoming causal edge.
    
    Mathematical Definition:
    -----------------------
    For each causal path (source → victim):
    1. Create Time Series during anomaly window:
       - Edge State Series: E_v[t] = 1 if edge exists, 0 if missing
       - Anomaly Score Series: S_v[t] = anomaly score for victim at time t
    
    2. Calculate Correlation:
       ρ_v = Corr_Spearman(E_v, S_v)
    
    3. Calculate TDS:
       TDS_v = 1 - |ρ_v|  (inverse correlation)
    
    4. Average across all victim nodes:
       TDS = mean(TDS_v)
    
    Interpretation:
    --------------
    - TDS ≈ 1.0: Low correlation → Scores stay high when edge is OFF → Good decoupling
    - TDS ≈ 0.0: High correlation → Scores drop when edge is OFF → Bad (brittle)
    - Original AERCA (no memory): TDS ≈ 0.0 (scores collapse when edge disappears)
    - StaR (with memory): TDS ≈ 1.0 (scores remain high due to memory)
    
    Parameters:
    ----------
    scores : np.ndarray
        Anomaly scores. Shape: (n_samples, n_timesteps, n_nodes)
    graphs : np.ndarray
        Adjacency matrices. Shape: (n_samples, n_timesteps, n_nodes, n_nodes)
    labels : np.ndarray
        Binary labels indicating root cause nodes (1 = root cause).
        Shape: (n_samples, n_nodes)
    timing_info : list
        List of dicts with 'start_time', 'end_time' per sample
    causal_paths : list, optional
        List of known causal paths. If None, inferred from root causes.
        Format: [(source_idx, victim_idx), ...]
    
    Returns:
    -------
    dict : {
        'tds_score': float,  # Overall TDS score
        'tds_per_victim': dict,  # TDS for each victim node
        'n_valid_victims': int,  # Number of victims with valid correlation
    }
    """
    n_samples, n_timesteps, n_nodes = scores.shape
    
    # Store TDS values for each victim node
    tds_values = []
    tds_per_victim = {}
    
    for sample_idx in range(n_samples):
        # Find root cause nodes (sources)
        root_causes = np.where(labels[sample_idx] == 1)[0]
        
        if len(root_causes) == 0:
            continue
        
        # Get anomaly window
        if timing_info and sample_idx < len(timing_info):
            start_time = timing_info[sample_idx].get('start_time', 0)
            end_time = timing_info[sample_idx].get('end_time', n_timesteps)
        else:
            start_time = 0
            end_time = n_timesteps
        
        # Ensure valid window
        start_time = max(0, start_time)
        end_time = min(end_time, n_timesteps)
        
        if start_time >= end_time or end_time - start_time < 3:
            # Need at least 3 timesteps for meaningful correlation
            continue
        
        # Identify causal paths
        if causal_paths is None:
            # Infer paths: for each root cause, find nodes it connects to
            paths = []
            for source in root_causes:
                # Find nodes that receive edges from this source at any time
                for victim in range(n_nodes):
                    if victim == source:
                        continue
                    # Check if edge source→victim exists at any time in window
                    edge_exists_anytime = False
                    for t in range(start_time, end_time):
                        if graphs[sample_idx, t, source, victim] > 0:
                            edge_exists_anytime = True
                            break
                    if edge_exists_anytime:
                        paths.append((source, victim))
        else:
            paths = causal_paths
        
        # For each causal path, compute TDS
        for source, victim in paths:
            # Extract time series for this victim in the anomaly window
            edge_state_series = []
            score_series = []
            
            for t in range(start_time, end_time):
                # Edge state: 1 if exists, 0 if missing
                edge_state = 1.0 if graphs[sample_idx, t, source, victim] > 0 else 0.0
                edge_state_series.append(edge_state)
                
                # Anomaly score for victim
                score = scores[sample_idx, t, victim]
                score_series.append(score)
            
            edge_state_series = np.array(edge_state_series)
            score_series = np.array(score_series)
            
            # Skip if no variation (edge always ON or always OFF)
            if np.std(edge_state_series) < 1e-6 or np.std(score_series) < 1e-6:
                continue
            
            # Skip if edge is always ON (no flickering to test)
            if np.all(edge_state_series == 1):
                continue
            
            # Compute Spearman correlation
            try:
                rho, p_value = spearmanr(edge_state_series, score_series)
                
                if not np.isnan(rho) and not np.isinf(rho):
                    # TDS = 1 - |ρ|
                    # High |ρ| → scores correlate with edge state → bad (brittle)
                    # Low |ρ| → scores independent of edge state → good (decoupled)
                    tds_victim = 1.0 - abs(rho)
                    
                    tds_values.append(tds_victim)
                    
                    # Track per-victim TDS
                    victim_key = f"victim_{victim}"
                    if victim_key not in tds_per_victim:
                        tds_per_victim[victim_key] = []
                    tds_per_victim[victim_key].append(tds_victim)
            except:
                # Skip if correlation computation fails
                continue
    
    # Compute overall TDS
    if len(tds_values) == 0:
        return {
            'tds_score': 0.0,
            'tds_per_victim': {},
            'n_valid_victims': 0,
            'warning': 'No valid victims found for TDS computation'
        }
    
    # Average TDS across all victims
    overall_tds = float(np.mean(tds_values))
    
    # Average per-victim TDS
    tds_per_victim_avg = {
        k: float(np.mean(v)) for k, v in tds_per_victim.items()
    }
    
    return {
        'tds_score': overall_tds,
        'tds_per_victim': tds_per_victim_avg,
        'n_valid_victims': len(tds_values),
        'tds_std': float(np.std(tds_values))
    }


def interpret_tds(tds_score: float) -> str:
    """
    Provide interpretation of TDS score.
    
    Parameters:
    ----------
    tds_score : float
        TDS score between 0 and 1
    
    Returns:
    -------
    str : Interpretation message
    """
    if tds_score >= 0.8:
        return ("Excellent temporal decoupling - model maintains high scores "
                "even when edges disappear (strong temporal memory)")
    elif tds_score >= 0.6:
        return ("Good temporal decoupling - model shows some resilience "
                "to edge disappearance (moderate temporal memory)")
    elif tds_score >= 0.4:
        return ("Moderate temporal decoupling - model partially depends "
                "on edge presence (weak temporal memory)")
    elif tds_score >= 0.2:
        return ("Poor temporal decoupling - model scores strongly correlate "
                "with edge state (minimal temporal memory)")
    else:
        return ("Very poor temporal decoupling - model is brittle and "
                "collapses when edges disappear (no temporal memory)")





