import torch
from math import log
from scipy.optimize import minimize
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score


def compute_kl_divergence(us, device: torch.device):
    """
    Compute the KL divergence between the empirical distribution of the input samples
    and an isotropic standard Gaussian distribution using PyTorch.

    Parameters:
    samples (Tensor): A 2D tensor with rows as samples and columns as features.

    Returns:
    Tensor: The KL divergence between the empirical distribution of the samples
            and the standard Gaussian distribution.
    """

    # Calculate the empirical mean and covariance matrix of the samples
    mean_p = torch.mean(us, dim=0)
    cov_p = torch.cov(us.t())

    # Dimensionality of the distribution
    d = mean_p.shape[0]

    eigenvalues = torch.linalg.eigvalsh(cov_p)
    condition_number = eigenvalues.max() / eigenvalues.clamp(min=1e-9).min()
    regularization_term = condition_number * 1e-6
    cov_p += torch.eye(d, device=device) * regularization_term
    # Ensure the covariance matrix is full rank
    # cov_p += 1e-9 * torch.eye(d).to(device)

    # Compute the trace term
    trace_term = torch.trace(cov_p)

    # Compute the product of means term (since mean_q is zero, this is just mean_p squared)
    means_term = torch.dot(mean_p, mean_p)

    # # Compute the determinant term
    # log_det_cov_p = torch.logdet(cov_p)
    try:
        L = torch.linalg.cholesky(cov_p)
        log_det_cov_p = 2 * torch.log(torch.diagonal(L)).sum()
    except RuntimeError:
        # Handle the case where Cholesky decomposition fails
        log_det_cov_p = torch.logdet(cov_p)

    # Compute the KL divergence using the formula
    kl_div = means_term + trace_term - d + log_det_cov_p
    if torch.isnan(kl_div).any():
        print('nan')
        print(f'mean_p: {mean_p}')
        print(f'cov_p: {cov_p}')
        print(f'trace_term: {trace_term}')
        print(f'means_term: {means_term}')
        print(f'log_det_cov_p: {log_det_cov_p}')
        print(f'kl_div: {kl_div}')
        raise ValueError('KL divergence is NaN')


    return kl_div


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sliding_window_view_torch(x, window_size: int):
    """
    A function to create a 2D sliding window view of a 2D PyTorch tensor.

    Args:
    x (torch.Tensor): The input 2D tensor.
    window_size (int): Window size.

    Returns:
    torch.Tensor: A tensor with the sliding windows.
    """
    # Calculate output shape
    output_shape = (x.size(0) - window_size + 1, window_size, x.size(1))
    # Calculate strides
    strides = (x.stride(0), x.stride(0), x.stride(1))
    # Create a view
    return x.as_strided(size=output_shape, stride=strides)


def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        hamming_dist = np.sum(np.abs(a_true_offdiag - a_pred_offdiag)) / len(a_true_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        hamming_dist = np.sum(np.abs(a_true.flatten() - a_pred.flatten())) / len(a_true.flatten())
    return accuracy, bal_accuracy, precision, recall, hamming_dist


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        if np.max(a_true_offdiag) == np.min(a_true_offdiag):
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def construct_training_dataset(data, order):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # data: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        time_idx_r = time_idx_r + offset + 200 * (r >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
        offset = np.max(time_idx_r)
    return data_out, response, time_idx

def grimshaw(peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
    ''' The Grimshaw's Trick Method

    The trick of thr Grimshaw's procedure is to reduce the two variables
    optimization problem to a signle variable equation.

    Args:
        peaks: peak nodes from original dataset.
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    '''
    # Handle empty peaks array
    if len(peaks) == 0:
        # Return default values when no peaks are found
        return 0.1, threshold * 0.1  # Small positive gamma and sigma
    
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates

    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)

    candidate_gamma = solve(function=lambda t: function(peaks, t),
                            dev_function=lambda t: dev_function(peaks, t),
                            bounds=(a + epsilon, -epsilon),
                            num_candidates=num_candidates
                            )
    candidate_sigma = solve(function=lambda t: function(peaks, t),
                            dev_function=lambda t: dev_function(peaks, t),
                            bounds=(b, c),
                            num_candidates=num_candidates
                            )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])

    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)

    for candidate in candidates:
        if candidate == 0 or np.isnan(candidate):
            continue
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best


def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1


def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u


def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n


def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(lambda x: obj_function(x, function, dev_function),
                            x0,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=[bounds]*len(x0)
                            )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else:
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood



def pot(data: np.array, risk: float = 1e-2, init_level: float = 0.98, num_candidates: int = 10,
        epsilon: float = 1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

    References:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        z: threshold searching by pot
        t: init threshold
    '''
    # Set init threshold0
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Handle case where no peaks are found
    if len(peaks) == 0:
        # Return a high threshold when no anomalies are detected
        return np.max(data) * 1.1, t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks,
                            threshold=t,
                            num_candidates=num_candidates,
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        z = t - sigma * log(r)

    return z, t

def topk(z_scores, label, threshold, k_range=500):
    ''' Top-k method

    Args:
        us: anomaly scores
        label: ground truth

    Returns:
        k: the number of top-k nodes
    '''
    z_scores = np.array(z_scores)
    us_above_threshold = np.where(z_scores > threshold, z_scores, 0.0)
    label = np.array(label)
    us_above_threshold = us_above_threshold.flatten()
    label = label.flatten()
    ranking = np.argsort(us_above_threshold)
    label_ind = np.where(label == 1)[0]
    k_lst = []
    
    # Safety check: if no anomalous labels, return zeros
    if len(label_ind) == 0:
        print("Warning: No anomalous labels found in topk evaluation. Returning zeros.")
        return np.zeros(k_range)
    
    for k in range(1, k_range+1):
        count = [1 if i in label_ind else 0 for i in ranking[-k:]]
        k_lst.append(sum(count)/min(k, len(label_ind)))
    return np.array(k_lst)

def topk_at_step(scores, labels, k_range=10):
    k_lst = []
    for i in range(len(labels)):
        if sum(labels[i]) > 0:
            ranking = np.argsort(scores[i])
            label_ind = np.where(labels[i] == 1)[0]
            for k in range(1, k_range + 1):
                count = [1 if i in label_ind else 0 for i in ranking[-k:]]
                k_lst.append(sum(count) / min(k, len(label_ind)))
    return np.array(k_lst).reshape(-1, k_range).mean(axis=0)


def compute_gaac(z_scores_list, connection_history_list, labels_list, timing_info_list, window_size=0):
    """
    Compute Graph-Aware Anomaly Contextualization (GAAC) metric.
    
    This metric evaluates how well a model's anomaly scores align with actual
    graph connectivity patterns during anomalous periods. Models that understand
    dynamic graphs should produce higher anomaly scores when root cause services
    are connected to downstream services.
    
    Args:
        z_scores_list: List of z-score arrays, shape (num_samples, T, num_vars)
        connection_history_list: List of connection matrices, shape (num_samples, T-1, num_vars, num_vars)
        labels_list: List of label arrays, shape (num_samples, T, num_vars)
        timing_info_list: List of dicts with 'anomaly_start', 'anomaly_end', 'root_cause_services'
        window_size: Window size used by model (to align timesteps)
        
    Returns:
        float: GAAC score in [0, 1], higher is better
    """
    if connection_history_list is None:
        print("Warning: connection_history_list is None. GAAC metric cannot be computed.")
        return -1.0
    
    cap_scores = []
    tcs_scores = []
    
    for sample_idx, (z_scores, conn_history, labels, timing_info) in enumerate(
        zip(z_scores_list, connection_history_list, labels_list, timing_info_list)
    ):
        # Extract anomaly window and root causes
        anomaly_start = timing_info.get('anomaly_start')
        anomaly_end = timing_info.get('anomaly_end')
        root_causes = timing_info.get('root_cause_services', [])
        
        if anomaly_start is None or anomaly_end is None or len(root_causes) == 0:
            continue
            
        # Adjust for window_size offset (z_scores start at window_size)
        # connection_history starts at t=1, so conn_history[0] corresponds to t=1
        # z_scores[0] corresponds to t=window_size (after sliding window)
        # We need to align them properly
        
        # Anomaly window in original time coordinates
        t_start = max(anomaly_start, window_size + 1)  # Earliest we have z-scores
        t_end = min(anomaly_end, len(z_scores) + window_size)
        
        if t_start >= t_end:
            continue
        
        # Convert to indices in z_scores and conn_history arrays
        z_start_idx = t_start - window_size
        z_end_idx = t_end - window_size
        conn_start_idx = t_start - 1  # connection_history[0] = t=1
        conn_end_idx = t_end - 1
        
        # Ensure indices are valid
        z_start_idx = max(0, z_start_idx)
        z_end_idx = min(len(z_scores), z_end_idx)
        conn_start_idx = max(0, conn_start_idx)
        conn_end_idx = min(len(conn_history), conn_end_idx)
        
        if z_start_idx >= z_end_idx or conn_start_idx >= conn_end_idx:
            continue
        
        # Get anomaly window data
        anom_z_scores = z_scores[z_start_idx:z_end_idx]  # (T_anom, num_vars)
        anom_connections = conn_history[conn_start_idx:conn_end_idx]  # (T_anom, num_vars, num_vars)
        
        # Ensure they have the same length (handle edge cases)
        min_len = min(len(anom_z_scores), len(anom_connections))
        anom_z_scores = anom_z_scores[:min_len]
        anom_connections = anom_connections[:min_len]
        
        if min_len == 0:
            continue
        
        # 1. Connection-Aware Precision (CAP)
        # For each timestep, compute graph-conditioned anomaly scores
        cap_timestep_scores = []
        for t_idx in range(min_len):
            z_t = anom_z_scores[t_idx]  # (num_vars,)
            conn_t = anom_connections[t_idx]  # (num_vars, num_vars)
            
            # Check which variables are connected (have any incoming connections)
            is_connected = np.sum(conn_t, axis=0) > 0  # (num_vars,)
            
            # Graph-conditioned anomaly scores (only for connected variables)
            gcas = z_t * is_connected
            
            # Numerator: sum of connected root cause scores
            root_cause_mask = np.zeros(len(z_t))
            root_cause_mask[root_causes] = 1
            numerator = np.sum(gcas * root_cause_mask)
            
            # Denominator: sum of all connected variable scores
            denominator = np.sum(gcas) + 1e-8
            
            cap_t = numerator / denominator
            cap_timestep_scores.append(cap_t)
        
        cap_sample = np.mean(cap_timestep_scores) if cap_timestep_scores else 0.0
        cap_scores.append(cap_sample)
        
        # 2. Temporal Consistency Score (TCS)
        # Correlation between number of root causes connected and their z-scores
        num_rc_connected = []
        sum_rc_zscores = []
        
        for t_idx in range(min_len):
            z_t = anom_z_scores[t_idx]
            conn_t = anom_connections[t_idx]
            
            # Count how many root causes are connected
            rc_connected_count = 0
            for rc in root_causes:
                if np.sum(conn_t[rc, :]) > 0:  # Root cause has outgoing connections
                    rc_connected_count += 1
            
            num_rc_connected.append(rc_connected_count)
            
            # Sum of root cause z-scores
            rc_zscore_sum = np.sum(z_t[root_causes])
            sum_rc_zscores.append(rc_zscore_sum)
        
        # Compute Pearson correlation
        if len(num_rc_connected) > 1 and np.std(num_rc_connected) > 0 and np.std(sum_rc_zscores) > 0:
            tcs_sample = np.corrcoef(num_rc_connected, sum_rc_zscores)[0, 1]
            # Only consider positive correlations (negative means inverse relationship, which is bad)
            tcs_sample = max(0.0, tcs_sample)
        else:
            tcs_sample = 0.0
        
        tcs_scores.append(tcs_sample)
    
    # Compute final GAAC score
    if len(cap_scores) == 0:
        print("Warning: No valid samples for GAAC computation.")
        return 0.0
    
    avg_cap = np.mean(cap_scores)
    avg_tcs = np.mean(tcs_scores) if tcs_scores else 0.0
    
    # Weighted combination: 70% CAP, 30% TCS
    gaac_score = 0.7 * avg_cap + 0.3 * avg_tcs
    
    return float(gaac_score)


# ==================== NEW DYNAMIC GRAPH METRICS ====================

def prepare_scores_for_metrics(
    z_scores_list: list,
    timing_info_list: list,
    max_timesteps: int = None
) -> np.ndarray:
    """
    Convert variable-length z_scores to fixed-size array for metrics.
    
    Parameters:
    -----------
    z_scores_list : list
        List of arrays, each shape (timesteps_i, n_nodes)
    timing_info_list : list
        List of dicts with timing information
    max_timesteps : int, optional
        Maximum timesteps for padding
    
    Returns:
    --------
    np.ndarray : Shape (n_samples, max_timesteps, n_nodes)
    """
    n_samples = len(z_scores_list)
    
    if n_samples == 0:
        return np.array([])
    
    n_nodes = z_scores_list[0].shape[1]
    
    if max_timesteps is None:
        max_timesteps = max(z.shape[0] for z in z_scores_list)
    
    # Initialize with zeros
    scores_array = np.zeros((n_samples, max_timesteps, n_nodes))
    
    for i, z_scores in enumerate(z_scores_list):
        n_timesteps = min(z_scores.shape[0], max_timesteps)
        scores_array[i, :n_timesteps, :] = z_scores[:n_timesteps, :]
    
    return scores_array


def compute_temporally_aware_path_awareness(
    scores: np.ndarray,
    graphs: np.ndarray,
    labels: np.ndarray,
    timing_info: list,
    k_values: list = None
) -> dict:
    """
    Compute Temporally-Aware Path Awareness at k (TPA@k).
    
    An improved version of CPA@k that specifically measures the model's memory-bridging
    capability by identifying reachable nodes based on a temporally-aggregated graph
    that represents all paths the model "could have learned" during the anomaly window.
    
    Mathematical Definition:
    ------------------------
    For each anomaly window (t_start to t_end):
    1. Identify Sources (S): The set of true root cause nodes
    2. Create Aggregated Graph (G_agg): For each edge (u, v), it exists in G_agg 
       if it existed in ANY timestep within that window:
       G_agg = Union_{t=t_start}^{t_end} G_t
    3. Find Temporally-Reachable Set (R_temp): Compute the set of all nodes reachable 
       from any source s ∈ S within G_agg. This represents the full "causal blast radius" 
       over time.
    4. Calculate TPA@k: At each timestep t in the window, get the model's TopK_Nodes(t).
       Compare this to the static R_temp:
       TPA@k_t = |TopK_Nodes(t) ∩ R_temp| / k
    5. The final score is the average TPA@k_t across all timesteps in all anomaly windows.
    
    Why This Works:
    ---------------
    - StaR: At time t, the edge Svc3 -> Svc1 is missing. The TGN "remembers" the path
      and scores Svc1 high (it's in TopK_Nodes(t)). Since Svc1 is in R_temp (because the 
      edge existed at t-1), TGN gets a high score. (Correctly rewarded)
      
    - Original AERCA: At time t, no edge = complete loss of causal signal. Svc1 scores low
      (not in TopK_Nodes(t)). The metric gives low score. (Correctly penalized)
    
    This metric directly measures if the model can use temporal memory to identify nodes
    on a causal path, even when the path is instantaneously broken.
    
    Parameters:
    -----------
    scores : np.ndarray
        Anomaly scores. Shape: (n_samples, n_timesteps, n_nodes)
    graphs : np.ndarray
        Adjacency matrices. Shape: (n_samples, n_timesteps, n_nodes, n_nodes)
    labels : np.ndarray
        Binary labels indicating root cause nodes (1 = root cause).
        Shape: (n_samples, n_nodes)
    timing_info : list
        List of dicts with 'start_time', 'end_time' per sample
    k_values : list, optional
        List of k values to compute TPA@k for. Default: [1, 3, 5, 10]
    
    Returns:
    --------
    dict : TPA@k scores for different k values (keys: 'k=1', 'k=3', 'k=5', 'k=10')
    
    Notes:
    ------
    - Unlike CPA@k which checks instantaneous reachability at each timestep,
      TPA@k checks reachability in the temporally-aggregated graph
    - This directly measures the model's ability to "remember" and "bridge" 
      missing edges using temporal information
    - Higher TPA@k indicates better temporal memory and path awareness
    """
    n_samples, n_timesteps, n_nodes = scores.shape
    
    # Default k values to match AC@k metrics
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    tpa_results = {f'k={kval}': [] for kval in k_values}
    
    # Helper function for BFS to find all reachable nodes
    def bfs_reachable(graph, sources):
        """BFS to find all nodes reachable from any source node."""
        visited = set()
        queue = list(sources)
        visited.update(sources)
        idx = 0
        
        while idx < len(queue):
            node = queue[idx]
            idx += 1
            
            # Find neighbors in the directed graph
            for neighbor in range(n_nodes):
                if graph[node, neighbor] > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited
    
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
        
        if start_time >= end_time:
            continue
        
        # Step 2: Create temporally-aggregated graph (G_agg)
        # Union of all edges that existed in ANY timestep within the anomaly window
        g_agg = np.zeros((n_nodes, n_nodes))
        for t in range(start_time, end_time):
            g_agg = np.maximum(g_agg, graphs[sample_idx, t])
        
        # Step 3: Find temporally-reachable set (R_temp)
        # All nodes reachable from any root cause in the aggregated graph
        reachable_temp = bfs_reachable(g_agg, root_causes)
        
        # If no nodes are reachable (isolated root causes), skip
        if len(reachable_temp) == 0:
            continue
        
        # Step 4: Calculate TPA@k at each timestep in the anomaly window
        for t in range(start_time, end_time):
            # Rank nodes by anomaly scores at this timestep
            scores_t = scores[sample_idx, t, :]
            ranked_nodes = np.argsort(-scores_t)  # Descending order
            
            # Compute TPA@k for different k values
            for kval in k_values:
                actual_k = min(kval, n_nodes)
                top_k = set(ranked_nodes[:actual_k])
                
                # Count how many top-k nodes are in the temporally-reachable set
                overlap = len(top_k & reachable_temp)
                tpa_results[f'k={kval}'].append(overlap / actual_k)
    
    # Step 5: Average across all timesteps and samples
    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in tpa_results.items()}

