"""
StaR-GC: Stateful Dynamic-Graph Root Cause Analysis
This module implements a TGN-based architecture integrated with Granger causality discovery
for Root Cause Analysis in multivariate time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalMemory(nn.Module):
    """
    Temporal Memory Module for maintaining node states over time.
    Each node has a memory vector that captures its historical behavior.
    """
    def __init__(self, num_nodes: int, memory_dim: int, device: torch.device):
        """
        Args:
            num_nodes: Number of nodes/variables in the system
            memory_dim: Dimensionality of memory vectors
            device: Torch device
        """
        super(TemporalMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.device = device
        
        # Initialize memory vectors for all nodes
        self.memory = nn.Parameter(
            torch.zeros(num_nodes, memory_dim, device=device),
            requires_grad=False
        )
        # Last update timestamp for each node
        self.last_update = nn.Parameter(
            torch.zeros(num_nodes, device=device),
            requires_grad=False
        )
        
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve memory for specified nodes."""
        return self.memory[node_ids]
    
    def set_memory(self, node_ids: torch.Tensor, values: torch.Tensor):
        """Update memory for specified nodes."""
        self.memory[node_ids] = values.detach()
    
    def reset_memory(self):
        """Reset all memory vectors to zero."""
        self.memory.data.fill_(0.0)
        self.last_update.data.fill_(0.0)


class TemporalEncoder(nn.Module):
    """
    Encodes temporal information using a learnable time embedding.
    """
    def __init__(self, time_dim: int):
        super(TemporalEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch_size,) tensor of timestamps
        Returns:
            time_encoding: (batch_size, time_dim) temporal encodings
        """
        # Add dimension for linear layer
        timestamps = timestamps.unsqueeze(-1).float()
        return self.w(timestamps)


class MessageFunction(nn.Module):
    """
    Computes messages between nodes based on their features and memory.
    """
    def __init__(self, node_dim: int, memory_dim: int, time_dim: int, message_dim: int):
        super(MessageFunction, self).__init__()
        input_dim = 2 * node_dim + 2 * memory_dim + time_dim
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
    
    def forward(self, src_features: torch.Tensor, dst_features: torch.Tensor,
                src_memory: torch.Tensor, dst_memory: torch.Tensor,
                time_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute messages between source and destination nodes.
        
        Args:
            src_features: (batch_size, node_dim)
            dst_features: (batch_size, node_dim)
            src_memory: (batch_size, memory_dim)
            dst_memory: (batch_size, memory_dim)
            time_encoding: (batch_size, time_dim)
        Returns:
            messages: (batch_size, message_dim)
        """
        # Concatenate all features
        combined = torch.cat([
            src_features, dst_features,
            src_memory, dst_memory,
            time_encoding
        ], dim=-1)
        return self.message_net(combined)


class MemoryUpdater(nn.Module):
    """
    Updates node memory using GRU based on aggregated messages.
    """
    def __init__(self, memory_dim: int, message_dim: int):
        super(MemoryUpdater, self).__init__()
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(message_dim, memory_dim)
    
    def forward(self, memory: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Update memory using GRU.
        
        Args:
            memory: (num_nodes, memory_dim) current memory
            messages: (num_nodes, message_dim) aggregated messages
        Returns:
            updated_memory: (num_nodes, memory_dim)
        """
        return self.gru(messages, memory)


class GrangerCausalityLayer(nn.Module):
    """
    Neural Granger Causality layer that learns causal relationships.
    This layer predicts how each variable influences others over time.
    """
    def __init__(self, num_vars: int, window_size: int, hidden_dim: int, memory_dim: int, num_hidden_layers: int = 2):
        super(GrangerCausalityLayer, self).__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Causal coefficient networks (one per lag)
        self.causal_nets = nn.ModuleList()
        for k in range(window_size):
            # Build network with configurable depth (like SENN)
            modules = []
            
            # Input layer
            input_dim = num_vars + memory_dim
            modules.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            
            # Hidden layers (configurable depth)
            for j in range(num_hidden_layers - 1):
                modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
            # Output layer
            modules.extend([nn.Linear(hidden_dim, num_vars * num_vars), nn.Tanh()])
            
            net = nn.Sequential(*modules)
            self.causal_nets.append(net)
    
    def forward(self, inputs: torch.Tensor, memory: torch.Tensor, adjacency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictions and causal coefficients.
        
        Args:
            inputs: (batch_size, window_size, num_vars) historical data
            memory: (batch_size, num_vars, memory_dim) node memories
            adjacency_mask: Optional (batch_size, num_vars, num_vars) adjacency mask to filter coefficients.
                          Used ONLY in ablation mode (when memory is disabled) to provide graph structure.
        Returns:
            predictions: (batch_size, num_vars) next-step predictions
            coefficients: (batch_size, window_size, num_vars, num_vars) causal coefficients
        """
        batch_size = inputs.shape[0]
        predictions = torch.zeros(batch_size, self.num_vars, device=inputs.device)
        coeffs_list = []
        
        for k in range(self.window_size):
            # Get features at lag k
            lag_features = inputs[:, k, :]  # (batch_size, num_vars)
            
            # For each variable, compute its influence using its memory
            coeffs_k = []
            for var_idx in range(self.num_vars):
                # Combine features and memory
                var_memory = memory[:, var_idx, :]  # (batch_size, memory_dim)
                combined = torch.cat([lag_features, var_memory], dim=-1)
                
                # Compute causal coefficients
                coeff = self.causal_nets[k](combined)  # (batch_size, num_vars^2)
                coeff = coeff.view(batch_size, self.num_vars, self.num_vars)
                coeffs_k.append(coeff)
            
            coeffs_k = torch.stack(coeffs_k, dim=1)  # (batch_size, num_vars, num_vars, num_vars)
            coeffs_k = coeffs_k.mean(dim=1)  # Average across variables
            
            # Apply adjacency mask ONLY when provided (ablation mode)
            # In normal mode with memory, graph structure is encoded in memory
            if adjacency_mask is not None:
                coeffs_k = coeffs_k * adjacency_mask
            
            coeffs_list.append(coeffs_k)
            
            # Compute contribution to prediction
            # coeffs_k: (batch_size, num_vars, num_vars)
            # lag_features: (batch_size, num_vars)
            predictions += torch.bmm(coeffs_k, lag_features.unsqueeze(-1)).squeeze(-1)
        
        # Stack all coefficients
        coefficients = torch.stack(coeffs_list, dim=1)  # (batch_size, window_size, num_vars, num_vars)
        
        return predictions, coefficients


class StaRGC(nn.Module):
    """
    Stateful Dynamic-Graph Root Cause Analysis (StaR-GC).
    
    This model combines:
    1. Temporal memory to track node states
    2. Message passing for information propagation
    3. Granger causality for causal relationship discovery
    """
    def __init__(self, num_vars: int, window_size: int, hidden_dim: int, 
                 memory_dim: int, time_dim: int, message_dim: int, device: torch.device,
                 num_hidden_layers: int = 2, disable_message_passing: bool = False):
        """
        Args:
            num_vars: Number of variables/nodes
            window_size: Length of historical window (max lag)
            hidden_dim: Hidden layer size for neural networks
            memory_dim: Dimension of memory vectors
            time_dim: Dimension of time encoding
            message_dim: Dimension of messages
            device: Torch device
            num_hidden_layers: Number of hidden layers in Granger causality networks
            disable_message_passing: If True, disable full temporal mechanism (messages+GRU) for ablation study
        """
        super(StaRGC, self).__init__()
        
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.disable_message_passing = disable_message_passing
        
        # Core TGN components
        self.memory = TemporalMemory(num_vars, memory_dim, device)
        self.time_encoder = TemporalEncoder(time_dim)
        self.message_function = MessageFunction(num_vars, memory_dim, time_dim, message_dim)
        self.memory_updater = MemoryUpdater(memory_dim, message_dim)
        
        # Granger causality component with configurable depth
        self.granger_layer = GrangerCausalityLayer(num_vars, window_size, hidden_dim, memory_dim, num_hidden_layers)
        
        # Embedding layer for current observations
        self.embedding = nn.Sequential(
            nn.Linear(num_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim)
        )
    
    def reset_memory(self):
        """Reset all node memories."""
        self.memory.reset_memory()
    
    def forward(self, inputs: torch.Tensor, timestamps: Optional[torch.Tensor] = None, adjacency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through StaR-GC.
        
        Args:
            inputs: (batch_size, window_size, num_vars) historical data
            timestamps: (batch_size, window_size) optional timestamps
            adjacency_mask: Optional (batch_size, num_vars, num_vars) adjacency mask to filter messages.
                          If provided, only messages from connected nodes (mask[i,j] > 0) are included.
        Returns:
            predictions: (batch_size, num_vars) next-step predictions
            coefficients: (batch_size, window_size, num_vars, num_vars) causal coefficients
        """
        batch_size = inputs.shape[0]
        
        # Get current memory for all nodes
        node_ids = torch.arange(self.num_vars, device=self.device)
        current_memory = self.memory.get_memory(node_ids)  # (num_vars, memory_dim)
        
        # Expand memory for batch
        batch_memory = current_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_vars, memory_dim)
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = torch.arange(self.window_size, device=self.device).float()
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        
        # Update memory based on latest observation
        latest_obs = inputs[:, -1, :]  # (batch_size, num_vars)
        
        # Compute messages between all pairs of nodes (ALWAYS respect adjacency_mask for fairness)
        messages = torch.zeros(batch_size, self.num_vars, self.message_function.message_net[0].out_features, 
                             device=self.device)
        
        # Use latest timestamp for message computation
        latest_time = timestamps[:, -1]  # (batch_size,)
        time_encoding = self.time_encoder(latest_time)  # (batch_size, time_dim)
        
        # Aggregate messages for each node
        for i in range(self.num_vars):
            node_messages = []
            for j in range(self.num_vars):
                if i != j:  # Don't send messages to self
                    # Apply adjacency mask: only include message if connection exists
                    if adjacency_mask is not None:
                        # adjacency_mask shape: (batch_size, num_vars, num_vars)
                        # Check if connection j->i exists for each batch item
                        connection_exists = adjacency_mask[:, j, i] > 0  # (batch_size,)
                        if not connection_exists.any():  # Skip if no connections exist in this batch
                            continue
                    
                    src_feat = latest_obs[:, j:j+1]  # (batch_size, 1)
                    dst_feat = latest_obs[:, i:i+1]  # (batch_size, 1)
                    src_mem = batch_memory[:, j, :]  # (batch_size, memory_dim)
                    dst_mem = batch_memory[:, i, :]  # (batch_size, memory_dim)
                    
                    # Expand features to match expected dimensions
                    src_feat_expanded = src_feat.expand(-1, self.num_vars)
                    dst_feat_expanded = dst_feat.expand(-1, self.num_vars)
                    
                    msg = self.message_function(
                        src_feat_expanded, dst_feat_expanded,
                        src_mem, dst_mem, time_encoding
                    )
                    
                    # Apply mask to message if adjacency_mask is provided
                    if adjacency_mask is not None:
                        # Zero out messages where connection doesn't exist
                        mask_value = adjacency_mask[:, j, i].unsqueeze(-1)  # (batch_size, 1)
                        msg = msg * mask_value
                    
                    node_messages.append(msg)
            
        # Aggregate messages (mean pooling)
        if node_messages:
            messages[:, i, :] = torch.stack(node_messages, dim=0).mean(dim=0)
        
        # ABLATION STUDY: Disable message passing (zero out messages)
        if self.disable_message_passing:
            messages = torch.zeros_like(messages)
        
        # ABLATION STUDY: Check if GRU temporal memory update is disabled
        if self.disable_message_passing:
            # ABLATION: Remove the full temporal mechanism (messages + GRU)
            # This tests whether StaR's advantage comes from:
            #   - Temporal graph mechanisms (message passing + GRU memory)
            #   - OR just from increased model capacity (extra memory dimensions)
            #
            # Implementation:
            #   1. Messages are zeroed out (no cross-node information)
            #   2. GRU update is skipped (no temporal propagation)
            #   3. Memory stays at zero throughout training
            #   4. Granger layer gets [lag_features, zero_memory]
            updated_memory = torch.zeros_like(batch_memory)
            # DO NOT update global memory - keep it at zeros throughout training
            # This isolates capacity effect vs. temporal mechanism effect
        else:
            # Update memory for each node in the batch using GRU
            updated_memory = torch.zeros_like(batch_memory)
            for i in range(self.num_vars):
                updated_memory[:, i, :] = self.memory_updater(
                    batch_memory[:, i, :],
                    messages[:, i, :]
                )
            
            # Update global memory (use mean across batch)
            new_memory = updated_memory.mean(dim=0)  # (num_vars, memory_dim)
            self.memory.set_memory(node_ids, new_memory)
        
        # Compute predictions using Granger causality with updated memory
        # CRITICAL: Two different approaches based on whether memory is used
        if self.disable_message_passing:
            # ABLATION MODE (no temporal mechanism): 
            # Messages are zeroed out, memory is zero
            # Apply adjacency_mask to coefficients to still provide graph structure info
            # This isolates: Does temporal mechanism help beyond just graph structure knowledge?
            predictions, coefficients = self.granger_layer(inputs, updated_memory, adjacency_mask)
        else:
            # NORMAL MODE (with temporal mechanism):
            # Messages computed, memory updated via GRU
            # Memory already encodes graph structure through masked message passing
            # Don't mask coefficients - let model use learned temporal patterns
            predictions, coefficients = self.granger_layer(inputs, updated_memory, adjacency_mask=None)
        
        return predictions, coefficients
    
    def get_causal_graph(self, coefficients: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
        """
        Extract causal graph from learned coefficients.
        
        Args:
            coefficients: (batch_size, window_size, num_vars, num_vars)
            quantile: Threshold quantile for determining edges
        Returns:
            causal_graph: (num_vars, num_vars) binary adjacency matrix
        """
        # Take maximum across time lags and batch
        max_coeffs = torch.max(torch.abs(coefficients), dim=1)[0]  # (batch_size, num_vars, num_vars)
        mean_coeffs = max_coeffs.mean(dim=0)  # (num_vars, num_vars)
        
        # Threshold using quantile
        threshold = torch.quantile(mean_coeffs.flatten(), quantile)
        causal_graph = (mean_coeffs >= threshold).float()
        
        return causal_graph


