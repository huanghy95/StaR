"""
StaR-GC with Flexible Memory Mechanisms (for ablation studies)

This module extends StaR-GC to support different memory update mechanisms:
- GRU (default)
- LSTM
- Transformer
- MLP (no recurrence)

This enables ablation studies to determine which memory mechanism works best for RCA.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from models.memory_updaters import create_memory_updater


class TemporalMemory(nn.Module):
    """
    Temporal Memory Module for maintaining node states over time.
    Each node has a memory vector that captures its historical behavior.
    """
    def __init__(self, num_nodes: int, memory_dim: int, device: torch.device):
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
    """Encodes temporal information using a learnable time embedding."""
    def __init__(self, time_dim: int):
        super(TemporalEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.unsqueeze(-1).float()
        return self.w(timestamps)


class MessageFunction(nn.Module):
    """Computes messages between nodes based on their features and memory."""
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
        combined = torch.cat([
            src_features, dst_features,
            src_memory, dst_memory,
            time_encoding
        ], dim=-1)
        return self.message_net(combined)


class GrangerCausalityLayer(nn.Module):
    """Neural Granger Causality layer that learns causal relationships."""
    def __init__(self, num_vars: int, window_size: int, hidden_dim: int, 
                 memory_dim: int, num_hidden_layers: int = 2):
        super(GrangerCausalityLayer, self).__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Causal coefficient networks (one per lag)
        self.causal_nets = nn.ModuleList()
        for k in range(window_size):
            modules = []
            input_dim = num_vars + memory_dim
            modules.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            
            for j in range(num_hidden_layers - 1):
                modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
            modules.extend([nn.Linear(hidden_dim, num_vars * num_vars), nn.Tanh()])
            net = nn.Sequential(*modules)
            self.causal_nets.append(net)
    
    def forward(self, inputs: torch.Tensor, memory: torch.Tensor, 
                adjacency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        predictions = torch.zeros(batch_size, self.num_vars, device=inputs.device)
        coeffs_list = []
        
        for k in range(self.window_size):
            lag_features = inputs[:, k, :]
            coeffs_k = []
            
            for var_idx in range(self.num_vars):
                var_memory = memory[:, var_idx, :]
                combined = torch.cat([lag_features, var_memory], dim=-1)
                coeff = self.causal_nets[k](combined)
                coeff = coeff.view(batch_size, self.num_vars, self.num_vars)
                coeffs_k.append(coeff)
            
            coeffs_k = torch.stack(coeffs_k, dim=1)
            coeffs_k = coeffs_k.mean(dim=1)
            
            if adjacency_mask is not None:
                coeffs_k = coeffs_k * adjacency_mask
            
            coeffs_list.append(coeffs_k)
            predictions += torch.bmm(coeffs_k, lag_features.unsqueeze(-1)).squeeze(-1)
        
        coefficients = torch.stack(coeffs_list, dim=1)
        return predictions, coefficients


class StaRGC_Flexible(nn.Module):
    """
    StaR-GC with flexible memory update mechanisms for ablation studies.
    
    Supports different memory updaters: GRU, LSTM, Transformer, MLP
    """
    def __init__(self, num_vars: int, window_size: int, hidden_dim: int, 
                 memory_dim: int, time_dim: int, message_dim: int, device: torch.device,
                 num_hidden_layers: int = 2, disable_message_passing: bool = False,
                 memory_updater_type: str = 'gru', num_attention_heads: int = 4):
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
            disable_message_passing: If True, disable full temporal mechanism
            memory_updater_type: Type of memory updater ('gru', 'lstm', 'transformer', 'mlp')
            num_attention_heads: Number of heads for transformer updater
        """
        super(StaRGC_Flexible, self).__init__()
        
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.disable_message_passing = disable_message_passing
        self.memory_updater_type = memory_updater_type
        
        # Core TGN components
        self.memory = TemporalMemory(num_vars, memory_dim, device)
        self.time_encoder = TemporalEncoder(time_dim)
        self.message_function = MessageFunction(num_vars, memory_dim, time_dim, message_dim)
        
        # Flexible memory updater
        self.memory_updater = create_memory_updater(
            memory_updater_type, memory_dim, message_dim, num_attention_heads
        )
        
        # Granger causality component
        self.granger_layer = GrangerCausalityLayer(
            num_vars, window_size, hidden_dim, memory_dim, num_hidden_layers
        )
        
        # Embedding layer for current observations
        self.embedding = nn.Sequential(
            nn.Linear(num_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim)
        )
    
    def reset_memory(self):
        """Reset all node memories."""
        self.memory.reset_memory()
        # Reset LSTM cell state if using LSTM
        if hasattr(self.memory_updater, 'reset_cell_state'):
            self.memory_updater.reset_cell_state(self.num_vars, self.device)
    
    def forward(self, inputs: torch.Tensor, timestamps: Optional[torch.Tensor] = None,
                adjacency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through StaR-GC with flexible memory updater.
        
        Args:
            inputs: (batch_size, window_size, num_vars) historical data
            timestamps: (batch_size, window_size) optional timestamps
            adjacency_mask: Optional (batch_size, num_vars, num_vars) adjacency mask
        
        Returns:
            predictions: (batch_size, num_vars) next-step predictions
            coefficients: (batch_size, window_size, num_vars, num_vars) causal coefficients
        """
        batch_size = inputs.shape[0]
        
        # Get current memory for all nodes
        node_ids = torch.arange(self.num_vars, device=self.device)
        current_memory = self.memory.get_memory(node_ids)
        batch_memory = current_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = torch.arange(self.window_size, device=self.device).float()
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        
        # Update memory based on latest observation
        latest_obs = inputs[:, -1, :]
        
        # Compute messages between all pairs of nodes
        messages = torch.zeros(batch_size, self.num_vars, 
                             self.message_function.message_net[0].out_features,
                             device=self.device)
        
        latest_time = timestamps[:, -1]
        time_encoding = self.time_encoder(latest_time)
        
        # Aggregate messages for each node
        for i in range(self.num_vars):
            node_messages = []
            for j in range(self.num_vars):
                if i != j:
                    # Apply adjacency mask if provided
                    if adjacency_mask is not None:
                        connection_exists = adjacency_mask[:, j, i] > 0
                        if not connection_exists.any():
                            continue
                    
                    src_feat = latest_obs[:, j:j+1]
                    dst_feat = latest_obs[:, i:i+1]
                    src_mem = batch_memory[:, j, :]
                    dst_mem = batch_memory[:, i, :]
                    
                    src_feat_expanded = src_feat.expand(-1, self.num_vars)
                    dst_feat_expanded = dst_feat.expand(-1, self.num_vars)
                    
                    msg = self.message_function(
                        src_feat_expanded, dst_feat_expanded,
                        src_mem, dst_mem, time_encoding
                    )
                    
                    if adjacency_mask is not None:
                        mask_value = adjacency_mask[:, j, i].unsqueeze(-1)
                        msg = msg * mask_value
                    
                    node_messages.append(msg)
            
            if node_messages:
                messages[:, i, :] = torch.stack(node_messages, dim=0).mean(dim=0)
        
        # ABLATION STUDY: Disable message passing
        if self.disable_message_passing:
            messages = torch.zeros_like(messages)
            updated_memory = torch.zeros_like(batch_memory)
        else:
            # Update memory using the specified updater type
            updated_memory = torch.zeros_like(batch_memory)
            for i in range(self.num_vars):
                updated_memory[:, i, :] = self.memory_updater(
                    batch_memory[:, i, :],
                    messages[:, i, :]
                )
            
            # Update global memory
            new_memory = updated_memory.mean(dim=0)
            self.memory.set_memory(node_ids, new_memory)
        
        # Compute predictions using Granger causality
        if self.disable_message_passing:
            predictions, coefficients = self.granger_layer(inputs, updated_memory, adjacency_mask)
        else:
            predictions, coefficients = self.granger_layer(inputs, updated_memory, adjacency_mask=None)
        
        return predictions, coefficients
    
    def get_causal_graph(self, coefficients: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
        """Extract causal graph from learned coefficients."""
        max_coeffs = torch.max(torch.abs(coefficients), dim=1)[0]
        mean_coeffs = max_coeffs.mean(dim=0)
        threshold = torch.quantile(mean_coeffs.flatten(), quantile)
        causal_graph = (mean_coeffs >= threshold).float()
        return causal_graph


