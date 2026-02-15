"""
Alternative Memory Updater Mechanisms for TGN Ablation Studies

This module provides different memory update mechanisms (GRU, LSTM, Transformer)
to test which temporal memory mechanism works best for RCA tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRUMemoryUpdater(nn.Module):
    """
    Updates node memory using GRU based on aggregated messages.
    This is the default mechanism used in TGN.
    """
    def __init__(self, memory_dim: int, message_dim: int):
        super(GRUMemoryUpdater, self).__init__()
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


class LSTMMemoryUpdater(nn.Module):
    """
    Updates node memory using LSTM based on aggregated messages.
    LSTM has additional cell state which might help capture long-term dependencies.
    """
    def __init__(self, memory_dim: int, message_dim: int):
        super(LSTMMemoryUpdater, self).__init__()
        self.memory_dim = memory_dim
        self.lstm = nn.LSTMCell(message_dim, memory_dim)
        # Cell state (long-term memory)
        self.cell_state = None
    
    def reset_cell_state(self, num_nodes: int, device: torch.device):
        """Reset LSTM cell state."""
        self.cell_state = torch.zeros(num_nodes, self.memory_dim, device=device)
    
    def forward(self, memory: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Update memory using LSTM.
        
        Args:
            memory: (num_nodes, memory_dim) current hidden state
            messages: (num_nodes, message_dim) aggregated messages
        Returns:
            updated_memory: (num_nodes, memory_dim) new hidden state
        """
        if self.cell_state is None or self.cell_state.shape[0] != memory.shape[0]:
            self.cell_state = torch.zeros_like(memory)
        
        new_hidden, new_cell = self.lstm(messages, (memory, self.cell_state))
        self.cell_state = new_cell.detach()  # Store cell state for next update
        return new_hidden


class TransformerMemoryUpdater(nn.Module):
    """
    Updates node memory using Transformer-based attention mechanism.
    This allows the model to attend to relevant parts of the message.
    """
    def __init__(self, memory_dim: int, message_dim: int, num_heads: int = 4):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Project messages to memory dimension if needed
        self.message_proj = nn.Linear(message_dim, memory_dim) if message_dim != memory_dim else nn.Identity()
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(memory_dim, memory_dim * 4),
            nn.ReLU(),
            nn.Linear(memory_dim * 4, memory_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(memory_dim)
        self.norm2 = nn.LayerNorm(memory_dim)
    
    def forward(self, memory: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Update memory using Transformer.
        
        Args:
            memory: (num_nodes, memory_dim) current memory
            messages: (num_nodes, message_dim) aggregated messages
        Returns:
            updated_memory: (num_nodes, memory_dim)
        """
        # Project messages to memory dimension
        messages_proj = self.message_proj(messages)
        
        # Combine memory and messages (treat as sequence)
        # Shape: (batch=1, seq_len=num_nodes, embed_dim=memory_dim)
        combined = torch.stack([memory, messages_proj], dim=1)  # (num_nodes, 2, memory_dim)
        
        # Self-attention (batch_first=True)
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Take the first position (corresponding to memory)
        attn_output = attn_output[:, 0, :]  # (num_nodes, memory_dim)
        
        # Residual connection and layer norm
        memory_updated = self.norm1(memory + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(memory_updated)
        
        # Residual connection and layer norm
        memory_final = self.norm2(memory_updated + ffn_output)
        
        return memory_final


class MLPMemoryUpdater(nn.Module):
    """
    Simple MLP-based memory updater (baseline without recurrent mechanisms).
    This tests whether temporal recurrence is actually necessary.
    """
    def __init__(self, memory_dim: int, message_dim: int):
        super(MLPMemoryUpdater, self).__init__()
        self.memory_dim = memory_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim + message_dim, memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh()  # Bounded output
        )
    
    def forward(self, memory: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Update memory using MLP.
        
        Args:
            memory: (num_nodes, memory_dim) current memory
            messages: (num_nodes, message_dim) aggregated messages
        Returns:
            updated_memory: (num_nodes, memory_dim)
        """
        combined = torch.cat([memory, messages], dim=-1)
        return self.mlp(combined)


def create_memory_updater(updater_type: str, memory_dim: int, message_dim: int, 
                          num_heads: int = 4) -> nn.Module:
    """
    Factory function to create different types of memory updaters.
    
    Args:
        updater_type: Type of updater ('gru', 'lstm', 'transformer', 'mlp')
        memory_dim: Dimension of memory vectors
        message_dim: Dimension of messages
        num_heads: Number of attention heads (for transformer only)
    
    Returns:
        Memory updater module
    """
    if updater_type.lower() == 'gru':
        return GRUMemoryUpdater(memory_dim, message_dim)
    elif updater_type.lower() == 'lstm':
        return LSTMMemoryUpdater(memory_dim, message_dim)
    elif updater_type.lower() == 'transformer':
        return TransformerMemoryUpdater(memory_dim, message_dim, num_heads)
    elif updater_type.lower() == 'mlp':
        return MLPMemoryUpdater(memory_dim, message_dim)
    else:
        raise ValueError(f"Unknown updater type: {updater_type}. "
                        f"Choose from: 'gru', 'lstm', 'transformer', 'mlp'")


