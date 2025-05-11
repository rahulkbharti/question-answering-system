import torch
from typing import Tuple

import torch.nn as nn

class EntailmentMemory(nn.Module):
    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))  # [K, D]
        self.proj = nn.Linear(hidden_dim, num_slots)  # Maps BART hidden states to memory slots

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [batch_size, hidden_dim] (e.g., BART's [z] token embedding)
        Returns:
            z: Entailment representation [batch_size, hidden_dim]
            attn_weights: Memory attention scores [batch_size, num_slots]
        """
        attn_weights = torch.softmax(self.proj(hidden_state), dim=-1)  # [batch_size, K]
        z = torch.einsum('bk,kd->bd', attn_weights, self.memory)      # [batch_size, D]
        return z, attn_weights
    

