import torch
from typing import Tuple

import torch.nn as nn

class DiscourseMemory(nn.Module):
    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))  # [L, D]
        self.proj = nn.Linear(hidden_dim, num_slots)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as ERM but for discourse features."""
        attn_weights = torch.softmax(self.proj(hidden_state), dim=-1)  # [batch_size, L]
        z_d = torch.einsum('bl,ld->bd', attn_weights, self.memory)     # [batch_size, D]
        return z_d, attn_weights