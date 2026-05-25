"""Global memory cross-attention block.

A small set of learned memory tokens attend to the projected coarse feature
tokens (and vice versa) before being broadcast back into a residual added to
the projected feature map. Used by the A3 / staged readout heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_memory_tokens: int = 8,
        num_heads: int = 4,
        memory_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_memory_tokens = int(num_memory_tokens)
        self.memory_tokens = nn.Parameter(
            torch.empty(self.num_memory_tokens, self.embed_dim)
        )
        nn.init.normal_(self.memory_tokens, mean=0.0, std=memory_init_std)
        self.norm_tokens = nn.LayerNorm(self.embed_dim)
        self.norm_memory = nn.LayerNorm(self.embed_dim)
        self.mem_to_tokens = nn.MultiheadAttention(
            self.embed_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features`` has shape ``[B, C, H, W]``; returns the same shape."""

        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens_n = self.norm_tokens(tokens)
        memory = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)
        memory_n = self.norm_memory(memory)
        # Tokens attend to memory
        attended, _ = self.mem_to_tokens(query=tokens_n, key=memory_n, value=memory_n)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm_tokens(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)
