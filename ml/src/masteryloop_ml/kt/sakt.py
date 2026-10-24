"""SAKT: self-attentive knowledge tracing (FR-5).
The question SAKT answers that BKT cannot: which of the learner's PAST
interactions matter for this next item. BKT carries one scalar per skill and
forgets the order; attention lets the model weight the three questions on the
same sub-concept the learner got wrong last week over the forty unrelated ones
in between.
Two details that are easy to get wrong and silently destroy the model:
 * the query is the NEXT item; the keys and values are the PAST interactions.
 Feeding the next item's correctness into the key stream is the leak, and
 the model then scores 0.99 AUC;
 * the causal mask must be strictly lower-triangular. An off-by-one lets
 position t attend to position t, which is the same leak by another route.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .encoding import PAD # the leak-prone part lives apart; see encoding.py

class Sakt(nn.Module):
    def __init__(self, n_items: int, max_len: int = 200, dim: int = 128,
                 heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.max_len = max_len
        # interaction = (item, correct) pair -> 2*n_items + 1 symbols, PAD at 0
        self.interaction = nn.Embedding(2 * n_items + 1, dim, padding_idx=PAD)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=PAD)
        self.position = nn.Embedding(max_len, dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                               batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim, 1)

    def forward(self, interactions: torch.Tensor,
                next_items: torch.Tensor) -> torch.Tensor:
        """interactions: (B, L) encoded past (item, correct) symbols.
        next_items: (B, L) the item being predicted at each position."""
        b, length = interactions.shape
        pos = torch.arange(length, device=interactions.device)
        keys = self.interaction(interactions) + self.position(pos)[None]
        query = self.item(next_items)
        # strictly upper-triangular mask: position t may attend to < t only
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                       device=interactions.device), diagonal=1)
        padding = interactions == PAD
        attended, _ = self.attention(query, keys, keys, attn_mask=causal,
                                     key_padding_mask=padding, need_weights=False)
        x = self.norm1(attended + query)
        return self.head(self.norm2(self.ffn(x) + x)).squeeze(-1)

@torch.no_grad()
def predict(model: nn.Module, interactions, next_items, device: str = "cpu"):
    model.eval().to(device)
    x = torch.as_tensor(np.asarray(interactions), dtype=torch.long, device=device)
    q = torch.as_tensor(np.asarray(next_items), dtype=torch.long, device=device)
    return torch.sigmoid(model(x, q)).cpu().numpy()
