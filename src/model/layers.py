"""
Graph Transformer Layers
========================
These layers are the building blocks of our score network.

WHY A GRAPH TRANSFORMER (not a standard Transformer)?
    Standard transformers treat all tokens equally -- every token attends to
    every other token. But molecules have STRUCTURE: atom A is bonded to
    atom B, not to atom C.  A Graph Transformer injects this structure by
    adding an "edge bias" to the attention scores: bonded atoms get higher
    attention weights.

    Think of it this way:
    - Standard Transformer: "What is every word's relationship to every other word?"
    - Graph Transformer: "What is every atom's relationship to every other atom,
      given that I KNOW which ones are bonded?"

HOW TIME CONDITIONING WORKS:
    Diffusion models need to know "how noisy is the input?" at each step.
    We encode the timestep t as a sinusoidal embedding (same idea as positional
    encoding in transformers) and add it to the node features. This tells the
    network: "at t=999, expect pure noise; at t=1, expect almost-clean data."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar timestep t -> a vector of dimension `dim`.

    Uses sinusoidal frequencies (same trick as Transformer positional encoding).
    Different frequencies let the model distinguish fine-grained time differences.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # MLP to project the raw sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) integer timesteps

        Returns:
            (batch_size, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        # Logarithmically spaced frequencies
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product: each timestep x each frequency
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(embedding)


class GraphTransformerLayer(nn.Module):
    """
    One layer of the Graph Transformer.

    Compared to a standard transformer layer, the key difference is the
    EDGE BIAS: we learn a per-head bias for each bond type (no-bond, single,
    double, triple, aromatic). This modulates attention so bonded atoms
    attend to each other more strongly.

    Architecture per layer:
        1. LayerNorm -> Multi-Head Self-Attention (with edge bias) -> Residual
        2. LayerNorm -> Feed-Forward Network -> Residual
        3. Mask out padding nodes
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_bond_types: int,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias: a learned scalar per (bond_type, attention_head)
        # This is what makes it a GRAPH transformer
        self.edge_bias = nn.Embedding(num_bond_types, num_heads)

        # Feed-forward network (standard transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, D)  node features
            adj:  (B, N, N)  adjacency matrix (integer bond types)
            mask: (B, N)     1.0 for real atoms, 0.0 for padding

        Returns:
            (B, N, D) updated node features
        """
        B, N, D = x.shape

        # ---- Multi-Head Self-Attention with Edge Bias ----
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V and reshape for multi-head
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, num_heads, N, head_dim)

        # Dot-product attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (B, num_heads, N, N)

        # Add edge bias based on bond types
        # adj is (B, N, N) with integer values 0-4
        edge_b = self.edge_bias(adj.long())       # (B, N, N, num_heads)
        edge_b = edge_b.permute(0, 3, 1, 2)       # (B, num_heads, N, N)
        attn = attn + edge_b

        # Mask out padding nodes: if either node i or j is padding, mask the pair
        # mask: (B, N) -> mask_2d: (B, 1, N, N)
        mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
        attn = attn.masked_fill(mask_2d == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked rows
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = residual + self.dropout(out)

        # ---- Feed-Forward Network ----
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        # Zero out padding positions
        x = x * mask.unsqueeze(-1)

        return x
