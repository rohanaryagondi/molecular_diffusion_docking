"""
Score Network (Noise Prediction Network)
=========================================
This is the neural network at the heart of the diffusion model.

WHAT DOES IT DO?
    Given a noisy molecular graph (X_t, A_t) and a timestep t, it predicts
    the noise (epsilon) that was added to the clean graph (X_0, A_0).

    Inputs:  noisy atoms X_t, noisy adjacency A_t, timestep t
    Outputs: predicted noise for atoms (eps_X), predicted noise for bonds (eps_A)

WHY PREDICT NOISE?
    This is the "epsilon-prediction" parameterization from DDPM (Ho et al., 2020).
    The training loss is simply:
        L = ||epsilon_true - epsilon_predicted||^2

    This is equivalent to learning the SCORE FUNCTION (gradient of log probability)
    at each noise level. The connection:
        score(x_t) = -epsilon_theta(x_t, t) / sqrt(1 - alpha_bar_t)

ARCHITECTURE:
    1. Embed atom features (one-hot -> hidden_dim)
    2. Add time embedding (so the network knows the noise level)
    3. Process through Graph Transformer layers (message passing)
    4. Output head for node noise (linear projection)
    5. Output head for adjacency noise (pairwise node features -> scalar)
"""

import torch
import torch.nn as nn

from src.model.layers import SinusoidalTimeEmbedding, GraphTransformerLayer


class ScoreNetwork(nn.Module):

    def __init__(
        self,
        num_atom_types: int = 10,
        num_bond_types: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Input Embeddings ---
        # Map one-hot atom features to hidden dimension
        self.atom_embed = nn.Linear(num_atom_types, hidden_dim)

        # Time embedding: scalar t -> vector of size hidden_dim
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # --- Graph Transformer Backbone ---
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, num_bond_types, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # --- Output Heads ---
        # Node noise prediction: hidden_dim -> num_atom_types
        self.x_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

        # Adjacency noise prediction: pairwise features -> scalar
        # For each pair (i, j), concatenate node features h_i and h_j,
        # then project to a single value (the predicted bond noise).
        self.a_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        X_t: torch.Tensor,
        A_t: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Args:
            X_t:  (B, N, num_atom_types) noisy node features (continuous)
            A_t:  (B, N, N) noisy adjacency (continuous)
            mask: (B, N) node mask
            t:    (B,) integer timesteps

        Returns:
            eps_X: (B, N, num_atom_types) predicted node noise
            eps_A: (B, N, N) predicted adjacency noise (symmetric)
        """
        B, N, _ = X_t.shape

        # Embed atoms and add time information
        h = self.atom_embed(X_t)                    # (B, N, D)
        t_emb = self.time_embed(t)                   # (B, D)
        h = h + t_emb.unsqueeze(1)                   # broadcast time to all nodes

        # For the GNN layers, we need integer adjacency for edge bias.
        # During training: A_t is noisy (continuous), so we round it for
        # the edge bias lookup, but the continuous A_t is what we're denoising.
        adj_discrete = A_t.round().long().clamp(0, 4)

        # Process through Graph Transformer layers
        for layer in self.layers:
            h = layer(h, adj_discrete, mask)

        h = self.final_norm(h)

        # --- Node noise prediction ---
        eps_X = self.x_output(h)  # (B, N, num_atom_types)

        # --- Adjacency noise prediction ---
        # Create pairwise features by concatenating h_i and h_j
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        h_pair = torch.cat([h_i, h_j], dim=-1)       # (B, N, N, 2D)

        eps_A = self.a_output(h_pair).squeeze(-1)    # (B, N, N)

        # Enforce symmetry: the adjacency noise for (i,j) should equal (j,i)
        eps_A = (eps_A + eps_A.transpose(1, 2)) / 2

        return eps_X, eps_A
