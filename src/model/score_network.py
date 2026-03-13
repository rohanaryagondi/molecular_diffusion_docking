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
    1. Embed atom features (atom_feature_dim -> hidden_dim)
    2. Add time embedding (so the network knows the noise level)
    3. Process through Graph Transformer layers (message passing)
    4. Output head for node noise (hidden_dim -> atom_feature_dim)
    5. Output head for adjacency noise (pairwise -> num_bond_types channels)
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
        atom_feature_dim: int = None,
        num_classes: int = 3,
    ):
        super().__init__()
        # atom_feature_dim defaults to num_atom_types for backward compatibility
        self.atom_feature_dim = atom_feature_dim or num_atom_types
        self.num_bond_types = num_bond_types

        # --- Input Embeddings ---
        self.atom_embed = nn.Linear(self.atom_feature_dim, hidden_dim)

        # Time embedding: scalar t -> vector of size hidden_dim
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # Class label embedding for classifier-free guidance
        # num_classes real classes + 1 null/unconditional token
        self.label_embed = nn.Embedding(num_classes + 1, hidden_dim)

        # --- Graph Transformer Backbone ---
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, num_bond_types, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # --- Output Heads ---
        # Node noise prediction: hidden_dim -> atom_feature_dim
        self.x_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.atom_feature_dim),
        )

        # Adjacency noise prediction: pairwise features -> num_bond_types channels
        # Each bond channel gets its own predicted noise value.
        self.a_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bond_types),
        )

    def forward(
        self,
        X_t: torch.Tensor,
        A_t: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            X_t:    (B, N, atom_feature_dim) noisy node features (continuous)
            A_t:    (B, N, N, num_bond_types) noisy adjacency (continuous, one-hot)
            mask:   (B, N) node mask
            t:      (B,) integer timesteps
            labels: (B,) class labels (0..num_classes-1), or None for unconditional

        Returns:
            eps_X: (B, N, atom_feature_dim) predicted node noise
            eps_A: (B, N, N, num_bond_types) predicted adjacency noise (symmetric)
        """
        B, N, _ = X_t.shape

        # Embed atoms and add time information
        h = self.atom_embed(X_t)                    # (B, N, D)
        t_emb = self.time_embed(t)                   # (B, D)
        h = h + t_emb.unsqueeze(1)                   # broadcast time to all nodes

        # Add class label embedding (classifier-free guidance)
        if labels is not None:
            l_emb = self.label_embed(labels)         # (B, D)
            h = h + l_emb.unsqueeze(1)               # broadcast label to all nodes

        # For the GNN layers, we need integer adjacency for edge bias.
        # A_t is (B, N, N, num_bond_types) continuous -- argmax to get integer type.
        adj_discrete = A_t.argmax(dim=-1)  # (B, N, N)

        # Process through Graph Transformer layers
        for layer in self.layers:
            h = layer(h, adj_discrete, mask)

        h = self.final_norm(h)

        # --- Node noise prediction ---
        eps_X = self.x_output(h)  # (B, N, atom_feature_dim)

        # --- Adjacency noise prediction ---
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        h_pair = torch.cat([h_i, h_j], dim=-1)       # (B, N, N, 2D)

        eps_A = self.a_output(h_pair)  # (B, N, N, num_bond_types)

        # Enforce symmetry: bond noise for (i,j) should equal (j,i)
        eps_A = (eps_A + eps_A.transpose(1, 2)) / 2

        return eps_X, eps_A
