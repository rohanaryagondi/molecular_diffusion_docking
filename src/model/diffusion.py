"""
Gaussian Diffusion Process (DDPM)
=================================
This implements the forward and reverse diffusion processes for molecular graphs.

THE BIG PICTURE:
    1. FORWARD (training): Take a clean molecule, add noise over T steps.
       At t=0 it's the real molecule; at t=T it's pure Gaussian noise.
       Crucially, we can jump to ANY timestep directly (no need to go step-by-step):
           x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    2. REVERSE (generation): Start from pure noise, remove noise step by step.
       At each step, our score network predicts what noise was added, and we
       subtract it (with some math to account for the stochastic process).

THE MATH:
    Forward:  q(x_t | x_0) = N(x_t; sqrt(ā_t) * x_0,  (1 - ā_t) * I)
    Reverse:  p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta, sigma_t^2 * I)

    where: mu_theta = (1/sqrt(a_t)) * (x_t - (b_t/sqrt(1-ā_t)) * eps_theta)
    and:   a_t = 1 - b_t,  ā_t = product(a_1 ... a_t),  b_t = noise schedule

CONNECTION TO SCORE-BASED MODELS:
    The score function is: s(x_t) = nabla_x log p(x_t)
    Our network approximates: s_theta(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - ā_t)
    So epsilon-prediction IS score estimation. This is why DDPM is a
    "score-based generative model."

MOLECULAR SPECIFICS:
    We apply diffusion jointly to both node features X and adjacency A.
    Both are treated as continuous during diffusion, then discretized at the end:
      - X: argmax -> atom type
      - A: round -> bond type (0 = no bond, 1-4 = bond types)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion:
    """
    Manages the diffusion noise schedule and sampling procedures.

    This is NOT a nn.Module -- it's a utility class that works with any
    score network. Think of it as the "diffusion engine" that the score
    network plugs into.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # ---- Noise Schedule ----
        # Linear schedule: beta increases linearly from beta_start to beta_end.
        # Small beta at start = gentle noise; large beta at end = aggressive noise.
        betas = torch.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # cumulative product

        # Pre-compute useful quantities (avoids recomputation during training)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(device)
        self.sqrt_alphas = torch.sqrt(alphas).to(device)

    def to(self, device):
        """Move all schedule tensors to a device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        return self

    # ---- Forward Process (Adding Noise) ----

    def q_sample(self, x_0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0) -- the forward process.

        This is the "closed-form" sampling: we can jump directly to any
        timestep t without iterating through 0, 1, ..., t-1.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0:   (B, ...) clean data (any shape)
            t:     (B,) integer timesteps
            noise: (B, ...) optional pre-generated noise

        Returns:
            x_t:   (B, ...) noisy data
            noise: (B, ...) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Reshape schedule values for broadcasting with arbitrary data shapes
        shape = [-1] + [1] * (x_0.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].reshape(shape)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].reshape(shape)

        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
        return x_t, noise

    # ---- Training Loss ----

    def training_loss(self, model, X_0, A_0, mask, adj_loss_weight=1.0):
        """
        Compute the denoising score matching loss for one batch.

        Steps:
            1. Sample random timesteps
            2. Add noise to X_0 and A_0
            3. Predict the noise using the score network
            4. Compute MSE loss (only on real atoms/bonds, not padding)

        Args:
            model:  ScoreNetwork instance
            X_0:    (B, N, F) clean node features
            A_0:    (B, N, N) clean adjacency
            mask:   (B, N) node mask
            adj_loss_weight: relative weight for adjacency loss

        Returns:
            total_loss, loss_X, loss_A
        """
        B = X_0.shape[0]
        device = X_0.device

        # 1. Random timesteps for each sample in the batch
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # 2. Add noise to both X and A
        X_t, noise_X = self.q_sample(X_0, t)
        A_t, noise_A = self.q_sample(A_0, t)

        # Symmetrize adjacency noise (since A is symmetric)
        noise_A = (noise_A + noise_A.transpose(1, 2)) / 2
        A_t = (A_t + A_t.transpose(1, 2)) / 2

        # 3. Predict noise
        pred_noise_X, pred_noise_A = model(X_t, A_t, mask, t)

        # 4. Compute masked MSE loss
        # Node loss: only on real atoms
        node_mask = mask.unsqueeze(-1)  # (B, N, 1)
        loss_X = F.mse_loss(
            pred_noise_X * node_mask,
            noise_X * node_mask,
            reduction="sum",
        ) / node_mask.sum().clamp(min=1)

        # Adjacency loss: only on real atom pairs
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        loss_A = F.mse_loss(
            pred_noise_A * edge_mask,
            noise_A * edge_mask,
            reduction="sum",
        ) / edge_mask.sum().clamp(min=1)

        total_loss = loss_X + adj_loss_weight * loss_A
        return total_loss, loss_X.item(), loss_A.item()

    # ---- Reverse Process (Generation) ----

    @torch.no_grad()
    def p_sample_step(self, model, X_t, A_t, mask, t_index):
        """
        One step of the reverse process: x_t -> x_{t-1}.

        The DDPM reverse step formula:
            x_{t-1} = (1/sqrt(a_t)) * (x_t - (b_t/sqrt(1-ā_t)) * eps_theta) + sigma_t * z

        where z ~ N(0, I) and sigma_t = sqrt(beta_t).
        At t=0 we don't add noise (the final sample should be deterministic).
        """
        B = X_t.shape[0]
        device = X_t.device

        t_tensor = torch.full((B,), t_index, device=device, dtype=torch.long)

        # Predict noise
        pred_noise_X, pred_noise_A = model(X_t, A_t, mask, t_tensor)

        # Retrieve schedule values
        beta_t = self.betas[t_index]
        alpha_t = self.alphas[t_index]
        alpha_bar_t = self.alpha_bars[t_index]
        sqrt_alpha_t = self.sqrt_alphas[t_index]

        # Compute the denoised mean
        coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean_X = (1.0 / sqrt_alpha_t) * (X_t - coeff * pred_noise_X)
        mean_A = (1.0 / sqrt_alpha_t) * (A_t - coeff * pred_noise_A)

        if t_index > 0:
            # Add stochastic noise (except at the final step)
            sigma = torch.sqrt(beta_t)
            noise_X = torch.randn_like(X_t) * sigma
            noise_A = torch.randn_like(A_t) * sigma
            noise_A = (noise_A + noise_A.transpose(1, 2)) / 2  # symmetrize
            X_t = mean_X + noise_X
            A_t = mean_A + noise_A
        else:
            X_t = mean_X
            A_t = mean_A

        # Mask out padding
        X_t = X_t * mask.unsqueeze(-1)
        A_t = A_t * mask.unsqueeze(1) * mask.unsqueeze(2)

        return X_t, A_t

    @torch.no_grad()
    def sample(self, model, num_samples, max_atoms, num_atom_types, mask=None):
        """
        Generate molecules by running the full reverse process: t=T -> t=0.

        Args:
            model:          trained ScoreNetwork
            num_samples:    how many molecules to generate
            max_atoms:      padded molecule size
            num_atom_types: number of atom types

        Returns:
            X: (num_samples, max_atoms, num_atom_types) generated node features
            A: (num_samples, max_atoms, max_atoms) generated adjacency
        """
        device = self.device
        model.eval()

        # Start from pure Gaussian noise
        X_t = torch.randn(num_samples, max_atoms, num_atom_types, device=device)
        A_t = torch.randn(num_samples, max_atoms, max_atoms, device=device)
        A_t = (A_t + A_t.transpose(1, 2)) / 2  # start symmetric

        # Use full mask (all nodes active) -- the model decides which to "turn off"
        if mask is None:
            mask = torch.ones(num_samples, max_atoms, device=device)

        # Reverse diffusion: T-1, T-2, ..., 1, 0
        for t in reversed(range(self.num_timesteps)):
            X_t, A_t = self.p_sample_step(model, X_t, A_t, mask, t)

            if t % 100 == 0:
                print(f"  Sampling step {self.num_timesteps - t}/{self.num_timesteps}")

        return X_t, A_t
