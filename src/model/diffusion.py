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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_alpha_bar_schedule(num_timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal (2021, Improved DDPM).

    Produces a smoother SNR curve than linear, preserving more signal at
    early timesteps and destroying it more gradually.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bars = f_t / f_t[0]
    # Derive betas from consecutive alpha_bar ratios
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, min=0.0, max=0.999).float()


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
        schedule: str = "cosine",
        importance_sampling: bool = True,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.importance_sampling = importance_sampling

        # ---- Noise Schedule ----
        if schedule == "cosine":
            betas = _cosine_alpha_bar_schedule(num_timesteps)
        else:
            # Linear schedule (original): beta increases linearly.
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Pre-compute useful quantities (avoids recomputation during training)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(device)
        self.sqrt_alphas = torch.sqrt(alphas).to(device)

        # Importance sampling weights: sample noisier timesteps more often
        if importance_sampling:
            weights = self.sqrt_one_minus_alpha_bars
            self.timestep_probs = (weights / weights.sum()).to(device)
        else:
            self.timestep_probs = None

    def to(self, device):
        """Move all schedule tensors to a device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        if self.timestep_probs is not None:
            self.timestep_probs = self.timestep_probs.to(device)
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
        T = self.num_timesteps

        # 1. Sample timesteps (importance-weighted or uniform)
        if self.importance_sampling and self.timestep_probs is not None:
            t = torch.multinomial(self.timestep_probs, B, replacement=True)
        else:
            t = torch.randint(0, T, (B,), device=device)

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
        # Expand edge_mask for multi-channel adjacency (B, N, N) -> (B, N, N, 1)
        if pred_noise_A.dim() == 4:
            edge_mask_a = edge_mask.unsqueeze(-1)
        else:
            edge_mask_a = edge_mask
        loss_A = F.mse_loss(
            pred_noise_A * edge_mask_a,
            noise_A * edge_mask_a,
            reduction="sum",
        ) / edge_mask_a.sum().clamp(min=1)

        # Apply importance sampling correction: weight each sample by 1/(T*p(t))
        # This keeps the expected loss unbiased despite non-uniform sampling.
        if self.importance_sampling and self.timestep_probs is not None:
            w = 1.0 / (T * self.timestep_probs[t] + 1e-8)
            w = w / w.mean()  # normalize so weights average to 1
            # Reweight: compute per-sample losses, then weight
            # Sum over all non-batch dims
            x_dims = tuple(range(1, pred_noise_X.dim()))
            a_dims = tuple(range(1, pred_noise_A.dim()))
            per_sample_loss_X = ((pred_noise_X - noise_X) ** 2 * node_mask).sum(dim=x_dims)
            per_sample_loss_A = ((pred_noise_A - noise_A) ** 2 * edge_mask_a).sum(dim=a_dims)
            loss_X = (w * per_sample_loss_X).mean() / node_mask[0].sum().clamp(min=1)
            loss_A = (w * per_sample_loss_A).mean() / edge_mask[0].sum().clamp(min=1)

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
        # A_t may be (B, N, N) or (B, N, N, C) -- mask spatial dims
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        if A_t.dim() == 4:
            A_t = A_t * edge_mask.unsqueeze(-1)
        else:
            A_t = A_t * edge_mask

        return X_t, A_t

    @torch.no_grad()
    def sample(self, model, num_samples, max_atoms, num_atom_types,
               num_bond_types=5, mask=None):
        """
        Generate molecules by running the full reverse process: t=T -> t=0.

        Args:
            model:          trained ScoreNetwork
            num_samples:    how many molecules to generate
            max_atoms:      padded molecule size
            num_atom_types: number of atom types (or atom_feature_dim)
            num_bond_types: number of bond type channels

        Returns:
            X: (num_samples, max_atoms, num_atom_types) generated node features
            A: (num_samples, max_atoms, max_atoms, num_bond_types) generated adjacency
        """
        device = self.device
        model.eval()

        # Start from pure Gaussian noise
        X_t = torch.randn(num_samples, max_atoms, num_atom_types, device=device)
        A_t = torch.randn(num_samples, max_atoms, max_atoms, num_bond_types, device=device)
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

    # ---- DDIM Sampling (Fast Deterministic Generation) ----

    @torch.no_grad()
    def ddim_sample(self, model, num_samples, max_atoms, num_atom_types,
                    num_bond_types=5, num_inference_steps=100, eta=0.0,
                    temperature=1.0, mask=None):
        """
        Generate molecules using DDIM (Song et al., 2020) for faster sampling.

        DDIM skips timesteps (e.g., 100 steps instead of 1000) and can be
        fully deterministic (eta=0). Temperature < 1 produces more conservative
        (higher validity) molecules.

        Args:
            model:              trained ScoreNetwork
            num_samples:        how many molecules to generate
            max_atoms:          padded molecule size
            num_atom_types:     atom feature dimension
            num_bond_types:     bond type channels
            num_inference_steps: number of denoising steps (< num_timesteps)
            eta:                0 = deterministic DDIM, 1 = stochastic (like DDPM)
            temperature:        scale noise predictions (< 1 = conservative)

        Returns:
            X, A: generated node features and adjacency tensors
        """
        device = self.device
        model.eval()
        T = self.num_timesteps

        # Compute strided timestep schedule
        step_size = T // num_inference_steps
        timesteps = list(range(0, T, step_size))[:num_inference_steps]
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        X_t = torch.randn(num_samples, max_atoms, num_atom_types, device=device)
        A_t = torch.randn(num_samples, max_atoms, max_atoms, num_bond_types, device=device)
        A_t = (A_t + A_t.transpose(1, 2)) / 2

        if mask is None:
            mask = torch.ones(num_samples, max_atoms, device=device)

        for i, t_cur in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            B = X_t.shape[0]
            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Predict noise
            eps_X, eps_A = model(X_t, A_t, mask, t_tensor)
            eps_X = eps_X * temperature
            eps_A = eps_A * temperature

            # Current and previous alpha_bar
            ab_t = self.alpha_bars[t_cur]
            ab_prev = self.alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # Predict x_0 from x_t and eps
            x0_pred_X = (X_t - torch.sqrt(1 - ab_t) * eps_X) / torch.sqrt(ab_t)
            x0_pred_A = (A_t - torch.sqrt(1 - ab_t) * eps_A) / torch.sqrt(ab_t)

            # DDIM variance
            sigma = eta * torch.sqrt(
                (1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)
            ) if t_cur > 0 else torch.tensor(0.0, device=device)

            # Direction pointing to x_t
            dir_X = torch.sqrt(1 - ab_prev - sigma ** 2) * eps_X
            dir_A = torch.sqrt(1 - ab_prev - sigma ** 2) * eps_A

            # DDIM step
            X_t = torch.sqrt(ab_prev) * x0_pred_X + dir_X
            A_t = torch.sqrt(ab_prev) * x0_pred_A + dir_A

            if sigma > 0:
                noise_X = torch.randn_like(X_t) * sigma
                noise_A = torch.randn_like(A_t) * sigma
                noise_A = (noise_A + noise_A.transpose(1, 2)) / 2
                X_t = X_t + noise_X
                A_t = A_t + noise_A

            # Symmetrize A
            A_t = (A_t + A_t.transpose(1, 2)) / 2

            # Mask padding
            X_t = X_t * mask.unsqueeze(-1)
            edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            if A_t.dim() == 4:
                A_t = A_t * edge_mask.unsqueeze(-1)
            else:
                A_t = A_t * edge_mask

            if (i + 1) % max(len(timesteps) // 5, 1) == 0:
                print(f"  DDIM step {i+1}/{len(timesteps)}")

        return X_t, A_t
