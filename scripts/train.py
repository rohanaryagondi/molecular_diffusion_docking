"""
Train the Molecular Diffusion Model
====================================
This is the main training script. It:
    1. Loads preprocessed molecular graphs
    2. Creates the score network and diffusion process
    3. Runs the training loop (denoising score matching)
    4. Saves checkpoints periodically

The training objective is simple:
    - Pick a random molecule from the dataset
    - Add random noise at a random timestep
    - Ask the network to predict what noise was added
    - Minimize the prediction error (MSE loss)

After enough training, the network learns to denoise at every noise level,
which is equivalent to learning the score function (gradient of log probability).

Usage:
    python scripts/train.py --config configs/default.yaml

    On Bouchet:
    sbatch scripts/slurm_train.sh
"""

import os
import sys
import argparse
import copy
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MolecularGraphDataset
from src.model.score_network import ScoreNetwork
from src.model.diffusion import GaussianDiffusion
from src.data.featurizer import graph_to_mol
from src.chemistry.validity import check_validity


# ---- EMA (Exponential Moving Average) ----

class EMA:
    """Maintains an exponential moving average of model parameters.

    EMA weights produce smoother, more robust predictions at generation time.
    Standard practice for diffusion models (decay=0.9999).
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: p.clone() for name, p in self.shadow.items()}

    def load_into(self, model):
        """Load EMA weights into a model (for generation)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])


def train(config_path: str):
    # ---- Load Config ----
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    diff_cfg = config["diffusion"]
    train_cfg = config["training"]

    # ---- Device Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Data ----
    print("Loading training data ...")
    train_dataset = MolecularGraphDataset(data_cfg["processed_path"], split="train")
    val_dataset = MolecularGraphDataset(data_cfg["processed_path"], split="val")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin,
    )

    # ---- Model ----
    atom_feature_dim = data_cfg.get("atom_feature_dim", data_cfg["num_atom_types"])
    guidance_cfg = config.get("guidance", {})
    model = ScoreNetwork(
        num_atom_types=data_cfg["num_atom_types"],
        num_bond_types=data_cfg["num_bond_types"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        atom_feature_dim=atom_feature_dim,
        num_classes=guidance_cfg.get("num_classes", 3),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ---- EMA ----
    ema_decay = train_cfg.get("ema_decay", 0.9999)
    ema = EMA(model, decay=ema_decay)

    # ---- Diffusion ----
    diffusion = GaussianDiffusion(
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        schedule=diff_cfg.get("schedule", "cosine"),
        importance_sampling=diff_cfg.get("importance_sampling", True),
        device=device,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # LR schedule: linear warmup then cosine decay (per step, not per epoch)
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * train_cfg["num_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision (faster on A100/H200)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("use_amp", False))
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"

    # ---- Checkpoint directory ----
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training Loop ----
    schedule_name = diff_cfg.get("schedule", "cosine")
    print(f"\nStarting training for {train_cfg['num_epochs']} epochs ...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']} (warmup: {warmup_steps} steps)")
    print(f"  Diffusion timesteps: {diff_cfg['num_timesteps']} ({schedule_name} schedule)")
    print(f"  EMA decay: {ema_decay}")
    print(f"  Mixed precision: {use_amp}")
    print()

    best_val_loss = float("inf")
    global_step = 0
    validate_every = train_cfg.get("validate_every", 0)
    num_classes = guidance_cfg.get("num_classes", 3)

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_x = 0.0
        epoch_loss_a = 0.0
        num_batches = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Dataset returns (X, A, mask) or (X, A, mask, label)
            if len(batch) == 4:
                X, A, mask, labels = batch
                labels = labels.to(device)
            else:
                X, A, mask = batch
                labels = None

            X = X.to(device)
            A = A.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Forward pass with optional mixed precision
            p_uncond = guidance_cfg.get("p_uncond", 0.1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_x, loss_a = diffusion.training_loss(
                    model, X, A, mask,
                    adj_loss_weight=train_cfg.get("adj_loss_weight", 1.0),
                    labels=labels,
                    p_uncond=p_uncond,
                    num_classes=num_classes,
                )

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (prevents exploding gradients)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # step per optimizer step (not per epoch)

            # Update EMA weights
            ema.update(model)

            epoch_loss += loss.item()
            epoch_loss_x += loss_x
            epoch_loss_a += loss_a
            num_batches += 1
            global_step += 1

            # Periodic logging
            if global_step % train_cfg["log_every"] == 0:
                print(
                    f"  Step {global_step} | "
                    f"Loss: {loss.item():.4f} "
                    f"(X: {loss_x:.4f}, A: {loss_a:.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        # ---- Epoch Summary ----
        avg_loss = epoch_loss / num_batches
        avg_loss_x = epoch_loss_x / num_batches
        avg_loss_a = epoch_loss_a / num_batches
        elapsed = time.time() - t_start

        print(
            f"Epoch {epoch}/{train_cfg['num_epochs']} | "
            f"Train Loss: {avg_loss:.4f} (X: {avg_loss_x:.4f}, A: {avg_loss_a:.4f}) | "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    X, A, mask, labels = batch
                    labels = labels.to(device)
                else:
                    X, A, mask = batch
                    labels = None
                X, A, mask = X.to(device), A.to(device), mask.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, _, _ = diffusion.training_loss(
                        model, X, A, mask, labels=labels,
                        p_uncond=0.0, num_classes=num_classes,
                    )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"  Val Loss: {avg_val_loss:.4f}")

        # ---- Validity Check (periodic) ----
        if validate_every > 0 and epoch % validate_every == 0:
            _run_validity_check(model, diffusion, data_cfg, device)

        # ---- Save Checkpoints ----
        if epoch % train_cfg["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "config": config,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "config": config,
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")

        print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def _run_validity_check(model, diffusion, data_cfg, device, num_samples=64):
    """Generate a small batch and report validity percentage."""
    model.eval()
    atom_feature_dim = data_cfg.get("atom_feature_dim", data_cfg["num_atom_types"])
    X_gen, A_gen = diffusion.sample(
        model,
        num_samples=num_samples,
        max_atoms=data_cfg["max_atoms"],
        num_atom_types=atom_feature_dim,
        num_bond_types=data_cfg["num_bond_types"],
    )
    nat = data_cfg["num_atom_types"]
    virtual_idx = nat - 1
    masks = (X_gen[:, :, :nat].argmax(dim=-1) != virtual_idx).float()
    valid = 0
    for i in range(num_samples):
        mol = graph_to_mol(X_gen[i].cpu(), A_gen[i].cpu(), masks[i].cpu())
        if mol is not None and check_validity(mol):
            valid += 1
    print(f"  Validity check: {valid}/{num_samples} = {valid/num_samples:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
