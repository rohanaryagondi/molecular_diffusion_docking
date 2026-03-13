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
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MolecularGraphDataset
from src.model.score_network import ScoreNetwork
from src.model.diffusion import GaussianDiffusion


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    # ---- Model ----
    model = ScoreNetwork(
        num_atom_types=data_cfg["num_atom_types"],
        num_bond_types=data_cfg["num_bond_types"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ---- Diffusion ----
    diffusion = GaussianDiffusion(
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Cosine annealing schedule (learning rate decays smoothly to ~0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"]
    )

    # Mixed precision (faster on A100/H200)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("use_amp", False))
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"

    # ---- Checkpoint directory ----
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training Loop ----
    print(f"\nStarting training for {train_cfg['num_epochs']} epochs ...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Diffusion timesteps: {diff_cfg['num_timesteps']}")
    print(f"  Mixed precision: {use_amp}")
    print()

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_x = 0.0
        epoch_loss_a = 0.0
        num_batches = 0
        t_start = time.time()

        for batch_idx, (X, A, mask) in enumerate(train_loader):
            X = X.to(device)
            A = A.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Forward pass with optional mixed precision
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_x, loss_a = diffusion.training_loss(
                    model, X, A, mask,
                    adj_loss_weight=train_cfg.get("adj_loss_weight", 1.0),
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

        scheduler.step()

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
            for X, A, mask in val_loader:
                X, A, mask = X.to(device), A.to(device), mask.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, _, _ = diffusion.training_loss(model, X, A, mask)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"  Val Loss: {avg_val_loss:.4f}")

        # ---- Save Checkpoints ----
        if epoch % train_cfg["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
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
                "config": config,
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")

        print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
