"""
Preprocess SMILES -> Molecular Graph Tensors
=============================================
Converts the raw SMILES strings from ZINC250K into dense graph tensors
that the diffusion model can train on.

For each molecule:
    SMILES string  ->  (X, A, mask) tensors

These are stacked into large tensors and saved as .pt files for fast loading.

This script is CPU-only and can be run on a login node or locally.
Typical runtime: ~5-10 minutes for 250K molecules.

Usage:
    python scripts/preprocess.py [--config configs/default.yaml]
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.featurizer import smiles_to_graph


def preprocess(config_path: str = "configs/default.yaml"):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    input_path = data_cfg["dataset_path"]
    output_dir = data_cfg["processed_path"]
    train_split = data_cfg["train_split"]

    os.makedirs(output_dir, exist_ok=True)

    # Load raw SMILES
    print(f"Loading SMILES from {input_path} ...")
    df = pd.read_csv(input_path)
    smiles_list = df["smiles"].tolist()
    print(f"Total molecules: {len(smiles_list)}")

    # Convert to graphs
    print("Converting SMILES to molecular graphs ...")
    all_X, all_A, all_masks = [], [], []
    failed = 0

    for smiles in tqdm(smiles_list, desc="Featurizing"):
        result = smiles_to_graph(smiles)
        if result is not None:
            X, A, mask = result
            all_X.append(X)
            all_A.append(A)
            all_masks.append(mask)
        else:
            failed += 1

    print(f"Successfully converted: {len(all_X)}")
    print(f"Failed/skipped: {failed}")

    # Stack into tensors
    X_tensor = torch.stack(all_X)       # (N, max_atoms, num_atom_types)
    A_tensor = torch.stack(all_A)       # (N, max_atoms, max_atoms)
    mask_tensor = torch.stack(all_masks) # (N, max_atoms)

    # Train/val split
    n_total = len(X_tensor)
    n_train = int(n_total * train_split)
    indices = torch.randperm(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Save train set
    train_path = os.path.join(output_dir, "train.pt")
    torch.save({
        "X": X_tensor[train_idx],
        "A": A_tensor[train_idx],
        "masks": mask_tensor[train_idx],
    }, train_path)
    print(f"Train set ({len(train_idx)} molecules) saved to {train_path}")

    # Save val set
    val_path = os.path.join(output_dir, "val.pt")
    torch.save({
        "X": X_tensor[val_idx],
        "A": A_tensor[val_idx],
        "masks": mask_tensor[val_idx],
    }, val_path)
    print(f"Val set ({len(val_idx)} molecules) saved to {val_path}")

    # Save the original SMILES for novelty checking later
    smiles_path = os.path.join(output_dir, "train_smiles.txt")
    successful_smiles = [s for s in smiles_list if smiles_to_graph(s) is not None]
    train_smiles = [successful_smiles[i] for i in train_idx.tolist() if i < len(successful_smiles)]
    with open(smiles_path, "w") as f:
        for s in train_smiles:
            f.write(s + "\n")
    print(f"Training SMILES saved to {smiles_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    preprocess(args.config)
