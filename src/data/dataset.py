"""
PyTorch Dataset for Preprocessed Molecular Graphs
==================================================
After running scripts/preprocess.py, molecular graphs are stored as a single
.pt file containing stacked tensors. This Dataset class loads them for training.

The DataLoader will automatically batch these into:
    X:    (batch_size, max_atoms, num_atom_types)
    A:    (batch_size, max_atoms, max_atoms)
    mask: (batch_size, max_atoms)
"""

import os
import torch
from torch.utils.data import Dataset


class MolecularGraphDataset(Dataset):
    """
    Loads preprocessed molecular graphs from a .pt file.

    Expected format of the .pt file (created by scripts/preprocess.py):
        {
            'X':     Tensor of shape (N, max_atoms, num_atom_types),
            'A':     Tensor of shape (N, max_atoms, max_atoms),
            'masks': Tensor of shape (N, max_atoms),
        }
    """

    def __init__(self, data_dir: str, split: str = "train"):
        filepath = os.path.join(data_dir, f"{split}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"{filepath} not found. Run scripts/preprocess.py first."
            )

        data = torch.load(filepath, weights_only=True)
        self.X = data["X"]          # (N, max_atoms, num_atom_types)
        self.A = data["A"]          # (N, max_atoms, max_atoms)
        self.masks = data["masks"]  # (N, max_atoms)

        print(f"Loaded {split} set: {len(self)} molecules")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.masks[idx]
