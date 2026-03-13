"""
PyTorch Dataset for Preprocessed Molecular Graphs
==================================================
After running scripts/preprocess.py, molecular graphs are stored as a single
.pt file containing stacked tensors. This Dataset class loads them for training.

The DataLoader will automatically batch these into:
    X:      (batch_size, max_atoms, atom_feature_dim)
    A:      (batch_size, max_atoms, max_atoms, num_bond_types)
    mask:   (batch_size, max_atoms)
    labels: (batch_size,)  -- QED bucket labels for classifier-free guidance
"""

import os
import torch
from torch.utils.data import Dataset


class MolecularGraphDataset(Dataset):
    """
    Loads preprocessed molecular graphs from a .pt file.

    Expected format of the .pt file (created by scripts/preprocess.py):
        {
            'X':      Tensor of shape (N, max_atoms, atom_feature_dim),
            'A':      Tensor of shape (N, max_atoms, max_atoms, num_bond_types),
            'masks':  Tensor of shape (N, max_atoms),
            'labels': Tensor of shape (N,) [optional, for guided generation],
        }
    """

    def __init__(self, data_dir: str, split: str = "train"):
        filepath = os.path.join(data_dir, f"{split}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"{filepath} not found. Run scripts/preprocess.py first."
            )

        data = torch.load(filepath, weights_only=True)
        self.X = data["X"]
        self.A = data["A"]
        self.masks = data["masks"]
        self.labels = data.get("labels", None)

        print(f"Loaded {split} set: {len(self)} molecules")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.X[idx], self.A[idx], self.masks[idx], self.labels[idx]
        return self.X[idx], self.A[idx], self.masks[idx]
