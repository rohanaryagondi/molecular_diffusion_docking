"""
Molecular Featurizer
====================
Converts between SMILES strings and dense graph tensor representations.

WHAT IS A MOLECULAR GRAPH?
    Atoms are nodes, bonds are edges. We represent each molecule as:
      X : (max_atoms, num_atom_types)  one-hot node features
      A : (max_atoms, max_atoms)       adjacency matrix (bond type integers)
      mask : (max_atoms,)              1.0 for real atoms, 0.0 for padding

WHY PAD TO A FIXED SIZE?
    Neural networks need fixed-size inputs within a batch. ZINC250K's largest
    molecule has 38 heavy atoms, so we pad everything to 38. Padding atoms
    get a special "virtual" type that the model learns to ignore.

WHY ONE-HOT ENCODING?
    Atom types are categorical (C, N, O, ...), not numerical. One-hot lets
    us treat them as continuous vectors during diffusion:
      Carbon   = [1, 0, 0, 0, ...]
      Nitrogen = [0, 1, 0, 0, ...]
    During noising, these become mixtures (e.g., [0.7, 0.2, 0.1, ...])
    and get discretized back (argmax) after generation.
"""

import torch
import numpy as np
from rdkit import Chem


# ---- Atom & Bond vocabulary ------------------------------------------------
# These are the 9 atom types found in ZINC250K. If a molecule contains
# any other element, we skip it (very rare in this dataset).
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1  # +1 for virtual/padding node

# Bond types encoded as integers in the adjacency matrix.
# 0 = no bond (default), 1-4 = real bond types.
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
NUM_BOND_TYPES = 5  # including 0 = no bond

# Reverse mapping for converting adjacency values back to RDKit bond types
REVERSE_BOND_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}

MAX_ATOMS = 38  # largest molecule in ZINC250K


# ---- SMILES -> Graph --------------------------------------------------------

def smiles_to_graph(smiles: str):
    """
    Convert a SMILES string to (X, A, mask) tensors.

    Returns None if the molecule can't be parsed or has unsupported atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms > MAX_ATOMS:
        return None

    # --- Node features: one-hot atom types ---
    X = torch.zeros(MAX_ATOMS, NUM_ATOM_TYPES)
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in ATOM_TYPES:
            X[i, ATOM_TYPES.index(symbol)] = 1.0
        else:
            return None  # unsupported atom

    # Padding atoms get the virtual type (last index)
    for i in range(num_atoms, MAX_ATOMS):
        X[i, -1] = 1.0

    # --- Adjacency matrix: bond types as integers ---
    A = torch.zeros(MAX_ATOMS, MAX_ATOMS)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 1)
        A[i, j] = float(bond_type)
        A[j, i] = float(bond_type)  # symmetric (undirected graph)

    # --- Node mask ---
    mask = torch.zeros(MAX_ATOMS)
    mask[:num_atoms] = 1.0

    return X, A, mask


# ---- Graph -> Molecule ------------------------------------------------------

def graph_to_mol(X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor):
    """
    Convert (X, A, mask) tensors back to an RDKit Mol object.

    After generation, X and A are continuous. We discretize by:
      - Atom type: argmax of the one-hot vector
      - Bond type: round the adjacency value to nearest integer

    Returns None if the resulting molecule is chemically invalid.
    """
    num_atoms = int(mask.sum().item())
    if num_atoms == 0:
        return None

    # Discretize atom types
    atom_indices = X[:num_atoms].argmax(dim=-1).tolist()

    # Build molecule atom by atom
    mol = Chem.RWMol()
    for idx in atom_indices:
        if idx < len(ATOM_TYPES):
            mol.AddAtom(Chem.Atom(ATOM_TYPES[idx]))
        else:
            mol.AddAtom(Chem.Atom("C"))  # virtual node fallback

    # Add bonds from upper triangle of adjacency
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bond_val = int(round(A[i, j].item()))
            bond_val = max(0, min(bond_val, 4))  # clamp to valid range
            if bond_val > 0:
                bond_type = REVERSE_BOND_MAP.get(
                    bond_val, Chem.rdchem.BondType.SINGLE
                )
                mol.AddBond(i, j, bond_type)

    # RDKit sanitization: checks valence rules, aromaticity, etc.
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def graph_to_smiles(X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> str:
    """Convert graph tensors to a canonical SMILES string (empty if invalid)."""
    mol = graph_to_mol(X, A, mask)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""
