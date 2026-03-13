"""
Molecular Featurizer
====================
Converts between SMILES strings and dense graph tensor representations.

WHAT IS A MOLECULAR GRAPH?
    Atoms are nodes, bonds are edges. We represent each molecule as:
      X : (max_atoms, atom_feature_dim)  node features
      A : (max_atoms, max_atoms, num_bond_types)  one-hot adjacency
      mask : (max_atoms,)              1.0 for real atoms, 0.0 for padding

NODE FEATURES:
    Each atom gets a feature vector combining:
      - Atom type (one-hot): C, N, O, F, P, S, Cl, Br, I, virtual  [10 dims]
      - Formal charge (one-hot): -2, -1, 0, +1, +2                 [5 dims]
      - Degree (one-hot): 0, 1, 2, 3, 4, 5                         [6 dims]
      - Implicit hydrogens (one-hot): 0, 1, 2, 3, 4                [5 dims]
      - Is aromatic (binary)                                         [1 dim]
      - Is in ring (binary)                                          [1 dim]
    Total: 28 dims. Only the first 10 (atom type) are used during
    discretization -- the rest provide chemical context for the GNN.

ADJACENCY:
    One-hot encoding of bond types: (max_atoms, max_atoms, 5) where
    channel 0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic.
    This respects the categorical nature of bonds (aromatic != "more than triple").

WHY ONE-HOT FOR BONDS?
    Treating bond types as a single integer (0-4) implies an ordinal
    relationship (single < double < triple < aromatic) that doesn't
    exist. One-hot + multi-channel diffusion lets each bond type evolve
    independently during noising/denoising, dramatically improving validity.
"""

import torch
import numpy as np
from rdkit import Chem


# ---- Atom & Bond vocabulary ------------------------------------------------
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1  # +1 for virtual/padding node

# Extended atom feature dimensions
CHARGE_CLASSES = [-2, -1, 0, 1, 2]  # 5 dims
DEGREE_CLASSES = [0, 1, 2, 3, 4, 5]  # 6 dims
IMPLICIT_H_CLASSES = [0, 1, 2, 3, 4]  # 5 dims
# Plus: is_aromatic (1), is_in_ring (1)

ATOM_FEATURE_DIM = NUM_ATOM_TYPES + len(CHARGE_CLASSES) + len(DEGREE_CLASSES) + len(IMPLICIT_H_CLASSES) + 2  # = 28

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
NUM_BOND_TYPES = 5  # including 0 = no bond

REVERSE_BOND_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}

MAX_ATOMS = 38


# ---- SMILES -> Graph --------------------------------------------------------

def _one_hot(value, classes):
    """Create a one-hot vector for value within classes list."""
    vec = [0.0] * len(classes)
    if value in classes:
        vec[classes.index(value)] = 1.0
    else:
        # Clamp to nearest endpoint
        if isinstance(value, (int, float)):
            if value <= classes[0]:
                vec[0] = 1.0
            else:
                vec[-1] = 1.0
    return vec


def smiles_to_graph(smiles: str):
    """
    Convert a SMILES string to (X, A, mask) tensors.

    Returns:
        X:    (MAX_ATOMS, ATOM_FEATURE_DIM) node features
        A:    (MAX_ATOMS, MAX_ATOMS, NUM_BOND_TYPES) one-hot adjacency
        mask: (MAX_ATOMS,) node mask

    Returns None if the molecule can't be parsed or has unsupported atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms > MAX_ATOMS:
        return None

    # --- Node features ---
    X = torch.zeros(MAX_ATOMS, ATOM_FEATURE_DIM)
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in ATOM_TYPES:
            return None  # unsupported atom

        features = []
        # Atom type one-hot (10 dims)
        atom_oh = [0.0] * NUM_ATOM_TYPES
        atom_oh[ATOM_TYPES.index(symbol)] = 1.0
        features.extend(atom_oh)
        # Formal charge (5 dims)
        features.extend(_one_hot(atom.GetFormalCharge(), CHARGE_CLASSES))
        # Degree (6 dims)
        features.extend(_one_hot(atom.GetDegree(), DEGREE_CLASSES))
        # Implicit hydrogens (5 dims)
        features.extend(_one_hot(atom.GetNumImplicitHs(), IMPLICIT_H_CLASSES))
        # Is aromatic (1 dim)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        # Is in ring (1 dim)
        features.append(1.0 if atom.IsInRing() else 0.0)

        X[i] = torch.tensor(features)

    # Padding atoms get the virtual atom type (index 9)
    for i in range(num_atoms, MAX_ATOMS):
        X[i, NUM_ATOM_TYPES - 1] = 1.0  # virtual type
        X[i, NUM_ATOM_TYPES + CHARGE_CLASSES.index(0)] = 1.0  # charge = 0

    # --- Adjacency matrix: one-hot bond types ---
    A = torch.zeros(MAX_ATOMS, MAX_ATOMS, NUM_BOND_TYPES)
    # Default: no-bond channel (index 0) is 1 for all non-bonded pairs
    A[:, :, 0] = 1.0
    for bond in mol.GetBonds():
        bi = bond.GetBeginAtomIdx()
        bj = bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 1)
        # Clear the no-bond channel, set the bond type channel
        A[bi, bj, 0] = 0.0
        A[bi, bj, bond_type] = 1.0
        A[bj, bi, 0] = 0.0
        A[bj, bi, bond_type] = 1.0

    # --- Node mask ---
    mask = torch.zeros(MAX_ATOMS)
    mask[:num_atoms] = 1.0

    return X, A, mask


# ---- Graph -> Molecule ------------------------------------------------------

def graph_to_mol(X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor):
    """
    Convert (X, A, mask) tensors back to an RDKit Mol object.

    Handles both one-hot adjacency (N, N, 5) and scalar adjacency (N, N).
    Atom type is determined by argmax of the first NUM_ATOM_TYPES dims of X.

    Returns None if the resulting molecule is chemically invalid.
    """
    num_atoms = int(mask.sum().item())
    if num_atoms == 0:
        return None

    # Discretize atom types (first 10 dims = atom type one-hot)
    atom_indices = X[:num_atoms, :NUM_ATOM_TYPES].argmax(dim=-1).tolist()

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
            if A.dim() == 3:
                # One-hot adjacency: argmax over bond type channels
                bond_val = int(A[i, j].argmax().item())
            else:
                # Scalar adjacency (legacy): round to nearest int
                bond_val = int(round(A[i, j].item()))
                bond_val = max(0, min(bond_val, 4))

            if bond_val > 0:
                bond_type = REVERSE_BOND_MAP.get(
                    bond_val, Chem.rdchem.BondType.SINGLE
                )
                mol.AddBond(i, j, bond_type)

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
