"""
Chemical Validity Checking
==========================
After the diffusion model generates continuous tensors, we discretize them
into atom types and bond types. But rounding doesn't guarantee chemistry!

WHAT MAKES A MOLECULE VALID?
    1. Valence rules: Each atom has a maximum number of bonds.
       Carbon=4, Nitrogen=3, Oxygen=2, etc. Violating this = invalid.
    2. Sanitization: RDKit checks aromaticity, ring geometry, and more.
    3. Connected: The molecule should be a single connected component
       (not disconnected fragments).
    4. Non-trivial: Must have at least 2 atoms.

WHY VALIDITY MATTERS:
    A generative model that produces 90% invalid molecules is useless in
    practice. Validity rate is the FIRST metric people check. State-of-the-art
    molecular diffusion models achieve >85% validity on ZINC250K.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter


def check_validity(mol) -> bool:
    """
    Check if an RDKit Mol object represents a valid molecule.

    Returns True if the molecule passes all checks.
    """
    if mol is None:
        return False

    try:
        # SanitizeMol checks valence, aromaticity, ring info, etc.
        Chem.SanitizeMol(mol)

        # Must have a valid SMILES representation
        smiles = Chem.MolToSmiles(mol)
        if not smiles or smiles == "":
            return False

        # Must have at least 2 atoms
        if mol.GetNumAtoms() < 2:
            return False

        # Check for disconnected fragments (we want single molecules)
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            return False

        return True
    except Exception:
        return False


def compute_validity_metrics(mol_list, training_smiles=None):
    """
    Compute validity, uniqueness, and novelty metrics for a list of molecules.

    These are the THREE standard metrics for molecular generation:
        - Validity:   fraction of generated molecules that are chemically valid
        - Uniqueness:  fraction of valid molecules that are distinct (no duplicates)
        - Novelty:    fraction of unique molecules NOT in the training set

    Args:
        mol_list:         list of RDKit Mol objects (some may be None)
        training_smiles:  set of SMILES strings from training data (for novelty)

    Returns:
        dict with metrics and lists of valid/unique/novel SMILES
    """
    total = len(mol_list)

    # --- Validity ---
    valid_smiles = []
    for mol in mol_list:
        if check_validity(mol):
            smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(smiles)

    validity = len(valid_smiles) / total if total > 0 else 0.0

    # --- Uniqueness ---
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    # --- Novelty ---
    if training_smiles is not None:
        novel_smiles = [s for s in unique_smiles if s not in training_smiles]
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0
    else:
        novel_smiles = unique_smiles
        novelty = None  # can't compute without training set

    # --- Atom type distribution ---
    atom_counts = Counter()
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for atom in mol.GetAtoms():
                atom_counts[atom.GetSymbol()] += 1

    return {
        "total_generated": total,
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "valid_smiles": valid_smiles,
        "unique_smiles": unique_smiles,
        "novel_smiles": novel_smiles,
        "atom_distribution": dict(atom_counts),
    }
