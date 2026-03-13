"""
Drug-Likeness Properties
========================
Not all valid molecules make good drugs. These filters help identify
drug-like candidates before expensive docking simulations.

LIPINSKI'S RULE OF FIVE (1997):
    A quick heuristic: most orally active drugs satisfy:
      - Molecular weight <= 500 Da
      - LogP (lipophilicity) <= 5
      - H-bond donors <= 5
      - H-bond acceptors <= 10
    Violations don't mean "bad drug" but they raise a flag.

QED (Quantitative Estimate of Drug-likeness):
    A single score (0-1) combining multiple properties.
    Higher = more drug-like. Introduced by Bickerton et al. (2012).

SA SCORE (Synthetic Accessibility):
    How hard is it to actually synthesize this molecule in a lab?
    Scale of 1 (easy) to 10 (very hard). From Ertl & Schuffenhauer (2009).
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def lipinski_rule_of_five(mol):
    """
    Check Lipinski's Rule of Five.

    Returns:
        dict with individual properties and whether the rule is satisfied
    """
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    return {
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "num_violations": violations,
        "passes": violations <= 1,  # typically allow 1 violation
    }


def compute_qed(mol):
    """Compute QED (Quantitative Estimate of Drug-likeness). Range: 0-1."""
    if mol is None:
        return 0.0
    try:
        return round(QED.qed(mol), 4)
    except Exception:
        return 0.0


def compute_properties(smiles: str):
    """
    Compute all drug-likeness properties for a SMILES string.

    Returns a dict with Lipinski properties, QED, ring count, and rotatable bonds.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    lipinski = lipinski_rule_of_five(mol)
    qed_score = compute_qed(mol)

    return {
        "smiles": smiles,
        "qed": qed_score,
        "num_atoms": mol.GetNumAtoms(),
        "num_rings": Descriptors.RingCount(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),  # topological polar surface area
        **lipinski,
    }
