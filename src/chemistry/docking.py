"""
Molecular Docking with AutoDock Vina
=====================================
Docking predicts how well a small molecule (ligand) fits into a protein's
binding pocket. The score (kcal/mol) estimates binding affinity:
    - More negative = stronger binding = better drug candidate
    - Typical good binders: -7 to -12 kcal/mol
    - Weak/no binding: > -5 kcal/mol

THE DOCKING PIPELINE:
    1. Prepare the protein (download PDB, remove water, add hydrogens)
    2. Prepare the ligand (SMILES -> 3D coordinates -> PDBQT format)
    3. Define a search box around the binding pocket
    4. Run Vina's scoring function (searches for optimal ligand pose)
    5. Return the best binding affinity score

DEPENDENCIES:
    - vina: Python bindings for AutoDock Vina
    - meeko: Converts RDKit molecules to PDBQT format for Vina
    - rdkit: 3D coordinate generation
    - biopython: PDB file downloading

    On Bouchet, install in your venv:
        pip install vina meeko biopython

    If vina won't install (it needs C++ compilation), you can:
        1. Use `module load AutoDock-Vina` if available
        2. Or install via conda: conda install -c conda-forge autodock-vina
"""

import os
import tempfile
import warnings
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_3d_coords(smiles: str, num_conformers: int = 1):
    """
    Generate a 3D conformer for a molecule from its SMILES string.

    This uses RDKit's ETKDG algorithm (Experimental-Torsion Knowledge Distance
    Geometry) which produces realistic 3D geometries by combining:
      - Distance geometry (initial guess)
      - Experimental torsion angle preferences
      - Force field optimization (MMFF94)

    Returns an RDKit Mol with 3D coordinates, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add explicit hydrogens (needed for 3D embedding and docking)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates with multiple conformers for better docking
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    num_confs = num_conformers if num_conformers > 1 else 1
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) == 0:
        # Retry with basic ETKDG
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDG())
        if len(conf_ids) == 0:
            return None

    # Optimize all conformer geometries with MMFF94 force field
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        # Pick the lowest energy conformer
        if results and len(results) > 0:
            energies = [(i, r[1]) for i, r in enumerate(results) if r[0] == 0]
            if energies:
                best_conf = min(energies, key=lambda x: x[1])[0]
                # Set the best conformer as the active one
                # (keep all conformers, first one is used by default)
    except Exception:
        pass  # keep unoptimized geometries

    return mol


def mol_to_pdbqt_string(mol):
    """
    Convert an RDKit Mol (with 3D coords) to PDBQT format string.

    Uses the meeko library which handles:
      - Atom typing for AutoDock
      - Torsion tree (rotatable bonds)
      - Partial charge assignment
    """
    try:
        from meeko import MoleculePreparation

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        # Get the first (and usually only) setup
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = preparator.write_pdbqt_string(setup)
            if is_ok:
                return pdbqt_string
        return None
    except ImportError:
        warnings.warn(
            "meeko not installed. Install with: pip install meeko\n"
            "Falling back to basic PDB format (less accurate for Vina)."
        )
        return _fallback_pdb_string(mol)


def _fallback_pdb_string(mol):
    """Basic PDB string generation when meeko is not available."""
    try:
        return Chem.MolToPDBBlock(mol)
    except Exception:
        return None


def download_protein(pdb_id: str, output_dir: str = "data/protein"):
    """
    Download a protein structure from the RCSB Protein Data Bank.

    Args:
        pdb_id: 4-character PDB code (e.g., "1M17" for EGFR kinase)
        output_dir: directory to save the PDB file

    Returns:
        path to the downloaded PDB file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path):
        print(f"Protein {pdb_id} already downloaded.")
        return output_path

    try:
        from Bio.PDB import PDBList

        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")

        # PDBList saves with a different naming convention; rename it
        downloaded = os.path.join(output_dir, f"pdb{pdb_id.lower()}.ent")
        if os.path.exists(downloaded):
            os.rename(downloaded, output_path)
    except ImportError:
        # Fallback: direct HTTP download
        import urllib.request

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, output_path)

    print(f"Saved protein to {output_path}")
    return output_path


def prepare_receptor(pdb_path: str):
    """
    Prepare a protein receptor for docking.

    Steps:
        1. Remove water molecules (HOH)
        2. Remove co-crystallized ligands and other HETATMs
        3. Write cleaned structure

    NOTE: For production docking, you'd also add hydrogens and assign
    charges using tools like ADFR's prepare_receptor. For this project,
    Vina handles basic receptor preparation internally.

    Returns:
        path to the cleaned PDB file
    """
    clean_path = pdb_path.replace(".pdb", "_clean.pdb")

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # Keep only ATOM records (protein backbone + sidechains)
    # Remove HETATM (water, ligands, ions)
    clean_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            clean_lines.append(line)
        elif line.startswith("END"):
            clean_lines.append(line)

    with open(clean_path, "w") as f:
        f.writelines(clean_lines)

    print(f"Cleaned receptor saved to {clean_path}")
    return clean_path


def dock_molecule(
    smiles: str,
    receptor_path: str,
    center: list,
    box_size: list,
    exhaustiveness: int = 8,
    num_modes: int = 9,
):
    """
    Dock a single molecule against a protein receptor.

    Args:
        smiles:          SMILES string of the ligand
        receptor_path:   path to receptor PDBQT/PDB file
        center:          [x, y, z] center of the binding pocket (Angstroms)
        box_size:        [sx, sy, sz] search box dimensions
        exhaustiveness:  search thoroughness (higher = slower but better)
        num_modes:       number of binding poses to evaluate

    Returns:
        dict with binding affinity and metadata, or None on failure
    """
    # Step 1: Generate 3D coordinates
    mol_3d = generate_3d_coords(smiles)
    if mol_3d is None:
        return None

    try:
        from vina import Vina

        # Step 2: Prepare ligand
        ligand_pdbqt = mol_to_pdbqt_string(mol_3d)
        if ligand_pdbqt is None:
            return None

        # Step 3: Set up Vina
        v = Vina(sf_name="vina")

        # Write temporary files for Vina
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdbqt", delete=False
        ) as f:
            f.write(ligand_pdbqt)
            ligand_path = f.name

        v.set_receptor(receptor_path)
        v.set_ligand_from_file(ligand_path)
        v.compute_vina_maps(center=center, box_size=box_size)

        # Step 4: Dock
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        energies = v.energies()

        # Clean up temp file
        os.unlink(ligand_path)

        # Best binding affinity (most negative = best)
        best_affinity = energies[0][0] if len(energies) > 0 else None

        return {
            "smiles": smiles,
            "binding_affinity_kcal_mol": round(best_affinity, 2),
            "num_poses": len(energies),
        }

    except ImportError:
        warnings.warn(
            "vina not installed. Install with: pip install vina\n"
            "Returning a placeholder score based on molecular properties."
        )
        return _fallback_scoring(smiles, mol_3d)

    except Exception as e:
        warnings.warn(f"Docking failed for {smiles}: {e}")
        return None


def _fallback_scoring(smiles, mol):
    """
    Simple heuristic scoring when Vina is not available.

    Uses a rough estimate based on molecular properties known to correlate
    with binding affinity. THIS IS NOT A SUBSTITUTE FOR REAL DOCKING --
    it's just a placeholder so you can test the pipeline end-to-end.
    """
    from rdkit.Chem import Descriptors

    mol_noh = Chem.RemoveHs(mol)
    mw = Descriptors.MolWt(mol_noh)
    logp = Descriptors.MolLogP(mol_noh)
    hba = Descriptors.NumHAcceptors(mol_noh)
    hbd = Descriptors.NumHDonors(mol_noh)
    rotatable = Descriptors.NumRotatableBonds(mol_noh)

    # Rough empirical estimate (NOT physically meaningful)
    # Larger molecules with moderate logP tend to bind better
    score = -0.01 * mw - 0.5 * logp - 0.3 * (hba + hbd) + 0.1 * rotatable
    score = max(min(score, -1.0), -15.0)  # clamp to reasonable range

    return {
        "smiles": smiles,
        "binding_affinity_kcal_mol": round(score, 2),
        "num_poses": 0,
        "method": "heuristic_fallback",
    }


def dock_batch(
    smiles_list: list,
    receptor_path: str,
    center: list,
    box_size: list,
    exhaustiveness: int = 8,
):
    """
    Dock a batch of molecules, returning results sorted by binding affinity.
    """
    results = []
    for i, smiles in enumerate(smiles_list):
        print(f"Docking molecule {i+1}/{len(smiles_list)}: {smiles[:50]}...")
        result = dock_molecule(
            smiles, receptor_path, center, box_size, exhaustiveness
        )
        if result is not None:
            results.append(result)

    # Sort by binding affinity (most negative first = best binders)
    results.sort(key=lambda x: x["binding_affinity_kcal_mol"])
    return results
