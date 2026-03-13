"""
Dock Generated Molecules Against Target Protein
================================================
Takes the generated molecules and docks them against a target protein
(EGFR kinase by default) to predict binding affinity.

Pipeline:
    1. Load generated molecules from CSV
    2. Download and prepare the target protein
    3. Dock each molecule using AutoDock Vina
    4. Rank by binding affinity (most negative = best)
    5. Select top 5 candidates

The final deliverable: 5 novel, chemically valid molecules with
predicted binding affinity scores.

Usage:
    python scripts/dock.py --molecules results/generated.csv

    On Bouchet:
    sbatch scripts/slurm_dock.sh
"""

import os
import sys
import argparse
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.docking import download_protein, prepare_receptor, dock_batch
from src.chemistry.properties import compute_properties


def dock(
    molecules_path: str,
    config_path: str = "configs/default.yaml",
    top_k: int = 5,
):
    # ---- Load Config ----
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dock_cfg = config["docking"]

    # ---- Load Generated Molecules ----
    print(f"Loading molecules from {molecules_path} ...")
    df = pd.read_csv(molecules_path)
    smiles_list = df["smiles"].tolist()
    print(f"Loaded {len(smiles_list)} molecules")

    # Filter for drug-like molecules (Lipinski passes) to reduce docking time
    print("Pre-filtering for drug-likeness ...")
    drug_like = []
    for smi in smiles_list:
        props = compute_properties(smi)
        if props and props["passes"]:
            drug_like.append(smi)

    print(f"Drug-like molecules: {len(drug_like)} / {len(smiles_list)}")

    if len(drug_like) == 0:
        print("No drug-like molecules found. Using all valid molecules instead.")
        drug_like = smiles_list

    # Limit to a reasonable number for docking (it's slow)
    max_dock = min(len(drug_like), 100)
    dock_candidates = drug_like[:max_dock]
    print(f"Docking {max_dock} molecules ...")

    # ---- Prepare Protein ----
    pdb_id = dock_cfg["target_pdb_id"]
    print(f"\nPreparing target protein: {dock_cfg['target_name']} (PDB: {pdb_id})")

    pdb_path = download_protein(pdb_id)
    receptor_path = prepare_receptor(pdb_path)

    # ---- Run Docking ----
    print(f"\nDocking against {dock_cfg['target_name']} ...")
    print(f"  Binding site center: {dock_cfg['center']}")
    print(f"  Search box: {dock_cfg['box_size']}")
    print(f"  Exhaustiveness: {dock_cfg['exhaustiveness']}")

    results = dock_batch(
        dock_candidates,
        receptor_path,
        center=dock_cfg["center"],
        box_size=dock_cfg["box_size"],
        exhaustiveness=dock_cfg["exhaustiveness"],
    )

    if not results:
        print("No successful docking results. Check dependencies (vina, meeko).")
        return

    # ---- Save All Results ----
    output_path = dock_cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nAll docking results saved to {output_path}")

    # ---- Top K Results ----
    print(f"\n{'='*60}")
    print(f"TOP {top_k} MOLECULES BY BINDING AFFINITY")
    print(f"Target: {dock_cfg['target_name']} (PDB: {pdb_id})")
    print(f"{'='*60}")

    for i, result in enumerate(results[:top_k]):
        print(f"\n  #{i+1}")
        print(f"  SMILES:           {result['smiles']}")
        print(f"  Binding Affinity: {result['binding_affinity_kcal_mol']:.2f} kcal/mol")

        # Also compute properties for the top candidates
        props = compute_properties(result["smiles"])
        if props:
            print(f"  Mol. Weight:      {props['molecular_weight']:.1f} Da")
            print(f"  LogP:             {props['logp']:.2f}")
            print(f"  QED:              {props['qed']:.3f}")
            print(f"  H-Bond Donors:    {props['h_bond_donors']}")
            print(f"  H-Bond Acceptors: {props['h_bond_acceptors']}")
            print(f"  Lipinski:         {'PASS' if props['passes'] else 'FAIL'}")

    # Save top K separately for the final report
    top_path = output_path.replace(".csv", f"_top{top_k}.csv")
    top_df = pd.DataFrame(results[:top_k])
    top_df.to_csv(top_path, index=False)
    print(f"\nTop {top_k} results saved to {top_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecules", default="results/generated.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    dock(args.molecules, args.config, args.top_k)
