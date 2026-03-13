"""
Dock Generated Molecules Against Target Protein
================================================
Takes the generated molecules and docks them against a target protein
(EGFR kinase by default) to predict binding affinity.

Pipeline:
    1. Load generated molecules from CSV
    2. Multi-stage filtering: Lipinski -> SA score -> TPSA -> QED
    3. Download and prepare the target protein
    4. Dock each molecule using AutoDock Vina
    5. Rank by binding affinity (most negative = best)
    6. Select top 5 candidates

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


def _filter_molecules(smiles_list):
    """Multi-stage filtering pipeline with statistics at each stage."""
    print(f"\n{'='*50}")
    print("FILTERING PIPELINE")
    print(f"{'='*50}")
    print(f"  Input molecules:     {len(smiles_list)}")

    # Compute all properties once
    all_props = []
    for smi in smiles_list:
        props = compute_properties(smi)
        if props:
            all_props.append(props)

    print(f"  Valid (parseable):   {len(all_props)}")
    if not all_props:
        return []

    # Stage 1: Lipinski
    lipinski_pass = [p for p in all_props if p["passes"]]
    print(f"  Lipinski pass:       {len(lipinski_pass)}")

    # Stage 2: SA score < 5 (synthesizable)
    sa_pass = [p for p in lipinski_pass if p.get("sa_score", 10) < 5]
    print(f"  SA score < 5:        {len(sa_pass)}")

    # Stage 3: TPSA 40-140 (good oral bioavailability)
    tpsa_pass = [p for p in sa_pass if 40 <= p.get("tpsa", 0) <= 140]
    print(f"  TPSA 40-140:         {len(tpsa_pass)}")

    # Stage 4: QED > 0.4 (drug-like)
    qed_pass = [p for p in tpsa_pass if p["qed"] > 0.4]
    print(f"  QED > 0.4:           {len(qed_pass)}")

    # Fall back to less filtered sets if too few candidates survive
    if len(qed_pass) >= 5:
        candidates = qed_pass
    elif len(tpsa_pass) >= 5:
        print("  (Relaxing QED filter)")
        candidates = tpsa_pass
    elif len(sa_pass) >= 5:
        print("  (Relaxing TPSA filter)")
        candidates = sa_pass
    elif len(lipinski_pass) >= 5:
        print("  (Relaxing SA filter)")
        candidates = lipinski_pass
    else:
        print("  (Using all valid molecules)")
        candidates = all_props

    # Sort by QED descending
    candidates.sort(key=lambda p: p["qed"], reverse=True)
    max_dock = min(len(candidates), 200)
    print(f"  Docking candidates:  {max_dock}")
    print(f"{'='*50}\n")

    return [p["smiles"] for p in candidates[:max_dock]]


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
    if df.empty or "smiles" not in df.columns:
        print("No molecules found in input file. Run generation first with a trained model.")
        return
    smiles_list = df["smiles"].dropna().tolist()
    if not smiles_list:
        print("No valid SMILES found in input file.")
        return
    print(f"Loaded {len(smiles_list)} molecules")

    # ---- Multi-Stage Filtering ----
    dock_candidates = _filter_molecules(smiles_list)
    if not dock_candidates:
        print("No candidates survived filtering. Try generating more molecules.")
        return

    # ---- Prepare Protein ----
    pdb_id = dock_cfg["target_pdb_id"]
    print(f"Preparing target protein: {dock_cfg['target_name']} (PDB: {pdb_id})")

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

        props = compute_properties(result["smiles"])
        if props:
            print(f"  Mol. Weight:      {props['molecular_weight']:.1f} Da")
            print(f"  LogP:             {props['logp']:.2f}")
            print(f"  QED:              {props['qed']:.3f}")
            print(f"  SA Score:         {props.get('sa_score', 'N/A')}")
            print(f"  TPSA:             {props['tpsa']:.1f}")
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
