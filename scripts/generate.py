"""
Generate Novel Molecules
========================
Uses the trained diffusion model to generate new molecular structures
by running the reverse diffusion process (denoising from pure noise).

Pipeline:
    1. Load trained score network from checkpoint
    2. Run reverse diffusion: noise -> molecules
    3. Discretize continuous tensors to atom types / bond types
    4. Validate with RDKit
    5. Check novelty against training set
    6. Compute drug-likeness properties
    7. Save results

Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt --num_samples 1000

    On Bouchet:
    sbatch scripts/slurm_generate.sh
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.featurizer import graph_to_smiles, graph_to_mol, MAX_ATOMS, NUM_ATOM_TYPES
from src.model.score_network import ScoreNetwork
from src.model.diffusion import GaussianDiffusion
from src.chemistry.validity import compute_validity_metrics
from src.chemistry.properties import compute_properties


def generate(checkpoint_path: str, num_samples: int, output_path: str):
    # ---- Load Checkpoint ----
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    data_cfg = config["data"]
    model_cfg = config["model"]
    diff_cfg = config["diffusion"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Rebuild Model ----
    atom_feature_dim = data_cfg.get("atom_feature_dim", data_cfg["num_atom_types"])
    guidance_cfg = config.get("guidance", {})
    num_classes = guidance_cfg.get("num_classes", 3)
    model = ScoreNetwork(
        num_atom_types=data_cfg["num_atom_types"],
        num_bond_types=data_cfg["num_bond_types"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        atom_feature_dim=atom_feature_dim,
        num_classes=num_classes,
    ).to(device)

    # Prefer EMA weights (smoother, better generation quality)
    if "ema_state_dict" in checkpoint:
        ema_sd = checkpoint["ema_state_dict"]
        model_sd = model.state_dict()
        for name in ema_sd:
            if name in model_sd:
                model_sd[name] = ema_sd[name]
        model.load_state_dict(model_sd)
        print(f"Model loaded with EMA weights (trained for {checkpoint['epoch']} epochs)")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    model.eval()

    # ---- Rebuild Diffusion ----
    diffusion = GaussianDiffusion(
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        schedule=diff_cfg.get("schedule", "cosine"),
        device=device,
    )

    # ---- Generate ----
    gen_cfg = config.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 256)

    sampler = gen_cfg.get("sampler", "ddpm")
    temperature = gen_cfg.get("temperature", 1.0)
    guidance_scale = guidance_cfg.get("guidance_scale", 1.0)
    guide_class = guidance_cfg.get("guide_class", 2)  # default: high QED
    print(f"\nGenerating {num_samples} molecules (sampler={sampler}, temp={temperature}, "
          f"guidance_scale={guidance_scale}, class={guide_class}) ...")
    all_X, all_A = [], []
    remaining = num_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        # Create guidance labels (all same class for targeted generation)
        if guidance_scale > 1.0:
            labels = torch.full((bs,), guide_class, device=device, dtype=torch.long)
        else:
            labels = None

        if sampler == "ddim":
            X_gen, A_gen = diffusion.ddim_sample(
                model,
                num_samples=bs,
                max_atoms=data_cfg["max_atoms"],
                num_atom_types=atom_feature_dim,
                num_bond_types=data_cfg["num_bond_types"],
                num_inference_steps=gen_cfg.get("ddim_steps", 100),
                eta=gen_cfg.get("ddim_eta", 0.0),
                temperature=temperature,
                labels=labels,
                guidance_scale=guidance_scale,
                num_classes=num_classes,
            )
        else:
            X_gen, A_gen = diffusion.sample(
                model,
                num_samples=bs,
                max_atoms=data_cfg["max_atoms"],
                num_atom_types=atom_feature_dim,
                num_bond_types=data_cfg["num_bond_types"],
                labels=labels,
                guidance_scale=guidance_scale,
                num_classes=num_classes,
            )
        all_X.append(X_gen.cpu())
        all_A.append(A_gen.cpu())
        remaining -= bs

    X_all = torch.cat(all_X, dim=0)
    A_all = torch.cat(all_A, dim=0)

    # ---- Convert to Molecules & Validate ----
    print("\nConverting to molecules and validating ...")

    # Create masks: nodes whose atom-type argmax is NOT the virtual type are "real"
    # Only look at the first num_atom_types dims (atom type one-hot portion)
    nat = data_cfg["num_atom_types"]
    virtual_type_idx = nat - 1
    masks = (X_all[:, :, :nat].argmax(dim=-1) != virtual_type_idx).float()

    mol_list = []
    smiles_list = []
    for i in range(num_samples):
        mol = graph_to_mol(X_all[i], A_all[i], masks[i])
        mol_list.append(mol)
        smiles_list.append(graph_to_smiles(X_all[i], A_all[i], masks[i]))

    # Load training SMILES for novelty check
    train_smiles_path = os.path.join(data_cfg["processed_path"], "train_smiles.txt")
    training_smiles = set()
    if os.path.exists(train_smiles_path):
        with open(train_smiles_path) as f:
            training_smiles = set(line.strip() for line in f)
        print(f"Loaded {len(training_smiles)} training SMILES for novelty check")

    # ---- Compute Metrics ----
    metrics = compute_validity_metrics(mol_list, training_smiles)

    print(f"\n{'='*50}")
    print(f"GENERATION RESULTS")
    print(f"{'='*50}")
    print(f"Total generated:  {metrics['total_generated']}")
    print(f"Valid:            {metrics['num_valid']} ({metrics['validity']:.1%})")
    print(f"Unique:           {metrics['num_unique']} ({metrics['uniqueness']:.1%})")
    if metrics["novelty"] is not None:
        print(f"Novel:            {metrics['num_novel']} ({metrics['novelty']:.1%})")
    print(f"Atom distribution: {metrics['atom_distribution']}")

    # ---- Compute Properties for Valid Molecules ----
    print("\nComputing drug-likeness properties ...")
    results = []
    for smiles in metrics["novel_smiles"]:
        props = compute_properties(smiles)
        if props:
            props["novel"] = True
            results.append(props)

    # ---- Save Results ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if results:
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame(columns=["smiles", "qed", "num_atoms", "molecular_weight", "logp", "novel"])
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} molecules with properties to {output_path}")

    # Print top candidates by QED
    if len(df) > 0:
        print(f"\nTop 10 by QED (drug-likeness):")
        top = df.nlargest(10, "qed")
        for _, row in top.iterrows():
            print(
                f"  QED={row['qed']:.3f}  MW={row['molecular_weight']:.0f}  "
                f"LogP={row['logp']:.2f}  {row['smiles'][:60]}"
            )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output", default="results/generated.csv")
    args = parser.parse_args()
    generate(args.checkpoint, args.num_samples, args.output)
