# Molecular Diffusion Docking

Score-based generative modeling (DDPM) for *de novo* molecular design, with docking against a target protein.

## What This Does

1. **Trains a diffusion model** on ~250K drug-like molecules (ZINC250K)
2. **Generates novel molecules** by denoising from random Gaussian noise
3. **Validates** chemical validity, uniqueness, and novelty
4. **Docks** candidates against EGFR kinase to predict binding affinity
5. **Selects top 5** molecules by binding score

## Quick Start

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Bouchet HPC (Yale YCRC)
```bash
# On login node:
module load CUDA/12.6.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the Pipeline
```bash
# Step 1: Download ZINC250K (~250K SMILES strings)
python scripts/download_data.py

# Step 2: Convert SMILES -> graph tensors (CPU, ~5 min)
python scripts/preprocess.py

# Step 3: Train diffusion model (GPU, ~4-8 hours)
sbatch scripts/slurm_train.sh        # Bouchet
# python scripts/train.py             # local

# Step 4: Generate molecules (GPU, ~30 min)
sbatch scripts/slurm_generate.sh

# Step 5: Dock against EGFR (CPU, ~2-4 hours)
sbatch scripts/slurm_dock.sh
```

## Project Structure
```
src/
  data/           SMILES -> molecular graph conversion
  model/          Graph Transformer + DDPM diffusion
  chemistry/      RDKit validation, drug properties, Vina docking
scripts/          Pipeline entry points + Slurm jobs
configs/          Hyperparameter YAML files
```

## Key Concepts

- **SMILES**: Text notation for molecules (`CC(=O)O` = acetic acid)
- **Molecular Graph**: Atoms=nodes, bonds=edges, represented as dense tensors
- **DDPM**: Forward process adds noise, reverse process (learned) removes it
- **Score Function**: The model learns the gradient of log-probability at each noise level
- **Docking**: Predicts how well a molecule fits a protein's binding pocket (kcal/mol)

## Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| RDKit | Molecular manipulation & validation |
| AutoDock Vina | Binding affinity scoring |
| meeko | Ligand preparation for Vina |
| BioPython | Protein structure downloading |
