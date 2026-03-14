# Molecular Diffusion Docking

Score-based generative modeling (DDPM) for *de novo* molecular design, with docking against EGFR kinase.

## What This Does

1. **Trains a diffusion model** on ~250K drug-like molecules (ZINC250K)
2. **Generates novel molecules** by denoising from random Gaussian noise
3. **Validates** chemical validity, uniqueness, and novelty
4. **Filters** by drug-likeness (Lipinski, SA score, QED, TPSA)
5. **Docks** top candidates against EGFR kinase to predict binding affinity
6. **Selects top 5** molecules by binding score

## Features

- **Graph Transformer** architecture with edge-type bias (~5M params)
- **Cosine noise schedule** (Nichol & Dhariwal 2021) for smoother SNR
- **Categorical adjacency** — one-hot bond types (none/single/double/triple/aromatic)
- **Enhanced atom features** — 28-dim (atom type + charge + degree + implicit H + aromaticity + ring)
- **DDIM sampling** — 10x faster generation with deterministic denoising
- **Classifier-free guidance** — QED-conditioned generation for high drug-likeness
- **EMA weights** — smoother generation from exponential moving average
- **Importance-weighted timestep sampling** — focus training on noisier steps
- **Multi-stage filtering** — Lipinski → SA < 5 → TPSA 40-140 → QED > 0.4
- **Multi-conformer docking** — multiple 3D poses per ligand, best energy selected
- **Auto-push results** — metrics pushed to GitHub after each pipeline stage

## Quick Start

### Local Setup
```bash
git clone https://github.com/rohanaryagondi/molecular_diffusion_docking.git
cd molecular_diffusion_docking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Bouchet HPC (Yale YCRC)
```bash
# On login node — clone to scratch (home has limited quota):
cd ~/scratch60
git clone https://github.com/rohanaryagondi/molecular_diffusion_docking.git
cd molecular_diffusion_docking

# Set up environment:
module load CUDA/12.6.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Configure git for auto-push (needed for push_results.sh):
git config user.name "Your Name"
git config user.email "your.email@yale.edu"
```

## Running the Pipeline

### Option A: Full Pipeline (Single Job)
Runs everything end-to-end and auto-pushes results to GitHub:
```bash
sbatch scripts/slurm_full_pipeline.sh
```

### Option B: Step by Step

```bash
# Step 1: Download ZINC250K (login node, ~1 min)
python scripts/download_data.py

# Step 2: Preprocess SMILES -> graph tensors + QED labels (CPU, ~5-10 min)
sbatch scripts/slurm_preprocess.sh
# or locally: python scripts/preprocess.py --config configs/default.yaml

# Step 3: Train diffusion model (GPU, ~4-8 hours on A100)
sbatch scripts/slurm_train.sh

# Step 4: Generate 1000 molecules with DDIM + guidance (GPU, ~30 min)
sbatch scripts/slurm_generate.sh

# Step 5: Filter + dock against EGFR (CPU, ~2-4 hours)
sbatch scripts/slurm_dock.sh
```

### Monitoring from Anywhere
Each slurm job auto-pushes results to `runs/` on GitHub after completion.
Check `runs/<timestamp>/` for:
- `training_summary.json` — epoch, val loss, config
- `train_metrics.txt` — per-epoch loss and validity checks
- `generated.csv` — all valid generated molecules with properties
- `docking_results.csv` — binding affinities for top candidates
- Error logs if anything failed

To manually push results at any time:
```bash
bash scripts/push_results.sh "my custom message"
```

### Local Testing (Quick Smoke Test)
```bash
# Use small config (200 molecules, 3 epochs, ~1 min)
python scripts/preprocess.py --config configs/test_local.yaml
python scripts/train.py --config configs/test_local.yaml
python scripts/generate.py --checkpoint checkpoints/best.pt --num_samples 20 --output results/generated.csv
python scripts/dock.py --molecules results/generated.csv --config configs/test_local.yaml
```

## Configuration

Two config files in `configs/`:

| Config | Purpose | Dataset | Epochs | Timesteps |
|--------|---------|---------|--------|-----------|
| `default.yaml` | Full training (A100) | ZINC250K | 200 | 1000 |
| `test_local.yaml` | Local smoke test | ZINC250K | 3 | 50 |

Key parameters in `default.yaml`:
- `guidance.guidance_scale: 3.0` — strength of QED guidance (higher = more drug-like)
- `guidance.guide_class: 2` — target QED bucket (0=low, 1=med, 2=high)
- `generation.temperature: 0.8` — lower = more conservative/valid molecules
- `generation.sampler: ddim` — fast sampling (100 steps vs 1000)
- `diffusion.schedule: cosine` — noise schedule type

## Project Structure
```
src/
  data/
    featurizer.py       SMILES <-> molecular graph tensor conversion
    dataset.py          PyTorch Dataset with optional QED labels
  model/
    layers.py           Graph Transformer with edge bias + sinusoidal time embed
    score_network.py    Noise prediction GNN with label conditioning
    diffusion.py        DDPM/DDIM forward/reverse + classifier-free guidance
  chemistry/
    validity.py         RDKit chemical validation
    properties.py       QED, SA score, Lipinski rule of five
    docking.py          AutoDock Vina interface + multi-conformer support
scripts/
  download_data.py      Fetch ZINC250K from URL
  preprocess.py         SMILES -> tensors + QED bucket labels
  train.py              Training loop with EMA, warmup, validity checks
  generate.py           DDIM generation with classifier-free guidance
  dock.py               Multi-stage filtering + Vina docking
  push_results.sh       Commit & push metrics to GitHub
  slurm_preprocess.sh   Slurm: preprocessing (CPU)
  slurm_train.sh        Slurm: training (GPU)
  slurm_generate.sh     Slurm: generation (GPU)
  slurm_dock.sh         Slurm: docking (CPU)
  slurm_full_pipeline.sh  Slurm: everything in one job
configs/
  default.yaml          Full training hyperparameters
  test_local.yaml       Tiny config for local testing
runs/                   Auto-pushed results from HPC (tracked in git)
```

## Key Concepts

- **SMILES**: Text notation for molecules (`CC(=O)O` = acetic acid)
- **Molecular Graph**: Atoms=nodes, bonds=edges, represented as dense tensors
- **DDPM**: Forward process adds noise, reverse process (learned) removes it
- **DDIM**: Deterministic skip-step sampling (10x faster than DDPM)
- **Classifier-Free Guidance**: Train with random label dropout, guide generation toward high-QED
- **Score Function**: The model learns the gradient of log-probability at each noise level
- **QED**: Quantitative Estimate of Drug-likeness (0-1, higher=better)
- **SA Score**: Synthetic Accessibility (1-10, lower=easier to synthesize)
- **Docking**: Predicts how well a molecule fits a protein's binding pocket (kcal/mol)

## Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| RDKit | Molecular manipulation & validation |
| AutoDock Vina | Binding affinity scoring |
| meeko | Ligand preparation for Vina |
| BioPython | Protein structure downloading |
