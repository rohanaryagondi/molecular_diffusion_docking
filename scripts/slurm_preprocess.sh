#!/bin/bash
#SBATCH --job-name=moldiff_prep
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# =============================================================================
# Molecular Diffusion Model - Preprocessing Job (Bouchet)
# =============================================================================
# CPU-only: converts SMILES -> graph tensors + QED labels.
# Run AFTER download_data.py, BEFORE training.
# Submit with:  sbatch scripts/slurm_preprocess.sh
# =============================================================================

# ---- Project directory ----
PROJECT_DIR="/nfs/roberts/project/pi_mg269/rag88/molecule_dd_module13/molecular_diffusion_docking"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

echo "======================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Working dir: $(pwd)"
echo "Start time: $(date)"
echo "======================================"

module purge

source .venv/bin/activate

mkdir -p data/processed logs

# ---- Download data if needed ----
if [ ! -f data/raw/zinc250k.csv ]; then
    echo "Downloading ZINC250K ..."
    python scripts/download_data.py
fi

# ---- Preprocess ----
python scripts/preprocess.py --config configs/default.yaml

echo "======================================"
echo "End time: $(date)"
echo "======================================"
