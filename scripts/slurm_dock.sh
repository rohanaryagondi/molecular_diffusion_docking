#!/bin/bash
#SBATCH --job-name=moldiff_dock
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dock_%j.out
#SBATCH --error=logs/dock_%j.err

# =============================================================================
# Molecular Docking Job (Bouchet) - CPU only
# =============================================================================
# AutoDock Vina is CPU-based, so no GPU needed.
# Run AFTER generation completes.
# Submit with:  sbatch scripts/slurm_dock.sh
# =============================================================================

echo "======================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

module purge

source .venv/bin/activate

mkdir -p results data/protein

python scripts/dock.py \
    --molecules results/generated.csv \
    --config configs/default.yaml \
    --top_k 5

echo "======================================"
echo "End time: $(date)"
echo "======================================"
