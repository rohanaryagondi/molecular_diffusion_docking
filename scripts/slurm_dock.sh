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

mkdir -p results data/protein

python scripts/dock.py \
    --molecules results/generated.csv \
    --config configs/default.yaml \
    --top_k 5

# ---- Push results to GitHub ----
echo "Pushing results to GitHub ..."
bash scripts/push_results.sh "Docking complete (job $SLURM_JOB_ID)" || echo "Push failed (non-fatal)"

echo "======================================"
echo "End time: $(date)"
echo "======================================"
