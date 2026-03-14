#!/bin/bash
#SBATCH --job-name=moldiff_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# =============================================================================
# Molecular Diffusion Model - Generation Job (Bouchet)
# =============================================================================
# Run AFTER training completes.
# Submit with:  sbatch scripts/slurm_generate.sh
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
module load CUDA/12.6.0

source .venv/bin/activate

mkdir -p results

python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --num_samples 1000 \
    --output results/generated.csv

# ---- Push results to GitHub ----
echo "Pushing results to GitHub ..."
bash scripts/push_results.sh "Generation complete (job $SLURM_JOB_ID)" || echo "Push failed (non-fatal)"

echo "======================================"
echo "End time: $(date)"
echo "======================================"
