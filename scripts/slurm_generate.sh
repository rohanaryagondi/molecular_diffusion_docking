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

echo "======================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
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
