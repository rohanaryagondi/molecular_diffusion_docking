#!/bin/bash
#SBATCH --job-name=moldiff_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# =============================================================================
# Molecular Diffusion Model - Training Job (Bouchet)
# =============================================================================
# Submit with:  sbatch scripts/slurm_train.sh
# Monitor with: squeue -u $USER
# Check logs:   tail -f logs/train_<jobid>.out
# =============================================================================

echo "======================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "======================================"

# ---- Modules ----
module purge
module load CUDA/12.6.0

# ---- Environment ----
# Create venv once: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
source .venv/bin/activate

# Verify GPU is visible
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---- Create log directory ----
mkdir -p logs checkpoints

# ---- Train ----
python scripts/train.py --config configs/default.yaml

# ---- Push results to GitHub ----
echo "Pushing results to GitHub ..."
bash scripts/push_results.sh "Training complete (job $SLURM_JOB_ID)" || echo "Push failed (non-fatal)"

echo "======================================"
echo "End time: $(date)"
echo "======================================"
