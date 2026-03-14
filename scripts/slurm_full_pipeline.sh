#!/bin/bash
#SBATCH --job-name=moldiff_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

# =============================================================================
# Full Pipeline: Preprocess -> Train -> Generate -> Dock -> Push Results
# =============================================================================
# Runs the entire pipeline in one job. Results are automatically pushed
# to GitHub so you can monitor progress without SSH.
#
# Submit with:  sbatch scripts/slurm_full_pipeline.sh
# =============================================================================

set -e

# ---- Project directory ----
PROJECT_DIR="/nfs/roberts/project/pi_mg269/rag88/molecule_dd_module13/molecular_diffusion_docking"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

echo "======================================"
echo "FULL PIPELINE"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "Start time: $(date)"
echo "======================================"

module purge
module load CUDA/12.6.0

source .venv/bin/activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

mkdir -p logs checkpoints results data/protein

# ---- Step 1: Download Data (if needed) ----
if [ ! -f data/raw/zinc250k.csv ]; then
    echo ""
    echo "=== STEP 1: Downloading ZINC250K ==="
    python scripts/download_data.py
fi

# ---- Step 2: Preprocess (if needed or data format changed) ----
if [ ! -f data/processed/train.pt ]; then
    echo ""
    echo "=== STEP 2: Preprocessing ==="
    python scripts/preprocess.py --config configs/default.yaml
fi

# ---- Step 3: Train ----
echo ""
echo "=== STEP 3: Training ==="
python scripts/train.py --config configs/default.yaml

# Push intermediate results after training
echo ""
echo "=== Pushing training results ==="
bash scripts/push_results.sh "Training complete (job $SLURM_JOB_ID)" || echo "Push failed (non-fatal)"

# ---- Step 4: Generate ----
echo ""
echo "=== STEP 4: Generating molecules ==="
python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --num_samples 1000 \
    --output results/generated.csv

# ---- Step 5: Dock ----
echo ""
echo "=== STEP 5: Docking ==="
python scripts/dock.py \
    --molecules results/generated.csv \
    --config configs/default.yaml \
    --top_k 5

# ---- Push final results ----
echo ""
echo "=== Pushing final results ==="
bash scripts/push_results.sh "Full pipeline complete (job $SLURM_JOB_ID)"

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "End time: $(date)"
echo "======================================"
