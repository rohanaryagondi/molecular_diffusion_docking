#!/bin/bash
# =============================================================================
# Push Results to GitHub
# =============================================================================
# Commits training logs, generation metrics, and docking results to the repo
# so you can monitor progress remotely without SSH-ing into Bouchet.
#
# Usage:
#   bash scripts/push_results.sh                  # auto-detect stage
#   bash scripts/push_results.sh "training done"  # custom message
# =============================================================================

set -e

cd "$(dirname "$0")/.."

MSG="${1:-Auto-push results from Bouchet ($(date '+%Y-%m-%d %H:%M'))}"
RUN_DIR="runs/$(date '+%Y%m%d_%H%M%S')"

mkdir -p "$RUN_DIR"

# ---- Collect training artifacts ----
if [ -f checkpoints/best.pt ]; then
    # Extract metrics from checkpoint (epoch, loss) without copying the big .pt
    python -c "
import torch, json
ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
info = {
    'epoch': ckpt.get('epoch', '?'),
    'val_loss': round(ckpt.get('val_loss', 0), 4),
    'config': ckpt.get('config', {}),
}
with open('$RUN_DIR/training_summary.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'Training summary: epoch={info[\"epoch\"]}, val_loss={info[\"val_loss\"]}')
" 2>/dev/null && echo "Saved training summary" || echo "Skipped training summary"
fi

# ---- Collect latest training log ----
LATEST_TRAIN_LOG=$(ls -t logs/train_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_TRAIN_LOG" ]; then
    # Save last 200 lines (epoch summaries, losses, validity checks)
    tail -200 "$LATEST_TRAIN_LOG" > "$RUN_DIR/train_log_tail.txt"
    # Extract key metrics lines
    grep -E "(Epoch |Val Loss|Validity|best model|Training complete)" "$LATEST_TRAIN_LOG" > "$RUN_DIR/train_metrics.txt" 2>/dev/null || true
    echo "Saved training log excerpts"
fi

# ---- Collect generation results ----
if [ -f results/generated.csv ]; then
    cp results/generated.csv "$RUN_DIR/generated.csv"
    echo "Saved generated molecules"
fi

# Collect generation log
LATEST_GEN_LOG=$(ls -t logs/generate_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_GEN_LOG" ]; then
    tail -50 "$LATEST_GEN_LOG" > "$RUN_DIR/generate_log_tail.txt"
    grep -E "(GENERATION RESULTS|Valid:|Unique:|Novel:|Top 10)" "$LATEST_GEN_LOG" > "$RUN_DIR/generation_metrics.txt" 2>/dev/null || true
    echo "Saved generation log"
fi

# ---- Collect docking results ----
if [ -f results/docking_results.csv ]; then
    cp results/docking_results.csv "$RUN_DIR/docking_results.csv"
    echo "Saved docking results"
fi

LATEST_DOCK_LOG=$(ls -t logs/dock_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_DOCK_LOG" ]; then
    tail -50 "$LATEST_DOCK_LOG" > "$RUN_DIR/dock_log_tail.txt"
    echo "Saved docking log"
fi

# ---- Collect any error logs ----
for errlog in logs/*_*.err; do
    if [ -f "$errlog" ] && [ -s "$errlog" ]; then
        basename_log=$(basename "$errlog")
        cp "$errlog" "$RUN_DIR/error_${basename_log}"
        echo "Saved error log: $basename_log"
    fi
done

# ---- Create a summary file ----
cat > "$RUN_DIR/README.txt" <<SUMMARY
Run: $(date '+%Y-%m-%d %H:%M:%S')
Host: $(hostname)
Job IDs: $(ls logs/*.out 2>/dev/null | sed 's/.*_\([0-9]*\)\.out/\1/' | sort -u | tail -3 | tr '\n' ' ')

Files in this run:
$(ls "$RUN_DIR/" | sed 's/^/  - /')
SUMMARY

# ---- Git commit and push ----
git add "$RUN_DIR/"
git commit -m "$(cat <<EOF
$MSG

Run dir: $RUN_DIR
Host: $(hostname)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)" || { echo "Nothing new to commit"; exit 0; }

git push origin main
echo ""
echo "Results pushed to GitHub: $RUN_DIR"
