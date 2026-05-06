#!/bin/bash
# ============================================================================
# Reproduce the 3-seed study (seeds 42, 123, 456) with λ=0.02.
# This is the main thesis result: mean AUROC 0.9833.
#
# Usage:
#   bash scripts/run_three_seeds.sh
#   GPU=1 bash scripts/run_three_seeds.sh        # use a specific GPU
#   OUTPUT_DIR=outputs/my_run bash scripts/run_three_seeds.sh
#
# Expected time: ~6–8 hours per seed, ~18–24 hours total on a 32 GB GPU.
# Each seed checkpoints independently — you can rerun a failed seed alone.
#
# Expected results (val AUROC at best checkpoint):
#   seed=42  → 0.9873  (test AUROC 0.9898)
#   seed=123 → 0.9886  (test AUROC 0.9914)
#   seed=456 → 0.9887  (test AUROC 0.9686)
#   Mean     → 0.9882  (mean test AUROC 0.9833)
# ============================================================================

set -e

GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/three_seeds}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=============================================="
echo "DiffusionOOD — 3-Seed Study (λ=0.02)"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

for SEED in 42 123 456; do
    echo ""
    echo "=============================================="
    echo "[$(date)] Starting seed $SEED (λ=0.02)"
    echo "=============================================="

    python scripts/train.py \
        --seed "$SEED" \
        --separation_loss_weight 0.02 \
        --batch_size 64 \
        --max_epochs 200 \
        --eval_interval 10 \
        --scoring_method difference \
        --timestep_mode uniform \
        --experiment_tag "thesis_seed${SEED}" \
        --wandb_mode disabled \
        --output_dir "${OUTPUT_DIR}/seed${SEED}"

    echo "[$(date)] Seed $SEED DONE"
    echo ""
done

echo "=============================================="
echo "All 3 seeds complete: $(date)"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "To compute mean AUROC across seeds, check the best_auroc values"
echo "in each seed's checkpoint filename or metrics.json."
