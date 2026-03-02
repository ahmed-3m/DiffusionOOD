#!/bin/bash
# ============================================================
# MASTER GAP-COMPLETION SCRIPT
# Completes all 4 gaps for the thesis in order.
# Usage:
#   tmux new -s gaps
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_all_gaps.sh 2>&1 | tee results/gaps.log
# ============================================================
set -e

CONDA_ENV="/system/apps/studentenv/mohammed/sdm"
PROJECT="/system/user/studentwork/mohammed/2025/diffusion_classifier_ood"
cd "$PROJECT"

echo "============================================================"
echo "  GAP COMPLETION — started $(date)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── STEP 1: GAP 1 + GAP 2 (no GPU — pure Python math) ───────
echo ""
echo ">>> STEP 1: GAP 1 + GAP 2 (no GPU: raw_scores → JSONs)"
conda run --no-capture-output -p "$CONDA_ENV" \
    python scripts/compute_gap1_gap2.py
echo "STEP 1 DONE: $(date)"

# ── STEP 2: GAP 3 (GPU — SVHN eval on 4 sep-ablation ckpts) ─
echo ""
echo ">>> STEP 2: GAP 3 (GPU: SVHN eval for lambda=0.0/0.001/0.05/0.1)"
conda run --no-capture-output -p "$CONDA_ENV" \
    python scripts/eval_svhn_sep_ablation.py
echo "STEP 2 DONE: $(date)"

# ── STEP 3: GAP 4 (regenerate tables + figures) ─────────────
echo ""
echo ">>> STEP 3: GAP 4 (regenerate all tables and figures)"
conda run --no-capture-output -p "$CONDA_ENV" \
    python scripts/generate_all_figures_tables.py
echo "STEP 3 DONE: $(date)"

echo ""
echo "============================================================"
echo "  ALL GAPS COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Check outputs:"
echo "  cat results_json/separation_loss_results.json"
echo "  cat results_json/external_ood_results.json"
echo "  ls results/figures/*.png"
