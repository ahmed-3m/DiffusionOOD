# Binary Diffusion Classifier for OOD Detection — Documentation

> **Master's Thesis Project** | Mohammed | February 2026

This directory documents all experiments, results, and design decisions for the
**Binary Conditional Diffusion Model (CDM)** for Out-of-Distribution (OOD) detection.

---

## 📁 Documentation Index

| File | Description |
|------|-------------|
| [experiments.md](experiments.md) | Full log of all experiments run (training, evaluation, ablations) |
| [results_summary.md](results_summary.md) | Final results table — all metrics in one place |
| [ablation_study.md](ablation_study.md) | Separation loss ablation study in detail |
| [dataset.md](dataset.md) | Dataset description, splits, augmentation, design decisions |
| [setup.md](setup.md) | Environment setup & how to reproduce all experiments |
| [gpu_log.md](gpu_log.md) | GPU usage log across servers (student06, student07, student10) |

---

## 🎯 Project Summary

The project trains a **conditional UNet diffusion model** in a binary classification
setting (ID vs. OOD) on CIFAR-10. At inference, OOD detection is performed by
computing the difference in diffusion reconstruction error between the two class
conditions — a method called the **Diffusion Classifier Score**.

### Model Architecture
- **ConditionalUNet** — 35.7M parameters (all trainable)
- Input: 32×32×3 images
- Block channels: (128, 256, 256, 256)
- Layers per block: 2
- Attention: AttnDownBlock2D + AttnUpBlock2D (3rd level)
- Class embedding: 2 classes (binary: ID=0, OOD=1)
- Noise schedule: squaredcos_cap_v2, 1000 timesteps
- Prediction type: epsilon (noise prediction)

### Key Contribution
A novel **Separation Loss** term added to the training objective encourages the model
to produce distinctly different reconstruction errors for ID vs. OOD samples,
directly optimising the OOD detection signal.

### Best Results
- **λ=0.01 (3 seeds):** 0.9882 ± 0.0006
- **λ=0.02 (3 seeds):** **0.9903 ± 0.0007** ← confirmed optimal ⭐
- **Best single run:** 0.9911 (λ=0.02, seed 42, epoch 29)
- **Best external OOD:** SVHN AUROC=0.9658 (λ=0.02, +4.0% vs λ=0.01)

### Training Hyperparameters
- Batch size: 64 (effective: 128 with accumulate_grad_batches=2)
- Learning rate: 1e-4 (cosine schedule, 5 epoch warmup)
- Max epochs: 200 (with early stopping, patience=30)
- Precision: 16-mixed (FP16 with dynamic scaling)
- Separation loss weight: λ=0.01 (optimal, found via ablation)
- OOD scoring: 15 Monte Carlo trials, difference scoring, mid_focus timesteps

---

## 📊 Final Figures

All figures are in [`results/figures/`](../results/figures/):

| Figure | Description |
|--------|-------------|
| `separation_loss_ablation_final.png` | ⭐ Main ablation — 6 λ weights |
| `score_distributions_all.png` | OOD score distributions per seed |
| `k_ablation.png` | Effect of number of MC trials K |
| `timestep_strategy_comparison.png` | Timestep sampling strategies |
| `scoring_method_comparison.png` | Difference vs ratio vs id_error scoring |
| `calibration_curves.png` | Calibration of OOD scores |
| `confusion_matrix.png` | Confusion matrix on test set |

---

## 📐 LaTeX Tables

All tables are in [`results/latex_tables/`](../results/latex_tables/):

- `main_results_table.tex` — Mean/std AUROC across seeds (**⚠️ needs regeneration**)
- `separation_loss_table.tex` — Ablation table (6 weights) (**⚠️ partially empty, needs regeneration**)
- `k_ablation_table.tex` — K sensitivity table ✅
- `scoring_method_table.tex` — Scoring method comparison ✅
- `timestep_strategy_table.tex` — Timestep strategy comparison ✅

---

## ⚠️ Known Issues / TODOs

1. `results/external_ood_results.json` is **empty** `{}` — external OOD eval needs re-running
2. `main_results_table.tex` has empty body — needs regeneration with seed results
3. `separation_loss_table.tex` has rows with 0.0 — needs updating with final ablation data
4. Separation loss ablation ran with `num_trials=15`, but K-ablation JSON has K={1,5,10,25,50,100}
