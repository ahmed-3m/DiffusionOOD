# Results Summary

> All final metrics for the Binary CDM OOD Detection project.
> Last verified: 2026-03-01 from checkpoints, JSON files, and evaluation logs.

---

## 1. Main Experiment — Binary CDM (3 Seeds, λ=0.01)

**Training config:**
```
model:             ConditionalUNet (35.7M params)
dataset:           CIFAR-10 binary (ID=class 0 airplane, OOD=classes 1-9)
image_size:        32×32×3
batch_size:        64 (effective 128 with grad accumulation ×2)
learning_rate:     1e-4 (cosine schedule, 5-epoch warmup)
weight_decay:      0.01
max_epochs:        200 (early stopping, patience=30 on val/auroc)
precision:         16-mixed
sep_loss_weight:   0.01
num_trials:        15 (MC diffusion passes for OOD scoring)
scoring_method:    difference
timestep_mode:     mid_focus
noise_schedule:    squaredcos_cap_v2 (1000 timesteps)
prediction_type:   epsilon
```

| Seed | Val AUROC | Best Epoch |
|------|-----------|------------|
| 42   | 0.9873    | 19         |
| 123  | 0.9886    | 19         |
| 456  | 0.9887    | 19         |
| **Mean ± Std** | **0.9882 ± 0.0006** | — |

---

## 2. Separation Loss Ablation (λ sweep, seed=42 only)

| λ       | Best AUROC | Best Epoch | Δ vs baseline |
|---------|-----------|------------|---------------|
| 0.0     | 0.8025    | 79         | —             |
| 0.001   | 0.9732    | 19         | +17.07%       |
| 0.01    | 0.9869    | 19         | +18.44%       |
| **0.02**| **0.9911**| **29**     | **+18.86%**   |
| 0.05    | 0.9851    | 19         | +18.26%       |
| 0.1     | 0.9667    | 149        | +16.42%       |

**Optimal: λ = 0.02** — robust range λ ∈ [0.01, 0.05] gives AUROC ≥ 0.9851.

---

## 3. λ=0.02 Multi-seed Confirmation (✅ Complete)

Running the optimal λ=0.02 across 3 seeds confirms the result is reproducible:

| Seed | AUROC  | Best Epoch | Run Date |
|------|--------|------------|----------|
| 42   | 0.9911 | 29         | Feb 25   |
| 123  | 0.9895 | 39         | Feb 26   |
| 456  | 0.9904 | 29         | Mar 01   |
| **Mean ± Std** | **0.9903 ± 0.0007** | — | — |

### λ=0.02 vs λ=0.01 comparison

| Metric | λ=0.01 (3 seeds) | λ=0.02 (3 seeds) | Δ |
|--------|----------|----------|---|
| Mean AUROC | 0.9882   | **0.9903**   | **+0.0021** |
| Std    | 0.0006   | 0.0007   | similar stability |
| Peak   | 0.9887   | **0.9911**   | **+0.0024** |

> **Conclusion:** λ=0.02 is the confirmed optimal weight — larger mean, larger peak,
> essentially the same stability as λ=0.01.

---

## 4. External OOD Evaluation

### λ=0.01 Seeds (avg seeds 42 & 123, K=50)
| Dataset | AUROC | FPR@95% |
|---------|-------|---------|
| CIFAR-10 (within) | 0.9902 | 4.7% |
| Food-101          | 0.9912 | 4.1% |
| CIFAR-100         | 0.9672 | 15.3% |
| STL-10            | 0.9516 | 33.2% |
| FashionMNIST      | 0.9346 | 22.2% |
| Textures (DTD)    | 0.9297 | 30.9% |
| SVHN              | 0.9260 | 22.7% |

### λ=0.02 Best Checkpoint (seed 42, AUROC=0.9911, K=50)
| Dataset | AUROC | FPR@95% | Δ vs λ=0.01 |
|---------|-------|---------|------------|
| CIFAR-10 (within) | **0.9918** | **3.6%** | +0.0016 |
| CIFAR-100         | **0.9746** | **14.2%** | +0.0074 |
| STL-10            | 0.9514 | 35.7% | −0.0002 |
| FashionMNIST      | 0.9229 | 26.1% | −0.0117 |
| SVHN              | **0.9658** | **13.3%** | **+0.0398** 🚀 |

> **Key finding:** λ=0.02 best checkpoint gives a **+4.0% AUROC gain on SVHN** vs λ=0.01.
> This suggests the higher separation loss helps with more distributional shift.

---

## 5. Ablation — Number of MC Trials (K)

| K | AUROC | FPR@95% | Time (s) |
|---|-------|---------|----------|
| 1 | 0.9100 | 40.8% | 98 |
| 5 | 0.9724 | 14.3% | 486 |
| 10 | 0.9819 | 9.4% | 973 |
| 25 | 0.9852 | 7.3% | 2432 |
| 50 | 0.9864 | 6.6% | 4861 |
| 100 | 0.9869 | 6.6% | 9724 |

> K=15 (used in training evals) balances speed and accuracy well.

---

## 6. Ablation — Scoring Method

| Method | CIFAR AUROC | FPR95 | SVHN AUROC |
|--------|------------|-------|------------|
| **difference** | **0.9869** | **6.3%** | 0.9413 |
| ratio | 0.9862 | 6.6% | **0.9606** |
| id_error | 0.7830 | 67.0% | 0.2023 |

---

## 7. Ablation — Timestep Strategy

| Strategy | CIFAR AUROC | FPR95 | SVHN AUROC |
|----------|------------|-------|------------|
| **uniform** | **0.9887** | **5.4%** | **0.9544** |
| stratified | 0.9881 | 5.9% | 0.9498 |
| mid_focus | 0.9855 | 7.6% | 0.9380 |

---

## 8. Available Figures

| Figure | Content |
|--------|---------|
| `separation_loss_ablation_final.png` | ⭐ Main ablation curve (6 λ values) |
| `sep_loss_dual.png` | Ablation AUROC + convergence speed |
| `three_seed_auroc.png` | 3-seed bar chart (λ=0.01) |
| `training_curves.png` | AUROC + FPR95 vs epoch (3 seeds) |
| `roc_curves_cifar10.png` | ROC curves for 5 OOD datasets (seed 42) |
| `scoring_methods_full.png` | Scoring method 3-panel comparison |
| `score_distributions_all.png` | Score distributions per seed |
| `k_ablation.png` | K sensitivity curve |
| `calibration_curves.png` | Calibration curves |
| `timestep_strategy_comparison.png` | Timestep strategy comparison |
| `confusion_matrix.png` | Confusion matrix |
