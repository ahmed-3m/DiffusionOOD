# Algorithm: Diffusion Classifier for OOD Detection

This document provides a detailed description of the scoring algorithm and the separation loss.

---

## Core Idea

A standard diffusion model is trained to denoise images conditioned on a class label.
During training, we show the model both ID images (c=0) and OOD-proxy images (c=1).

At inference, the model has learned two different denoising functions — one for each condition.
An ID image will be denoised more accurately under c=0 than under c=1.
An OOD image will show the opposite pattern, or at least a smaller gap.

The OOD score is simply the difference in reconstruction error across the two conditions.

---

## Algorithm 1: diffusion_classifier_score

```
Input:  image x, model f_θ, scheduler, K trials
Output: OOD score s(x), predicted label ŷ

For k = 1 to K:
    sample timestep t_k ~ p(t)          # e.g. uniform over [1, T]
    sample noise ε_k ~ N(0, I)
    compute noisy image: x̃_k = sqrt(ᾱ_t) · x + sqrt(1 - ᾱ_t) · ε_k

    predict noise under ID condition:
        ε̂_0 = f_θ(x̃_k, t_k, c=0)
    predict noise under OOD condition:
        ε̂_1 = f_θ(x̃_k, t_k, c=1)

    compute per-condition MSE:
        e_0[k] = ||ε̂_0 - ε_k||²
        e_1[k] = ||ε̂_1 - ε_k||²

Average across K trials:
    Ē_0 = mean_k(e_0[k])
    Ē_1 = mean_k(e_1[k])

OOD score:    s(x) = Ē_0 - Ē_1
Prediction:   ŷ = argmin_c(Ē_c)     # 0=ID, 1=OOD
```

**Scoring methods:**

| Method | Formula | Notes |
|--------|---------|-------|
| `difference` (default) | `Ē_0 - Ē_1` | Lower FPR@95 within-CIFAR |
| `ratio` | `Ē_0 / (Ē_1 + ε)` | Better on some external sets |
| `id_error` | `Ē_0` only | Collapses on SVHN; not recommended |

---

## Separation Loss

Without an explicit training signal, both conditions can converge to similar predictions — the model learns a marginal distribution over images rather than truly class-conditional distributions.

The separation loss fixes this by maximising the divergence between the two denoising predictions on the same noisy input:

```
L_sep = -MSE(f_θ(x̃, t, c=0), f_θ(x̃, t, c=1))
```

The total training loss is:

```
L_total = L_MSE + λ · L_sep
```

where `L_MSE` is the standard denoising objective over the correct class label, and `λ` controls how strongly the two conditions are pushed apart.

**Effect of λ:**

- `λ=0`: Both conditions collapse to similar predictions → AUROC ≈ 80%.
- `λ=0.001`: Even a very small signal separates the conditions → AUROC jumps to ~97%.
- `λ=0.02`: Optimal balance → AUROC = 99.03% ± 0.07% (three-seed average).
- `λ=0.10`: Separation objective dominates denoising → training destabilises → AUROC drops.

---

## Timestep Sampling

The choice of which timesteps to sample affects the quality of the OOD signal.

| Strategy | Description | Within-CIFAR AUROC | SVHN AUROC |
|----------|-------------|---------------------|------------|
| `uniform` | t ~ U[1, 1000] | **98.9%** | **95.4%** |
| `stratified` | Equal-width bins | 98.8% | 95.0% |
| `mid_focus` | Truncated N(μ=300, σ=150) | 98.5% | 93.8% |

Uniform sampling performs best. Restricting to intermediate timesteps (mid-focus) discards useful signal at both the low- and high-noise extremes.

---

## Monte Carlo Trials (K)

The OOD score is averaged over K random timestep samples to reduce variance.

| K | AUROC | Time / 10K images |
|---|-------|-------------------|
| 1 | 91.0% | ~2 min |
| 10 | 98.2% | ~16 min |
| 25 | 98.5% | ~40 min |
| 50 | 98.9% | ~81 min |
| 100 | 98.95% | ~162 min |

Performance saturates after K=25. **K=10 is the recommended default** for latency-sensitive applications.
