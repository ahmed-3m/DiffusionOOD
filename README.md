# DiffusionOOD — Conditional Diffusion Model for CIFAR-10 OOD Detection

[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-ahmed--3m%2FDiffusionOOD-blue)](https://huggingface.co/ahmed-3m/DiffusionOOD)
[![GitHub](https://img.shields.io/badge/GitHub-ahmed--3m%2FDiffusionOOD-black)](https://github.com/ahmed-3m/DiffusionOOD)

**Thesis:** *Conditional Diffusion Models as Generative Classifiers for Out-of-Distribution Detection in Inkjet Print Quality Control*
**Author:** Ahmed Mohammed — MSc AI, Johannes Kepler University Linz (2026)
**Supervisor:** Univ.-Prof. Dr. Sepp Hochreiter

---

## What This Does

A binary Conditional Diffusion Model (CDM) trained on a single CIFAR-10 class detects out-of-distribution images by comparing reconstruction errors under two conditions:

- **c=0 (ID)**: in-distribution class (airplane, class 0)
- **c=1 (OOD proxy)**: all other CIFAR-10 classes during training

OOD score = `E_t[ ||ε - ε_θ(x_t, t, c=ID)||² ] − E_t[ ||ε - ε_θ(x_t, t, c=OOD)||² ]`

Higher score → more likely OOD. This is **Algorithm 1** from the thesis.

**Key results (3-seed average, CIFAR-10 within-split):**
- Mean AUROC: **98.33%** (best seed: **98.98%**)
- Separation loss gain: **+18.86 pp** (λ=0: 80.25% → λ=0.02: 99.11%)

---

## Pretrained Weights on HuggingFace

All trained checkpoints: **[https://huggingface.co/ahmed-3m/DiffusionOOD](https://huggingface.co/ahmed-3m/DiffusionOOD)**

| File on HF | AUROC (val) | Test AUROC | Params |
|---|---|---|---|
| `models/main_training/seed42_best_auroc0.9873.ckpt` | 0.9873 | **0.9898** | 68.79 M |
| `models/main_training/seed123_best_auroc0.9886.ckpt` | 0.9886 | **0.9914** | 68.79 M |
| `models/main_training/seed456_best_auroc0.9887.ckpt` | 0.9887 | **0.9686** | 68.79 M |
| `models/separation_loss_ablation/sep_loss_lambda_0p0_epoch79_auroc0.8025.ckpt` | 0.8025 | — | 68.79 M |
| `models/separation_loss_ablation/sep_loss_lambda_0p02_epoch29_auroc0.9911.ckpt` | **0.9911** | — | 68.79 M |
| `models/separation_loss_ablation/sep_loss_lambda_0p1_epoch149_auroc0.9667.ckpt` | 0.9667 | — | 68.79 M |
| `models/raw_scores/seed42_cifar10_id_scores.pt` | — | 1000 ID scores | score tensor |
| `models/raw_scores/seed42_cifar10_ood_scores.pt` | — | 9000 OOD scores | score tensor |

> **Which to use for evaluation?** Start with `seed42_best_auroc0.9873.ckpt` — it gives the thesis headline result of 98.98% test AUROC.

---

## Quick Start

### Path A — Evaluate with pretrained weights (~10 min)

```bash
# 1. Clone and install
git clone https://github.com/ahmed-3m/DiffusionOOD
cd DiffusionOOD
pip install -r requirements.txt

# 2. Download pretrained checkpoint from HuggingFace
python download_weights.py

# 3. Evaluate (CIFAR-10 is auto-downloaded to ./data)
python scripts/evaluate.py \
    --checkpoint_path models/seed42_best.ckpt \
    --num_trials 10 \
    --data_dir ./data
```

Expected output: AUROC ≈ 0.989, FPR@95 ≈ 0.047.

---

### Path B — Train from scratch (~6–8 hours on 32 GB GPU)

```bash
# Single run (seed=42, λ=0.02, thesis best)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --seed 42 \
    --separation_loss_weight 0.02 \
    --batch_size 64 \
    --max_epochs 200 \
    --eval_interval 10 \
    --scoring_method difference \
    --timestep_mode uniform \
    --experiment_tag thesis_seed42 \
    --wandb_mode disabled \
    --output_dir outputs/seed42
```

---

### Path C — Reproduce the 3-seed study

```bash
bash scripts/run_three_seeds.sh
```

Expected: mean AUROC 0.9833 (seeds 42/123/456), individual: 0.9898 / 0.9914 / 0.9686.

---

## Step-by-Step Installation

### Requirements

- Python 3.9+
- CUDA GPU (tested on Quadro GV100 32 GB; minimum ~8 GB for inference, ~12 GB for training)
- ~500 MB for CIFAR-10 dataset (auto-downloaded)
- ~800 MB for all pretrained checkpoints

### Install

```bash
git clone https://github.com/ahmed-3m/DiffusionOOD
cd DiffusionOOD
pip install -r requirements.txt
```

Core packages:
- `torch>=2.0`, `torchvision>=0.15`
- `lightning>=2.1` (PyTorch Lightning)
- `diffusers>=0.25` (HuggingFace Diffusers — provides `UNet2DModel`)
- `huggingface-hub>=0.20`
- `wandb>=0.16` (optional; use `--wandb_mode disabled` to skip)

---

## Step-by-Step: Evaluate with Pretrained Weights

**Step 1 — Download weights**

```bash
python download_weights.py
```

Downloads to `models/`:
- `models/seed42_best.ckpt` — thesis headline model (AUROC 0.9898)
- `models/seed123_best.ckpt`
- `models/seed456_best.ckpt`
- `models/sep_lambda_0p0.ckpt` — baseline (AUROC 0.8025)
- `models/sep_lambda_0p02.ckpt` — separation loss best (AUROC 0.9911)

**Step 2 — Run evaluation**

```bash
python scripts/evaluate.py \
    --checkpoint_path models/seed42_best.ckpt \
    --num_trials 10 \
    --data_dir ./data
```

CIFAR-10 is automatically downloaded to `./data` on first run.

**Step 3 — View results**

```
eval_results/
├── metrics.json     ← AUROC, FPR@95, AUPR
├── scores_id.pt     ← per-sample scores for ID samples
├── scores_ood.pt    ← per-sample scores for OOD samples
└── roc_curve.png    ← ROC curve plot
```

**Step 4 — External OOD evaluation**

```bash
python scripts/evaluate_external_ood.py \
    --checkpoint_path models/seed42_best.ckpt \
    --num_trials 10 \
    --datasets cifar100 svhn places365 fashionmnist textures stl10
```

Expected (seed42): AUROC range 90.50–99.27% across external datasets.

---

## Step-by-Step: Train from Scratch

**Step 1 — Single training run (λ=0.02, thesis setting)**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --seed 42 \
    --separation_loss_weight 0.02 \
    --batch_size 64 \
    --max_epochs 200 \
    --eval_interval 10 \
    --scoring_method difference \
    --timestep_mode uniform \
    --experiment_tag thesis_seed42 \
    --wandb_mode disabled \
    --output_dir outputs/seed42
```

Key arguments:

| Argument | Thesis value | Notes |
|---|---|---|
| `--separation_loss_weight` | 0.02 | λ for class separation loss; 0.0 = DDPM baseline |
| `--scoring_method` | difference | `difference` = e₀−e₁ (best); `ratio` or `id_error` also available |
| `--timestep_mode` | uniform | `uniform` wins; `stratified` or `mid_focus` also available |
| `--batch_size` | 64 | CIFAR-10, 32×32 — fits on 8 GB GPU |
| `--max_epochs` | 200 | Best AUROC typically at epoch 20–40 with sep loss |
| `--eval_interval` | 10 | Run validation AUROC every N epochs |
| `--seed` | 42 / 123 / 456 | Set for reproducibility |

**Step 2 — Train the baseline (λ=0)**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --seed 42 \
    --separation_loss_weight 0.0 \
    --batch_size 64 \
    --max_epochs 200 \
    --experiment_tag baseline_seed42 \
    --wandb_mode disabled \
    --output_dir outputs/baseline
```

Expected: converges to AUROC ≈ 0.80 (thesis: 0.8025 at epoch 79).

---

## Step-by-Step: Separation Loss Ablation

Reproduces Figure 6.6 / Table 6.2 of the thesis (λ ∈ {0, 0.001, 0.01, 0.02, 0.05, 0.1}):

```bash
python scripts/run_ablations.py \
    --lambda_values 0.0 0.001 0.01 0.02 0.05 0.1 \
    --seed 42 \
    --max_epochs 200 \
    --output_dir outputs/sep_loss_ablation
```

Expected results (seed=42, val AUROC at best checkpoint):

| λ | Best Val AUROC | Epoch |
|---|---|---|
| 0.0 | 0.8025 | 79 |
| 0.001 | 0.9732 | 19 |
| 0.01 | 0.9869 | — |
| **0.02** | **0.9911** | **29** |
| 0.05 | 0.9851 | 19 |
| 0.1 | 0.9667 | 149 |

---

## Step-by-Step: K Ablation (Monte Carlo Trials)

Reproduces Figure 6.3 / Table 6.x of the thesis (K ∈ {1, 5, 10, 25, 50, 100}):

```bash
python scripts/evaluate.py \
    --checkpoint_path models/seed42_best.ckpt \
    --num_trials_list 1 5 10 25 50 100 \
    --output_dir outputs/k_ablation
```

Expected results (K=10 is thesis default — optimal trade-off):

| K | AUROC | Time/sample |
|---|---|---|
| 1 | 0.9100 | 0.010 s |
| 5 | 0.9724 | 0.049 s |
| **10** | **0.9819** | **0.097 s** |
| 25 | 0.9852 | 0.243 s |
| 50 | 0.9864 | 0.486 s |
| 100 | 0.9869 | 0.972 s |

---

## Results

### Main Results (3-Seed Evaluation)

| Seed | Val AUROC | Test AUROC | FPR@95 |
|---|---|---|---|
| seed=42 | 0.9873 | **0.9898** | 0.047 |
| seed=123 | 0.9886 | **0.9914** | 0.046 |
| seed=456 | 0.9887 | **0.9686** | 0.122 |
| **Mean** | 0.9882 | **0.9833** | 0.072 |

> Thesis headline result: **98.98%** AUROC (seed=42). ✅ Exact match verified from stored score tensors.

### Separation Loss Ablation (seed=42)

| λ | AUROC | Gain vs λ=0 |
|---|---|---|
| 0.0 (baseline) | 0.8025 | — |
| 0.001 | 0.9732 | +17.1 pp |
| 0.01 | 0.9869 | +18.4 pp |
| **0.02** | **0.9911** | **+18.9 pp** |
| 0.05 | 0.9851 | +18.3 pp |
| 0.1 | 0.9667 | +16.4 pp |

> Thesis reports: **+18.8 pp** ✅

### External OOD Generalization (seed=42)

| Dataset | AUROC |
|---|---|
| CIFAR-10 (within-split) | **0.9898** |
| Food101 | 0.9927 |
| CIFAR-100 | 0.9697 |
| STL-10 | 0.9521 |
| FashionMNIST | 0.9403 |
| Textures | 0.9284 |
| SVHN | 0.9050 |

---

## Architecture

```
CIFAR-10 image (32×32×3) + noisy version x_t
            │
    ┌───────▼────────────────────────────────┐
    │  UNet2DModel (HuggingFace Diffusers)   │
    │  Channels: (128, 256, 256, 256)         │
    │  Attention: at 16×16 resolution        │
    │  Class conditioning: 2 embeddings      │
    │  (c=0: ID class,  c=1: OOD proxy)      │
    └───────┬────────────────────────────────┘
            │  predicted noise ε̂
    ┌───────▼────────────────────────────────┐
    │  Algorithm 1 Scoring (K trials)        │
    │  score = mean_t[ e(x,t,c=0) − e(x,t,c=1) ]  │
    │  e(x,t,c) = ||ε − ε̂(x_t,t,c)||²       │
    └────────────────────────────────────────┘
```

- **Parameters:** 68.79 M (UNet2DModel, verified via smoke test)
- **Diffusion schedule:** Cosine cap (squaredcos_cap_v2), T=1000
- **Training:** 200 epochs, AdamW (lr=1e-4), batch=64, AMP 16-bit
- **Separation loss:** pushes ID/OOD class embeddings apart; λ=0.02 is optimal

---

## Repository Structure

```
DiffusionOOD/
├── README.md                           ← this file
├── requirements.txt
├── download_weights.py                 ← download pretrained checkpoints from HF
│
├── configs/
│   └── default.py                      ← all hyperparameters (dataclasses)
│
├── src/
│   ├── model.py                        ← UNet2DModel wrapper + model card
│   ├── data.py                         ← CIFAR-10 binary data module
│   ├── lightning_module.py             ← training loop + separation loss
│   ├── scoring.py                      ← Algorithm 1 implementation
│   ├── metrics.py                      ← AUROC, FPR@95, AUPR
│   ├── plotting.py                     ← all visualization functions
│   └── utils.py                        ← callbacks, checkpointing, HF upload
│
├── scripts/
│   ├── train.py                        ← training entry point
│   ├── evaluate.py                     ← evaluation entry point
│   ├── run_ablations.py                ← separation loss λ sweep
│   ├── evaluate_external_ood.py        ← external dataset evaluation
│   ├── generate_all_figures.py         ← regenerate all thesis figures
│   └── run_three_seeds.sh              ← reproduce the 3-seed study
│
├── tests/                              ← unit tests
│
└── assets/                             ← figures and documentation
```

---

## Companion Repository

The companion **InkjetOOD** repo applies the same CDM approach to industrial inkjet print quality control:
- GitHub: [https://github.com/ahmed-3m/InkjetOOD](https://github.com/ahmed-3m/InkjetOOD)
- HuggingFace: [https://huggingface.co/ahmed-3m/InkjetOOD](https://huggingface.co/ahmed-3m/InkjetOOD)

---

## Citation

```bibtex
@mastersthesis{mohammed2026diffusionood,
  title   = {Conditional Diffusion Models as Generative Classifiers for
             Out-of-Distribution Detection in Inkjet Print Quality Control},
  author  = {Mohammed, Ahmed},
  school  = {Johannes Kepler University Linz},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## License

MIT License — see `LICENSE` file.
