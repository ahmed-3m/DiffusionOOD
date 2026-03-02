#!/usr/bin/env python3
"""
GAP 1 + GAP 2 (no GPU needed):
- Compute 3-seed lambda=0.02 stats -> results_json/separation_loss_results.json
- Compute external OOD mean/std from raw_scores/ -> results_json/external_ood_results.json

Run from project root:
    python scripts/compute_gap1_gap2.py
"""
import os, json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch

PROJECT = Path("/system/user/studentwork/mohammed/2025/diffusion_classifier_ood")
RAW = PROJECT / "results/raw_scores"
OUT = PROJECT / "results_json"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fpr_at_tpr95(id_scores, ood_scores):
    """FPR at 95% TPR. Higher score = more OOD."""
    tpr_target = 0.95
    thresh = np.percentile(ood_scores, (1 - tpr_target) * 100)
    fpr = np.mean(id_scores >= thresh)
    return float(fpr)

def auroc_from_scores(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))

def load_pt(path):
    return torch.load(path, map_location="cpu").numpy().flatten()

# ─────────────────────────────────────────────
# GAP 2: External OOD from raw scores
# ─────────────────────────────────────────────
EXT_DATASETS = ["svhn", "cifar100", "fashionmnist", "textures", "food101", "stl10"]
SEEDS = [42, 123, 456]

print("=== GAP 2: Computing external OOD metrics from raw scores ===\n")

per_seed = {}
for seed in SEEDS:
    sid = f"seed{seed}"
    id_path = RAW / f"{sid}_cifar10_id_scores.pt"
    if not id_path.exists():
        print(f"  MISSING: {id_path}"); continue
    id_scores = load_pt(id_path)
    per_seed[seed] = {}
    for ds in EXT_DATASETS:
        ood_path = RAW / f"{sid}_{ds}_scores.pt"
        if not ood_path.exists():
            print(f"  MISSING: {ood_path}"); continue
        ood_scores = load_pt(ood_path)
        auroc = auroc_from_scores(id_scores, ood_scores)
        fpr95 = fpr_at_tpr95(id_scores, ood_scores)
        per_seed[seed][ds] = {"auroc": round(auroc, 4), "fpr95": round(fpr95, 3)}
        print(f"  seed{seed} {ds:15s}: AUROC={auroc:.4f}  FPR95={fpr95:.3f}")
    print()

# Aggregate across seeds
ext_ood_results = {}
for ds in EXT_DATASETS:
    aurocs = [per_seed[s][ds]["auroc"] for s in SEEDS if ds in per_seed.get(s, {})]
    fprs   = [per_seed[s][ds]["fpr95"] for s in SEEDS if ds in per_seed.get(s, {})]
    if not aurocs: continue
    ext_ood_results[ds] = {
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std":  round(float(np.std(aurocs)),  4),
        "fpr95_mean": round(float(np.mean(fprs)),   3),
        "fpr95_std":  round(float(np.std(fprs)),    3),
        "per_seed":   {str(s): per_seed[s].get(ds, {}) for s in SEEDS},
    }
    print(f"  AGG {ds:15s}: AUROC={ext_ood_results[ds]['auroc_mean']:.4f}±{ext_ood_results[ds]['auroc_std']:.4f}  FPR95={ext_ood_results[ds]['fpr95_mean']:.3f}")

out_ext = OUT / "external_ood_results.json"
with open(out_ext, "w") as f:
    json.dump(ext_ood_results, f, indent=2)
print(f"\nSaved: {out_ext}\n")

# ─────────────────────────────────────────────
# GAP 1: separation_loss_results.json
# ─────────────────────────────────────────────
print("=== GAP 1: Building separation_loss_results.json ===\n")

# lambda=0.02 three-seed AUROC (from checkpoints, hardcoded verified values)
l02_seeds = {"42": 0.9911, "123": 0.9895, "456": 0.9904}
l02_mean  = float(np.mean(list(l02_seeds.values())))
l02_std   = float(np.std(list(l02_seeds.values())))

# Within-CIFAR AUROC per lambda (seed42 ablation runs)
# lambda=0.01 seed42 within-cifar: compute from raw scores
id42   = load_pt(RAW / "seed42_cifar10_id_scores.pt")
ood42  = load_pt(RAW / "seed42_cifar10_ood_scores.pt")
l01_auroc = auroc_from_scores(id42, ood42)
l01_fpr95 = fpr_at_tpr95(id42, ood42)
print(f"  lambda=0.01 seed42 within-CIFAR: AUROC={l01_auroc:.4f} FPR95={l01_fpr95:.3f}")

sep_results = {
    "weights": [0.0, 0.001, 0.01, 0.02, 0.05, 0.1],
    "within_cifar": {
        "0.0":   {"auroc": 0.8025, "fpr95": None, "std": None},
        "0.001": {"auroc": 0.9732, "fpr95": None, "std": None},
        "0.01":  {"auroc": round(l01_auroc, 4), "fpr95": round(l01_fpr95, 3), "std": None, "note": "seed42"},
        "0.02":  {"auroc": round(l02_mean, 4),  "fpr95": None, "std": round(l02_std, 4),
                  "seeds": l02_seeds, "note": "mean of seeds 42/123/456"},
        "0.05":  {"auroc": 0.9851, "fpr95": None, "std": None},
        "0.1":   {"auroc": 0.9667, "fpr95": None, "std": None},
    },
    "svhn": {
        # lambda=0.01 seed42: compute from raw scores
        "0.01": {"auroc": None, "fpr95": None},
        # lambda=0.02: from our lambda02_ood_results.json
        "0.02": {"auroc": 0.9658, "fpr95": 0.133, "note": "seed42 best ckpt"},
        # Others need GPU eval (GAP 3 script will fill these)
        "0.0": {"auroc": None, "fpr95": None},
        "0.001": {"auroc": None, "fpr95": None},
        "0.05": {"auroc": None, "fpr95": None},
        "0.1": {"auroc": None, "fpr95": None},
    }
}

# Fill lambda=0.01 SVHN from raw scores
svhn42 = load_pt(RAW / "seed42_svhn_scores.pt")
l01_svhn_auroc = auroc_from_scores(id42, svhn42)
l01_svhn_fpr95 = fpr_at_tpr95(id42, svhn42)
sep_results["svhn"]["0.01"] = {"auroc": round(l01_svhn_auroc, 4), "fpr95": round(l01_svhn_fpr95, 3)}
print(f"  lambda=0.01 seed42 SVHN:         AUROC={l01_svhn_auroc:.4f} FPR95={l01_svhn_fpr95:.3f}")
print(f"  lambda=0.02 3-seed mean:          AUROC={l02_mean:.4f} ± {l02_std:.4f}")

out_sep = OUT / "separation_loss_results.json"
with open(out_sep, "w") as f:
    json.dump(sep_results, f, indent=2)
print(f"\nSaved: {out_sep}")
print("\n=== DONE (no-GPU part) ===")
print("Next: run scripts/eval_svhn_sep_ablation.sh on GPU for GAP 3 (lambda=0.0/0.001/0.05/0.1)")
