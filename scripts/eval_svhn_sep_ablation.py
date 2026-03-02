#!/usr/bin/env python3
"""
GAP 3: Evaluate sep-ablation checkpoints (lambda=0.0/0.001/0.05/0.1) on SVHN.
Then merge results into results_json/separation_loss_results.json.
Run AFTER compute_gap1_gap2.py.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/eval_svhn_sep_ablation.py
"""
import os, sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

from src.lightning_module import DiffusionClassifierOOD
from src.scoring import diffusion_classifier_score

PROJECT = Path("/system/user/studentwork/mohammed/2025/diffusion_classifier_ood")
RAW     = PROJECT / "results/raw_scores"
OUT     = PROJECT / "results_json"
DATA    = PROJECT / "data"

# Sep-ablation checkpoints (seed42 only)
CHECKPOINTS = {
    "0.0":   PROJECT / "results/sep_loss_ablation/2026-02-21_05-04-31_sep_0.0/best-epoch=79-val/auroc=0.8025.ckpt",
    "0.001": PROJECT / "results/sep_loss_ablation/2026-02-21_22-03-43_sep_0.001/best-epoch=19-val/auroc=0.9732.ckpt",
    "0.05":  PROJECT / "results/sep_loss_ablation/2026-02-23_02-04-16_sep_0.05/best-epoch=19-val/auroc=0.9851.ckpt",
    "0.1":   PROJECT / "results/sep_loss_ablation/2026-02-22_14-36-51_sep_0.1/best-epoch=149-val/auroc=0.9667.ckpt",
}
NUM_TRIALS  = 50
BATCH_SIZE  = 64
N_SAMPLES   = 2000   # cap for speed

def get_transform():
    return transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def fpr_at_tpr95(id_scores, ood_scores):
    thresh = np.percentile(ood_scores, 5.0)   # 95th TPR = 5th percentile of OOD (highest = more OOD)
    return float(np.mean(id_scores >= thresh))

def auroc(id_s, ood_s):
    labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    scores = np.concatenate([id_s, ood_s])
    return float(roc_auc_score(labels, scores))

def score_dataset(model, loader, device, n_max=N_SAMPLES):
    scores = []
    scheduler = model.scheduler
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
            result = diffusion_classifier_score(model, scheduler, imgs,
                                                num_trials=NUM_TRIALS,
                                                scoring_method="difference",
                                                timestep_mode="mid_focus")
            s = result[0] if isinstance(result, tuple) else result
            scores.extend(s.cpu().numpy().tolist())
            if len(scores) >= n_max:
                break
    return np.array(scores[:n_max])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tfm = get_transform()

    # Load ID set (CIFAR-10 class 0, from raw_scores)
    id_scores_ref = torch.load(RAW / "seed42_cifar10_id_scores.pt", map_location="cpu").numpy().flatten()
    print(f"ID scores loaded from raw_scores: {len(id_scores_ref)} samples")

    # Load SVHN
    svhn = torchvision.datasets.SVHN(root=DATA, split='test', transform=tfm, download=True)
    svhn_loader = DataLoader(svhn, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"SVHN: {len(svhn)} samples\n")

    svhn_results = {}

    for lam, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            # try fuzzy search
            pattern = f"sep_{lam.replace('.','_') if lam != '0.0' else '0.0'}"
            found = list((PROJECT / "results/sep_loss_ablation").glob(f"*{lam.replace('.','_')}*/best-*.ckpt"))
            if not found:
                print(f"  lambda={lam}: CHECKPOINT NOT FOUND — skipping")
                continue
            ckpt_path = found[0]

        print(f"=== lambda={lam} ===")
        print(f"  Checkpoint: {ckpt_path}")
        model = DiffusionClassifierOOD.load_from_checkpoint(str(ckpt_path))
        model.eval().to(device)

        svhn_scores = score_dataset(model, svhn_loader, device)
        a = auroc(id_scores_ref[:len(svhn_scores)], svhn_scores)
        f = fpr_at_tpr95(id_scores_ref[:len(svhn_scores)], svhn_scores)
        svhn_results[lam] = {"auroc": round(a, 4), "fpr95": round(f, 3)}
        print(f"  SVHN: AUROC={a:.4f}  FPR95={f:.3f}\n")

        del model
        torch.cuda.empty_cache()

    # Merge into separation_loss_results.json
    sep_path = OUT / "separation_loss_results.json"
    with open(sep_path) as f:
        sep = json.load(f)

    for lam, res in svhn_results.items():
        sep["svhn"][lam] = res
        print(f"Updated svhn[{lam}] = {res}")

    with open(sep_path, "w") as f:
        json.dump(sep, f, indent=2)
    print(f"\nUpdated: {sep_path}")
    print("\nFinal SVHN ablation:")
    for lam in ["0.0","0.001","0.01","0.02","0.05","0.1"]:
        v = sep["svhn"].get(lam, {})
        print(f"  lambda={lam}: {v}")

if __name__ == "__main__":
    main()
