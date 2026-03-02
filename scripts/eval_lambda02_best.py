#!/usr/bin/env python3
"""
Evaluate the best lambda=0.02 checkpoint (AUROC=0.9911, seed42, epoch29)
on all external OOD datasets.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_lambda02_best.py \
        --data_dir ./data \
        --output results/lambda02_ood_results.json
"""
import os, sys, json, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from src.lightning_module import DiffusionClassifierOOD
from src.scoring import diffusion_classifier_score
from src.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT = "results/sep_loss_ablation/2026-02-24_22-57-58_sep_0.02/best-epoch=29-val/auroc=0.9911.ckpt"

def get_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def get_gray_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="results/lambda02_ood_results.json")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {CHECKPOINT}")

    model = DiffusionClassifierOOD.load_from_checkpoint(CHECKPOINT)
    model.eval().to(device)

    tfm = get_transform()
    gray_tfm = get_gray_transform()

    # --- ID reference: CIFAR-10 class 0 (airplane) ---
    cifar_test = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=tfm, download=True)
    id_idx = [i for i, (_, y) in enumerate(cifar_test) if y == 0]
    ood_idx = [i for i, (_, y) in enumerate(cifar_test) if y != 0]
    id_loader = DataLoader(torch.utils.data.Subset(cifar_test, id_idx[:1000]), batch_size=args.batch_size, shuffle=False)
    ood_loader = DataLoader(torch.utils.data.Subset(cifar_test, ood_idx[:1000]), batch_size=args.batch_size, shuffle=False)

    datasets_to_eval = {}

    # Within-CIFAR
    datasets_to_eval["within_cifar"] = {
        "id_loader": id_loader,
        "ood_loader": ood_loader,
    }

    # External datasets
    external = {}
    try:
        svhn = torchvision.datasets.SVHN(root=args.data_dir, split='test', transform=tfm, download=True)
        external["svhn"] = DataLoader(torch.utils.data.Subset(svhn, list(range(min(2000, len(svhn))))), batch_size=args.batch_size)
        logger.info(f"SVHN loaded: {len(svhn)} samples")
    except Exception as e:
        logger.warning(f"SVHN failed: {e}")

    try:
        c100 = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=tfm, download=True)
        external["cifar100"] = DataLoader(torch.utils.data.Subset(c100, list(range(2000))), batch_size=args.batch_size)
        logger.info(f"CIFAR-100 loaded: {len(c100)} samples")
    except Exception as e:
        logger.warning(f"CIFAR-100 failed: {e}")

    try:
        stl = torchvision.datasets.STL10(root=args.data_dir, split='test', transform=tfm, download=True)
        external["stl10"] = DataLoader(torch.utils.data.Subset(stl, list(range(min(2000, len(stl))))), batch_size=args.batch_size)
    except Exception as e:
        logger.warning(f"STL-10 failed: {e}")

    try:
        fm = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False, transform=gray_tfm, download=True)
        external["fashionmnist"] = DataLoader(torch.utils.data.Subset(fm, list(range(2000))), batch_size=args.batch_size)
    except Exception as e:
        logger.warning(f"FashionMNIST failed: {e}")

    def score_loader(loader):
        scores = []
        scheduler = model.scheduler
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0].to(device)
                elif isinstance(batch, dict):
                    imgs = batch['images'].to(device)
                else:
                    imgs = batch.to(device)
                result = diffusion_classifier_score(
                    model, scheduler, imgs,
                    num_trials=args.num_trials,
                    scoring_method="difference",
                    timestep_mode="mid_focus"
                )
                # result may be (scores, extra) or just scores
                s = result[0] if isinstance(result, tuple) else result
                scores.extend(s.cpu().numpy().tolist())
        return np.array(scores)

    results = {"checkpoint": CHECKPOINT, "num_trials": args.num_trials, "datasets": {}}

    # Score ID samples once
    logger.info("Scoring ID (CIFAR-10 airplane)...")
    id_scores = score_loader(id_loader)
    logger.info(f"  ID scores: mean={id_scores.mean():.4f} std={id_scores.std():.4f}")

    # Within-CIFAR
    logger.info("Scoring OOD within-CIFAR...")
    ood_scores_cifar = score_loader(ood_loader)
    all_scores = np.concatenate([id_scores, ood_scores_cifar])
    all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores_cifar))])
    metrics = compute_all_metrics(all_labels, all_scores)
    results["datasets"]["within_cifar"] = metrics
    logger.info(f"  within_cifar: AUROC={metrics['auroc']:.4f} FPR95={metrics['fpr95']:.3f}")

    # External datasets
    for ds_name, loader in external.items():
        logger.info(f"Scoring {ds_name}...")
        ext_scores = score_loader(loader)
        ext_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ext_scores))])
        ext_all = np.concatenate([id_scores, ext_scores])
        metrics = compute_all_metrics(ext_labels, ext_all)
        results["datasets"][ds_name] = metrics
        logger.info(f"  {ds_name}: AUROC={metrics['auroc']:.4f} FPR95={metrics['fpr95']:.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to: {args.output}")

    # Print summary
    print("\n=== SUMMARY: lambda=0.02 best checkpoint (AUROC=0.9911) ===")
    for ds, m in results["datasets"].items():
        print(f"  {ds:20s}: AUROC={m['auroc']:.4f}  FPR95={m['fpr95']:.3f}")

if __name__ == "__main__":
    main()
