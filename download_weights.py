"""
download_weights.py
===================
Download pretrained checkpoints from HuggingFace (ahmed-3m/DiffusionOOD).

Usage:
    python download_weights.py               # download all checkpoints (default)
    python download_weights.py --main-only   # download only the 3 main seed checkpoints
    python download_weights.py --scores      # also download raw score tensors

All files are saved to models/ inside this repo.
CIFAR-10 data is NOT downloaded here — it is auto-downloaded by evaluate.py.
"""

import argparse
import sys
from pathlib import Path


HF_REPO = "ahmed-3m/DiffusionOOD"

# Main checkpoints (3-seed study, thesis headline results)
MAIN_CHECKPOINTS = {
    "seed42_best.ckpt":  "models/main_training/seed42_best_auroc0.9873.ckpt",
    "seed123_best.ckpt": "models/main_training/seed123_best_auroc0.9886.ckpt",
    "seed456_best.ckpt": "models/main_training/seed456_best_auroc0.9887.ckpt",
}

# Separation-loss ablation checkpoints
SEP_LOSS_CHECKPOINTS = {
    "sep_lambda_0p0.ckpt":  "models/separation_loss_ablation/sep_loss_lambda_0p0_epoch79_auroc0.8025.ckpt",
    "sep_lambda_0p02.ckpt": "models/separation_loss_ablation/sep_loss_lambda_0p02_epoch29_auroc0.9911.ckpt",
    "sep_lambda_0p1.ckpt":  "models/separation_loss_ablation/sep_loss_lambda_0p1_epoch149_auroc0.9667.ckpt",
}

# Raw score tensors (for fast AUROC recomputation without re-running inference)
RAW_SCORES = {
    "seed42_cifar10_id_scores.pt":  "models/raw_scores/seed42_cifar10_id_scores.pt",
    "seed42_cifar10_ood_scores.pt": "models/raw_scores/seed42_cifar10_ood_scores.pt",
}


def download_file(repo_id: str, hf_path: str, local_path: Path) -> bool:
    """Download one file from HuggingFace Hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface-hub>=0.20")
        sys.exit(1)

    if local_path.exists():
        print(f"  [skip] {local_path.name} already exists")
        return True

    print(f"  Downloading {hf_path} → {local_path} ...", end="", flush=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            local_dir=local_path.parent,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place the file with a different name; rename if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path:
            downloaded_path.rename(local_path)
        print(" done")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download DiffusionOOD pretrained weights")
    parser.add_argument("--main-only", action="store_true",
                        help="Download only the 3 main seed checkpoints (not ablations)")
    parser.add_argument("--scores", action="store_true",
                        help="Also download raw score tensors (.pt files)")
    parser.add_argument("--models-dir", default="models",
                        help="Directory to save checkpoints (default: models/)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"HuggingFace repo : {HF_REPO}")
    print(f"Save directory   : {models_dir.resolve()}")
    print()

    files_to_download = dict(MAIN_CHECKPOINTS)
    if not args.main_only:
        files_to_download.update(SEP_LOSS_CHECKPOINTS)
    if args.scores:
        files_to_download.update(RAW_SCORES)

    print(f"Downloading {len(files_to_download)} file(s):")
    failed = []
    for local_name, hf_path in files_to_download.items():
        local_path = models_dir / local_name
        ok = download_file(HF_REPO, hf_path, local_path)
        if not ok:
            failed.append(local_name)

    print()
    if failed:
        print(f"WARNING: {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  - {f}")
        print()
        print("You can retry with: python download_weights.py")
        sys.exit(1)
    else:
        print("All downloads complete.")
        print()
        print("Quick-start evaluation:")
        print("  python scripts/evaluate.py \\")
        print("      --checkpoint_path models/seed42_best.ckpt \\")
        print("      --num_trials 10 \\")
        print("      --data_dir ./data")
        print()
        print("Expected: AUROC ≈ 0.989, FPR@95 ≈ 0.047")


if __name__ == "__main__":
    main()
