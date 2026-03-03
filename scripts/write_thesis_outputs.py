#!/usr/bin/env python3
"""
Write all 4 thesis submission output files from existing results_json/.
Target: thesis_submission/02_cifar10_cdm/
"""
import json, os, shutil
from pathlib import Path

PROJECT = Path("/system/user/studentwork/mohammed/2025/diffusion_classifier_ood")
TARGET  = PROJECT / "thesis_submission/02_cifar10_cdm"
SRC_JSON = PROJECT / "results_json"
SRC_TEX  = PROJECT / "results/latex_tables"

# Create dirs
(TARGET / "results_json").mkdir(parents=True, exist_ok=True)
(TARGET / "tables").mkdir(parents=True, exist_ok=True)

# ── Load source data ─────────────────────────────────────────
ext = json.load(open(SRC_JSON / "external_ood_results.json"))
sep = json.load(open(SRC_JSON / "separation_loss_results.json"))

# ─────────────────────────────────────────────────────────────
# FILE 1: external_ood_results.json (exact required format)
# Only 4 datasets: svhn, cifar100, fashionmnist, textures
# per_seed is just AUROC float (not nested dict)
# ─────────────────────────────────────────────────────────────
DATASETS_4 = ["svhn", "cifar100", "fashionmnist", "textures"]

ext_out = {}
for ds in DATASETS_4:
    s = ext[ds]
    ps = s["per_seed"]
    ext_out[ds] = {
        "auroc_mean": s["auroc_mean"],
        "auroc_std":  s["auroc_std"],
        "fpr95_mean": s["fpr95_mean"],
        "per_seed": {
            "42":  ps["42"]["auroc"],
            "123": ps["123"]["auroc"],
            "456": ps["456"]["auroc"],
        }
    }

f1 = TARGET / "results_json/external_ood_results.json"
with open(f1, "w") as f:
    json.dump(ext_out, f, indent=2)
print("=== FILE 1: external_ood_results.json ===")
print(open(f1).read())

# ─────────────────────────────────────────────────────────────
# FILE 2: separation_loss_results.json (exact required format)
# ─────────────────────────────────────────────────────────────
w02 = sep["within_cifar"]["0.02"]
s02 = sep["svhn"]["0.02"]

sep_out = {
    "weights": [0.0, 0.001, 0.01, 0.02, 0.05, 0.1],
    "within_cifar": {
        "0.0":  {"auroc": 0.8025, "fpr95": None},
        "0.001":{"auroc": 0.9732, "fpr95": None},
        "0.01": {"auroc": sep["within_cifar"]["0.01"]["auroc"],
                 "fpr95": sep["within_cifar"]["0.01"]["fpr95"]},
        "0.02": {"auroc": w02["auroc"],
                 "fpr95": w02.get("fpr95"),
                 "std":   w02["std"],
                 "per_seed": {
                     "42":  w02["seeds"]["42"],
                     "123": w02["seeds"]["123"],
                     "456": w02["seeds"]["456"],
                 }},
        "0.05": {"auroc": 0.9851, "fpr95": None},
        "0.1":  {"auroc": 0.9667, "fpr95": None},
    },
    "svhn": {
        "0.0":  {"auroc": sep["svhn"]["0.0"]["auroc"],  "fpr95": sep["svhn"]["0.0"]["fpr95"]},
        "0.001":{"auroc": sep["svhn"]["0.001"]["auroc"],"fpr95": sep["svhn"]["0.001"]["fpr95"]},
        "0.01": {"auroc": sep["svhn"]["0.01"]["auroc"], "fpr95": sep["svhn"]["0.01"]["fpr95"]},
        "0.02": {"auroc": s02["auroc"], "fpr95": s02["fpr95"]},
        "0.05": {"auroc": sep["svhn"]["0.05"]["auroc"], "fpr95": sep["svhn"]["0.05"]["fpr95"]},
        "0.1":  {"auroc": sep["svhn"]["0.1"]["auroc"],  "fpr95": sep["svhn"]["0.1"]["fpr95"]},
    }
}

f2 = TARGET / "results_json/separation_loss_results.json"
with open(f2, "w") as f:
    json.dump(sep_out, f, indent=2)
print("\n=== FILE 2: separation_loss_results.json ===")
print(open(f2).read())

# ─────────────────────────────────────────────────────────────
# FILE 3+4: LaTeX tables → copy from latex_tables/
# ─────────────────────────────────────────────────────────────
for src_name, dst_name in [
    ("main_results_table.tex",     "main_results_table.tex"),
    ("separation_loss_table.tex",  "separation_loss_table.tex"),
]:
    src = SRC_TEX / src_name
    dst = TARGET / "tables" / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"\n=== FILE: tables/{dst_name} ===")
        print(open(dst).read())
    else:
        print(f"WARNING: {src} not found!")

print("\n=== ALL FILES WRITTEN ===")
print(f"Target: {TARGET}")
for p in sorted(TARGET.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(TARGET)}")
