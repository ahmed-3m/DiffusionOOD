import numpy as np
import pytest
from src.metrics import (
    compute_auroc,
    compute_fpr_at_tpr,
    compute_aupr,
    compute_all_metrics,
    format_metrics_table,
)


def test_perfect_auroc():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auroc(labels, scores) == pytest.approx(1.0)


def test_random_auroc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    auroc = compute_auroc(labels, scores)
    assert 0.4 < auroc < 0.6


def test_single_class_auroc():
    labels = np.ones(10)
    scores = np.random.rand(10)
    assert compute_auroc(labels, scores) == 0.5


def test_fpr_at_tpr():
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0])
    fpr, _ = compute_fpr_at_tpr(labels, scores, target_tpr=0.8)
    assert 0.0 <= fpr <= 1.0


def test_compute_all_metrics_keys():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.3, 0.7, 0.8])
    preds = np.array([0, 0, 1, 1])
    metrics = compute_all_metrics(labels, scores, preds)
    assert "auroc" in metrics
    assert "fpr95" in metrics
    assert "aupr" in metrics
    assert "accuracy" in metrics


def test_format_metrics_table():
    metrics = {"auroc": 0.99, "fpr95": 0.05, "aupr": 0.98}
    table = format_metrics_table(metrics)
    assert "AUROC" in table
    assert "0.9900" in table
