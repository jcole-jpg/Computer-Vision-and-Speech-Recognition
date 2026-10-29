"""Session 6 MLP recipe: shapes, parameter count, seeded reproducibility."""
import numpy as np
import pandas as pd
import pytest
import torch

from milk10k import mlp


def _toy(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, d)) + y[:, None]            # classes separable by a shift
    return mlp.to_tensor(X), torch.tensor(y)


def test_parameter_count_matches_session_6_slide():
    # 33-64-3: (33*64 + 64) + (64*3 + 3) = 2371
    assert sum(p.numel() for p in mlp.make_mlp(33).parameters()) == 2371
    assert sum(p.numel() for p in mlp.make_mlp(3072).parameters()) == 3072 * 64 + 64 + 64 * 3 + 3


def test_same_seed_same_weights_different_seed_differs():
    a, b, c = mlp.make_mlp(10, seed=3), mlp.make_mlp(10, seed=3), mlp.make_mlp(10, seed=4)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(next(a.parameters()), next(c.parameters()))


def test_fit_is_reproducible_and_learns():
    X, y = _toy()
    runs = []
    for _ in range(2):
        m = mlp.make_mlp(5, hidden=(16,), seed=0)
        runs.append(mlp.fit(m, X, y, X, y, epochs=30, lr=1e-2))
    pd.testing.assert_frame_equal(runs[0], runs[1])
    assert runs[0]["train_loss"].iloc[-1] < runs[0]["train_loss"].iloc[0]
    assert runs[0]["val_f1"].iloc[-1] > 0.6


def test_inverse_frequency_weights_balance_the_classes():
    y = torch.tensor([0] * 8 + [1] * 1 + [2] * 1)
    w = mlp.inverse_frequency_weights(y)
    counts = torch.bincount(y).float()
    assert torch.allclose(w * counts, torch.full((3,), len(y) / 3))


def test_scores_keys_and_range():
    X, y = _toy()
    s = mlp.scores(mlp.make_mlp(5, seed=0), X, y)
    assert set(s) == {"accuracy", "macro_f1"} and 0 <= s["macro_f1"] <= 1
