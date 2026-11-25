"""Milestone 4: a ViT next to the CNN, compared fairly (Session 9, slide 25).

    same data     the seed-42 lesion splits, dermoscopic image, 224x224 stretched, ImageNet normalisation
    same head     both backbones FROZEN; one linear head = standardised features + multinomial
                  logistic regression with balanced class weights; C picked on VALIDATION from the same grid
    same metric   validation macro-F1 chooses; the test split is scored once per backbone

Features come from ``scripts/extract_frozen_features.py`` (ResNet-18 ``fc = Identity`` -> 512 numbers,
ViT-B/16 ``heads = Identity`` -> the 768-number class token), cached in
``data/processed/final/frozen_features.npz``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0)


def fit_head(X_tr, y_tr, C: float) -> object:
    head = make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    return head.fit(X_tr, y_tr)


def compare(features: dict, labels: pd.Series, split: pd.Series, backbones=("resnet18", "vit_b_16"),
            classes: list[str] | None = None) -> pd.DataFrame:
    """One row per backbone: chosen C, validation and test macro-F1 / accuracy."""
    rows = []
    tr, va, te = (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()
    y = labels.to_numpy()
    for name in backbones:
        X = features[name]
        best = max(C_GRID, key=lambda C: f1_score(y[va], fit_head(X[tr], y[tr], C).predict(X[va]),
                                                   average="macro", labels=classes, zero_division=0))
        head = fit_head(X[tr], y[tr], best)
        row = {"backbone": name, "features": X.shape[1], "C": best}
        for part, m in (("val", va), ("test", te)):
            pred = head.predict(X[m])
            row[f"{part}_macro_f1"] = f1_score(y[m], pred, average="macro", labels=classes, zero_division=0)
            row[f"{part}_accuracy"] = accuracy_score(y[m], pred)
        rows.append(row)
    return pd.DataFrame(rows)
