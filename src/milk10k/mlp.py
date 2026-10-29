"""The Session 6 MLP recipe, importable: build, train with mini-batches, score.

Same recipe as the class notebook - Adam, class-weighted cross-entropy, a seeded
initialisation - so the homework notebook and the tests use exactly the code the
class used instead of a drifting copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import CLASSES


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(X), dtype=torch.float32)


def encode_labels(series: pd.Series) -> torch.Tensor:
    return torch.tensor(series.map({c: i for i, c in enumerate(CLASSES)}).to_numpy())


def inverse_frequency_weights(y: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """N / (K * n_c): the rare class counts as much as the common ones in the loss."""
    return len(y) / (n_classes * torch.bincount(y, minlength=n_classes).float())


def make_mlp(n_in: int, hidden: tuple[int, ...] = (64,), dropout: float = 0.0,
             n_out: int = 3, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)                      # same initial weights for everyone
    layers, d_in = [], n_in
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d_in = h
    layers.append(nn.Linear(d_in, n_out))        # logits; softmax lives inside the loss
    return nn.Sequential(*layers)


def fit(model: nn.Module, Xtr, ytr, Xva, yva, class_weights=None, epochs: int = 50,
        lr: float = 1e-3, weight_decay: float = 0.0, batch_size: int = 64,
        seed: int = 0) -> pd.DataFrame:
    """Train in place; return one row per epoch: train_loss, val_loss, val_f1."""
    g = torch.Generator().manual_seed(seed)      # shuffling order is reproducible too
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            history.append({"epoch": epoch,
                            "train_loss": total / len(Xtr),
                            "val_loss": loss_fn(val_logits, yva).item(),
                            "val_f1": f1_score(yva, val_logits.argmax(1), average="macro")})
    return pd.DataFrame(history).set_index("epoch")


def scores(model: nn.Module, X, y) -> dict[str, float]:
    """Accuracy and macro-F1 of ``model`` on (X, y)."""
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}
