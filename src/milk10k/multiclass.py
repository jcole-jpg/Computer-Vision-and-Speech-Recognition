"""The 11-class stretch target: encoding, metrics, and the collapse back to benign / malignant.

Why a separate module: ``features.CLASSES`` and ``mlp.encode_labels`` are written for the
three-class ``diagnosis_1`` target. The stretch target predicts ``dx`` (BCC, NV, BKL, ...),
which changes three things the three-class code never had to decide:

* **Which classes count in the headline.** MAL_OTH has 9 lesions (6 train, 1 validation,
  2 test). Its F1 is reported but never trusted, so the headline macro-F1 averages the other
  ten classes (``EVALUABLE``). ``labels.LOW_SUPPORT_CLASSES`` records that decision.
* **How uncertainty is stated.** With 7 test lesions in four classes, a point estimate is
  an anecdote. ``bootstrap_ci`` resamples lesions and gives an interval for any metric.
* **How an 11-way answer becomes a clinical one.** Every ``dx`` class is wholly benign or
  wholly malignant (``data.MALIGNANT``), so P(malignant) is the sum of the malignant
  columns. :func:`binary_report` scores that decision with the homework's cost of a miss.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data import CLASS_CODES, MALIGNANT
from .labels import DX_MAP, LOW_SUPPORT_CLASSES

#: The 11 classes in the column order of training_gt.csv (index = integer label).
CLASSES: list[str] = list(CLASS_CODES)
#: The ten classes with enough lesions to score; the headline macro-F1 averages these.
EVALUABLE: list[str] = [c for c in CLASSES if c not in LOW_SUPPORT_CLASSES]
#: Column indices of the malignant classes in a probability matrix ordered as ``CLASSES``.
MALIGNANT_IDX: list[int] = [i for i, c in enumerate(CLASSES) if c in MALIGNANT]

#: Binary cost, rows = truth (Benign, Malignant), columns = prediction. The two outer
#: numbers are the Benign/Malignant corner of ``metrics.DEFAULT_COST``: a missed malignant
#: lesion costs 50, an unnecessary work-up costs 1.
BINARY_COST = pd.DataFrame([[0, 1], [50, 0]], index=["Benign", "Malignant"],
                           columns=["Benign", "Malignant"])


# --------------------------------------------------------------------------
# Labels and weights
# --------------------------------------------------------------------------

def encode(labels) -> np.ndarray:
    """``dx`` strings -> integers 0..10 in ``CLASSES`` order; raises on an unknown code."""
    labels = pd.Series(list(labels))
    unknown = set(labels.unique()) - set(DX_MAP)
    if unknown:
        raise KeyError(f"not 11-class labels: {sorted(unknown)}")
    return labels.map(DX_MAP).to_numpy()


def is_malignant(labels) -> np.ndarray:
    """True where the ``dx`` code is a malignant class (Indeterminate AKIEC included)."""
    return np.isin(np.asarray(list(labels), dtype=object), list(MALIGNANT))


def power_weights(y, power: float = 1.0) -> dict[str, float]:
    """Class weights ``(N / (K * n_c)) ** power``, from the TRAIN labels only.

    ``power=0`` is no weighting, ``1`` is the sklearn 'balanced' rule (a class a tenth as
    common weighs ten times as much), ``0.5`` the square-root compromise. With a 300:1
    imbalance the full rule gives MAL_OTH a weight above 100, which is why it is a knob.
    """
    y = pd.Series(list(y))
    counts = y.value_counts()
    n, k = len(y), len(CLASSES)
    return {c: float((n / (k * counts[c])) ** power) if counts.get(c, 0) else 0.0
            for c in CLASSES}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def macro_f1(y_true, y_pred, labels: list[str] = EVALUABLE) -> float:
    """Macro-F1 over ``labels`` (default: the ten evaluable classes). A class with no
    true and no predicted lesion scores 0 rather than being skipped."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_report(y_true, y_pred, labels: list[str] = CLASSES) -> pd.DataFrame:
    """Precision / recall / F1 / support per class, indexed by class code."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)


def top_k_accuracy(y_true, proba: np.ndarray, k: int = 3) -> float:
    """Share of lesions whose true class is among the ``k`` highest-probability classes.
    ``proba`` columns follow ``CLASSES``."""
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([DX_MAP[t] in row for t, row in zip(y_true, top)]))


def summary_scores(y_true, y_pred, proba: np.ndarray | None = None) -> dict[str, float]:
    """The numbers every model is judged on, in one dict."""
    out = {
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_all11": macro_f1(y_true, y_pred, CLASSES),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if proba is not None:
        out["top3_accuracy"] = top_k_accuracy(y_true, proba, 3)
    return out


def bootstrap_ci(y_true, y_pred, metric: Callable = macro_f1, n_boot: int = 2000,
                 seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for ``metric(y_true, y_pred)``, resampling lesions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx])
             for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def per_class_f1_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Per-class F1 with a bootstrap interval; the width is the honest message for the
    classes with fewer than ten test lesions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.array([per_class_report(y_true[i], y_pred[i])["f1"].to_numpy()
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    base = per_class_report(y_true, y_pred)
    base["f1_lo"], base["f1_hi"] = np.percentile(boots, [2.5, 97.5], axis=0)
    return base


def top_confusions(y_true, y_pred, top: int = 8) -> pd.DataFrame:
    """The most frequent wrong (truth -> prediction) pairs, with the share of the true
    class they account for."""
    df = pd.DataFrame({"truth": list(y_true), "predicted": list(y_pred)})
    totals = df["truth"].value_counts()
    wrong = (df[df["truth"] != df["predicted"]]
             .groupby(["truth", "predicted"]).size().rename("n").reset_index()
             .sort_values("n", ascending=False).head(top))
    wrong["share_of_true_class"] = wrong.apply(lambda r: r["n"] / totals[r["truth"]], axis=1)
    return wrong.reset_index(drop=True)


# --------------------------------------------------------------------------
# From 11 classes back to the clinical question
# --------------------------------------------------------------------------

def p_malignant(proba: np.ndarray) -> np.ndarray:
    """P(malignant) = sum of the malignant columns of an (n, 11) probability matrix."""
    return np.asarray(proba)[:, MALIGNANT_IDX].sum(axis=1)


def binary_report(y_true, malignant_score, threshold: float = 0.5) -> dict[str, float]:
    """Sensitivity, specificity, AUC and cost per lesion of the benign / malignant decision.

    ``y_true`` are ``dx`` codes; ``malignant_score`` is P(malignant) per lesion.
    The cost uses ``BINARY_COST`` (a miss costs 50, a false alarm 1).
    """
    truth = is_malignant(y_true)
    score = np.asarray(malignant_score)
    pred = score >= threshold
    sens = float((pred & truth).sum() / truth.sum())
    spec = float((~pred & ~truth).sum() / (~truth).sum())
    cost = ((~pred & truth).sum() * BINARY_COST.loc["Malignant", "Benign"]
            + (pred & ~truth).sum() * BINARY_COST.loc["Benign", "Malignant"]) / len(truth)
    return {"sensitivity": sens, "specificity": spec,
            "auc": float(roc_auc_score(truth, score)), "cost_per_lesion": float(cost),
            "threshold": float(threshold)}


def threshold_for_sensitivity(y_true, malignant_score, target: float = 0.95) -> float:
    """Largest threshold at which sensitivity is still at least ``target`` - the operating
    point a screening tool would use instead of 0.5. Choose it on validation, not test."""
    truth = is_malignant(y_true)
    pos = np.sort(np.asarray(malignant_score)[truth])
    k = int(np.floor((1 - target) * len(pos)))      # lesions we may miss
    return float(pos[k])
