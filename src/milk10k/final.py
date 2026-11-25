"""Milestone 5: the final 11-class model, ``predict()`` and the leaderboard ``submission.csv``.

The final model = the average of two probability vectors (see ``FinalModel``):
  1. ResNet-18 fine-tuned in full, 11 outputs (``train``), and
  2. a frozen ViT-B/16 + logistic-regression head (``train_vit_head``, Milestone 4).

The CNN uses the same recipe as the Session 7 pick (``cnn.py``, strategy 3), only the head changes:

    input       the dermoscopic image of each lesion, stretched to 224x224 (PIL bilinear), RGB
    network     ResNet-18, ImageNet weights, every layer fine-tuned, head ``Linear(512, 11)``
    loss        cross-entropy with class weights ``(N / (K * n_c)) ** 0.75`` from TRAIN only
                (power 0.75 = the value chosen by CV in the 11-class stretch notebook)
    optimiser   Adam, head 1e-3, backbone 1e-4, batch 32, 10 epochs
    selection   the epoch with the best VALIDATION macro-F1; the test split is scored once

The one rule of Session 10 (slide 23): a new image is prepared exactly like a validation
image. ``eval_tensor`` is that single preparation, used by validation, by ``predict`` and by
the submission, so there is no train/serve skew.

The leaderboard (challenge.isic-archive.com/landing/milk10k) scores macro-F1 over the 11
classes after binarising every probability at >= 0.5, column by column. Raw softmax values
would leave lesions with no class and rare classes (DF, MEL) under 0.5. The submission
therefore writes DECISIONS (1.0 / 0.0): class c is 1.0 when its probability reaches a
per-class threshold tuned on VALIDATION (``tune_thresholds``). On the internal test split,
scored once, this rule gave 0.497 against 0.463 for argmax and 0.481 for raw >= 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

from . import cnn
from .config import ARTIFACT_DIR, IMAGE_DIR, PROCESSED_DIR, SPLIT_DIR
from .multiclass import CLASSES, power_weights

#: Column order the leaderboard expects (identical to ``CLASSES`` / training_gt.csv).
LEADERBOARD_COLUMNS: list[str] = ["lesion"] + CLASSES
RESULT_DIR: Path = ARTIFACT_DIR / "final"
MODEL_PATH: Path = PROCESSED_DIR / "final" / "resnet18_11class.pt"   # 45 MB, git-ignored
WEIGHT_POWER = 0.75


# --------------------------------------------------------------------------
# Finding images: the flat ISIC layout (images/ISIC_x.jpg) or the challenge zip layout
# (MILK10k_*_Input/IL_lesion/ISIC_x.jpg)
# --------------------------------------------------------------------------

def image_index(root: Path | str) -> dict[str, Path]:
    """``isic_id -> path`` for every .jpg under ``root``, whatever the folder layout."""
    index = {p.stem: p for p in Path(root).rglob("*.jpg")}
    if not index:
        raise FileNotFoundError(f"no .jpg images under {root}")
    return index


def eval_tensor(path: Path | str, size: int = cnn.IMAGE_SIZE) -> torch.Tensor:
    """The validation transform as a uint8 ``(3, size, size)`` tensor: RGB, PIL bilinear
    resize (stretched, not cropped). Normalisation happens in ``cnn.normalise``."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR))
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def stack(paths) -> torch.Tensor:
    return torch.stack([eval_tensor(p) for p in paths])


def dermoscopic(split_csv: Path | str) -> pd.DataFrame:
    """One row per lesion: its dermoscopic image and 11-class label (column ``label``)."""
    df = pd.read_csv(split_csv)
    df = df[df["image_type"].str.startswith("dermoscopic")].sort_values("lesion_id")
    assert df["lesion_id"].is_unique, "more than one dermoscopic image for a lesion"
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def load_splits(image_root: Path | str = IMAGE_DIR) -> dict[str, tuple[pd.DataFrame, torch.Tensor, torch.Tensor]]:
    """``{split: (table, X_uint8, y)}`` for train / val / test (dermoscopic only)."""
    index, out = image_index(image_root), {}
    for split in ("train", "val", "test"):
        df = dermoscopic(SPLIT_DIR / f"{split}.csv")
        missing = sorted(set(df["isic_id"]) - set(index))
        if missing:                                   # fail loudly (Milestone 1, B1)
            raise FileNotFoundError(f"{len(missing)} {split} images missing, e.g. {missing[:3]}")
        X = stack([index[i] for i in df["isic_id"]])
        y = torch.tensor(df["label"].map({c: k for k, c in enumerate(CLASSES)}).to_numpy())
        out[split] = (df, X, y)
    return out


def class_weight_tensor(train_labels) -> torch.Tensor:
    w = power_weights(train_labels, power=WEIGHT_POWER)
    return torch.tensor([w[c] for c in CLASSES], dtype=torch.float32)


def train(image_root: Path | str = IMAGE_DIR, epochs: int = 10, seed: int = 0) -> dict:
    """Train the final model, save its weights and its numbers, return the numbers."""
    data = load_splits(image_root)
    (dtr, Xtr, ytr), (dva, Xva, yva), (dte, Xte, yte) = data["train"], data["val"], data["test"]
    weights = class_weight_tensor(dtr["label"])
    model = cnn.resnet18(trainable=None, n_out=len(CLASSES), seed=seed)
    fit = cnn.fit_cnn(model, Xtr, ytr, Xva, yva, class_weights=weights, epochs=epochs,
                      lr_head=1e-3, lr_backbone=1e-4, batch_size=32, seed=seed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODEL_PATH)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"epochs": epochs, "seed": seed, "best_epoch": fit["best_epoch"] + 1,
              "seconds_per_epoch": fit["seconds_per_epoch"],
              "class_weights": dict(zip(CLASSES, weights.tolist())),
              "n_lesions": {"train": len(dtr), "val": len(dva), "test": len(dte)}}
    for name, (df, X, y) in data.items():
        if name == "train":
            continue
        proba = softmax(cnn.predict(model, X))
        pred = proba.argmax(1)
        result[name] = split_scores(y.numpy(), pred)
        out = df[["lesion_id", "isic_id", "label"]].copy()
        out["pred"] = [CLASSES[k] for k in pred]
        out[CLASSES] = proba
        out.to_csv(RESULT_DIR / f"{name}_predictions.csv", index=False)
    fit["history"].to_csv(RESULT_DIR / "history.csv")
    (RESULT_DIR / "scores.json").write_text(json.dumps(result, indent=1))
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def split_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``macro_f1_11``: F1 averaged over ALL 11 classes, a never-predicted class counting 0,
    as on the leaderboard (one-hot argmax = binarised at 0.5). ``macro_f1_present``: only the
    classes that occur in ``y_true`` (MAL_OTH can be absent from a split). Plus accuracy."""
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = np.isin(labels, np.unique(y_true))
    return {"macro_f1_11": float(per_class.mean()),
            "macro_f1_present": float(per_class[present].mean()),
            "accuracy": float((y_true == y_pred).mean()),
            "per_class_f1": dict(zip(CLASSES, np.round(per_class, 4).tolist()))}


# --------------------------------------------------------------------------
# Inference: predict() on a new image, and the leaderboard file
# --------------------------------------------------------------------------

def load_model(path: Path | str = MODEL_PATH, dev: torch.device | None = None) -> torch.nn.Module:
    model = cnn.resnet18(trainable=None, n_out=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(dev or torch.device("cpu")).eval()


# --------------------------------------------------------------------------
# The final model: fine-tuned ResNet-18 + frozen ViT-B/16 head, probabilities averaged
# --------------------------------------------------------------------------
#
# Milestone 4 compared the two backbones fairly (compare.py). The CNN fine-tuned in full and
# the frozen ViT with a logistic-regression head make different mistakes; the plain average
# of their probabilities was the best of the weights {0, .25, .5, .75, 1} on VALIDATION
# (macro-F1 0.463 vs 0.422 for the CNN alone and 0.375 for the ViT alone). The test split,
# scored once afterwards, agreed (0.463 vs 0.416 / 0.401).

VIT_HEAD_PATH: Path = PROCESSED_DIR / "final" / "vit_head.joblib"
FEATURES_PATH: Path = PROCESSED_DIR / "final" / "frozen_features.npz"
VIT_C = 0.1            # chosen on validation by compare.compare (11 classes)
VIT_WEIGHT = 0.5       # chosen on validation from {0, 0.25, 0.5, 0.75, 1}


def train_vit_head(features: Path | str = FEATURES_PATH, out: Path | str = VIT_HEAD_PATH):
    """Fit the logistic-regression head on the TRAIN lesions' frozen ViT features and save it."""
    import joblib
    from .compare import fit_head
    z = np.load(features, allow_pickle=True)
    tr = z["split"] == "train"
    head = fit_head(z["vit_b_16"][tr], z["label"][tr], VIT_C)
    assert list(head.classes_) == CLASSES
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, out)
    return head


class FinalModel:
    """``proba(X_uint8)`` = VIT_WEIGHT * ViT head + (1 - VIT_WEIGHT) * fine-tuned ResNet-18.
    Both see the SAME uint8 224x224 tensor, normalised with the ImageNet statistics."""

    def __init__(self, resnet: torch.nn.Module | None = None, vit_head=None):
        import joblib
        import torchvision
        self.resnet = resnet or load_model()
        self.vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = torch.nn.Identity()
        self.vit.eval()
        self.vit_head = vit_head if vit_head is not None else joblib.load(VIT_HEAD_PATH)

    @torch.no_grad()
    def proba(self, X_uint8: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        p_cnn = softmax(cnn.predict(self.resnet, X_uint8, batch_size=batch_size))
        feats = np.concatenate([self.vit(cnn.normalise(X_uint8[i:i + batch_size])).numpy()
                                for i in range(0, len(X_uint8), batch_size)])
        p_vit = self.vit_head.predict_proba(feats)
        return VIT_WEIGHT * p_vit + (1 - VIT_WEIGHT) * p_cnn


def predict(image_path: Path | str, model: "FinalModel | None" = None) -> dict:
    """Diagnosis of ONE new dermoscopic image, prepared exactly like a validation image
    (``eval_tensor``: RGB, 224x224, no augmentation; both networks in eval mode, no gradients).

    Returns ``{"label": best class, "probabilities": {class: p}, "p_malignant": sum}``.
    """
    model = model or FinalModel()
    proba = model.proba(eval_tensor(image_path).unsqueeze(0))[0]
    probs = dict(zip(CLASSES, proba.round(4).tolist()))
    p_mal = float(sum(probs[c] for c in ("BCC", "MEL", "SCCKA", "MAL_OTH")))
    return {"label": CLASSES[int(proba.argmax())], "probabilities": probs, "p_malignant": p_mal}


def evaluate(image_root: Path | str = IMAGE_DIR, model: "FinalModel | None" = None) -> dict:
    """Run the final model through the deployment path on val and test; save predictions."""
    model = model or FinalModel()
    index, result = image_index(image_root), {}
    for split in ("val", "test"):
        df = dermoscopic(SPLIT_DIR / f"{split}.csv")
        proba = model.proba(stack([index[i] for i in df["isic_id"]]))
        y = df["label"].map({c: k for k, c in enumerate(CLASSES)}).to_numpy()
        result[split] = split_scores(y, proba.argmax(1))
        out = df[["lesion_id", "isic_id", "label"]].copy()
        out["pred"] = [CLASSES[k] for k in proba.argmax(1)]
        out[CLASSES] = proba
        out.to_csv(RESULT_DIR / f"{split}_predictions.csv", index=False)
    (RESULT_DIR / "scores.json").write_text(json.dumps(result, indent=1))
    return result


THRESHOLD_PATH: Path = RESULT_DIR / "per_class_thresholds_val.csv"


def tune_thresholds(val_predictions: Path | str = RESULT_DIR / "val_predictions.csv",
                    out: Path | str = THRESHOLD_PATH) -> pd.Series:
    """One threshold per class, each maximising that class's F1 on VALIDATION (grid 0.05..0.95).
    The leaderboard binarises every column on its own, so a class whose probability rarely
    reaches 0.5 (DF, MEL, BEN_OTH) can still be predicted once its threshold is lowered."""
    d = pd.read_csv(val_predictions)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    # ties (several cuts with the same F1) go to the cut closest to the default 0.5
    thr = pd.Series({c: max(grid, key=lambda t: (f1_score(d["label"] == c, d[c] >= t, zero_division=0),
                                                 -abs(t - 0.5)))
                     for c in CLASSES}, name="threshold")
    thr.to_csv(out, header=True)
    return thr


def make_submission(test_image_root: Path | str, test_metadata_csv: Path | str,
                    out_csv: Path | str = RESULT_DIR / "submission.csv",
                    model: "FinalModel | None" = None, rule: str = "thresholds") -> pd.DataFrame:
    """479 rows x 12 columns: ``lesion`` + the 11 classes as 1.0 / 0.0 decisions.

    ``rule="thresholds"`` (submitted): class c is 1.0 when p_c >= its validation threshold, so a
    lesion can get 0, 1 or several classes, exactly what the leaderboard scores.
    ``rule="argmax"``: the single most likely class is 1.0 (written to submission_argmax.csv).
    The raw averaged probabilities go to submission_probabilities.csv.
    """
    model = model or FinalModel()
    meta = pd.read_csv(test_metadata_csv)
    derm = meta[meta["image_type"].str.startswith("dermoscopic")].sort_values("lesion_id")
    assert derm["lesion_id"].is_unique
    index = image_index(test_image_root)
    missing = sorted(set(derm["isic_id"]) - set(index))
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images missing, e.g. {missing[:3]}")
    proba = model.proba(stack([index[i] for i in derm["isic_id"]]))
    probs = pd.DataFrame(proba, columns=CLASSES)
    probs.insert(0, "lesion", derm["lesion_id"].to_numpy())
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(out_csv.with_name("submission_probabilities.csv"), index=False)
    argmax = probs.copy()
    argmax[CLASSES] = np.eye(len(CLASSES))[proba.argmax(1)]
    check_submission(argmax, n_rows=len(derm), one_class=True)
    argmax.to_csv(out_csv.with_name("submission_argmax.csv"), index=False)
    if rule == "argmax":
        return argmax
    thr = pd.read_csv(THRESHOLD_PATH, index_col=0)["threshold"].reindex(CLASSES).to_numpy()
    hard = probs.copy()
    hard[CLASSES] = (proba >= thr).astype(float)
    check_submission(hard, n_rows=len(derm))
    hard.to_csv(out_csv, index=False)
    return hard


def check_submission(df: pd.DataFrame, n_rows: int = 479, one_class: bool = False) -> None:
    """The format check the leaderboard runs, run first so an upload never bounces."""
    assert list(df.columns) == LEADERBOARD_COLUMNS, f"columns: {list(df.columns)}"
    assert len(df) == n_rows and df["lesion"].is_unique, "one row per lesion"
    v = df[CLASSES].to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), "values in [0, 1]"
    if one_class:
        assert ((v >= 0.5).sum(1) == 1).all(), "exactly one class >= 0.5 per lesion"
