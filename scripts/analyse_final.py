"""Milestone 4 + 5 analysis: confusion matrices, misclassified images, fairness, scoreboard.

    python scripts/analyse_final.py --images data/raw/milk10k/images --metadata data/raw/milk10k/metadata.csv

Reads the Session 7 pick (data/processed/session07/resnet18_s3_full.pt, 3 classes) and the final
11-class model (data/processed/final/resnet18_11class.pt). Writes tables to artifacts/final/ and
figures to reports/figures/final_*.png. The test split is only scored, never used to choose.
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from milk10k import cnn, final
from milk10k.config import IMAGE_DIR, METADATA_CSV
from milk10k.features import CLASSES as CLASSES3
from milk10k.multiclass import CLASSES as CLASSES11

OUT, FIG = ROOT / "artifacts" / "final", ROOT / "reports" / "figures"
MALIGNANT = ["BCC", "MEL", "SCCKA", "MAL_OTH"]


def plot_cm(cm, labels, title, path):
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 3, 0.55 * len(labels) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fairness(df, truth, pred, labels):
    rows = []
    for col in ("skin_tone_class", "sex"):
        for g, part in df.groupby(col):
            rows.append({"group": col, "value": g, "n_lesions": len(part),
                         "macro_f1": f1_score(part[truth], part[pred], labels=labels, average="macro", zero_division=0),
                         "accuracy": float((part[truth] == part[pred]).mean()),
                         "note": "too small to conclude" if len(part) < 30 else ""})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=str(IMAGE_DIR))
    p.add_argument("--metadata", default=str(METADATA_CSV))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    index = final.image_index(a.images)
    meta = pd.read_csv(a.metadata)
    meta = meta[meta.image_type.str.startswith("dermoscopic")].set_index("lesion_id")
    results = {}

    # ---- M4: the 3-class pick (Session 7, ResNet-18 fully fine-tuned) ----------------------
    m3 = cnn.resnet18(trainable=None, n_out=3, pretrained=False)
    m3.load_state_dict(torch.load(ROOT / "data/processed/session07/resnet18_s3_full.pt", map_location="cpu"))
    test = final.dermoscopic(ROOT / "artifacts/splits/test.csv")
    proba = final.softmax(cnn.predict(m3, final.stack([index[i] for i in test.isic_id])))
    t3 = test[["lesion_id", "isic_id", "diagnosis_1", "label"]].copy()
    t3["pred"] = [CLASSES3[k] for k in proba.argmax(1)]
    t3[[f"p_{c}" for c in CLASSES3]] = proba
    t3.to_csv(OUT / "m4_test_predictions_3class.csv", index=False)
    cm3 = confusion_matrix(t3.diagnosis_1, t3.pred, labels=CLASSES3)
    pd.DataFrame(cm3, index=CLASSES3, columns=CLASSES3).to_csv(OUT / "m4_confusion_test_3class.csv")
    plot_cm(cm3, CLASSES3, "ResNet-18 fine-tuned, 3 classes, test (749 lesions)", FIG / "final_cm_3class.png")
    t3 = t3.join(meta[["skin_tone_class", "sex", "age_approx", "site"]], on="lesion_id")
    fairness(t3, "diagnosis_1", "pred", CLASSES3).to_csv(OUT / "m4_fairness_3class.csv", index=False)
    results["m4_test_macro_f1"] = f1_score(t3.diagnosis_1, t3.pred, average="macro")
    results["m4_test_recall"] = dict(zip(CLASSES3, recall_score(t3.diagnosis_1, t3.pred, labels=CLASSES3, average=None).round(3).tolist()))

    # misclassified: the 10 most confident mistakes
    t3["p_pred"] = proba.max(1)
    wrong = t3[t3.diagnosis_1 != t3.pred].sort_values("p_pred", ascending=False).head(10)
    wrong.to_csv(OUT / "m4_misclassified_top10.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6))
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        with Image.open(index[r.isic_id]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"true {r.label} ({r.diagnosis_1[:5]})\npred {r.pred[:5]} p={r.p_pred:.2f} · tone {r.skin_tone_class}", fontsize=8)
        ax.axis("off")
    fig.suptitle("The 10 most confident mistakes on test (3-class ResNet-18)")
    fig.tight_layout(); fig.savefig(FIG / "final_misclassified.png", dpi=110); plt.close(fig)

    # ---- M5: the 11-class model ---------------------------------------------------------
    for split in ("val", "test"):
        d = pd.read_csv(OUT / f"{split}_predictions.csv")
        cm = confusion_matrix(d.label, d.pred, labels=CLASSES11)
        pd.DataFrame(cm, index=CLASSES11, columns=CLASSES11).to_csv(OUT / f"confusion_{split}_11class.csv")
        plot_cm(cm, CLASSES11, f"Final model (ResNet-18 + ViT), 11 classes, {split} ({len(d)} lesions)", FIG / f"final_cm_11class_{split}.png")
        f1 = f1_score(d.label, d.pred, labels=CLASSES11, average=None, zero_division=0)
        support = d.label.value_counts().reindex(CLASSES11).fillna(0).astype(int)
        pc = pd.DataFrame({"class": CLASSES11, "support": support.to_numpy(), "f1": f1.round(3),
                           "recall": recall_score(d.label, d.pred, labels=CLASSES11, average=None, zero_division=0).round(3),
                           "n_predicted": d.pred.value_counts().reindex(CLASSES11).fillna(0).astype(int).to_numpy()})
        pc.to_csv(OUT / f"per_class_{split}_11class.csv", index=False)
        dm = d.join(meta[["skin_tone_class", "sex"]], on="lesion_id")
        fairness(dm, "label", "pred", CLASSES11).to_csv(OUT / f"fairness_{split}_11class.csv", index=False)
        mal_true, mal_pred = d.label.isin(MALIGNANT), d.pred.isin(MALIGNANT)
        results[f"{split}_11"] = {"macro_f1_11": float(f1.mean()),
                                  "zero_f1_classes": [c for c, v in zip(CLASSES11, f1) if v == 0],
                                  "malignant_sensitivity": float((mal_true & mal_pred).sum() / mal_true.sum()),
                                  "malignant_specificity": float((~mal_true & ~mal_pred).sum() / (~mal_true).sum())}
    pc = pd.read_csv(OUT / "per_class_test_11class.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(pc["class"], pc["f1"], color="#3b6ea8")
    for i, (f, s) in enumerate(zip(pc.f1, pc.support)):
        ax.text(i, f + 0.02, f"n={s}", ha="center", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("F1 (test)"); ax.set_title("Per-class F1 of the 11-class model on test")
    plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(FIG / "final_per_class_f1.png", dpi=150); plt.close(fig)
    (OUT / "analysis.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
