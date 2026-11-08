"""Generate notebooks/03_data_preparation_in_class.ipynb (Session 3 "Code in class").

    python scripts/build_session3_class_notebook.py --run

The code written in class (Session 3 - Data Preparation for Machine Learning Projects), run on the
full MILK10k download in this repo (all 10,480 images) instead of the 10-image sample.
"""

from __future__ import annotations

import argparse

from nbbuild import ROOT, Builder

nb = Builder(ROOT / "notebooks" / "03_data_preparation_in_class.ipynb")
md, code = nb.md, nb.code

md("""
# Session 3 — Data Preparation for Machine Learning Projects
**Code in class** · Computer Vision & Speech Recognition · Joshua Cole

The code from class, executed on the full MILK10k download (10,480 images, 5,240 lesions) rather than the 10-image sample
the course repo ships with. The only change to the class code is the data location (`data/raw/milk10k/`).
""")

code(r'''
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# repo root = the first parent that contains the raw MILK10k download
ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
            if (p / "data" / "raw" / "milk10k" / "metadata.csv").is_file())
DATA_DIR = ROOT / "data" / "raw" / "milk10k"
METADATA_PATH = DATA_DIR / "metadata.csv"                 # per IMAGE
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"    # per LESION (one-hot diagnosis)

IMG_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", DATA_DIR / "images"))

def image_path(isic_id: str) -> Path:
    """isic_id -> path of the .jpg file under IMG_DIR."""
    return IMG_DIR / f"{isic_id}.jpg"

def available_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the rows whose image file exists on disk (a no-op with the full dataset)."""
    mask = df["isic_id"].apply(lambda iid: image_path(iid).exists())
    return df[mask].reset_index(drop=True)

def dominant_class(gt: pd.DataFrame) -> pd.Series:
    """gt is one-hot with exactly one positive class per lesion -> one string label per lesion."""
    return gt.drop(columns="lesion_id").idxmax(axis=1)

def imbalance_ratio(gt: pd.DataFrame, lesion_ids: Optional[pd.Series] = None) -> float:
    """max class count / min class count, computed on a given subset of lesions."""
    sub = gt if lesion_ids is None else gt[gt["lesion_id"].isin(set(lesion_ids))]
    counts = sub.drop(columns="lesion_id").sum()
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())

print("Images folder :", IMG_DIR.relative_to(ROOT))
print("Sample only?  :", IMG_DIR == ROOT / "data" / "images")
''')

md("## Why data preparation matters + MILK10k recap")
code(r'''
df = pd.read_csv(METADATA_PATH)   # per IMAGE  (10,480 rows)
gt = pd.read_csv(GT_PATH)          # per LESION (5,240 rows, one-hot diagnosis)
print("metadata:", df.shape, "| gt:", gt.shape)
print("unique lesions:", df["lesion_id"].nunique())

# Exactly 2 images per lesion?
print("\nimages per lesion:")
print(df.groupby("lesion_id").size().value_counts())
print("\nimage types:")
print(df["image_type"].value_counts())
df.head(3)
''')

md("""
## Train / Validation / Test — three jobs

| Split | Used for | Touched |
|---|---|---|
| Train | fitting weights | every step |
| Validation | tuning, choosing epochs, comparing models | every epoch |
| Test | one honest final estimate | once |

The rule to remember: if you tune on test, it becomes a validation set.
""")

md("## The group-leakage trap\nWhat goes wrong if we split image by image? Do the naive thing and measure the damage.")
code(r'''
from sklearn.model_selection import train_test_split

# NAIVE: split rows (images), stratified on diagnosis_1
naive_train, naive_test = train_test_split(
    df, test_size=0.2, stratify=df["diagnosis_1"], random_state=0
)
leaked = set(naive_train["lesion_id"]) & set(naive_test["lesion_id"])
n_test_lesions = naive_test["lesion_id"].nunique()
print(f"Lesions present in BOTH train and test: {len(leaked)}")
print(f"That is {len(leaked) / n_test_lesions:.0%} of the test lesions -> the model has already 'seen' them.")
''')

md("## MILK10k class distribution (real numbers)")
code(r'''
class_cols = [c for c in gt.columns if c != "lesion_id"]
dist = gt[class_cols].sum().sort_values(ascending=False).astype(int)
table = pd.DataFrame({"lesions": dist, "% of total": (100 * dist / len(gt)).round(1)})
display(table)

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.bar(table.index, table["lesions"], color="tab:blue")
ax.bar_label(bars)
ax.set_ylabel("# lesions")
ax.set_title("MILK10k — lesions per diagnosis class")
plt.show()
''')

md("## Why is MILK10k so imbalanced? (sampling bias)\nMost lesions were biopsied: doctors biopsy what worries them, so this is not the real-world prevalence.")
code(r'''
lesions = df.drop_duplicates("lesion_id")
print("diagnosis_1 (per lesion):")
print((lesions["diagnosis_1"].value_counts(normalize=True) * 100).round(1))
print("\ndiagnosis_confirm_type (per image):")
print((df["diagnosis_confirm_type"].value_counts(normalize=True) * 100).round(1))
''')

md("## Metrics that survive imbalance\nWith ~48 % BCC, a model that always predicts BCC looks \"okay\" on accuracy and is useless.")
code(r'''
from sklearn.metrics import accuracy_score, f1_score

y = dominant_class(gt)          # one label per lesion, e.g. 'BCC'
y_dummy = pd.Series("BCC", index=y.index)   # a "model" that always says BCC
print(f"accuracy : {accuracy_score(y, y_dummy):.3f}   <- looks decent...")
print(f"macro-F1 : {f1_score(y, y_dummy, average='macro', zero_division=0):.3f}   <- reveals the truth")
''')

md("## The cost of errors is not symmetric\nA missed melanoma (false negative) is far worse than an unnecessary biopsy (false positive). Look at per-class recall, not just the average.")
code(r'''
from sklearn.metrics import classification_report

report = classification_report(y, y_dummy, zero_division=0, output_dict=True)
print(f"Recall on MEL: {report['MEL']['recall']:.2f}   <- every melanoma missed")
print(f"Recall on BCC: {report['BCC']['recall']:.2f}")

# Illustrative (made-up) costs: the operating point is a *clinical* decision
COST_FN, COST_FP = 100, 1
n_mel = int((y == "MEL").sum())
print(f"\nIllustrative cost of the always-BCC model on melanomas alone: {n_mel} misses x {COST_FN} = {n_mel * COST_FN}")
''')

md("## Strategies for class imbalance\nClass weights · oversampling · undersampling · targeted augmentation · stratified splitting.")
code(r'''
from sklearn.utils.class_weight import compute_class_weight

classes = np.array(sorted(y.unique()))
counts = y.value_counts()

# 1) CLASS WEIGHTS: rare class => bigger weight in the loss
weights = compute_class_weight("balanced", classes=classes, y=y)
print(pd.Series(weights, index=classes).round(2).sort_values(ascending=False))
''')
code(r'''
# 2) OVERSAMPLING: sample lesions with probability ~ 1 / class_count  (this is with replacement)
sample_w = y.map(1.0 / counts)
oversampled = y.sample(n=len(y), replace=True, weights=sample_w, random_state=0)
print("Oversampled class counts (roughly uniform now, but rare-class rows are REPEATED):")
print(oversampled.value_counts().sort_index())

# 3) UNDERSAMPLING: cut every class down to the smallest one
n_min = counts.min()
print(f"\nUndersampling to {n_min}/class keeps only {n_min * len(counts)} of {len(y)} lesions -> throws data away")
''')

md("## Stratified + grouped splitting\nGroup by `lesion_id` and stratify by diagnosis: `StratifiedGroupKFold`.")
md("### Cross-validation for rare classes\nWith rare classes (DF: 52, MAL_OTH: 9) a single split may leave a class out. k-fold rotates the held-out fold so every lesion is validated exactly once.")
code(r'''
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
fold_counts, seen = {}, []
for k, (tr_idx, va_idx) in enumerate(sgkf.split(gt, y, groups=gt["lesion_id"])):
    fold_counts[f"fold {k}"] = y.iloc[va_idx].value_counts()
    seen.extend(gt["lesion_id"].iloc[va_idx])

# Every lesion validated exactly once:
assert len(seen) == len(set(seen)) == len(gt)

# Validation-class counts per fold — notice the rare classes still show up in every fold
pd.DataFrame(fold_counts).fillna(0).astype(int).loc[classes]
''')

md("""
## What is data leakage? (types)

| Type | Example |
|---|---|
| Group | same `lesion_id` in train and test |
| Preprocessing | normalising with mean/std computed on all data |
| Duplicate | near-identical re-photographs on both sides |
| Target | a feature that encodes the label |

Below: preprocessing leakage and target leakage.
""")
code(r'''
# --- Preprocessing leakage: statistics must come from TRAIN only ---
tr_idx, te_idx = next(sgkf.split(gt, y, groups=gt["lesion_id"]))
train_l = set(gt["lesion_id"].iloc[tr_idx])
test_l = set(gt["lesion_id"].iloc[te_idx])

avail = available_subset(df)          # only images that exist on disk
tr_sample = avail[avail["lesion_id"].isin(train_l)].sample(min(60, avail["lesion_id"].isin(train_l).sum()), random_state=0)
te_sample = avail[avail["lesion_id"].isin(test_l)].sample(min(60, avail["lesion_id"].isin(test_l).sum()), random_state=0)

def mean_rgb(ids):
    means = []
    for iid in ids:
        im = Image.open(image_path(iid))
        im.draft("RGB", (128, 128))           # fast JPEG downscale on load
        means.append(np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32).mean(axis=(0, 1)) / 255)
    return np.mean(means, axis=0)

print("CORRECT (train only)  :", mean_rgb(tr_sample["isic_id"]).round(4))
print("LEAKY  (train + test) :", mean_rgb(pd.concat([tr_sample, te_sample])["isic_id"]).round(4))
print("-> the difference is small here, but the *principle* is: the test set must not influence ANY fitted quantity.")
''')
code(r'''
# --- Target leakage: does a column secretly encode the label? ---
print((pd.crosstab(df["diagnosis_confirm_type"], df["diagnosis_1"], normalize="index") * 100).round(1))
print("\nIf a column predicts the label this strongly, it is NOT available at prediction time -> don't use it as an input.")
''')

md("## Case study — shortcut learning in medical imaging\nModels can learn acquisition artefacts (scanner, ruler, pen ink) instead of pathology. Check whether such fields correlate with the label.")
code(r'''
print("image_manipulation vs diagnosis_1 (row %):")
print((pd.crosstab(df["image_manipulation"].fillna("none"), df["diagnosis_1"], normalize="index") * 100).round(1))
print("\nimage_type vs diagnosis_1 (row %):")
print((pd.crosstab(df["image_type"], df["diagnosis_1"], normalize="index") * 100).round(1))
''')

md("## Data-quality checks before you split\nAudit missingness first. A NaN is only a bug if the field should always be populated.")
code(r'''
miss = df.isna().sum()
audit = pd.DataFrame({"missing": miss, "% missing": (100 * miss / len(df)).round(1)})
audit = audit[audit["missing"] > 0].sort_values("missing", ascending=False)
display(audit)
''')
code(r'''
# Actions from the slide
clean = df.copy()
clean["age_approx"] = clean["age_approx"].fillna(clean["age_approx"].median())   # numeric -> median
clean["anatom_site_general"] = clean["anatom_site_general"].fillna("unknown")    # categorical -> explicit level
# diagnosis_4 / melanocytic: NaN = "not applicable" -> leave alone
print(clean[["age_approx", "anatom_site_general"]].isna().sum())
''')

md("## Augmentation — why and what\nLabel-preserving transforms: flip · rotate · brightness/contrast · crop/zoom. Train set only.\n\n### Real augmentations on a MILK10k image")
code(r'''
import torchvision.transforms.functional as TF

sample = available_subset(df).iloc[0]
img = Image.open(image_path(sample["isic_id"])).convert("RGB")
print(sample["isic_id"], "|", sample["image_type"], "|", sample["diagnosis_1"], "| size:", img.size)

w, h = img.size
variants = {
    "original":        img,
    "horizontal flip": TF.hflip(img),
    "rotate 15°":      TF.rotate(img, 15),
    "brightness ×1.45": TF.adjust_brightness(img, 1.45),
    "zoom crop":       TF.resized_crop(img, top=h // 10, left=w // 10, height=int(h * 0.8), width=int(w * 0.8), size=(h, w)),
}
fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3))
for ax, (name, im) in zip(axes, variants.items()):
    ax.imshow(im); ax.set_title(name); ax.axis("off")
plt.tight_layout(); plt.show()
''')

md("""
### Augmentation cautions for medical images
Vertical flip: fine for skin (no canonical "up"), wrong for X-rays. Aggressive colour jitter is dangerous: redness and
pigmentation are diagnostic. Crops must not cut the lesion out. Compare a conservative jitter with an aggressive one:
""")
code(r'''
import torch
import torchvision.transforms as T

torch.manual_seed(0)
mild = T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15)
aggressive = T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.9, hue=0.5)
panels = {"original": img,
          "mild #1": mild(img), "mild #2": mild(img),
          "AGGRESSIVE #1": aggressive(img), "AGGRESSIVE #2": aggressive(img)}
fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
for ax, (name, im) in zip(axes, panels.items()):
    ax.imshow(im); ax.set_title(name); ax.axis("off")
plt.tight_layout(); plt.show()
print("Hue shifts of +-0.5 turn skin green/blue -- no longer a plausible lesion. Every augmentation is a claim: 'this never changes the label'.")
''')

md("""
## A robust pipeline — the order matters
**Load + validate → Group + stratify split → Freeze test set → Augment (train only) → Dataset / DataLoader**

Takeaways: (1) split on the grouping unit (`lesion_id`), not the row; (2) MILK10k is imbalanced by construction, so accuracy
misleads; (3) `StratifiedGroupKFold` is leak-free and class-balanced; (4) leakage is silent, so suspiciously good results
deserve an audit; (5) augmentations must respect the medical meaning of the image.

### Recap — loading MILK10k
`df` is per image, `gt` is per lesion (one-hot). They share `lesion_id`.
""")
code(r'''
df = pd.read_csv(METADATA_PATH)
gt = pd.read_csv(GT_PATH)
print(df.shape, gt.shape)
print(df["lesion_id"].nunique(), "unique lesions")
''')

md("### A grouped, stratified split\n`idxmax` turns the one-hot columns into a single dominant class per lesion (used for stratification). We split at lesion level, and only afterwards join back to images.")
code(r'''
from sklearn.model_selection import StratifiedGroupKFold

y = gt.drop(columns="lesion_id").idxmax(axis=1)   # dominant class per lesion
groups = gt["lesion_id"]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(sgkf.split(gt, y, groups))
train_lesions = set(gt.loc[train_idx, "lesion_id"])
test_lesions = set(gt.loc[test_idx, "lesion_id"])
print(len(train_lesions), "train lesions |", len(test_lesions), "test lesions")
''')

md("### Verify zero leakage\nTrust, but verify: never skip the assert.")
code(r'''
overlap = train_lesions & test_lesions
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} lesions in both splits!"

train_df = df[df["lesion_id"].isin(train_lesions)]
test_df = df[df["lesion_id"].isin(test_lesions)]
print(f"train: {len(train_df)} images / {len(train_lesions)} lesions")
print(f"test:  {len(test_df)} images / {len(test_lesions)} lesions")
assert len(train_df) == 2 * len(train_lesions) and len(test_df) == 2 * len(test_lesions)
''')
code(r'''
# Bonus: stratification worked? Class proportions should be ~equal in both splits
props = pd.DataFrame({
    "train %": y.loc[train_idx].value_counts(normalize=True) * 100,
    "test %":  y.loc[test_idx].value_counts(normalize=True) * 100,
}).round(1)
props
''')

md("### Quantifying imbalance\nOne number for the problem. Compute it on the train split only: never peek at test proportions.")
code(r'''
counts = (gt.loc[gt["lesion_id"].isin(train_lesions), gt.columns.drop("lesion_id")]
            .sum().sort_values(ascending=False))
ratio = counts.max() / counts.min()
print(counts)
print(f"\nimbalance ratio (max/min class) = {ratio:.1f}x")

counts.plot.bar(logy=True, figsize=(7, 3), title="Train class counts (log scale)")
plt.show()
''')

md("## Basic data-quality checks\n`Image.verify()` catches truncated / corrupt files without fully decoding them.")
code(r'''
from PIL import UnidentifiedImageError

avail = available_subset(df)
n = min(300, len(avail))
bad_files = []
for iid in avail["isic_id"].sample(n, random_state=0):
    try:
        with Image.open(image_path(iid)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError):
        bad_files.append(iid)
print(f"{len(bad_files)} unreadable files out of {n} checked")
print(df[["age_approx", "anatom_site_general"]].isna().mean().round(4))
''')
code(r'''
# Bonus: image sizes — do all images share one resolution? (if not -> we need a resize step)
sizes = [Image.open(image_path(i)).size for i in avail["isic_id"].sample(min(100, len(avail)), random_state=0)]
print(pd.Series(sizes).value_counts().head())
''')

md("## An augmentation pipeline\nMild, bounded parameters. `eval_transform` has no randomness: validation and test must be reproducible.")
code(r'''
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ToTensor(),
])
eval_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
print(train_transform)
''')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute the notebook and save outputs")
    args = ap.parse_args()
    p = nb.write()
    print(f"wrote {p.relative_to(ROOT)}  ({len(nb.cells)} cells)")
    if args.run:
        nb.run()
        print("executed with outputs saved")
