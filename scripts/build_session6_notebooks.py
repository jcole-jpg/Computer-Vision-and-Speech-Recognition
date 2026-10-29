"""Generate the two Session 6 notebooks.

    notebooks/06_neural_networks.ipynb  the class notebook (class.ipynb), pointed at this repo's
                                        data/features/baseline, plus the executed mini-challenge
    notebooks/06_homework.ipynb         scoreboard, nn.Conv2d answer, MLP on raw 32x32x3 pixels

    python scripts/build_session6_notebooks.py          # write both
    python scripts/build_session6_notebooks.py --run    # write, execute, save outputs

Needs ``class.ipynb`` (the instructor's Session 6 notebook, not committed) in the repo root and
``artifacts/session05/baseline_scores.json`` (run ``build_session5_notebook.py --run`` first).
"""

from __future__ import annotations

import argparse
import json
import re

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

from nbbuild import ROOT, SETUP, Builder

CLASS_NB = ROOT / "class.ipynb"
BASELINE = ROOT / "artifacts" / "session05" / "baseline_scores.json"


def _strip(src: str, pattern: str) -> str:
    """Drop instructor reference numbers quoted in trailing comments (our splits give other numbers)."""
    return re.sub(r"[ \t]*#[ \t]*(?:" + pattern + r")[ \t]*$", "", src, flags=re.M)


def _swap(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"class.ipynb changed, cannot patch: {old[:50]!r}"
    return src.replace(old, new)


# ===================================================================================
# 1. the class notebook, patched
# ===================================================================================

DATA_CELL = r'''
from PIL import Image
from scipy import ndimage

# ---- Session 5 ---------------------------------------------------------------
def color_stats(rgb):
    return np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))])

def color_histogram(rgb, bins=8):
    n_pixels = rgb.shape[0] * rgb.shape[1]
    return np.concatenate([np.histogram(rgb[:, :, c], bins=bins, range=(0, 256))[0] / n_pixels
                           for c in range(3)])

def texture_features(rgb):
    gray = rgb.mean(axis=2)
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_sq_mean = ndimage.uniform_filter(gray**2, size=9)
    return np.array([np.hypot(gx, gy).mean(),
                     (local_sq_mean - local_mean**2).mean(),
                     ndimage.laplace(gray).var()])

def extract_features(isic_id):
    rgb = np.array(Image.open(mk.config.IMAGE_DIR / f"{isic_id}.jpg").convert("RGB").resize((200, 150)), dtype=float)
    return np.concatenate([color_stats(rgb), color_histogram(rgb), texture_features(rgb)])
# ------------------------------------------------------------------------------------------

# This repo's committed, lesion-level splits (Session 3: leak-free), dermoscopic image per lesion.
frames = F.load_lesion_splits()
train_df, val_df, test_df = [frames[k].assign(isic_id=frames[k]["derm_id"]) for k in ("train", "val", "test")]

# The same 33 features, from the cache written by src/milk10k/features.py (extracted once for all images).
cols = F.FEATURE_GROUPS["color_stats"] + F.FEATURE_GROUPS["color_hist"] + F.FEATURE_GROUPS["texture"]
feats = F.feature_table(pd.concat([d["isic_id"] for d in (train_df, val_df, test_df)]),
                        cache=F.FEATURE_DIR / "features.npz")
data = {f"X_{n}": feats.loc[d["isic_id"], cols].to_numpy()
        for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
assert np.allclose(data["X_val"][:5], np.stack([extract_features(i) for i in val_df["isic_id"][:5]]))   # cache == the code above

X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
print(X_train.shape, X_val.shape, X_test.shape)

# Session 5 results on THESE splits (artifacts/session05, written by notebooks/05_classical_baseline.ipynb)
baseline = json.loads((ROOT / "artifacts" / "session05" / "baseline_scores.json").read_text())
grid = pd.read_csv(ROOT / "artifacts" / "session05" / "feature_model_grid.csv")
g33 = grid[grid["feature set"] == "Session 5 set (33)"].set_index("model")["val_f1"]
FOREST = baseline["forest_33"]["val_macro_f1"]
print(f"Session 5 on the same 33 features, val macro-F1: majority {g33['majority']:.3f}, "
      f"logreg {g33['logistic regression']:.3f}, kNN {g33['kNN (k=15)']:.3f}, forest {FOREST:.3f}")
'''

INTRO_CELL = """
**Adapted for this repository.** This is the Session 6 class notebook with three changes: (1) the course helper package
`common` is replaced by this repo's own code (`src/milk10k`: lesion-level splits from `artifacts/splits`, the 33 features
cached by `features.py`); (2) every "Session 5" reference number is read from *our* `artifacts/session05` results, because our
splits differ from the instructor's, so the numbers quoted in the slides (forest 0.534, MLP 0.493) do not apply; (3) the
mini-challenge (§11) is answered below it. The homework is in `06_homework.ipynb`.
"""


def build_class_notebook() -> Builder:
    src = nbf.read(CLASS_NB, as_version=4)
    nb = Builder(ROOT / "notebooks" / "06_neural_networks.ipynb")
    base = json.loads(BASELINE.read_text())
    cells = [c.source for c in src.cells]

    for i, c in enumerate(src.cells):
        text = c.source
        if i == 2:
            text = SETUP + "\nimport copy\nimport torch\nimport torch.nn as nn\n"
        elif i == 28:
            text = DATA_CELL
        elif i == 29:
            text = _strip(text, r"tensor\(.*")
        elif i == 30:
            text = _strip(text, r"tensor\(.*")
        elif i == 33:
            text = _strip(text, r"\d+ batches per epoch")
        elif i == 34:
            text = _strip(text, r"last row:.*")
        elif i == 35:
            text = _swap(text, 'axes[1].axhline(0.534, color="k", ls="--", label="Session 5 forest (0.534)")',
                         'axes[1].axhline(FOREST, color="k", ls="--", label=f"Session 5 forest ({FOREST:.3f})")')
        elif i == 36:
            text = _strip(_swap(text, "(Session 5: majority 0.272, logreg 0.437, kNN 0.474, forest 0.534)",
                                "(Session 5: majority {g33['majority']:.3f}, logreg {g33['logistic regression']:.3f}, "
                                "kNN {g33['kNN (k=15)']:.3f}, forest {FOREST:.3f})"), r"0\.\d+")
        elif i == 40:
            text = _strip(text, r"train 0\.\d+.*")
        elif i == 42:
            text = _strip(text, r"0\.\d+\s+0\.\d+")
            text = _swap(text, 'baseline = json.loads((ROOT / "session_05_classical_ml" / "_output" / "baseline_scores.json").read_text())\n', "")
            text = _swap(text, '[baseline["val_macro_f1"], baseline["test_macro_f1"]]',
                         '[baseline["forest_33"]["val_macro_f1"], baseline["forest_33"]["test_macro_f1"]]')
        elif i == 43:
            text = _swap(text, 'out_dir = ROOT / "session_06_neural_networks" / "_output"\nout_dir.mkdir(exist_ok=True)',
                         'out_dir = ROOT / "artifacts" / "session06"\nout_dir.mkdir(parents=True, exist_ok=True)')
        elif i == 45:
            text = (f"Change **one thing at a time**, train, and note the final **validation** macro-F1:\n\n"
                    f"1. learning rate: `lr=0.1`, then `lr=1e-5`\n2. width: `hidden=(4,)`, then `hidden=(256,)`\n"
                    f"3. depth: `hidden=(64, 64)`\n\n"
                    f"Number to beat: Session 5's random forest on the same 33 features, **{base['forest_33']['val_macro_f1']:.3f}** "
                    f"(validation macro-F1 on this repo's splits; the slides' 0.534 came from different splits). "
                    f"The test set stays closed.")
        new = new_markdown_cell(text) if c.cell_type == "markdown" else new_code_cell(text)
        nb.cells.append(new)
        if i == 0:
            nb.md(INTRO_CELL)
        if i == 45:
            add_mini_challenge(nb, base)
    return nb


MINI_CODE = r'''
# one change at a time, everything else as in the baseline run: 50 epochs, batch 64, Adam, class-weighted loss, seed 0
runs = {
    "baseline  (64,)  lr 1e-3":    ("-",              dict(hidden=(64,)),     dict(lr=1e-3)),
    "lr = 0.1":                    ("learning rate",  dict(hidden=(64,)),     dict(lr=0.1)),
    "lr = 1e-5":                   ("learning rate",  dict(hidden=(64,)),     dict(lr=1e-5)),
    "hidden = (4,)":               ("width",          dict(hidden=(4,)),      dict(lr=1e-3)),
    "hidden = (256,)":             ("width",          dict(hidden=(256,)),    dict(lr=1e-3)),
    "hidden = (64, 64)":           ("depth",          dict(hidden=(64, 64)),  dict(lr=1e-3)),
}
t0 = time.perf_counter()
curves, rows = {}, []
for name, (what, arch, opt) in runs.items():
    m = make_mlp(**arch)
    h = fit(m, **opt)
    curves[name] = h
    rows.append({"run": name, "changed": what, "parameters": sum(p.numel() for p in m.parameters()),
                 "train loss": h.train_loss.iloc[-1], "val loss": h.val_loss.iloc[-1],
                 "final val macro-F1": h.val_f1.iloc[-1],
                 "best epoch (val F1)": f"{h.val_f1.max():.3f} @ {int(h.val_f1.idxmax())}"})
mini = pd.DataFrame(rows).set_index("run")
mini["vs baseline"] = mini["final val macro-F1"] - mini.iloc[0]["final val macro-F1"]
mini["vs Session 5 forest"] = mini["final val macro-F1"] - FOREST
print(f"{time.perf_counter() - t0:.0f} s")
mini.round(3)
'''

NOISE_CODE = r'''
# How much does a run move with the random seed alone? (initial weights + mini-batch order), five seeds, validation only
seed_f1 = {}
for name in ["baseline  (64,)  lr 1e-3", "hidden = (256,)", "hidden = (64, 64)"]:
    _, arch, opt = runs[name]
    seed_f1[name] = np.array([fit(make_mlp(seed=s, **arch), **opt).val_f1.iloc[-1] for s in range(5)])
    print(f"{name:26s} seeds 0-4: {seed_f1[name].round(3)}   mean {seed_f1[name].mean():.3f}   std {seed_f1[name].std():.3f}")
seed_f1 = seed_f1["baseline  (64,)  lr 1e-3"]

f = curves["lr = 0.1"].val_f1
print(f"lr = 0.1, val macro-F1 over the 50 epochs: min {f.min():.3f}, median {f.median():.3f}, max {f.max():.3f}; "
      f"epochs below 0.35: {(f < 0.35).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
cols = dict(zip(runs, ["#2E6FBA", "#F08200", "#D9534F", "#7BAFD4", "#1B3F73", "#4DB6AC"]))
for name, h in curves.items():
    axes[0].plot(h.index, h.val_f1, color=cols[name], label=name, lw=1.4)
    axes[1].plot(h.index, h.val_loss, color=cols[name], label=name, lw=1.4)
axes[0].axhline(FOREST, color="k", ls="--", lw=1, label=f"Session 5 forest ({FOREST:.3f})")
axes[0].axhspan(seed_f1.mean() - seed_f1.std(), seed_f1.mean() + seed_f1.std(), color="#2E6FBA", alpha=0.12)
axes[0].set_title("validation macro-F1  (band = seed std of the baseline)"); axes[0].set_xlabel("epoch")
axes[1].set_title("validation loss"); axes[1].set_xlabel("epoch"); axes[1].set_yscale("log"); axes[1].legend(fontsize=7.5, frameon=False)
plt.tight_layout(); plt.show()
'''


def add_mini_challenge(nb: Builder, base: dict) -> None:
    nb.code(MINI_CODE)
    nb.code(NOISE_CODE)
    nb.md(MINI_READING)
    nb.md("**Homework before Session 7** is answered in `06_homework.ipynb` (scoreboard, `nn.Conv2d` arguments, MLP on raw pixels).")


MINI_READING = r"""
### Reading the mini-challenge

Final validation macro-F1, one change at a time (seed 0): baseline 0.489 · `lr=0.1` **0.280** · `lr=1e-5` **0.383** · `hidden=(4,)` 0.454 · `hidden=(256,)` 0.496 · `hidden=(64, 64)` 0.513. Number to beat: Session 5 forest, 0.575. **Nothing beats it** (best −0.062). Test split not touched.

- **`lr=0.1` is too large.** The steps overshoot, so the loss never settles (train loss 1.014, val loss 2.07, both near where training starts). F1 swings from epoch to epoch between 0.22 and 0.49 (median 0.37, final epoch 0.280), so its final number is a coin flip, not a result.
- **`lr=1e-5` is too small.** 50 epochs of 59 small steps barely move the weights: train loss 1.017, F1 0.383, and still rising at the last epoch (best epoch = 49). Nothing is broken; it would need roughly 100× more steps.
- **Width.** `(4,)` has 151 weights and cannot fit much: F1 0.454, but the lowest validation loss (0.861) because it cannot overfit either. `(256,)` has 9,475 weights, trains lower (0.545) but validates worse (1.30); its F1 is the baseline's within noise.
- **Depth.** `(64, 64)` gives 0.513, the best of the six, but it is one seed. Over seeds 0-4 the baseline is 0.488 ± 0.008, `(256,)` 0.498 ± 0.011 and `(64, 64)` 0.491 ± 0.011: **width above 64 and a second layer are indistinguishable from the baseline**; seed 0 was a lucky draw for the deep net. Only the learning rate (both directions) and the very narrow net move F1 by more than the seed noise.
- **Loss and F1 disagree.** For every network with room to learn, validation loss starts rising after roughly 15-25 epochs while F1 stays flat: F1 only looks at which class wins, the loss also punishes confidence. Early stopping on validation loss would be the cheap next step; more capacity is not.

**Choice:** keep `lr=1e-3`, one hidden layer of 64 (simplest net that reaches the plateau). On 33 summary statistics and 3,742 lesions the forest remains the stronger model, with the MLP about 0.08 behind on validation.
"""


# ===================================================================================
# 2. the homework notebook
# ===================================================================================

def build_homework() -> Builder:
    nb = Builder(ROOT / "notebooks" / "06_homework.ipynb")
    md, code = nb.md, nb.code

    md(r"""
# Session 6 homework — scoreboard, `nn.Conv2d`, and an MLP on raw pixels

1. Put the MLP's validation and test macro-F1 on the scoreboard next to Session 5's forest.
2. Read the `nn.Conv2d` docs: which arguments are Session 4's kernel size, stride and padding?
3. Train the MLP on raw pixels (32×32×3 = 3,072 numbers): better or worse than 33 features, and why?

Everything uses this repo's lesion-level splits (`artifacts/splits`, seed 42) and the MLP recipe of the class notebook
(`src/milk10k/mlp.py`: Adam lr 1e-3, 50 epochs, batch 64, class-weighted cross-entropy, scaler fitted on train only).
The test split is scored **once per final model**; nothing is chosen on it.
""")

    code(SETUP + r'''
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from milk10k import filters as flt, mlp
from scipy import ndimage

IMG = mk.config.IMAGE_DIR
OUT = mk.config.ARTIFACT_DIR / "session06"
OUT.mkdir(parents=True, exist_ok=True)
base = json.loads((mk.config.ARTIFACT_DIR / "session05" / "baseline_scores.json").read_text())

frames = F.load_lesion_splits()                                     # one row per lesion; dermoscopic image = derm_id
ids = pd.concat([f["derm_id"] for f in frames.values()])
feats = F.feature_table(ids, cache=F.FEATURE_DIR / "features.npz")
y = {k: mlp.encode_labels(v["diagnosis_1"]) for k, v in frames.items()}
w = mlp.inverse_frequency_weights(y["train"])
print({k: len(v) for k, v in frames.items()}, "class weights", w.numpy().round(2))

def prepare(X):
    """train-only StandardScaler -> float32 tensors per split."""
    sc = StandardScaler().fit(X["train"])
    return {k: mlp.to_tensor(sc.transform(v)) for k, v in X.items()}, sc

def train_and_score(X, hidden=(64,), dropout=0.0, weight_decay=0.0, test=True, seed=0):
    m = mlp.make_mlp(X["train"].shape[1], hidden=hidden, dropout=dropout, seed=seed)
    hist = mlp.fit(m, X["train"], y["train"], X["val"], y["val"], class_weights=w, weight_decay=weight_decay, seed=seed)
    out = {"model": m, "hist": hist, "params": sum(p.numel() for p in m.parameters()),
           "val": mlp.scores(m, X["val"], y["val"])}
    if test:
        out["test"] = mlp.scores(m, X["test"], y["test"])
    return out
''')

    # ---------------------------------------------------------------- 1
    md(r"""
## 1. The MLP on the 33 features, and the scoreboard

Same network as the class notebook: 33 → 64 → 3 (2,371 parameters). The features are Session 5's 33 (colour statistics, colour histogram, texture) of the dermoscopic image.
""")
    code(r'''
X33, _ = prepare({k: F.build_matrix("Session 5 set (33)", v, feats) for k, v in frames.items()})
r33 = train_and_score(X33)
print(f"MLP 33-64-3: {r33['params']:,} parameters   val macro-F1 {r33['val']['macro_f1']:.3f}   "
      f"test macro-F1 {r33['test']['macro_f1']:.3f}   test accuracy {r33['test']['accuracy']:.3f}")
cls = OUT / "mlp_scores.json"
if cls.exists():
    c = json.loads(cls.read_text())
    print(f"class notebook run: val {c['val_macro_f1']:.3f}   test {c['test_macro_f1']:.3f}   test accuracy {c['test_accuracy']:.3f}"
          "   (same recipe, a different mini-batch order)")
''')

    # ---------------------------------------------------------------- 3 (computed before the table)
    md(r"""
## 3. MLP on raw pixels (done before the table so the scoreboard is complete)

Each dermoscopic photograph is shrunk to 32×32 (the 4:3 photo is squashed to a square) and flattened to 3,072 numbers,
scaled per pixel-channel with train-only statistics, then fed to the same network: 3,072 → 64 → 3. No hand-made feature: the
network has to find colour and texture structure from raw intensities by itself.
""")
    code(r'''
def load_pixels(isic_id):
    return F.load_rgb(IMG / f"{isic_id}.jpg", size=(32, 32)).astype(np.float32)        # (32, 32, 3)

t0 = time.perf_counter()
with ThreadPoolExecutor(8) as pool:
    raw = {k: np.stack(list(pool.map(load_pixels, v["derm_id"]))) for k, v in frames.items()}
print({k: v.shape for k, v in raw.items()}, f"{time.perf_counter() - t0:.0f} s")
Xraw, raw_scaler = prepare({k: v.reshape(len(v), -1) for k, v in raw.items()})        # 3,072 columns

rraw = train_and_score(Xraw)
rreg = train_and_score(Xraw, dropout=0.3, weight_decay=1e-3, test=False)              # one untuned regularised try (val only)
print(f"raw 3072-64-3:        {rraw['params']:,} parameters   val {rraw['val']['macro_f1']:.3f}   "
      f"test {rraw['test']['macro_f1']:.3f}   test accuracy {rraw['test']['accuracy']:.3f}")
print(f"raw + dropout/decay:  val {rreg['val']['macro_f1']:.3f}  (validation only)")
''')

    code(r'''
# Is raw > features on validation more than seed noise? Five seeds each (validation only), plus per-class F1.
from sklearn.metrics import f1_score
five = {name: [train_and_score(X, test=False, seed=s) for s in range(5)]
        for name, X in [("33 features", X33), ("raw pixels", Xraw)]}
spread = pd.DataFrame({name: [r["val"]["macro_f1"] for r in rs] for name, rs in five.items()})
print("val macro-F1 over seeds 0-4:\n", spread.round(3).T.assign(mean=spread.mean(), std=spread.std(ddof=0)).round(3).to_string())

def per_class_f1(model, X):
    model.eval()
    with torch.no_grad():
        pred = model(X["val"]).argmax(1)
    return f1_score(y["val"], pred, average=None, labels=[0, 1, 2])
pc = pd.DataFrame({"33 features": per_class_f1(r33["model"], X33), "raw pixels": per_class_f1(rraw["model"], Xraw)}, index=F.CLASSES)
pc.round(3).assign(**{"val lesions": np.bincount(y["val"].numpy(), minlength=3)})
''')

    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=False)
for ax, (name, r) in zip(axes[:2], [(f"33 features ({r33['params']:,} params)", r33), (f"raw pixels ({rraw['params']:,} params)", rraw)]):
    r["hist"][["train_loss", "val_loss"]].plot(ax=ax, color=[mk.ORANGE, mk.BLUE], title=name)
    ax.set_xlabel("epoch"); ax.set_ylim(0.1, 3.8)
axes[2].plot(r33["hist"].val_f1, color=mk.BLUE, label="33 features")
axes[2].plot(rraw["hist"].val_f1, color=mk.ORANGE, label="raw pixels")
axes[2].plot(rreg["hist"].val_f1, color=mk.AQUA, label="raw + dropout 0.3 + decay 1e-3")
axes[2].axhline(base["forest_33"]["val_macro_f1"], color="k", ls="--", lw=1, label="Session 5 forest (33)")
axes[2].set_title("validation macro-F1"); axes[2].set_xlabel("epoch"); axes[2].legend(fontsize=7.5, frameon=False)
plt.tight_layout(); plt.show()
pd.DataFrame({name: {"parameters": r["params"], "final train loss": round(r["hist"].train_loss.iloc[-1], 3), "final val loss": round(r["hist"].val_loss.iloc[-1], 3),
                     "min val loss (epoch)": f"{r['hist'].val_loss.min():.3f} @ {int(r['hist'].val_loss.idxmin())}",
                     "final val macro-F1": round(r["hist"].val_f1.iloc[-1], 3)}
              for name, r in [("33 features", r33), ("raw pixels", rraw), ("raw + dropout/decay", rreg)]})
''')

    md(r"""
**Why? A test of the "no spatial prior" argument.** A fully-connected layer treats pixel (5, 7) and pixel (5, 8) as unrelated inputs and
learns a separate weight for every position. A lesion shifted a few pixels is, to it, an entirely new input. The 33 hand-made features
(colour statistics, histograms) do not depend on where things are. To check this, every validation image is rolled sideways by
12.5 % of its width (the content wraps around) and both models are scored again, without retraining.
""")
    code(r'''
def val_inputs(shift_frac):
    """Validation images rolled sideways, then turned into (33 features, raw 32x32 pixels) exactly as in training."""
    f33, px = [], []
    for i in frames["val"]["derm_id"]:
        rgb = F.load_rgb(IMG / f"{i}.jpg")                                           # 200 x 150
        rgb = np.roll(rgb, int(shift_frac * rgb.shape[1]), axis=1)
        f33.append(np.concatenate([F.color_stats(rgb), F.color_histogram(rgb), F.texture_features(rgb)]))
        small = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(rgb.astype(np.uint8)).resize((32, 32)), dtype=np.float32)
        px.append(small.ravel())
    return np.stack(f33), np.stack(px)

sc33 = StandardScaler().fit(F.build_matrix("Session 5 set (33)", frames["train"], feats))
inputs = {frac: val_inputs(frac) for frac in (0.0, 0.125)}
rows = []
for name, scaler, col in [("33 features", sc33, 0), ("raw pixels", raw_scaler, 1)]:
    for frac, arrays in inputs.items():
        X = mlp.to_tensor(scaler.transform(arrays[col]))
        f1 = [mlp.scores(r["model"], X, y["val"])["macro_f1"] for r in five[name]]
        rows.append({"model": name, "shift": f"{frac:.1%} of width", "val macro-F1 (mean of 5 seeds)": np.mean(f1), "std": np.std(f1)})
shift = pd.DataFrame(rows).set_index(["model", "shift"])
shift.round(3)
''')

    # ---------------------------------------------------------------- 1b scoreboard
    md(r"""
## 1b. Scoreboard

Validation and test macro-F1 of everything so far, on the same splits. "Session 5 forest (33)" is the like-for-like row (same 33 inputs
as the MLP); "Session 5 best" is the winner of the Session 5 grid, which also sees the clinical photograph.
""")
    code(r'''
def row(name, val, test, params, acc=None):
    return {"model": name, "parameters": params, "val macro-F1": val, "test macro-F1": test, "test accuracy": acc}
board = pd.DataFrame([
    row("majority class",                                  base["majority"]["val_macro_f1"], base["majority"]["test_macro_f1"], 0, 0.694),
    row("Session 5 forest, 33 features",                   base["forest_33"]["val_macro_f1"], base["forest_33"]["test_macro_f1"], None),
    row("Session 5 best: forest, derm + clinical (66)",    base["val_macro_f1"], base["test_macro_f1"], None, base["test_accuracy"]),
    row("Session 6 MLP 33-64-3 (33 features)",             r33["val"]["macro_f1"], r33["test"]["macro_f1"], r33["params"], r33["test"]["accuracy"]),
    row("Session 6 MLP 3072-64-3 (raw 32x32x3 pixels)",    rraw["val"]["macro_f1"], rraw["test"]["macro_f1"], rraw["params"], rraw["test"]["accuracy"]),
]).set_index("model")
board.round(3).to_csv(OUT / "scoreboard.csv")
(OUT / "homework_scores.json").write_text(json.dumps({
    "mlp_33": {"val_macro_f1": round(r33["val"]["macro_f1"], 3), "test_macro_f1": round(r33["test"]["macro_f1"], 3)},
    "mlp_raw_pixels": {"val_macro_f1": round(rraw["val"]["macro_f1"], 3), "test_macro_f1": round(rraw["test"]["macro_f1"], 3)},
    "mlp_raw_regularised_val_only": round(rreg["val"]["macro_f1"], 3)}, indent=2))
board.round(3)
''')

    # ---------------------------------------------------------------- 2 Conv2d
    md(r"""
## 2. `nn.Conv2d`: which arguments are Session 4's kernel size, stride and padding?
""")
    code(r'''
import inspect
print("nn.Conv2d" + str(inspect.signature(nn.Conv2d.__init__)).replace("(self, ", "("))
''')
    md(r"""
| Session 4 idea | `nn.Conv2d` argument | meaning |
|---|---|---|
| **kernel size** (3×3 Sobel, 7×7 Gabor…) | `kernel_size` | height × width of the window the filter slides over (an int `3` or a pair `(3, 3)`) |
| **stride** (how far the window moves) | `stride` (default 1) | step in pixels between two positions; stride 2 halves the output |
| **padding** (border handling; the `mode=` of `ndimage`) | `padding` (default 0), `padding_mode` | rows/columns added around the image so the window fits at the border; `padding_mode="zeros"`/`"reflect"`/`"replicate"` ≈ filling with 0 / `reflect` / `nearest` |
| **the kernel values** (the Sobel numbers) | `weight`, shape `(out_channels, in_channels, kH, kW)` | *learned* instead of hand-written |
| number of filters in the bank | `out_channels` | one output map per filter |
| grayscale vs RGB input | `in_channels` | 1 or 3; each filter has one 2-D kernel per input channel, summed |

Two more arguments exist but were not part of Session 4: `dilation` (spacing between kernel taps) and `groups`. Output size per axis:
`⌊(H + 2·padding − kernel_size) / stride⌋ + 1`. `Conv2d` computes cross-correlation (the kernel is *not* flipped), like `ndimage.correlate` and the Session 4 code.
""")
    code(r'''
# 1) A Conv2d holding the Sobel kernel IS Session 4's Sobel filter.
gray = F.load_rgb(IMG / f"{frames['val']['derm_id'].iloc[0]}.jpg").mean(axis=2)                # 150 x 200
x = torch.tensor(gray, dtype=torch.float32)[None, None]                                         # (batch, channels, H, W)
conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor(flt.SOBEL_X, dtype=torch.float32)                             # kernel size 3x3, values = Sobel
    out = conv(x)[0, 0].numpy()
print("Conv2d(Sobel) vs ndimage.sobel, max |difference|:", np.abs(out - ndimage.sobel(gray, axis=1, mode="nearest")).max())

# 2) The output-size formula, checked.
rows = []
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 1), (7, 2, 3), (5, 1, 2)]:
    o = nn.Conv2d(1, 1, kernel_size=k, stride=s, padding=p)(x).shape[-2:]
    rows.append({"kernel_size": k, "stride": s, "padding": p, "output (H, W)": tuple(o),
                 "formula (H, W)": ((150 + 2 * p - k) // s + 1, (200 + 2 * p - k) // s + 1)})
display(pd.DataFrame(rows))

# 3) Session 4's learned filters were exactly this layer: ResNet-18 conv1.
c1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
print("ResNet-18 conv1:", tuple(c1.weight.shape), f"{c1.weight.numel():,} weights; 224x224x3 in ->", tuple(c1(torch.zeros(1, 3, 224, 224)).shape[1:]), "out")
print(f"same image size through nn.Linear(3*224*224, 64): {3 * 224 * 224 * 64:,} weights - convolution shares one small kernel across all positions")
''')

    # ---------------------------------------------------------------- verdict
    md(VERDICT)
    return nb


VERDICT = r"""
## Answers

**1. Scoreboard** (above): the MLP on the 33 features scores **0.489 validation / 0.480 test** macro-F1, against **0.575 / 0.528** for the Session 5 forest on the same inputs (and 0.568 / 0.536 for the Session 5 winner, which also sees the clinical photograph; majority class 0.273). The MLP is 0.09 behind on validation, far beyond its seed noise (± 0.008); on test the 0.05 gap is inside the forest's bootstrap interval ([0.452, 0.597] in Session 5).

**2. `nn.Conv2d`:** `kernel_size` is Session 4's kernel size, `stride` is the step the window moves between positions, `padding` is the border handling (`padding_mode` chooses what fills it). The kernel *values*, the Sobel numbers, become the learned `weight` tensor, and `out_channels` is the number of filters in the bank. A `Conv2d` loaded with the Sobel kernel reproduces `ndimage.sobel` to float precision (difference 9e-5).

**3. Raw pixels (32 × 32 × 3 = 3,072 inputs): not better, roughly equal in macro-F1 and worse in every other way.**

| | 33 features | raw pixels |
|---|---|---|
| parameters | 2,371 | 196,867 |
| test macro-F1 (seed 0) | 0.480 | 0.484 |
| validation macro-F1, mean ± std over seeds 0-4 | 0.480 ± 0.007 | 0.485 ± 0.037 |
| final train / validation loss | 0.65 / 1.03 | 0.14 / 3.53 |

- The raw-pixel network's headline validation score (0.549) is the best of its five seeds; the average (0.485) is the same as the features'. Its higher test accuracy (0.724 against 0.637) does not carry over to macro-F1, which is a tie on test.
- **Why it does not win.** (a) *Overfitting.* 196,867 weights for 3,742 lesions: training loss falls to 0.14 while validation loss is lowest at epoch 6 (1.17) and ends at 3.53. Dropout 0.3 plus weight decay brings the final validation loss to 2.33 and F1 to 0.522 (one untuned try, validation only): better behaved, no better in F1. (b) *No notion of position or neighbourhood.* Every pixel is an unrelated input, so the network can only use per-pixel colour and brightness, the information the colour statistics already summarise; the texture and edge measures need neighbouring pixels, and 32 × 32 has blurred most of that away. (c) *Variance.* Five times larger across seeds than for the features.
- **Translation check.** Rolling each validation image by 12.5 % of its width costs the raw-pixel network 0.486 → 0.470 and the features nothing (0.480 → 0.481). The direction matches the argument that a fully-connected layer has no translation prior, but the drop is smaller than the seed spread (0.04), so this is weak evidence, not a proof.
- **What the 33 features have that raw pixels lack:** they are translation-invariant summaries, designed by a person, fewer than the number of training examples.  To get the benefit of raw pixels without the cost you need layers that share weights across positions: convolutions. `Conv2d(3, 64, 7)` has 9,408 weights, a `Linear` layer from a 224 × 224 × 3 image to 64 units has 9.6 million.
- Caveat: one split, 17 validation Indeterminate lesions (per-class F1 for that class is 0.26 raw vs 0.13 features, a difference of one or two lesions), so differences of a few hundredths in macro-F1 are not reliable.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute the notebooks and save outputs")
    args = ap.parse_args()
    for nb in (build_class_notebook(), build_homework()):
        p = nb.write()
        print(f"wrote {p.relative_to(ROOT)}  ({len(nb.cells)} cells)")
        if args.run:
            print("executing...")
            t = nb.run()
            print("executed with outputs saved")
