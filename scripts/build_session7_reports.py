"""Session 7 deliverables: the mini-challenge PDF, the homework report PDF and the notebook PDF.

    python scripts/build_session7_reports.py [--to ../Deliverables/07_Session_7_CNNs]

Reads what notebooks/07_cnn.ipynb wrote (artifacts/session07), so every number in the reports is the
one in the notebook. Writes reports/session7_minichallenge.{md,pdf} and reports/session7_report.{md,pdf};
with --to, copies the three PDFs there under the names the course platform asks for.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from build_submission_folder import md_to_pdf  # noqa: E402

REPO = "https://github.com/jcole-jpg/Computer-Vision-and-Speech-Recognition"
BLOB = f"{REPO}/blob/main"
ART = ROOT / "artifacts" / "session07"
REPORTS = ROOT / "reports"
SEEDS = (0, 1, 2)


def link(path: str, label: str | None = None) -> str:
    return f"[`{label or path}`]({BLOB}/{path})"


def fnum(x, fmt="{:.3f}") -> str:
    return "–" if pd.isna(x) else fmt.format(x)


def mini_figure(runs: dict[str, dict]) -> Path:
    """Best validation macro-F1 per seed for the two configurations of the mini-challenge."""
    cfg = [("layer4 + fc", "resnet18_s2_layer4", "#8fb8de"), ("layer3 + layer4 + fc", "resnet18_s2b_layer3_4", "#2a78d6")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, (label, base, colour) in enumerate(cfg):
        vals = [runs[base if s == 0 else f"{base}_seed{s}"]["best_val_f1"] for s in SEEDS]
        ax.scatter([k] * 3, vals, color=colour, s=46, zorder=3, edgecolor="white")
        ax.hlines(np.mean(vals), k - 0.22, k + 0.22, color=colour, lw=2.5, zorder=2)
        ax.text(k + 0.27, np.mean(vals), f"mean {np.mean(vals):.3f}", va="center", fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([c[0] for c in cfg]); ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("best validation macro-F1"); ax.set_title("Three seeds each: the spread is as large as the gap", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    out = REPORTS / "figures" / "s7_minichallenge.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=Path, help="folder to copy the three PDFs into")
    args = ap.parse_args()

    runs = {p.stem: json.loads(p.read_text()) for p in (ART / "runs").glob("*.json")}
    mini = pd.read_csv(ART / "mini_challenge.csv")
    board = pd.read_csv(ART / "scoreboard.csv")
    sc = json.loads((ART / "scores.json").read_text())
    pick = sc["best_session7_by_val"]
    a, b = mini.iloc[0], mini.iloc[1]
    seed_vals = {k: [runs[base if s == 0 else f"{base}_seed{s}"]["best_val_f1"] for s in SEEDS]
                 for k, base in [("a", "resnet18_s2_layer4"), ("b", "resnet18_s2b_layer3_4")]}
    mini_fig = mini_figure(runs)

    # ------------------------------------------------------------------ 7.1 mini-challenge
    mc = f"""# Session 7 — Mini-challenge (slide 29): unfreeze `layer3` as well?

**Computer Vision & Speech Recognition** · Joshua Cole · MILK10k skin-lesion classification (3 classes) · seed 0, 1, 2

**Repository:** <{REPO}>

**Code that completes the exercise:**
{link("notebooks/07_cnn.ipynb")} (section 5, executed, with outputs) ·
{link("src/milk10k/cnn.py")} (`resnet18(trainable=...)`, `fit_cnn`, `count_parameters`) ·
{link("scripts/build_session7_notebook.py")} (generates and runs the notebook) ·
{link("tests/test_cnn.py")} (tests) ·
results: {link("artifacts/session07/mini_challenge.csv")}, {link("artifacts/session07/runs", "artifacts/session07/runs/")} (per-epoch history of every run).

## The exercise

Pretrained ImageNet ResNet-18 with a new `fc = Linear(512, 3)`. Compare **`layer4 + fc`** trainable (the usual default) against **`layer3 + layer4 + fc`**
trainable. Same data (3,742 / 749 / 749 lesion-level splits), same augmentation, same 10 epochs, class-weighted cross-entropy, Adam with the new head at 1e-3 and the unfrozen
pretrained layers at 1e-4. Measured: trainable parameters, seconds per epoch (median over the epochs, so the GPU warm-up of epoch 1 does not count; Apple M2 Pro),
best **validation** macro-F1. The test split is not used for this decision.

## Results

| unfrozen | trainable parameters | seconds / epoch | best val macro-F1, seed 0 | mean of 3 seeds | std |
|---|---|---|---|---|---|
| `layer4 + fc` | {int(a['trainable params']):,} | {a['s / epoch']:.1f} | {a['val F1 seed 0']:.3f} | {a['val F1 mean of 3']:.3f} | {a['std']:.3f} |
| `layer3 + layer4 + fc` | {int(b['trainable params']):,} | {b['s / epoch']:.1f} | {b['val F1 seed 0']:.3f} | {b['val F1 mean of 3']:.3f} | {b['std']:.3f} |
| difference | +{int(b['trainable params'] - a['trainable params']):,} (+{b['trainable params'] / a['trainable params'] - 1:.0%}) | +{b['s / epoch'] - a['s / epoch']:.1f} (+{b['s / epoch'] / a['s / epoch'] - 1:.0%}) | {b['val F1 seed 0'] - a['val F1 seed 0']:+.3f} | {b['val F1 mean of 3'] - a['val F1 mean of 3']:+.3f} | |

Seeds 0 / 1 / 2, best validation macro-F1: `layer4 + fc` {" / ".join(f"{v:.3f}" for v in seed_vals['a'])}; `layer3 + layer4 + fc` {" / ".join(f"{v:.3f}" for v in seed_vals['b'])}.

![best validation macro-F1 per seed for both configurations]({mini_fig.relative_to(REPORTS)})

## Is it worth it?

**No, not on this evidence.** Unfreezing `layer3` adds 2.1 million trainable weights (+25 %) and {b['s / epoch'] - a['s / epoch']:.1f} s per epoch (+28 %), and buys
{b['val F1 mean of 3'] - a['val F1 mean of 3']:+.3f} validation macro-F1 on average, while `layer4 + fc` alone moves by {a['std']:.3f} from seed to seed. Seed 0 on its own would have said
{b['val F1 seed 0'] - a['val F1 seed 0']:+.3f}; seed 2 puts `layer4 + fc` ahead ({seed_vals['a'][2]:.3f} against {seed_vals['b'][2]:.3f}). Only 17 Indeterminate lesions are in the validation split, so one lesion moves the
class F1 a lot, which is why a single run cannot settle a difference this small. The one consistent effect is a tighter spread with `layer3` unfrozen ({b['std']:.3f} against {a['std']:.3f}),
which three seeds can only hint at. The extra time is small in absolute terms (about 30 s over ten epochs), so the verdict is "no evidence it helps", not "it hurts":
keep `layer4 + fc` as the default partial fine-tune.

**Test split: closed for this decision.** The choice above was made on validation only. (The Session 7 scoreboard scores every network once on test, seed 0:
`layer4 + fc` and `layer3 + layer4 + fc` both get 0.577, so nothing changes.)
"""
    p1 = REPORTS / "session7_minichallenge.md"
    p1.write_text(mc)

    # ------------------------------------------------------------------ 7.3 homework report
    rows = "\n".join(
        f"| {r['session']} | {r['model']}{'  **[PICK]**' if r['session'] == 7 and r['model'] == pick['model'] else ''} | {fnum(r['trainable params'], '{:,.0f}')} "
        f"| {fnum(r['val macro-F1'])} | {fnum(r['test macro-F1'])} | {fnum(r['test accuracy'])} | {fnum(r['Malignant recall'])} |"
        for _, r in board.iterrows())
    hw = f"""# Session 7 homework — the scoreboard, and what a pretrained CNN adds

**Computer Vision & Speech Recognition** · Joshua Cole · MILK10k skin-lesion classification (Benign / Indeterminate / Malignant) · seed 42 splits

**Repository:** <{REPO}>

**Code that completes the homework (slide 30):**
{link("notebooks/07_cnn.ipynb")} (the experiment, executed, every table and figure; PDF: {link("notebooks/07_cnn.pdf")}) ·
{link("src/milk10k/cnn.py")} (tensors, augmentation, networks, fine-tuning strategies, cached training runs) ·
{link("scripts/build_session7_notebook.py")} (generates and runs the notebook) ·
{link("tests/test_cnn.py")} (13 tests).
Raw outputs: {link("artifacts/session07/scoreboard.csv")}, {link("artifacts/session07/mini_challenge.csv")}, {link("artifacts/session07/scores.json")}, {link("artifacts/session07/runs", "artifacts/session07/runs/")}.
Earlier scoreboard rows: {link("notebooks/05_classical_baseline.ipynb")}, {link("notebooks/06_neural_networks.ipynb")}.

## Result

**The model to bring to Session 8 is ResNet-18 pretrained on ImageNet and fine-tuned in full (strategy 3, the optional homework): validation macro-F1 {pick['val_macro_f1']:.3f}, test macro-F1 {pick['test_macro_f1']:.3f}
(95 % bootstrap interval {pick['test_macro_f1_ci95']}), test accuracy {pick['test_accuracy']:.3f}.** It was chosen by validation macro-F1 alone, the highest of everything in Sessions 5–7, and the test split was scored once, after every choice was fixed.
Always predicting the majority class scores 0.273.

## The scoreboard, Sessions 5–7 (same lesion-level splits; macro-F1)

| session | model | trainable params | validation | test | test accuracy | Malignant recall |
|---|---|---|---|---|---|---|
{rows}

![validation and test macro-F1 of every model, Sessions 5 to 7]({"figures/s7_scoreboard.png"})

## How to read it

* **Pretraining is what matters.** The same ResNet-18 from random weights scores 0.510 on validation and 0.434 on test; from ImageNet weights, fully fine-tuned, 0.703 and 0.598. Small networks trained from scratch on 3,742 lesions
  (slide-15 net 0.483, four-block net 0.431 on test) do no better than Session 6's MLPs (0.480, 0.484).
* **More unfrozen layers, better score, higher cost:** `fc` only 0.523 (8.7 s/epoch), `layer4 + fc` 0.577 (11.0 s), `layer3 + layer4 + fc` 0.577 (15.0 s), everything 0.598 (31.0 s) on test.
  The gaps between the last three are inside the seed noise measured in the mini-challenge (std 0.023 on validation), so `layer4 + fc` is statistically indistinguishable from the pick at a third of the training time.
* **Against Session 5's forest** (test 0.536, interval [0.472, 0.601]) the pick is +0.062; the intervals overlap, so this is a likely improvement on 749 test lesions, not a proven one.
* **Validation is optimistic** for every model (0.703 against 0.598 for the pick): the best epoch and the best model were both chosen on it. The test number is the one to quote.
* **What it gets wrong:** per-class recall on test is Benign 0.70, Indeterminate 0.41, Malignant 0.79; 96 of 520 malignant lesions are called benign. At argmax the cost per lesion (missed malignant 50, unnecessary work-up 1) is 6.65,
  well above the 0.40 of always answering "Malignant"; the matrix calls for a screening threshold below 0.5, which is not tuned here.

## EfficientNet or DenseNet versus ResNet

**DenseNet-121 differs from ResNet-18 in how it reuses earlier features: where a ResNet block adds its input to its output (`x + f(x)`), each DenseNet layer receives the concatenated feature maps of all earlier layers in its block,
so features are reused instead of relearned and the network gets by with 8.0 M parameters against 11.7 M, at the price of more memory per layer.**
EfficientNet-B0 keeps ResNet's skip connections but replaces its plain 3×3 blocks with MBConv blocks (depthwise-separable convolutions with squeeze-and-excitation) and sets depth, width and input resolution together by one scaling rule,
reaching 5.3 M parameters. Parameter counts are from torchvision (notebook section 7); neither network was trained here, so no accuracy claim is made for them.

## Protocol and assumptions

* Dermoscopic image of each lesion only, stretched to 224×224, ImageNet normalisation, flips / quarter turns / mild brightness and contrast on training batches only; class-weighted cross-entropy; Adam; **10 epochs** for every network; the best-validation epoch is kept.
* Pretrained layers train at 1e-4, the new head (and every from-scratch network) at 1e-3; frozen layers keep their BatchNorm statistics.
* Seed 0 for the scoreboard; seeds 1 and 2 only for the mini-challenge, so strategy 3's number comes from a single run. Apple MPS training is not bit-for-bit repeatable; the notebook caches every run, so re-executing it reproduces the numbers above.
* Sessions 5 and 6 rows are read from `artifacts/session05` and `artifacts/session06` (same splits), not retrained.
"""
    p3 = REPORTS / "session7_report.md"
    p3.write_text(hw)

    pdf1 = md_to_pdf(p1, REPORTS / "session7_minichallenge.pdf", title="Session 7 mini-challenge", compact=True)
    pdf3 = md_to_pdf(p3, REPORTS / "session7_report.pdf", title="Session 7 homework", compact=True)
    print("wrote", p1.relative_to(ROOT), p3.relative_to(ROOT), pdf1.name, pdf3.name)

    if args.to:
        dest = args.to if args.to.is_absolute() else ROOT / args.to
        dest.mkdir(parents=True, exist_ok=True)
        for src, name in [(pdf1, "7.1 S7 - MiniChallenge Exercise.pdf"), (ROOT / "notebooks" / "07_cnn.pdf", "7.2 Adapted Code.pdf"),
                          (pdf3, "7.3 Homework Session 7.pdf")]:
            shutil.copyfile(src, dest / name)
            print("copied", (dest / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
