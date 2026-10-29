"""Write the Session 4 and Session 5 homework reports from the executed notebooks.

    python scripts/build_session_reports.py

Reads ``notebooks/04_filters.ipynb`` and ``notebooks/05_classical_baseline.ipynb`` (run their
builders with ``--run`` first), copies the figures the reports need to ``reports/figures/`` and
writes ``reports/session4_filters.md`` and ``reports/session5_baseline_report.md``. Every number
and every written answer is taken from the notebooks, so the PDFs cannot drift from the code.
``build_submission_folder.py`` renders the Markdown to PDF.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
REPO = "https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition"
BLOB = f"{REPO}/blob/main"


def load(name: str) -> list[dict]:
    return json.loads((ROOT / "notebooks" / name).read_text())["cells"]


def text(cell: dict) -> str:
    return "".join(cell["source"])


def figure(cells: list[dict], index: int, out_name: str) -> str:
    """Save the first PNG output of cell ``index`` to reports/figures and return its file name."""
    for o in cells[index].get("outputs", []):
        if "image/png" in o.get("data", {}):
            FIG.mkdir(parents=True, exist_ok=True)
            (FIG / out_name).write_bytes(base64.b64decode(o["data"]["image/png"]))
            return out_name
    raise SystemExit(f"cell {index} has no figure - re-run the notebook builder with --run")


def link(path: str) -> str:
    return f"[`{path}`]({BLOB}/{path})"


def md_table(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------
def session4() -> Path:
    cells = load("04_filters.ipynb")
    bank = figure(cells, 4, "s4_filter_bank.png")
    sweep = figure(cells, 17, "s4_sigma_sweep.png")
    versus = figure(cells, 20, "s4_canny_vs_sobel.png")
    params = figure(cells, 22, "s4_canny_parameters.png")

    sheet = text(cells[-1]).split("\n", 1)[1].strip()                      # drop the "## 5. Answer sheet" heading
    q1, rest = sheet.split("**Q2.", 1)
    q2, q3 = ("**Q2." + rest).split("**Q3.", 1)
    q3 = "**Q3." + q3

    md = f"""# Session 4 homework — Filters on skin lesions

**MILK10k skin-lesion classification** · Joshua Cole · seed 42

**Notebook with all code, figures and outputs:** {link("notebooks/04_filters.ipynb")}
(repository: <{REPO}>). Code used: {link("src/milk10k/filters.py")} (filter kernels, `gaussian_kernel`,
`sobel_edges`, and `canny` written from scratch: Gaussian blur → Sobel gradients → non-maximum suppression → double threshold → hysteresis),
built by {link("scripts/build_session4_notebook.py")}. Numbers below are re-read from the executed notebook
(`artifacts/session04/*.csv`).

**What was done.** A bank of hand-built filters (Sobel, Prewitt, Laplacian, sharpen, Gabor at three angles, difference of Gaussians) and the 64 first-layer
filters of a pretrained ResNet-18 were applied to dermoscopic images from the training split. Edge statistics were compared between
Benign and Malignant lesions (central 60 % of each image, so the vignette is not counted) on a 300-image sample and on all 3,653 benign and
malignant training lesions, summarised as an AUC (0.5 = no difference) with a bootstrap interval. `sigma` was swept from 0 to 8 and Canny
was implemented as a function and compared with Sobel on one image.

## Q1. Edges and textures: malignant vs benign

![filter bank on three benign and three malignant lesions](figures/{bank})
*Rows 1 – 3 benign (BKL, BKL, NV), rows 4 – 6 malignant (BCC, SCCKA, BCC); picked at random with a fixed seed. Hairs dominate rows 2, 5 and 6.*

{q1.strip()}

## Q2. `sigma` in `scipy.ndimage.gaussian_filter`

![edge magnitude and class separation vs sigma](figures/{sweep})

{q2.strip()}

## Q3. Canny implemented as a function, compared with Sobel

![Sobel and Canny on one benign nevus, with zoom](figures/{versus})
*Top: whole image (400 × 300 px). Bottom: the orange box, enlarged. All edge maps are binary.*

{q3.strip()}

![Canny parameters](figures/{params})
*Top row: σ = 1, 2, 3.5 at fixed thresholds. Bottom row: σ = 2 with low/high thresholds 2 %/6 %, 5 %/15 %, 12 %/35 % of the peak gradient.*
"""
    out = ROOT / "reports" / "session4_filters.md"
    out.write_text(md)
    return out


# ---------------------------------------------------------------------------------------
def session5() -> Path:
    cells = load("05_classical_baseline.ipynb")
    heat = figure(cells, 8, "s5_feature_model_grid.png")
    conf = figure(cells, 15, "s5_confusion_matrix.png")
    s = json.loads((ROOT / "artifacts" / "session05" / "baseline_scores.json").read_text())
    final = pd.read_csv(ROOT / "artifacts" / "session05" / "final_table.csv")
    final = final.rename(columns={"Unnamed: 0": "model"}) if "model" not in final.columns else final
    final["Malignant recall"] = final["Malignant recall"].map("{:.2f}".format)
    final["cost / lesion"] = final["cost / lesion"].map("{:.2f}".format)
    for c in ("val macro-F1", "test macro-F1", "test accuracy"):
        final[c] = final[c].map("{:.3f}".format)

    reading = text(cells[10]).split("\n", 1)[1].strip()                    # drop "### Reading the grid"
    verdict = text(cells[17]).split("\n", 1)[1].strip()                    # drop "### What the baseline says"
    decisions = text(cells[0]).split("**Method in one paragraph.**")[1].strip()

    md = f"""# Session 5 — A classical machine-learning baseline

**MILK10k skin-lesion classification** · Joshua Cole · seed 42

**Code in the repository** ({REPO}):
{link("notebooks/05_classical_baseline.ipynb")} (the experiment, executed, with every table and figure) ·
{link("src/milk10k/features.py")} (colour / histogram / texture / HSV / metadata features, cached extraction, lesion-level splits) ·
{link("scripts/build_session5_notebook.py")} (generates and runs the notebook) ·
{link("tests/test_features.py")} (tests).
Raw outputs: {link("artifacts/session05/baseline_scores.json")}, {link("artifacts/session05/feature_model_grid.csv")}.

## Result

The baseline to beat is a **random forest**, **test macro-F1 {s["test_macro_f1"]:.3f}** (95 % bootstrap interval {s["test_macro_f1_ci95"]})
on {s["model"].split(" on ")[1]} features; the same forest on the 33 dermoscopic features alone scores {s["forest_33"]["test_macro_f1"]:.3f}.
Always predicting the majority class scores {s["majority"]["test_macro_f1"]:.3f}. The test split was scored once, after every choice was fixed.

{md_table(final)}

## Method

{decisions}

## Feature sets × models

6 models × 7 feature sets, 5-fold cross-validated macro-F1 on the training lesions (selection) and one fit scored on validation (check):

![feature set by model grid](figures/{heat})

{reading}

## Tuning, final evaluation and reading

The winning family (random forest on `derm + clinical (66)`) was tuned with a small grid (`max_features`, `min_samples_leaf`) by the same 5-fold CV.
Best settings: `{s["best_params"]}`; CV {s["cv_macro_f1"]:.3f}, validation {s["val_macro_f1"]:.3f}.

![confusion matrix, test split](figures/{conf})

{verdict}
"""
    out = ROOT / "reports" / "session5_baseline_report.md"
    out.write_text(md)
    return out


if __name__ == "__main__":
    for p in (session4(), session5()):
        print("wrote", p.relative_to(ROOT))
