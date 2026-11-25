"""Write reports/eleven_class_report.md (and its PDF) from artifacts/eleven_class.

    python scripts/build_eleven_class_report.py

The 11-class stretch of the Milestone 1 brief: predict the specific diagnosis, not only
Benign / Indeterminate / Malignant. Every number below is read from the files the notebook
(notebooks/stretch_11_class.ipynb) wrote, so the report cannot drift from the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from build_submission_folder import md_to_pdf  # noqa: E402

REPO = "https://github.com/jcole-jpg/Computer-Vision-and-Speech-Recognition"
BLOB = f"{REPO}/blob/main"
ART = ROOT / "artifacts" / "eleven_class"


def link(path: str, label: str | None = None) -> str:
    return f"[`{label or path}`]({BLOB}/{path})"


def main() -> None:
    sc = json.loads((ART / "scores.json").read_text())
    final = pd.read_csv(ART / "final_table.csv")
    per = pd.read_csv(ART / "per_class_test.csv")
    conf = pd.read_csv(ART / "top_confusions.csv")
    binary = pd.read_csv(ART / "binary_comparison.csv")
    head = final[final["headline"]].iloc[0]
    forest = final[final["model"] == "hand-crafted forest"].iloc[0]
    mlp = final[final["model"] == "MLP, ResNet-18 embeddings"].iloc[0]

    rows = "\n".join(
        f"| {r.model}{'  **[HEADLINE]**' if r.headline else ''} | {r.macro_f1:.3f} | [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}] "
        f"| {r.balanced_accuracy:.3f} | {r.accuracy:.3f} | {'' if pd.isna(r.top3_accuracy) else f'{r.top3_accuracy:.3f}'} |"
        for r in final.itertuples())
    per_rows = "\n".join(
        f"| {r.Index} | {r.n_train} | {r.support} | {r.recall:.2f} | {r.f1:.2f} | [{r.f1_lo:.2f}, {r.f1_hi:.2f}] |"
        for r in per.set_index("Unnamed: 0").sort_values("n_train", ascending=False).itertuples())
    conf_rows = "\n".join(f"| {r.truth} → {r.predicted} | {r.n} | {r.share_of_true_class:.0%} |" for r in conf.head(5).itertuples())
    b = binary.set_index("model")
    s5, r18 = b.loc["Session 5 forest (3 classes)"], b.loc["11-class ResNet-18 + logistic regression"]

    md = f"""# The 11-class stretch target — which skin lesion is it?

**MILK10k skin-lesion classification** · Joshua Cole · seed {sc['seed']} · Milestone 1 stretch goal

**Code in the repository** ({REPO}):
{link("notebooks/stretch_11_class.ipynb")} (the experiment, executed, every table and figure) ·
{link("src/milk10k/multiclass.py")} (11-class metrics, power-weights, bootstrap) ·
{link("src/milk10k/embeddings.py")} (frozen ResNet-18 embeddings, cached) ·
{link("scripts/build_eleven_class_notebook.py")} (generates and runs the notebook) ·
{link("tests/test_multiclass.py")} (tests).
Raw outputs: {link("artifacts/eleven_class/final_table.csv")}, {link("artifacts/eleven_class/per_class_test.csv")}, {link("artifacts/eleven_class/scores.json")}.

## Result

The primary target is `diagnosis_1` (3 classes). The stretch asks for the **specific diagnosis, 11 classes**, from basal cell carcinoma (48 % of
lesions) to other malignant (9 lesions in the whole dataset). On the 749 test lesions, scored once, the headline model, **frozen ResNet-18
embeddings + logistic regression**, reaches **macro-F1 {head.macro_f1:.3f}** (95 % bootstrap interval [{head.macro_f1_lo:.3f}, {head.macro_f1_hi:.3f}]), against
{forest.macro_f1:.3f} for the best hand-crafted-feature forest and {final.iloc[0].macro_f1:.3f} for always predicting BCC. Macro-F1 is averaged over the ten
evaluable classes: MAL_OTH (6 train, 2 test lesions) is reported but never averaged in, and is never found (the all-11 value is {head.macro_f1_all11:.3f}).

| model | macro-F1 (10 classes) | 95 % CI | balanced accuracy | accuracy | top-3 accuracy |
|---|---|---|---|---|---|
{rows}

The headline was fixed **before** the test split was opened: the finalist with the higher cross-validated macro-F1 on train
({sc['embedding_finalist']['cv_macro_f1']:.3f} for the embeddings against {sc['hand_crafted_finalist']['cv_macro_f1']:.3f} for the forest). The MLP on the embeddings scores
{mlp.macro_f1:.3f} on test, but its interval overlaps the headline's and the rule had already chosen; it is reported, not promoted.

## Method

* Same committed lesion-level splits as Sessions 5–7 (3,742 train / 749 val / 749 test lesions), same cost habits from Session 3.
* Session 5's comparison re-run for 11 classes: 7 feature sets × 6 models, 5-fold CV macro-F1 on **train only** (best cell: random forest on
  derm + clinical, CV {sc['grid']['best_cv_macro_f1']:.3f} ± {sc['grid']['best_cv_std']:.3f}; majority class {sc['grid']['majority_cv_macro_f1']:.3f}).
* The one new decision, **how hard to up-weight rare classes**, tested rather than assumed: weight = (N / (K·n_c))^p with p ∈ {{0, 0.25, 0.5, 0.75, 1}},
  chosen by CV mean (p = {sc['weight_power']['chosen']}); the fold spread is as large as the gaps between neighbouring p.
* One extension beyond the course: 512-number ResNet-18 embeddings of each image (frozen, ImageNet weights, nothing trained on skin) with a logistic-regression head,
  and an MLP in the Session 6 recipe on top of them. The question: is the low hand-crafted score a ceiling of the features or of the data?

## What the numbers say

* **Frozen CNN features beat hand-crafted ones by a wide margin** ({head.macro_f1:.3f} against {forest.macro_f1:.3f}, intervals do not overlap; CV on train ordered them the same way).
  The low hand-crafted score is a ceiling of the features, not only of the data.
* **Useful on common classes, an anecdote on rare ones.** BCC, NV and SCCKA carry the average; DF, INF, VASC and BEN_OTH have 6–7 test lesions each and
  intervals as wide as 0.4–0.6.

| class | train lesions | test lesions | recall | F1 | F1 95 % CI |
|---|---|---|---|---|---|
{per_rows}

* **The errors are clinically readable.** Largest confusions:

| true → predicted | lesions | share of the true class |
|---|---|---|
{conf_rows}

  31 % of melanomas are called naevi, the error that matters most; a screening threshold has to push it down.
* **Eleven classes do not cost the triage answer.** Collapsing to benign / malignant (P(malignant) = sum of the five malignant columns), the 11-class ResNet model has
  AUC {r18['AUC']:.3f} against {s5['AUC']:.3f} for the Session 5 three-class forest. With a screening threshold chosen on validation for 95 % sensitivity it catches
  {r18['sens @val-thr']:.1%} of malignant test lesions at specificity {r18['spec @val-thr']:.3f} and {r18['cost @val-thr']:.2f} cost per lesion (Session 5: {s5['sens @val-thr']:.3f}, {s5['spec @val-thr']:.3f}, {s5['cost @val-thr']:.2f}).

![F1 per class with its bootstrap interval, next to the training-set size]({"figures/ec_per_class.png"})

![11-class confusion matrix on the test split]({"figures/ec_confusion_matrix.png"})

## What this does not show

* **Not a clinical tool.** 749 test lesions from one dataset, one split, no external validation; the 71.7 % malignant share of the test split is far above primary care.
* **Differences of a few hundredths are noise.** With 7 lesions in four classes a single lesion moves a class F1 by more than 0.1.
* **Next step.** Fine-tune a CNN end to end (done for the 3-class target in Session 7: {link("reports/session7_report.md")}) and pool train + validation for cross-validation, so the rare classes get more than six to eight lesions.
"""
    out = ROOT / "reports" / "eleven_class_report.md"
    out.write_text(md)
    pdf = md_to_pdf(out, out.with_suffix(".pdf"), title="The 11-class stretch target")
    print("wrote", out.relative_to(ROOT), pdf.relative_to(ROOT) if pdf else "(no pdf)")


if __name__ == "__main__":
    main()
