"""EDA first outputs -> reports/eda_first_outputs.pdf

    python scripts/build_eda_first_outputs.py

Four things, one lesion per row (data/processed/lesions.csv, 5,240 lesions):
  1. samples per class (diagnosis_1)       - bar plot + table
  2. samples per subclass (11 diagnoses)   - bar plot + table
  3. age range per class                   - box plot (whiskers = min/max) + table
  4. sex distribution per class            - 100 % stacked bars + table

The figures go to reports/figures/fo_*.png, the page is rendered as HTML and printed
to PDF with headless Chrome (same route as scripts/export_notebook_pdf.py).
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_notebook_pdf import to_pdf  # noqa: E402
from milk10k.data import CLASS_NAMES, MALIGNANT  # noqa: E402
from milk10k.viz import INK2, MUTED, apply_style, class_color  # noqa: E402

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "eda_first_outputs.pdf"
REPO_URL = "https://github.com/jcole-jpg/Computer-Vision-and-Speech-Recognition"

D1_ORDER = ["Benign", "Indeterminate", "Malignant"]
D1_COLOR = {"Benign": class_color("NV"), "Indeterminate": MUTED, "Malignant": class_color("MEL")}
SEX_COLOR = {"female": "#7a5cc7", "male": "#3f8f8a"}


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "lesions.csv")
    assert len(df) == 5240 and df.lesion_id.is_unique
    return df


def fig_path(name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG / name


def bar_counts(counts: pd.Series, colors: list[str], title: str, path: Path, *, horizontal: bool,
               figsize: tuple[float, float]) -> None:
    total = counts.sum()
    fig, ax = plt.subplots(figsize=figsize)
    labels = [f"{v:,}  ({v / total:.1%})" for v in counts.values]
    if horizontal:
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
        ax.set_xlabel("lesions")
        ax.grid(axis="y", visible=False)
        for b, t in zip(bars, labels[::-1]):
            ax.text(b.get_width() + total * 0.008, b.get_y() + b.get_height() / 2, t,
                    va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, counts.max() * 1.22)
    else:
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.6)
        ax.set_ylabel("lesions")
        ax.grid(axis="x", visible=False)
        for b, t in zip(bars, labels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + total * 0.01, t,
                    ha="center", fontsize=9, color=INK2)
        ax.set_ylim(0, counts.max() * 1.12)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def fig_age(df: pd.DataFrame, order11: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [3, 8]}, sharey=True)
    for ax, col, order, colors, title in (
        (axes[0], "diagnosis_1", D1_ORDER, [D1_COLOR[c] for c in D1_ORDER], "By class (diagnosis_1)"),
        (axes[1], "dx", order11, [class_color(c) for c in order11], "By subclass (11 diagnoses)"),
    ):
        data = [df.loc[df[col] == c, "age"].dropna().values for c in order]
        bp = ax.boxplot(data, whis=(0, 100), patch_artist=True, widths=0.6,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.8},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                        boxprops={"edgecolor": MUTED})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(order) + 1), order, rotation=35 if len(order) > 3 else 0,
                      ha="right" if len(order) > 3 else "center")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("age (years)")
    fig.suptitle("Age range per class — box = middle 50 %, line = median, whiskers = youngest / oldest",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def fig_sex(df: pd.DataFrame, order11: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [3, 8]}, sharey=False)
    for ax, col, order, title in (
        (axes[0], "diagnosis_1", D1_ORDER, "By class (diagnosis_1)"),
        (axes[1], "dx", order11, "By subclass (11 diagnoses)"),
    ):
        ct = pd.crosstab(df[col], df["sex"]).reindex(order)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        y = np.arange(len(order))[::-1]
        ax.barh(y, pct["female"].values, color=SEX_COLOR["female"], height=0.7, label="female")
        ax.barh(y, pct["male"].values, left=pct["female"].values, color=SEX_COLOR["male"], height=0.7, label="male")
        for yi, f, n in zip(y, pct["female"].values, ct.sum(axis=1).values):
            ax.text(f / 2, yi, f"{f:.0f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            ax.text(f + (100 - f) / 2, yi, f"{100 - f:.0f}%", ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")
            ax.text(101.5, yi, f"n={n:,}", va="center", fontsize=8, color=MUTED)
        ax.set_yticks(y, order)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of lesions")
        ax.axvline(50, color="white", lw=0.8, alpha=0.7)
        ax.grid(False)
        ax.set_title(title)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(df: pd.DataFrame, *, cls: str = "") -> str:
    return df.to_html(index=False, escape=False, classes=cls, border=0)


def age_table(df: pd.DataFrame, col: str, order: list[str], label: str) -> pd.DataFrame:
    rows = []
    for c in order:
        a = df.loc[df[col] == c, "age"]
        s = a.dropna()
        rows.append({label: c, "n": f"{len(a):,}", "missing": int(a.isna().sum()),
                     "min": f"{s.min():.0f}", "Q1": f"{s.quantile(.25):.0f}", "median": f"{s.median():.0f}",
                     "Q3": f"{s.quantile(.75):.0f}", "max": f"{s.max():.0f}",
                     "mean ± sd": f"{s.mean():.1f} ± {s.std():.1f}"})
    return pd.DataFrame(rows)


def sex_table(df: pd.DataFrame, col: str, order: list[str], label: str) -> pd.DataFrame:
    ct = pd.crosstab(df[col], df["sex"]).reindex(order)
    out = pd.DataFrame({label: order, "female": [f"{v:,}" for v in ct["female"]],
                        "male": [f"{v:,}" for v in ct["male"]],
                        "% female": [f"{v:.1f}" for v in ct["female"] / ct.sum(axis=1) * 100],
                        "% male": [f"{v:.1f}" for v in ct["male"] / ct.sum(axis=1) * 100]})
    return out


def uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    apply_style()
    df = load()
    n = len(df)

    # ---- 1. diagnosis_1 ------------------------------------------------------------------
    c1 = df.diagnosis_1.value_counts().reindex(D1_ORDER)
    f1 = fig_path("fo_1_class_counts.png")
    bar_counts(c1, [D1_COLOR[c] for c in D1_ORDER], "Lesions per class (diagnosis_1)", f1,
               horizontal=False, figsize=(6.2, 3.3))
    t1 = pd.DataFrame({"class (diagnosis_1)": D1_ORDER, "lesions": [f"{v:,}" for v in c1],
                       "share": [f"{v / n:.1%}" for v in c1], "images": [f"{2 * v:,}" for v in c1]})

    # ---- 2. 11 diagnoses -----------------------------------------------------------------
    c11 = df.dx.value_counts()
    order11 = list(c11.index)
    f2 = fig_path("fo_2_subclass_counts.png")
    bar_counts(c11, [class_color(c) for c in order11], "Lesions per subclass (11 diagnoses)", f2,
               horizontal=True, figsize=(7.2, 4.3))
    parent = df.groupby("dx").diagnosis_1.value_counts().unstack(fill_value=0)
    rows = []
    for c in order11:
        p = parent.loc[c]
        p = p[p > 0].sort_values(ascending=False)
        parents = " / ".join(f"{k} ({v})" if len(p) > 1 else k for k, v in p.items())
        rows.append({"subclass": c, "name": CLASS_NAMES[c], "diagnosis_1": parents,
                     "lesions": f"{c11[c]:,}", "share": f"{c11[c] / n:.1%}"})
    t2 = pd.DataFrame(rows)

    # ---- 3. age --------------------------------------------------------------------------
    f3 = fig_path("fo_3_age_range.png")
    fig_age(df, order11, f3)
    t3a = age_table(df, "diagnosis_1", D1_ORDER, "class (diagnosis_1)")
    t3b = age_table(df, "dx", order11, "subclass")
    miss = df[df.age.isna()]
    miss_note = (f"{len(miss)} lesions have no recorded age "
                 f"({', '.join(f'{k}: {v}' for k, v in miss.dx.value_counts().items())}); "
                 "they are left out of the statistics but counted in <i>n</i>.")

    # ---- 4. sex --------------------------------------------------------------------------
    f4 = fig_path("fo_4_sex_distribution.png")
    fig_sex(df, order11, f4)
    t4a = sex_table(df, "diagnosis_1", D1_ORDER, "class (diagnosis_1)")
    t4b = sex_table(df, "dx", order11, "subclass")

    # ---- readings (computed, not typed) --------------------------------------------------
    med = df.groupby("dx").age.median().sort_values()
    fem = (pd.crosstab(df.dx, df.sex)["female"] / pd.crosstab(df.dx, df.sex).sum(axis=1) * 100).sort_values()
    tot_f = (df.sex == "female").mean() * 100
    read_counts = (f"Malignant is {c1['Malignant'] / n:.0%} of the lesions and Benign {c1['Benign'] / n:.0%}; "
                   f"BCC alone is {c11['BCC'] / n:.0%}. The rarest subclass, MAL_OTH, has {c11['MAL_OTH']} lesions "
                   f"({c11['BCC'] // c11['MAL_OTH']}× fewer than BCC). This is a referral population, not a screening one.")
    read_age = (f"Median age runs from {med.iloc[0]:.0f} ({med.index[0]}) to {med.iloc[-1]:.0f} ({med.index[-1]}). "
                f"Benign lesions are younger (median {df[df.diagnosis_1 == 'Benign'].age.median():.0f}) than Malignant "
                f"({df[df.diagnosis_1 == 'Malignant'].age.median():.0f}) and Indeterminate "
                f"({df[df.diagnosis_1 == 'Indeterminate'].age.median():.0f}); ages span "
                f"{df.age.min():.0f}–{df.age.max():.0f} overall.")
    read_sex = (f"Overall {tot_f:.0f}% of lesions are female. The most male-skewed subclass is {fem.index[0]} "
                f"({fem.iloc[0]:.0f}% female) and the most female-skewed is {fem.index[-1]} ({fem.iloc[-1]:.0f}% female); "
                f"Indeterminate lesions are {t4a.set_index('class (diagnosis_1)').loc['Indeterminate', '% male']}% male.")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
  @page {{ size: A4; margin: 14mm 13mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1b1b1b; font-size: 10pt; line-height: 1.4; }}
  h1 {{ font-size: 19pt; margin: 0 0 2pt; }}
  .sub {{ color: #52514e; margin-bottom: 8pt; }}
  h2 {{ font-size: 12.5pt; margin: 14pt 0 5pt; padding-bottom: 3pt; border-bottom: 1px solid #d9d8d4; break-after: avoid; }}
  h2.np {{ break-before: page; margin-top: 0; }}
  img {{ display: block; margin: 4pt auto; max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 4pt 0 6pt; break-inside: avoid; }}
  th, td {{ text-align: left; padding: 2.5pt 6pt; border-bottom: 1px solid #e6e5e1; }}
  th {{ color: #52514e; font-weight: 600; background: #f6f5f1; }}
  td:nth-child(n+2) {{ font-variant-numeric: tabular-nums; }}
  .note {{ color: #52514e; font-size: 8.8pt; margin: 2pt 0 4pt; }}
  .read {{ background: #fafaf8; border: 1px solid #d9d8d4; border-radius: 6px; padding: 5pt 10pt; margin: 6pt 0; break-inside: avoid; }}
  a {{ color: #2a78d6; }}
</style></head><body>
<h1>MILK10k — EDA first outputs</h1>
<div class="sub">Computer Vision &amp; Speech Recognition · Joshua Cole · code:
  <a href="{REPO_URL}">{REPO_URL}</a></div>
<div class="note">Unit = lesion (one row per lesion, {n:,} lesions = {2 * n:,} images: one clinical close-up and one
dermatoscopic image each). Age and sex belong to the lesion, so image counts are exactly twice the lesion counts.
Source: <code>data/processed/lesions.csv</code>, built from <code>metadata.csv</code> + <code>training_gt.csv</code>;
script <code>scripts/build_eda_first_outputs.py</code>.</div>

<h2>1. Samples per class (diagnosis_1)</h2>
<img src="{uri(f1)}" style="width:60%">
{table(t1)}

<h2>2. Samples per subclass (11 diagnoses)</h2>
<img src="{uri(f2)}" style="width:72%">
{table(t2)}
<div class="read">{read_counts} AKIEC is split between two classes: the 123 Indeterminate lesions are all AKIEC.</div>

<h2 class="np">3. Age range per class</h2>
<img src="{uri(f3)}">
{table(t3a)}
{table(t3b)}
<div class="note">{miss_note}</div>
<div class="read">{read_age}</div>

<h2 class="np">4. Sex distribution per class</h2>
<img src="{uri(f4)}">
{table(t4a)}
{table(t4b)}
<div class="read">{read_sex}</div>
</body></html>"""

    with tempfile.TemporaryDirectory() as tmp:
        h = Path(tmp) / "eda_first_outputs.html"
        h.write_text(html, encoding="utf-8")
        to_pdf(h, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
