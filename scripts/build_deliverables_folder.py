"""Assemble Deliverables/: every submission, numbered and named as in the course PDF.

    python scripts/build_deliverables_folder.py

Order and titles follow "ALL-COMPUTER VISION AND SPEECH RECOGNITION.pdf" (Sessions 1-7).
Nothing here is a source of truth: every file is copied from where it is generated.
Session 7 is copied in by its own build, so this script only lists what is already in
Deliverables/07_Session_7 and never touches it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Deliverables"

# (session folder, [(deliverable title, file name in the folder, source in the repo)])
PLAN = [
    ("01_Session_1_Introduction", [
        ("1.1  EDA - Some first outputs", "1.1 EDA - Some first outputs.pdf", "reports/eda_first_outputs.pdf"),
        ("1.2  Homework Session 1 (repo link + editor screenshot)", "1.2 Homework Session 1.pdf", "exercises/exercise_1.pdf"),
    ]),
    ("02_Session_2_Image_Representation", [
        ("2.1  Exercise - Exploring Resolution and Intensity (RGB, Grayscale, Histogram, Normalization)",
         "2.1 Exercise - Exploring Resolution and Intensity.pdf", "notebooks/02_resolution_and_intensity.pdf"),
        ("2.1  the same notebook (.ipynb, committed to the repo)",
         "2.1 Exercise - Exploring Resolution and Intensity.ipynb", "notebooks/02_resolution_and_intensity.ipynb"),
        ("2.1  extra: the Session 2 image-basics exercises done in class",
         "2.1 extra - Session 2 image basics in class.pdf", "exercises/exercise_2.pdf"),
        ("2.2  Homework - Extended Exploratory Data Analysis (EDA) & Data Pipeline (report + repo link)",
         "2.2 Homework - Extended EDA and Data Pipeline.pdf", "exercises/exercise_3.pdf"),
    ]),
    ("03_Session_3_Data_Preparation", [
        ("3.1  Code in class (outputs visible)", "3.1 Code in class.pdf", "notebooks/03_data_preparation_in_class.pdf"),
        ("3.2  Homework Session 3, Part A (notebook as PDF)", "3.2 Homework Session 3 - Part A notebook.pdf",
         "Finished/01_Part_A_notebook/homework_part_a.pdf"),
        ("3.2  Part A notebook (.ipynb)", "3.2 Homework Session 3 - Part A notebook.ipynb",
         "Finished/01_Part_A_notebook/homework_part_a.ipynb"),
        ("3.2  Part B, Course Project Milestone 1 report", "3.2 Homework Session 3 - Part B Milestone 1 report.pdf",
         "Finished/02_Part_B_milestone1/milestone1_report.pdf"),
        ("3.2  Part B, SUBMISSION.md (a direct link to each file)", "3.2 Homework Session 3 - Part B SUBMISSION.md",
         "SUBMISSION.md"),
    ]),
    ("04_Session_4_Filters_and_Convolution", [
        ("4.1  Python Code done in Class (notebook as PDF)", "4.1 Python Code done in Class.pdf", "notebooks/04_filters.pdf"),
        ("4.2  Homework (single PDF of answers + link to the notebook)", "4.2 Homework.pdf",
         "Finished/04_session4_filters/session4_filters.pdf"),
    ]),
    ("05_Session_5_Classical_ML", [
        ("5    no deliverable is listed for Session 5; supplementary: baseline report",
         "5 supplementary - Classical baseline report.pdf", "Finished/05_session5_baseline/session5_baseline_report.pdf"),
        ("5    supplementary: baseline notebook as PDF", "5 supplementary - Classical baseline notebook.pdf",
         "notebooks/05_classical_baseline.pdf"),
    ]),
    ("06_Session_6_Neural_Networks", [
        ("6.1  Mini-Challenge (learning rate, width, depth; updated code as PDF)", "6.1 Mini-Challenge.pdf",
         "notebooks/06_neural_networks.pdf"),
        ("6.2  Homework (.ipynb as PDF: scoreboard, nn.Conv2d, raw-pixel MLP)", "6.2 Homework.pdf",
         "notebooks/06_homework.pdf"),
    ]),
]
S7_DIR = "07_Session_7_CNNs"
S7_TITLES = "7.1  S7 - MiniChallenge Exercise  |  7.2  Adapted Code  |  7.3  Homework Session 7"


def main() -> None:
    lines = [
        "# Deliverables: Computer Vision & Speech Recognition (Sessions 1-7)",
        "",
        "Joshua Cole. Listed in the order and with the titles of the course page. Upload each file to its slot.",
        "Repository (linked from the PDFs): https://github.com/jcole-jpg/Computer-Vision-and-Speech-Recognition",
        "",
    ]
    missing: list[str] = []
    for folder, items in PLAN:
        d = OUT / folder
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
        lines += [f"## {folder.replace('_', ' ')}", ""]
        for title, name, src in items:
            s = ROOT / src
            if not s.exists():
                missing.append(src)
                continue
            shutil.copy2(s, d / name)
            lines.append(f"- {title}: `{folder}/{name}`")
        lines.append("")

    s7 = OUT / S7_DIR
    lines += [f"## {S7_DIR.replace('_', ' ')}", "", f"{S7_TITLES}", ""]
    files = sorted(p.name for p in s7.glob("*")) if s7.exists() else []
    lines += [f"- `{S7_DIR}/{n}`" for n in files] or ["- (not yet built)"]
    lines += ["", "Sessions 8-10 are outside this set. Session 4.1 (clarifications) lists no deliverable."]
    (OUT / "00_README.md").write_text("\n".join(lines) + "\n")

    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"Deliverables/: {n} files")
    if missing:
        print("MISSING sources:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
