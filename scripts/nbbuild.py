"""Tiny shared helper for the notebook builders (scripts/build_session*_notebook*.py).

    nb = Builder(ROOT / "notebooks" / "05_x.ipynb")
    nb.md("..."); nb.code("..."); nb.write(); nb.run()

Notebooks are generated from Python so the cell order, the prose and the analysis code
have one source of truth and "Restart & Run All" is reproducible by construction.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''
# --- setup: find the repo root from wherever the notebook is opened -----------
import sys, json, time, warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "src" / "milk10k").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import milk10k as mk
from milk10k import features as F

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
'''


class Builder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.cells: list = []

    def md(self, text: str) -> None:
        self.cells.append(new_markdown_cell(text.strip("\n")))

    def code(self, text: str) -> None:
        self.cells.append(new_code_cell(text.strip("\n")))

    def write(self) -> Path:
        nb = new_notebook(cells=self.cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(nbf.writes(nb))
        return self.path

    def run(self, timeout: int = 3600) -> None:
        from nbclient import NotebookClient

        nb = nbf.read(self.path, as_version=4)
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(nb, str(self.path))
