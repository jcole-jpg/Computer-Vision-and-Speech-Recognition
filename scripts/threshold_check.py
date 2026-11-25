"""Pool question 11.25: does the leaderboard's 0.5 threshold, or per-class thresholds tuned on
validation, beat the argmax decision?  Reads artifacts/final/{val,test}_predictions.csv."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from milk10k.multiclass import CLASSES

def load(split):
    d = pd.read_csv(ROOT / f"artifacts/final/{split}_predictions.csv")
    Y = np.eye(11)[[CLASSES.index(c) for c in d.label]]
    return Y, d[CLASSES].to_numpy()

(Yv, Pv), (Yt, Pt) = load("val"), load("test")
f1 = lambda Y, B: f1_score(Y, B, average="macro", zero_division=0)
argmax = lambda P: np.eye(11)[P.argmax(1)]
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
from milk10k.final import tune_thresholds
thr = tune_thresholds().reindex(CLASSES).to_numpy()
rows = [("argmax", f1(Yv, argmax(Pv)), f1(Yt, argmax(Pt))),
        ("raw probabilities >= 0.5", f1(Yv, Pv >= 0.5), f1(Yt, Pt >= 0.5)),
        ("per-class thresholds tuned on val (submitted)", f1(Yv, Pv >= thr), f1(Yt, Pt >= thr))]
out = pd.DataFrame(rows, columns=["decision rule", "val_macro_f1", "test_macro_f1"]).round(4)
out["lesions with no class (test)"] = [0, int(((Pt >= 0.5).sum(1) == 0).sum()), int(((Pt >= thr).sum(1) == 0).sum())]
out.to_csv(ROOT / "artifacts/final/threshold_check.csv", index=False)
print(out.to_string(index=False)); print(dict(zip(CLASSES, thr)))
pc = pd.DataFrame({"class": CLASSES, "test_support": Yt.sum(0).astype(int),
                   "f1_argmax": f1_score(Yt, argmax(Pt), average=None, zero_division=0).round(3),
                   "f1_thresholds": f1_score(Yt, Pt >= thr, average=None, zero_division=0).round(3),
                   "threshold": thr})
pc.to_csv(ROOT / "artifacts/final/per_class_test_thresholds.csv", index=False)
print(pc.to_string(index=False))
