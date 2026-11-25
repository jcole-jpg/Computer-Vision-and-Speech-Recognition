"""Milestone 4: frozen ResNet-18 vs frozen ViT-B/16, same split, same head, same metric.

    python scripts/compare_cnn_vit.py      # after scripts/extract_frozen_features.py

Writes artifacts/final/cnn_vs_vit.csv (3-class and 11-class rows).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from milk10k.compare import compare
from milk10k.multiclass import CLASSES

z = np.load(ROOT / "data/processed/final/frozen_features.npz", allow_pickle=True)
splits = pd.concat([pd.read_csv(ROOT / f"artifacts/splits/{s}.csv") for s in ("train", "val", "test")])
dx1 = splits.drop_duplicates("lesion_id").set_index("lesion_id")["diagnosis_1"]
split = pd.Series(z["split"])
r3 = compare(z, pd.Series(dx1.loc[z["lesion"]].to_numpy()), split)
r3.insert(0, "target", "3 classes (diagnosis_1)")
r11 = compare(z, pd.Series(z["label"]), split, classes=CLASSES)
r11.insert(0, "target", "11 classes")
out = pd.concat([r3, r11]).round(4)
out.to_csv(ROOT / "artifacts/final/cnn_vs_vit.csv", index=False)
print(out.to_string(index=False))
