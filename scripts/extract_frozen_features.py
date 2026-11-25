"""Milestone 4: frozen ResNet-18 and ViT-B/16 features of every lesion's dermoscopic image.

    python scripts/extract_frozen_features.py [--start 0 --stop 5240]

Same preprocessing for both backbones (stretched 224x224, ImageNet mean/std, no augmentation).
ResNet-18 with ``fc = Identity`` gives the 512 pooled numbers; ViT-B/16 with ``heads = Identity``
gives the 768-number class token. Run in chunks (--start/--stop) on a slow CPU; the chunks are
merged into data/processed/final/frozen_features.npz. ~14 s per 100 lesions on a 4-core CPU.
"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

from milk10k import cnn, final
from milk10k.config import IMAGE_DIR

OUT = ROOT / "data" / "processed" / "final"

p = argparse.ArgumentParser()
p.add_argument("--images", default=str(IMAGE_DIR))
p.add_argument("--start", type=int, default=0)
p.add_argument("--stop", type=int, default=10**9)
a = p.parse_args()

df = pd.concat([final.dermoscopic(ROOT / f"artifacts/splits/{s}.csv") for s in ("train", "val", "test")],
               ignore_index=True)
A, B = a.start, min(a.stop, len(df))
index = final.image_index(a.images)
net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = nn.Identity(); net.eval()
vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1"); vit.heads = nn.Identity(); vit.eval()
feats, t0 = {"resnet18": [], "vit_b_16": []}, time.time()
with torch.no_grad():
    for k in range(A, B, 16):
        x = cnn.normalise(final.stack([index[i] for i in df.isic_id.iloc[k:min(k + 16, B)]]))
        feats["resnet18"].append(net(x).numpy())
        feats["vit_b_16"].append(vit(x).numpy())
        print(f"{k}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
sub = df.iloc[A:B]
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / f"frozen_chunk_{A}.npz", ids=sub.isic_id.to_numpy(), lesion=sub.lesion_id.to_numpy(),
         split=sub.split.to_numpy(), label=sub.label.to_numpy(),
         **{k: np.concatenate(v) for k, v in feats.items()})
chunks = sorted(OUT.glob("frozen_chunk_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
if sum(len(np.load(c)["ids"]) for c in chunks) == len(df):            # all chunks done -> merge
    zs = [np.load(c, allow_pickle=True) for c in chunks]
    np.savez(OUT / "frozen_features.npz", **{k: np.concatenate([z[k] for z in zs]) for k in zs[0].files})
    print("merged ->", OUT / "frozen_features.npz")
