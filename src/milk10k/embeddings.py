"""Frozen ResNet-18 embeddings: the hand-crafted features' stronger competitor.

Sessions 4-6 describe an image with 33 numbers chosen by hand (colour, histogram, texture).
This module describes the same image with the 512 numbers of the last pooling layer of an
ImageNet-pretrained ResNet-18, *without training the network*: no gradient ever reaches the
weights, so there is no new overfitting risk and the lesion-level splits stay valid. Only
the small classifier on top is fitted, and only on the training split.

It is an extension beyond the course so far (convolutional networks come later). It exists
to answer one question about the 11-class result: is the low score a ceiling of the
hand-crafted features, or of the data?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import ARTIFACT_DIR, IMAGE_DIR

EMBEDDING_CACHE: Path = ARTIFACT_DIR / "features" / "resnet18_embeddings.npz"
EMBEDDING_DIM = 512
#: Stretched (not cropped) to 224x224: a lesion at the edge of the frame is not cut off.
_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _Images(Dataset):
    def __init__(self, ids: list[str], image_dir: Path):
        self.ids, self.image_dir = ids, image_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.image_dir / f"{self.ids[i]}.jpg") as img:
            return _TRANSFORM(img.convert("RGB"))


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _backbone() -> torch.nn.Module:
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()                  # keep the 512-d pooled features
    return net.eval()


@torch.no_grad()
def resnet18_embeddings(isic_ids, cache: Path | str | None = EMBEDDING_CACHE,
                        image_dir: Path | str = IMAGE_DIR, batch_size: int = 64,
                        num_workers: int = 6) -> pd.DataFrame:
    """512-d embedding per image, indexed by ``isic_id``. Cached at ``cache`` (.npz,
    float16) and only the ids missing from the cache are computed."""
    ids = list(dict.fromkeys(isic_ids))
    cache = Path(cache) if cache else None
    have = pd.DataFrame(columns=range(EMBEDDING_DIM), dtype=np.float32)
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        have = pd.DataFrame(z["X"].astype(np.float32), index=z["ids"])
    todo = [i for i in ids if i not in have.index]
    if todo:
        device, net = _device(), _backbone()
        net.to(device)
        loader = DataLoader(_Images(todo, Path(image_dir)), batch_size=batch_size,
                            num_workers=num_workers)
        chunks = [net(batch.to(device)).cpu().numpy() for batch in loader]
        new = pd.DataFrame(np.vstack(chunks), index=todo)
        have = pd.concat([have, new]) if len(have) else new
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, X=have.to_numpy().astype(np.float16),
                                ids=have.index.to_numpy(str))
    out = have.loc[ids].astype(np.float32)
    out.columns = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    out.index.name = "isic_id"
    return out


def embedding_matrix(frame: pd.DataFrame, emb: pd.DataFrame, views=("derm", "clinical")
                     ) -> np.ndarray:
    """Design matrix of the lesions in ``frame``: the embedding of each requested view,
    side by side (``derm`` first)."""
    col = {"derm": "derm_id", "clinical": "clinical_id"}
    return np.hstack([emb.loc[frame[col[v]]].to_numpy() for v in views])
