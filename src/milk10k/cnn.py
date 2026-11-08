"""Session 7: convolutional networks on the dermoscopic images.

    slide_cnn        the 306,147-parameter network traced by hand on slide 15
    scratch_cnn      a small modern CNN (conv-BN-ReLU-pool x4, global average pooling)
    resnet18         ImageNet ResNet-18 with a new 3-class head and a freezing strategy
    fit_cnn          Session 6's loop (Adam, class-weighted loss) + a clock per epoch
    predict, scores  logits and macro-F1 / accuracy of a model on uint8 image tensors

The images are shrunk to 224x224 once and kept in RAM as uint8 (5,240 x 150 KB = 0.8 GB), so
an epoch costs the network and nothing else. Augmentation (slide 24, Session 3) is applied
to the batch on the device: random flips and quarter turns (a lesion has no up), plus a
mild brightness / contrast jitter within the limits the Session 3 audit showed to be
label-safe. Frozen BatchNorm layers are put in eval mode: ``requires_grad = False`` stops
their weights, not their running statistics, and a "frozen" backbone must stay frozen.
"""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .config import ARTIFACT_DIR, IMAGE_DIR, PROCESSED_DIR
from .features import CLASSES

IMAGE_SIZE = 224
TENSOR_CACHE: Path = PROCESSED_DIR / "derm224_uint8.npz"
RUN_DIR: Path = ARTIFACT_DIR / "session07" / "runs"       # history + numbers of every run (committed)
STATE_DIR: Path = PROCESSED_DIR / "session07"              # best weights of every run (git-ignored)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

#: name -> prefixes of the ResNet-18 parameters that train (``None`` = all of them).
STRATEGIES: dict[str, tuple[str, ...] | None] = {
    "1 feature extraction (fc)": ("fc",),
    "2 partial (layer4 + fc)": ("layer4", "fc"),
    "2b partial (layer3 + layer4 + fc)": ("layer3", "layer4", "fc"),
    "3 full fine-tuning": None,
}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Images -> tensors
# --------------------------------------------------------------------------

def _load(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:     # stretched, not cropped: nothing at the frame edge is lost
        return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR))


def image_tensors(isic_ids, size: int = IMAGE_SIZE, image_dir: Path | str = IMAGE_DIR,
                  cache: Path | str | None = TENSOR_CACHE) -> torch.Tensor:
    """``(N, 3, size, size)`` uint8 tensor, one image per id, in the order given.
    Cached at ``cache`` (.npz) and rebuilt if the requested ids or size differ."""
    ids = np.asarray(list(isic_ids), dtype=str)
    cache = Path(cache) if cache else None
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if z["X"].shape[1] == size and np.array_equal(z["ids"], ids):
            return torch.from_numpy(z["X"])
    image_dir = Path(image_dir)
    with ThreadPoolExecutor(max_workers=8) as pool:
        arrays = list(pool.map(lambda i: _load(image_dir / f"{i}.jpg", size), ids))
    X = np.stack(arrays).transpose(0, 3, 1, 2).copy()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, ids=ids)
    return torch.from_numpy(X)


def labels_tensor(series: pd.Series) -> torch.Tensor:
    return torch.tensor(series.map({c: i for i, c in enumerate(CLASSES)}).to_numpy())


def normalise(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float, then ``(x/255 - mean) / std`` per channel (slide 24)."""
    x = x_uint8.float() / 255.0
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def augment(x_uint8: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Random flips, quarter turns and a mild brightness / contrast jitter, per image.
    Takes uint8, returns normalised float: labels never change."""
    n, dev = len(x_uint8), x_uint8.device
    x = x_uint8.float() / 255.0
    r = lambda *shape: torch.rand(*shape, generator=gen).to(dev)           # noqa: E731
    x = torch.where((r(n) < 0.5).view(n, 1, 1, 1), x.flip(3), x)
    x = torch.where((r(n) < 0.5).view(n, 1, 1, 1), x.flip(2), x)
    turns = (r(n) * 4).long().view(n, 1, 1, 1)
    x = torch.where(turns == 1, x.rot90(1, (2, 3)), x)
    x = torch.where(turns == 2, x.rot90(2, (2, 3)), x)
    x = torch.where(turns == 3, x.rot90(3, (2, 3)), x)
    brightness = 1 + (r(n) - 0.5) * 0.2                                    # x0.9 .. x1.1
    contrast = 1 + (r(n) - 0.5) * 0.2
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast.view(n, 1, 1, 1) + mean) * brightness.view(n, 1, 1, 1)
    return (x.clamp(0, 1) - _MEAN.to(dev)) / _STD.to(dev)


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

def slide_cnn(n_out: int = 3, size: int = IMAGE_SIZE, seed: int = 0) -> nn.Sequential:
    """Slide 15 exactly: conv 3->16, pool, conv 16->32, pool, flatten, linear. 306,147 weights
    at 224x224, 98 % of them in the last layer."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * (size // 4) ** 2, n_out))


def scratch_cnn(n_out: int = 3, widths: tuple[int, ...] = (32, 64, 128, 256),
                dropout: float = 0.3, seed: int = 0) -> nn.Sequential:
    """Conv-BN-ReLU-pool blocks, then each map is averaged to one number (slide 15's
    "modern CNNs"): the head has ``widths[-1] * n_out + n_out`` weights, not 100,352 * n_out."""
    torch.manual_seed(seed)
    layers, c_in = [], 3
    for c_out in widths:
        layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                   nn.ReLU(), nn.MaxPool2d(2)]
        c_in = c_out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_out)]
    return nn.Sequential(*layers)


def resnet18(trainable: tuple[str, ...] | None = ("layer4", "fc"), n_out: int = 3,
             pretrained: bool = True, seed: int = 0) -> nn.Module:
    """ResNet-18 with its 1000-way head replaced by a fresh ``Linear(512, n_out)``.
    Only parameters whose name starts with one of ``trainable`` keep ``requires_grad``
    (``None`` trains everything). ``pretrained=False`` is the from-scratch control."""
    torch.manual_seed(seed)
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Linear(512, n_out)
    for name, p in net.named_parameters():
        p.requires_grad = trainable is None or name.startswith(trainable)
    return net


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> list[dict]:
    """Slide 24: the new head learns at ``lr_head``, the unfrozen pretrained layers at the
    smaller ``lr_backbone``. Frozen parameters are left out of the optimiser altogether."""
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    return [{"params": g, "lr": lr} for g, lr in ((head, lr_head), (body, lr_backbone)) if g]


def set_train_mode(model: nn.Module) -> None:
    """``model.train()``, except BatchNorm layers with no trainable parameter stay in eval
    mode, so a frozen layer keeps its ImageNet running statistics."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
                p.requires_grad for p in m.parameters()):
            m.eval()


# --------------------------------------------------------------------------
# Training and scoring
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, X_uint8: torch.Tensor, batch_size: int = 128,
            dev: torch.device | None = None) -> np.ndarray:
    """Logits ``(N, n_out)`` of ``model`` on uint8 images, eval mode, no augmentation."""
    dev = dev or next(model.parameters()).device
    model.eval()
    return np.concatenate([model(normalise(X_uint8[i:i + batch_size].to(dev))).cpu().numpy()
                           for i in range(0, len(X_uint8), batch_size)])


def scores(y_true, logits) -> dict[str, float]:
    pred = np.asarray(logits).argmax(1)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro"))}


def fit_cnn(model: nn.Module, Xtr: torch.Tensor, ytr: torch.Tensor, Xva: torch.Tensor,
            yva: torch.Tensor, class_weights: torch.Tensor | None = None, epochs: int = 8,
            lr_head: float = 1e-3, lr_backbone: float = 1e-4, batch_size: int = 32,
            seed: int = 0, augmenting: bool = True, dev: torch.device | None = None,
            verbose: bool = True) -> dict:
    """Train ``model`` in place and restore the epoch with the best validation macro-F1.

    Returns ``history`` (one row per epoch: train_loss, val_loss, val_f1, seconds),
    ``best_epoch``, ``best_val_f1``, ``seconds_per_epoch`` and ``trainable``. The validation
    set picks the epoch; the test set is never seen here.
    """
    dev = dev or device()
    model.to(dev)
    gen = torch.Generator().manual_seed(seed)             # shuffling and augmentation: reproducible
    optimizer = torch.optim.Adam(param_groups(model, lr_head, lr_backbone))
    loss_fn = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(dev))
    yva_np, ytr_dev = yva.numpy(), ytr.to(dev)
    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        set_train_mode(model)
        order, total = torch.randperm(len(Xtr), generator=gen), 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = order[i:i + batch_size]
            xb = Xtr[idx].to(dev)
            xb = augment(xb, gen) if augmenting else normalise(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), ytr_dev[idx.to(dev)])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if dev.type == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - t0
        val_logits = predict(model, Xva, dev=dev)
        val_loss = float(loss_fn(torch.from_numpy(val_logits).to(dev), yva.to(dev)))
        val_f1 = scores(yva_np, val_logits)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_loss": val_loss,
                        "val_f1": val_f1, "seconds": seconds})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  loss {total / len(Xtr):.3f}  "
                  f"val loss {val_loss:.3f}  val F1 {val_f1:.3f}  {seconds:5.1f}s")
    model.load_state_dict(best_state)
    hist = pd.DataFrame(history).set_index("epoch")
    return {"history": hist, "best_epoch": best_epoch, "best_val_f1": best_f1,
            "seconds_per_epoch": float(hist["seconds"].median()),   # median: epoch 1 pays the GPU warm-up
            "trainable": count_parameters(model)["trainable"]}


def run_cached(name: str, model: nn.Module, Xtr, ytr, Xva, yva, run_dir: Path = RUN_DIR,
               state_dir: Path = STATE_DIR, **fit_kwargs) -> dict:
    """:func:`fit_cnn`, or the saved result of the very same run (same name and settings).

    A fine-tuning run is minutes of GPU time; saving it makes "Restart & Run All" cheap and
    lets the scoreboard be rebuilt without retraining. The best weights go to ``state_dir``
    (``<name>.pt``), the history and numbers to ``run_dir`` (``<name>.json``). Delete both
    to force a retrain.
    """
    config = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in fit_kwargs.items()
              if k not in ("dev", "verbose")}
    config.update(n_train=len(Xtr), trainable=count_parameters(model)["trainable"])
    json_path, state_path = Path(run_dir) / f"{name}.json", Path(state_dir) / f"{name}.pt"
    if json_path.is_file() and state_path.is_file():
        saved = json.loads(json_path.read_text())
        if saved["config"] == json.loads(json.dumps(config)):
            model.load_state_dict(torch.load(state_path, map_location="cpu"))
            saved["history"] = pd.DataFrame(saved["history"]).set_index("epoch")
            print(f"  {name}: loaded the saved run (best epoch {saved['best_epoch'] + 1}, "
                  f"val F1 {saved['best_val_f1']:.3f})")
            return saved
    result = fit_cnn(model, Xtr, ytr, Xva, yva, **fit_kwargs)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, state_path)
    out = {**result, "config": config}
    json_path.write_text(json.dumps({**out, "history": result["history"].reset_index().to_dict("records")},
                                    indent=1))
    return out
