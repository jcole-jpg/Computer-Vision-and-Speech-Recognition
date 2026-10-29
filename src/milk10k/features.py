"""Hand-crafted image features for the classical baseline (Session 5) and its inputs.

    color_stats       mean + std of R, G, B                       6 numbers
    color_histogram   8-bin normalised histogram per channel     24 numbers
    texture_features  edge density, roughness, Laplacian variance  3 numbers
    hsv_stats         mean + std of saturation and value           4 numbers

The first three together are Session 5's 33 features. ``feature_table`` extracts all
of them for a list of images once, in parallel, and caches the result so the notebooks
re-run in seconds. ``load_lesion_splits`` replaces the course's ``make_splits``: it
reads the committed leak-free split files (``artifacts/splits``) and returns one row
per lesion, so a lesion's two photographs can never straddle train and test.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage

from .config import ARTIFACT_DIR, IMAGE_DIR, SPLIT_DIR
from .data import build_lesion_table

#: (width, height) every image is shrunk to before measuring - Session 5's choice.
FEATURE_SIZE: tuple[int, int] = (200, 150)
CLASSES: list[str] = ["Benign", "Indeterminate", "Malignant"]
FEATURE_DIR: Path = ARTIFACT_DIR / "features"

_CHANNELS = ("R", "G", "B")
FEATURE_GROUPS: dict[str, list[str]] = {
    "color_stats": [f"{s}_{c}" for s in ("mean", "std") for c in _CHANNELS],
    "color_hist": [f"hist_{c}{i}" for c in _CHANNELS for i in range(8)],
    "texture": ["edge_density", "roughness", "laplacian_var"],
    "hsv": ["sat_mean", "sat_std", "val_mean", "val_std"],
}
FEATURE_NAMES: list[str] = [n for g in FEATURE_GROUPS.values() for n in g]


# --------------------------------------------------------------------------
# One image -> numbers
# --------------------------------------------------------------------------

def load_rgb(path: Path | str, size: tuple[int, int] = FEATURE_SIZE) -> np.ndarray:
    """Open an image, force RGB, shrink to ``size`` (w, h), return float (h, w, 3)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize(size), dtype=float)


def color_stats(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))])


def color_histogram(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    n_pixels = rgb.shape[0] * rgb.shape[1]
    return np.concatenate([np.histogram(rgb[:, :, c], bins=bins, range=(0, 256))[0] / n_pixels
                           for c in range(3)])


def texture_features(rgb: np.ndarray) -> np.ndarray:
    """Edge density (mean Sobel magnitude), roughness (mean 9x9 local variance),
    sharpness (variance of the Laplacian), all on the simple-average grayscale."""
    gray = rgb.mean(axis=2)
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_sq_mean = ndimage.uniform_filter(gray**2, size=9)
    return np.array([np.hypot(gx, gy).mean(),
                     (local_sq_mean - local_mean**2).mean(),
                     ndimage.laplace(gray).var()])


def hsv_stats(rgb: np.ndarray) -> np.ndarray:
    """Mean/std of saturation and value (hue is circular, so it is left out)."""
    hsv = rgb_to_hsv(rgb / 255.0)
    s, v = hsv[..., 1], hsv[..., 2]
    return np.array([s.mean(), s.std(), v.mean(), v.std()])


def extract_features(path: Path | str, size: tuple[int, int] = FEATURE_SIZE) -> np.ndarray:
    """All features of one image, in ``FEATURE_NAMES`` order (33 Session 5 + 4 HSV)."""
    rgb = load_rgb(path, size)
    return np.concatenate([color_stats(rgb), color_histogram(rgb),
                           texture_features(rgb), hsv_stats(rgb)])


def _extract_id(isic_id: str, image_dir: Path) -> np.ndarray:
    return extract_features(image_dir / f"{isic_id}.jpg")


# --------------------------------------------------------------------------
# Many images -> a cached table
# --------------------------------------------------------------------------

def feature_table(isic_ids, cache: Path | str | None = None,
                  workers: int | None = None, image_dir: Path | str = IMAGE_DIR) -> pd.DataFrame:
    """Features for every id, indexed by ``isic_id``. Cached at ``cache`` (.npz) and
    only the ids missing from the cache are computed."""
    ids = list(dict.fromkeys(isic_ids))
    cache = Path(cache) if cache else None
    have = pd.DataFrame(columns=FEATURE_NAMES, dtype=float)
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if list(z["names"]) == FEATURE_NAMES:
            have = pd.DataFrame(z["X"], index=z["ids"], columns=FEATURE_NAMES)
    todo = [i for i in ids if i not in have.index]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:   # PIL/numpy/scipy release the GIL
            rows = list(pool.map(partial(_extract_id, image_dir=Path(image_dir)), todo,
                                 chunksize=64))
        have = pd.concat([have, pd.DataFrame(np.vstack(rows), index=todo,
                                             columns=FEATURE_NAMES)])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, X=have.to_numpy(), ids=have.index.to_numpy(str),
                                names=np.array(FEATURE_NAMES))
    out = have.loc[ids]
    out.index.name = "isic_id"
    return out


# --------------------------------------------------------------------------
# Lesion-level splits and the metadata feature set
# --------------------------------------------------------------------------

def load_lesion_splits(split_dir: Path | str = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """``{"train"|"val"|"test": lesions}`` - one row per lesion with
    ``lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site`` from
    :func:`build_lesion_table`, assigned by the committed split files."""
    split_dir = Path(split_dir)
    assign = pd.concat([pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "val", "test")])
    assign = assign.drop_duplicates("lesion_id").set_index("lesion_id")["split"]
    lesions = build_lesion_table()
    lesions["split"] = lesions["lesion_id"].map(assign)
    missing = lesions["split"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} lesions have no split assignment in {split_dir}")
    return {s: lesions[lesions["split"] == s].reset_index(drop=True)
            for s in ("train", "val", "test")}


def metadata_features(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Age, sex and body site as numbers. Medians and categories come from the
    TRAIN frame only, so nothing about val/test leaks into the encoding."""
    train = frames["train"]
    age_median = train["age"].median()
    sexes = sorted(train["sex"].dropna().unique())
    sites = sorted(train["site"].dropna().unique())

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"age": df["age"].fillna(age_median),
                            "age_missing": df["age"].isna().astype(float),
                            "site_missing": df["site"].isna().astype(float)})
        for s in sexes:
            out[f"sex_{s}"] = (df["sex"] == s).astype(float)
        for s in sites:
            out[f"site_{s}"] = (df["site"] == s).astype(float)
        return out.reset_index(drop=True)

    return {name: encode(df) for name, df in frames.items()}


#: name -> (groups taken from the derm image, groups from the clinical image, add metadata)
FEATURE_SETS: dict[str, tuple[list[str], list[str], bool]] = {
    "colour stats (6)":              (["color_stats"], [], False),
    "colour stats + histogram (30)": (["color_stats", "color_hist"], [], False),
    "Session 5 set (33)":            (["color_stats", "color_hist", "texture"], [], False),
    "33 + HSV (37)":                 (["color_stats", "color_hist", "texture", "hsv"], [], False),
    "33 + metadata":                 (["color_stats", "color_hist", "texture"], [], True),
    "derm + clinical (66)":          (["color_stats", "color_hist", "texture"],
                                      ["color_stats", "color_hist", "texture"], False),
    "metadata only":                 ([], [], True),
}


def matrix_columns(name: str, meta: pd.DataFrame | None = None) -> list[str]:
    """Column names of the design matrix of feature set ``name`` (clinical-image
    columns carry a ``clin_`` prefix, metadata columns come last)."""
    derm_groups, clin_groups, add_meta = FEATURE_SETS[name]
    cols = [c for g in derm_groups for c in FEATURE_GROUPS[g]]
    cols += ["clin_" + c for g in clin_groups for c in FEATURE_GROUPS[g]]
    if add_meta:
        cols += list(meta.columns)
    return cols


def build_matrix(name: str, frame: pd.DataFrame, feats: pd.DataFrame,
                 meta: pd.DataFrame | None = None) -> np.ndarray:
    """Design matrix of feature set ``name`` for the lesions in ``frame``."""
    derm_groups, clin_groups, add_meta = FEATURE_SETS[name]
    parts = []
    if derm_groups:
        cols = [c for g in derm_groups for c in FEATURE_GROUPS[g]]
        parts.append(feats.loc[frame["derm_id"], cols].to_numpy())
    if clin_groups:
        cols = [c for g in clin_groups for c in FEATURE_GROUPS[g]]
        parts.append(feats.loc[frame["clinical_id"], cols].to_numpy())
    if add_meta:
        parts.append(meta.to_numpy())
    return np.hstack(parts)
