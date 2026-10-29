"""Session 5 features: shape, determinism, the numbers on images with a known answer."""
import numpy as np
import pytest
from PIL import Image

from milk10k import features as F


def _constant_rgb(value=100.0, size=(40, 30)):
    return np.full((size[1], size[0], 3), value)


def test_feature_names_match_groups():
    assert len(F.FEATURE_NAMES) == 6 + 24 + 3 + 4
    assert len(set(F.FEATURE_NAMES)) == len(F.FEATURE_NAMES)


def test_constant_image_has_known_stats():
    rgb = _constant_rgb(100.0)
    assert F.color_stats(rgb).tolist() == [100, 100, 100, 0, 0, 0]
    hist = F.color_histogram(rgb)
    assert hist.shape == (24,) and hist.sum() == pytest.approx(3.0)
    assert hist[100 // 32] == 1.0                       # all pixels in one bin, per channel
    assert F.texture_features(rgb) == pytest.approx([0, 0, 0])  # flat = no edges, no texture


def test_texture_rises_with_noise():
    rng = np.random.default_rng(0)
    flat = _constant_rgb(128.0)
    noisy = np.clip(flat + rng.normal(0, 20, flat.shape), 0, 255)
    assert (F.texture_features(noisy) > F.texture_features(flat)).all()


def test_extract_features_is_deterministic_and_33_plus_hsv(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "a.jpg"
    Image.fromarray(rng.integers(0, 256, (150, 200, 3), dtype=np.uint8)).save(path)
    a, b = F.extract_features(path), F.extract_features(path)
    assert a.shape == (len(F.FEATURE_NAMES),)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_feature_table_caches_and_reuses(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(2)
    for i in range(3):
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(img_dir / f"ISIC_{i}.jpg")
    cache = tmp_path / "f.npz"
    first = F.feature_table(["ISIC_0", "ISIC_1"], cache=cache, workers=1, image_dir=img_dir)
    again = F.feature_table(["ISIC_1", "ISIC_0", "ISIC_2"], cache=cache, workers=1,
                           image_dir=img_dir)
    assert list(again.index) == ["ISIC_1", "ISIC_0", "ISIC_2"]
    np.testing.assert_allclose(again.loc[["ISIC_0", "ISIC_1"]].to_numpy(),
                               first.loc[["ISIC_0", "ISIC_1"]].to_numpy())


def test_real_splits_are_lesion_disjoint():
    try:
        frames = F.load_lesion_splits()
    except FileNotFoundError:
        pytest.skip("MILK10k metadata not available locally")
    ids = {k: set(v["lesion_id"]) for k, v in frames.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) \
        and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in frames.values()) == 5240


def test_metadata_encoding_uses_train_categories_only():
    import pandas as pd
    train = pd.DataFrame({"age": [30.0, 50.0, None], "sex": ["male", "female", "male"],
                          "site": ["head/neck", None, "head/neck"]})
    val = pd.DataFrame({"age": [None], "sex": ["male"], "site": ["unseen site"]})
    enc = F.metadata_features({"train": train, "val": val})
    assert list(enc["train"].columns) == list(enc["val"].columns)
    assert enc["val"].loc[0, "age"] == 40.0             # train median, not val's
    assert enc["val"].loc[0, "age_missing"] == 1.0
    assert enc["val"].filter(like="site_head").iloc[0, 0] == 0.0   # unseen -> all zeros
