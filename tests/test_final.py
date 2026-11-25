"""Milestone 5: the leaderboard format check, the inference transform and predict()."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from milk10k import cnn, final
from milk10k.multiclass import CLASSES


def _submission(n=5):
    df = pd.DataFrame(np.eye(11)[np.arange(n) % 11], columns=CLASSES)
    df.insert(0, "lesion", [f"IL_{i:07d}" for i in range(n)])
    return df


def test_columns_are_the_leaderboard_order():
    assert final.LEADERBOARD_COLUMNS == ["lesion", "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                                         "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def test_check_submission_accepts_a_valid_file():
    final.check_submission(_submission(), n_rows=5, one_class=True)
    df = _submission()
    df.loc[0, CLASSES] = 0.0                      # thresholds may leave a lesion with no class
    final.check_submission(df, n_rows=5)


@pytest.mark.parametrize("breakage", ["column order", "duplicate lesion", "out of range", "no class"])
def test_check_submission_rejects(breakage):
    df = _submission()
    if breakage == "column order":
        df = df[["lesion"] + CLASSES[::-1]]
    elif breakage == "duplicate lesion":
        df.loc[1, "lesion"] = df.loc[0, "lesion"]
    elif breakage == "out of range":
        df.loc[0, "BCC"] = 1.5
    else:
        df.loc[0, CLASSES] = 0.09            # a softmax whose top class is < 0.5: no class at all
    with pytest.raises(AssertionError):
        final.check_submission(df, n_rows=5, one_class=True)


def _image(path, size=(600, 450), seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_eval_tensor_is_deterministic_and_224(tmp_path):
    _image(tmp_path / "ISIC_0000001.jpg")
    a, b = final.eval_tensor(tmp_path / "ISIC_0000001.jpg"), final.eval_tensor(tmp_path / "ISIC_0000001.jpg")
    assert a.shape == (3, 224, 224) and a.dtype == torch.uint8
    assert torch.equal(a, b)


def test_image_index_reads_both_layouts(tmp_path):
    (tmp_path / "IL_1").mkdir()
    _image(tmp_path / "IL_1" / "ISIC_0000002.jpg")
    _image(tmp_path / "ISIC_0000003.jpg")
    assert set(final.image_index(tmp_path)) == {"ISIC_0000002", "ISIC_0000003"}


def test_predict_returns_11_probabilities(tmp_path):
    _image(tmp_path / "x.jpg")
    net = cnn.resnet18(trainable=None, n_out=11, pretrained=False).eval()

    class Stub:                                   # FinalModel's interface, without downloading ViT
        def proba(self, X):
            return final.softmax(cnn.predict(net, X))
    model = Stub()
    out = final.predict(tmp_path / "x.jpg", model=model)
    assert set(out["probabilities"]) == set(CLASSES)
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["label"] in CLASSES and 0 <= out["p_malignant"] <= 1
    assert final.predict(tmp_path / "x.jpg", model=model) == out     # eval mode: same answer twice


def test_tune_thresholds_picks_a_lower_cut_for_a_shy_class(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        label = CLASSES[i % 11]
        p = rng.uniform(0, 0.05, 11)
        p[CLASSES.index(label)] = 0.3 if label == "DF" else 0.8   # DF is right but never reaches 0.5
        rows.append({"label": label, **dict(zip(CLASSES, p))})
    pd.DataFrame(rows).to_csv(tmp_path / "val.csv", index=False)
    thr = final.tune_thresholds(tmp_path / "val.csv", tmp_path / "thr.csv")
    assert thr["DF"] <= 0.3 and thr["BCC"] == 0.5
