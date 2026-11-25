"""The 11-class stretch target: label order, weights, metrics, and the binary collapse."""
import numpy as np
import pandas as pd
import pytest

from milk10k import multiclass as mc
from milk10k.data import MALIGNANT


def test_class_order_matches_the_label_map():
    assert mc.CLASSES == ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH",
                          "MEL", "NV", "SCCKA", "VASC"]
    assert list(mc.encode(mc.CLASSES)) == list(range(11))
    assert "MAL_OTH" not in mc.EVALUABLE and len(mc.EVALUABLE) == 10


def test_encode_rejects_unknown_codes():
    with pytest.raises(KeyError, match="XYZ"):
        mc.encode(["NV", "XYZ"])


def test_malignant_columns_are_exactly_the_malignant_classes():
    assert {mc.CLASSES[i] for i in mc.MALIGNANT_IDX} == set(MALIGNANT)
    assert list(mc.is_malignant(["NV", "MEL", "AKIEC", "VASC"])) == [False, True, True, False]


def test_power_weights_interpolate_between_none_and_balanced():
    y = ["BCC"] * 80 + ["NV"] * 20 + ["MAL_OTH"] * 1      # 101 lesions, 3 classes seen
    none, half, full = (mc.power_weights(y, p) for p in (0, 0.5, 1))
    assert all(w == 1.0 for c, w in none.items() if c in {"BCC", "NV", "MAL_OTH"})
    assert full["MAL_OTH"] > full["NV"] > full["BCC"]
    assert full["MAL_OTH"] / full["BCC"] == pytest.approx(80)            # inverse frequency
    assert half["MAL_OTH"] / half["BCC"] == pytest.approx(80 ** 0.5)
    assert full["DF"] == 0.0                                             # absent class: no weight


def test_macro_f1_ignores_mal_oth_but_all11_does_not():
    y_true = ["BCC", "NV", "MAL_OTH", "MAL_OTH"]
    y_pred = ["BCC", "NV", "NV", "BCC"]
    assert mc.macro_f1(y_true, y_pred, ["BCC", "NV"]) < 1.0          # false positives cost
    # MAL_OTH is never found: counting it as a class drags the average down.
    assert mc.macro_f1(y_true, y_pred, mc.CLASSES) < mc.macro_f1(y_true, y_pred)


def test_macro_f1_perfect_and_empty_classes():
    y = ["BCC", "NV", "MEL", "BKL"]
    assert mc.macro_f1(y, y, ["BCC", "NV", "MEL", "BKL"]) == 1.0
    assert mc.macro_f1(y, y) < 1.0              # six evaluable classes never occur -> scored 0


def test_per_class_report_support_and_recall():
    rep = mc.per_class_report(["NV", "NV", "MEL"], ["NV", "MEL", "MEL"])
    assert rep.loc["NV", "support"] == 2 and rep.loc["NV", "recall"] == 0.5
    assert rep.loc["MEL", "precision"] == 0.5 and rep.loc["MEL", "recall"] == 1.0
    assert rep.loc["DF", "support"] == 0 and len(rep) == 11


def test_top_k_accuracy():
    proba = np.zeros((2, 11))
    proba[0, mc.CLASSES.index("NV")] = 0.6
    proba[0, mc.CLASSES.index("BKL")] = 0.3
    proba[1, mc.CLASSES.index("BCC")] = 0.7
    proba[1, mc.CLASSES.index("MEL")] = 0.2
    y = ["BKL", "NV"]
    assert mc.top_k_accuracy(y, proba, k=1) == 0.0
    assert mc.top_k_accuracy(y, proba, k=2) == 0.5
    assert mc.top_k_accuracy(y, proba, k=11) == 1.0


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["BCC", "NV", "MEL"], 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, "BKL")
    labels = ["BCC", "NV", "MEL"]
    metric = lambda t, p: mc.macro_f1(t, p, labels)                    # noqa: E731
    lo, hi = mc.bootstrap_ci(y_true, y_pred, metric, n_boot=300)
    assert lo < metric(y_true, y_pred) < hi
    assert (lo, hi) == mc.bootstrap_ci(y_true, y_pred, metric, n_boot=300)
    lo2, hi2 = mc.bootstrap_ci(y_true[:60], y_pred[:60], metric, n_boot=300)
    assert hi2 - lo2 > hi - lo                                          # fewer lesions -> wider


def test_per_class_ci_is_wide_for_a_tiny_class():
    y_true = ["NV"] * 90 + ["DF"] * 5 + ["BCC"] * 5
    y_pred = ["NV"] * 90 + ["DF", "DF", "NV", "NV", "NV"] + ["BCC"] * 5
    rep = mc.per_class_f1_ci(y_true, y_pred, n_boot=300)
    assert (rep.loc["DF", "f1_hi"] - rep.loc["DF", "f1_lo"]) > \
           (rep.loc["NV", "f1_hi"] - rep.loc["NV", "f1_lo"])


def test_top_confusions_ranks_by_count():
    y_true = ["NV"] * 4 + ["MEL"] * 2
    y_pred = ["BKL", "BKL", "BKL", "NV", "NV", "MEL"]
    top = mc.top_confusions(y_true, y_pred)
    assert top.iloc[0].to_dict() == {"truth": "NV", "predicted": "BKL", "n": 3,
                                     "share_of_true_class": 0.75}
    assert len(top) == 2


def test_p_malignant_sums_the_malignant_columns():
    proba = np.zeros((2, 11))
    proba[0, mc.CLASSES.index("BCC")] = 0.4
    proba[0, mc.CLASSES.index("MEL")] = 0.1
    proba[0, mc.CLASSES.index("NV")] = 0.5
    proba[1, mc.CLASSES.index("NV")] = 1.0
    assert mc.p_malignant(proba) == pytest.approx([0.5, 0.0])


def test_binary_report_counts_a_miss_as_fifty_times_a_false_alarm():
    y_true = ["MEL", "BCC", "NV", "NV"]
    score = np.array([0.2, 0.9, 0.8, 0.1])     # one miss (MEL), one false alarm (NV)
    rep = mc.binary_report(y_true, score)
    assert rep["sensitivity"] == 0.5 and rep["specificity"] == 0.5
    assert rep["cost_per_lesion"] == pytest.approx((50 + 1) / 4)
    assert rep["auc"] == pytest.approx(0.75)


def test_threshold_for_sensitivity_reaches_the_target():
    rng = np.random.default_rng(1)
    y_true = np.array(["MEL"] * 200 + ["NV"] * 200)
    score = np.concatenate([rng.beta(5, 2, 200), rng.beta(2, 5, 200)])
    t = mc.threshold_for_sensitivity(y_true, score, target=0.95)
    assert mc.binary_report(y_true, score, t)["sensitivity"] >= 0.95
    assert t < 0.5                                   # reaching 95 % needs a lower bar than 0.5
