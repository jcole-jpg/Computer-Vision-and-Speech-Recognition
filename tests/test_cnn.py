"""Session 7: the networks and the training helpers, on random tensors (no images, no download)."""

import pytest
import torch
from torch import nn

from milk10k import cnn


def test_slide_15_network_has_the_hand_traced_shapes_and_parameters():
    net = cnn.slide_cnn()
    x = torch.zeros(1, 3, 224, 224)
    shapes = []
    for layer in net:
        x = layer(x)
        shapes.append(tuple(x.shape[1:]))
    assert shapes[0] == (16, 224, 224) and shapes[2] == (16, 112, 112)
    assert shapes[3] == (32, 112, 112) and shapes[5] == (32, 56, 56)
    assert shapes[6] == (100_352,) and shapes[7] == (3,)
    assert cnn.count_parameters(net)["total"] == 306_147          # 448 + 4,640 + 301,059


def test_conv_layer_is_21_500_times_smaller_than_the_dense_layer():
    conv = sum(p.numel() for p in nn.Conv2d(3, 32, 3, padding=1).parameters())
    dense = sum(p.numel() for p in nn.Linear(224 * 224 * 3, 128).parameters())
    assert (conv, dense) == (896, 19_267_712)
    assert round(dense / conv) == 21_504


@pytest.mark.parametrize("strategy, trainable", [
    ("1 feature extraction (fc)", 1_539),
    ("2 partial (layer4 + fc)", 8_395_267),
    ("2b partial (layer3 + layer4 + fc)", 10_494_979),
    ("3 full fine-tuning", 11_178_051),
])
def test_resnet18_strategies_unfreeze_exactly_the_slide_23_weights(strategy, trainable):
    net = cnn.resnet18(cnn.STRATEGIES[strategy], pretrained=False)
    counts = cnn.count_parameters(net)
    assert counts["trainable"] == trainable and counts["total"] == 11_178_051


def test_param_groups_give_the_head_a_larger_lr_and_skip_frozen_weights():
    net = cnn.resnet18(("layer4", "fc"), pretrained=False)
    head, body = cnn.param_groups(net, 1e-3, 1e-4)
    assert head["lr"] == 1e-3 and body["lr"] == 1e-4
    assert sum(p.numel() for p in head["params"]) == 1_539
    assert sum(p.numel() for p in body["params"]) == 8_393_728          # layer4 only
    assert len(cnn.param_groups(cnn.resnet18(("fc",), pretrained=False), 1e-3, 1e-4)) == 1


def test_frozen_batchnorm_stays_in_eval_mode_while_training():
    net = cnn.resnet18(("layer4", "fc"), pretrained=False)
    cnn.set_train_mode(net)
    assert net.training and not net.bn1.training and not net.layer1[0].bn1.training
    assert net.layer4[0].bn1.training
    before = net.bn1.running_mean.clone()
    net(torch.randn(4, 3, 64, 64))
    assert torch.equal(net.bn1.running_mean, before)                   # frozen layer: statistics untouched


def test_augment_keeps_the_shape_and_is_reproducible_with_a_seed():
    x = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    a = cnn.augment(x, torch.Generator().manual_seed(1))
    b = cnn.augment(x, torch.Generator().manual_seed(1))
    c = cnn.augment(x, torch.Generator().manual_seed(2))
    assert a.shape == (8, 3, 32, 32) and a.dtype == torch.float32
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_augment_only_rearranges_pixels_up_to_the_light_jitter():
    # a flat image has no geometry to change: only brightness / contrast can differ
    x = torch.full((4, 3, 16, 16), 128, dtype=torch.uint8)
    out = cnn.augment(x, torch.Generator().manual_seed(0)) * cnn._STD + cnn._MEAN
    assert torch.allclose(out, torch.full_like(out, 128 / 255), atol=0.06)


def test_normalise_matches_the_slide_24_worked_example():
    x = torch.zeros(1, 3, 1, 1, dtype=torch.uint8)
    x[0, 0] = 200                                                       # red = 200
    assert cnn.normalise(x)[0, 0, 0, 0].item() == pytest.approx((200 / 255 - 0.485) / 0.229)
    assert round((200 / 255 - 0.485) / 0.229, 2) == 1.31


def test_fit_cnn_learns_a_separable_toy_problem_and_restores_the_best_epoch(tmp_path):
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 3, (96,), generator=g)
    X = torch.full((96, 3, 16, 16), 60, dtype=torch.uint8)
    for c in range(3):
        X[y == c, c] = 200                                              # class c = channel c bright
    net = cnn.scratch_cnn(widths=(8, 16), seed=0)
    res = cnn.fit_cnn(net, X[:64], y[:64], X[64:], y[64:], epochs=6, lr_head=1e-2, lr_backbone=1e-2,
                      augmenting=False, dev=torch.device("cpu"), verbose=False)
    assert len(res["history"]) == 6 and res["best_val_f1"] > 0.9
    assert cnn.scores(y[64:].numpy(), cnn.predict(net, X[64:], dev=torch.device("cpu")))["macro_f1"] \
        == pytest.approx(res["best_val_f1"])                            # best weights were restored


def test_run_cached_reloads_the_same_run_and_retrains_when_settings_change(tmp_path):
    X = torch.randint(0, 256, (40, 3, 16, 16), dtype=torch.uint8)
    y = torch.randint(0, 3, (40,))
    kw = dict(run_dir=tmp_path / "runs", state_dir=tmp_path / "state", epochs=2, lr_backbone=1e-3,
              augmenting=False, dev=torch.device("cpu"), verbose=False)
    first = cnn.run_cached("toy", cnn.scratch_cnn(widths=(4,)), X[:30], y[:30], X[30:], y[30:], **kw)
    again_model = cnn.scratch_cnn(widths=(4,), seed=5)
    again = cnn.run_cached("toy", again_model, X[:30], y[:30], X[30:], y[30:], **kw)
    assert again["best_val_f1"] == first["best_val_f1"]
    saved = torch.load(tmp_path / "state" / "toy.pt")
    assert all(torch.equal(saved[k], v) for k, v in again_model.state_dict().items())
    kw["epochs"] = 3                                                    # different settings -> new run
    third = cnn.run_cached("toy", cnn.scratch_cnn(widths=(4,)), X[:30], y[:30], X[30:], y[30:], **kw)
    assert len(third["history"]) == 3
