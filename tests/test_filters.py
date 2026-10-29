"""Session 4 filters: kernels agree with scipy, Canny behaves like a Canny detector."""
import numpy as np
import pytest
from scipy import ndimage

from milk10k import filters as flt


def _step(h=40, w=40, col=20):
    img = np.zeros((h, w))
    img[:, col:] = 200.0
    return img


def test_sobel_kernels_match_scipy():
    rng = np.random.default_rng(0)
    img = rng.random((30, 30)) * 255
    np.testing.assert_allclose(flt.apply_kernel(img, flt.SOBEL_X), ndimage.sobel(img, axis=1))
    np.testing.assert_allclose(flt.apply_kernel(img, flt.SOBEL_Y), ndimage.sobel(img, axis=0))


@pytest.mark.parametrize("sigma", [0.7, 1.0, 2.0, 4.0])
def test_gaussian_kernel_matches_scipy_and_sums_to_one(sigma):
    k = flt.gaussian_kernel(sigma)
    assert k.sum() == pytest.approx(1.0)
    impulse = np.zeros(2 * len(k) + 1)
    impulse[len(impulse) // 2] = 1.0
    np.testing.assert_allclose(ndimage.gaussian_filter1d(impulse, sigma, truncate=4.0)
                               [len(impulse) // 2 - len(k) // 2: len(impulse) // 2 + len(k) // 2 + 1],
                               k, atol=1e-12)


def test_wider_sigma_means_wider_kernel_and_less_edge_energy():
    assert len(flt.gaussian_kernel(4)) > len(flt.gaussian_kernel(1))
    rng = np.random.default_rng(1)
    img = rng.random((60, 60)) * 255
    dens = [flt.edge_density(ndimage.gaussian_filter(img, s)) for s in (0.5, 1, 2, 4)]
    assert dens == sorted(dens, reverse=True)


def test_canny_finds_a_vertical_step_and_nothing_else():
    edges = flt.canny(_step(), sigma=1.0)
    assert edges.dtype == bool
    cols = np.where(edges.any(axis=0))[0]
    assert cols.min() >= 18 and cols.max() <= 21        # only around the step at column 20
    assert edges[5:-5].any(axis=1).all()                # present on every interior row


def test_canny_edge_is_thinner_than_sobel():
    img = ndimage.gaussian_filter(_step(), 1.5)
    assert flt.canny(img, sigma=0.5).sum() < flt.sobel_edges(img, threshold=0.2).sum()


def test_canny_flat_image_has_no_edges():
    assert not flt.canny(np.full((30, 30), 90.0)).any()


def test_hysteresis_keeps_weak_pixels_only_when_connected_to_strong():
    thin = np.zeros((5, 12))
    thin[2, 0] = 10.0                       # strong
    thin[2, 1:5] = 4.0                      # weak chain touching the strong pixel
    thin[0, 9] = 4.0                        # weak but isolated
    out = flt.hysteresis(thin, low=3.0, high=8.0)
    assert out[2, :5].all() and not out[0, 9]


def test_canny_diagonal_edge_is_found():
    yy, xx = np.mgrid[:50, :50]
    img = np.where(xx + yy > 50, 200.0, 0.0)
    edges = flt.canny(img, sigma=1.0)
    ys, xs = np.nonzero(edges[5:-5, 5:-5])
    assert len(ys) > 20 and np.abs((xs + 5) + (ys + 5) - 50).max() <= 2
