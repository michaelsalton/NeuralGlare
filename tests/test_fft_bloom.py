from pathlib import Path

import numpy as np
import pytest
import torch

from neuralglare.baselines.fft_bloom import (
    centre_psf,
    centre_psf_blend,
    conv_same_torch,
    fft_bloom,
    fit_uniform_kernel,
    scatter_term,
)
from neuralglare.reference.convolve import PSFSet, convolve_uniform, reference

PSF_PATH = Path(__file__).parent.parent / "data" / "psf" / "psf_quick.npz"


needs_psf = pytest.mark.skipif(not PSF_PATH.exists(),
                               reason="needs data/psf/psf_quick.npz (run reference/psf.py)")


@pytest.fixture(scope="module")
def ps():
    return PSFSet(PSF_PATH)


@pytest.fixture(scope="module")
def centre_delta(ps):
    d = np.zeros((ps.height, ps.width, 3), np.float32)
    d[ps.height // 2, ps.width // 2] = 1.0
    return d


@needs_psf
def test_centre_psf_is_normalised(ps):
    k = centre_psf(ps)
    assert k.shape == (ps.K, ps.K, 3)
    np.testing.assert_allclose(k.sum(axis=(0, 1)), 1.0, rtol=1e-4)


@needs_psf
def test_centred_point_matches_reference(ps, centre_delta):
    # At the frame centre the reference applies exactly the blend of the nodes around
    # it, so a uniform kernel equal to that blend must reproduce the reference there.
    out = fft_bloom(centre_delta, centre_psf_blend(ps), ps)
    ref = reference(centre_delta, ps)[0]
    np.testing.assert_allclose(out, ref, atol=1e-6)


@needs_psf
def test_precomputed_scatter_gives_same_result(ps, centre_delta):
    k = centre_psf(ps)
    s = scatter_term(centre_delta, ps)
    np.testing.assert_array_equal(fft_bloom(centre_delta, k, ps),
                                  fft_bloom(centre_delta, k, ps, scatter=s))


def test_conv_same_torch_matches_scipy():
    rng = np.random.default_rng(0)
    img = rng.random((37, 50, 3))
    for kshape in [(5, 5), (7, 3), (4, 6)]:          # odd, non-square, even sizes
        k = rng.random(kshape + (3,))
        expected = convolve_uniform(img, k)
        got = conv_same_torch(torch.from_numpy(img), torch.from_numpy(k)).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_fit_uniform_kernel_recovers_known_kernel():
    # References made with a known kernel: the fit, started from a flat kernel,
    # must find it again.
    rng = np.random.default_rng(0)
    true_k = rng.random((5, 5, 3))
    true_k /= true_k.sum(axis=(0, 1))
    imgs = [rng.random((32, 32, 3)) * 10 for _ in range(2)]
    pairs = [(im, convolve_uniform(im, true_k)) for im in imgs]
    scatters = [np.zeros_like(im) for im in imgs]
    init = np.full((5, 5, 3), 1 / 25)
    k, history = fit_uniform_kernel(pairs, scatters, 0.0, init, steps=300, lr=0.1, device="cpu")
    assert history[-1] < history[0] / 100
    np.testing.assert_allclose(k, true_k, atol=5e-3)
