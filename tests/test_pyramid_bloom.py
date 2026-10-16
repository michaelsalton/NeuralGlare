from dataclasses import replace

import numpy as np
import pytest

from neuralglare.baselines.pyramid_bloom import (
    BLUR_H_WEIGHTS,
    BLUR_V_WEIGHTS,
    URPBloomParams,
    bloom_basis,
    bloom_tint,
    downsample_chain,
    effective_kernel,
    gaussian9,
    level_weights,
    mip_sizes,
    prefilter,
    pyramid_bloom,
    sample_bilinear,
    shader_constants,
    soft_threshold,
    srgb_to_linear,
    texel_uv,
    upsample_chain,
)
from neuralglare.eval.metrics import log_loss
from neuralglare.io import luminance


def test_blur_weights_sum_to_one():
    # A blur must move light around, never add or remove it.
    assert BLUR_H_WEIGHTS.sum() == pytest.approx(1.0, abs=1e-6)
    assert BLUR_V_WEIGHTS.sum() == pytest.approx(1.0, abs=1e-6)


def test_sample_at_texel_centres_is_exact():
    rng = np.random.default_rng(0)                  # seeded: same "random" image every run
    img = rng.random((7, 9, 3)).astype(np.float32)
    out = sample_bilinear(img, *texel_uv(7, 9))
    np.testing.assert_allclose(out, img, atol=1e-7)


def test_sample_halfway_between_texels_averages():
    img = np.array([[[0.0], [10.0]]], np.float32)   # 1 row, 2 texels, 1 channel
    u = np.array([[0.5]])                           # the shared edge of texel 0 and texel 1
    v = np.array([[0.5]])
    assert sample_bilinear(img, u, v)[0, 0, 0] == pytest.approx(5.0)


def test_sample_clamps_to_edge():
    img = np.array([[[1.0], [2.0], [3.0]]], np.float32)
    u = np.array([[-0.5, 1.5]])                     # far outside the texture on both sides
    v = np.array([[0.5, 0.5]])
    np.testing.assert_allclose(sample_bilinear(img, u, v)[0, :, 0], [1.0, 3.0])


@pytest.mark.parametrize("h, w", [(1080, 1920), (135, 240), (67, 120)])  # even and odd sizes
def test_blur_preserves_constant(h, w):
    c = np.full((h, w, 3), 3.5, np.float32)
    oh, ow = h >> 1, w >> 1                         # URP halves sizes with integer >> 1
    down = gaussian9(c, oh, ow, "h")
    blurred = gaussian9(down, oh, ow, "v")
    np.testing.assert_allclose(down, 3.5, atol=1e-5)
    np.testing.assert_allclose(blurred, 3.5, atol=1e-5)


def test_blur_v_five_taps_equal_nine_tap_gaussian():
    # A unit impulse through BlurV must give back the 9 Gaussian weights:
    # the 5 bilinear taps really are the 9-tap kernel.
    d = np.zeros((21, 1, 1), np.float32)
    d[10] = 1.0
    col = gaussian9(d, 21, 1, "v")[6:15, 0, 0]
    np.testing.assert_allclose(col, BLUR_H_WEIGHTS, atol=1e-6)


def test_bad_axis_raises():
    with pytest.raises(ValueError):
        gaussian9(np.zeros((4, 4, 3), np.float32), 4, 4, "x")


def test_srgb_to_linear_known_values():
    assert srgb_to_linear(0.0) == pytest.approx(0.0)
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    assert srgb_to_linear(0.9) == pytest.approx(0.7874, abs=1e-4)   # URP's default threshold


def _grey(b):
    """A single grey pixel of brightness b, shape (1, 1, 3)."""
    return np.full((1, 1, 3), b, np.float64)


def test_soft_threshold_regions():
    t, k = 1.0, 0.5
    assert soft_threshold(_grey(0.4), t, k).max() == 0.0                      # below t - k: cut
    np.testing.assert_allclose(soft_threshold(_grey(3.0), t, k), 2.0, atol=1e-6)  # above: b - t
    mid = soft_threshold(_grey(1.0), t, k).max()                              # at t: toe value
    assert mid == pytest.approx(k * k / (4 * k + 1e-4), rel=1e-6)


def test_soft_threshold_is_continuous_at_knee_edges():
    t, k, e = 1.0, 0.5, 1e-6
    for b in (t - k, t + k):
        lo = soft_threshold(_grey(b - e), t, k).max()
        hi = soft_threshold(_grey(b + e), t, k).max()
        assert hi - lo == pytest.approx(0.0, abs=1e-4)


def test_soft_threshold_preserves_hue():
    c = np.array([[[8.0, 4.0, 1.0]]])
    out = soft_threshold(c, 1.0, 0.5)
    np.testing.assert_allclose(out / out.max(), c / c.max(), rtol=1e-6)


def test_soft_threshold_zero_is_identity():
    rng = np.random.default_rng(0)
    c = rng.random((4, 4, 3)) * 100 + 1e-3
    np.testing.assert_allclose(soft_threshold(c, 0.0, 0.0), c, rtol=1e-6)


@pytest.mark.parametrize("shift", [1, 2])   # Downscale: Half, Quarter
def test_prefilter_output_size(shift):
    img = np.ones((1080, 1920, 3), np.float32)
    out = prefilter(img, 0.0, 0.0, 65472.0, 1080 >> shift, 1920 >> shift)
    assert out.shape == (1080 >> shift, 1920 >> shift, 3)


def test_prefilter_applies_clamp_max():
    img = np.full((4, 4, 3), 1e6, np.float32)
    out = prefilter(img, 0.0, 0.0, 65472.0, 2, 2)
    np.testing.assert_allclose(out, 65472.0, rtol=1e-6)


def test_shader_constants_defaults():
    threshold, knee, scatter = shader_constants(URPBloomParams())
    assert threshold == pytest.approx(0.7874, abs=1e-4)
    assert knee == pytest.approx(0.5 * threshold)
    assert scatter == pytest.approx(0.68)          # 0.05 + 0.9 * 0.7


@pytest.mark.parametrize("downscale, expected", [
    ("half",    [(540, 960), (270, 480), (135, 240), (67, 120), (33, 60), (16, 30)]),
    ("quarter", [(270, 480), (135, 240), (67, 120), (33, 60), (16, 30), (8, 15)]),
])
def test_mip_sizes_1080p(downscale, expected):
    assert mip_sizes(1080, 1920, URPBloomParams(downscale=downscale)) == expected


def test_mip_sizes_tiny_image_keeps_one_mip():
    assert mip_sizes(4, 4, URPBloomParams()) == [(2, 2)]


def test_downsample_preserves_constant():
    img = np.full((1080, 1920, 3), 2.0, np.float32)
    for m in downsample_chain(img, URPBloomParams(threshold=0.0)):
        np.testing.assert_allclose(m, 2.0, atol=1e-5)


def test_downsample_conserves_energy():
    # A point source far from the edges: each mip has 1/4 the texels of the previous
    # one, so sum(mip_k) * 4**k must stay equal to sum(mip_0). Light is moved, not lost.
    img = np.zeros((1024, 1024, 3), np.float32)
    img[512, 512] = 1e4
    mips = downsample_chain(img, URPBloomParams(threshold=0.0))
    e0 = mips[0].astype(np.float64).sum()
    for k, m in enumerate(mips):
        assert m.astype(np.float64).sum() * 4**k == pytest.approx(e0, rel=1e-6)


def test_upsample_chain_is_lerp():
    mips = [np.full((4, 4, 1), 1.0, np.float32), np.full((2, 2, 1), 3.0, np.float32)]
    np.testing.assert_allclose(upsample_chain(mips, 0.25), 1.5)   # lerp(1, 3, 0.25)


def test_bloom_tint_white_and_normalized():
    np.testing.assert_allclose(bloom_tint((1.0, 1.0, 1.0)), 1.0)
    assert luminance(bloom_tint((1.0, 0.5, 0.2))) == pytest.approx(1.0)


def test_zero_intensity_is_identity():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 100).astype(np.float32)
    comp, bloom = pyramid_bloom(img, URPBloomParams(intensity=0.0))
    np.testing.assert_array_equal(comp, img)
    assert not bloom.any()


def test_constant_image_doubles():
    img = np.full((1080, 1920, 3), 2.0, np.float32)
    comp, _ = pyramid_bloom(img, URPBloomParams(threshold=0.0, intensity=1.0))
    np.testing.assert_allclose(comp, 4.0, atol=1e-5)


def test_bloom_adds_energy():
    # The pyramid conserves energy, so the bloom holds as much light as the input, and the
    # additive composite therefore doubles it. A real PSF would keep the total at 1x.
    img = np.zeros((1024, 1024, 3), np.float32)
    img[512, 512] = 1e4
    comp, bloom = pyramid_bloom(img, URPBloomParams(threshold=0.0, intensity=1.0))
    e_in = img.astype(np.float64).sum()
    assert bloom.astype(np.float64).sum() == pytest.approx(e_in, rel=1e-6)
    assert comp.astype(np.float64).sum() == pytest.approx(2 * e_in, rel=1e-6)


def test_linear_when_threshold_is_zero():
    rng = np.random.default_rng(0)
    a = (rng.random((256, 256, 3)) * 10).astype(np.float32)
    b = (rng.random((256, 256, 3)) * 10).astype(np.float32)
    p = URPBloomParams(threshold=0.0, intensity=1.0)
    f = lambda x: pyramid_bloom(x, p)[1]
    np.testing.assert_allclose(f(a + b), f(a) + f(b), rtol=1e-5, atol=1e-4)


def test_tiny_image_single_mip():
    comp, _ = pyramid_bloom(np.ones((4, 4, 3), np.float32),
                            URPBloomParams(threshold=0.0, intensity=1.0))
    np.testing.assert_allclose(comp, 2.0)


def test_effective_kernel_energy():
    k = effective_kernel(URPBloomParams(threshold=0.0, intensity=1.0), 1024, 1024, 512, 512)
    np.testing.assert_allclose(k.astype(np.float64).sum(axis=(0, 1)), 2.0, rtol=1e-5)


def test_effective_kernel_needs_zero_threshold():
    with pytest.raises(ValueError):
        effective_kernel(URPBloomParams(threshold=0.9), 64, 64, 32, 32)


@pytest.mark.parametrize("n", [1, 3, 8])
@pytest.mark.parametrize("s", [0.05, 0.5, 0.95])
def test_level_weights_sum_to_one(s, n):
    assert level_weights(s, n).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("downscale", ["half", "quarter"])
def test_basis_reproduces_upsample_chain(downscale):
    rng = np.random.default_rng(0)
    img = (rng.random((135, 240, 3)) * 10).astype(np.float32)
    p = URPBloomParams(threshold=0.0, downscale=downscale)
    basis = bloom_basis(img, p)
    _, _, s = shader_constants(p)
    for n in range(1, len(basis) + 1):
        expected = upsample_chain(downsample_chain(img, replace(p, max_iterations=n)), s)
        got = sum(w * b for w, b in zip(level_weights(s, n), basis))
        np.testing.assert_allclose(got, expected, atol=1e-5)


def test_log_loss():
    x = np.full((4, 4, 3), 2.0)
    assert log_loss(x, x, 1e-3) == 0.0
    assert log_loss(10 * x, x, 0.0) == pytest.approx(np.log(10) ** 2)
