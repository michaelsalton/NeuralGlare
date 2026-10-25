"""slang/pyramid.slang reproduces the NumPy URP downsample chain (threshold 0, no clamp).

The NumPy baseline computes texel coordinates in float64; the shader in float32. A float32
coordinate near texel 1000 is only accurate to ~1e-4 texels, so a bilinear weight can be off
by that much, times whatever the neighbouring source texel holds. Errors are therefore measured
against the brightest source texel in each tap footprint, not against the output pixel itself.
"""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from scipy.ndimage import maximum_filter

from neuralglare.baselines.pyramid_bloom import (
    URPBloomParams,
    downsample_chain,
    gaussian9,
    mip_sizes,
    prefilter,
)

SLANG_DIR = Path(__file__).parents[1] / "slang"
COORD_ULPS = 4          # allowed coordinate error, in float32 ulps of the source size
FOOTPRINT = 10          # source texels: covers the widest tap (BlurH, ±8 texels) plus bilinear


@pytest.fixture(scope="module")
def module():
    device = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[SLANG_DIR])
    return spy.Module.load_from_file(device, "train_pyramid.slang")


@pytest.fixture
def image():
    """HDR noise over 7 decades (1e-3 .. 1e4) plus a sun-like 4e5 pixel: worst-case contrast."""
    rng = np.random.default_rng(0)
    img = (10.0 ** rng.uniform(-3, 4, (135, 241, 3))).astype(np.float32)
    img[70, 120] = 4e5
    return img


def to_float3_tensor(device, img):
    """(H, W, 3) float32 array -> slangpy tensor of shape (H, W) with float3 elements."""
    t = spy.Tensor.empty(device, shape=img.shape[:2], dtype="float3")
    t.copy_from_numpy(np.ascontiguousarray(img, np.float32))
    return t


def run_pass(module, fn, src, h, w):
    out = fn(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
             src=to_float3_tensor(module.device, src), _result="numpy")
    return out.reshape(h, w, 3)


def local_source_max(src, h, w):
    """For each (h, w) output texel: the brightest source value within FOOTPRINT texels."""
    m = maximum_filter(np.abs(src).max(axis=-1), size=2 * FOOTPRINT + 1)
    ys = np.minimum(((np.arange(h) + 0.5) * src.shape[0] / h).astype(int), src.shape[0] - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * src.shape[1] / w).astype(int), src.shape[1] - 1)
    return m[np.ix_(ys, xs)][..., None]


def assert_close(got, expected, src, name):
    assert got.shape == expected.shape, name
    # float32 resolves a coordinate near N texels to spacing(N): ~1.5e-5 at 241, ~1.2e-4 at 1920
    tolerance = COORD_ULPS * np.spacing(np.float32(max(src.shape[:2])))
    err = np.abs(got - expected) / local_source_max(src, *got.shape[:2])
    assert err.max() < tolerance, f"{name}: error {err.max():.2e} of local max (< {tolerance:.1e})"


def test_prefilter(module, image):
    h, w = image.shape[0] // 2, image.shape[1] // 2
    expected = prefilter(image, 0.0, 0.0, np.inf, h, w)
    assert_close(run_pass(module, module.prefilter_pass, image, h, w), expected, image, "prefilter")


def test_blur_passes(module, image):
    h, w = image.shape[0] // 2, image.shape[1] // 2
    blurred_h = gaussian9(image, h, w, "h")
    assert_close(run_pass(module, module.blur_h_pass, image, h, w), blurred_h, image, "blur_h")
    expected = gaussian9(blurred_h, h, w, "v")
    assert_close(run_pass(module, module.blur_v_pass, blurred_h, h, w), expected, blurred_h,
                 "blur_v")


def test_chain_matches_numpy(module, image):
    """The whole chain, each pass fed the NumPy result so errors don't compound."""
    p = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)
    sizes = mip_sizes(*image.shape[:2], p)
    expected = downsample_chain(image, p)
    assert len(expected) == len(sizes)
    assert_close(run_pass(module, module.prefilter_pass, image, *sizes[0]), expected[0], image,
                 "mip 0")
    for k, (h, w) in enumerate(sizes[1:], start=1):
        blurred_h = run_pass(module, module.blur_h_pass, expected[k - 1], h, w)
        got = run_pass(module, module.blur_v_pass, blurred_h, h, w)
        assert_close(got, expected[k], expected[k - 1], f"mip {k}")
