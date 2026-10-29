from pathlib import Path

import numpy as np
import pytest

from neuralglare.baselines.unreal_fft_bloom import (
    quantize_r11g11b10,
    unreal_bloom_as_run,
    unreal_fft_bloom,
    unreal_kernel,
)


def rgb(*values):
    return np.array(values, np.float32).reshape(1, -1, 3)


def test_quantize_matches_unreal_capture():
    # data/captures/unreal/unreal_beforebloom_pointgrid.exr stores the 1e4 points as 9984
    np.testing.assert_array_equal(quantize_r11g11b10(rgb(1e4, 1e4, 1e4)), rgb(9984, 9984, 9984))


def test_quantize_mantissa_bits():
    # 6-bit mantissa (R, G): step 2**-6 above 1; 5-bit (B): step 2**-5
    q = quantize_r11g11b10(rgb(1 + 2 ** -6, 1 + 2 ** -6, 1 + 2 ** -6))
    np.testing.assert_array_equal(q, rgb(1 + 2 ** -6, 1 + 2 ** -6, 1.0))


def test_quantize_truncates():
    # just below the next step stays on the lower one (round-to-nearest would go up)
    x = 1 + 0.99 * 2 ** -6
    np.testing.assert_array_equal(quantize_r11g11b10(rgb(x, x, 0.99 * 2 ** -14)),
                                  rgb(1.0, 1.0, 31 * 2 ** -19))


def test_quantize_unsigned_and_clamped():
    q = quantize_r11g11b10(rgb(-1.0, 1e6, 1e6))
    np.testing.assert_array_equal(q, rgb(0.0, 65024.0, 64512.0))


def test_quantize_denormals():
    # below 2**-14 the step is 2**-14 / 2**m
    q = quantize_r11g11b10(rgb(2 ** -20, 2 ** -20, 2 ** -20))
    np.testing.assert_array_equal(q, rgb(2 ** -20, 2 ** -20, 0.0))


def test_quantize_is_idempotent():
    # 2**-20 to 2**15: denormals up to just below the clamp
    x = (2.0 ** np.random.default_rng(0).uniform(-20, 15, (64, 64, 3))).astype(np.float32)
    q = quantize_r11g11b10(x)
    np.testing.assert_array_equal(quantize_r11g11b10(q), q)
    # truncation: never above the input, at most one step below, 2**-m relative (m = 5 for blue)
    assert np.all(q <= x)
    assert np.all(x - q <= np.maximum(x * 2.0 ** -5, 2.0 ** -19))


def test_delta_kernel_is_unchanged():
    # all energy in the centre texel: everything becomes pass-through, nothing is lost
    K = np.zeros((9, 9, 3), np.float32)
    K[4, 4] = 1.0
    K_eff, info = unreal_kernel(K)
    np.testing.assert_allclose(K_eff, K)
    assert info["tint"] == [1.0, 1.0, 1.0]


def test_kernel_energy_is_the_tint():
    rng = np.random.default_rng(1)
    K = rng.random((15, 15, 3)) * 0.01
    K[7, 7] = (5.0, 3.0, 1.0)
    K /= K.sum((0, 1))
    K_eff, info = unreal_kernel(K)
    assert info["centre"] == (7, 7)
    np.testing.assert_allclose(K_eff.sum((0, 1)), info["tint"], rtol=1e-6)
    assert max(info["tint"]) == 1.0


def test_unreal_fft_bloom_of_a_point_is_the_kernel():
    K = np.random.default_rng(2).random((7, 7, 3)).astype(np.float32)
    K[3, 3] = 1.0
    K /= K.sum((0, 1))
    img = np.zeros((21, 21, 3), np.float32)
    img[10, 10] = 1.0                                    # exactly representable
    out = unreal_fft_bloom(img, K)
    np.testing.assert_allclose(out[7:14, 7:14], unreal_kernel(K)[0], atol=1e-6)


def test_apply_constants_match_the_capture():
    # data/captures/unreal/unreal_fft_constants_pointgrid.json, centre kernel of the default run
    path = Path(__file__).parent.parent / "outputs/01_gap_study/default/fits/fft_kernels.npz"
    if not path.exists():
        pytest.skip("needs the default run's fft_kernels.npz")
    _, info = unreal_kernel(np.load(path)["centre"])
    np.testing.assert_allclose(info["scene_color_apply"], [0.8505343, 0.6121117, 0.4075874],
                               rtol=1e-6)
    np.testing.assert_allclose(info["fft_multiply"], [0.1494656, 0.1712394, 0.1557920],
                               rtol=1e-6)


def test_bloom_as_run_channel_gains():
    # a point through the as-run bloom: each channel's energy is its multiply constant,
    # times its own kernel sum over the sum it is (wrongly) normalised by
    K = np.zeros((9, 9, 3))
    K[4, 4] = (0.6, 0.5, 0.4)
    K[4, 2] = K[2, 4] = (0.1, 0.125, 0.15)
    K[4, 6] = K[6, 4] = (0.1, 0.125, 0.15)
    img = np.zeros((31, 31, 3), np.float32)
    img[15, 15] = 1024.0
    bloom = unreal_bloom_as_run(img, K)
    _, info = unreal_kernel(K)
    s, mult = np.array(info["clamped_kernel_sum"]), np.array(info["fft_multiply"])
    expected = 1024.0 * mult[[0, 1, 0]] * s / s[[0, 2, 1]]
    np.testing.assert_allclose(bloom.sum((0, 1)), expected, rtol=2 ** -5)
