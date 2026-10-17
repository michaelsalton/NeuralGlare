import numpy as np

from neuralglare.eval import metrics as m


def test_window_is_centred_and_clipped():
    img = np.arange(100 * 200).reshape(100, 200)
    w = m.window(img, 50, 40, 10)
    assert w.shape == (10, 10) and w[5, 5] == img[40, 50]
    assert m.window(img, 2, 3, 10).shape == (8, 7)             # clipped at the top-left


def test_log_psnr_and_energy():
    ref = np.full((8, 8, 3), 2.0, np.float32)
    assert abs(m.energy_error(ref * 1.1, ref) - 0.1) < 1e-6
    peak = m.log_peak(ref, 1e-3)
    # identical images: infinite PSNR is capped by the 1e-30 floor, so it is very large
    assert m.log_psnr(ref, ref, 1e-3, peak) > 200
    # doubling every value: log error = ln 2 everywhere
    expected = 10 * np.log10(peak ** 2 / np.log((4 + 1e-3) / (2 + 1e-3)) ** 2)
    assert abs(m.log_psnr(ref * 2, ref, 1e-3, peak) - expected) < 1e-6


def test_flip_map_shape_and_identity():
    ref = np.random.default_rng(0).uniform(0, 10, (32, 48, 3)).astype(np.float32)
    err, params = m.flip_map(ref, ref)
    assert err.shape == (32, 48) and float(err.max()) < 1e-6
    assert params["startExposure"] <= params["stopExposure"]
