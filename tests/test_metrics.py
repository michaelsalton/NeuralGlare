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


def test_log_abs_loss():
    ref = np.full((8, 8, 3), 2.0, np.float32)
    assert m.log_abs_loss(ref, ref, 1e-3) == 0.0
    assert abs(m.log_abs_loss(ref * 2, ref, 0.0) - np.log(2)) < 1e-6   # sign-free: ln 2
    assert abs(m.log_abs_loss(ref / 2, ref, 0.0) - np.log(2)) < 1e-6   # either way
    assert m.LOG_LOSSES["sq"] is m.log_loss and m.fit_suffix("sq") == ""


def test_flip_constant_matches_flip():
    """FLIP_K is FLIP's own constant: a constant-1 reference gets start exposure log2(K)."""
    one = np.ones((8, 8, 3), np.float32)
    _, params = m.flip_map(one, one)
    assert abs(params["startExposure"] - np.log2(m.FLIP_K)) < 1e-5


def test_flip_floor_leaves_natural_images_alone():
    ref = np.random.default_rng(0).uniform(0.1, 10, (32, 48, 3)).astype(np.float32)
    pred = ref * 1.2
    err0, p0 = m.flip_map(pred, ref)
    err1, p1 = m.flip_map(pred, ref, floor=1e-3)               # median ~5 >> floor
    assert p1 == p0 and np.array_equal(err1, err0)


def test_flip_floor_bounds_black_frames():
    """A sparse bright point on black: without a floor, a haze far below the floor scores as
    a large error; with it, the haze is invisible and the start exposure is FLIP's own."""
    ref = np.zeros((64, 64, 3), np.float32)
    ref[32, 32] = 1e4
    haze = ref + 1e-6
    err0, p0 = m.flip_map(haze, ref)
    err1, p1 = m.flip_map(haze, ref, floor=1e-3)
    assert float(err0.mean()) > 0.5 and float(err1.mean()) < 0.05
    assert abs(p1["startExposure"] - p0["startExposure"]) < 1e-5
    assert abs(p1["stopExposure"] - np.log2(m.FLIP_K / 1e-3)) < 1e-5


def test_flip_floor_matches_log_loss_visibility():
    """With flip_floor(eps), FLIP and the log loss agree on what is visible: a haze 10x
    below eps is (nearly) invisible to both, a haze at eps is a large error in both."""
    eps = 1e-3
    ref = np.zeros((64, 64, 3), np.float32)
    ref[32, 32] = 1e4
    faint, _ = m.flip_map(ref + 0.1 * eps, ref, floor=m.flip_floor(eps))
    at_eps, _ = m.flip_map(ref + eps, ref, floor=m.flip_floor(eps))
    assert float(faint.mean()) < 0.25 and float(at_eps.mean()) > 0.75


def test_flip_map_shape_and_identity():
    ref = np.random.default_rng(0).uniform(0, 10, (32, 48, 3)).astype(np.float32)
    err, params = m.flip_map(ref, ref)
    assert err.shape == (32, 48) and float(err.max()) < 1e-6
    assert params["startExposure"] <= params["stopExposure"]
