"""Quality metrics shared by the baselines, training and evaluation.

All metrics take linear HDR images (never clamped or tonemapped).
"""

import numpy as np

from neuralglare.io import luminance


def log_loss(pred, ref, eps):
    """Mean squared difference of log(x + eps). It weights relative errors equally at
    every brightness, so the 1e4 sources don't drown out the 1e-3 halo."""
    return float(np.mean((np.log(pred + eps) - np.log(ref + eps)) ** 2))


def log_abs_loss(pred, ref, eps):
    """Mean absolute difference of log(x + eps). Grows linearly with a small error, as FLIP
    does, where log_loss shrinks it quadratically (decision log 2026-11-03)."""
    return float(np.mean(np.abs(np.log(pred + eps) - np.log(ref + eps))))


LOG_LOSSES = {"sq": log_loss, "abs": log_abs_loss}


def fit_suffix(loss):
    """File-name suffix of a baseline fitted with this loss: none for the squared log loss
    (the original fits), "_abs" for the absolute one."""
    return "" if loss == "sq" else f"_{loss}"


def log_psnr(pred, ref, eps, peak):
    """PSNR of the log(x + eps) images. `peak` is the log range the PSNR is relative to;
    pass the same value for every window of a frame, e.g. log_peak(full reference, eps)."""
    return float(10.0 * np.log10(peak ** 2 / max(log_loss(pred, ref, eps), 1e-30)))


def log_peak(ref, eps):
    """Log range of a reference frame: log(max + eps) - log(eps)."""
    return float(np.log(ref.max() + eps) - np.log(eps))


def energy_error(pred, ref):
    """Relative energy difference, sum(pred) / sum(ref) - 1, in float64."""
    return float(pred.astype(np.float64).sum() / ref.astype(np.float64).sum() - 1.0)


def window(img, x, y, size):
    """size x size crop of img centred on pixel (x, y), clipped at the frame edge."""
    h = size // 2
    cx, cy = int(np.rint(x)), int(np.rint(y))
    return img[max(cy - h, 0):cy - h + size, max(cx - h, 0):cx - h + size]


FLIP_K = 2.1188732  # FLIP 1.7 (ACES): exposure c shows luminance K * 2^-c near white; a
                    # constant-1 image gets startExposure log2(K) (test_metrics.py)
FLIP_FLOOR_PER_EPS = 10.0  # calibrated against the log loss (test_metrics.py): with
                           # floor = 10 eps, a haze far below eps scores ~0 in FLIP, as in
                           # the loss; a haze at eps scores high in both. floor = eps scored
                           # a 0.1 eps haze 0.87 while the loss scored it ~0.


def flip_floor(eps):
    """FLIP exposure floor matching a log loss with this eps (see flip_exposures)."""
    return FLIP_FLOOR_PER_EPS * eps


def flip_exposures(ref, floor):
    """FLIP's exposure range for ref, with the stop exposure capped by `floor`.

    FLIP uses start = log2(K / max luminance), stop = log2(K / median luminance). A mostly
    black reference (the point grid: median 0) sends stop to +24 stops, where a 1e-6 haze
    looks like a full error. Here stop = log2(K / max(median, floor)): identical to FLIP's
    own range when median >= floor (returns None: let FLIP choose), bounded otherwise."""
    lum = luminance(np.maximum(ref, 0))
    median, peak = float(np.median(lum)), float(lum.max())
    if median >= floor:
        return None
    start = float(np.log2(FLIP_K / peak))
    stop = max(float(np.log2(FLIP_K / floor)), start)
    return {"startExposure": start, "stopExposure": stop,
            "numExposures": max(2, int(np.ceil(stop - start)))}


def flip_map(pred, ref, ppd=67.0, floor=None):
    """HDR-FLIP error map (H, W) of pred against ref, and the parameters FLIP used.
    FLIP picks its exposure range from the reference, so maps of different predictions
    against one reference are comparable. `floor` (luminance, e.g. the log-loss eps) caps
    the brightest exposure; see flip_exposures. FLIP needs non-negative input: FFT
    round-off can leave tiny negatives, which are zeroed for this metric only."""
    import flip_evaluator as flip
    parameters = {"ppd": ppd}
    if floor is not None:
        parameters |= flip_exposures(ref, floor) or {}
    err, _, params = flip.evaluate(np.ascontiguousarray(np.maximum(ref, 0), np.float32),
                                   np.ascontiguousarray(np.maximum(pred, 0), np.float32),
                                   "HDR", applyMagma=False, computeMeanError=False,
                                   parameters=parameters)
    return np.asarray(err, np.float32).reshape(ref.shape[:2]), params
