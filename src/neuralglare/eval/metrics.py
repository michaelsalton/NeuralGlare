"""Quality metrics shared by the baselines, training and evaluation.

All metrics take linear HDR images (never clamped or tonemapped).
"""

import numpy as np


def log_loss(pred, ref, eps):
    """Mean squared difference of log(x + eps). It weights relative errors equally at
    every brightness, so the 1e4 sources don't drown out the 1e-3 halo."""
    return float(np.mean((np.log(pred + eps) - np.log(ref + eps)) ** 2))


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


def flip_map(pred, ref, ppd=67.0):
    """HDR-FLIP error map (H, W) of pred against ref, and the parameters FLIP used.
    FLIP picks its exposure range from the reference, so maps of different predictions
    against one reference are comparable. FLIP needs non-negative input: FFT round-off
    can leave tiny negatives, which are zeroed for this metric only."""
    import flip_evaluator as flip
    err, _, params = flip.evaluate(np.ascontiguousarray(np.maximum(ref, 0), np.float32),
                                   np.ascontiguousarray(np.maximum(pred, 0), np.float32),
                                   "HDR", applyMagma=False, computeMeanError=False,
                                   parameters={"ppd": ppd})
    return np.asarray(err, np.float32).reshape(ref.shape[:2]), params
