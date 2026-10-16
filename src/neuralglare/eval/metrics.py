"""Quality metrics shared by the baselines, training and evaluation.

All metrics take linear HDR images (never clamped or tonemapped).
"""

import numpy as np


def log_loss(pred, ref, eps):
    """Mean squared difference of log(x + eps). It weights relative errors equally at
    every brightness, so the 1e4 sources don't drown out the 1e-3 halo."""
    return float(np.mean((np.log(pred + eps) - np.log(ref + eps)) ** 2))
