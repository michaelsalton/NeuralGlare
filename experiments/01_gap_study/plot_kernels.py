"""Radial profile of the centre PSF (reference) vs URP pyramid bloom's effective kernel.

Writes outputs/01_gap_study/<run>/report/kernel_profiles.png.

Usage (from the repo root):
    uv run python experiments/01_gap_study/plot_kernels.py --run quick
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from neuralglare.baselines.pyramid_bloom import URPBloomParams, effective_kernel
from neuralglare.paths import RUNS, psf_path, run_dir
from neuralglare.reference.convolve import PSFSet, reference


def radial_profile(img, y, x):
    """Mean green-channel value in 1 px rings around (y, x)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - y, xx - x).astype(np.int64).ravel()
    return np.bincount(r, img[..., 1].ravel()) / np.maximum(np.bincount(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="quick",
                    help="config: picks the PSF and output folder")
    args = ap.parse_args()

    ps = PSFSet(psf_path(args.run))
    h, w = ps.height, ps.width
    y, x = h // 2, w // 2
    delta = np.zeros((h, w, 3), np.float32)
    delta[y, x] = 1.0
    ref, diff, scat, _ = reference(delta, ps)
    eta = ps.scatter.energy_fraction

    fig, ax = plt.subplots(figsize=(7, 5))
    r = np.arange(1, int(np.hypot(y, x)))            # out to the frame corner
    for img, label, style in [(ref, "reference (total)", "k-"),
                              ((1 - eta) * diff, "  diffraction part", "k:"),
                              (eta * scat, "  scatter part", "k--")]:
        ax.loglog(r, radial_profile(img, y, x)[r], style, label=label)
    for s in (0.3, 0.7, 0.95):
        p = URPBloomParams(threshold=0.0, intensity=1.0, scatter=s)
        ax.loglog(r, radial_profile(effective_kernel(p, h, w, y, x), y, x)[r],
                  label=f"URP bloom, scatter {s}")
    ax.set_ylim(1e-11, 1)                            # below this is FFT round-off noise
    ax.set_xlabel("distance from source (px)")
    ax.set_ylabel("energy per pixel (unit source)")
    title = "Centre PSF vs URP pyramid bloom (threshold 0, intensity 1)"
    if ps.placeholder:
        title += "\nPLACEHOLDER lens parameters: not for reporting"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = f"{run_dir('01_gap_study', args.run, 'report')}/kernel_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
