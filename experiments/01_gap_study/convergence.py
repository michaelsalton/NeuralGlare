"""Convergence check for the numerical settings of the reference PSFs.

Each numerical setting of lens_params.DEFAULT is doubled on its own and the PSF at the
frame corner (H = 1, where aberrations are largest) is recomputed. The change is
    relative L1 = sum |PSF_doubled - PSF_default| / sum PSF_default   (all channels)
  rays          rays per side of the traced pupil grid
  table_px      pupil table resolution
  wavelengths   spectral samples, 12 -> 24 over the same 420-680 nm
  kernel_px     PSF size; the doubled kernel is cropped to the default size, so the change
                includes the energy the default kernel truncates
Separately, the node grid: the exact PSF at the centre of the corner cell against the
bilinear blend of its 4 node PSFs, which is what convolve.py applies there.

Target: every change below TOLERANCE. Writes outputs/01_gap_study/default/checks/
convergence.json.

Usage:
    uv run python experiments/01_gap_study/convergence.py
"""

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np

from neuralglare.paths import run_dir
from neuralglare.reference import lens_params as lp
from neuralglare.reference import lens_trace
from neuralglare.reference.convolve import hat_weights
from neuralglare.reference.psf import field_nodes, psf_rgb

TOLERANCE = 0.01


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / np.abs(b).sum())


def field_at(cfg, x, y):
    """(H, unit direction) of pixel (x, y), as psf.field_nodes defines them."""
    cx, cy = (cfg.camera.width_px - 1) / 2, (cfg.camera.height_px - 1) / 2
    dx, dy = x - cx, y - cy
    d = math.hypot(dx, dy)
    return d / math.hypot(cx, cy), (dx / d, dy / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="convergence.json",
                    help="file name in the default run's checks/ folder")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["rays", "table_px", "wavelengths", "kernel_px", "grid"],
                    help="run only these checks")
    args = ap.parse_args()
    run = set(args.only or ["rays", "table_px", "wavelengths", "kernel_px", "grid"])
    base = lp.DEFAULT
    t0 = time.time()

    # corner node and the 4 nodes of the corner cell
    nodes = {(n[0], n[1]): n for n in field_nodes(base)}
    corner = nodes[(0, 0)]
    cell = [nodes[(r, c)] for r in (0, 1) for c in (0, 1)]
    xm, ym = round((cell[0][2] + cell[1][2]) / 2), round((cell[0][3] + cell[2][3]) / 2)
    Hm, dm = field_at(base, xm, ym)
    heights = [corner[4]] + ([n[4] for n in cell] + [Hm] if "grid" in run else [])

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    log("tracing default settings")
    tab = lens_trace.load_or_build(base, heights, verbose=False)
    ref = psf_rgb(base, tab, corner[4], corner[5])
    results = {}

    variants = {
        "rays": replace(base, trace=replace(base.trace, rays=2 * base.trace.rays - 1)),
        "table_px": replace(base, trace=replace(base.trace, table_px=2 * base.trace.table_px - 1)),
        "wavelengths": replace(base, spectral=replace(base.spectral, wavelengths_um=tuple(
            np.round(np.linspace(0.42, 0.68, 2 * len(base.spectral.wavelengths_um)), 4)))),
    }
    for name, cfg in variants.items():
        if name not in run:
            continue
        log(f"{name}: tracing")
        t = lens_trace.load_or_build(cfg, [corner[4]], verbose=False)
        results[name] = rel_l1(psf_rgb(cfg, t, corner[4], corner[5]), ref)
        log(f"{name}: change {results[name]:.3%}")

    if "kernel_px" in run:
        log("kernel_px: computing doubled kernel")
        big = replace(base, kernel_px=2 * base.kernel_px + 1)
        p = psf_rgb(big, tab, corner[4], corner[5])
        o = (big.kernel_px - base.kernel_px) // 2
        results["kernel_px"] = rel_l1(p[o:o + base.kernel_px, o:o + base.kernel_px], ref)
        results["kernel_px_energy_outside"] = float(1 - p[o:o + base.kernel_px,
                                                          o:o + base.kernel_px].sum() / p.sum())
        log(f"kernel_px: change {results['kernel_px']:.3%}")

    if "grid" in run:
        log("grid: exact PSF at the corner-cell centre vs bilinear blend of its nodes")
        wx = hat_weights(base.camera.width_px, base.grid[1])[:, xm]
        wy = hat_weights(base.camera.height_px, base.grid[0])[:, ym]
        blend = sum(wy[n[0]] * wx[n[1]] * psf_rgb(base, tab, n[4], n[5]) for n in cell)
        results["grid"] = rel_l1(blend, psf_rgb(base, tab, Hm, dm))
        log(f"grid: change {results['grid']:.3%}")

    checks = {k: v for k, v in results.items() if k != "kernel_px_energy_outside"}
    verdict = {k: ("PASS" if v < TOLERANCE else "FAIL") for k, v in checks.items()}
    print(f"\n{'setting':<14}{'default':>10}{'change':>10}  (tolerance {TOLERANCE:.0%})")
    defaults = {"rays": base.trace.rays, "table_px": base.trace.table_px,
                "wavelengths": len(base.spectral.wavelengths_um), "kernel_px": base.kernel_px,
                "grid": "x".join(map(str, base.grid))}
    for k, v in checks.items():
        print(f"{k:<14}{defaults[k]!s:>10}{v:>10.3%}  {verdict[k]}")
    if "kernel_px_energy_outside" in results:
        print(f"energy outside the default kernel: {results['kernel_px_energy_outside']:.3%}")

    out = f"{run_dir('01_gap_study', 'default', 'checks')}/{args.out}"   # checks lp.DEFAULT
    with open(out, "w") as fh:
        json.dump({"tolerance": TOLERANCE, "defaults": defaults, "relative_l1": results,
                   "verdict": verdict, "corner_H": corner[4],
                   "cell_centre_px": [xm, ym], "cell_centre_H": Hm}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
