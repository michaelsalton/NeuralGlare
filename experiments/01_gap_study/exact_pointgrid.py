"""Exact reference for the point grid: every source gets the PSF computed at its own pixel.

The general reference (convolve.py) blends the PSFs of the surrounding grid nodes. Between
nodes that blend is not the real PSF (10-41% relative L1 on the 6x8 grid; convergence.py),
and every point-grid source sits between nodes. Sources are sparse, so here each one is
simply intensity x PSF(its own H and direction), placed at its pixel and cropped to the
frame, with no interpolation. This is the primary gap-study measurement.

Usage:
    uv run python experiments/01_gap_study/exact_pointgrid.py --config default \
        --out data/reference_default/pointgrid_ref_exact.exr
"""

import argparse
import json
import math
import os
import time

import numpy as np

from neuralglare.io import read_hdr, write_exr, write_preview
from neuralglare.reference import lens_params as lp
from neuralglare.reference import lens_trace
from neuralglare.reference.psf import psf_rgb


def field_at(cfg, x, y):
    """(H, unit direction) of pixel (x, y), as psf.field_nodes defines them."""
    cx, cy = (cfg.camera.width_px - 1) / 2, (cfg.camera.height_px - 1) / 2
    dx, dy = x - cx, y - cy
    d = math.hypot(dx, dy)
    return d / math.hypot(cx, cy), ((dx / d, dy / d) if d > 1e-9 else (0.0, 1.0))


def splat(shape, points, psfs):
    """Sum of value x PSF centred on each (x, y), cropped to the frame. Returns
    (image, energy lost off the frame)."""
    h, w = shape[:2]
    K = psfs[0].shape[0]
    p = K // 2
    canvas = np.zeros((h + K - 1, w + K - 1, 3), np.float64)
    for (x, y, value), k in zip(points, psfs):
        canvas[y:y + K, x:x + K] += value[None, None, :] * k
    out = canvas[p:p + h, p:p + w]
    return out.astype(np.float32), float(canvas.sum() - out.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["quick", "default"], default="default")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--out", default="data/reference_default/pointgrid_ref_exact.exr")
    args = ap.parse_args()
    cfg = lp.QUICK if args.config == "quick" else lp.DEFAULT

    with open(os.path.join(args.inputs, "manifest.json")) as fh:
        meta = json.load(fh)["images"]["pointgrid"]
    img = read_hdr(os.path.join(args.inputs, "pointgrid.exr"))
    points = [(x, y, img[y, x].astype(np.float64)) for y in meta["ys"] for x in meta["xs"]]
    fields = [field_at(cfg, x, y) for x, y, _ in points]

    t0 = time.time()
    tab = lens_trace.load_or_build(cfg, [f[0] for f in fields])
    wrgb = cfg.spectral.rgb_weights()
    psfs = []
    for i, (H, d) in enumerate(fields):
        psfs.append(psf_rgb(cfg, tab, H, d, wrgb))
        print(f"  point {i + 1:>2}/{len(fields)}  H={H:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    out, lost = splat(img.shape, points, psfs)
    e_in = float(img.astype(np.float64).sum())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_exr(args.out, out)
    write_preview(args.out.replace(".exr", "_preview.png"), out)
    report = {"config": args.config, "points": len(points), "energy_input": e_in,
              "energy_out_in_frame": float(out.astype(np.float64).sum()),
              "lost_off_frame": lost, "placeholder": lp.PLACEHOLDER,
              "wavelengths": len(cfg.spectral.wavelengths_um)}
    with open(args.out.replace(".exr", "_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out}; energy in {e_in:.6g}, out {report['energy_out_in_frame']:.6g}, "
          f"lost off frame {lost:.3g}")


if __name__ == "__main__":
    main()
