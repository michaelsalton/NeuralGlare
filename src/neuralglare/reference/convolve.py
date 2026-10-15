"""Spatially varying glare: the ground-truth reference for the gap study.

    reference = (1 - η) · Σ_k PSF_k ⊛ (w_k · I)   +   η · S ⊛ I
                ───────── diffraction ─────────       ─ scatter ─

  PSF_k  diffraction PSF at field node k (from reference/psf.py)
  w_k    bilinear "hat" weight of node k; Σ_k w_k = 1 at every pixel
  S, η   spatially invariant veiling-glare kernel and its energy fraction

Weighting the *source* image before convolving (rather than blending outputs) is the
physically correct form for light spreading from each point: a source at pixel p
spreads with Σ_k w_k(p)·PSF_k, the PSF interpolated at p. `--check` verifies this.

Each node's weighted image is non-zero only over its neighbouring cells, so the
convolutions run on crops and are cheap. Light spread past the frame edge is lost
(as on a real sensor) and reported.

Also exposes convolve_uniform() for the uniform-kernel FFT baseline.

Usage (from the repo root):
    uv run python -m neuralglare.reference.convolve --psf data/psf/psf_quick.npz \
        --inputs data/inputs/*.exr --out data/reference --check
"""

import argparse
import json
import os

import numpy as np
from scipy.signal import fftconvolve

from neuralglare.io import read_hdr, write_exr, write_preview
from neuralglare.reference import lens_params as lp
from neuralglare.reference.psf import scatter_kernel


# ------------------------------------------------------------------ PSF set
class PSFSet:
    def __init__(self, path):
        d = np.load(path)
        self.psf = d["psf"].astype(np.float32)          # (rows, cols, K, K, 3)
        self.node_pos = d["node_pos_px"]
        self.cfg = json.loads(str(d["config"]))
        self.placeholder = bool(d["placeholder"])
        self.rows, self.cols, self.K = self.psf.shape[:3]
        self.width = self.cfg["camera"]["width_px"]
        self.height = self.cfg["camera"]["height_px"]
        self.scatter = lp.Scatter(**self.cfg["scatter"])


def hat_weights(n_px, n_nodes):
    """(n_nodes, n_px) piecewise-linear weights over evenly spaced nodes that include
    both ends; columns sum to 1."""
    if n_nodes == 1:
        return np.ones((1, n_px), np.float32)
    nodes = np.linspace(0, n_px - 1, n_nodes)
    x = np.arange(n_px)
    w = 1.0 - np.abs(x[None, :] - nodes[:, None]) / (nodes[1] - nodes[0])
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def interpolated_psf(ps, x, y):
    """PSF at pixel (x, y): the same bilinear blend the reference applies."""
    wx = hat_weights(ps.width, ps.cols)[:, int(x)]
    wy = hat_weights(ps.height, ps.rows)[:, int(y)]
    return np.einsum("r,c,rcijk->ijk", wy, wx, ps.psf)


# ------------------------------------------------------------ convolutions
def convolve_diffraction(image, ps):
    """Σ_k PSF_k ⊛ (w_k · I), output cropped to the frame. Returns (out, lost_energy)."""
    H, W, _ = image.shape
    K, p = ps.K, ps.K // 2
    wx = hat_weights(W, ps.cols)
    wy = hat_weights(H, ps.rows)
    canvas = np.zeros((H + K - 1, W + K - 1, 3), np.float64)
    for r in range(ps.rows):
        ys = np.nonzero(wy[r])[0]
        y0, y1 = ys[0], ys[-1] + 1
        for c in range(ps.cols):
            xs = np.nonzero(wx[c])[0]
            x0, x1 = xs[0], xs[-1] + 1
            sub = image[y0:y1, x0:x1] * (wy[r, y0:y1, None, None] * wx[c, None, x0:x1, None])
            if not sub.any():
                continue
            conv = fftconvolve(sub, ps.psf[r, c], mode="full", axes=(0, 1))
            canvas[y0:y0 + conv.shape[0], x0:x0 + conv.shape[1]] += conv
    out = canvas[p:p + H, p:p + W]
    lost = canvas.sum() - out.sum()
    return out.astype(np.float32), float(lost)


def convolve_uniform(image, kernel_rgb):
    """Single kernel everywhere (the uniform FFT-bloom baseline). 'same' crop."""
    k = kernel_rgb if kernel_rgb.ndim == 3 else kernel_rgb[..., None]
    return fftconvolve(image, k, mode="same", axes=(0, 1)).astype(np.float32)


def convolve_scatter(image, scatter):
    """η-free scatter term S ⊛ I, with a kernel large enough to cover the frame."""
    H, W, _ = image.shape
    S = scatter_kernel(scatter, (2 * H - 1, 2 * W - 1))
    return convolve_uniform(image, S)


def reference(image, ps):
    """Full reference plus its components and an energy report."""
    eta = ps.scatter.energy_fraction
    diff, lost = convolve_diffraction(image, ps)
    scat = convolve_scatter(image, ps.scatter) if eta > 0 else np.zeros_like(image)
    out = (1.0 - eta) * diff + eta * scat
    e_in = float(image.astype(np.float64).sum())
    energy = {
        "input": e_in,
        "output_in_frame": float(out.astype(np.float64).sum()),
        "diffraction_lost_off_frame": lost,
        "relative_error": (float(out.sum()) - e_in) / e_in if e_in else 0.0,
    }
    return out, diff, scat, energy


# ------------------------------------------------------------------- check
def check_point_sources(image, diff, ps, max_points=500):
    """For sparse inputs (e.g. the point grid): the diffraction output around each
    source must equal intensity × the PSF interpolated at that source."""
    lum = image.sum(-1)
    ys, xs = np.nonzero(lum)
    if len(ys) == 0 or len(ys) > max_points:
        print("  check skipped (input is not a sparse point-source image)")
        return None
    p = ps.K // 2
    errs = []
    for y, x in zip(ys, xs):
        if not (p <= y < image.shape[0] - p and p <= x < image.shape[1] - p):
            continue  # PSF crosses the frame edge; not comparable
        expected = image[y, x][None, None, :] * interpolated_psf(ps, x, y)
        actual = diff[y - p:y + p + 1, x - p:x + p + 1]
        errs.append(float(np.abs(actual - expected).sum() / expected.sum()))
    worst = max(errs) if errs else float("nan")
    status = "OK" if worst < 1e-3 else "FAIL"
    print(f"  check: {len(errs)} sources, worst relative L1 error {worst:.2e}  [{status}]")
    return worst


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", required=True, help="npz from reference/psf.py")
    ap.add_argument("--inputs", nargs="+", required=True, help="input EXR files")
    ap.add_argument("--out", default="data/reference")
    ap.add_argument("--check", action="store_true", help="verify on sparse point-source inputs")
    ap.add_argument("--save-components", action="store_true",
                    help="also write the diffraction and scatter terms separately")
    args = ap.parse_args()

    ps = PSFSet(args.psf)
    if ps.placeholder:
        print("WARNING: PSFs were computed from PLACEHOLDER lens parameters. "
              "Do not report results computed with them.\n")
    os.makedirs(args.out, exist_ok=True)
    report = {"psf": args.psf, "placeholder": ps.placeholder, "images": {}}

    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        image = read_hdr(path)
        if image.shape[:2] != (ps.height, ps.width):
            raise ValueError(f"{path}: {image.shape[1]}x{image.shape[0]} does not match the "
                             f"PSF camera {ps.width}x{ps.height}")
        print(f"{name}")
        out, diff, scat, energy = reference(image, ps)
        write_exr(os.path.join(args.out, f"{name}_ref.exr"), out)
        write_preview(os.path.join(args.out, f"{name}_ref_preview.png"), out)
        if args.save_components:
            write_exr(os.path.join(args.out, f"{name}_diffraction.exr"), diff)
            write_exr(os.path.join(args.out, f"{name}_scatter.exr"), scat)
        print(f"  energy: in {energy['input']:.4g}, out {energy['output_in_frame']:.4g} "
              f"(relative {energy['relative_error']:+.2%}; off-frame loss is expected)")
        entry = {"energy": energy}
        if args.check:
            entry["check_worst_rel_l1"] = check_point_sources(image, diff, ps)
        report["images"][name] = entry

    with open(os.path.join(args.out, "reference_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()