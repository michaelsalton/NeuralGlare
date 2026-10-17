"""Spatially varying, physically based point spread functions (Fraunhofer optics).

For each node of a field grid spanning the image (corners included) and each
wavelength:
    pupil field  E = A(u; H, λ) * exp(i 2π W(u; H, λ))
    PSF_λ        = |FFT(E)|²,  integrated exactly onto the image pixel grid
then the wavelengths are combined into RGB with the sensor response.

W (in waves at λ) and the vignetting part of A come from a real-ray trace of the lens
prescription (lens_trace.py), tabulated over the image-space ray direction u, where
|u| = 1 is the f-number cone. A is the diaphragm polygon, cut where rays cross the stop,
times the lens-rim vignetting.

Sampling (why it is correct for every wavelength):
    FFT size M = K·b, where K = kernel size in pixels and b = sub-samples per pixel.
    The focal-plane sample spacing is λ·F#·D_s / M (D_s = 2R, R = samples per unit u).
    Choosing D_s = K·p / (λ·F#) (p = pixel pitch) makes the spacing exactly p / b for
    every λ, so b×b blocks sum to one image pixel. Nyquist (M ≥ 2 × pupil extent, and
    off-axis the pupil reaches |u| = u_max, a little over 1) fixes
    b ≥ 2p·u_max / (λ_min·F#). Red PSFs therefore come out wider than blue ones in pixels,
    as they should.

Outputs (npz): psf[rows, cols, K, K, 3] (each RGB channel sums to 1), node positions,
field heights, and all config values. The scatter halo is NOT baked into the grid:
it is spatially invariant and much wider than K, so convolve.py applies it
separately via scatter_kernel().

Usage (from the repo root):
    python -m reference.psf --config quick --out data/psf/psf_quick.npz
    python -m reference.psf --config default --out data/psf/psf_default.npz
"""

import argparse
import dataclasses
import json
import math
import os
import time

import cv2
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import map_coordinates

from neuralglare.reference import lens_params as lp
from neuralglare.reference import lens_trace


# ------------------------------------------------------------------ pupil model
def pupil_function(x, y, R, ap, tab, ih, il, field_dir):
    """Transmission A and wavefront W (waves at this wavelength) on pupil samples x, y
    (relative to the pupil centre; R samples = one unit of u) for table field ih and
    wavelength il. The table's meridional axis is laid along field_dir."""
    fx, fy = field_dir
    u_mer = (x * fx + y * fy) / R
    u_sag = (-x * fy + y * fx) / R                      # along perp = (-fy, fx)
    U, N = float(tab["U"]), tab["opd"].shape[-1]
    idx = np.stack([(u_mer + U) * ((N - 1) / (2 * U)), (u_sag + U) * ((N - 1) / (2 * U))])

    def sample(name):
        return map_coordinates(tab[name][ih, il], idx, order=1, mode="nearest")

    W = sample("opd")
    soft = ap.edge_softness
    vignette = np.clip(sample("vig") * R / soft + 0.5, 0.0, 1.0)   # u → samples

    # Diaphragm: regular polygon, vertices on the stop circle, fixed in image orientation,
    # evaluated where each ray crosses the stop (stop radii), turned into a distance in
    # pupil samples by dividing by the local gradient.
    s_mer, s_sag = sample("stop_y"), sample("stop_x")
    sx = s_mer * fx - s_sag * fy                        # stop point in image orientation
    sy = s_mer * fy + s_sag * fx
    n = ap.blades
    rot = math.radians(ap.rotation_deg)
    proj = np.full(x.shape, -np.inf, np.float32)
    for k in range(n):
        phi = rot + (2 * k + 1) * math.pi / n           # edge normals
        np.maximum(proj, sx * math.cos(phi) + sy * math.sin(phi), out=proj)
    inside = math.cos(math.pi / n) - proj               # > 0 inside, in stop radii
    gy, gx = np.gradient(inside)
    blades = np.clip(inside / np.maximum(np.hypot(gx, gy), 1e-9) / soft + 0.5, 0.0, 1.0)
    return blades * vignette, W


# --------------------------------------------------------- monochromatic PSF
def subsamples_per_pixel(cfg, u_max=1.0):
    lam_min = min(cfg.spectral.wavelengths_um)
    b = math.ceil(2.0 * cfg.camera.pixel_pitch_um * u_max / (lam_min * cfg.camera.f_number))
    return b if b % 2 else b + 1    # odd, so pixel bins are centred on the PSF origin


def psf_mono(cfg, lam_um, il, ih, field_dir, tab):
    """K×K PSF for one wavelength and field position, integrated over pixels, sum 1."""
    K, b = cfg.kernel_px, subsamples_per_pixel(cfg, float(tab["u_max"]))
    M = K * b
    R = 0.5 * K * cfg.camera.pixel_pitch_um / (lam_um * cfg.camera.f_number)  # samples per unit u

    # Evaluate the pupil only on its bounding box, then embed in the M×M field.
    half = math.ceil(float(tab["u_max"]) * R + 2 * cfg.aperture.edge_softness) + 1
    c = M // 2
    coords = np.arange(-half, half + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)          # x → image +x (right), y → image +y (down)
    A, W = pupil_function(x, y, R, cfg.aperture, tab, ih, il, field_dir)
    phase = 2.0 * np.pi * W

    E = np.zeros((M, M), np.complex64)
    E[c - half:c + half + 1, c - half:c + half + 1] = (A * np.exp(1j * phase)).astype(np.complex64)

    F = sfft.fftshift(sfft.fft2(sfft.ifftshift(E), workers=-1))
    P = (F.real ** 2 + F.imag ** 2).astype(np.float64)
    P = P.reshape(K, b, K, b).sum(axis=(1, 3))   # exact pixel integration
    return P / P.sum()


# ------------------------------------------------------------- field grid
def field_nodes(cfg):
    """Node positions in pixels (including frame edges/corners), field height H
    (1 at the corners) and unit field direction per node."""
    rows, cols = cfg.grid
    Wpx, Hpx = cfg.camera.width_px, cfg.camera.height_px
    xs = np.linspace(0, Wpx - 1, cols)
    ys = np.linspace(0, Hpx - 1, rows)
    cx, cy = (Wpx - 1) / 2, (Hpx - 1) / 2
    half_diag = math.hypot(cx, cy)
    nodes = []
    for r, y in enumerate(ys):
        for col, x in enumerate(xs):
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            H = d / half_diag
            direction = (dx / d, dy / d) if d > 1e-9 else (0.0, 1.0)
            nodes.append((r, col, x, y, H, direction))
    return nodes


def psf_rgb(cfg, tab, H, field_dir, wrgb=None):
    """K×K×3 PSF at field height H (must be in the table) and direction; each channel
    sums to 1."""
    K = cfg.kernel_px
    wrgb = cfg.spectral.rgb_weights() if wrgb is None else wrgb
    ih = lens_trace.height_index(tab, H)
    acc = np.zeros((K, K, 3))
    for li, lam in enumerate(cfg.spectral.wavelengths_um):
        acc += psf_mono(cfg, lam, li, ih, field_dir, tab)[..., None] * wrgb[li][None, None, :]
    return acc / acc.sum(axis=(0, 1), keepdims=True)


def compute_psf_grid(cfg, verbose=True):
    rows, cols = cfg.grid
    K = cfg.kernel_px
    lams = cfg.spectral.wavelengths_um
    wrgb = cfg.spectral.rgb_weights()                    # (n_lambda, 3)
    psf = np.zeros((rows, cols, K, K, 3), np.float32)
    nodes = field_nodes(cfg)
    tab = lens_trace.load_or_build(cfg, [n[4] for n in nodes], verbose=verbose)

    if verbose:
        b = subsamples_per_pixel(cfg, float(tab["u_max"]))
        print(f"grid {rows}x{cols}, K={K}, {len(lams)} wavelengths, "
              f"FFT {K * b}^2 (b={b}), F/{cfg.camera.f_number}, lens {cfg.lens.name}")
    t0 = time.time()
    for i, (r, c, x, y, H, d) in enumerate(nodes):
        psf[r, c] = psf_rgb(cfg, tab, H, d, wrgb)
        if verbose:
            print(f"  node {i + 1:>3}/{len(nodes)}  H={H:.2f}  ({time.time() - t0:.0f}s)")
    pos = np.array([[n[2], n[3]] for n in nodes]).reshape(rows, cols, 2)
    Hs = np.array([n[4] for n in nodes]).reshape(rows, cols)
    # Exact on-axis PSF (the table always holds H = 0). The grid has no node at the frame
    # centre, so the blend there is not the on-axis PSF; the centre-PSF baseline uses this.
    centre = psf_rgb(cfg, tab, 0.0, (0.0, 1.0), wrgb).astype(np.float32)
    return psf, pos, Hs, centre


# ------------------------------------------------------------- scatter halo
def scatter_kernel(scatter, size):
    """Spatially invariant veiling-glare kernel (lens_params.Scatter), sum 1.
    `size` is an odd int or an odd (height, width) pair; make it ~2× the image so the
    halo from any pixel covers the whole frame."""
    s = scatter
    kh, kw = (size, size) if np.isscalar(size) else size
    ry = np.arange(kh) - kh // 2
    rx = np.arange(kw) - kw // 2
    rr = rx[None, :] ** 2 + ry[:, None] ** 2
    k = (rr + s.core_radius_px ** 2) ** (-s.exponent / 2.0)
    return (k / k.sum()).astype(np.float32)


# ---------------------------------------------------------------- reporting
def summarize(psf, Hs):
    """Per-node indicators of spatial variation, compared with the centre-most node."""
    rows, cols = Hs.shape
    rc = np.unravel_index(np.argmin(Hs), Hs.shape)
    ref = psf[rc].mean(-1)
    K = psf.shape[2]
    yy, xx = np.mgrid[:K, :K] - K // 2
    print(f"\n{'node':<8}{'H':>6}{'L1 vs centre':>14}{'centroid px':>14}{'EE r<=2px':>11}")
    for r in range(rows):
        for c in range(cols):
            p = psf[r, c].mean(-1)
            l1 = np.abs(p - ref).sum()
            cy, cx = (p * yy).sum(), (p * xx).sum()
            ee = p[(xx ** 2 + yy ** 2) <= 4].sum()
            print(f"({r},{c}){'':<3}{Hs[r, c]:>6.2f}{l1:>14.4f}{f'({cx:+.2f},{cy:+.2f})':>14}{ee:>11.4f}")
    print("\nL1 vs centre: 0 = identical shape, 2 = no overlap. This is a first hint of how much"
          "\na single uniform kernel will miss at each part of the frame.")


def preview(psf, path, decades=7):
    """Tile all node PSFs (log scale per channel, normalised per tile) into one PNG."""
    rows, cols, K = psf.shape[:3]
    pad = 4
    canvas = np.zeros((rows * (K + pad) + pad, cols * (K + pad) + pad, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            p = psf[r, c] / psf[r, c].max()
            v = np.clip((np.log10(p + 10.0 ** -(decades + 1)) + decades) / decades, 0, 1)
            y0, x0 = pad + r * (K + pad), pad + c * (K + pad)
            canvas[y0:y0 + K, x0:x0 + K] = v
    scale = max(1, 600 // canvas.shape[0])
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, (canvas[..., ::-1] * 255).astype(np.uint8))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["quick", "default"], default="quick")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = lp.QUICK if args.config == "quick" else lp.DEFAULT
    out = args.out or f"data/psf/psf_{args.config}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if lp.PLACEHOLDER:
        print("WARNING: lens_params.py contains PLACEHOLDER values. "
              "Do not report results computed with them.\n")

    psf, pos, Hs, centre = compute_psf_grid(cfg)
    np.savez_compressed(out, psf=psf, node_pos_px=pos, field_height=Hs, centre_psf=centre,
                        config=json.dumps(dataclasses.asdict(cfg)),
                        placeholder=lp.PLACEHOLDER)
    preview(psf, out.replace(".npz", "_preview.png"))
    summarize(psf, Hs)
    print(f"\nwrote {out} and {out.replace('.npz', '_preview.png')}")


if __name__ == "__main__":
    main()