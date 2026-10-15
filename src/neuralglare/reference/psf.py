"""Spatially varying, physically based point spread functions (Fraunhofer optics).

For each node of a field grid spanning the image (corners included) and each
wavelength:
    pupil field  E = A(x, y; H) * exp(i 2π W(x, y; H) λ_ref / λ)
    PSF_λ        = |FFT(E)|²,  integrated exactly onto the image pixel grid
then the wavelengths are combined into RGB with the sensor response.

Sampling (why it is correct for every wavelength):
    FFT size M = K·b, where K = kernel size in pixels and b = sub-samples per pixel.
    The focal-plane sample spacing is λ·F#·D_s / M (D_s = pupil diameter in samples).
    Choosing D_s = K·p / (λ·F#) (p = pixel pitch) makes the spacing exactly p / b for
    every λ, so b×b blocks sum to one image pixel. Nyquist (M ≥ 2·D_s) fixes
    b ≥ 2p / (λ_min·F#). Red PSFs therefore come out wider than blue ones in pixels,
    as they should.

Outputs (npz): psf[rows, cols, K, K, 3] (each RGB channel sums to 1), node positions,
field heights, and all config values. The scatter halo is NOT baked into the grid:
it is spatially invariant and much wider than K, so convolve.py applies it
separately via scatter_kernel().

Usage (from the repo root):
    uv run python -m neuralglare.reference.psf --config quick --out data/psf/psf_quick.npz
    uv run python -m neuralglare.reference.psf --config default --out data/psf/psf_default.npz

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

from neuralglare.reference import lens_params as lp


# ------------------------------------------------------------------ pupil model
def aperture_mask(x, y, R, ap, H, field_dir):
    """Polygonal diaphragm (vertices on the circle of radius R) intersected with a
    shifted vignetting circle. x, y in pupil samples relative to the pupil centre."""
    n = ap.blades
    soft = ap.edge_softness
    rot = math.radians(ap.rotation_deg)
    r_in = R * math.cos(math.pi / n)
    proj = np.full(x.shape, -np.inf, np.float32)
    for k in range(n):
        phi = rot + (2 * k + 1) * math.pi / n          # edge normals
        np.maximum(proj, x * math.cos(phi) + y * math.sin(phi), out=proj)
    mask = np.clip((r_in - proj) / soft + 0.5, 0.0, 1.0)

    if H > 0 and ap.vignette_shift > 0:
        cx = ap.vignette_shift * H * R * field_dir[0]
        cy = ap.vignette_shift * H * R * field_dir[1]
        d = np.hypot(x - cx, y - cy)
        mask *= np.clip((ap.vignette_radius * R - d) / soft + 0.5, 0.0, 1.0)
    return mask


def wavefront_waves(x, y, R, seidel, H, field_dir):
    """Seidel wavefront in waves at seidel.lambda_ref_um. rho·cos(theta) is the pupil
    coordinate along the field direction."""
    rho2 = (x * x + y * y) / (R * R)
    rc = (x * field_dir[0] + y * field_dir[1]) / R      # rho cos(theta)
    return (seidel.W040 * rho2 * rho2
            + seidel.W131 * H * rho2 * rc
            + seidel.W222 * H * H * rc * rc
            + seidel.W220 * H * H * rho2)


# --------------------------------------------------------- monochromatic PSF
def subsamples_per_pixel(cfg):
    lam_min = min(cfg.spectral.wavelengths_um)
    return int(math.ceil(2.0 * cfg.camera.pixel_pitch_um / (lam_min * cfg.camera.f_number)))


def psf_mono(cfg, lam_um, H, field_dir, seidel):
    """K×K PSF for one wavelength and field position, integrated over pixels, sum 1."""
    K, b = cfg.kernel_px, subsamples_per_pixel(cfg)
    M = K * b
    R = 0.5 * K * cfg.camera.pixel_pitch_um / (lam_um * cfg.camera.f_number)  # pupil radius, samples

    # Evaluate the pupil only on its bounding box, then embed in the M×M field.
    half = int(math.ceil(R + 2 * cfg.aperture.edge_softness)) + 1
    c = M // 2
    coords = np.arange(-half, half + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)          # x → image +x (right), y → image +y (down)
    A = aperture_mask(x, y, R, cfg.aperture, H, field_dir)
    W = wavefront_waves(x, y, R, seidel, H, field_dir)
    phase = (2.0 * np.pi * seidel.lambda_ref_um / lam_um) * W

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


def compute_psf_grid(cfg, verbose=True):
    rows, cols = cfg.grid
    K = cfg.kernel_px
    seidel = cfg.seidel.at_f_number(cfg.camera.f_number)
    lams = cfg.spectral.wavelengths_um
    wrgb = cfg.spectral.rgb_weights()                    # (n_lambda, 3)
    psf = np.zeros((rows, cols, K, K, 3), np.float32)
    nodes = field_nodes(cfg)

    if verbose:
        b = subsamples_per_pixel(cfg)
        print(f"grid {rows}x{cols}, K={K}, {len(lams)} wavelengths, "
              f"FFT {K * b}^2 (b={b}), F/{cfg.camera.f_number}")
    t0 = time.time()
    for i, (r, c, x, y, H, d) in enumerate(nodes):
        acc = np.zeros((K, K, 3))
        for li, lam in enumerate(lams):
            acc += psf_mono(cfg, lam, H, d, seidel)[..., None] * wrgb[li][None, None, :]
        psf[r, c] = acc / acc.sum(axis=(0, 1), keepdims=True)
        if verbose:
            print(f"  node {i + 1:>3}/{len(nodes)}  H={H:.2f}  ({time.time() - t0:.0f}s)")
    pos = np.array([[n[2], n[3]] for n in nodes]).reshape(rows, cols, 2)
    Hs = np.array([n[4] for n in nodes]).reshape(rows, cols)
    return psf, pos, Hs


# ------------------------------------------------------------- scatter halo
def scatter_kernel(cfg, size):
    """Spatially invariant veiling-glare kernel, `size`×`size` (odd), sum 1.
    Choose `size` large (up to ~2× the image) since the halo is wide."""
    s = cfg.scatter
    r = np.arange(size) - size // 2
    rr = r[None, :] ** 2 + r[:, None] ** 2
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

    psf, pos, Hs = compute_psf_grid(cfg)
    np.savez_compressed(out, psf=psf, node_pos_px=pos, field_height=Hs,
                        config=json.dumps(dataclasses.asdict(cfg)),
                        placeholder=lp.PLACEHOLDER)
    preview(psf, out.replace(".npz", "_preview.png"))
    summarize(psf, Hs)
    print(f"\nwrote {out} and {out.replace('.npz', '_preview.png')}")


if __name__ == "__main__":
    main()