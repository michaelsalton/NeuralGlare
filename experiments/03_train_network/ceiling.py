"""Phase 2, step 2: ceiling analysis on the point grid. Each oracle is the exact reference with
one feature removed from every PSF (the bank's test PSFs, which reproduce
pointgrid_ref_exact.exr to float rounding), scored against the exact reference with
train_points' measurement.

    identity    the bank PSFs unchanged: the control, ~0
    round       angular average per channel: no spikes or coma, colour kept
    grey        luminance in every channel: no colour, shape kept
    round_grey  both: a round, grey, centred kernel (the Step 1 model's class)
    round_median, round_median_grey
                as round and round_grey, with the ring median in place of the mean
    aperture    median over the 14 rotations by 360/14 per channel. INVALID: the PSF isn't
                14-fold symmetric, and rotation smears the core (decision log 2026-11-04)
    aperture_d7 the 7-blade polygon's symmetry: halo median over 7 rotations by 360/7 and
                their mirrors, core (r <= 5 px) ring median. INVALID: rotating on the pixel
                grid blurs the spikes by ~0.016 FLIP (decision log 2026-11-04)
    identity_fine, aperture_fine
                from the 5x oversampled test PSFs (psf_bank.py --oversample 5 --only test):
                the sampling control, and aperture_d7 rotated at 5x, then summed to pixels
    sym_core, sym_rays, sym_glow, sym_not_core, sym_not_rays, sym_not_glow,
    sym_rays_w3, sym_glow_w3
                the coma split (decision log 2026-11-17): aperture_fine applied only inside a
                mask, identity_fine elsewhere. The masks (coma_masks) partition the PSF into
                core (aperture_fine's round core), rays (within 2 px of a ray line) and glow
                (the rest); sym_<piece> symmetrises one piece, sym_not_<piece> the other two,
                *_w3 the 3 px ray band (robustness)
    sym_annulus, sym_rays_far, sym_glow_far, rays_far_bright
                ray follow-ups (decision log 2026-11-17, an amendment after the coma split):
                the ray band split at ANNULUS_PX into the 5.25-9 px annulus and the far rays;
                rays_far_bright is sym_rays_far with each ray's exact light restored per
                channel (exact brightness, symmetric shape)
    rays_far_profile
                ray profile oracle (decision log 2026-11-17, second amendment): as
                rays_far_bright, the light restored per ray and radial segment
    xs_exact, xs_line, xs_line_bg, xs_line_skirt, xs_line_cskirt, xs_line_cskirt_bg
                ray cross-section oracle (decision log 2026-11-26): on the 5x bank, every cell
                at r >= ANNULUS_PX belongs to its nearest ray's wedge; per (ray, channel) the
                cross-ray profile is a class fitted with parameters a + b ln(r / XS_R0_PX),
                each 1 px radius bin keeping its exact light. xs_exact is the exact profile
                averaged per radius bin (the floor); the others a Gaussian line, plus a flat
                glow (_bg), a Gaussian (skirt) or Lorentzian (cskirt) skirt

Caveat found on the first run (decision log 2026-11-04): an oracle is only a ceiling if it is
the best kernel without its feature. The angular *mean* is the best round kernel in linear
squared error, not in FLIP or the log loss: it spreads the spikes' energy into a ring of haze
between them. round and round_grey score worse in FLIP than the trained Step 1 model, so they
are not ceilings. The ring *median* is the best round kernel in the absolute log loss (the
training loss), so the median oracles decide the questions (amended, same date). grey keeps
the luminance exactly, so it is a reasonable bound on colour.

What each result means was fixed before the first run (decision log 2026-11-04). FLIP isn't
additive, so differences between rows rank the features rather than split the error exactly.

    uv run python experiments/03_train_network/ceiling.py --run default
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erf

from neuralglare.io import luminance, read_hdr
from neuralglare.paths import REFS, RUNS, psf_bank_path, run_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "01_gap_study"))
sys.path.insert(0, str(ROOT / "experiments" / "03_train_network"))
from exact_pointgrid import splat
from train_points import (
    COMPARE,
    EPS,
    EXPERIMENT,
    INTENSITY,
    strongest_baselines,
    summarize,
    table,
    window_records,
)

STUDY = "ceiling"
STEP1 = ("baseline", "train_points_lerp_abs_fitted_lr0.0001")   # the current best
CONTROL_FLIP = 1e-3                 # identity must score below this in every bin
SPIKES = 14                         # 7 aperture blades: 14 spikes, 360/14 apart (measured in the bank)
BLADES = 7                          # the PSF's symmetry: 7 rotations by 360/7, plus a mirror
CORE_PX = 5                         # aperture_d7: ring median within this radius (99.7 % of the energy)
APERTURE_CONTROL_FLIP = 0.01        # aperture oracles at the H ~ 0 source (already symmetric) must score below
RAY_HALF_WIDTH_PX = 2.0             # coma split: ray band, perpendicular px (ray FWHM 1.2-2.4, decision log 2026-11-05)
ROBUST_HALF_WIDTH_PX = 3.0          # coma split: the robustness band
VALIDITY_SLACK = 1e-3               # coma split: a one-piece oracle may exceed aperture_fine by this much
ANNULUS_PX = 9.0                    # ray follow-ups: the +-2 px band fills the ring inside this (0.449 * 9 ~ 4 px)
RAY_PROFILE_BINS_PX = (ANNULUS_PX, 16.0, 32.0)   # ray profile oracle: lower edges of the radial segments
XS_R0_PX = 16.0                     # ray cross-section oracle: each parameter is a + b * ln(r / XS_R0_PX)
XS_FIT_EPS = EPS / INTENSITY        # the log loss's eps as a share of the source (each PSF channel sums to 1)
XS_ENOUGH = 0.005                   # a class is enough if within this of xs_exact at centre and edge
XS_VARIANTS = ("xs_exact", "xs_line", "xs_line_bg", "xs_line_skirt", "xs_line_cskirt",
               "xs_line_cskirt_bg")
CURRENT = ("ray_bright", "train_points_bright_abs_fitted_lr0.001_ee0.02_mips1")   # the current best
# (init, lower, upper) of each piece's parameters, each one as [value at XS_R0_PX, slope per e-fold of r]
XS_LINE = ([0.0, 0.0, np.log(0.5), 0.0],                         # offset, ln width
           [-2, -2, np.log(0.05), -2], [2, 2, np.log(4.0), 2])
XS_SKIRT = ([np.log(0.05), 0.0, 0.0, 0.0, np.log(2.0), 0.0],     # ln weight, offset, ln width
            [-12, -4, -3, -3, np.log(0.3), -2], [2, 4, 3, 3, np.log(20.0), 2])
XS_GLOW = ([np.log(1e-3), 0.0], [-25, -6], [0, 6])               # ln level per px


def rings(K):
    """Half-pixel ring index round(2r) of every pixel of a K x K PSF, around its centre pixel
    (the source). Half-pixel rings keep the core's first distances (0, 1, √2, 2) apart;
    further out a ring holds ~π·r pixels."""
    c = K // 2
    y, x = np.mgrid[:K, :K]
    return np.rint(2 * np.hypot(x - c, y - c)).astype(int)


def round_psf(psf):
    """Average over angle around the source pixel, per channel: every pixel gets the mean of
    its ring. Each ring keeps its energy, so each channel still sums to 1 and colour is
    kept."""
    K = psf.shape[0]
    ring = rings(K).ravel()
    count = np.maximum(np.bincount(ring), 1)          # some ring indices hold no pixel
    out = np.empty(psf.shape, np.float64)
    for ch in range(3):
        mean = np.bincount(ring, weights=psf[..., ch].ravel()) / count
        out[..., ch] = mean[ring].reshape(K, K)
    return out


def round_median_psf(psf, grey=False):
    """Per ring, the median: the best round kernel in the absolute log loss. log(I·v + eps)
    is monotonic in v, so a ring's median minimises its sum of |log(out + eps) -
    log(ref + eps)| for any intensity I and eps. grey=True pools the three channels into one
    median, the best round grey kernel. The median ignores the spikes' energy, so this
    doesn't conserve energy: a looser class than the model's, which a ceiling may be."""
    ring = rings(psf.shape[0])
    out = np.empty(psf.shape, np.float64)
    for r in np.unique(ring):
        sel = ring == r
        out[sel] = np.median(psf[sel]) if grey else np.median(psf[sel], axis=0)
    return out


def grey_psf(psf):
    """Every channel replaced by the PSF's luminance: same luminance at every pixel, no
    colour. The luminance weights sum to 1 and each channel sums to 1, so the result does."""
    return np.repeat(luminance(psf)[..., None], 3, axis=-1)


def aperture_psf(psf, order=1):
    """Per channel, the per-pixel median over the PSF rotated about the source pixel by every
    multiple of 360/14: what the 7-blade aperture's symmetry keeps (spikes, round profile,
    colour) stays, what points one way (coma) goes. A pixel's 14 copies are its orbit under
    that symmetry, so their median is the best symmetric value in the absolute log loss, as
    the ring median is for round kernels. The rotations interpolate (spline order), which
    blurs the thin spikes slightly; the control at the H ~ 0 source bounds it."""
    copies = [psf] + [ndimage.rotate(psf, 360 * k / SPIKES, axes=(1, 0), reshape=False, order=order)
                      for k in range(1, SPIKES)]
    return np.maximum(np.median(copies, axis=0), 0)


def aperture_d7_psf(psf):
    """The 7-blade polygon's symmetry, per channel. Halo (r > CORE_PX): the per-pixel median
    over the PSF rotated by every multiple of 360/7 and the same rotations of its mirror about
    the horizontal axis (exact on the grid; the axis measured exact at H = 0.001), cubic,
    negatives clipped. Core: the ring median, which needs no interpolation (rotating the
    few-pixel core smears it) and is allowed by the symmetry. Keeps the spikes, the round
    profile and colour; removes what points one way (coma)."""
    copies = []
    for k in (psf, psf[::-1]):
        copies += [k] + [ndimage.rotate(k, 360 * j / BLADES, axes=(1, 0), reshape=False, order=3)
                         for j in range(1, BLADES)]
    out = np.maximum(np.median(copies, axis=0), 0)
    core = rings(psf.shape[0]) <= 2 * CORE_PX                    # ring index is round(2r)
    out[core] = round_median_psf(psf)[core]
    return out


def bin_fine(fine, s):
    """A K·s x K·s PSF of 1/s-pixel cells summed back to K x K pixels."""
    K = fine.shape[0] // s
    return fine.reshape(K, s, K, s, 3).sum(axis=(1, 3))


def aperture_fine_psf(psf, fine, s):
    """aperture_d7 on the s-times oversampled PSF (psf_bank.py --oversample), where a 1-2 px
    spike spans s-2s samples and cubic rotation no longer blurs it at the pixel scale. Halo:
    the per-sample median over the polygon's 14 copies, summed back to pixels; core
    (r <= CORE_PX): the ring median of the pixel PSF, as aperture_d7. The median is taken per
    fine sample, so this is the best symmetric kernel in the absolute log loss only
    approximately."""
    copies = []
    for k in (fine, fine[::-1]):                     # mirror about the horizontal axis
        copies += [k] + [ndimage.rotate(k, 360 * j / BLADES, axes=(1, 0), reshape=False, order=3)
                         for j in range(1, BLADES)]
    out = bin_fine(np.maximum(np.median(copies, axis=0), 0), s)
    core = rings(psf.shape[0]) <= 2 * CORE_PX
    out[core] = round_median_psf(psf)[core]
    return out


PIECES = ("core", "rays", "glow")


def coma_masks(K, half_width=RAY_HALF_WIDTH_PX):
    """Partition of a K x K PSF about its centre pixel into core / rays / glow (decision log
    2026-11-17). core: aperture_fine's round core (ring index <= 2 * CORE_PX, r <= 5.25 px);
    rays: outside it, within half_width px perpendicular distance of one of the 14 ray lines
    (angles k * 360/14 from arctan2(dy, dx), y down as glare.slang's RAY_DIRS); glow: the rest.
    Built from radius and the ray angles only, so each mask is invariant under the PSF's
    symmetry and symmetrising inside it is a restriction of the aperture_fine median.
    Perpendicular distance rather than angle, since a ray has a constant width in pixels; rays
    are 0.449 r px apart, so a +-2 px band fills the ring up to r ~ 9 px. "ray" (not a piece)
    is every pixel's nearest ray index k, whose direction is k * 360/14."""
    c = K // 2
    y, x = np.mgrid[:K, :K]
    dx, dy = x - c, y - c
    step = 2 * np.pi / SPIKES
    shifted = np.arctan2(dy, dx) + step / 2
    off = shifted % step - step / 2                             # angle to the nearest ray
    dist = np.hypot(dx, dy) * np.abs(np.sin(off))               # perpendicular px to it
    core = rings(K) <= 2 * CORE_PX
    rays = ~core & (dist <= half_width + 1e-9)    # pixels exactly half_width away are rays on both
                                                   # sides of a ray (float rounding broke the mirror)
    ray = np.floor(shifted / step).astype(int) % SPIKES          # the ray `off` was measured from
    return {"core": core, "rays": rays, "glow": ~core & ~rays, "ray": ray}


FAR_PIECES = ("core", "annulus", "rays_far", "glow_far")


def far_masks(K, half_width=RAY_HALF_WIDTH_PX):
    """The coma split's pieces with the ray band split at ANNULUS_PX (decision log 2026-11-17,
    ray follow-ups): core unchanged; annulus = every pixel outside the core with r < ANNULUS_PX,
    whatever the band says; rays_far, glow_far = the band and the glow at r >= ANNULUS_PX."""
    m = coma_masks(K, half_width)
    c = K // 2
    y, x = np.mgrid[:K, :K]
    near = np.hypot(x - c, y - c) < ANNULUS_PX
    return {"core": m["core"], "annulus": ~m["core"] & near, "rays_far": m["rays"] & ~near,
            "glow_far": m["glow"] & ~near, "ray": m["ray"]}


def radial_segment(K, edges=RAY_PROFILE_BINS_PX):
    """Each pixel's radial segment about the centre pixel: -1 inside edges[0], then i for
    edges[i] <= r < edges[i + 1], the last segment open to the kernel's corners."""
    c = K // 2
    y, x = np.mgrid[:K, :K]
    return np.digitize(np.hypot(x - c, y - c), edges) - 1


def restore_ray_light(kernel, exact, rays, groups):
    """Per group label g and channel, scale the kernel's pixels in rays & (groups == g) so their
    sum is the exact kernel's there, keeping the kernel's shape inside each group. With the ray
    index as the label, each ray gets its exact brightness (the best a per-ray, per-channel
    weight could do); with ray x segments, its exact coarse falloff too. A group the kernel
    holds no light in stays as is."""
    out = kernel.copy()
    for g in np.unique(groups[rays]):
        sel = rays & (groups == g)
        have, want = out[sel].sum(axis=0), exact[sel].sum(axis=0)
        out[sel] = out[sel] * np.divide(want, have, out=np.ones_like(want), where=have > 0)
    return out


def masked_psf(sym, exact, mask):
    """The symmetric kernel inside the mask (K x K bool), the exact one outside."""
    return np.where(mask[..., None], sym, exact)


def cell_gauss(d, mu, sigma, s):
    """A unit-area Gaussian across the ray, integrated over a 1/s px cell: a 0.3 px line on
    0.2 px cells would alias if point-sampled."""
    h, z = 0.5 / s, np.sqrt(2) * sigma
    return 0.5 * (erf((d + h - mu) / z) - erf((d - h - mu) / z))


def cell_cauchy(d, mu, gamma, s):
    """A unit-area Lorentzian (half-width gamma) integrated over a 1/s px cell: tails ~ 1/d^2,
    as a diffraction spike's sinc^2 cross-section."""
    h = 0.5 / s
    return (np.arctan((d + h - mu) / gamma) - np.arctan((d - h - mu) / gamma)) / np.pi


def xs_bounds(variant):
    """(init, lower, upper) parameter vectors of a cross-section class."""
    parts = ([XS_LINE] + ([XS_SKIRT] if "skirt" in variant else [])
             + ([XS_GLOW] if variant.endswith("_bg") else []))
    return tuple(np.concatenate([np.asarray(p[i], float) for p in parts]) for i in range(3))


def cross_profile(p, d, x, variant, s):
    """The class's unnormalised cross-section at signed distance d (px) from the ray line, at
    x = ln(r / XS_R0_PX): a Gaussian line, plus a Gaussian ("skirt") or Lorentzian ("cskirt")
    skirt, plus a flat glow ("_bg"). Every parameter is linear in x."""
    def lin(i):
        return p[i] + p[i + 1] * x
    out = cell_gauss(d, lin(0), np.exp(lin(2)), s)
    i = 4
    if "skirt" in variant:
        skirt = cell_cauchy if "cskirt" in variant else cell_gauss
        out = out + np.exp(lin(i)) * skirt(d, lin(i + 2), np.exp(lin(i + 4)), s)
        i += 6
    if variant.endswith("_bg"):
        out = out + np.exp(lin(i)) / s
    return out


def fit_cross(t, d, r, variant, s):
    """One (ray, channel) wedge: the fine cells' exact values t at distance d across the ray and
    radius r. Returns the class's best reconstruction and its parameters (None for xs_exact).
    Each 1 px radius bin keeps its exact light (the along-ray profile is exact), so only the
    cross-section is tested. Fitted by soft-L1 on the log error, a smooth stand-in for the
    absolute log loss. xs_exact: the exact profile averaged per radius bin and 1/s px of d."""
    eps = XS_FIT_EPS / s ** 2
    _, inv = np.unique(np.floor(r).astype(int), return_inverse=True)
    light = np.bincount(inv, weights=t)

    def recon(shape):
        z = np.bincount(inv, weights=shape)
        return light[inv] * shape / np.where(z > 0, z, 1.0)[inv]

    if variant == "xs_exact":
        share = t / np.where(light > 0, light, 1.0)[inv]
        _, g = np.unique(inv * 10000 + np.rint(d * s).astype(int), return_inverse=True)
        return recon((np.bincount(g, weights=share) / np.bincount(g))[g]), None
    x = np.log(r / XS_R0_PX)
    p0, lo, hi = xs_bounds(variant)
    fit = least_squares(lambda p: np.log(recon(cross_profile(p, d, x, variant, s)) + eps)
                        - np.log(t + eps), p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1)
    return recon(cross_profile(fit.x, d, x, variant, s)), fit.x


def xs_geometry(K, s):
    """Per fine cell of the K*s grid, about the source (pixel K // 2's centre): radius r (px),
    nearest ray, signed perpendicular distance d to it (coma_masks' angles), and
    far = r >= ANNULUS_PX."""
    c = K // 2
    q = (np.arange(K * s) + 0.5) / s - 0.5 - c          # cell centres in pixels
    x, y = np.meshgrid(q, q)                            # y down, as coma_masks
    step = 2 * np.pi / SPIKES
    shifted = np.arctan2(y, x) + step / 2
    r = np.hypot(x, y)
    return {"far": r >= ANNULUS_PX, "ray": np.floor(shifted / step).astype(int) % SPIKES,
            "d": r * np.sin(shifted % step - step / 2), "r": r}


def separable_rays(job):
    """(fine PSF, variant) -> (pixel kernel, {(ray, channel): parameters}): the far wedges
    rebuilt by the class, the rest exact. One job per PSF, for a process pool."""
    fine, variant = job
    fine = fine.astype(np.float64)
    geo = xs_geometry(fine.shape[0] // OVERSAMPLE, OVERSAMPLE)
    out, params = np.zeros(fine.shape), {}
    for k in range(SPIKES):
        sel = geo["far"] & (geo["ray"] == k)
        for ch in range(3):
            out[..., ch][sel], params[(k, ch)] = fit_cross(fine[..., ch][sel], geo["d"][sel],
                                                           geo["r"][sel], variant, OVERSAMPLE)
    return bin_fine(np.where(geo["far"][..., None], out, fine), OVERSAMPLE), params


def xs_fit_summary(params, records):
    """Medians per H bin of xs_line_cskirt_bg's fits at r = XS_R0_PX: the widths and weights an
    engine term would need."""
    rows = {b: [] for b in BINS}
    for rec, fits in zip(records, params):
        b = "centre" if rec["H"] < 1 / 3 else "mid" if rec["H"] < 2 / 3 else "edge"
        rows[b] += list(fits.values())
    keys = {"line_sigma_px": lambda p: np.exp(p[2]), "line_sigma_slope": lambda p: p[3],
            "skirt_weight": lambda p: np.exp(p[4]), "skirt_gamma_px": lambda p: np.exp(p[8]),
            "glow_per_px": lambda p: np.exp(p[10])}
    return {b: {k: float(np.median([f(p) for p in ps])) for k, f in keys.items()}
            for b, ps in rows.items()}


# Coma split (decision log 2026-11-17): oracle -> (pieces symmetrised, ray band half-width)
COMA_SPLIT = {"sym_core": (("core",), RAY_HALF_WIDTH_PX),
              "sym_rays": (("rays",), RAY_HALF_WIDTH_PX),
              "sym_glow": (("glow",), RAY_HALF_WIDTH_PX),
              "sym_not_core": (("rays", "glow"), RAY_HALF_WIDTH_PX),
              "sym_not_rays": (("core", "glow"), RAY_HALF_WIDTH_PX),
              "sym_not_glow": (("core", "rays"), RAY_HALF_WIDTH_PX),
              "sym_rays_w3": (("rays",), ROBUST_HALF_WIDTH_PX),
              "sym_glow_w3": (("glow",), ROBUST_HALF_WIDTH_PX)}


# Oracles of the oversampled bank: f(pixel PSF, fine PSF, s); run when the bank exists
FINE_ORACLES = {"identity_fine": lambda k, f, s: bin_fine(f, s),
                "aperture_fine": aperture_fine_psf}


ORACLES = {"identity": lambda k: k, "round": round_psf, "grey": grey_psf,
           "round_grey": lambda k: grey_psf(round_psf(k)),
           "round_median": round_median_psf,
           "round_median_grey": lambda k: round_median_psf(k, grey=True),
           "aperture": aperture_psf,
           "aperture_d7": aperture_d7_psf}
LABELS = {"identity": "Oracle: identity (control)",
          "round": "Oracle: round, ring mean (no spikes / coma)",
          "grey": "Oracle: grey (no colour)",
          "round_grey": "Oracle: round + grey, ring mean",
          "round_median": "Oracle: round, ring median (best round, abs log loss)",
          "round_median_grey": "Oracle: round + grey, ring median (best of Step 1's class, abs log loss)",
          "aperture": "Oracle: aperture-symmetric, 14-fold median (INVALID: control failed, decision log 2026-11-04)",
          "aperture_d7": "Oracle: polygon-symmetric, 7 rotations + mirror, round core (INVALID: control failed, interpolation floor, decision log 2026-11-04)",
          "identity_fine": "Oracle: identity from the 5x bank, summed to pixels (sampling control)",
          "aperture_fine": "Oracle: polygon-symmetric at 5x, round core (spikes kept, no coma)",
          "sym_core": "Coma split: core symmetrised (round core), rays and glow exact",
          "sym_rays": "Coma split: rays symmetrised (+-2 px band), core and glow exact",
          "sym_glow": "Coma split: glow symmetrised, core and rays (+-2 px) exact",
          "sym_not_core": "Coma split: rays and glow symmetrised, core exact",
          "sym_not_rays": "Coma split: core and glow symmetrised, rays exact",
          "sym_not_glow": "Coma split: core and rays symmetrised, glow exact",
          "sym_rays_w3": "Coma split: rays symmetrised (+-3 px band, robustness)",
          "sym_glow_w3": "Coma split: glow symmetrised (+-3 px ray band, robustness)",
          "sym_annulus": "Ray follow-up: 5.25-9 px annulus symmetrised, the rest exact",
          "sym_rays_far": "Ray follow-up: rays beyond 9 px symmetrised, the rest exact",
          "sym_glow_far": "Ray follow-up: glow beyond 9 px symmetrised, the rest exact",
          "rays_far_bright": "Ray follow-up: rays beyond 9 px symmetric in shape, exact in light per ray",
          "rays_far_profile": "Ray profile: rays beyond 9 px symmetric across, exact in light per ray and segment",
          "xs_exact": "Ray cross-section: exact profile per 1 px of radius and channel (floor)",
          "xs_line": "Ray cross-section: one Gaussian line per ray and channel",
          "xs_line_bg": "Ray cross-section: line + flat glow",
          "xs_line_skirt": "Ray cross-section: line + Gaussian skirt",
          "xs_line_cskirt": "Ray cross-section: line + Lorentzian skirt",
          "xs_line_cskirt_bg": "Ray cross-section: line + Lorentzian skirt + flat glow"}
OVERSAMPLE = 5                      # aperture_fine: psf_bank.py --oversample
BINS = ("centre", "mid", "edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--aperture-order", type=int, choices=[1, 3], default=1,
                    help="aperture oracle's rotation interpolation: 1 bilinear; 3 cubic, the "
                         "pre-registered fallback if the H ~ 0 control fails (output *_cubic)")
    args = ap.parse_args()
    suffix = "" if args.aperture_order == 1 else "_cubic"
    oracles = {**ORACLES, "aperture": lambda k: aperture_psf(k, args.aperture_order)}

    with np.load(ROOT / psf_bank_path(args.run)) as z:
        bank = {k: z[k] for k in z.files}
    phase0, phase0_fit, gap = strongest_baselines(args.run)
    size, ppd = gap["window_px"], gap["flip_ppd"]
    assert gap["eps"] == EPS, "log eps differs from the gap study's"
    img = read_hdr(str(ROOT / "data" / "inputs" / "pointgrid.exr"))[..., :3]
    ref = read_hdr(str(ROOT / REFS[args.run] / "pointgrid_ref_exact.exr"))[..., :3]
    test = bank["split"] == "test"
    positions = [tuple(int(v) for v in p) for p in bank["pos_px"][test]]
    points = [(x, y, img[y, x].astype(np.float64)) for x, y in positions]
    psfs = bank["psf"][test].astype(np.float64)
    fine_path = ROOT / psf_bank_path(args.run, OVERSAMPLE)
    assert fine_path.exists(), f"{fine_path} missing: run psf_bank.py --oversample {OVERSAMPLE} --only test"
    with np.load(fine_path) as z:
        assert int(z["oversample"]) == OVERSAMPLE
        fine_test = z["split"] == "test"
        assert [tuple(int(v) for v in p) for p in z["pos_px"][fine_test]] == positions, (
            "the oversampled bank's test positions differ from the pixel bank's")
        fines = z["psf"][fine_test]                  # float32, (35, K·s, K·s, 3)

    results, fine_kernels = {}, {}

    def score(name, kernels):
        out, _ = splat(img.shape, points, kernels)
        records = window_records(out, ref, positions, size, ppd)
        results[name] = {"summary": summarize(records), "records": records}
        print(f"{name}: FLIP " + " / ".join(f"{results[name]['summary'][b]['flip']:.4f}"
                                            for b in BINS), flush=True)

    for name, edit in {**oracles, **FINE_ORACLES}.items():
        if name in FINE_ORACLES:
            kernels = [edit(k, f.astype(np.float64), OVERSAMPLE) for k, f in zip(psfs, fines)]
            fine_kernels[name] = kernels
        else:
            kernels = [edit(k) for k in psfs]
        score(name, kernels)

    # Coma split (decision log 2026-11-17): every masked oracle is derived from the two fine
    # kernel lists, so the rotations run once
    sym, exact = fine_kernels["aperture_fine"], fine_kernels["identity_fine"]
    K = psfs.shape[1]
    masks = {w: coma_masks(K, w) for w in (RAY_HALF_WIDTH_PX, ROBUST_HALF_WIDTH_PX)}
    for m in masks.values():                     # union control (0a): a partition ...
        assert (sum(m[p].astype(int) for p in PIECES) == 1).all(), "coma masks don't partition the PSF"
    for s_k, e_k in zip(sym, exact):             # ... whose pieces, applied in turn, rebuild aperture_fine
        k = e_k
        for p in PIECES:
            k = masked_psf(s_k, k, masks[RAY_HALF_WIDTH_PX][p])
        assert np.array_equal(k, s_k), "the coma pieces don't rebuild aperture_fine"
    for name, (pieces, width) in COMA_SPLIT.items():
        mask = np.logical_or.reduce([masks[width][p] for p in pieces])
        score(name, [masked_psf(s_k, e_k, mask) for s_k, e_k in zip(sym, exact)])

    # Ray follow-ups (decision log 2026-11-17, amendment after the coma split)
    fm = far_masks(K)
    assert (sum(fm[p].astype(int) for p in FAR_PIECES) == 1).all(), "far masks don't partition the PSF"
    for s_k, e_k in zip(sym, exact):
        k = e_k
        for p in FAR_PIECES:
            k = masked_psf(s_k, k, fm[p])
        assert np.array_equal(k, s_k), "the far pieces don't rebuild aperture_fine"
    for p in FAR_PIECES[1:]:
        score(f"sym_{p}", [masked_psf(s_k, e_k, fm[p]) for s_k, e_k in zip(sym, exact)])
    score("rays_far_bright", [restore_ray_light(masked_psf(s_k, e_k, fm["rays_far"]), e_k,
                                                fm["rays_far"], fm["ray"])
                              for s_k, e_k in zip(sym, exact)])
    # Ray profile oracle (decision log 2026-11-17, second amendment): light per ray and segment
    segment_groups = fm["ray"] * len(RAY_PROFILE_BINS_PX) + radial_segment(K)
    score("rays_far_profile", [restore_ray_light(masked_psf(s_k, e_k, fm["rays_far"]), e_k,
                                                 fm["rays_far"], segment_groups)
                               for s_k, e_k in zip(sym, exact)])
    # Ray cross-section oracle (decision log 2026-11-26): one process per PSF
    xs_params = {}
    with ProcessPoolExecutor() as pool:
        for variant in XS_VARIANTS:
            done = list(pool.map(separable_rays, [(f, variant) for f in fines]))
            xs_params[variant] = [p for _, p in done]
            score(variant, [k for k, _ in done])

    study, tag = STEP1
    step1 = json.loads((Path(run_dir(EXPERIMENT, args.run, "eval", study)) /
                        f"{tag}_eval.json").read_text())["results"]["trained"]["summary"]
    control = all(results["identity"]["summary"][b]["flip"] < CONTROL_FLIP for b in BINS)

    def flip(summary, b):
        return summary[b]["flip"]

    rg = results["round_median_grey"]["summary"]                   # amended: decision log 2026-11-04
    budget = {b: {"profile": flip(step1, b) - flip(rg, b),
                  "roundness": flip(rg, b) - flip(results["grey"]["summary"], b),
                  "colour": flip(rg, b) - flip(results["round_median"]["summary"], b)}
              for b in BINS}
    target = phase0["fft_centre"]["edge"]["flip"]
    answers = {"control_passes": control,
               "median_below_step1": all(flip(rg, b) < flip(step1, b) for b in BINS),
               "round_grey_reaches_target": flip(rg, "edge") < target,
               "largest_at_edge": max(budget["edge"], key=budget["edge"].get)}

    # Spikes vs coma (decision log 2026-11-04): from aperture_fine; the invalid oracles'
    # controls kept for the record
    def centre_control(name):
        rec = min(results[name]["records"], key=lambda r: r["H"])
        return {"H": rec["H"], "flip": rec["flip"], "passes": rec["flip"] < APERTURE_CONTROL_FLIP}

    ap_s, rm = results["aperture_fine"]["summary"], results["round_median"]["summary"]
    split = {b: {"spike_gain": flip(rm, b) - flip(ap_s, b), "coma_remainder": flip(ap_s, b)}
             for b in BINS}
    answers |= {"aperture_control_invalid_14fold": centre_control("aperture"),
                "aperture_control_invalid_d7": centre_control("aperture_d7"),
                "sampling_control": all(results["identity_fine"]["summary"][b]["flip"] < CONTROL_FLIP
                                        for b in BINS),
                "aperture_control": centre_control("aperture_fine"),
                "aperture_below_round_median": all(flip(ap_s, b) < flip(rm, b) for b in BINS),
                "aperture_reaches_target": flip(ap_s, "edge") < target,
                "first": ("spikes" if split["edge"]["spike_gain"] >= split["edge"]["coma_remainder"]
                          else "field-aligned offsets")}

    # Coma split (decision log 2026-11-17)
    one = {p: results[f"sym_{p}"]["summary"] for p in PIECES}
    valid = all(flip(one[p], b) <= flip(ap_s, b) + VALIDITY_SLACK for p in PIECES for b in BINS)
    picked = max(PIECES, key=lambda p: flip(one[p], "edge")) if valid else None
    w2 = flip(one["rays"], "edge") - flip(one["glow"], "edge")
    w3 = (flip(results["sym_rays_w3"]["summary"], "edge")
          - flip(results["sym_glow_w3"]["summary"], "edge"))
    answers["coma_split"] = {
        "centre_controls": {name: centre_control(name) for name in COMA_SPLIT},
        "valid": valid,
        "picked": picked,
        "picked_alone_reaches_target": (flip(results[f"sym_not_{picked}"]["summary"], "edge") < target
                                        if picked else None),
        "rays_vs_glow_order_robust": bool(np.sign(w2) == np.sign(w3))}

    # Ray follow-ups (decision log 2026-11-17, amendment)
    far_names = ("sym_annulus", "sym_rays_far", "sym_glow_far", "rays_far_bright")
    fs = {name: results[name]["summary"] for name in far_names}
    bounds = {"sym_annulus": "sym_rays", "sym_rays_far": "sym_rays", "sym_glow_far": "sym_glow",
              "rays_far_bright": "sym_rays_far"}
    validity = {name: all(flip(fs[name], b) <= flip(results[bound]["summary"], b) + VALIDITY_SLACK
                          for b in BINS)
                for name, bound in bounds.items()}
    where = ("rays_far" if flip(fs["sym_rays_far"], "edge") >= flip(fs["sym_annulus"], "edge")
             else "annulus")
    floor = centre_control("sym_rays_far")["flip"]
    gain = flip(fs["sym_rays_far"], "edge") - flip(fs["rays_far_bright"], "edge")
    remainder = flip(fs["rays_far_bright"], "edge") - floor
    answers["ray_followups"] = {
        "centre_controls": {name: centre_control(name) for name in far_names},
        "validity": validity,
        "where": where,
        "floor": floor, "brightness_gain": gain, "shape_remainder": remainder,
        "what": ("brightness" if gain >= remainder else "shape") if where == "rays_far" else None}

    # Ray profile oracle (decision log 2026-11-17, second amendment)
    prof = results["rays_far_profile"]["summary"]
    along = flip(fs["rays_far_bright"], "edge") - flip(prof, "edge")
    across = flip(prof, "edge") - floor
    answers["ray_profile"] = {
        "centre_control": centre_control("rays_far_profile"),
        "valid": all(flip(prof, b) <= flip(fs["rays_far_bright"], b) + VALIDITY_SLACK for b in BINS),
        "along_gain": along, "across_remainder": across,
        "first": "along" if along >= across else "across",
        "close": abs(along - across) < 0.005}

    # Ray cross-section oracle (decision log 2026-11-26)
    current = json.loads((Path(run_dir(EXPERIMENT, args.run, "eval", CURRENT[0])) /
                          f"{CURRENT[1]}_eval.json").read_text())["results"]["trained"]["summary"]
    xs = {v: results[v]["summary"] for v in XS_VARIANTS}
    best = min(XS_VARIANTS[1:], key=lambda v: flip(xs[v], "centre") + flip(xs[v], "edge"))
    line_gap = {b: flip(current, b) - flip(xs["xs_line"], b) for b in BINS}
    answers["ray_cross_section"] = {
        "control": all(flip(xs["xs_exact"], b) < APERTURE_CONTROL_FLIP for b in BINS),
        "current": {"tag": CURRENT[1], "summary": current},
        "line_gap": line_gap,
        "gains_over_line": {v: {b: flip(xs["xs_line"], b) - flip(xs[v], b) for b in BINS}
                            for v in XS_VARIANTS[2:]},
        "best": best,
        "best_remainder": {b: flip(xs[best], b) - flip(xs["xs_exact"], b) for b in BINS},
        "enough": all(flip(xs[best], b) - flip(xs["xs_exact"], b) <= XS_ENOUGH
                      for b in ("centre", "edge")),
        "first": {b: "line" if line_gap[b] >= flip(xs["xs_line"], b) - flip(xs[best], b)
                  else "skirt / glow" for b in BINS},
        "fits": xs_fit_summary(xs_params["xs_line_cskirt_bg"], results["identity"]["records"])}

    rows = [(f"{COMPARE[b]} ({phase0_fit[b]} fit)", phase0[b]) for b in COMPARE]
    rows += [("Step 1 baseline: MLP(uv)", step1)]
    rows += [(LABELS[name], r["summary"]) for name, r in results.items()]
    budget_lines = "\n".join(
        f"| {b} | {budget[b]['profile']:+.4f} | {budget[b]['roundness']:+.4f} | "
        f"{budget[b]['colour']:+.4f} |" for b in BINS)

    def out_dir(stage):
        return Path(run_dir(EXPERIMENT, args.run, stage, STUDY))

    split_lines = "\n".join(
        f"| {b} | {split[b]['spike_gain']:+.4f} | {split[b]['coma_remainder']:.4f} |" for b in BINS)
    c = answers["aperture_control"]

    (out_dir("eval") / f"ceiling{suffix}_eval.json").write_text(json.dumps(
        {"eps": EPS, "window_px": size, "flip_ppd": ppd, "aperture_order": args.aperture_order,
         "step1": {"tag": tag, "summary": step1}, "phase0": phase0, "phase0_fit": phase0_fit,
         "budget": budget, "split": split, "answers": answers, "results": results}, indent=2))
    report = (f"# Phase 2, {STUDY}: ceiling analysis on the point grid ({args.run})\n\n"
              f"{table(rows)}\n\n"
              f"Each oracle is the exact reference with one feature removed from every PSF. The "
              f"ring-mean oracles aren't ceilings (see the module docstring); the ring-median "
              f"ones are the best round kernels in the absolute log loss. Medians over the 35 "
              f"point-grid windows ({size} px), as train_points.py.\n\n"
              f"Median FLIP differences, from the ring-median oracles (a ranking: FLIP isn't "
              f"additive):\n\n"
              f"| Bin | profile: Step 1 - round+grey | roundness: round+grey - grey | "
              f"colour: round+grey - round |\n|---|---|---|---|\n{budget_lines}\n\n"
              f"Pre-registered questions (decision log 2026-11-04, amended):\n"
              f"- control: identity FLIP < {CONTROL_FLIP:g} in every bin: "
              f"**{answers['control_passes']}**\n"
              f"- validity: ring-median round+grey below Step 1 in FLIP in every bin: "
              f"**{answers['median_below_step1']}**\n"
              f"- a round, grey kernel can reach the target (round+grey edge FLIP < "
              f"{target:.3f}): **{answers['round_grey_reaches_target']}**\n"
              f"- largest difference at the edge, which picks the next ablation: "
              f"**{answers['largest_at_edge']}**\n\n"
              f"Spikes vs coma, from the oversampled polygon-symmetric oracle aperture_fine (the invalid 14-fold "
              f"oracle used interpolation order {args.aperture_order}):\n\n"
              f"| Bin | spike gain: round median - aperture_fine | coma remainder: aperture_fine |\n"
              f"|---|---|---|\n{split_lines}\n\n"
              f"Pre-registered questions (decision log 2026-11-04, oversampled polygon-symmetric "
              f"oracle):\n"
              f"- sampling control: identity_fine FLIP < {CONTROL_FLIP:g} in every bin: "
              f"**{answers['sampling_control']}**\n"
              f"- control: aperture_fine FLIP at the source nearest H = 0 (H = {c['H']:.3f}) < "
              f"{APERTURE_CONTROL_FLIP:g}: {c['flip']:.4f}, **{c['passes']}**\n"
              f"- validity: aperture_fine below round median in every bin: "
              f"**{answers['aperture_below_round_median']}**\n"
              f"- a centred spike term could reach the target (aperture_fine edge FLIP < "
              f"{target:.3f}): **{answers['aperture_reaches_target']}**\n"
              f"- built first: **{answers['first']}**\n")

    cs = answers["coma_split"]
    builds = {"glow": "field-aligned radial glow", "rays": "per-ray weights by angle to the field direction",
              "core": "a longer radial offset pair", None: "not ranked (validity failed)"}
    split_rows = "\n".join(
        f"| `{name}` | {' + '.join(pieces)} | {w:g} px | "
        + " / ".join(f"{flip(results[name]['summary'], b):.4f}" for b in BINS)
        + f" | {results[name]['summary']['edge']['energy'] * 100:+.2f} % | "
        f"{cs['centre_controls'][name]['flip']:.4f} |"
        for name, (pieces, w) in COMA_SPLIT.items())
    reach = cs["picked_alone_reaches_target"]
    report += (f"\nComa split (decision log 2026-11-17): aperture_fine ({' / '.join(f'{flip(ap_s, b):.4f}' for b in BINS)}) "
               f"applied only to the named pieces, the exact PSF elsewhere. Pieces: core = r <= "
               f"{CORE_PX + 0.25:g} px (aperture_fine's round core), rays = within the band of a ray line, "
               f"glow = the rest.\n\n"
               f"| Oracle | Symmetrised | Ray band | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 |\n"
               f"|---|---|---|---|---|---|\n{split_rows}\n\n"
               f"Pre-registered questions (decision log 2026-11-17):\n"
               f"- centre control: every masked oracle < {APERTURE_CONTROL_FLIP:g} at the source nearest "
               f"H = 0: **{all(v['passes'] for v in cs['centre_controls'].values())}**\n"
               f"- validity: each one-piece oracle <= aperture_fine + {VALIDITY_SLACK:g} in every bin: "
               f"**{cs['valid']}**\n"
               f"- largest one-piece edge FLIP, built first: **{cs['picked']}** ({builds[cs['picked']]})\n"
               f"- the picked piece alone could reach the target (sym_not_{cs['picked']} edge FLIP < "
               f"{target:.4f}): **{reach}**\n"
               f"- rays vs glow order the same with the +-3 px band: **{cs['rays_vs_glow_order_robust']}**\n")

    rf = answers["ray_followups"]
    pieces = {"sym_annulus": f"annulus (5.25 <= r < {ANNULUS_PX:g} px)",
              "sym_rays_far": f"rays (+-2 px), r >= {ANNULUS_PX:g} px",
              "sym_glow_far": f"glow, r >= {ANNULUS_PX:g} px",
              "rays_far_bright": f"rays r >= {ANNULUS_PX:g} px, shape only (light restored per ray)"}
    far_rows = "\n".join(
        f"| `{name}` | {pieces[name]} | "
        + " / ".join(f"{flip(fs[name], b):.4f}" for b in BINS)
        + f" | {fs[name]['edge']['energy'] * 100:+.2f} % | {rf['centre_controls'][name]['flip']:.4f} | "
        f"{bounds[name]} | **{rf['validity'][name]}** |"
        for name in far_names)
    whats = {"brightness": "per-ray weights by angle to the field direction (SpikeMLP)",
             "shape": "the rays' shape: an angle-dependent per-ray blur", None: "not asked (the annulus comes first)"}
    report += (f"\nRay follow-ups (decision log 2026-11-17, an amendment after the coma split): the ray band "
               f"split at {ANNULUS_PX:g} px, and the far rays' brightness restored.\n\n"
               f"| Oracle | Symmetrised | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 | Bounded by | Valid |\n"
               f"|---|---|---|---|---|---|---|\n{far_rows}\n\n"
               f"Pre-registered questions (decision log 2026-11-17, ray follow-ups):\n"
               f"- centre control: every new oracle < {APERTURE_CONTROL_FLIP:g} at the source nearest H = 0: "
               f"**{all(v['passes'] for v in rf['centre_controls'].values())}**\n"
               f"- validity: each oracle <= its bound + {VALIDITY_SLACK:g} in every bin: "
               f"**{all(rf['validity'].values())}**\n"
               f"- where (sym_rays_far vs sym_annulus at the edge): **{rf['where']}**\n"
               f"- what: brightness gain (sym_rays_far - rays_far_bright, edge) {rf['brightness_gain']:+.4f} vs "
               f"shape remainder (rays_far_bright edge - floor {rf['floor']:.4f}) {rf['shape_remainder']:+.4f}: "
               f"**{rf['what']}** ({whats[rf['what']]})\n")

    rp = answers["ray_profile"]
    firsts = {"along": "along the ray: per-ray weights and profiles by angle to the field direction",
              "across": "across the ray: an angle-dependent per-ray blur"}
    report += (f"\nRay profile oracle (decision log 2026-11-17, a second amendment): the far rays' light "
               f"restored per ray and radial segment ({' / '.join(f'{e:g}' for e in RAY_PROFILE_BINS_PX)} px).\n\n"
               f"| Oracle | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 |\n|---|---|---|---|\n"
               f"| `rays_far_profile` | " + " / ".join(f"{flip(prof, b):.4f}" for b in BINS)
               + f" | {prof['edge']['energy'] * 100:+.2f} % | {rp['centre_control']['flip']:.4f} |\n\n"
               f"Pre-registered questions (decision log 2026-11-17, ray profile oracle):\n"
               f"- centre control: < {APERTURE_CONTROL_FLIP:g} at the source nearest H = 0: "
               f"**{rp['centre_control']['passes']}**\n"
               f"- validity: rays_far_profile <= rays_far_bright + {VALIDITY_SLACK:g} in every bin: "
               f"**{rp['valid']}**\n"
               f"- along-ray gain (rays_far_bright - rays_far_profile, edge) {along:+.4f} vs across-ray "
               f"remainder (rays_far_profile edge - floor {floor:.4f}) {across:+.4f}: **{rp['first']}** "
               f"({firsts[rp['first']]}); within 0.005: **{rp['close']}**\n")

    xc = answers["ray_cross_section"]
    xs_rows = "\n".join(f"| `{v}` | " + " / ".join(f"{flip(xs[v], b):.4f}" for b in BINS)
                        + f" | {xs[v]['edge']['energy'] * 100:+.2f} % |" for v in XS_VARIANTS)
    report += (f"\nRay cross-section oracle (decision log 2026-11-26): far rays (r >= {ANNULUS_PX:g} px) "
               f"rebuilt per ray and channel from a cross-section class with parameters linear in "
               f"ln(r / {XS_R0_PX:g} px), the light per 1 px of radius exact.\n\n"
               f"| Oracle | FLIP c / m / e | Energy, edge |\n|---|---|---|\n{xs_rows}\n"
               f"| `{CURRENT[0]}` (the model) | " + " / ".join(f"{flip(current, b):.4f}" for b in BINS)
               + " | |\n\n"
               f"Pre-registered questions (decision log 2026-11-26, ray cross-section oracle):\n"
               f"- control: xs_exact < {APERTURE_CONTROL_FLIP:g} in every bin: **{xc['control']}**\n"
               f"- line gap (model - xs_line): "
               + " / ".join(f"{xc['line_gap'][b]:+.4f}" for b in BINS) + "\n"
               f"- best class: **{xc['best']}**, remainder over the floor "
               + " / ".join(f"{xc['best_remainder'][b]:+.4f}" for b in BINS)
               + f"; enough (<= {XS_ENOUGH:g} at centre and edge): **{xc['enough']}**\n"
               f"- first fix: " + ", ".join(f"{b} **{xc['first'][b]}**" for b in BINS) + "\n"
               f"- fits at r = {XS_R0_PX:g} px (xs_line_cskirt_bg, medians): "
               + "; ".join(f"{b}: " + ", ".join(f"{k} {v:.3g}" for k, v in xc["fits"][b].items())
                           for b in BINS) + "\n")
    (out_dir("report") / f"ceiling{suffix}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
