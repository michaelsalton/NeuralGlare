"""Pre-fit of the per-ray cross filter (glare.slang RayMLP; decision log 2026-11-19).

The uv MLPs' field-dependent outputs stayed near their constant init (2026-11-17), so RayMLP starts
from a regression onto measured targets. Per training source and ray: the 3 non-negative taps that
best map the on-axis source's far ray band (ceiling.far_masks rays_far: r >= 9 px, +-2 px of the
ray line), shifted by -1 / 0 / +1 px across the ray, onto this source's (NNLS in relative error,
luminance, normalised to sum 1). RayMLP is then fitted to those taps through glare.slang
(train_glare.slang cross_mix): Adam on the light-weighted KL divergence, every other parameter
fixed. Training frames only: the validation frame selects between runs.

Per-ray brightness (BrightMLP, decision log 2026-11-19) is pre-fitted alike: per training source
and ray, the target gain is the ray's far-band luminance over the base's, normalised to mean 1
over the family (the gain is 7 softmax over a family, so it can only move light between them);
BrightMLP is regressed onto it through train_glare.slang ray_gain on the light-weighted squared
log error.

Imported lazily by train_points.py (ceiling.py imports train_points).
"""

import numpy as np
import torch
from ceiling import SPIKES, far_masks
from scipy import ndimage
from scipy.optimize import nnls

from neuralglare.io import luminance

CROSS_OFFSETS = (-1, 0, 1)      # px along n_k: glare.slang's tap m moves light by (m - 1) px
NNLS_FLOOR = 0.05               # relative-error weight 1 / (e + NNLS_FLOOR * max e), as the scratch fits
PREFIT_STEPS = 500
PREFIT_LR = 1e-2


def field_vector(pos, w, h):
    """glare.slang field_offset at a texel: H * d_hat, (x, y) with y down, corners at length 1."""
    p = np.asarray(pos, np.float64) + 0.5 - np.array([w, h]) / 2
    return p / np.hypot(w / 2, h / 2)


def ray_normal(k):
    """n_k = (-e_y, e_x) of glare.slang's RAY_DIRS[k] = (cos, sin)(k * 360 / 14), y down."""
    a = 2 * np.pi * k / SPIKES
    return np.array([-np.sin(a), np.cos(a)])


def cross_targets(bank, shape, sel):
    """Rows (ray, v_x, v_y, w_-1, w_0, w_+1, light) for every source in sel and ray, and the H of
    the on-axis base (the source in sel nearest the frame centre)."""
    h, w = shape[:2]
    sel = np.asarray(sel)
    base_i = sel[np.argmin(bank["field_height"][sel])]
    base = luminance(bank["psf"][base_i].astype(np.float64))
    fm = far_masks(base.shape[0])
    bands = [fm["rays_far"] & (fm["ray"] == k) for k in range(SPIKES)]
    copies = []
    for k in range(SPIKES):
        n = ray_normal(k)
        copies.append([ndimage.shift(base, (m * n[1], m * n[0]), order=1)[bands[k]]
                       for m in CROSS_OFFSETS])
    rows = []
    for i in sel:
        exact = luminance(bank["psf"][i].astype(np.float64))
        v = field_vector(bank["pos_px"][i], w, h)
        for k in range(SPIKES):
            e = exact[bands[k]]
            A = np.stack(copies[k], 1)
            wgt = 1.0 / (e + NNLS_FLOOR * e.max() + 1e-30)
            taps, _ = nnls(A * wgt[:, None], e * wgt)
            taps = taps / taps.sum() if taps.sum() > 0 else np.array([0.0, 1.0, 0.0])
            rows.append((k, *v, *taps, e.sum()))
    return np.array(rows), float(bank["field_height"][base_i])


def prefit_cross(module, params, offset, rows, steps=PREFIT_STEPS, lr=PREFIT_LR):
    """params with RayMLP (params[offset:]) regressed onto the rows' taps through cross_mix, and
    the light-weighted KL divergence (cross-entropy minus the targets' entropy; 0 = every target
    reproduced) before and after."""
    fixed = params[:offset].cuda()
    cross = params[offset:].clone().cuda().requires_grad_(True)
    light = rows[:, 6] / rows[:, 6].sum()
    t = rows[:, 3:6]
    entropy = float(-(light[:, None] * t * np.log(np.where(t > 0, t, 1.0))).sum())

    def cuda(a):
        return torch.tensor(a, dtype=torch.float32, device="cuda")

    groups = [(k, cuda(rows[sel, 1:3]), cuda(rows[sel, 3:6]), cuda(light[sel]))
              for k in range(SPIKES) for sel in [rows[:, 0] == k]]

    def kl():
        p = torch.cat([fixed, cross])
        ce = 0.0
        for k, v, target, wt in groups:
            a = module.cross_mix(v=v, params=p, offset=offset, ray=k)       # (n, 3)
            ce = ce - (wt[:, None] * target * torch.log(a + 1e-12)).sum()
        return ce - entropy

    optimizer = torch.optim.Adam([cross], lr=lr)
    start = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = kl()
        start = loss.item() if start is None else start
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        end = kl().item()
    return torch.cat([fixed, cross.detach()]).cpu(), (start, end)


def bright_targets(bank, shape, sel):
    """Rows (ray, v_x, v_y, gain, light) for every source in sel and ray: the ray's far-band light
    over the on-axis base's (the source in sel nearest the frame centre), normalised to mean 1 over
    the ray's family; light is the ray's far-band light. And the base's H."""
    h, w = shape[:2]
    sel = np.asarray(sel)
    base_i = sel[np.argmin(bank["field_height"][sel])]
    base = luminance(bank["psf"][base_i].astype(np.float64))
    fm = far_masks(base.shape[0])
    bands = [fm["rays_far"] & (fm["ray"] == k) for k in range(SPIKES)]
    base_light = np.array([base[b].sum() for b in bands])
    rows = []
    for i in sel:
        exact = luminance(bank["psf"][i].astype(np.float64))
        v = field_vector(bank["pos_px"][i], w, h)
        light = np.array([exact[b].sum() for b in bands])
        ratio = light / base_light
        for k in range(SPIKES):
            fam = ratio[k % 2::2]
            rows.append((k, *v, ratio[k] / fam.mean(), light[k]))
    return np.array(rows), float(bank["field_height"][base_i])


def prefit_bright(module, params, offset, rows, steps=PREFIT_STEPS, lr=PREFIT_LR):
    """params with BrightMLP (params[offset:]) regressed onto the rows' gains through ray_gain, and
    the light-weighted mean squared log gain error before and after."""
    fixed = params[:offset].cuda()
    bright = params[offset:].clone().cuda().requires_grad_(True)
    light = rows[:, 4] / rows[:, 4].sum()

    def cuda(a):
        return torch.tensor(a, dtype=torch.float32, device="cuda")

    groups = [(k, cuda(rows[sel, 1:3]), cuda(np.log(rows[sel, 3])), cuda(light[sel]))
              for k in range(SPIKES) for sel in [rows[:, 0] == k]]

    def err():
        p = torch.cat([fixed, bright])
        total = 0.0
        for k, v, target, wt in groups:
            g = module.ray_gain(v=v, params=p, offset=offset, ray=k)        # (n,)
            total = total + (wt * (torch.log(g) - target) ** 2).sum()
        return total

    optimizer = torch.optim.Adam([bright], lr=lr)
    start = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = err()
        start = loss.item() if start is None else start
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        end = err().item()
    return torch.cat([fixed, bright.detach()]).cpu(), (start, end)
