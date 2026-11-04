"""Figures of one Phase 2 study: how its runs look next to the exact lens, around point sources
at the centre, mid-field and the corner of the frame.

Every model comes from the crops train_points.py saves with each run (eval/*_views.npz: output
around the VIEW_H sources and each level's blend map, both from the Slang model), so no figure
re-runs an architecture that slang/ no longer holds. The reference and Phase 0's strongest
tuned baselines (FFT with the centre PSF, fitted pyramid bloom; train_points.strongest_baselines)
are rendered here on the same point-grid frame.

    uv run python experiments/03_train_network/views.py --study radial_input

Writes outputs/03_train_network/<run>/<study>/report/views/:
    1_look.png            tonemapped, as a viewer would see it
    2_log_luminance.png   7 decades of luminance: core, halo, spikes
    3_error_ratio.png     log10((out + eps) / (ref + eps)), as the loss sees it
    4_radial_profile.png  mean luminance in rings around the source
    5_loss_curves.png     training and validation loss per run
    6_blend_maps.png      each level's learned blend t over the screen
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from neuralglare.eval import metrics as m
from neuralglare.io import display, luminance, read_hdr
from neuralglare.paths import REFS, RUNS, psf_path, run_dir
from neuralglare.reference.convolve import PSFSet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "01_gap_study"))
sys.path.insert(0, str(ROOT / "experiments" / "03_train_network"))
from evaluate_gap import load_baselines
from train_points import EPS, EXPERIMENT, VIEW_PX, view_crop

# Per study, the runs it compares: (study folder, run tag, label). The first is the reference
# point of the study (the current best when it ran); add an entry for every new ablation.
STUDY_RUNS = {
    "radial_input": [
        ("baseline", "_lerp_abs_fitted_lr0.0001", "Step 1 baseline: MLP(uv)"),
        ("radial_input", "_lerp_h_abs_fitted_lr0.0001", "Radial input: MLP(H)"),
    ],
}
BASELINES = {"fft_centre": "Phase 0: FFT, centre PSF", "pyramid": "Phase 0: pyramid (fitted)"}
EXPOSURE = 0.05                     # 1_look: Reinhard exposure, as the Step 1 views
DECADES = 7                         # 2_log_luminance: range below the reference peak
FLOOR = 1e-6                        # 4_radial_profile: luminance floor of the log axis

# Reference palette of the dataviz guide: surface, ink, categorical slots in fixed order
# (blue, orange, aqua, yellow) for the four compared models, diverging blue / grey / orange
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIVERGING = LinearSegmentedColormap.from_list("too_dark_bright", ["#eb6834", "#f0efec", "#2a78d6"])
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CONSTANT = 1e-3                     # 6_blend_maps: relative range below which a level is flat
OUTSIDE = "#c8c7c1"                 # 3_error_ratio: outside the reference PSF's support

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
                     "font.size": 9})


def load_runs(run, study):
    """Per compared run: label, its saved view crops and its eval record."""
    runs = []
    for folder, tag, label in STUDY_RUNS[study]:
        def path(stage, suffix, folder=folder, tag=tag):
            return Path(run_dir(EXPERIMENT, run, stage, folder)) / f"train_points{tag}{suffix}"
        runs.append({"label": label, "views": dict(np.load(path("eval", "_views.npz"))),
                     "eval": json.loads(path("eval", "_eval.json").read_text())})
    pos = runs[0]["views"]["pos"]
    assert all(np.array_equal(r["views"]["pos"], pos) for r in runs), "runs saw other sources"
    return runs


def baseline_crops(run, fit, pos):
    """Phase 0's strongest tuned baselines (the fit train_points chose) and the exact
    reference, cropped around the view sources exactly like the runs' outputs."""
    img = read_hdr(str(ROOT / "data" / "inputs" / "pointgrid.exr"))[..., :3].astype(np.float64)
    ref = read_hdr(str(ROOT / REFS[run] / "pointgrid_ref_exact.exr"))[..., :3]
    ps = PSFSet(psf_path(run))
    fits = run_dir("01_gap_study", run, "fits")
    crops = {"reference": np.stack([view_crop(ref, x, y) for x, y in pos])}
    for b in BASELINES:
        render, _ = load_baselines(fits, ps, m.fit_suffix(fit[b]))
        out = render[b](img)
        crops[b] = np.stack([view_crop(out, x, y) for x, y in pos])
    return crops, ps.K


def grid(rows, cols, title):
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols + 0.9, 2.15 * rows + 0.6),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def label_grid(axes, names, hs):
    for ax, name in zip(axes[0], names):
        ax.set_title(textwrap.fill(name, 22), fontsize=8)
    for ax, h in zip(axes[:, 0], hs):
        ax.set_ylabel(f"H = {h:.2f}", fontsize=9)


def log_lum(crop):
    return np.log10(np.maximum(luminance(crop), 1e-12))


def fig_look(columns, hs, path):
    fig, axes = grid(len(hs), len(columns), f"What it looks like (Reinhard, exposure {EXPOSURE}), "
                     f"{VIEW_PX} px around a source")
    for c, (_, crops) in enumerate(columns):
        for r in range(len(hs)):
            axes[r, c].imshow(display(crops[r], EXPOSURE), interpolation="nearest")
    label_grid(axes, [n for n, _ in columns], hs)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_log(columns, hs, path):
    vmax = max(log_lum(c).max() for c in columns[0][1])
    fig, axes = grid(len(hs), len(columns), "Log luminance: the whole dynamic range, incl. faint "
                     "halo and spikes")
    for c, (_, crops) in enumerate(columns):
        for r in range(len(hs)):
            im = axes[r, c].imshow(log_lum(crops[r]), cmap="magma", vmin=vmax - DECADES, vmax=vmax,
                                   interpolation="nearest")
    label_grid(axes, [n for n, _ in columns], hs)
    fig.colorbar(im, ax=axes, shrink=0.5, label=f"log10 luminance ({DECADES} decades)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_error(columns, hs, K, path):
    """Column 0 shows the reference; the others the log ratio to it. Grey: outside the
    reference PSF's K x K support, where the reference is 0 by construction."""
    ref = columns[0][1]
    vmax = max(log_lum(c).max() for c in ref)
    yy, xx = np.mgrid[:VIEW_PX, :VIEW_PX] - VIEW_PX // 2
    outside = (np.abs(yy) > K // 2) | (np.abs(xx) > K // 2)
    fig, axes = grid(len(hs), len(columns), f"Error as the loss sees it: log10((out + eps) / "
                     f"(ref + eps)), eps = {EPS:g}.\nGrey frame: outside the reference PSF. Black "
                     f"contour: output below -eps")
    for r in range(len(hs)):
        axes[r, 0].imshow(log_lum(ref[r]), cmap="magma", vmin=vmax - DECADES, vmax=vmax,
                          interpolation="nearest")
        lr = luminance(ref[r])
        for c, (_, crops) in enumerate(columns[1:], start=1):
            ratio = np.log10((luminance(crops[r]) + EPS) / (lr + EPS))
            rgba = DIVERGING(np.clip((ratio + 1) / 2, 0, 1))
            rgba[outside] = matplotlib.colors.to_rgba(OUTSIDE)
            axes[r, c].imshow(rgba, interpolation="nearest")
            negative = (crops[r] < -EPS).any(-1)      # below -eps: log(out + eps) undefined
            if negative.any():
                axes[r, c].contour(negative, levels=[0.5], colors=INK, linewidths=0.8)
    label_grid(axes, [n for n, _ in columns], hs)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, ax=axes, shrink=0.5,
                 label=f"log10 ratio (eps {EPS:g}): blue too bright, orange too dark")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def ring_profile(crop):
    """Mean luminance in 1 px rings around the crop centre (the source)."""
    yy, xx = np.mgrid[:VIEW_PX, :VIEW_PX] - VIEW_PX // 2
    ring = np.rint(np.hypot(yy, xx)).astype(int)
    lum = luminance(crop)
    radii = np.arange(VIEW_PX // 2)
    return radii, np.array([lum[ring == r].mean() for r in radii])


def fig_profile(columns, hs, path):
    fig, axes = plt.subplots(1, len(hs), figsize=(4.6 * len(hs), 3.8), squeeze=False)
    fig.suptitle("Radial profile: core (left of each plot) and halo (right)", fontsize=11)
    for r, ax in enumerate(axes[0]):
        for c, (name, crops) in enumerate(columns):
            radii, prof = ring_profile(crops[r])
            style = ({"color": INK, "lw": 2.4} if c == 0 else
                     {"color": SERIES[c - 1], "lw": 1.6})
            ax.plot(radii, np.maximum(prof, FLOOR), label=name, **style)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlim(0, VIEW_PX // 2)
        ax.set_title(f"H = {hs[r]:.2f}")
        ax.set_xlabel("distance from source (px)")
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("mean luminance in ring (log)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_loss(runs, path):
    """Training (solid) and validation (dashed) loss; one panel per loss kind, one y-scale each."""
    kinds = sorted({r["eval"]["args"]["loss"] for r in runs})
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 3.8), squeeze=False)
    names = {"abs": "absolute log error", "sq": "squared log error"}
    for ax, kind in zip(axes[0], kinds):
        for i, r in enumerate(runs):
            if r["eval"]["args"]["loss"] != kind:
                continue
            curve = r["eval"]["curve"]
            steps = [p["step"] for p in curve]
            color = SERIES[2 + i]
            ax.plot(steps, [p["train_loss"] for p in curve], color=color, lw=2, marker="o",
                    ms=4, label=f"{r['label']}: train")
            ax.plot(steps, [p["val_loss"] for p in curve], color=color, lw=2, ls="--",
                    label=f"{r['label']}: validation")
        ax.set_title(names[kind])
        ax.set_xlabel("step")
        ax.set_ylabel(names[kind])
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_blend(runs, path):
    """Each level's blend t over the screen, one row per run. Colour range per level, shared
    by the runs, so a row-to-row difference is a real one; the printed range shows how much
    a level varies at all. A level whose t varies by < CONSTANT relative is drawn flat: its
    variation is float noise (pass-through levels sit at t ~ 1e-4). Drawn 16:9 like the screen,
    so a function of H shows as circles."""
    n = runs[0]["views"]["blend"].shape[0]
    fig, axes = plt.subplots(len(runs), n, figsize=(2.3 * n + 0.6, 1.55 * len(runs) + 0.9),
                             squeeze=False)
    fig.suptitle("Learned blend t = sigmoid(MLP) per level over the screen (dark: more of the "
                 "level below; colour range per level)", fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n):
        lo = min(r["views"]["blend"][k].min() for r in runs)
        hi = max(r["views"]["blend"][k].max() for r in runs)
        flat = hi - lo < CONSTANT * hi
        for i, r in enumerate(runs):
            t = r["views"]["blend"][k]
            ax = axes[i, k]
            ax.imshow(np.zeros_like(t) if flat else t, cmap=BLUE_RAMP,
                      vmin=0 if flat else lo, vmax=1 if flat else hi, interpolation="nearest",
                      extent=(0, 16, 9, 0))
            ax.set_xlabel(f"t = {t.mean():.2g} (constant)" if flat else
                          f"t {t.min():.3g} – {t.max():.3g}", fontsize=7)
        axes[0, k].set_title("full resolution" if k == n - 1 else f"level {k}", fontsize=8)
    for i, r in enumerate(runs):
        axes[i, 0].set_ylabel(r["label"], fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--study", choices=sorted(STUDY_RUNS), required=True)
    args = ap.parse_args()

    runs = load_runs(args.run, args.study)
    pos, hs = runs[0]["views"]["pos"], runs[0]["views"]["h"]
    crops, K = baseline_crops(args.run, runs[0]["eval"]["phase0_fit"], pos)
    fit = runs[0]["eval"]["phase0_fit"]
    columns = ([("Reference (exact lens)", crops["reference"])]
               + [(f"{BASELINES[b]} ({fit[b]})", crops[b]) for b in BASELINES]
               + [(r["label"], r["views"]["out"]) for r in runs])

    out = Path(run_dir(EXPERIMENT, args.run, "report", args.study)) / "views"
    out.mkdir(exist_ok=True)
    fig_look(columns, hs, out / "1_look.png")
    fig_log(columns, hs, out / "2_log_luminance.png")
    fig_error(columns, hs, K, out / "3_error_ratio.png")
    fig_profile(columns, hs, out / "4_radial_profile.png")
    fig_loss(runs, out / "5_loss_curves.png")
    fig_blend(runs, out / "6_blend_maps.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
