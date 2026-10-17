"""Gap study evaluation: how far each real-time baseline is from the spatially varying
reference, as a function of field height H.

Each baseline is re-rendered from its saved fit (fft_kernels.npz, pyramid_fit.json) and
compared with the reference in a window around every light source:
    log_mse    mean squared difference of log(x + eps): the loss the baselines were fitted to
    log_psnr   PSNR of the log images; peak = log(max ref + eps) - log(eps) of the frame
    flip       mean HDR-FLIP in the window. The FLIP map is computed once per full frame, so
               every window of a frame shares one exposure range
    energy     window energy relative to the reference
The point grid is the clean measurement: one source at 35 positions, so only H changes.
The HDRI crops check the trend on real content, at their measured H (manifest
H_measured).

Outputs (outputs/01_gap_study/): gap_eval.json (every window, summary, versions),
gap_table.md, gap_vs_H.png, motivation_<image>.png.

Usage:
    uv run python experiments/01_gap_study/evaluate_gap.py
"""

import argparse
import importlib.metadata
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from neuralglare.baselines.fft_bloom import fft_bloom
from neuralglare.baselines.pyramid_bloom import URPBloomParams, pyramid_bloom
from neuralglare.eval import metrics as m
from neuralglare.io import display, luminance, read_hdr
from neuralglare.reference.convolve import PSFSet

# Fixed categorical order (validated palette, light surface); colour follows the baseline.
BASELINES = ["fft_centre", "fft_best_fit", "pyramid", "no_bloom"]
LABELS = {"fft_centre": "FFT, centre PSF", "fft_best_fit": "FFT, best-fit PSF",
          "pyramid": "Pyramid bloom (fitted)", "no_bloom": "No bloom"}
COLORS = {"fft_centre": "#2a78d6", "fft_best_fit": "#eb6834",
          "pyramid": "#1baf7a", "no_bloom": "#eda100"}
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
BINS = [("centre", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("edge", 2 / 3, 1.01)]
METRICS = ["flip", "log_mse", "log_psnr", "energy"]

# Proceed / pivot criterion, fixed on 2026-10-17 before the default-config run (decision
# log). The best uniform kernel is the FFT baseline with the lowest median FLIP over all
# point-grid windows. PROCEED if, on the point grid, its median edge FLIP is at least
# DECISION_RATIO x its median centre FLIP, and the HDRI crops agree in direction
# (edge / centre > 1). Otherwise PIVOT (RESEARCH_PLAN.md, Phase 0 "Decision").
UNIFORM = ["fft_centre", "fft_best_fit"]
DECISION_RATIO = 2.0


def decide(records, summary):
    def overall(b):
        return float(np.median([r["flip"] for r in records
                                if r["kind"] == "synthetic" and r["baseline"] == b]))
    best = min(UNIFORM, key=overall)
    grid = summary["synthetic"][best]["edge_over_centre_flip"]
    crops = summary["hdri_crop"][best]["edge_over_centre_flip"]
    ok = grid is not None and crops is not None and grid >= DECISION_RATIO and crops > 1
    return {"best_uniform": best, "grid_edge_over_centre_flip": grid,
            "crops_edge_over_centre_flip": crops, "required_ratio": DECISION_RATIO,
            "verdict": "PROCEED" if ok else "PIVOT"}


def field_height(x, y, w, h):
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return float(np.hypot(x - cx, y - cy) / np.hypot(cx, cy))


def load_baselines(args, ps):
    kernels = np.load(os.path.join(os.path.dirname(args.fft), "fft_kernels.npz"))
    with open(args.pyramid) as fh:
        best = json.load(fh)["best"]
    params = URPBloomParams(**{**best, "tint": tuple(best["tint"])})
    eta = ps.scatter.energy_fraction
    if eta > 0:
        raise NotImplementedError("scatter is on: pass fft_bloom a precomputed scatter term")

    def fft(k):
        return lambda img: fft_bloom(img, k, ps, scatter=np.zeros_like(img))

    render = {"fft_centre": fft(kernels["centre"]), "fft_best_fit": fft(kernels["best_fit"]),
              "pyramid": lambda img: pyramid_bloom(img, params)[0],
              "no_bloom": lambda img: img}
    return render, params


def sources(name, entry):
    """[(x, y)] light positions of one input, from the manifest."""
    if entry["type"] == "synthetic":
        return [(float(x), float(y)) for y in entry["ys"] for x in entry["xs"]]
    return [tuple(entry["light_px"])]


def evaluate_image(name, entry, img, ref, render, size, eps, ppd):
    records, outs = [], {}
    peak = m.log_peak(ref, eps)
    for b in BASELINES:
        out = render[b](img)
        outs[b] = out
        fmap, flip_params = m.flip_map(out, ref, ppd)
        for x, y in sources(name, entry):
            w_out, w_ref = m.window(out, x, y, size), m.window(ref, x, y, size)
            records.append({
                "image": name, "kind": entry["type"], "baseline": b, "x": x, "y": y,
                "H": field_height(x, y, img.shape[1], img.shape[0]),
                "flip": float(m.window(fmap, x, y, size).mean()),
                "log_mse": m.log_loss(w_out, w_ref, eps),
                "log_psnr": m.log_psnr(w_out, w_ref, eps, peak),
                "energy": m.energy_error(w_out, w_ref),
                "flip_exposure": [flip_params["startExposure"], flip_params["stopExposure"]]})
    return records, outs


def summarize(records):
    """Median of each metric per (kind, baseline, H bin), and edge / centre ratios."""
    out = {}
    for kind in ("synthetic", "hdri_crop"):
        for b in BASELINES:
            rows = [r for r in records if r["kind"] == kind and r["baseline"] == b]
            cell = {}
            for bin_name, lo, hi in BINS:
                sel = [r for r in rows if lo <= r["H"] < hi]
                cell[bin_name] = {k: (float(np.median([r[k] for r in sel])) if sel else None)
                                  for k in METRICS}
                cell[bin_name]["n"] = len(sel)
            for k in ("flip", "log_mse"):
                c, e = cell["centre"][k], cell["edge"][k]
                cell[f"edge_over_centre_{k}"] = (e / c) if c and e is not None else None
            out.setdefault(kind, {})[b] = cell
    return out


def table_markdown(summary):
    lines = []
    for kind, title in (("synthetic", "Point grid (same source, only H changes)"),
                        ("hdri_crop", "HDRI crops (at measured H)")):
        lines += [f"### {title}", "",
                  ("| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | "
                   "log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |"),
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b in BASELINES:
            c = summary[kind][b]

            def f(v, fmt="{:.4f}"):
                return "-" if v is None else fmt.format(v)
            lines.append(
                f"| {LABELS[b]} | {f(c['centre']['flip'])} | {f(c['mid']['flip'])} | "
                f"{f(c['edge']['flip'])} | {f(c['edge_over_centre_flip'], '{:.2f}x')} | "
                f"{f(c['centre']['log_mse'], '{:.2e}')} | {f(c['mid']['log_mse'], '{:.2e}')} | "
                f"{f(c['edge']['log_mse'], '{:.2e}')} | "
                f"{f(c['edge_over_centre_log_mse'], '{:.2f}x')} | "
                f"{f(c['edge']['energy'], '{:+.2%}')} |")
        n = {name: summary[kind][BASELINES[0]][name]["n"] for name, _, _ in BINS}
        lines += ["", (f"Medians over windows. Windows per bin: {n}. Bins: centre H < 1/3, "
                       "mid 1/3-2/3, edge >= 2/3."), ""]
    return "\n".join(lines)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_vs_h(records, path):
    """One panel per metric (never two y-scales on one axis). Point grid: median per H
    (line + markers); HDRI crops: hollow markers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (k, ylabel) in zip(axes, (("flip", "HDR-FLIP in window (lower is better)"),
                                      ("log_mse", "log-MSE in window (lower is better)"))):
        style(ax)
        for b in BASELINES:
            grid = [r for r in records if r["kind"] == "synthetic" and r["baseline"] == b]
            hs = sorted({round(r["H"], 3) for r in grid})
            med = [np.median([r[k] for r in grid if round(r["H"], 3) == h]) for h in hs]
            ax.plot(hs, med, color=COLORS[b], linewidth=2, marker="o", markersize=5,
                    label=LABELS[b])
            crops = [r for r in records if r["kind"] == "hdri_crop" and r["baseline"] == b]
            ax.scatter([r["H"] for r in crops], [r[k] for r in crops], s=64,
                       facecolors="none", edgecolors=COLORS[b], linewidths=1.5)
        ax.set_xlabel("Field height H (0 = centre, 1 = corner)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_yscale("log")
        ax.set_xlim(-0.02, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(BASELINES), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Error vs field height. Lines: point grid (median per H); "
                 "hollow circles: HDRI crops", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# Diverging: two hues (orange = too dark, blue = too bright) around a neutral grey.
ERR_CMAP = LinearSegmentedColormap.from_list(
    "err", ["#eb6834", "#f0efec", "#2a78d6"])


def plot_motivation(name, entry, ref, outs, path, zoom=192, decades=6, err_decades=0.3):
    """Row 1: full frames (display tonemap), zoom box marked. Row 2: log-luminance zoom
    on the light, one shared range. Row 3: log10(baseline / reference) in the zoom: where
    each uniform method puts too much (blue) or too little (orange) light."""
    shown = ["reference"] + BASELINES[:3]
    frames = {"reference": ref, **outs}
    # crops: the light; point grid: its first source, the top-left corner point
    x, y = sources(name, entry)[0]
    fig, axes = plt.subplots(3, len(shown), figsize=(4 * len(shown), 8.2), facecolor=SURFACE)
    ref_lum = luminance(m.window(ref, x, y, zoom))
    vmax = np.log10(ref_lum.max())
    em = None
    for i, k in enumerate(shown):
        top, mid, bot = axes[:, i]
        top.imshow(display(frames[k]))
        top.add_patch(plt.Rectangle((x - zoom / 2, y - zoom / 2), zoom, zoom,
                                    fill=False, edgecolor=SURFACE, linewidth=1))
        lum = luminance(m.window(frames[k], x, y, zoom))
        im = mid.imshow(np.log10(np.maximum(lum, 1e-12)), cmap="gray",
                        vmin=vmax - decades, vmax=vmax)
        if k == "reference":
            bot.set_axis_off()
            bot.text(0.5, 0.5, "log10(method / reference)\nin the zoom window",
                     ha="center", va="center", color=INK_2, fontsize=9, transform=bot.transAxes)
        else:
            ratio = np.log10(np.maximum(lum, 1e-12) / np.maximum(ref_lum, 1e-12))
            em = bot.imshow(ratio, cmap=ERR_CMAP, vmin=-err_decades, vmax=err_decades)
        title = "Reference (spatially varying)" if k == "reference" else LABELS[k]
        top.set_title(title, color=INK, fontsize=10)
        for ax in (top, mid, bot):
            ax.set_xticks([])
            ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[1, :], fraction=0.015, pad=0.01)
    cb.set_label(f"log10 luminance ({decades} decades)", color=INK_2)
    if em is not None:
        cb = fig.colorbar(em, ax=axes[2, :], fraction=0.015, pad=0.01)
        cb.set_label("log10 ratio: blue = too bright, orange = too dark", color=INK_2)
    h = entry.get("H_measured", field_height(x, y, ref.shape[1], ref.shape[0]))
    fig.suptitle(f"{name}: light at H = {h:.2f}", color=INK)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", default="data/psf/psf_quick.npz")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--refs", default="data/reference")
    ap.add_argument("--fft", default="outputs/01_gap_study/fft_bloom.json")
    ap.add_argument("--pyramid", default="outputs/01_gap_study/pyramid_fit.json")
    ap.add_argument("--out", default="outputs/01_gap_study")
    ap.add_argument("--pointgrid-ref", default=None,
                    help="exact point-grid reference (exact_pointgrid.py) instead of "
                         "<refs>/pointgrid_ref.exr")
    ap.add_argument("--window", type=int, default=None, help="window size, px (default 2K)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--ppd", type=float, default=67.0, help="FLIP pixels per degree")
    ap.add_argument("--figure", nargs="*", default=["pointgrid", "goegap_road_4k_corner",
                                                    "cobblestone_street_night_4k_corner"])
    args = ap.parse_args()

    with open(os.path.join(args.refs, "reference_report.json")) as fh:
        ref_psf = json.load(fh)["psf"]
    with open(args.fft) as fh:
        fft_psf = json.load(fh)["psf"]
    for what, p in (("references", ref_psf), ("FFT baseline", fft_psf)):
        if os.path.normpath(p) != os.path.normpath(args.psf):
            raise ValueError(f"{what} were made with {p}, not {args.psf}")
    ps = PSFSet(args.psf)
    if ps.placeholder:
        print("WARNING: PLACEHOLDER lens parameters. Do not report these results.\n")
    size = args.window or 2 * ps.K
    render, pyr_params = load_baselines(args, ps)
    with open(os.path.join(args.inputs, "manifest.json")) as fh:
        manifest = json.load(fh)
    os.makedirs(args.out, exist_ok=True)

    records = []
    for name, entry in manifest["images"].items():
        print(f"{name}")
        img = read_hdr(os.path.join(args.inputs, f"{name}.exr"))
        ref_path = (args.pointgrid_ref if name == "pointgrid" and args.pointgrid_ref
                    else os.path.join(args.refs, f"{name}_ref.exr"))
        ref = read_hdr(ref_path)
        recs, outs = evaluate_image(name, entry, img, ref, render, size, args.eps, args.ppd)
        records += recs
        if name in args.figure:
            plot_motivation(name, entry, ref, outs,
                            os.path.join(args.out, f"motivation_{name}.png"))

    summary = summarize(records)
    decision = decide(records, summary)
    table = table_markdown(summary)
    verdict = (f"**Decision: {decision['verdict']}**. Best uniform kernel: "
               f"{LABELS[decision['best_uniform']]}; point-grid edge/centre FLIP "
               f"{decision['grid_edge_over_centre_flip']:.2f}x (required >= "
               f"{DECISION_RATIO:.1f}x); HDRI crops {decision['crops_edge_over_centre_flip']:.2f}x "
               f"(required > 1).")
    if ps.placeholder:
        verdict += " PLACEHOLDER lens parameters: not a valid decision."
    if not args.pointgrid_ref:
        verdict += (" Point-grid reference is grid-interpolated (10-41% PSF error between"
                    " nodes): not a valid decision; pass --pointgrid-ref.")
    print("\n" + table + "\n" + verdict)
    with open(os.path.join(args.out, "gap_table.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Gap study ({os.path.basename(args.psf)}, window {size} px)\n\n"
                 f"{verdict}\n\n{table}")
    plot_vs_h(records, os.path.join(args.out, "gap_vs_H.png"))
    versions = {p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "flip-evaluator", "rayoptics", "torch")}
    with open(os.path.join(args.out, "gap_eval.json"), "w") as fh:
        json.dump({"psf": args.psf, "placeholder": ps.placeholder, "window_px": size,
                   "pointgrid_ref": args.pointgrid_ref or "interpolated",
                   "eps": args.eps, "flip_ppd": args.ppd, "pyramid_params": pyr_params.__dict__,
                   "fft_kernels": os.path.join(os.path.dirname(args.fft), "fft_kernels.npz"),
                   "bins": BINS, "versions": versions, "decision": decision, "summary": summary,
                   "records": records}, fh, indent=2)
    print(f"\nwrote {args.out}/gap_eval.json, gap_table.md, gap_vs_H.png")


if __name__ == "__main__":
    main()
