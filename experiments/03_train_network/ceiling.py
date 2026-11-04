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
from pathlib import Path

import numpy as np
from scipy import ndimage

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
          "aperture_fine": "Oracle: polygon-symmetric at 5x, round core (spikes kept, no coma)"}
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

    results = {}
    for name, edit in {**oracles, **FINE_ORACLES}.items():
        if name in FINE_ORACLES:
            kernels = [edit(k, f.astype(np.float64), OVERSAMPLE) for k, f in zip(psfs, fines)]
        else:
            kernels = [edit(k) for k in psfs]
        out, _ = splat(img.shape, points, kernels)
        records = window_records(out, ref, positions, size, ppd)
        results[name] = {"summary": summarize(records), "records": records}
        print(f"{name}: FLIP " + " / ".join(f"{results[name]['summary'][b]['flip']:.4f}"
                                            for b in BINS), flush=True)

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
    (out_dir("report") / f"ceiling{suffix}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
