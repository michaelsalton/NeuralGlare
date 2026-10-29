"""Unreal check: the pass threshold, fixed before any Unreal convolution-bloom output is seen.

Never reads data/captures/. For each scene, from NumPy alone:

    a      = Q(I) ⊛ K          prediction (a): the physical centre kernel
    b      = Q(I) ⊛ K_eff      prediction (b): the kernel as Unreal processes it
    b_raw  =   I  ⊛ K_eff      (b) without the R11G11B10 input quantisation Q

    signal = how far (a) is from (b): what the check has to detect
    noise  = how far b_raw is from (b): error from the input format alone

Both predictions are pure convolution, not fft_bloom: Unreal gets only K, no scatter term.
FLIP is the mean HDR-FLIP over 127x127 windows on the 35 point-grid sources, and over the
full frame for HDRI crops. Energy is per channel, relative to (b).

    uv run python experiments/01_gap_study/unreal_threshold.py
"""

import argparse
import datetime
import importlib.metadata
import json
import os

import numpy as np

from neuralglare.baselines.unreal_fft_bloom import quantize_r11g11b10, unreal_kernel
from neuralglare.eval import metrics as m
from neuralglare.io import read_hdr
from neuralglare.paths import RUNS, run_dir
from neuralglare.reference.convolve import convolve_uniform

SCENES = ("pointgrid", "ferndale_studio_12_4k_center")   # fit in R11G11B10 (max < 64512)


def flip_score(pred, ref, windows, ppd):
    fmap, _ = m.flip_map(pred, ref, ppd)
    if not windows:
        return float(fmap.mean())
    return float(np.mean([m.window(fmap, x, y, size).mean() for x, y, size in windows]))


def channel_energy(pred, ref, windows):
    """Per-channel energy of pred relative to ref, summed over the windows (or the frame)."""
    crops = [(pred, ref)] if not windows else [
        (m.window(pred, x, y, s), m.window(ref, x, y, s)) for x, y, s in windows]
    p = sum(c[0].astype(np.float64).sum((0, 1)) for c in crops)
    r = sum(c[1].astype(np.float64).sum((0, 1)) for c in crops)
    return (p / r - 1.0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--ppd", type=float, default=67.0, help="FLIP pixels per degree")
    args = ap.parse_args()
    fits = run_dir("01_gap_study", args.run, "fits")
    K = np.load(f"{fits}/fft_kernels.npz")["centre"]
    K_eff, kernel_info = unreal_kernel(K)
    with open(os.path.join(args.inputs, "manifest.json")) as fh:
        manifest = json.load(fh)["images"]

    scenes = {}
    for name in SCENES:
        I = read_hdr(os.path.join(args.inputs, f"{name}.exr"))
        entry = manifest[name]
        windows = ([(x, y, K.shape[0]) for y in entry["ys"] for x in entry["xs"]]
                   if entry["type"] == "synthetic" else [])
        Iq = quantize_r11g11b10(I)
        a, b, b_raw = (convolve_uniform(Iq, K), convolve_uniform(Iq, K_eff),
                       convolve_uniform(I, K_eff))
        signal = {"flip": flip_score(a, b, windows, args.ppd),
                  "energy": channel_energy(a, b, windows)}
        noise = {"flip": flip_score(b_raw, b, windows, args.ppd),
                 "energy": channel_energy(b_raw, b, windows)}
        scenes[name] = {"region": "35 source windows" if windows else "full frame",
                        "signal": signal, "noise": noise,
                        "noise_over_signal_flip": noise["flip"] / signal["flip"]}
        print(f"{name} ({scenes[name]['region']}): FLIP(a, b) = {signal['flip']:.4f}, "
              f"FLIP(b_raw, b) = {noise['flip']:.4f}; energy a/b - 1 = "
              f"{np.round(signal['energy'], 4)}, b_raw/b - 1 = {np.round(noise['energy'], 4)}")

    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "flip-evaluator")}
    path = f"{run_dir('01_gap_study', args.run, 'checks')}/unreal_threshold.json"
    with open(path, "w") as fh:
        json.dump({"written": datetime.datetime.now().isoformat(timespec="seconds"),
                   "run": args.run, "kernel": f"{fits}/fft_kernels.npz (centre)",
                   "kernel_processing": kernel_info, "flip_ppd": args.ppd,
                   "versions": versions, "scenes": scenes}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
