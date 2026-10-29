"""Unreal check, step 5: Unreal's convolution bloom against the NumPy predictions.

    U = SceneColor · SceneColorApply + bloom      rebuilt from the capture, as the tonemapper does
    (a) Q(I) ⊛ K                                 the physical centre kernel          pre-registered
    (b) Q(I) ⊛ K_eff                             Unreal's kernel processing, as designed
    (c) unreal_fft_bloom_as_run(I, K)            (b) plus the 5.8 FFT channel mix-ups and the
                                                 R11G11B10 bloom texture             post hoc

The verdict uses only (a), (b) and the rule fixed in checks/unreal_threshold.json before
any Unreal convolution output was seen: Unreal ≈ (b) if FLIP(U, b) ≤ ½·FLIP(a, b) and every
channel's energy is within 2 % of (b); Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b). (c) was
written after seeing the output, so it is reported but never changes the verdict.

Needs engines/unreal/Baseline/extract_capture.py run on the captures first.

    uv run python experiments/01_gap_study/unreal_check.py
"""

import argparse
import datetime
import json
import os

import numpy as np
from unreal_threshold import SCENES, channel_energy, flip_score

from neuralglare.baselines.unreal_fft_bloom import (
    quantize_r11g11b10,
    unreal_bloom_as_run,
    unreal_fft_bloom_as_run,
    unreal_kernel,
)
from neuralglare.io import read_hdr
from neuralglare.paths import RUNS, run_dir
from neuralglare.reference.convolve import convolve_uniform

CAPTURES = "data/captures/unreal"
CAPTURE_NAME = {"pointgrid": "pointgrid", "ferndale_studio_12_4k_center": "ferndale"}
ENERGY_TOLERANCE = 0.02                  # pre-registered
APPLY = "Bloom.FFT.SceneColorApplyParameters"


def score(pred, U, windows, ppd):
    return {"flip": flip_score(U, pred, windows, ppd), "energy": channel_energy(U, pred, windows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--ppd", type=float, default=67.0, help="FLIP pixels per degree")
    args = ap.parse_args()
    fits = run_dir("01_gap_study", args.run, "fits")
    with open(f"{run_dir('01_gap_study', args.run, 'checks')}/unreal_threshold.json") as fh:
        threshold = json.load(fh)
    K = np.load(f"{fits}/fft_kernels.npz")["centre"]
    K_eff, info = unreal_kernel(K)
    with open(os.path.join(args.inputs, "manifest.json")) as fh:
        manifest = json.load(fh)["images"]

    scenes, rows = {}, []
    for name in SCENES:
        cap = CAPTURE_NAME[name]
        with open(f"{CAPTURES}/unreal_fft_constants_{cap}.json") as fh:
            constants = json.load(fh)
        scene_color = read_hdr(f"{CAPTURES}/unreal_beforebloom_{cap}.exr")
        bloom = read_hdr(f"{CAPTURES}/unreal_bloom_{cap}.exr")
        U = scene_color * np.float32(constants[APPLY][:3]) + bloom

        I = read_hdr(os.path.join(args.inputs, f"{name}.exr"))
        Iq = quantize_r11g11b10(I)
        entry = manifest[name]
        windows = ([(x, y, K.shape[0]) for y in entry["ys"] for x in entry["xs"]]
                   if entry["type"] == "synthetic" else [])
        a, b = convolve_uniform(Iq, K), convolve_uniform(Iq, K_eff)
        c = unreal_fft_bloom_as_run(I, K)
        bloom_c = unreal_bloom_as_run(Iq, K)

        s = {p: score(pred, U, windows, args.ppd) for p, pred in (("a", a), ("b", b), ("c", c))}
        tau = 0.5 * threshold["scenes"][name]["signal"]["flip"]
        approx_b = bool(s["b"]["flip"] <= tau
                        and max(abs(e) for e in s["b"]["energy"]) <= ENERGY_TOLERANCE)
        differs_a = bool(s["a"]["flip"] > s["b"]["flip"])
        scenes[name] = {
            "region": "35 source windows" if windows else "full frame",
            "scene_color_bit_exact": bool(np.array_equal(scene_color, Iq)),
            "apply_constants": {"capture": constants[APPLY][:3],
                                "predicted": info["scene_color_apply"]},
            "scores": s, "tau": tau, "energy_tolerance": ENERGY_TOLERANCE,
            "verdict": {"unreal_approx_b": approx_b, "unreal_differs_from_a": differs_a,
                        "pass": approx_b and differs_a},
            "post_hoc_c_bloom_texture": {
                "energy": channel_energy(bloom, bloom_c, windows),
                "l1_relative": (np.abs(bloom - bloom_c).sum((0, 1))
                                / bloom.astype(np.float64).sum((0, 1))).tolist(),
                "bit_exact_fraction": (bloom == bloom_c).mean((0, 1)).tolist()}}
        rows.append((name, scenes[name]))
        print(f"{name}: pass={scenes[name]['verdict']['pass']} "
              f"(FLIP U-a {s['a']['flip']:.3f}, U-b {s['b']['flip']:.3f} vs tau {tau:.3f}, "
              f"U-c {s['c']['flip']:.3f}; energy U/b-1 {np.round(s['b']['energy'], 4)}, "
              f"U/c-1 {np.round(s['c']['energy'], 4)})")

    report = run_dir("01_gap_study", args.run, "report")
    pct = lambda e: " / ".join(f"{100 * x:+.2f}" for x in e)
    rule = ("Pre-registered rule (`checks/unreal_threshold.json`): Unreal ≈ (b) if "
            "FLIP(U, b) ≤ τ = ½·FLIP(a, b) and every channel's energy is within 2 % of (b); "
            "Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b). (c) is post hoc and does not change the "
            "verdict.")
    header = ("| Scene | FLIP U–a | FLIP U–b | τ | FLIP U–c (post hoc) | Energy U/b − 1, % | "
              "Energy U/c − 1, % | Verdict |")
    lines = ["# Unreal check (UE 5.8.4, centre kernel)", "", rule, "", header,
             "|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        s = r["scores"]
        lines.append(f"| `{name}` | {s['a']['flip']:.3f} | {s['b']['flip']:.3f} | "
                     f"{r['tau']:.3f} | {s['c']['flip']:.3f} | {pct(s['b']['energy'])} | "
                     f"{pct(s['c']['energy'])} | {'pass' if r['verdict']['pass'] else 'fail'} |")
    with open(f"{report}/unreal_check.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    path = f"{run_dir('01_gap_study', args.run, 'eval')}/unreal_check.json"
    with open(path, "w") as fh:
        json.dump({"written": datetime.datetime.now().isoformat(timespec="seconds"),
                   "run": args.run, "engine": "UE 5.8.4 (CL 58902052), DX12, RTX 2070, "
                   "driver 621.01", "kernel_processing": info, "flip_ppd": args.ppd,
                   "scenes": scenes}, fh, indent=2)
    print(f"\nwrote {path} and {report}/unreal_check.md")


if __name__ == "__main__":
    main()
