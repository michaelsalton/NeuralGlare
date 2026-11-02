"""Phase 2, step 1: train the Phase 1 architecture on exact point-light PSFs (the baseline).

Model: the Phase 1 pyramid (2 taps per level, grey weights, input uv) plus one level at full
resolution, whose "mip" is the input frame itself:
    out = w0(uv) * input + w1(uv) * bilinear(up[0])
Phase 1's up[0] is half resolution, while the lens reference is the full frame convolved with
the PSF, most of whose energy sits in a core a few pixels wide. The input path carries the core,
the pyramid the halo. Same upsample kernel, one more network in the weight buffer.

Data: the PSF bank (psf_bank.py). The 144 training positions are packed into 9 fixed frames by
(row mod 3, col mod 3) of their grid cell, so sources are >= 241 px apart and their PSFs never
overlap; every reference is the bank's exact PSFs splatted at intensity 1e4 (as the point grid).
Model and optics are linear, so fixed frames lose nothing against random ones.

Loss: mean squared difference of log(max(out, 0) + EPS) and log(ref + EPS), EPS = 1e-7 x
intensity = 1e-3 (the gap study's eps), plus (out- / EPS)^2: the weights are unconstrained, an
output can go negative, and the clamped log has no gradient there.

Evaluation and criterion, fixed before the first run (decision log 2026-10-31): exactly the gap
study's measurement (evaluate_gap.py) on the 35 point-grid test sources, compared with Phase 0:
    (1) beats the fitted pyramid: median FLIP below it in every H bin
    (2) closes the edge gap: median edge FLIP below the best uniform kernel's, and
        edge / centre FLIP <= 2x
Expected in advance: centre FLIP stays above the FFT centre kernel's (round grey Gaussians can't
form the aperture spikes or colour fringes).

slangpy 0.43.1 decides from the first upsample_chain_pass call whether it is differentiable, so
training runs before any evaluation (see train_pyramid.py).

--init fitted (sanity check, decision log 2026-11-02) starts every level at Phase 0's fitted
pyramid bloom as constant weights instead of Phase 1's, so the "Init" row must reproduce Phase 0's
fitted-pyramid numbers; outputs get a "_fitted" suffix.

    uv run python experiments/03_train_network/train_points.py --run default
    uv run python experiments/03_train_network/train_points.py --run default --init fitted
"""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import slangpy as spy
import torch

from neuralglare.baselines.pyramid_bloom import URPBloomParams, shader_constants
from neuralglare.eval import metrics as m
from neuralglare.io import read_hdr
from neuralglare.paths import REFS, RUNS, psf_bank_path, run_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "01_gap_study"))
sys.path.insert(0, str(ROOT / "experiments" / "02_slang_plumbing"))
from evaluate_gap import BINS, field_height
from exact_pointgrid import splat
from train_levels import HIDDEN, IN, OUT, environment, init_params
from train_levels import OUT_DIR as PHASE1_DIR
from train_pyramid import PARAM_COUNT, slang_mips

EXPERIMENT = "03_train_network"
INTENSITY = 1e4                     # per channel, as the Phase 0 point grid
EPS = 1e-7 * INTENSITY              # 7 decades below the source; = the gap study's eps
P = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)     # as Phase 1
LOG_EVERY = 250
COMPARE = {"fft_centre": "Phase 0: FFT, exact centre PSF",
           "pyramid": "Phase 0: pyramid bloom (fitted)"}
EDGE_RATIO = 2.0                    # criterion (2), the Phase 0 proceed rule's ratio


def train_frames(bank, rows, cols):
    """Bank indices of the 9 training frames: training positions grouped by (row mod 3,
    col mod 3) of their grid cell. Same-class cells are >= 3 cells apart."""
    idx = np.nonzero(bank["split"] == "train")[0]
    assert len(idx) == rows * cols, "bank and split.toml disagree on the training grid"
    K = bank["psf"].shape[1]
    frames = []
    for dr in range(3):
        for dc in range(3):
            sel = [int(i) for n, i in enumerate(idx) if (n // cols) % 3 == dr and (n % cols) % 3 == dc]
            pos = bank["pos_px"][sel].astype(np.float64)
            dist = np.hypot(*(pos[:, None] - pos[None]).transpose(2, 0, 1))
            assert dist[~np.eye(len(sel), dtype=bool)].min() >= K, "PSFs overlap in a frame"
            frames.append(sel)
    return frames


def fitted_pyramid_init(gen, n_levels, fit):
    """Phase 0's fitted pyramid bloom as constant per-level weights (output layer 0, bias w).
    URP with m mips: up[k] = lerp(mip_k, up(up[k+1]), s) for k < m-1, up[m-1] = mip_{m-1};
    out = input + intensity * up(up[0]). Hidden layers stay random so uv can still be learned."""
    p = URPBloomParams(**{**fit, "tint": tuple(fit["tint"])})
    assert p.tint == (1.0, 1.0, 1.0), "a tint isn't representable by grey weights"
    assert p.max_iterations <= n_levels - 1, "fit uses more mips than the model has"
    _, _, s = shader_constants(p)
    m = p.max_iterations
    weights = [(1 - s, s) if k < m - 1 else (1.0, 0.0) for k in range(n_levels - 1)]
    weights.append((1.0, p.intensity))                  # full-resolution level
    nets = []
    for w in weights:
        net = init_params(gen)
        net[IN * HIDDEN + HIDDEN:-OUT] = 0              # W2: output independent of uv
        net[-OUT:] = torch.tensor(w)                    # b2: the constant weights
        nets.append(net)
    return torch.cat(nets)


def render(bank, sel, shape):
    """Input frame (sources at INTENSITY) and its exact reference, from the bank's PSFs."""
    img = np.zeros(shape, np.float32)
    points = []
    for i in sel:
        x, y = (int(v) for v in bank["pos_px"][i])
        img[y, x] = INTENSITY
        points.append((x, y, np.full(3, INTENSITY)))
    ref, _ = splat(shape, points, [bank["psf"][i].astype(np.float64) for i in sel])
    return img, ref


def level(module, mip, low, params, k):
    """Upsample level k at mip's resolution: w0 * mip + w1 * bilinear(low), network k."""
    h, w = mip.shape[:2]
    return module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), mip=mip,
                                      low=low, params=params, offset=k * PARAM_COUNT)


def glare(module, mips, src, params):
    """The pyramid (levels 0..n-1) and the full-resolution level n, whose mip is the input."""
    out = mips[-1]
    for k in reversed(range(len(mips) - 1)):
        out = level(module, mips[k], out, params, k)
    return level(module, src, out, params, len(mips) - 1)


def prepare(module, img, ref=None):
    mips = slang_mips(module, img, P)
    src = torch.from_numpy(np.ascontiguousarray(img, np.float32)).cuda()
    return mips, src, (None if ref is None else torch.from_numpy(ref).cuda())


def log_loss(out, ref):
    data = (torch.log(out.clamp(min=0) + EPS) - torch.log(ref + EPS)) ** 2
    negative = ((-out).clamp(min=0) / EPS) ** 2
    return data.mean() + negative.mean()


def train_loss(module, data, params):
    """The training objective itself: mean loss over all training frames. One frame's loss
    isn't comparable across steps (frames hold 15 or 18 sources)."""
    with torch.no_grad():
        return float(np.mean([log_loss(glare(module, mips, src, params), ref).item()
                              for mips, src, ref in data]))


def evaluate(module, params, img, ref, positions, size, ppd):
    """Per test source, the gap study's window metrics. FLIP from one full-frame map."""
    mips, src, _ = prepare(module, img)
    with torch.no_grad():
        out = glare(module, mips, src, params).cpu().numpy()
    fmap, _ = m.flip_map(out, ref, ppd)
    pos_out = np.maximum(out, 0)
    h, w = img.shape[:2]
    records = []
    for x, y in positions:
        w_out, w_ref = m.window(pos_out, x, y, size), m.window(ref, x, y, size)
        records.append({"x": int(x), "y": int(y), "H": field_height(x, y, w, h),
                        "flip": float(m.window(fmap, x, y, size).mean()),
                        "log_mse": m.log_loss(w_out, w_ref, EPS),
                        "energy": m.energy_error(m.window(out, x, y, size), w_ref)})
    return records, float((out < 0).mean())


def summarize(records):
    cell = {}
    for name, lo, hi in BINS:
        sel = [r for r in records if lo <= r["H"] < hi]
        cell[name] = {k: float(np.median([r[k] for r in sel])) for k in ("flip", "log_mse", "energy")}
        cell[name]["n"] = len(sel)
    cell["edge_over_centre_flip"] = cell["edge"]["flip"] / cell["centre"]["flip"]
    return cell


def verdict(model, phase0):
    pyramid, fft = phase0["pyramid"], phase0["fft_centre"]
    beats = all(model[b]["flip"] < pyramid[b]["flip"] for b in ("centre", "mid", "edge"))
    closes = (model["edge"]["flip"] < fft["edge"]["flip"]
              and model["edge_over_centre_flip"] <= EDGE_RATIO)
    return {"beats_fitted_pyramid": beats, "closes_edge_gap": closes,
            "centre_above_fft_centre_as_expected": model["centre"]["flip"] > fft["centre"]["flip"]}


def table(rows):
    lines = [("| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | "
              "log-MSE edge | energy edge |"), "|---|---|---|---|---|---|---|---|"]
    for label, c in rows:
        lines.append(f"| {label} | {c['centre']['flip']:.4f} | {c['mid']['flip']:.4f} | "
                     f"{c['edge']['flip']:.4f} | {c['edge_over_centre_flip']:.2f}x | "
                     f"{c['centre']['log_mse']:.2e} | {c['edge']['log_mse']:.2e} | "
                     f"{c['edge']['energy']:+.2%} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["phase1", "fitted"], default="phase1")
    args = ap.parse_args()
    tag = ("" if args.init == "phase1" else f"_{args.init}") + \
          ("" if args.lr == ap.get_default("lr") else f"_lr{args.lr:g}")

    split = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase2_points"]
    with np.load(ROOT / psf_bank_path(args.run)) as z:
        bank = {k: z[k] for k in z.files}
    cam = json.loads(str(bank["config"]))["camera"]
    shape = (cam["height_px"], cam["width_px"], 3)
    gap = json.loads((ROOT / "outputs" / "01_gap_study" / args.run / "eval" / "gap_eval.json")
                     .read_text())
    size, ppd = gap["window_px"], gap["flip_ppd"]
    assert size == 2 * bank["psf"].shape[1], "window differs from the gap study's"
    assert gap["eps"] == EPS, "log eps differs from the gap study's"
    phase0 = {b: gap["summary"]["synthetic"][b] for b in COMPARE}

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    module = spy.Module.load_from_file(device, "train_glare.slang")

    frames = train_frames(bank, *split["train_grid"])
    data = [prepare(module, *render(bank, sel, shape)) for sel in frames]
    n_levels = len(data[0][0])                      # pyramid levels + the full-resolution one
    if args.init == "fitted":
        fit = json.loads((ROOT / "outputs" / "01_gap_study" / args.run / "fits" / "pyramid_fit.json")
                         .read_text())["best"]
        init = fitted_pyramid_init(gen, n_levels, fit)
    else:
        phase1 = np.load(PHASE1_DIR / "train_pyramid_params.npy")
        assert phase1.size == (n_levels - 1) * PARAM_COUNT, "Phase 1 weights have another level count"
        init = torch.cat([torch.from_numpy(phase1), init_params(gen)])
    print(f"{len(frames)} training frames ({min(map(len, frames))}-{max(map(len, frames))} "
          f"sources), {n_levels} levels, {init.numel()} parameters")

    # Train first (slangpy: the first upsample_chain_pass call must carry gradients)
    params = init.clone().cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    curve = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        mips, src, ref = data[rng.integers(len(data))]
        optimizer.zero_grad()
        out = glare(module, mips, src, params)
        assert out.grad_fn is not None, "slangpy dropped autograd (see train_pyramid.py)"
        loss = log_loss(out, ref)
        loss.backward()
        if step == 1:                               # after the first gradient call, before any update
            curve.append({"step": 0, "train_loss": train_loss(module, data, params)})
            print(f"step     0: train loss {curve[0]['train_loss']:.3e}", flush=True)
        optimizer.step()
        schedule.step()
        if step % LOG_EVERY == 0 or step == args.steps:
            full = train_loss(module, data, params)
            curve.append({"step": step, "loss": loss.item(), "train_loss": full})
            print(f"step {step:5d}: train loss {full:.3e}  (this frame {loss.item():.3e}, "
                  f"{time.perf_counter() - t0:.0f}s)", flush=True)
    seconds = time.perf_counter() - t0

    test_img = read_hdr(str(ROOT / "data" / "inputs" / "pointgrid.exr"))[..., :3]
    test_ref = read_hdr(str(ROOT / REFS[args.run] / "pointgrid_ref_exact.exr"))[..., :3]
    test_pos = [tuple(p) for p in bank["pos_px"][bank["split"] == "test"]]
    results = {}
    for name, p in (("init", init.cuda()), ("trained", params.detach())):
        records, negative = evaluate(module, p, test_img, test_ref, test_pos, size, ppd)
        results[name] = {"summary": summarize(records), "negative_fraction": negative,
                         "records": records}
    trained = results["trained"]["summary"]
    answers = verdict(trained, phase0)

    fits = Path(run_dir(EXPERIMENT, args.run, "fits"))
    export = Path(run_dir(EXPERIMENT, args.run, "export"))
    np.save(fits / f"train_points{tag}_params.npy", params.detach().cpu().numpy())
    params.detach().cpu().numpy().astype("<f4").tofile(export / f"glare_params{tag}.bytes")
    (Path(run_dir(EXPERIMENT, args.run, "eval")) / f"train_points{tag}_eval.json").write_text(json.dumps({
        "environment": environment(device), "args": vars(args), "eps": EPS,
        "intensity": INTENSITY, "window_px": size, "flip_ppd": ppd, "levels": n_levels,
        "param_count": PARAM_COUNT, "frames": frames, "seconds": seconds, "curve": curve,
        "phase0": phase0, "answers": answers, "results": results}, indent=2))

    rows = [(COMPARE[b], phase0[b]) for b in COMPARE]
    start = {"phase1": "Phase 1 weights", "fitted": "fitted pyramid"}[args.init]
    rows += [(f"Init ({start}, before training)", results["init"]["summary"]),
             ("**Trained (this run)**", trained)]
    report = (f"# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs ({args.run})\n\n"
              f"{table(rows)}\n\n"
              f"Medians over the 35 point-grid windows ({size} px). Bins: centre H < 1/3 "
              f"(n = {trained['centre']['n']}), mid (n = {trained['mid']['n']}), edge >= 2/3 "
              f"(n = {trained['edge']['n']}). Negative output texels: "
              f"{results['trained']['negative_fraction']:.2%}. Learning rate {args.lr:g}; "
              f"training loss (all {len(data)} frames) {curve[0]['train_loss']:.4f} at step 0, "
              f"{curve[-1]['train_loss']:.4f} at step {curve[-1]['step']}.\n\n"
              f"Pre-registered questions (decision log 2026-10-31):\n"
              f"- (1) beats the fitted pyramid in every bin: **{answers['beats_fitted_pyramid']}**\n"
              f"- (2) closes the edge gap (edge FLIP < {phase0['fft_centre']['edge']['flip']:.3f} "
              f"and edge/centre <= {EDGE_RATIO:g}x): **{answers['closes_edge_gap']}**\n"
              f"- expected: centre FLIP above the FFT centre kernel's "
              f"({phase0['fft_centre']['centre']['flip']:.3f}): "
              f"**{answers['centre_above_fft_centre_as_expected']}**\n")
    (Path(run_dir(EXPERIMENT, args.run, "report")) / f"train_points{tag}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
