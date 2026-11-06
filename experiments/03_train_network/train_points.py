"""Phase 2, step 1: train the Phase 1 architecture on exact point-light PSFs (the baseline).

Model: the Phase 1 pyramid (2 taps per level, grey weights, input uv) plus one level at full
resolution, whose "mip" is the input frame itself:
    out = (1 - t(uv)) * input + t(uv) * bilinear(up[0]),   t = sigmoid(MLP(uv))
(Step 2, coma: every level has 4 taps, w = softmax(MLP(uv)) over mip(x),
mip(x + o), mip(x - o) and the level below, o = H * d_hat texels; glare.slang. Step 2, spikes,
--arch spikes: 14 streaked rays from mip 0, two families, mixed into the full-resolution output by
softmax(SpikeMLP(uv)); glare.slang. --arch offsets runs without them.)
Phase 1's up[0] is half resolution, while the lens reference is the full frame convolved with
the PSF, most of whose energy sits in a core a few pixels wide. The input path carries the core,
the pyramid the halo. Same upsample kernel, one more network in the weight buffer. Every level
is a lerp (glare.slang, decision log 2026-11-02): weights positive and summing to 1, so outputs
are never negative and light is never lost.

Data: the PSF bank (psf_bank.py). The 144 training positions are packed into 9 fixed frames by
(row mod 3, col mod 3) of their grid cell, so sources are >= 241 px apart and their PSFs never
overlap; every reference is the bank's exact PSFs splatted at intensity 1e4 (as the point grid).
Model and optics are linear, so fixed frames lose nothing against random ones. One frame
(split.toml val_frame) is held out: 8 frames train, the validation frame selects between runs.

Loss: mean difference of log(out + EPS) and log(ref + EPS), EPS = 1e-7 x intensity = 1e-3 (the
gap study's eps). --loss abs (default) takes its absolute value, --loss sq its square (the gap
study's log-MSE). Squaring shrinks small errors quadratically, so a faint haze that FLIP sees
barely counts; the absolute error grows linearly with it, as FLIP does (decision log 2026-11-03).
No negative-output penalty: the lerp can't produce negatives.

Selection between runs: lowest mean FLIP over the validation frame's windows.

Evaluation and criterion, fixed before the first run (decision log 2026-10-31): exactly the gap
study's measurement (evaluate_gap.py) on the 35 point-grid test sources, compared with Phase 0's
strongest tuned baselines (the lower-FLIP of the sq and abs fits, strongest_baselines):
    (1) beats the fitted pyramid: median FLIP below it in every H bin
    (2) closes the edge gap: median edge FLIP below the best uniform kernel's, and
        edge / centre FLIP <= 2x
Expected in advance: centre FLIP stays above the FFT centre kernel's (round grey Gaussians can't
form the aperture spikes or colour fringes).

slangpy 0.43.1 decides from the first upsample_chain_pass call whether it is differentiable, so
training runs before any evaluation (see train_pyramid.py).

--init fitted (default) starts every level at Phase 0's fitted pyramid bloom as a constant
blend, energy-normalised (see fitted_pyramid_init), so the "Init" row is close to Phase 0's
fitted-pyramid numbers but not identical. --init neutral starts every level at t = 0.5.
Outputs are named train_points_<arch>_<loss>_<init>_lr<lr>, plus _seed<n> for --seed other than 0
(seed spread, decision log 2026-11-03). Older runs kept their names: before
the lerp (2-weight networks), and before the validation frame (squared loss on all 9 frames,
train_points_<arch>_<init>_lr<lr>).

    uv run python experiments/03_train_network/train_points.py --run default --study baseline
    uv run python experiments/03_train_network/train_points.py --run default --study spikes
    uv run python experiments/03_train_network/train_points.py --run default --study coma --arch offsets --seed 1
                                                    # delta re-measurement (decision log 2026-11-04)

--study names the folder the run's files go to, outputs/03_train_network/<run>/<study>/<stage>/
(STUDIES; decision log 2026-11-03): one folder per Step 2 ablation, so a study's reports, raw
records, weights and figures stay together.
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
from train_levels import (
    engine_layout,
    environment,
    has_spikes,
    init_params,
    level_mlp,
    spike_layout,
)
from train_pyramid import level_params, slang_mips

EXPERIMENT = "03_train_network"
STUDIES = ("setup", "baseline", "radial_input", "coma", "spikes", "spikes_full")   # one per ablation
# --arch, part of every output name, and the image the spike rays are streaked from (None: no
# spike term). glare.slang runs all three: offsets (4 taps per level, softmax(MLP(uv)), study
# coma), spikes (offsets plus rays from mip 0, study spikes, current best), spikes_full (rays from
# the input frame, study spikes_full; decision log 2026-11-05). Earlier: lerp (MLP(uv), Step 1)
# and lerp_h (MLP(H), radial_input), which glare.slang no longer runs.
SPIKE_SOURCE = {"spikes_full": "input", "spikes": "mip0", "offsets": None}
ARCHS = tuple(SPIKE_SOURCE)
OFFSET_SHARE = 0.01                 # each offset tap's share of the mip's weight at init
                                    # (decision log 2026-11-04)
SPIKE_SHARE = 1e-3                  # each ray family's share of the light at init: the bank's spikes
                                    # carry ~0.25 % of the PSF's light, ~1.2e-3 per family
SPIKE_DECAY = {"mip0": 8.0,         # init ray profile exp(-d / decay), d in texels of the rays'
               "input": 16.0}       # source: 16 full-resolution px either way (2026-11-04, -05)
T_PASS = 1e-4                       # default blend of a pass-through level (sigmoid never
                                    # reaches 0); its gradient carries t(1 - t), so a tiny t
                                    # nearly freezes the level (decision log 2026-11-02)
INTENSITY = 1e4                     # per channel, as the Phase 0 point grid
EPS = 1e-7 * INTENSITY              # 7 decades below the source; = the gap study's eps
P = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)     # as Phase 1
LOG_EVERY = 250
COMPARE = {"fft_centre": "Phase 0: FFT, exact centre PSF",
           "pyramid": "Phase 0: pyramid bloom (fitted)"}
EDGE_RATIO = 2.0                    # criterion (2), the Phase 0 proceed rule's ratio
VIEW_H = (0.0, 0.5, 0.88)           # views.py: the test sources nearest these field heights
VIEW_PX = 160                       # views.py: crop size around each view source
BLEND_GRID = (27, 48)               # views.py: (rows, cols) of the screen grid for the blend maps


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


def fitted_pyramid_init(gen, net, n_levels, fit, t_pass=T_PASS):
    """Phase 0's fitted pyramid bloom as a constant blend per level (output layer 0, biases
    the log of the tap weights, the low tap's = t). URP with m mips: up[k] = lerp(mip_k, up(up[k+1]), s) for k < m-1,
    up[m-1] = mip_{m-1}: t = s, then t = t_pass. URP's last step out = input + I * up(up[0])
    adds light; the energy-conserving lerp with the same shape is that divided by 1 + I,
    t = I / (1 + I). Hidden layers stay random so uv can still be learned."""
    p = URPBloomParams(**{**fit, "tint": tuple(fit["tint"])})
    assert p.tint == (1.0, 1.0, 1.0), "a tint isn't representable by grey weights"
    assert p.max_iterations <= n_levels - 1, "fit uses more mips than the model has"
    _, _, s = shader_constants(p)
    m = p.max_iterations
    blends = [s if k < m - 1 else t_pass for k in range(n_levels - 1)]
    blends.append(p.intensity / (1 + p.intensity))      # full-resolution level
    hidden_end = net.inputs * net.hidden + net.hidden  # mlp.slang layout: W1, b1, W2, b2
    nets = []
    for t in blends:
        w = init_params(gen, net)
        w[hidden_end:-net.outputs] = 0                  # W2: output independent of uv
        # b2 = log of the target weights: softmax(log w) = w, since they sum to 1. The low tap
        # keeps t; the offset taps take OFFSET_SHARE each of the mip's 1 - t
        assert net.outputs == 4, "init written for glare.slang's 4 taps"
        e = OFFSET_SHARE
        w[-net.outputs:] = torch.tensor(np.log([(1 - t) * (1 - 2 * e), (1 - t) * e,
                                                (1 - t) * e, t]), dtype=w.dtype)
        nets.append(w)
    return torch.cat(nets)


def spike_init(gen, module, share=SPIKE_SHARE, decay=SPIKE_DECAY):
    """The spike section: SpikeMLP with its output layer 0 and biases log(1 - 2 share, share,
    share), so each family starts with `share` of the light everywhere, then the profile logits
    z[p][n] = -n 4^p / decay for both families. A ray's weight at distance d is the product of
    one softmax weight per base-4 digit of d, so these logits give exactly exp(-d / decay)
    (normalised over 0..63 texels)."""
    lay = spike_layout(module)
    net = lay.net
    w = init_params(gen, net)
    w[net.inputs * net.hidden + net.hidden:-net.outputs] = 0   # W2: output independent of uv
    w[-net.outputs:] = torch.tensor(np.log([1 - 2 * share, share, share]), dtype=w.dtype)
    z = [-n * lay.taps ** p / decay for p in range(lay.passes) for n in range(lay.taps)]
    return torch.cat([w, torch.tensor(z * 2, dtype=w.dtype)])


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
                                      low=low, params=params,
                                      offset=k * level_mlp(module).param_count)


def spikes(module, source, base, params, offset, families=False):
    """The spike term (glare.slang): every ray streaked from source (mip 0 or the input frame,
    SPIKE_SOURCE) in its passes, each family's rays summed, then mixed into the full-resolution
    output base by SpikeMLP. offset: where the spike section starts in params (SpikeMLP, then the
    profile logits). families: also return the two family sums (even, odd), for unity_parity.py."""
    lay = spike_layout(module)
    profile = offset + lay.net.param_count
    h, w = source.shape[:2]
    grid = spy.grid((h, w))
    acc = [torch.zeros_like(source), torch.zeros_like(source)]      # even, odd family
    for ray in range(lay.rays):
        s = source
        for stage in range(lay.passes - 1):
            s = module.streak_pass(texel=grid, src=s, params=params, profile=profile, ray=ray,
                                   stage=stage)
        acc[ray % 2] = module.streak_accumulate_pass(texel=grid, src=s, acc=acc[ray % 2],
                                                     params=params, profile=profile, ray=ray,
                                                     stage=lay.passes - 1)
    h, w = base.shape[:2]
    out = module.spike_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), base=base,
                            even=acc[0], odd=acc[1], params=params, offset=offset)
    return (out, acc) if families else out


def glare(module, mips, src, params, source="mip0"):
    """The pyramid (levels 0..n-1), the full-resolution level n, whose mip is the input, and
    the spike term on top if params hold one, its rays streaked from `source` (SPIKE_SOURCE)."""
    out = mips[-1]
    for k in reversed(range(len(mips) - 1)):
        out = level(module, mips[k], out, params, k)
    out = level(module, src, out, params, len(mips) - 1)
    if has_spikes(module, params.numel(), len(mips)):
        out = spikes(module, {"mip0": mips[0], "input": src}[source], out, params,
                     len(mips) * level_mlp(module).param_count)
    return out


def prepare(module, img, ref=None):
    mips = slang_mips(module, img, P)
    src = torch.from_numpy(np.ascontiguousarray(img, np.float32)).cuda()
    return mips, src, (None if ref is None else torch.from_numpy(ref).cuda())


def log_loss(out, ref, kind):
    """Mean log error, log(out + EPS) - log(ref + EPS): absolute ("abs") or squared ("sq")."""
    d = torch.log(out + EPS) - torch.log(ref + EPS)
    return (d.abs() if kind == "abs" else d ** 2).mean()


def frames_loss(module, data, params, kind, source="mip0"):
    """The objective over a set of frames (training or validation): mean of the per-frame
    losses. One frame's loss isn't comparable across steps (frames hold 15 or 18 sources)."""
    with torch.no_grad():
        return float(np.mean([log_loss(glare(module, mips, src, params, source), ref, kind).item()
                              for mips, src, ref in data]))


def window_records(out, ref, positions, size, ppd):
    """Per test source, the gap study's window metrics of an output frame. FLIP from one
    full-frame map. Shared with ceiling.py, so models and oracles are measured alike."""
    fmap, _ = m.flip_map(out, ref, ppd, floor=m.flip_floor(EPS))   # exposure floor: decision log 2026-11-03
    pos_out = np.maximum(out, 0)
    h, w = ref.shape[:2]
    records = []
    for x, y in positions:
        w_out, w_ref = m.window(pos_out, x, y, size), m.window(ref, x, y, size)
        records.append({"x": int(x), "y": int(y), "H": field_height(x, y, w, h),
                        "flip": float(m.window(fmap, x, y, size).mean()),
                        "log_mse": m.log_loss(w_out, w_ref, EPS),
                        "energy": m.energy_error(m.window(out, x, y, size), w_ref)})
    return records


def evaluate(module, params, img, ref, positions, size, ppd, source="mip0"):
    """Per test source, the gap study's window metrics of the model's output."""
    mips, src, _ = prepare(module, img)
    with torch.no_grad():
        out = glare(module, mips, src, params, source).cpu().numpy()
    return window_records(out, ref, positions, size, ppd), float((out < 0).mean())


def view_sources(positions, w, h):
    """The test sources nearest VIEW_H, one per target: the rows of views.py's figures."""
    hs = np.array([field_height(x, y, w, h) for x, y in positions])
    return [positions[int(np.argmin(np.abs(hs - t)))] for t in VIEW_H]


def view_crop(img, x, y):
    """VIEW_PX crop centred on (x, y), zero outside the frame: the 6 % grid margin is 65 px, less
    than half a crop, but the PSF (127 px) always lies inside the frame."""
    r = VIEW_PX // 2
    padded = np.pad(img, ((r, r), (r, r), (0, 0)))
    return padded[int(y):int(y) + VIEW_PX, int(x):int(x) + VIEW_PX]


def save_views(module, params, img, positions, path, source="mip0"):
    """What views.py draws, saved with every run so a figure never has to re-run an old
    architecture: the output (float32, unclamped) in VIEW_PX crops around the view sources,
    and each level's blend t over the screen (and the spike mix weights, if any), both from the
    Slang model."""
    h, w = img.shape[:2]
    mips, src, _ = prepare(module, img)
    extra = {}
    with torch.no_grad():
        out = glare(module, mips, src, params, source).cpu().numpy()
        gv, gu = np.meshgrid(np.linspace(0, 1, BLEND_GRID[0]), np.linspace(0, 1, BLEND_GRID[1]),
                             indexing="ij")
        uv = torch.tensor(np.stack([gu.ravel(), gv.ravel()], -1), dtype=torch.float32,
                          device="cuda")
        blend = np.stack([module.filter_weights(uv=uv, size=spy.int2(w, h),
                                                params=level_params(module, params, k))
                          [:, 3].cpu().numpy().reshape(BLEND_GRID)     # the low tap: t
                          for k in range(len(mips))])
        if has_spikes(module, params.numel(), len(mips)):
            mix = module.spike_mix(uv=uv, params=params,
                                   offset=len(mips) * level_mlp(module).param_count)
            extra["spike_mix"] = mix.cpu().numpy().T.reshape(3, *BLEND_GRID).astype(np.float32)
    pos = view_sources(positions, w, h)
    crops = np.stack([view_crop(out, x, y) for x, y in pos])
    np.savez_compressed(path, pos=np.array(pos), h=[field_height(x, y, w, h) for x, y in pos],
                        out=crops.astype(np.float32), blend=blend.astype(np.float32), **extra)


def summarize(records):
    cell = {}
    for name, lo, hi in BINS:
        sel = [r for r in records if lo <= r["H"] < hi]
        cell[name] = {k: float(np.median([r[k] for r in sel])) for k in ("flip", "log_mse", "energy")}
        cell[name]["n"] = len(sel)
    cell["edge_over_centre_flip"] = cell["edge"]["flip"] / cell["centre"]["flip"]
    return cell


def strongest_baselines(run):
    """Phase 0 point-grid summaries: per baseline, whichever fit (squared or absolute log
    loss, metrics.LOG_LOSSES) has the lower mean FLIP over the H bins, i.e. the strongest
    tuned baseline (decision log 2026-11-03). The absolute-loss pyramid fit turns bloom off,
    so this is the squared-loss pyramid today. Returns (summaries, fit loss per baseline,
    the squared-loss gap_eval for the shared settings)."""
    gaps = {loss: json.loads((ROOT / "outputs" / "01_gap_study" / run / "eval" /
                              f"gap_eval{m.fit_suffix(loss)}.json").read_text())
            for loss in m.LOG_LOSSES}
    settings = ("window_px", "flip_ppd", "eps")
    for loss, g in gaps.items():
        assert g.get("fit_loss", "sq") == loss, f"gap_eval for {loss} wasn't fitted with it"
        assert all(g[k] == gaps["sq"][k] for k in settings), "gap evals differ in settings"

    def mean_flip(loss, b):
        return np.mean([gaps[loss]["summary"]["synthetic"][b][c]["flip"]
                        for c in ("centre", "mid", "edge")])

    fit = {b: min(gaps, key=lambda loss: mean_flip(loss, b)) for b in COMPARE}  # ties: sq
    return ({b: gaps[fit[b]]["summary"]["synthetic"][b] for b in COMPARE}, fit, gaps["sq"])


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


def train(module, data, val_data, init, args, rng):
    """Adam on the log loss, one random training frame per step. The curve logs the mean loss
    over the training and the validation frames at step 0 (after the first gradient call,
    before any update) and every LOG_EVERY steps. slangpy: the first upsample_chain_pass call
    must carry gradients."""
    source = SPIKE_SOURCE[args.arch] or "mip0"
    params = init.clone().cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    curve = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        mips, src, ref = data[rng.integers(len(data))]
        optimizer.zero_grad()
        out = glare(module, mips, src, params, source)
        assert out.grad_fn is not None, "slangpy dropped autograd (see train_pyramid.py)"
        loss = log_loss(out, ref, args.loss)
        loss.backward()
        if step == 1:                               # after the first gradient call, before any update
            curve.append({"step": 0,
                          "train_loss": frames_loss(module, data, params, args.loss, source),
                          "val_loss": frames_loss(module, val_data, params, args.loss, source)})
            print(f"step     0: train loss {curve[0]['train_loss']:.3e}  "
                  f"val loss {curve[0]['val_loss']:.3e}", flush=True)
        optimizer.step()
        schedule.step()
        if step % LOG_EVERY == 0 or step == args.steps:
            full = frames_loss(module, data, params, args.loss, source)
            val = frames_loss(module, val_data, params, args.loss, source)
            curve.append({"step": step, "loss": loss.item(), "train_loss": full, "val_loss": val})
            print(f"step {step:5d}: train loss {full:.3e}  val loss {val:.3e}  (this frame "
                  f"{loss.item():.3e}, {time.perf_counter() - t0:.0f}s)", flush=True)
    return params.detach(), curve, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    ap.add_argument("--study", choices=STUDIES, required=True,
                    help="output folder: Step 1 setup, the baseline, or one Step 2 ablation")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["abs", "sq"], default="abs",
                    help="absolute or squared log error (decision log 2026-11-03)")
    ap.add_argument("--arch", choices=ARCHS, default=ARCHS[0],
                    help="spikes_full / spikes: with the spike term, rays from the input frame / "
                         "mip 0; offsets: without (the coma model)")
    ap.add_argument("--init", choices=["fitted", "neutral"], default="fitted")
    ap.add_argument("--t-pass", type=float, default=T_PASS,
                    help="--init fitted: blend of the levels URP leaves unused")
    ap.add_argument("--rescore", action="store_true",
                    help="don't train: re-evaluate this run's saved weights (fits/) and keep its "
                         "training curve")
    args = ap.parse_args()
    tag = f"_{args.arch}_{args.loss}_{args.init}_lr{args.lr:g}"
    if args.init == "fitted" and args.t_pass != T_PASS:
        tag += f"_tpass{args.t_pass:g}"
    if args.seed != 0:
        tag += f"_seed{args.seed}"

    split = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase2_points"]
    with np.load(ROOT / psf_bank_path(args.run)) as z:
        bank = {k: z[k] for k in z.files}
    cam = json.loads(str(bank["config"]))["camera"]
    shape = (cam["height_px"], cam["width_px"], 3)
    phase0, phase0_fit, gap = strongest_baselines(args.run)
    size, ppd = gap["window_px"], gap["flip_ppd"]
    assert size == 2 * bank["psf"].shape[1], "window differs from the gap study's"
    assert gap["eps"] == EPS, "log eps differs from the gap study's"

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    module = spy.Module.load_from_file(device, "train_glare.slang")

    frames = train_frames(bank, *split["train_grid"])
    val_sel = frames.pop(split["val_frame"])
    val_img, val_ref = render(bank, val_sel, shape)
    val_data = [prepare(module, val_img, val_ref)]
    data = [prepare(module, *render(bank, sel, shape)) for sel in frames]
    n_levels = len(data[0][0])                      # pyramid levels + the full-resolution one
    if args.init == "fitted":
        fit = json.loads((ROOT / "outputs" / "01_gap_study" / args.run / "fits" / "pyramid_fit.json")
                         .read_text())["best"]
        init = fitted_pyramid_init(gen, level_mlp(module), n_levels, fit, args.t_pass)
    else:
        init = torch.cat([init_params(gen, level_mlp(module)) for _ in range(n_levels)])
    source = SPIKE_SOURCE[args.arch]
    if source:                                      # drawn after the levels: their init is unchanged
        init = torch.cat([init, spike_init(gen, module, decay=SPIKE_DECAY[source])])
    print(f"{len(frames)} training frames ({min(map(len, frames))}-{max(map(len, frames))} "
          f"sources), 1 validation frame ({len(val_sel)} sources), {n_levels} levels, "
          f"{init.numel()} parameters")

    def out(stage):
        return Path(run_dir(EXPERIMENT, args.run, stage, args.study))

    fits = out("fits")
    eval_path = out("eval") / f"train_points{tag}_eval.json"
    if args.rescore:
        params = torch.from_numpy(np.load(fits / f"train_points{tag}_params.npy")).cuda()
        old = json.loads(eval_path.read_text())
        assert old["args"]["steps"] == args.steps and old["args"]["seed"] == args.seed, (
            "pass the run's own --steps / --seed")
        curve, seconds = old["curve"], old["seconds"]
    else:
        params, curve, seconds = train(module, data, val_data, init, args, rng)

    test_img = read_hdr(str(ROOT / "data" / "inputs" / "pointgrid.exr"))[..., :3]
    test_ref = read_hdr(str(ROOT / REFS[args.run] / "pointgrid_ref_exact.exr"))[..., :3]
    test_pos = [tuple(p) for p in bank["pos_px"][bank["split"] == "test"]]
    val_pos = [tuple(p) for p in bank["pos_px"][val_sel]]
    results = {}
    for name, p in (("init", init.cuda()), ("trained", params)):
        records, negative = evaluate(module, p, test_img, test_ref, test_pos, size, ppd,
                                     source or "mip0")
        val_records, _ = evaluate(module, p, val_img, val_ref, val_pos, size, ppd,
                                  source or "mip0")
        results[name] = {"summary": summarize(records), "negative_fraction": negative,
                         "records": records,
                         "val": {"flip": float(np.mean([r["flip"] for r in val_records])),
                                 "log_mse": float(np.mean([r["log_mse"] for r in val_records])),
                                 "loss": frames_loss(module, val_data, p, args.loss,
                                                     source or "mip0"),
                                 "records": val_records}}
    trained = results["trained"]["summary"]
    answers = verdict(trained, phase0)

    export = out("export")
    np.save(fits / f"train_points{tag}_params.npy", params.cpu().numpy())
    save_views(module, params, test_img, test_pos, out("eval") / f"train_points{tag}_views.npz",
               source or "mip0")
    params.cpu().numpy().astype("<f4").tofile(export / f"glare_params{tag}.bytes")
    (export / f"glare_params{tag}.json").write_text(json.dumps(       # the engines' layout file
        engine_layout(module, n_levels, source is not None, source), indent=2))
    eval_path.write_text(json.dumps({
        "environment": environment(device), "args": vars(args), "eps": EPS, "flip_floor": m.flip_floor(EPS),
        "intensity": INTENSITY, "window_px": size, "flip_ppd": ppd, "levels": n_levels,
        "param_count": level_mlp(module).param_count, "arch": args.arch, "spike_source": source,
        "spike_param_count": init.numel() - n_levels * level_mlp(module).param_count,
        "frames": frames, "val_frame": val_sel, "seconds": seconds,
        "curve": curve,
        "phase0": phase0, "phase0_fit": phase0_fit, "answers": answers, "results": results},
        indent=2))

    rows = [(f"{COMPARE[b]} ({phase0_fit[b]} fit)", phase0[b]) for b in COMPARE]
    start = {"fitted": f"fitted pyramid, energy-normalised, t_pass {args.t_pass:g}",
             "neutral": "t = 0.5"}[args.init]
    rows += [(f"Init ({start}, before training)", results["init"]["summary"]),
             ("**Trained (this run)**", trained)]
    report = (f"# Phase 2, {args.study}: `{args.arch}` on exact point-light PSFs ({args.run})\n\n"
              f"{table(rows)}\n\n"
              f"Medians over the 35 point-grid windows ({size} px). Bins: centre H < 1/3 "
              f"(n = {trained['centre']['n']}), mid (n = {trained['mid']['n']}), edge >= 2/3 "
              f"(n = {trained['edge']['n']}). Negative output texels: "
              f"{results['trained']['negative_fraction']:.2%}. Learning rate {args.lr:g}; "
              f"{args.loss} log loss, training ({len(data)} frames) "
              f"{curve[0]['train_loss']:.4f} at step 0, "
              f"{curve[-1]['train_loss']:.4f} at step {curve[-1]['step']}.\n\n"
              f"Validation frame ({len(val_sel)} sources; selects between runs, decision log "
              f"2026-11-03): mean FLIP {results['init']['val']['flip']:.4f} at init, "
              f"**{results['trained']['val']['flip']:.4f}** trained; {args.loss} log loss "
              f"{results['init']['val']['loss']:.4f} at init, "
              f"{results['trained']['val']['loss']:.4f} trained.\n\n"
              f"Pre-registered questions (decision log 2026-10-31):\n"
              f"- (1) beats the fitted pyramid in every bin: **{answers['beats_fitted_pyramid']}**\n"
              f"- (2) closes the edge gap (edge FLIP < {phase0['fft_centre']['edge']['flip']:.3f} "
              f"and edge/centre <= {EDGE_RATIO:g}x): **{answers['closes_edge_gap']}**\n"
              f"- expected: centre FLIP above the FFT centre kernel's "
              f"({phase0['fft_centre']['centre']['flip']:.3f}): "
              f"**{answers['centre_above_fft_centre_as_expected']}**\n")
    (out("report") / f"train_points{tag}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
