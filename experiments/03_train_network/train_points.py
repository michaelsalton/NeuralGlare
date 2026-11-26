"""Phase 2, step 1: train the Phase 1 architecture on exact point-light PSFs (the baseline).

Model: the Phase 1 pyramid (2 taps per level, grey weights, input uv) plus one level at full
resolution, whose "mip" is the input frame itself:
    out = (1 - t(uv)) * input + t(uv) * bilinear(up[0]),   t = sigmoid(MLP(uv))
(Step 2, coma: every level has 4 taps, w = softmax(MLP(uv)) over mip(x),
mip(x + o), mip(x - o) and the level below, o = H * d_hat texels; glare.slang. Step 2, spikes,
--arch spikes: 14 streaked rays from mip 0, two families, mixed into the full-resolution output by
softmax(SpikeMLP(uv)); glare.slang. --arch per_channel: every softmax, the levels' and the spike
mix's, has one set of logits per colour channel, the rays streaked from the input frame.)
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
Outputs are named train_points_<arch>_<loss>_<init>_lr<lr>, plus _mips<n> for --mips other than
LEGACY_MIPS (study mip_levels, decision log 2026-11-10) and _seed<n> for --seed other than 0
(seed spread, decision log 2026-11-03). Older runs kept their names: before
the lerp (2-weight networks), and before the validation frame (squared loss on all 9 frames,
train_points_<arch>_<init>_lr<lr>). Runs before 2026-11-10 have 6 mips: re-score them with --mips 6.

    uv run python experiments/03_train_network/train_points.py --run default --study per_channel --mips 6
    uv run python experiments/03_train_network/train_points.py --run default --study per_channel --mips 6 --seed 1
                                                    # delta re-measurement (decision log 2026-11-04)
    uv run python experiments/03_train_network/train_points.py --run default --study mip_levels --mips 2
    uv run python experiments/03_train_network/train_points.py --run default --study mip_levels
                                                    # 1 mip, the default (decision log 2026-11-10)
    uv run python experiments/03_train_network/train_points.py --run default --study mip_levels --mips 0
                                                    # no blur: the low tap reads the input
    uv run python experiments/03_train_network/train_points.py --run default --study tent_upsample --mips 2 --t-low 0.05
                                                    # tent upsample, light starting in mip 0 (2026-11-12)

The low tap is a 3x3 tent since 2026-11-12 (glare.slang sample_tent), bilinear before: runs from
before that date were trained with bilinear, so --rescore on them scores a different model.
--t-low sets the pyramid levels' (not the full-resolution level's) low-tap share at init, in
place of the fit's scatter; tagged _tlow<t>.

    uv run python experiments/03_train_network/train_points.py --run default --study shoulder_tap
                                                    # 5th tap, lr 1e-3, 1 mip: the defaults (2026-11-13)
    uv run python experiments/03_train_network/train_points.py --run default --study shoulder_tap --tent-share 0.01

--arch shoulder (2026-11-13) adds a fifth tap to every level, the 3x3 tent of the level's own mip:
at the full-resolution level a round +-1 px shoulder that works at the frame centre, where the
offset taps collapse onto the source (o = 0). --tent-share is its share at init (tagged
_tshare<s> when not TENT_SHARE). The learning rate defaults to 1e-3 since 2026-11-13 (LR); the
tag always carries it, so older lr 1e-4 names keep their meaning.

    uv run python experiments/03_train_network/train_points.py --run default --study core_loss --ee-weight <w>
                                                    # core (encircled-energy) loss (2026-11-16)

--ee-weight (2026-11-16) adds the core loss to the objective at that weight (tag _ee<w>): per
source, per ring of round(distance) 0..EE_RADIUS px, per channel, the |log| error of the ring's
light as a share of the source (ee_loss). The per-pixel log loss weighs a source's ~9 core pixels
against its ~64k halo pixels, so it can't tell cores apart; the core loss counts each source once
per ring. Every report logs the validation core loss, whatever the weight.

    uv run python experiments/03_train_network/train_points.py --run default --study wide_core --ee-weight 0.02
                                                    # 6th tap, the own-mip 5x5 binomial (2026-11-16)
    uv run python experiments/03_train_network/train_points.py --run default --study wide_core --ee-weight 0.02 --wide-share 0.01

--arch wide (2026-11-16) adds a sixth tap to every level, the 5x5 binomial of the level's own mip
(sample_binomial5, a Gaussian of sigma = 1 texel): at the full-resolution level the off-axis +-2 px
shoulder. --wide-share is its share at init (tagged _wshare<s> when not WIDE_SHARE); TENT_SHARE
is 0.70 for it (0.85 for shoulder), both from the reference at the frame centre. The shoulder runs
(shoulder_tap, core_loss) can't be re-scored: glare.slang runs 6 taps.

    uv run python experiments/03_train_network/train_points.py --run default --study ray_cross --arch cross --ee-weight 0.02
                                                    # per-ray cross filter (2026-11-19)

--arch cross (2026-11-19) filters each ray across itself after its last streak pass: 3 taps at
-1 / 0 / +1 px along the ray's normal, weights softmax(RayMLP(e_k . v, n_k . v, family)), v = H *
d_hat. RayMLP starts as the plain ray and is pre-fitted to per-source, per-ray NNLS taps measured
on the training frames' PSFs (cross_prefit.py) before training.

    uv run python experiments/03_train_network/train_points.py --run default --study ray_bright --arch bright --ee-weight 0.02
                                                    # per-ray brightness (2026-11-19)

--steps 0 exports the untrained init (tag _init), for a timing-only test (cost doesn't depend on
the weights; efficiency step 2a, decision log 2026-11-24).

--arch bright (2026-11-19) is cross plus a gain per ray: g_k = 7 softmax over the family's 7 rays
of BrightMLP(e_k . v, n_k . v, family), so light moves between a family's rays (SpikeMLP still
sets each family's total). Each ray's cross pass adds e^{z_k} times the ray and the composite
divides each family by its normaliser instead of 7. BrightMLP starts at g = 1 (the cross init
exactly) and is pre-fitted to the per-ray light ratios of the training frames' PSFs (cross_prefit.py).

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
from neuralglare.storage import FORMATS as RAY_STORAGES
from neuralglare.storage import store_as

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "01_gap_study"))
sys.path.insert(0, str(ROOT / "experiments" / "02_slang_plumbing"))
from evaluate_gap import BINS, field_height
from exact_pointgrid import splat
from train_levels import (
    bright_mlp,
    engine_layout,
    environment,
    init_params,
    level_mlp,
    ray_mlp,
    spike_layout,
    spike_parts,
)
from train_pyramid import level_params, slang_mips

EXPERIMENT = "03_train_network"
STUDIES = ("setup", "baseline", "radial_input", "coma", "spikes", "spikes_full",
           "per_channel", "mip_levels", "tent_upsample", "shoulder_tap",
           "core_loss", "wide_core", "ray_cross", "ray_bright")             # one per ablation
# --arch, part of every output name, and the image the spike rays are streaked from (None: no
# spike term). glare.slang runs wide (shoulder plus a sixth tap per level, the 5x5 binomial of the
# level's own mip; study wide_core, decision log 2026-11-16), cross (wide plus a 3-tap filter
# across each ray after its last streak pass, RayMLP; study ray_cross, 2026-11-19) and bright
# (cross plus a gain per ray, BrightMLP; study ray_bright, 2026-11-19). Earlier, which glare.slang no
# longer runs: shoulder (per_channel plus a fifth tap per level, the tent of the level's own mip;
# shoulder_tap, core_loss), per_channel (4 taps per level, one softmax per colour channel; per_channel,
# mip_levels, tent_upsample), and with grey weights lerp (MLP(uv), Step 1), lerp_h (MLP(H),
# radial_input), offsets (4 taps per level, study coma), spikes (rays from mip 0) and spikes_full
# (rays from the input frame).
SPIKE_SOURCE = {"wide": "input", "cross": "input", "bright": "input"}
ARCHS = tuple(SPIKE_SOURCE)
CROSS_ARCHS = ("cross", "bright")   # archs with the per-ray cross filter (decision log 2026-11-19)
BRIGHT_ARCHS = ("bright",)          # archs with per-ray brightness (decision log 2026-11-19)
CROSS_EPS = 1e-3                    # each side tap's weight at init, before the pre-fit
OFFSET_SHARE = 0.01                 # each offset tap's share of the mip's weight at init
                                    # (decision log 2026-11-04)
TENT_SHARE = 0.70                   # full-resolution level's own-tent and 5x5 binomial tap shares at
WIDE_SHARE = 0.15                   # init, from the reference (2026-11-16): the best mix of centre,
                                    # tent, low tap and binomial for the validation source nearest the
                                    # frame centre (H = 0.05) under the core loss is 0.14 / 0.70 /
                                    # 0.02 / 0.15. The 5-tap shoulder arch had TENT_SHARE 0.85 (tent
                                    # alone: 1 - 0.75 s = 0.357 at H = 0, decision log 2026-11-13)
TENT_SHARE_PYRAMID = 0.01           # the pyramid levels' own-tent and binomial shares at init
WIDE_SHARE_PYRAMID = 0.01           # (as OFFSET_SHARE)
LR = 1e-3                           # default learning rate: the tent_upsample lr study (2026-11-13)
SPIKE_SHARE = 1e-3                  # each ray family's share of the light at init: the bank's spikes
                                    # carry ~0.25 % of the PSF's light, ~1.2e-3 per family
SPIKE_DECAY = {"mip0": 8.0,         # init ray profile exp(-d / decay), d in texels of the rays'
               "input": 16.0}       # source: 16 full-resolution px either way (2026-11-04, -05)
T_PASS = 1e-4                       # default blend of a pass-through level (sigmoid never
                                    # reaches 0); its gradient carries t(1 - t), so a tiny t
                                    # nearly freezes the level (decision log 2026-11-02)
INTENSITY = 1e4                     # per channel, as the Phase 0 point grid
EPS = 1e-7 * INTENSITY              # 7 decades below the source; = the gap study's eps
EE_RADIUS = 4                       # core loss: rings 0..4 px around each source, by
                                    # round(distance) (decision log 2026-11-16)
EE_DELTA = 1e-4                     # floor of a ring's light, as a share of the source: EPS's
                                    # role for rings
MIPS = 1                            # default mip count: the mip_levels study's best, no mip 1
                                    # disc (decision log 2026-11-10)
LEGACY_MIPS = 6                     # Phase 1's chain: runs before 2026-11-10, untagged
LOG_EVERY = 250
COMPARE = {"fft_centre": "Phase 0: FFT, exact centre PSF",
           "pyramid": "Phase 0: pyramid bloom (fitted)"}
EDGE_RATIO = 2.0                    # criterion (2), the Phase 0 proceed rule's ratio
VIEW_H = (0.0, 0.5, 0.88)           # views.py: the test sources nearest these field heights
VIEW_PX = 160                       # views.py: crop size around each view source
BLEND_GRID = (27, 48)               # views.py: (rows, cols) of the screen grid for the blend maps
CHANNELS = 3                        # glare.slang's softmax_rgb: logits channel-major, R, G, B
VIEW_CHANNEL = 1                    # the channel saved as `blend` / `spike_mix` (G), as before
RAY_STORE = "fp32"                  # the format the engine stores the streak passes' outputs in (rayA /
                                    # rayB; family sums stay float32, spikes()); a name from
                                    # neuralglare.storage.FORMATS; --ray-storage, efficiency
                                    # step 1a (2026-11-24)
RAY_STACK = True                    # brightness at the output's resolution: the composite reads every
                                    # ray's last streak pass from a stack and forms the families itself
                                    # (glare.slang spike_composite_stack; efficiency step 1c, decision
                                    # log 2026-11-24), as GlareChain.RayStack. Same arithmetic as the
                                    # per-ray passes (test_spikes.py test_stack_*)
RAY_PRE_EXPOSURE = 2.0 ** -4        # the rays' pre-exposure k (glare.slang streak; decision log
                                    # 2026-11-24): a real sun (3.8e5) fits fp16's 65504, the 1e-3 loss
                                    # floor of a 1e4 source lands at fp16's smallest normal; a power
                                    # of two, so float32 results don't depend on it. As GlareChain's
                                    # DefaultPreExposure


def ray_scale(k, reads_source, writes_family):
    """A ray pass's scale (glare.slang streak): k if it reads the source, 1/k if it writes the
    family sum, both if it does both. As GlareChain.RayScale."""
    return (k if reads_source else 1.0) * (1.0 / k if writes_family else 1.0)


def stored(x, store):
    """x as the engine stores it between passes (format name `store`), read back as float32:
    rounded toward zero as the RTX 2070's UAV stores on D3D12 (D3D allows that or round-to-nearest;
    decision log 2026-11-24: modelling round-to-nearest left Unity's fp16 ray families 0.1 % darker
    on every channel). neuralglare.storage.store_as; straight-through gradient."""
    return store_as(x, store)


def pyramid(mips):
    """The fixed downsample chain with `mips` mips: URP's, without threshold or clamp (as Phase 1)."""
    return URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=mips)


def level_count(mips):
    """Level networks for a list of mips: one per mip below the coarsest, then the
    full-resolution one. 0 mips still has the full-resolution level (decision log 2026-11-10)."""
    return max(len(mips), 1)


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


def fitted_pyramid_init(gen, net, n_levels, fit, t_pass=T_PASS, t_low=None,
                        tent_share=TENT_SHARE, wide_share=WIDE_SHARE):
    """Phase 0's fitted pyramid bloom as a constant blend per level (output layer 0, biases
    the log of the tap weights, the low tap's = t). URP with m mips: up[k] = lerp(mip_k, up(up[k+1]), s) for k < m-1,
    up[m-1] = mip_{m-1}: t = s, then t = t_pass. URP's last step out = input + I * up(up[0])
    adds light; the energy-conserving lerp with the same shape is that divided by 1 + I,
    t = I / (1 + I). Every channel starts with the same logits (grey). Hidden layers stay random
    so uv can still be learned. A model with fewer mips than the fit truncates it: its coarsest
    mip takes the light the fit sends to the dropped ones (decision log 2026-11-10). t_low, if
    given, replaces every pyramid level's t (not the full-resolution level's): with 2 mips, a
    small t_low starts the light in mip 0 rather than the wide mip 1 (decision log 2026-11-12).
    The own-tent tap takes tent_share of the full-resolution level's 1 - t (the reference's 1 px
    shoulder) and TENT_SHARE_PYRAMID of each pyramid level's (decision log 2026-11-13); the own-mip
    5x5 binomial tap wide_share and WIDE_SHARE_PYRAMID alike (the ±2 px shoulder, 2026-11-16)."""
    p = URPBloomParams(**{**fit, "tint": tuple(fit["tint"])})
    assert p.tint == (1.0, 1.0, 1.0), "a tint isn't representable by grey weights"
    _, _, s = shader_constants(p)
    m = p.max_iterations
    blends = [s if k < m - 1 else t_pass for k in range(n_levels - 1)]
    if t_low is not None:
        blends = [t_low] * (n_levels - 1)
    blends.append(p.intensity / (1 + p.intensity))      # full-resolution level
    hidden_end = net.inputs * net.hidden + net.hidden  # mlp.slang layout: W1, b1, W2, b2
    shares = [(TENT_SHARE_PYRAMID, WIDE_SHARE_PYRAMID)] * (n_levels - 1) + [(tent_share, wide_share)]
    nets = []
    for t, (sh, wd) in zip(blends, shares):
        w = init_params(gen, net)
        w[hidden_end:-net.outputs] = 0                  # W2: output independent of uv
        # b2 = log of the target weights: softmax(log w) = w, since they sum to 1. The low tap
        # keeps t; of the mip's 1 - t, the offset taps take OFFSET_SHARE each, the own-mip tent sh
        # and the own-mip binomial wd. glare.slang's order: mip, mip(+o), mip(-o), low, tent(mip),
        # binomial5(mip)
        assert net.outputs == 6 * CHANNELS, "init written for glare.slang's 6 taps per channel"
        e = OFFSET_SHARE
        assert 2 * e + sh + wd < 1, "the centre tap needs a positive share"
        taps = [(1 - t) * (1 - 2 * e - sh - wd), (1 - t) * e, (1 - t) * e, t, (1 - t) * sh,
                (1 - t) * wd]
        w[-net.outputs:] = torch.tensor(np.tile(np.log(taps), CHANNELS), dtype=w.dtype)
        nets.append(w)
    return torch.cat(nets)


def spike_init(gen, module, share=SPIKE_SHARE, decay=SPIKE_DECAY):
    """The spike section: SpikeMLP with its output layer 0 and biases log(1 - 2 share, share,
    share) in every channel, so each family starts with `share` of the light everywhere, then the
    profile logits z[p][n] = -n 4^p / decay for both families. A ray's weight at distance d is
    the product of one softmax weight per base-4 digit of d, so these logits give exactly
    exp(-d / decay) (normalised over 0..63 texels)."""
    lay = spike_layout(module)
    net = lay.net
    w = init_params(gen, net)
    w[net.inputs * net.hidden + net.hidden:-net.outputs] = 0   # W2: output independent of uv
    assert net.outputs == 3 * CHANNELS, "init written for glare.slang's 3 mix weights per channel"
    w[-net.outputs:] = torch.tensor(np.tile(np.log([1 - 2 * share, share, share]), CHANNELS),
                                    dtype=w.dtype)
    z = [-n * lay.taps ** p / decay for p in range(lay.passes) for n in range(lay.taps)]
    return torch.cat([w, torch.tensor(z * 2, dtype=w.dtype)])


def cross_init(gen, module, eps=CROSS_EPS):
    """RayMLP with its output layer 0 and biases log(eps, 1 - 2 eps, eps): every ray starts as the
    plain ray to within 2 eps of its light, everywhere on the screen. cross_prefit then fits it."""
    net = ray_mlp(module)
    w = init_params(gen, net)
    w[net.inputs * net.hidden + net.hidden:-net.outputs] = 0   # W2: output independent of the field
    assert net.outputs == 3, "init written for glare.slang's 3 cross taps"
    w[-net.outputs:] = torch.tensor(np.log([eps, 1 - 2 * eps, eps]), dtype=w.dtype)
    return w


def bright_init(gen, module):
    """BrightMLP with its output layer and bias 0: every logit 0, so every gain is 1 everywhere and
    the term starts as the cross model exactly. cross_prefit then fits it."""
    net = bright_mlp(module)
    w = init_params(gen, net)
    w[net.inputs * net.hidden + net.hidden:] = 0                # W2 and b2
    return w


def load_bank(run):
    """A run's point-grid split (split.toml phase2_points), PSF bank and frame shape (H, W, 3)."""
    split = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase2_points"]
    with np.load(ROOT / psf_bank_path(run)) as z:
        bank = {k: z[k] for k in z.files}
    cam = json.loads(str(bank["config"]))["camera"]
    return split, bank, (cam["height_px"], cam["width_px"], 3)


def eval_frames(run, split, bank, shape):
    """The frames every run is scored on, as main() scores them and unity_parity.py score scores
    the engine's output (efficiency step 1e, decision log 2026-11-24): {"val": (img, ref,
    positions), "test": (img, ref, positions)}. The validation frame is held out of the training
    frames (split.toml val_frame); the test frame is the Phase 0 point grid and its exact
    reference."""
    val_sel = train_frames(bank, *split["train_grid"])[split["val_frame"]]
    val_img, val_ref = render(bank, val_sel, shape)
    test_img = read_hdr(str(ROOT / "data" / "inputs" / f"{split['test']}.exr"))[..., :3]
    test_ref = read_hdr(str(ROOT / REFS[run] / f"{split['test']}_ref_exact.exr"))[..., :3]
    return {"val": (val_img, val_ref, [tuple(p) for p in bank["pos_px"][val_sel]]),
            "test": (test_img, test_ref,
                     [tuple(p) for p in bank["pos_px"][bank["split"] == "test"]])}


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


def spikes(module, source, base, params, offset, families=False, cross=False, bright=False,
           store=None, pre_exposure=RAY_PRE_EXPOSURE, stack=None):
    """The spike term (glare.slang): every ray streaked from source (mip 0 or the input frame,
    SPIKE_SOURCE) in its passes, each family's rays summed, then mixed into the full-resolution
    output base by SpikeMLP. offset: where the spike section starts in params (SpikeMLP, then the
    profile logits). cross: each ray filtered across itself after its last pass (RayMLP, after
    the profile logits; decision log 2026-11-19). bright: and weighted by e^{z_k} (BrightMLP,
    after RayMLP), each family divided by its normaliser in the composite. families: also return
    the two family sums (even, odd), for unity_parity.py. store: the format each streak pass's
    output is kept in, as the engine's rayA / rayB (None: RAY_STORE). The family sums and the
    composite stay float32, as Unity's targets: with brightness a family sum holds e^{z_k} times
    each ray before the composite divides by the normaliser, up to ~9x the source (9.1e4 for the
    1e4 point grid), past fp16's 65504. pre_exposure: the rays' k (ray_scale), so a stored streak
    holds k times the light; the engine's must match for parity (unity_run.json preExposure).
    stack: the ray-stack path (RAY_STACK when None; only with brightness and source at base's
    resolution): every ray's last pass stored, stacked, and the composite does the cross filter,
    brightness and family sums. No family sums exist then: families returns None for them."""
    assert cross or not bright, "per-ray brightness needs the cross filter"
    stackable = bright and source.shape == base.shape
    stack = (RAY_STACK and stackable) if stack is None else stack
    assert stackable or not stack, "the ray stack needs brightness and rays at base's resolution"
    store = store or RAY_STORE
    lay = spike_layout(module)
    profile = offset + lay.net.param_count
    cross_offset = profile + lay.profile_count
    h, w = source.shape[:2]
    grid = spy.grid((h, w))
    acc = [torch.zeros_like(source), torch.zeros_like(source)]      # even, odd family
    if cross and torch.is_grad_enabled():
        # slangpy 0.43.1 on Vulkan can't run ray_cross_accumulate_pass's backward with a DiffTensor
        # `acc` that has no gradient buffer (as glare()'s lone level); the zeros' gradient is unused
        acc = [a.requires_grad_() for a in acc]
    k = pre_exposure
    last = []                                       # stack: each ray's last pass, as stored
    for ray in range(lay.rays):
        s = source
        for stage in range(lay.passes if cross else lay.passes - 1):
            s = stored(module.streak_pass(texel=grid, src=s, params=params, profile=profile,
                                          ray=ray, stage=stage,
                                          scale=ray_scale(k, stage == 0, False)),
                       store)
        if stack:
            last.append(s)
        elif cross:                                   # the cross pass does the accumulating
            accumulate = (module.ray_bright_accumulate_pass if bright
                          else module.ray_cross_accumulate_pass)
            acc[ray % 2] = accumulate(texel=grid, out_size=spy.int2(w, h), src=s, acc=acc[ray % 2],
                                      params=params, offset=cross_offset, ray=ray,
                                      scale=ray_scale(k, False, True))
        else:
            acc[ray % 2] = module.streak_accumulate_pass(texel=grid, src=s, acc=acc[ray % 2],
                                                         params=params, profile=profile, ray=ray,
                                                         stage=lay.passes - 1,
                                                         scale=ray_scale(k, lay.passes == 1, True))
    h, w = base.shape[:2]
    if stack:                                       # the stack is stored as the streaks (rayStack)
        out = module.spike_stack_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), base=base,
                                      rays=torch.stack(last), params=params, offset=offset,
                                      scale=ray_scale(k, False, True))
        return (out, None) if families else out
    composite = module.spike_bright_pass if bright else module.spike_pass
    out = composite(texel=spy.grid((h, w)), out_size=spy.int2(w, h), base=base,
                    even=acc[0], odd=acc[1], params=params, offset=offset)
    return (out, acc) if families else out


def glare(module, mips, src, params, source="mip0"):
    """The pyramid (levels 0..n-1), the full-resolution level n, whose mip is the input, and
    the spike term on top if params hold one, its rays streaked from `source` (SPIKE_SOURCE).
    0 mips: the full-resolution level's low image is the input itself, so its low tap is a
    second centre tap (a bilinear fetch at texel centres, at equal size, is exact): no blur."""
    out = mips[-1] if mips else src
    if len(mips) <= 1 and torch.is_grad_enabled():
        # slangpy 0.43.1 on Vulkan can't run the backward of a lone level whose DiffTensor `low`
        # has no gradient buffer (the chain's levels don't hit this); the mip's gradient is unused
        out = out.detach().requires_grad_()
    for k in reversed(range(len(mips) - 1)):
        out = level(module, mips[k], out, params, k)
    out = level(module, src, out, params, level_count(mips) - 1)
    spikes_on, cross_on, bright_on = spike_parts(module, params.numel(), level_count(mips))
    if spikes_on:
        assert source == "input" or mips, "rays from mip 0 need a mip"
        out = spikes(module, src if source == "input" else mips[0], out, params,
                     level_count(mips) * level_mlp(module).param_count, cross=cross_on,
                     bright=bright_on)
    return out


def path_outputs(module, mips, src, params, source="mip0"):
    """The output split by the image its light enters from: the input frame, then each mip.
    Every pass is a weighted sum of taps whose weights depend on uv only, so the output is
    linear in (src, mips) jointly and the parts sum to glare(). The spike rays land in the
    part of the image they're streaked from (SPIKE_SOURCE): the input's, or mip 0's."""
    parts = []
    for keep in range(-1, len(mips)):
        m = [x if k == keep else torch.zeros_like(x) for k, x in enumerate(mips)]
        s = src if keep == -1 else torch.zeros_like(src)
        parts.append(glare(module, m, s, params, source))
    return ["input"] + [f"mip{k}" for k in range(len(mips))], parts


def prepare(module, img, ref=None, *, mips):
    mips_ = slang_mips(module, img, pyramid(mips)) if mips else []   # mip_sizes clips to >= 1
    src = torch.from_numpy(np.ascontiguousarray(img, np.float32)).cuda()
    return mips_, src, (None if ref is None else torch.from_numpy(ref).cuda())


def log_loss(out, ref, kind):
    """Mean log error, log(out + EPS) - log(ref + EPS): absolute ("abs") or squared ("sq")."""
    d = torch.log(out + EPS) - torch.log(ref + EPS)
    return (d.abs() if kind == "abs" else d ** 2).mean()


def ring_index(radius=EE_RADIUS):
    """The (2r+1)^2 patch's offsets (dy, dx) and each one's ring, round(distance); -1 beyond r."""
    d = torch.arange(-radius, radius + 1)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    ring = torch.round(torch.hypot(dx.float(), dy.float())).long()
    ring[ring > radius] = -1
    return dy.flatten(), dx.flatten(), ring.flatten()


def ring_energy(img, src, radius=EE_RADIUS):
    """(sources, rings, 3): img's light in each ring around every source of the input frame src
    (black but for the sources, >= 241 px apart, so the patches never overlap). Pixels outside
    the frame count 0 (two training sources sit at y = 0, 1; their reference is clipped alike)."""
    ys, xs = torch.nonzero(src[..., 0] > 0, as_tuple=True)
    h, w = src.shape[:2]
    dy, dx, ring = (t.to(img.device) for t in ring_index(radius))
    keep = ring >= 0
    py, px = ys[:, None] + dy[keep], xs[:, None] + dx[keep]               # (S, P)
    inside = ((py >= 0) & (py < h) & (px >= 0) & (px < w)).to(img.dtype)
    patch = img[py.clamp(0, h - 1), px.clamp(0, w - 1)] * inside[..., None]   # (S, P, 3)
    zero = torch.zeros(len(ys), radius + 1, 3, device=img.device, dtype=img.dtype)
    return zero.index_add(1, ring[keep], patch)                          # out-of-place: autograd


def ee_loss(out, ref, src):
    """Core loss: mean |log| error of each source's light per ring (encircled energy by ring),
    as shares of the source, per channel. Each source counts once per ring, so its few core
    pixels weigh as much as its halo (decision log 2026-11-16)."""
    e_out = ring_energy(out, src) / INTENSITY
    e_ref = ring_energy(ref, src) / INTENSITY
    return (torch.log(e_out + EE_DELTA) - torch.log(e_ref + EE_DELTA)).abs().mean()


def objective(out, ref, src, args):
    """The training objective: the log loss, plus the core loss at weight args.ee_weight."""
    loss = log_loss(out, ref, args.loss)
    return loss + args.ee_weight * ee_loss(out, ref, src) if args.ee_weight else loss


def frames_loss(module, data, params, args, source="mip0", loss=None):
    """A loss over a set of frames (training or validation): mean of the per-frame losses;
    loss(out, ref, src) defaults to the objective. One frame's loss isn't comparable across
    steps (frames hold 15 or 18 sources)."""
    loss = loss or (lambda out, ref, src: objective(out, ref, src, args))
    with torch.no_grad():
        return float(np.mean([loss(glare(module, mips, src, params, source), ref, src).item()
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


def evaluate(module, params, img, ref, positions, size, ppd, source="mip0", *, mips):
    """Per test source, the gap study's window metrics of the model's output."""
    mips, src, _ = prepare(module, img, mips=mips)
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


def save_views(module, params, img, positions, path, source="mip0", *, mips):
    """What views.py draws, saved with every run so a figure never has to re-run an old
    architecture: the output (float32, unclamped) in VIEW_PX crops around the view sources,
    and each level's blend t over the screen (and the spike mix weights, if any), both from the
    Slang model: VIEW_CHANNEL's as `blend` / `spike_mix`, so views.py reads every run alike, and
    all channels' as `blend_rgb` / `spike_mix_rgb`. Also the output split by path
    (path_outputs), cropped alike, as `paths` / `path_names`."""
    h, w = img.shape[:2]
    mips, src, _ = prepare(module, img, mips=mips)
    extra = {}
    with torch.no_grad():
        out = glare(module, mips, src, params, source).cpu().numpy()
        names, parts = path_outputs(module, mips, src, params, source)
        parts = [p.cpu().numpy() for p in parts]
        err = np.abs(sum(parts) - out).max() / np.abs(out).max()
        assert err < 1e-5, f"paths don't sum to the output ({err:.1e}): a pass isn't linear"
        gv, gu = np.meshgrid(np.linspace(0, 1, BLEND_GRID[0]), np.linspace(0, 1, BLEND_GRID[1]),
                             indexing="ij")
        uv = torch.tensor(np.stack([gu.ravel(), gv.ravel()], -1), dtype=torch.float32,
                          device="cuda")
        blend = np.stack([[module.filter_weights(uv=uv, size=spy.int2(w, h),
                                                 params=level_params(module, params, k), channel=c)
                           [:, 3].cpu().numpy().reshape(BLEND_GRID)    # the low tap: t
                           for c in range(CHANNELS)] for k in range(level_count(mips))])
        extra["blend_rgb"] = blend.astype(np.float32)                  # (level, channel, row, col)
        for key, tap in (("tent_rgb", 4), ("wide_rgb", 5)):   # the own-mip tent / binomial taps
            extra[key] = np.stack([[module.tap_weight(uv=uv, params=level_params(module, params, k),
                                                      channel=c, tap=tap).cpu().numpy().reshape(BLEND_GRID)
                                    for c in range(CHANNELS)] for k in range(level_count(mips))]
                                  ).astype(np.float32)
        spikes_on, cross_on, bright_on = spike_parts(module, params.numel(), level_count(mips))
        if spikes_on:
            mix = np.stack([module.spike_mix(uv=uv, params=params,
                                             offset=level_count(mips) * level_mlp(module).param_count,
                                             channel=c).cpu().numpy().T.reshape(3, *BLEND_GRID)
                            for c in range(CHANNELS)]).astype(np.float32)   # (channel, weight, ...)
            extra |= {"spike_mix": mix[VIEW_CHANNEL], "spike_mix_rgb": mix}
        if cross_on:                                # the cross filter's taps over the screen
            v = ((uv - 0.5) * torch.tensor([w, h], dtype=torch.float32, device="cuda")
                 / float(np.hypot(w / 2, h / 2)))                       # field_offset at uv
            off = (level_count(mips) * level_mlp(module).param_count
                   + spike_layout(module).param_count)
            extra["cross"] = np.stack([
                module.cross_mix(v=v, params=params, offset=off, ray=k).cpu().numpy()
                .T.reshape(3, *BLEND_GRID) for k in range(spike_layout(module).rays)]
            ).astype(np.float32)                                         # (ray, tap, row, col)
        if bright_on:                               # each ray's gain over the screen
            off_b = off + ray_mlp(module).param_count
            extra["bright"] = np.stack([
                module.ray_gain(v=v, params=params, offset=off_b, ray=k).cpu().numpy()
                .reshape(BLEND_GRID) for k in range(spike_layout(module).rays)]
            ).astype(np.float32)                                         # (ray, row, col)
    pos = view_sources(positions, w, h)
    crops = np.stack([view_crop(out, x, y) for x, y in pos])
    extra["paths"] = np.stack([[view_crop(p, x, y) for x, y in pos]
                               for p in parts]).astype(np.float32)   # (path, view, y, x, rgb)
    extra["path_names"] = np.array(names)
    np.savez_compressed(path, pos=np.array(pos), h=[field_height(x, y, w, h) for x, y in pos],
                        out=crops.astype(np.float32),
                        blend=blend[:, VIEW_CHANNEL].astype(np.float32), **extra)


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
        loss = objective(out, ref, src, args)
        loss.backward()
        if step == 1:                               # after the first gradient call, before any update
            terms = {"train_log_loss": frames_loss(module, data, params, args, source,
                                                   lambda o, r, s: log_loss(o, r, args.loss)),
                     "train_core_loss": frames_loss(module, data, params, args, source, ee_loss)}
            curve.append({"step": 0,
                          "train_loss": frames_loss(module, data, params, args, source),
                          "val_loss": frames_loss(module, val_data, params, args, source)} | terms)
            print(f"step     0: train loss {curve[0]['train_loss']:.3e}  "
                  f"val loss {curve[0]['val_loss']:.3e}  (log loss {terms['train_log_loss']:.3e}, "
                  f"core loss {terms['train_core_loss']:.3e})", flush=True)
        optimizer.step()
        schedule.step()
        if step % LOG_EVERY == 0 or step == args.steps:
            full = frames_loss(module, data, params, args, source)
            val = frames_loss(module, val_data, params, args, source)
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
    ap.add_argument("--lr", type=float, default=LR,
                    help="Adam learning rate (default 1e-3 since 2026-11-13; 1e-4 before)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["abs", "sq"], default="abs",
                    help="absolute or squared log error (decision log 2026-11-03)")
    ap.add_argument("--arch", choices=ARCHS, default=ARCHS[0],
                    help="wide: shoulder's model (per-channel weights, own-mip tent) plus a sixth "
                         "tap per level, the 5x5 binomial of the level's own mip; cross: wide plus "
                         "a 3-tap filter across each ray, RayMLP, pre-fitted; bright: cross plus a "
                         "gain per ray, BrightMLP, pre-fitted (decision log 2026-11-19)")
    ap.add_argument("--init", choices=["fitted", "neutral"], default="fitted")
    ap.add_argument("--t-pass", type=float, default=T_PASS,
                    help="--init fitted: blend of the levels URP leaves unused")
    ap.add_argument("--rescore", action="store_true",
                    help="don't train: re-evaluate this run's saved weights (fits/) and keep its "
                         "training curve")
    ap.add_argument("--mips", type=int, default=MIPS,
                    help="mips in the pyramid (study mip_levels; decision log 2026-11-10)")
    ap.add_argument("--t-low", type=float, default=None,
                    help="pyramid levels' low-tap share at init, in place of the fit's scatter "
                         "(--init fitted; study tent_upsample, decision log 2026-11-12)")
    ap.add_argument("--tent-share", type=float, default=TENT_SHARE,
                    help="full-resolution level's own-tent tap share at init (--init fitted; "
                         "study shoulder_tap, decision log 2026-11-13)")
    ap.add_argument("--wide-share", type=float, default=WIDE_SHARE,
                    help="full-resolution level's own-mip 5x5 binomial tap share at init (--init "
                         "fitted; study wide_core, decision log 2026-11-16)")
    ap.add_argument("--ee-weight", type=float, default=0.0,
                    help="weight of the core (encircled-energy) loss; 0 = the log loss alone "
                         "(study core_loss, decision log 2026-11-16)")
    ap.add_argument("--ray-storage", choices=tuple(RAY_STORAGES), default="fp32",
                    help="format the engine stores the ray intermediates in (efficiency step 1a, "
                         "decision log 2026-11-24); fp16 rounds them as the GPU does")
    args = ap.parse_args()
    assert args.t_low is None or (args.init == "fitted" and args.mips >= 2), \
        "--t-low needs --init fitted and a pyramid level (--mips >= 2)"
    tag = f"_{args.arch}_{args.loss}_{args.init}_lr{args.lr:g}"
    if args.init == "fitted" and args.t_pass != T_PASS:
        tag += f"_tpass{args.t_pass:g}"
    if args.t_low is not None:
        tag += f"_tlow{args.t_low:g}"
    if args.init == "fitted" and args.tent_share != TENT_SHARE:
        tag += f"_tshare{args.tent_share:g}"
    if args.init == "fitted" and args.wide_share != WIDE_SHARE:
        tag += f"_wshare{args.wide_share:g}"
    if args.ee_weight:
        tag += f"_ee{args.ee_weight:g}"
    if args.mips != LEGACY_MIPS:
        tag += f"_mips{args.mips}"
    if args.seed != 0:
        tag += f"_seed{args.seed}"
    if args.steps == 0:                 # the untrained init (e.g. exported for a timing-only test)
        tag += "_init"
    fit_tag = tag                       # the trained weights' name; --rescore loads these
    if args.ray_storage != "fp32":
        tag += f"_ray{args.ray_storage}"
    global RAY_STORE
    RAY_STORE = args.ray_storage

    split, bank, shape = load_bank(args.run)
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
    scored = eval_frames(args.run, split, bank, shape)          # shared with unity_parity.py score
    val_img, val_ref, val_pos = scored["val"]
    val_data = [prepare(module, val_img, val_ref, mips=args.mips)]
    data = [prepare(module, *render(bank, sel, shape), mips=args.mips) for sel in frames]
    n_levels = level_count(data[0][0])              # pyramid levels + the full-resolution one
    if args.init == "fitted":
        fit = json.loads((ROOT / "outputs" / "01_gap_study" / args.run / "fits" / "pyramid_fit.json")
                         .read_text())["best"]
        init = fitted_pyramid_init(gen, level_mlp(module), n_levels, fit, args.t_pass, args.t_low,
                                   args.tent_share, args.wide_share)
    else:
        init = torch.cat([init_params(gen, level_mlp(module)) for _ in range(n_levels)])
    source = SPIKE_SOURCE[args.arch]
    if source:                                      # drawn after the levels: their init is unchanged
        init = torch.cat([init, spike_init(gen, module, decay=SPIKE_DECAY[source])])
    cross_prefit = None
    if args.arch in CROSS_ARCHS:                    # drawn last: the levels' and rays' init unchanged
        # lazy: ceiling.py, which cross_prefit imports, imports this file
        from cross_prefit import cross_targets, prefit_cross
        offset = init.numel()
        init = torch.cat([init, cross_init(gen, module)])
        train_sel = sorted(i for sel in frames for i in sel)                # training frames only
        rows, base_h = cross_targets(bank, shape, train_sel)
        init, (kl0, kl1) = prefit_cross(module, init, offset, rows)
        cross_prefit = {"base_H": base_h, "samples": len(rows), "kl_start": kl0, "kl_end": kl1}
        print(f"cross pre-fit: {len(rows)} (source, ray) targets, base H = {base_h:.4f}, "
              f"KL {kl0:.4f} -> {kl1:.4f}", flush=True)
    bright_prefit = None
    if args.arch in BRIGHT_ARCHS:                   # after the cross filter: its init unchanged
        from cross_prefit import bright_targets, prefit_bright
        offset = init.numel()
        init = torch.cat([init, bright_init(gen, module)])
        rows, base_h = bright_targets(bank, shape, train_sel)
        init, (e0, e1) = prefit_bright(module, init, offset, rows)
        bright_prefit = {"base_H": base_h, "samples": len(rows), "err_start": e0, "err_end": e1}
        print(f"brightness pre-fit: {len(rows)} (source, ray) targets, base H = {base_h:.4f}, "
              f"light-weighted squared log gain error {e0:.4f} -> {e1:.4f}", flush=True)
    print(f"{len(frames)} training frames ({min(map(len, frames))}-{max(map(len, frames))} "
          f"sources), 1 validation frame ({len(val_sel)} sources), {n_levels} levels, "
          f"{init.numel()} parameters")

    def out(stage):
        return Path(run_dir(EXPERIMENT, args.run, stage, args.study))

    fits = out("fits")
    eval_path = out("eval") / f"train_points{tag}_eval.json"
    if args.rescore:
        params = torch.from_numpy(np.load(fits / f"train_points{fit_tag}_params.npy")).cuda()
        old = json.loads((out("eval") / f"train_points{fit_tag}_eval.json").read_text())
        assert (old["args"]["steps"] == args.steps and old["args"]["seed"] == args.seed
                and old["args"].get("mips", LEGACY_MIPS) == args.mips), (
            "pass the run's own --steps / --seed / --mips")
        curve, seconds = old["curve"], old["seconds"]
    else:
        params, curve, seconds = train(module, data, val_data, init, args, rng)

    test_img, test_ref, test_pos = scored["test"]
    results = {}
    for name, p in (("init", init.cuda()), ("trained", params)):
        records, negative = evaluate(module, p, test_img, test_ref, test_pos, size, ppd,
                                     source or "mip0", mips=args.mips)
        val_records, _ = evaluate(module, p, val_img, val_ref, val_pos, size, ppd,
                                  source or "mip0", mips=args.mips)
        results[name] = {"summary": summarize(records), "negative_fraction": negative,
                         "records": records,
                         "val": {"flip": float(np.mean([r["flip"] for r in val_records])),
                                 "log_mse": float(np.mean([r["log_mse"] for r in val_records])),
                                 "loss": frames_loss(module, val_data, p, args,
                                                     source or "mip0"),
                                 "core_loss": frames_loss(module, val_data, p, args,
                                                          source or "mip0", ee_loss),
                                 "records": val_records}}
    trained = results["trained"]["summary"]
    answers = verdict(trained, phase0)

    export = out("export")
    np.save(fits / f"train_points{tag}_params.npy", params.cpu().numpy())
    save_views(module, params, test_img, test_pos, out("eval") / f"train_points{tag}_views.npz",
               source or "mip0", mips=args.mips)
    params.cpu().numpy().astype("<f4").tofile(export / f"glare_params{tag}.bytes")
    (export / f"glare_params{tag}.json").write_text(json.dumps(       # the engines' layout file
        engine_layout(module, n_levels, source is not None, source,
                      cross=args.arch in CROSS_ARCHS, bright=args.arch in BRIGHT_ARCHS,
                      ray_storage=args.ray_storage), indent=2))
    eval_path.write_text(json.dumps({
        "environment": environment(device), "args": vars(args), "eps": EPS, "flip_floor": m.flip_floor(EPS),
        "intensity": INTENSITY, "window_px": size, "flip_ppd": ppd, "levels": n_levels,
        "param_count": level_mlp(module).param_count, "arch": args.arch, "spike_source": source,
        "spike_param_count": init.numel() - n_levels * level_mlp(module).param_count,  # with cross, bright
        "cross_param_count": ray_mlp(module).param_count if args.arch in CROSS_ARCHS else 0,
        "cross_prefit": cross_prefit,
        "bright_param_count": bright_mlp(module).param_count if args.arch in BRIGHT_ARCHS else 0,
        "bright_prefit": bright_prefit,
        "frames": frames, "val_frame": val_sel, "seconds": seconds,
        "curve": curve,
        "phase0": phase0, "phase0_fit": phase0_fit, "answers": answers, "results": results},
        indent=2))

    rows = [(f"{COMPARE[b]} ({phase0_fit[b]} fit)", phase0[b]) for b in COMPARE]
    start = {"fitted": f"fitted pyramid, energy-normalised, t_pass {args.t_pass:g}"
                       + ("" if args.t_low is None else f", t_low {args.t_low:g}")
                       + f", tent share {args.tent_share:g}, binomial share {args.wide_share:g}",
             "neutral": "t = 0.5"}[args.init]
    objective_name = f"{args.loss} log loss" + (f" + {args.ee_weight:g} x core loss"
                                                if args.ee_weight else "")
    rows += [(f"Init ({start}, before training)", results["init"]["summary"]),
             ("**Trained (this run)**", trained)]
    report = (f"# Phase 2, {args.study}: `{args.arch}` on exact point-light PSFs ({args.run})\n\n"
              f"{table(rows)}\n\n"
              f"Medians over the 35 point-grid windows ({size} px). Bins: centre H < 1/3 "
              f"(n = {trained['centre']['n']}), mid (n = {trained['mid']['n']}), edge >= 2/3 "
              f"(n = {trained['edge']['n']}). Negative output texels: "
              f"{results['trained']['negative_fraction']:.2%}. Learning rate {args.lr:g}; "
              f"{objective_name}, training ({len(data)} frames) "
              + (f"{curve[0]['train_loss']:.4f} at step 0, "
                 f"{curve[-1]['train_loss']:.4f} at step {curve[-1]['step']}.\n\n" if curve else
                 "not trained (--steps 0: the init, as exported).\n\n") +
              f"Validation frame ({len(val_sel)} sources; selects between runs, decision log "
              f"2026-11-03): mean FLIP {results['init']['val']['flip']:.4f} at init, "
              f"**{results['trained']['val']['flip']:.4f}** trained; {objective_name} "
              f"{results['init']['val']['loss']:.4f} at init, "
              f"{results['trained']['val']['loss']:.4f} trained; core loss "
              f"{results['init']['val']['core_loss']:.4f} at init, "
              f"{results['trained']['val']['core_loss']:.4f} trained.\n\n"
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
