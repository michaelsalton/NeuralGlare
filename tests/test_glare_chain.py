"""The chained upsample passes (slang/train_glare.slang, upsample_chain_pass) before any training.

With every level's MLP set to output URP's constant blend t = s (bias logit(s)), the Slang chain must
reproduce the NumPy URP upsample chain with its bilinear low fetch replaced by glare.slang's 3x3
tent (sample_tent, decision log 2026-11-12; the URP baseline itself stays bilinear), and a loss on
the final image must send gradients into every level's parameters. A failure here is plumbing,
not learning.

slangpy 0.43.1 decides from the first call of an entry point (per argument signature) whether
it is differentiable: if that first call's params tensor has no requires_grad, later calls with
requires_grad silently get no autograd history. So every call here passes params that require
grad, and chain() checks the history is there. Training scripts must do the same (train first,
or make the first call with requires_grad), never evaluate with a no-grad params tensor first.
"""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
import torch

from neuralglare.baselines.pyramid_bloom import (
    URPBloomParams,
    downsample_chain,
    sample_bilinear,
    shader_constants,
    texel_uv,
)

SLANG_DIR = Path(__file__).parents[1] / "slang"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch")


@pytest.fixture(scope="module")
def module():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[SLANG_DIR])
    return spy.Module.load_from_file(device, "train_glare.slang")


@pytest.fixture(scope="module")
def pyramid():
    """Mips of a 1e-1 .. 1e1 noise image and URP's scatter. Moderate contrast keeps the
    float32 (Slang) vs float64 (NumPy) bilinear coordinate error below the tolerance."""
    rng = np.random.default_rng(0)
    img = (10.0 ** rng.uniform(-1, 1, (270, 480, 3))).astype(np.float32)
    p = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)
    _, _, scatter = shader_constants(p)
    return downsample_chain(img, p), scatter


def param_count(module):
    """LevelMLP.PARAM_COUNT, from glare.slang itself (not restated here)."""
    return int(module.level_mlp_sizes().w)


@needs_cuda
def test_level_mlp_sizes_match_mlp_layout(module):
    """PARAM_COUNT is mlp.slang's layout for the sizes it reports: W1, b1, W2, b2. Python sizes
    and offsets the shared parameter buffer with these numbers (train_levels.level_mlp)."""
    v = module.level_mlp_sizes()
    i, h, o, n = int(v.x), int(v.y), int(v.z), int(v.w)
    assert n == (i * h + h) + (h * o + o)


def sample_tent(img, u, v):
    """3x3 tent of bilinear taps one texel of img apart, as slang/pyramid.slang sample_tent."""
    h, w = img.shape[:2]
    k = {-1: 1.0, 0: 2.0, 1: 1.0}
    return sum(k[i] * k[j] * sample_bilinear(img, u + i / w, v + j / h)
               for i in (-1, 0, 1) for j in (-1, 0, 1)) / 16.0


def sample_binomial5(img, u, v):
    """5x5 binomial [1 4 6 4 1]^2 / 256, the tent applied twice, from whole-texel loads (no
    linear-sampling trick, so it checks pyramid.slang sample_binomial5's 1.2-texel offsets)."""
    h, w = img.shape[:2]
    k = {-2: 1.0, -1: 4.0, 0: 6.0, 1: 4.0, 2: 1.0}
    return sum(k[i] * k[j] * sample_bilinear(img, u + i / w, v + j / h)
               for i in k for j in k) / 256.0


def tent_up0(mips, scatter):
    """URP's upsample chain with the tent in place of the bilinear low fetch (glare.slang)."""
    up = mips[-1]
    for mip in reversed(mips[:-1]):
        h, w = mip.shape[:2]
        up = mip + scatter * (sample_tent(up, *texel_uv(h, w)) - mip)
    return up


TAPS = 6        # glare.slang: mip, mip(+o), mip(-o), low, tent(mip), binomial5(mip), per channel
LOW, TENT, WIDE = 3, 4, 5


def bias(count, outputs, k, c, tap):
    """Index of level k's output bias for channel c's tap (channel-major, TAPS per channel)."""
    return (k + 1) * count - outputs + TAPS * c + tap


def exact_params(n_levels, scatter, count, outputs=TAPS * 3):
    """Every weight and hidden bias 0, so the hidden layer outputs leaky_relu(0) = 0 and each
    MLP outputs its output biases: softmax(0, -30, -30, logit(s), -30, -30) = (1 - s, ~0, ~0, s,
    ~0, ~0) in every channel, URP's answer with the offset and own-mip taps off. scatter: one s for all
    channels, or one per channel (R, G, B)."""
    s = np.broadcast_to(scatter, 3)
    params = torch.zeros(n_levels * count)
    for k in range(n_levels):
        for c in range(3):
            for tap in (1, 2, TENT, WIDE):                   # off: weight e^-30
                params[bias(count, outputs, k, c, tap)] = -30.0
            params[bias(count, outputs, k, c, LOW)] = float(np.log(s[c] / (1 - s[c])))
    return params


def chain(module, mips, params):
    out = mips[-1]
    for k in reversed(range(len(mips) - 1)):
        h, w = mips[k].shape[:2]
        out = module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                         mip=mips[k], low=out, params=params,
                                         offset=k * param_count(module))
        if torch.is_grad_enabled() and params.requires_grad:
            assert out.grad_fn is not None, "slangpy dropped autograd (see module docstring)"
    return out


@needs_cuda
def test_exact_weights_reproduce_tent_urp(module, pyramid):
    mips, scatter = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    params = exact_params(len(mips) - 1, scatter, param_count(module)).cuda().requires_grad_(True)
    out = chain(module, gpu, params)
    np.testing.assert_allclose(out.detach().cpu().numpy(), tent_up0(mips, scatter), rtol=1e-4)


@needs_cuda
def test_level_mlp_outputs_six_taps_per_channel(module):
    """Per-channel weights (decision log 2026-11-09): 6 taps x R, G, B, channel-major (the fifth,
    the own-mip tent, decision log 2026-11-13; the sixth, the own-mip 5x5 binomial, 2026-11-16)."""
    assert int(module.level_mlp_sizes().z) == TAPS * 3


@needs_cuda
@pytest.mark.parametrize("on, reference", [(TENT, sample_tent), (WIDE, sample_binomial5)],
                         ids=["tent", "binomial5"])
def test_own_mip_tap_is_its_filter_of_the_mip(module, pyramid, on, reference):
    """With only the fifth (sixth) tap on, a level outputs the 3x3 tent (5x5 binomial) of its own
    mip, whatever the low image: the +-1 (+-2) texel shoulder the full-resolution level gets from
    the input."""
    mips, _ = pyramid
    count = param_count(module)
    params = torch.full((count,), 0.0)
    for c in range(3):
        for tap in range(TAPS):
            params[bias(count, TAPS * 3, 0, c, tap)] = 0.0 if tap == on else -30.0
    params = params.cuda().requires_grad_(True)
    h, w = mips[0].shape[:2]
    out = module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                     mip=torch.from_numpy(mips[0]).cuda(),
                                     low=torch.from_numpy(mips[1]).cuda(), params=params, offset=0)
    np.testing.assert_allclose(out.detach().cpu().numpy(),
                               reference(mips[0], *texel_uv(h, w)), rtol=1e-4)


@needs_cuda
def test_per_channel_blends_are_independent(module, pyramid):
    """A different blend per channel: each output channel is the tent chain with its own scatter,
    so no channel's weights leak into another's."""
    mips, _ = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    s = (0.3, 0.6, 0.8)
    params = exact_params(len(mips) - 1, s, param_count(module)).cuda().requires_grad_(True)
    out = chain(module, gpu, params).detach().cpu().numpy()
    for c in range(3):
        np.testing.assert_allclose(out[..., c], tent_up0(mips, s[c])[..., c], rtol=1e-4)


@needs_cuda
def test_gradient_reaches_every_level(module, pyramid):
    mips, scatter = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    params = exact_params(len(mips) - 1, scatter, param_count(module)).cuda().requires_grad_(True)
    chain(module, gpu, params).sum().backward()
    grad = params.grad.reshape(len(mips) - 1, param_count(module))
    # d(sum of up0) / d(bias of level k's blend) is nonzero for every level (B channel's low tap)
    low = TAPS * 2 + LOW - TAPS * 3
    assert torch.all(grad[:, low].abs() > 0), grad[:, low]


@needs_cuda
def test_gradient_matches_finite_difference(module, pyramid):
    """The chain is linear in each level's blend t = sigmoid(bias), so a central difference is
    off by sigmoid's curvature, h^2/6 * |1 - 6s + 6s^2| relative (5e-4 at h = 0.1), plus float32
    round-off that grows as 1/h (up to 1.2e-3 at h = 0.01, deepest level: its gradient carries
    the factor s(1 - s)). h = 0.03 keeps both below 3e-4: checks the backward pass through every
    low input, not just its sign. The three channels' blend biases move together (the grey
    direction), against the sum of their gradients: one channel alone carries a third of the
    signal over the same round-off, which takes the deepest level past the tolerance."""
    mips, scatter = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    n_levels = len(mips) - 1
    params = exact_params(n_levels, scatter, param_count(module)).cuda().requires_grad_(True)
    chain(module, gpu, params).sum().backward()
    h = 3e-2
    for k in range(n_levels):
        i = [bias(param_count(module), TAPS * 3, k, c, LOW) for c in range(3)]   # level k's blend
        plus, minus = params.detach().clone(), params.detach().clone()
        plus[i] += h
        minus[i] -= h
        with torch.no_grad():
            fd = (chain(module, gpu, plus).double().sum()
                  - chain(module, gpu, minus).double().sum()) / (2 * h)
        assert params.grad[i].sum().item() == pytest.approx(fd.item(), rel=1e-3), k


@needs_cuda
def test_torch_layout_downsample_matches_numpy(module):
    """train_glare.slang's downsample wrappers (torch (H, W, 3) tensors, used by
    train_pyramid.py) give the same mips as the NumPy chain."""
    rng = np.random.default_rng(1)
    img = (10.0 ** rng.uniform(-1, 1, (270, 480, 3))).astype(np.float32)
    p = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)
    ref = downsample_chain(img, p)
    h, w = ref[0].shape[:2]
    mip = module.prefilter_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                src=torch.from_numpy(img).cuda())
    np.testing.assert_allclose(mip.cpu().numpy(), ref[0], rtol=2e-4)
    for r in ref[1:]:
        h, w = r.shape[:2]
        blurred = module.blur_h_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), src=mip)
        mip = module.blur_v_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), src=blurred)
        np.testing.assert_allclose(mip.cpu().numpy(), r, rtol=2e-4)
