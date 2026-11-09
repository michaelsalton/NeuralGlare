"""The chained upsample passes (slang/train_glare.slang, upsample_chain_pass) before any training.

With every level's MLP set to output URP's constant blend t = s (bias logit(s)), the Slang chain must
reproduce the NumPy URP upsample chain, and a loss on the final image must send gradients into
every level's parameters. A failure here is plumbing, not learning.

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


def urp_up0(mips, scatter):
    up = mips[-1]
    for mip in reversed(mips[:-1]):
        h, w = mip.shape[:2]
        up = mip + scatter * (sample_bilinear(up, *texel_uv(h, w)) - mip)
    return up


def exact_params(n_levels, scatter, count, outputs=12):
    """Every weight and hidden bias 0, so the hidden layer outputs leaky_relu(0) = 0 and each
    MLP outputs its output biases: softmax(0, -30, -30, logit(s)) = (1 - s, ~0, ~0, s) in every
    channel, URP's answer with the offset taps off. scatter: one s for all channels, or one per
    channel (R, G, B); the biases are channel-major, 4 per channel."""
    s = np.broadcast_to(scatter, 3)
    params = torch.zeros(n_levels * count)
    for k in range(n_levels):
        for c in range(3):
            end = (k + 1) * count - outputs + 4 * (c + 1)   # channel c: mip, mip(+o), mip(-o), low
            params[end - 3] = params[end - 2] = -30.0       # offset taps off: weight e^-30
            params[end - 1] = float(np.log(s[c] / (1 - s[c])))
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
def test_exact_weights_reproduce_urp(module, pyramid):
    mips, scatter = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    params = exact_params(len(mips) - 1, scatter, param_count(module)).cuda().requires_grad_(True)
    out = chain(module, gpu, params)
    np.testing.assert_allclose(out.detach().cpu().numpy(), urp_up0(mips, scatter), rtol=1e-4)


@needs_cuda
def test_level_mlp_outputs_four_taps_per_channel(module):
    """Per-channel weights (decision log 2026-11-09): 4 taps x R, G, B, channel-major."""
    assert int(module.level_mlp_sizes().z) == 12


@needs_cuda
def test_per_channel_blends_are_independent(module, pyramid):
    """A different blend per channel: each output channel is URP's chain with its own scatter,
    so no channel's weights leak into another's."""
    mips, _ = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    s = (0.3, 0.6, 0.8)
    params = exact_params(len(mips) - 1, s, param_count(module)).cuda().requires_grad_(True)
    out = chain(module, gpu, params).detach().cpu().numpy()
    for c in range(3):
        np.testing.assert_allclose(out[..., c], urp_up0(mips, s[c])[..., c], rtol=1e-4)


@needs_cuda
def test_gradient_reaches_every_level(module, pyramid):
    mips, scatter = pyramid
    gpu = [torch.from_numpy(m).cuda() for m in mips]
    params = exact_params(len(mips) - 1, scatter, param_count(module)).cuda().requires_grad_(True)
    chain(module, gpu, params).sum().backward()
    grad = params.grad.reshape(len(mips) - 1, param_count(module))
    # d(sum of up0) / d(bias of level k's blend) is nonzero for every level
    assert torch.all(grad[:, -1].abs() > 0), grad[:, -1]


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
        end = (k + 1) * param_count(module)
        i = [end - 12 + 4 * c + 3 for c in range(3)]     # level k's blend bias per channel
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
