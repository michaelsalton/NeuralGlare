"""The spike term (slang/glare.slang streak / spike_composite, via train_glare.slang) before any
training: rays go where and as far as the design says, keep their light, get zero outside the
frame, mix like the levels' softmax, and pass gradients to their logits and inputs.

As in test_glare_chain.py, every call passes params that require grad (slangpy 0.43.1 decides
from an entry point's first call whether it is differentiable).
"""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
import torch

SLANG_DIR = Path(__file__).parents[1] / "slang"
H, W = 160, 200                 # a mip-0-sized test image
SRC = (100, 80)                 # (x, y) of the impulse, > 63 texels from every border

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch")


@pytest.fixture(scope="module")
def module():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[SLANG_DIR])
    return spy.Module.load_from_file(device, "train_glare.slang")


def sizes(module):
    v = module.spike_sizes()
    return int(v.x), int(v.y), int(v.z), int(v.w)       # rays, passes, taps, profile logits


def impulse(x=SRC[0], y=SRC[1]):
    img = torch.zeros(H, W, 3, device="cuda")
    img[y, x] = 1.0
    return img


def ray(module, img, params, k):
    _, passes, _, _ = sizes(module)
    for stage in range(passes):
        img = module.streak_pass(texel=spy.grid((H, W)), src=img, params=params, profile=0,
                                 ray=k, stage=stage)
    return img


def logits(module, value=0.0):
    return torch.full((sizes(module)[3],), value, device="cuda", requires_grad=True)


@needs_cuda
def test_sizes(module):
    rays, passes, taps, count = sizes(module)
    assert (rays, count) == (14, 2 * passes * taps)
    assert taps ** passes == 64, "the passes' base-taps digits must cover 0..63 texels"
    v = module.spike_mlp_sizes()
    i, h, o, n = int(v.x), int(v.y), int(v.z), int(v.w)
    assert o == 9 and n == (i * h + h) + (h * o + o)     # 3 mix weights x R, G, B


@needs_cuda
def test_uniform_profile_is_a_continuous_ray(module):
    """Logits 0: every pass uniform, so ray 0 (along +x) is 64 texels of exactly 1/64: every
    distance 0..63 reached once, no gaps, nothing off the ray."""
    out = ray(module, impulse(), logits(module), 0).detach().cpu().numpy()[..., 0]
    x, y = SRC
    np.testing.assert_allclose(out[y, x:x + 64], 1 / 64, rtol=1e-6)
    assert out.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.abs(out[y, x:x + 64]).sum() == pytest.approx(out.sum(), rel=1e-6)


@needs_cuda
@pytest.mark.parametrize("k", range(14))
def test_every_ray_keeps_its_light_and_points_along_its_angle(module, k):
    params = torch.randn(sizes(module)[3], generator=torch.Generator().manual_seed(k)).cuda()
    out = ray(module, impulse(), params.requires_grad_(True), k).detach().cpu().numpy()[..., 0]
    assert out.sum() == pytest.approx(1.0, rel=1e-5)
    assert out.min() >= 0
    yy, xx = np.mgrid[:H, :W]
    d = np.array([xx - SRC[0], yy - SRC[1]], np.float64)
    e = np.array([np.cos(2 * np.pi * k / 14), np.sin(2 * np.pi * k / 14)])
    perp = (d * np.array([-e[1], e[0]])[:, None, None]).sum(0)
    along = (d * e[:, None, None]).sum(0)
    assert np.abs((out * perp).sum()) < 1e-3, "light off the ray's line"
    assert (out * along).sum() > 0 and out[along < -1.5].sum() < 1e-6, "ray points backwards"


@needs_cuda
def test_families_use_their_own_logits(module):
    """Ray 0 (even) reads logits 0..11, ray 1 (odd) 12..23: changing one family's logits leaves
    the other family's rays unchanged."""
    a, b = logits(module), logits(module)
    with torch.no_grad():
        b[12:] = 3.0 * torch.arange(12, device="cuda") / 12
    for k, same in ((0, True), (1, False)):
        ra = ray(module, impulse(), a, k).detach()
        rb = ray(module, impulse(), b, k).detach()
        assert torch.equal(ra, rb) == same, k


@needs_cuda
def test_zero_outside_the_frame(module):
    """A source 10 texels from the right border: ray 0 runs off the frame, and the light past
    the border is lost, not piled onto the border texel as clamp-to-edge would."""
    x, y = W - 11, SRC[1]
    out = ray(module, impulse(x, y), logits(module), 0).detach().cpu().numpy()[..., 0]
    np.testing.assert_allclose(out[y, x:], 1 / 64, rtol=1e-6)
    assert out.sum() == pytest.approx(11 / 64, rel=1e-5)


@needs_cuda
def test_accumulate_adds_to_the_running_sum(module):
    _, passes, _, _ = sizes(module)
    params = logits(module)
    s = impulse()
    for stage in range(passes - 1):
        s = module.streak_pass(texel=spy.grid((H, W)), src=s, params=params, profile=0, ray=3,
                               stage=stage)
    acc = torch.rand(H, W, 3, device="cuda")
    got = module.streak_accumulate_pass(texel=spy.grid((H, W)), src=s, acc=acc, params=params,
                                        profile=0, ray=3, stage=passes - 1)
    want = acc + ray(module, impulse(), params, 3)
    torch.testing.assert_close(got.detach(), want.detach(), rtol=1e-6, atol=1e-7)


def spike_params(module, bias, profile=0.0):
    """SpikeMLP with every weight 0 and output biases `bias` (3 for every channel, or 9,
    channel-major), then the profile logits."""
    v = module.spike_mlp_sizes()
    p = torch.zeros(int(v.w) + sizes(module)[3])
    p[int(v.w) - 9:int(v.w)] = torch.tensor(bias * 3 if len(bias) == 3 else bias)
    p[int(v.w):] = profile
    return p.cuda().requires_grad_(True)


@needs_cuda
def test_composite_mixes_with_softmax_weights(module):
    """Biases (b0, b1, b2): out = w0 * base + (w1 * even + w2 * odd) / 7 with w = softmax(b),
    even / odd bilinearly upsampled from half resolution."""
    rng = np.random.default_rng(0)
    base = torch.from_numpy(rng.uniform(0, 1, (2 * H, 2 * W, 3)).astype(np.float32)).cuda()
    even = torch.from_numpy(rng.uniform(0, 1, (H, W, 3)).astype(np.float32)).cuda()
    odd = torch.from_numpy(rng.uniform(0, 1, (H, W, 3)).astype(np.float32)).cuda()
    b = np.array([0.5, -1.0, 0.25])
    w = np.exp(b) / np.exp(b).sum()
    params = spike_params(module, b.tolist())
    got = module.spike_pass(texel=spy.grid((2 * H, 2 * W)), out_size=spy.int2(2 * W, 2 * H),
                            base=base, even=even, odd=odd, params=params, offset=0)
    up = torch.nn.functional.interpolate               # bilinear, half-pixel centres, clamped
    e = up(even.permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=False)
    o = up(odd.permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=False)
    want = w[0] * base + (w[1] * e[0] + w[2] * o[0]).permute(1, 2, 0) / 7
    torch.testing.assert_close(got.detach(), want.float(), rtol=1e-5, atol=1e-6)


@needs_cuda
def test_composite_mixes_each_channel_with_its_own_weights(module):
    """Channel-major biases, a different mix per channel (decision log 2026-11-09): channel c of
    the output is w_c[0] * base + (w_c[1] * even + w_c[2] * odd) / 7, w_c = softmax(b_c)."""
    base = torch.rand(2 * H, 2 * W, 3, device="cuda")
    even = torch.rand(H, W, 3, device="cuda")
    odd = torch.rand(H, W, 3, device="cuda")
    b = np.array([[0.5, -1.0, 0.25], [2.0, 0.0, -3.0], [-1.0, 1.5, 0.5]])     # R, G, B
    w = np.exp(b) / np.exp(b).sum(1, keepdims=True)
    got = module.spike_pass(texel=spy.grid((2 * H, 2 * W)), out_size=spy.int2(2 * W, 2 * H),
                            base=base, even=even, odd=odd,
                            params=spike_params(module, b.ravel().tolist()), offset=0)
    up = torch.nn.functional.interpolate
    e = up(even.permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=False)
    o = up(odd.permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=False)
    e, o = e[0].permute(1, 2, 0), o[0].permute(1, 2, 0)
    for c in range(3):         # atol 1e-5: float32 bilinear coordinates, 1.6e-6 seen on 1 texel
        want = w[c, 0] * base[..., c] + (w[c, 1] * e[..., c] + w[c, 2] * o[..., c]) / 7
        torch.testing.assert_close(got.detach()[..., c], want.float(), rtol=1e-5, atol=1e-5)


@needs_cuda
def test_spikes_off_is_the_base(module):
    """Biases (0, -30, -30): the composite returns the full-resolution output (to e^-30)."""
    base = torch.rand(2 * H, 2 * W, 3, device="cuda")
    ones = torch.ones(H, W, 3, device="cuda")
    got = module.spike_pass(texel=spy.grid((2 * H, 2 * W)), out_size=spy.int2(2 * W, 2 * H),
                            base=base, even=ones, odd=ones,
                            params=spike_params(module, [0.0, -30.0, -30.0]), offset=0)
    torch.testing.assert_close(got.detach(), base, rtol=1e-6, atol=1e-7)


@needs_cuda
def test_profile_gradient_matches_finite_difference(module):
    """A loss that weighs distance along the ray (total light can't depend on the logits), so
    each pass's logits have a gradient; checked against a central difference, through all three
    chained passes."""
    weight = torch.arange(W, device="cuda", dtype=torch.float32)[None, :, None].expand(H, W, 3)

    def loss(p):
        return (ray(module, impulse(), p, 0) * weight).sum()

    params = torch.zeros(sizes(module)[3], device="cuda")
    params[:12] = torch.linspace(-0.5, 0.5, 12, device="cuda")
    params.requires_grad_(True)
    loss(params).backward()
    h = 1e-2
    for i in (0, 5, 11):                    # one tap of each pass of the even family
        plus, minus = params.detach().clone(), params.detach().clone()
        plus[i] += h
        minus[i] -= h
        with torch.no_grad():
            fd = (loss(plus).double() - loss(minus).double()) / (2 * h)
        assert params.grad[i].item() == pytest.approx(fd.item(), rel=2e-3), i
    assert params.grad[12:].abs().max() == 0, "ray 0 must not touch the odd family's logits"
