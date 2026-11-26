"""The spike term (slang/glare.slang streak / spike_composite, via train_glare.slang) before any
training: rays go where and as far as the design says, keep their light, get zero outside the
frame, mix like the levels' softmax, and pass gradients to their logits and inputs. The per-ray
cross filter (ray_cross_accumulate, decision log 2026-11-19) moves light across its ray by the
tap the design says, keeps light, reads the field in the ray's frame and passes gradients. Per-ray
brightness (ray_bright_accumulate / spike_composite_bright, 2026-11-19) starts as the cross term,
is 1 on axis, sums to 7 over a family, matches the gains it is defined by, and passes gradients.

As in test_glare_chain.py, every call passes params that require grad (slangpy 0.43.1 decides
from an entry point's first call whether it is differentiable).
"""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
import torch

from neuralglare.storage import half_toward_zero

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
                                 ray=k, stage=stage, scale=1.0)
    return img


def logits(module, value=0.0):
    return torch.full((sizes(module)[3],), value, device="cuda", requires_grad=True)


@needs_cuda
def test_sizes(module):
    rays, passes, taps, count = sizes(module)
    assert (passes, taps) == (3, 4)
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
    """Ray 0 (even) reads the first half of the logits (0..11 with 4 taps), ray 1 (odd) the
    second: changing one family's logits leaves the other family's rays unchanged."""
    half = sizes(module)[3] // 2
    a, b = logits(module), logits(module)
    with torch.no_grad():
        b[half:] = 3.0 * torch.arange(half, device="cuda") / half
    for k, same in ((0, True), (1, False)):
        ra = ray(module, impulse(), a, k).detach()
        rb = ray(module, impulse(), b, k).detach()
        assert torch.equal(ra, rb) == same, k


@needs_cuda
def test_exponential_init_is_exact(module):
    """train_points.spike_init's logits z[p][n] = -n taps^p / decay give ray 0 the profile
    exp(-d / decay) over d = 0..63, normalised: the product of one softmax weight per base-taps
    digit of d."""
    _, passes, taps, count = sizes(module)
    decay = 16.0
    z = [-n * taps ** p / decay for p in range(passes) for n in range(taps)]
    params = torch.tensor(z * 2, device="cuda").requires_grad_(True)
    assert params.numel() == count
    out = ray(module, impulse(), params, 0).detach().cpu().numpy()[..., 0]
    want = np.exp(-np.arange(64) / decay)
    x, y = SRC
    np.testing.assert_allclose(out[y, x:x + 64], want / want.sum(), rtol=1e-5)


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
                               stage=stage, scale=1.0)
    acc = torch.rand(H, W, 3, device="cuda")
    got = module.streak_accumulate_pass(texel=spy.grid((H, W)), src=s, acc=acc, params=params,
                                        profile=0, ray=3, stage=passes - 1, scale=1.0)
    want = acc + ray(module, impulse(), params, 3)
    torch.testing.assert_close(got.detach(), want.detach(), rtol=1e-6, atol=1e-7)


def scaled_ray(module, img, params, k, texel_step):
    _, passes, _, _ = sizes(module)
    h, w = img.shape[:2]
    for stage in range(passes):
        img = module.streak_at_scale_pass(texel=spy.grid((h, w)), src=img, params=params,
                                          profile=0, ray=k, stage=stage, scale=1.0,
                                          texel_step=texel_step)
    return img


@needs_cuda
@pytest.mark.parametrize("k", (0, 3))
def test_streak_at_scale_one_is_streak(module, k):
    """texel_step = 1 is streak exactly (efficiency step 2c): streak calls it with 1."""
    params = torch.randn(sizes(module)[3], generator=torch.Generator().manual_seed(k)).cuda()
    params.requires_grad_(True)
    got = scaled_ray(module, impulse(), params, k, 1.0).detach()
    torch.testing.assert_close(got, ray(module, impulse(), params, k).detach(), rtol=0, atol=0)


@needs_cuda
def test_half_resolution_ray_keeps_its_length_in_pixels(module):
    """texel_step = 0.5 on an image of 2-pixel texels: uniform logits put ray 0 (along +x) over
    0..31.5 texels, i.e. 0..63 output pixels, keeping its light; its centroid is half the
    full-resolution one (31.5 / 2 texels)."""
    out = scaled_ray(module, impulse(), logits(module), 0, 0.5).detach().cpu().numpy()[..., 0]
    x, y = SRC
    assert out.sum() == pytest.approx(1.0, rel=1e-5)
    assert np.abs(out[y]).sum() == pytest.approx(out.sum(), rel=1e-6)    # stays on its row
    centroid = (out[y] * (np.arange(W) - x)).sum() / out[y].sum()
    assert centroid == pytest.approx(31.5 / 2, rel=1e-5)
    assert out[y, x + 33:].sum() == 0 and out[y, :x].sum() == 0          # nothing past 31.5


def pre_exposed_ray(module, img, params, k, half):
    """Ray 3 as an engine runs it with pre-exposure k (glare.slang streak; decision log
    2026-11-24): k into the first pass, 1/k out of the last, each stored output but the last
    rounded to fp16 if `half` (as rayA / rayB; the last feeds a float32 family sum)."""
    _, passes, _, _ = sizes(module)
    for stage in range(passes):
        scale = (k if stage == 0 else 1.0) * (1.0 / k if stage == passes - 1 else 1.0)
        img = module.streak_pass(texel=spy.grid((H, W)), src=img, params=params, profile=0,
                                 ray=3, stage=stage, scale=scale)
        if half and stage < passes - 1:
            img = half_toward_zero(img)
    return img.detach()


@needs_cuda
def test_pre_exposure_is_exact_and_keeps_a_sun_in_fp16(module):
    """A power-of-two k changes nothing in float32 (bit for bit). A sun-bright impulse (3.8e5, as
    goegap_road_4k's) overflows fp16 storage without pre-exposure (saturating at 65504, rounding
    toward zero), and fits with k = 1/16, within fp16's rounding of the float32 ray."""
    params = logits(module)
    sun = 3.8e5 * impulse()
    plain = pre_exposed_ray(module, sun, params, 1.0, half=False)
    assert torch.equal(pre_exposed_ray(module, sun, params, 1 / 16, half=False), plain)
    clipped = pre_exposed_ray(module, sun, params, 1.0, half=True)
    assert not torch.allclose(clipped, plain, rtol=0.1)        # saturated, silently
    got = pre_exposed_ray(module, sun, params, 1 / 16, half=True)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, plain, rtol=2e-3, atol=1e-3)


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


# ------------------------------------------------------------- per-ray cross filter (2026-11-19)

def cross_sizes(module):
    v = module.ray_mlp_sizes()
    return int(v.x), int(v.y), int(v.z), int(v.w)       # inputs, hidden, taps, parameters


def cross_params(module, bias, profile=0.0):
    """Profile logits (all `profile`), then RayMLP with every weight 0 and output biases `bias`:
    the same 3 tap weights softmax(bias) at every texel."""
    count, n = sizes(module)[3], cross_sizes(module)[3]
    p = torch.zeros(count + n)
    p[:count] = profile
    p[-3:] = torch.tensor(bias)
    return p.cuda().requires_grad_(True)


def crossed(module, img, params, k, acc=None):
    """Ray k through all its passes, then the cross filter (RayMLP after the profile logits)."""
    _, passes, _, count = sizes(module)
    for stage in range(passes):
        img = module.streak_pass(texel=spy.grid((H, W)), src=img, params=params, profile=0,
                                 ray=k, stage=stage, scale=1.0)
    # requires_grad: slangpy 0.43.1 on Vulkan needs a gradient buffer on every DiffTensor in a backward
    acc = torch.zeros(H, W, 3, device="cuda", requires_grad=True) if acc is None else acc
    return module.ray_cross_accumulate_pass(texel=spy.grid((H, W)), out_size=spy.int2(W, H),
                                            src=img, acc=acc, params=params, offset=count, ray=k, scale=1.0)


@needs_cuda
def test_cross_sizes(module):
    i, h, o, n = cross_sizes(module)
    assert (i, o) == (3, 3) and n == (i * h + h) + (h * o + o)


@needs_cuda
@pytest.mark.parametrize("k", (0, 3, 8))
def test_cross_centre_tap_is_the_plain_ray(module, k):
    """Biases (-30, 0, -30): the centre tap only, so the term is the plain ray plus the running sum."""
    p = cross_params(module, [-30.0, 0.0, -30.0])
    acc = torch.rand(H, W, 3, device="cuda")
    got = crossed(module, impulse(), p, k, acc)
    want = acc + ray(module, impulse(), p, k)
    torch.testing.assert_close(got.detach(), want.detach(), rtol=1e-6, atol=1e-7)


@needs_cuda
@pytest.mark.parametrize("tap, dy", ((0, -1), (2, 1)))
def test_cross_tap_moves_the_ray_across(module, tap, dy):
    """Ray 0 runs along +x, so n_0 = (0, 1): tap 0 moves the whole ray one texel to -y, tap 2 to +y."""
    b = [-30.0, -30.0, -30.0]
    b[tap] = 0.0
    out = crossed(module, impulse(), cross_params(module, b), 0).detach().cpu().numpy()[..., 0]
    x, y = SRC
    np.testing.assert_allclose(out[y + dy, x:x + 64], 1 / 64, rtol=1e-5)
    assert out.sum() == pytest.approx(1.0, rel=1e-5)


@needs_cuda
@pytest.mark.parametrize("k", range(14))
def test_cross_filter_keeps_light_when_uniform(module, k):
    """Random biases, weights 0: one positive filter summing to 1 at every texel, a convolution, so
    a point's light is kept exactly. (With field-dependent weights only approximately: they are
    taken at the receiving texel.)"""
    b = torch.randn(3, generator=torch.Generator().manual_seed(k)).tolist()
    out = crossed(module, impulse(), cross_params(module, b), k).detach().cpu().numpy()[..., 0]
    assert out.sum() == pytest.approx(1.0, rel=1e-5)
    assert out.min() >= 0


@needs_cuda
def test_cross_mix_sees_the_field_in_the_ray_frame(module):
    """The input is (e_k . v, n_k . v, family): v rotated by two rays' angle, read by ray k + 2 (same
    family), gives the same weights; at the frame centre (v = 0) there is one filter per family."""
    count, n = sizes(module)[3], cross_sizes(module)[3]
    p = torch.randn(count + n, generator=torch.Generator().manual_seed(0)).cuda().requires_grad_(True)
    v = torch.rand(64, 2, generator=torch.Generator().manual_seed(1)).cuda() * 2 - 1
    a = 2 * 2 * np.pi / 14
    rot = torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=torch.float32,
                       device="cuda")
    for k in (0, 1, 5, 13):
        w0 = module.cross_mix(v=v, params=p, offset=count, ray=k)
        w1 = module.cross_mix(v=v @ rot.T, params=p, offset=count, ray=(k + 2) % 14)
        torch.testing.assert_close(w0.detach(), w1.detach(), rtol=1e-5, atol=1e-5)
    zero = torch.zeros(1, 2, device="cuda")
    at_centre = [module.cross_mix(v=zero, params=p, offset=count, ray=k).detach() for k in range(14)]
    for k in range(2, 14):
        torch.testing.assert_close(at_centre[k], at_centre[k % 2])
    assert not torch.allclose(at_centre[0], at_centre[1]), "the family input must change the weights"


@needs_cuda
def test_cross_gradient_matches_finite_difference(module):
    """A loss that weighs distance across ray 0 (y), so the side taps' biases have a gradient;
    checked against a central difference through the three passes and the cross filter."""
    weight = torch.arange(H, device="cuda", dtype=torch.float32)[:, None, None].expand(H, W, 3)
    count, n = sizes(module)[3], cross_sizes(module)[3]

    def loss(p):
        return (crossed(module, impulse(), p, 0) * weight).sum()

    params = cross_params(module, [0.3, 0.0, -0.2])
    loss(params).backward()
    h = 1e-2
    for i in (count + n - 3, count + n - 1):          # output biases of taps 0 and 2
        plus, minus = params.detach().clone(), params.detach().clone()
        plus[i] += h
        minus[i] -= h
        with torch.no_grad():
            fd = (loss(plus).double() - loss(minus).double()) / (2 * h)
        assert params.grad[i].item() == pytest.approx(fd.item(), rel=2e-3), i


# ------------------------------------------------------------- per-ray brightness (2026-11-19)

def bright_sizes(module):
    v = module.bright_mlp_sizes()
    return int(v.x), int(v.y), int(v.z), int(v.w)       # inputs, hidden, outputs, parameters


def full_spike_params(module, seed=None, bright_scale=1.0):
    """[SpikeMLP][profile logits][RayMLP][BrightMLP] with the composite at offset 0, and the
    (profile, RayMLP, BrightMLP) starts. seed None: all 0 but the mix biases; else random (the
    BrightMLP part times bright_scale)."""
    s, count = int(module.spike_mlp_sizes().w), sizes(module)[3]
    r, b = cross_sizes(module)[3], bright_sizes(module)[3]
    if seed is None:
        p = torch.zeros(s + count + r + b)
    else:
        p = torch.randn(s + count + r + b, generator=torch.Generator().manual_seed(seed)) * 0.5
        p[s + count + r:] *= bright_scale
    p[s - 9:s] = torch.tensor([0.5, -0.5, -1.0] * 3)                 # mix biases, every channel
    return p.cuda().requires_grad_(True), (s, s + count, s + count + r)


def family_rays(module, img, params, offsets, bright):
    """Every ray of img through its passes and the cross pass (with e^{z_k} if bright), summed
    per family: the (even, odd) the composite reads."""
    profile, ray_off, _ = offsets
    passes = sizes(module)[1]
    acc = [torch.zeros(H, W, 3, device="cuda", requires_grad=True) for _ in range(2)]
    for k in range(14):
        s = img
        for stage in range(passes):
            s = module.streak_pass(texel=spy.grid((H, W)), src=s, params=params, profile=profile,
                                   ray=k, stage=stage, scale=1.0)
        accumulate = (module.ray_bright_accumulate_pass if bright
                      else module.ray_cross_accumulate_pass)
        acc[k % 2] = accumulate(texel=spy.grid((H, W)), out_size=spy.int2(W, H), src=s,
                                acc=acc[k % 2], params=params, offset=ray_off, ray=k, scale=1.0)
    return acc


def bright_term(module, img, params, offsets, bright=True):
    """The spike term of img onto a zero base (spike_bright_pass, or spike_pass if not bright)."""
    even, odd = family_rays(module, img, params, offsets, bright)
    composite = module.spike_bright_pass if bright else module.spike_pass
    base = torch.zeros(H, W, 3, device="cuda", requires_grad=True)
    return composite(texel=spy.grid((H, W)), out_size=spy.int2(W, H), base=base, even=even,
                     odd=odd, params=params, offset=0)


def texel_grid():
    """(uv, field_offset) at every texel of an (H, W) frame, each (H * W, 2), row-major."""
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                            indexing="ij")
    p = torch.stack([xx + 0.5, yy + 0.5], -1).float().reshape(-1, 2)
    size = torch.tensor([W, H], dtype=torch.float32, device="cuda")
    return p / size, (p - size / 2) / float(np.hypot(W / 2, H / 2))


@needs_cuda
def test_bright_sizes(module):
    i, h, o, n = bright_sizes(module)
    assert (i, o) == (3, 1) and n == (i * h + h) + (h * o + o)


@needs_cuda
def test_bright_zero_is_the_cross_term(module):
    """BrightMLP all 0: every e^{z_k} = 1 and Z = 7, so the term is the cross model's."""
    params, offsets = full_spike_params(module)
    with torch.no_grad():
        params[offsets[1]:offsets[2]] = torch.randn(
            offsets[2] - offsets[1], generator=torch.Generator().manual_seed(3)).cuda()
    got = bright_term(module, impulse(), params, offsets, bright=True)
    want = bright_term(module, impulse(), params, offsets, bright=False)
    torch.testing.assert_close(got.detach(), want.detach(), rtol=1e-6, atol=1e-9)


@needs_cuda
def test_bright_gain_is_one_on_axis_and_sums_to_seven(module):
    """At v = 0 a family's rays all see the same input, so every gain is 1 whatever the weights;
    anywhere, a family's 7 gains are positive and sum to 7."""
    params, (_, _, off) = full_spike_params(module, seed=0, bright_scale=4.0)
    zero = torch.zeros(1, 2, device="cuda")
    for k in range(14):
        g = module.ray_gain(v=zero, params=params, offset=off, ray=k).detach()
        torch.testing.assert_close(g, torch.ones_like(g), rtol=1e-6, atol=1e-6)
    v = torch.rand(64, 2, generator=torch.Generator().manual_seed(1)).cuda() * 2 - 1
    gains = torch.stack([module.ray_gain(v=v, params=params, offset=off, ray=k).detach()
                         for k in range(14)])
    assert gains.min() > 0
    assert not torch.allclose(gains, torch.ones_like(gains), atol=1e-3), "the gains must vary"
    for fam in (0, 1):
        torch.testing.assert_close(gains[fam::2].sum(0), torch.full((64,), 7.0, device="cuda"),
                                   rtol=1e-5, atol=1e-5)


@needs_cuda
def test_bright_brightens_the_ray_along_the_field(module):
    """BrightMLP z = 2 relu(e_k . v): at v along e_0, ray 0 (e . v largest) gets the largest gain
    of its family, above 1, and the rays pointing away from v less than 1."""
    params, (_, _, off) = full_spike_params(module)
    i, h, _, _ = bright_sizes(module)
    with torch.no_grad():
        params[off + 0] = 1.0                    # hidden unit 0 reads e . v (W1[0][0])
        params[off + i * h + h] = 2.0            # the output reads hidden unit 0 (W2[0][0])
    v = torch.tensor([[0.8, 0.0]], device="cuda")
    g = torch.stack([module.ray_gain(v=v, params=params, offset=off, ray=k).detach()[0]
                     for k in range(0, 14, 2)])  # the even family: rays 0, 2, .., 12
    assert g.argmax() == 0 and g[0] > 1
    assert (g[2:5] < 1).all(), "rays 4, 6, 8 point away from v"


@needs_cuda
def test_bright_term_matches_its_gains(module):
    """The pulled-out normaliser: the term equals w1 * sum_even g_k S'_k / 7 + w2 * sum_odd ...,
    g_k = ray_gain at each texel and S'_k the plain cross-filtered rays, for random weights."""
    params, offsets = full_spike_params(module, seed=2, bright_scale=4.0)
    img = torch.zeros(H, W, 3, device="cuda")
    img[30, 40] = 1.0                            # off-centre sources, so the gains vary
    img[120, 170] = 2.0
    got = bright_term(module, img, params, offsets, bright=True).detach()
    profile, ray_off, off = offsets
    uv, v = texel_grid()
    passes = sizes(module)[1]
    fams = [torch.zeros(H, W, 3, device="cuda"), torch.zeros(H, W, 3, device="cuda")]
    for k in range(14):
        s = img
        for stage in range(passes):
            s = module.streak_pass(texel=spy.grid((H, W)), src=s, params=params, profile=profile,
                                   ray=k, stage=stage, scale=1.0)
        zero = torch.zeros(H, W, 3, device="cuda", requires_grad=True)
        sk = module.ray_cross_accumulate_pass(texel=spy.grid((H, W)), out_size=spy.int2(W, H),
                                              src=s, acc=zero, params=params, offset=ray_off,
                                              ray=k, scale=1.0).detach()
        g = module.ray_gain(v=v, params=params, offset=off, ray=k).detach().reshape(H, W, 1)
        fams[k % 2] += g * sk / 7
    w = torch.stack([module.spike_mix(uv=uv, params=params, offset=0, channel=c).detach()
                     for c in range(3)], -1).reshape(H, W, 3, 3)          # (y, x, weight, channel)
    want = w[:, :, 1] * fams[0] + w[:, :, 2] * fams[1]
    assert got.sum().item() > 0
    # atol: float32 rounding of the two orders of summation, 5e-8 seen against a 7e-3 peak
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5 * want.abs().max().item())


@needs_cuda
def test_bright_gradient_matches_finite_difference(module):
    """A loss that weighs position, so moving light between rays changes it; checked against a
    central difference on BrightMLP weights through both passes. Its output bias is common to
    every ray, so softmax cancels it: gradient 0."""
    params, offsets = full_spike_params(module, seed=4, bright_scale=2.0)
    off = offsets[2]
    i, h, _, _ = bright_sizes(module)
    img = torch.zeros(H, W, 3, device="cuda")
    img[30, 40] = 1.0
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                            indexing="ij")
    weight = (xx + 2 * yy).float()[..., None].expand(H, W, 3)

    def loss(p):
        return (bright_term(module, img, p, offsets) * weight).sum()

    loss(params).backward()
    step = 1e-2
    for j in (off + 1, off + i * h + h + 2):          # a hidden weight and an output weight
        plus, minus = params.detach().clone(), params.detach().clone()
        plus[j] += step
        minus[j] -= step
        with torch.no_grad():
            fd = (loss(plus).double() - loss(minus).double()) / (2 * step)
        assert params.grad[j].item() == pytest.approx(fd.item(), rel=5e-3, abs=1e-6), j
    # The bias's gradient is the difference of two equal terms (the rays' e^{z_k} and Z), so in
    # float32 it is noise: 5e-5 of the largest BrightMLP gradient seen
    bright_grad = params.grad[off:off + bright_sizes(module)[3]]
    assert abs(bright_grad[-1].item()) < 1e-3 * bright_grad.abs().max().item(), "the bias cancels"


# -------------------------------------------- the ray stack (efficiency step 1c, 2026-11-24)

def stack_term(module, img, params, offsets, scale=1.0):
    """The spike term of img onto a zero base from the ray stack (spike_stack_pass): every ray's
    last streak pass stacked, the cross filter, brightness and family sums in the composite."""
    profile = offsets[0]
    passes = sizes(module)[1]
    last = []
    for k in range(14):
        s = img
        for stage in range(passes):
            s = module.streak_pass(texel=spy.grid((H, W)), src=s, params=params, profile=profile,
                                   ray=k, stage=stage, scale=1.0)
        last.append(s)
    base = torch.zeros(H, W, 3, device="cuda", requires_grad=True)
    return module.spike_stack_pass(texel=spy.grid((H, W)), out_size=spy.int2(W, H), base=base,
                                   rays=torch.stack(last), params=params, offset=0, scale=scale)


@needs_cuda
def test_stack_matches_the_per_ray_passes(module):
    """Same arithmetic, same order: the stack composite equals ray_bright_accumulate summed per
    family + spike_composite_bright, for random weights and off-centre sources."""
    params, offsets = full_spike_params(module, seed=2, bright_scale=4.0)
    img = torch.zeros(H, W, 3, device="cuda")
    img[30, 40] = 1.0
    img[120, 170] = 2.0
    img[5, 195] = 3.0                            # rays leave the frame: the zero border
    got = stack_term(module, img, params, offsets).detach()
    want = bright_term(module, img, params, offsets, bright=True).detach()
    assert got.sum().item() > 0
    # atol: the per-ray composite reads each family with sample_bilinear at texel centres, where
    # uv * size - 0.5 misses the integer by up to an ulp (1.5e-5 texel at x = 196), blending in a
    # sliver of the neighbours: 7e-6 of the peak seen, next to a source. The stack reads its rays
    # at integer offsets, so it is the exact one.
    torch.testing.assert_close(got, want, rtol=1e-6, atol=2e-5 * want.abs().max().item())


@needs_cuda
def test_stack_scale_undoes_the_pre_exposure(module):
    """A stack of k-scaled rays with scale 1/k (k a power of two) is the unscaled term exactly."""
    params, offsets = full_spike_params(module, seed=2, bright_scale=4.0)
    img = torch.zeros(H, W, 3, device="cuda")
    img[30, 40] = 1.0
    want = stack_term(module, img, params, offsets).detach()
    got = stack_term(module, img / 16, params, offsets, scale=16.0).detach()
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@needs_cuda
def test_stack_gradient_matches_the_per_ray_passes(module):
    """The stack's backward (through RaySlice, the loop and the stacked tensor) gives the per-ray
    chain's parameter gradient."""
    params, offsets = full_spike_params(module, seed=4, bright_scale=2.0)
    img = torch.zeros(H, W, 3, device="cuda")
    img[30, 40] = 1.0
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                            indexing="ij")
    weight = (xx + 2 * yy).float()[..., None].expand(H, W, 3)
    grads = []
    for term in (stack_term, bright_term):
        p = params.detach().clone().requires_grad_(True)
        (term(module, img, p, offsets) * weight).sum().backward()
        grads.append(p.grad.clone())
    got, want = grads
    off = offsets[2]                                  # BrightMLP's start; it runs to the end
    torch.testing.assert_close(got[:off], want[:off], rtol=1e-4,
                               atol=1e-6 * want.abs().max().item())
    # BrightMLP: the e^{z_k} in each ray's term and in Z are one product here, two kernels in the
    # per-ray chain, whose float32 cancellation is noise there (3e-4 of the block's largest
    # gradient seen). The output bias cancels exactly (test_bright_gradient...): the stack's is
    # the smaller noise.
    scale = want[off:].abs().max().item()
    torch.testing.assert_close(got[off:], want[off:], rtol=0, atol=1e-3 * scale)
    assert abs(got[-1].item()) <= abs(want[-1].item()) + 1e-6 * scale, "the bias cancels"
