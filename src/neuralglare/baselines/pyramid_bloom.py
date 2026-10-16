"""
Pyramid Bloom implementation modelled after the bloom from Unity 6.6 URP 17.6

"""

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import minimize

from neuralglare.eval.metrics import log_loss
from neuralglare.io import luminance, read_hdr

URP_VERSION = "URP 17.6.0 (Unity 6.6)"
DOWNSCALE_SHIFT = {"half": 1, "quarter": 2}

def texel_uv(h, w):
    """UV of every texel centre of an (h, w) target, as a fullscreen pass sees it."""
    v, u = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    return u, v


def sample_bilinear(img, u, v):
    """Bilinear fetch with clamp-to-edge addressing, like a GPU linear-clamp sampler.

    img: (H, W, C). u, v: same-shaped arrays in UV space (texel i's centre at (i + 0.5) / size).
    Returns an array of shape (*u.shape, C).
    """
    H, W = img.shape[:2]
    x = u * W - 0.5                        # continuous texel coords: integers are texel centres
    y = v * H - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None]               # blend weight toward the right/lower neighbour
    fy = (y - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)         # clamp-to-edge: clamp the indices, keep the weights
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)

# BlurH: 9 taps spaced 2 source texels apart (= 1 destination texel); it also does the 2x downsample.
BLUR_H_OFFSETS = np.arange(-4, 5) * 2.0
BLUR_H_WEIGHTS = np.array([0.01621622, 0.05405405, 0.12162162, 0.19459459, 0.22702703,
                           0.19459459, 0.12162162, 0.05405405, 0.01621622])
# BlurV: the same 9-tap Gaussian at equal resolution, in 5 bilinear taps.
BLUR_V_OFFSETS = np.array([-3.23076923, -1.38461538, 0.0, 1.38461538, 3.23076923])
BLUR_V_WEIGHTS = np.array([0.07027027, 0.31621622, 0.22702703, 0.31621622, 0.07027027])


def gaussian9(src, out_h, out_w, axis):
    """URP's separable 9-tap Gaussian, as a gather from src into an (out_h, out_w) target.

    axis="h": FragBlurH (9 taps; with out = src >> 1 this is also the downsample).
    axis="v": FragBlurV (5 bilinear taps; out = src size).
    Offsets are in *source* texels, as in the shader (_BlitTexture_TexelSize).
    """
    H, W = src.shape[:2]
    u, v = texel_uv(out_h, out_w)
    if axis == "h":
        offsets, weights, du, dv = BLUR_H_OFFSETS, BLUR_H_WEIGHTS, 1.0 / W, 0.0
    elif axis == "v":
        offsets, weights, du, dv = BLUR_V_OFFSETS, BLUR_V_WEIGHTS, 0.0, 1.0 / H
    else:
        raise ValueError(f"axis must be 'h' or 'v', got {axis!r}")
    out = np.zeros((out_h, out_w, src.shape[2]), np.float32)
    for o, w in zip(offsets, weights):
        out += w * sample_bilinear(src, u + o * du, v + o * dv)
    return out


def srgb_to_linear(x):
    """Unity's Mathf.GammaToLinearSpace: the exact piecewise sRGB curve, not pow(x, 2.2)."""
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def soft_threshold(color, threshold, knee):
    """URP's soft-knee threshold on the brightest channel. Linear-space threshold and knee.

    Below threshold - knee: 0. Above threshold + knee: brightness reduced by `threshold`.
    In between: a quadratic toe, so pixels fade in instead of popping.
    """
    brightness = color.max(axis=-1, keepdims=True)
    softness = np.clip(brightness - threshold + knee, 0.0, 2.0 * knee)
    softness = softness * softness / (4.0 * knee + 1e-4)
    multiplier = np.maximum(brightness - threshold, softness) / np.maximum(brightness, 1e-4)
    return color * multiplier


def prefilter(image, threshold, knee, clamp_max, out_h, out_w):
    """FragPrefilter (non-HQ path): one bilinear fetch into the downscaled target,
    then clamp, soft threshold, and clamp to positive."""
    color = sample_bilinear(image, *texel_uv(out_h, out_w))
    color = np.minimum(color, clamp_max)
    color = soft_threshold(color, threshold, knee)
    return np.maximum(color, 0.0).astype(np.float32)


@dataclass(frozen=True)
class URPBloomParams:
    """Bloom Volume settings in inspector units, with URP's defaults."""
    threshold: float = 0.9          # gamma space, as the slider shows it
    intensity: float = 0.0          # URP default: 0 = bloom off
    scatter: float = 0.7            # 0..1, remapped to 0.05..0.95 for the shader
    tint: tuple = (1.0, 1.0, 1.0)
    clamp_max: float = 65472.0
    high_quality: bool = False      # 13-tap prefilter + bicubic upsample (not implemented yet)
    downscale: str = "half"         # "half" or "quarter"
    max_iterations: int = 6


def shader_constants(p):
    """What URP's C# derives from the Volume before uploading _Params."""
    threshold = float(srgb_to_linear(p.threshold))   # Mathf.GammaToLinearSpace
    knee = 0.5 * threshold                           # hard-coded knee
    scatter = 0.05 + 0.9 * p.scatter                 # Mathf.Lerp(0.05, 0.95, scatter)
    return threshold, knee, scatter


def mip_sizes(h, w, p):
    """(h, w) of every mip, starting at the prefilter target, as URP computes them."""
    shift = DOWNSCALE_SHIFT[p.downscale]
    th, tw = h >> shift, w >> shift
    iterations = int(np.floor(np.log2(max(th, tw)) - 1))
    count = int(np.clip(iterations, 1, p.max_iterations))
    sizes = [(th, tw)]
    for _ in range(1, count):
        th, tw = max(1, th >> 1), max(1, tw >> 1)
        sizes.append((th, tw))
    return sizes


def downsample_chain(image, p):
    """mip 0 = prefilter (not blurred); mip i = BlurV(BlurH(mip i-1)) at half the size."""
    threshold, knee, _ = shader_constants(p)
    sizes = mip_sizes(*image.shape[:2], p)
    mips = [prefilter(image, threshold, knee, p.clamp_max, *sizes[0])]
    for h, w in sizes[1:]:
        mips.append(gaussian9(gaussian9(mips[-1], h, w, "h"), h, w, "v"))
    return mips


def upsample_chain(mips, scatter):
    """FragUpsample from the smallest mip up: up[i] = lerp(mip[i], bilinear(up[i+1]), scatter).
    Returns up[0], at the prefilter resolution."""
    up = mips[-1]
    for high in reversed(mips[:-1]):
        h, w = high.shape[:2]
        low = sample_bilinear(up, *texel_uv(h, w))
        up = high + scatter * (low - high)          # lerp(high, low, scatter)
    return up.astype(np.float32)


def bloom_tint(tint):
    """URP's C#: tint.linear divided by its luminance, so the tint changes hue, not brightness."""
    t = srgb_to_linear(tint)
    luma = luminance(t)
    return t / luma if luma > 0 else np.ones(3)


def pyramid_bloom(image, p):
    """Full URP bloom. Returns (composite, bloom), both at the input resolution."""
    if p.high_quality:
        raise NotImplementedError("HQ filtering: check the 17.6 Bloom.shader first")
    _, _, scatter = shader_constants(p)
    bloom_low = upsample_chain(downsample_chain(image, p), scatter)
    h, w = image.shape[:2]
    bloom = sample_bilinear(bloom_low, *texel_uv(h, w)) * (p.intensity * bloom_tint(p.tint))
    return (image + bloom).astype(np.float32), bloom.astype(np.float32)


def effective_kernel(p, h, w, y, x):
    """Composite response of an (h, w) frame to a unit point source at (y, x): the
    pyramid's equivalent of a PSF. Only defined for threshold == 0, where bloom is linear."""
    if p.threshold != 0.0:
        raise ValueError("pyramid bloom only has a kernel (is linear) when threshold == 0")
    delta = np.zeros((h, w, 3), np.float32)
    delta[y, x] = 1.0
    return pyramid_bloom(delta, p)[0]


def level_weights(scatter, n):
    """Weight of mip k in up[0] for an n-level pyramid: the lerp chain unrolled.
    (1 - s) s^k for every level but the coarsest, which keeps s^(n-1). Sums to 1."""
    w = (1.0 - scatter) * scatter ** np.arange(n)
    w[-1] = scatter ** (n - 1)
    return w


def bloom_basis(image, p, n_levels=8):
    """Each mip carried up to mip 0's size through the chain's bilinear upsamples.
    Independent of scatter, intensity and tint, so for any n <= len(basis):
        upsample_chain(first n mips, s) == sum_k level_weights(s, n)[k] * basis[k]"""
    mips = downsample_chain(image, replace(p, max_iterations=n_levels))
    basis = []
    for k, m in enumerate(mips):
        b = m
        for j in range(k - 1, -1, -1):
            h, w = mips[j].shape[:2]
            b = sample_bilinear(b, *texel_uv(h, w))
        basis.append(b)
    return basis


def sample_pixels(image, ref, basis, rng, fraction):
    """A fixed random subset of pixels: input, reference and every basis image there.
    The basis is fetched with the composite's final bilinear upsample."""
    h, w = image.shape[:2]
    idx = rng.choice(h * w, size=int(h * w * fraction), replace=False)
    ys, xs = np.unravel_index(idx, (h, w))
    u, v = (xs + 0.5) / w, (ys + 0.5) / h
    bases = np.stack([sample_bilinear(b, u, v) for b in basis])   # (levels, P, 3)
    return image[ys, xs], ref[ys, xs], bases


def _fit_loss(x, n, samples, p, eps):
    """Joint log loss of a (scatter UI, log10 intensity) guess with an n-level pyramid."""
    scatter_ui, log_intensity = x
    _, _, s = shader_constants(replace(p, scatter=scatter_ui))
    w = level_weights(s, n)
    return np.mean([log_loss(i + 10 ** log_intensity * np.tensordot(w, b[:n], axes=1), r, eps)
                    for i, r, b in samples])


def fit(pairs, thresholds=(0.0, 0.9), downscales=("half",),
        max_levels=8, eps=1e-3, fraction=1 / 16, seed=0):
    """Joint fit of URP bloom to (image, reference) pairs. Grid over threshold, downscale
    and max_iterations; Nelder-Mead over (scatter, log10 intensity). Tint stays white."""
    results = []
    for threshold in thresholds:
        for downscale in downscales:
            p = URPBloomParams(threshold=threshold, downscale=downscale)
            rng = np.random.default_rng(seed)          # same pixels in every grid cell
            samples = [sample_pixels(img, ref, bloom_basis(img, p, max_levels), rng, fraction)
                       for img, ref in pairs]
            n_available = min(b.shape[0] for _, _, b in samples)

            for n in range(2, n_available + 1):
                res = minimize(_fit_loss, x0=(0.7, -1.0), args=(n, samples, p, eps), method="Nelder-Mead",
                               bounds=[(0.0, 1.0), (-4.0, 1.0)],
                               options={"xatol": 1e-3, "fatol": 1e-7})
                row = {"threshold": threshold, "downscale": downscale, "max_iterations": n,
                       "scatter": float(res.x[0]), "intensity": float(10 ** res.x[1]),
                       "loss": float(res.fun), "evaluations": int(res.nfev), "converged": bool(res.success)}
                results.append(row)
                print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()))
    best = min(results, key=lambda r: r["loss"])
    params = URPBloomParams(threshold=best["threshold"], intensity=best["intensity"],
                            scatter=best["scatter"], downscale=best["downscale"],
                            max_iterations=best["max_iterations"])
    return params, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="input EXRs; globs are expanded")
    ap.add_argument("--refs", default="data/reference")
    ap.add_argument("--out", default="outputs/01_gap_study/pyramid_fit.json")
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()

    paths = sorted(f for pattern in args.inputs for f in glob.glob(pattern))
    names = [os.path.splitext(os.path.basename(f))[0] for f in paths]
    pairs = [(read_hdr(f), read_hdr(os.path.join(args.refs, f"{n}_ref.exr")))
             for f, n in zip(paths, names)]
    with open(os.path.join(args.refs, "reference_report.json")) as fh:
        placeholder = json.load(fh)["placeholder"]
    if placeholder:
        print("WARNING: references use PLACEHOLDER lens parameters. Do not report these fits.\n")

    best, table = fit(pairs, eps=args.eps)

    per_image = {}                                   # exact re-score on full frames
    for n, (img, ref) in zip(names, pairs):
        comp, _ = pyramid_bloom(img, best)
        e_in = img.astype(np.float64).sum()
        per_image[n] = {"loss": log_loss(comp, ref, args.eps),
                        "loss_no_bloom": log_loss(img, ref, args.eps),
                        "energy_relative": float(comp.astype(np.float64).sum() / e_in - 1)}
        print(f"{n:<45} loss {per_image[n]['loss']:.4f} "
              f"(no bloom {per_image[n]['loss_no_bloom']:.4f}), "
              f"energy {per_image[n]['energy_relative']:+.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"urp_version": URP_VERSION, "numpy": np.__version__,
                   "placeholder_references": placeholder, "inputs": names, "eps": args.eps,
                   "best": asdict(best), "grid": table, "per_image": per_image}, fh, indent=2)
    print(f"\nbest: {best}\nwrote {args.out}")


if __name__ == "__main__":
    main()

