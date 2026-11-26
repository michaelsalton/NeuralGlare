// Fixed downsample chain of the glare pyramid, mirroring URP 17.6 bloom
// without its threshold and clamp: inputs stay HDR

// Code reads images through the IImage interface, so the same source runs on Slangpy
// tensors (training) and on engine textures (wrappers in engines)
// A bilinear tap is the image's job (IImage.bilinear / bilinear_zero): training images do it by hand
// from four loads (bilinear_from_loads), engine textures may use the hardware filter (efficiency
// step 1e, decision log 2026-11-24). The filters built from taps stay here and in glare.slang

module pyramid;

public interface IImage
{
    public int2 size();
    [Differentiable] public float3 load(int2 texel);
    // A bilinear tap at continuous texel coordinates p (integers are texel centres): clamp-to-edge
    // (bilinear) or 0 outside the image (bilinear_zero). Training images do it from four loads
    // (bilinear_from_loads); engine textures may use the hardware filter (efficiency step 1e).
    [Differentiable] public float3 bilinear(no_diff float2 p);
    [Differentiable] public float3 bilinear_zero(no_diff float2 p);
}

// A stack of same-sized images read by slice (efficiency step 1c, decision log 2026-11-24): every
// ray's last streak pass (the ray stack). Engine textures are texture arrays; training arrays
// are tensors with a leading slice axis.
public interface IImageArray
{
    public int2 size();
    [Differentiable] public float3 load(int slice, int2 texel);
    [Differentiable] public float3 bilinear(int slice, no_diff float2 p);
    [Differentiable] public float3 bilinear_zero(int slice, no_diff float2 p);
}

// One slice of an image array as an image, so every filter on IImage runs on it unchanged.
public struct ImageSlice<A : IImageArray> : IImage
{
    public A images;
    public int slice;

    public int2 size() { return images.size(); }
    [Differentiable] public float3 load(int2 texel) { return images.load(slice, texel); }
    [Differentiable] public float3 bilinear(no_diff float2 p) { return images.bilinear(slice, p); }
    [Differentiable] public float3 bilinear_zero(no_diff float2 p) { return images.bilinear_zero(slice, p); }
}

// UV of a texel centre in a target of the given size
public float2 texel_uv(int2 texel, int2 size)
{
    return (float2(texel) + 0.5) / float2(size);
}

// bilinear fetch with clamp-to-edge addressing, like a linear-clamp sampler
[Differentiable]
public float3 sample_bilinear<I : IImage>(I img, no_diff float2 uv)
{
    return img.bilinear(uv * float2(img.size()) - 0.5); // continuous texel coords: integers are texel centres
}

// A clamp-to-edge bilinear tap at texel coordinates p from four loads (IImage.bilinear by hand)
[Differentiable]
public float3 bilinear_from_loads<I : IImage>(I img, no_diff float2 p)
{
    int2 size = img.size();
    float2 p0 = no_diff floor(p); // geometry only: gradients flow through the loads, not the weights
    float2 f = p - p0; // blend weight toward the right/lower neighbour
    int2 i0 = clamp(int2(p0), int2(0), size - 1);
    int2 i1 = clamp(int2(p0) + 1, int2(0), size - 1);
    float3 top = lerp(img.load(int2(i0.x, i0.y)), img.load(int2(i1.x, i0.y)), f.x);
    float3 bot = lerp(img.load(int2(i0.x, i1.y)), img.load(int2(i1.x, i1.y)), f.x);
    return lerp(top, bot, f.y);
}

// 3x3 tent filter (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare",
// SIGGRAPH 2014): 9 bilinear taps one texel of img apart, weights 1 2 1 / 2 4 2 / 1 2 1, / 16.
// The taps are one texel of the source apart (not of the target), so a single bright texel
// spreads into a round blob instead of staying a bilinear block. The weights sum to 1, so light is
// conserved (clamp-to-edge, as sample_bilinear). Decision log 2026-11-12.
[Differentiable]
public float3 sample_tent<I : IImage>(I img, no_diff float2 uv)
{
    float2 d = 1.0 / float2(img.size()); // one texel of img, in uv (geometry: size() is an int)
    float3 s = 4.0 * sample_bilinear(img, uv);
    s += 2.0 * (sample_bilinear(img, uv + float2(d.x, 0.0)) + sample_bilinear(img, uv - float2(d.x, 0.0))
              + sample_bilinear(img, uv + float2(0.0, d.y)) + sample_bilinear(img, uv - float2(0.0, d.y)));
    s += sample_bilinear(img, uv + d) + sample_bilinear(img, uv - d)
       + sample_bilinear(img, uv + float2(d.x, -d.y)) + sample_bilinear(img, uv + float2(-d.x, d.y));
    return s / 16.0;
}

// 5x5 binomial, [1 4 6 4 1]^2 / 256 (the tent applied twice, a Gaussian of sigma = 1 texel), from 9
// bilinear taps by linear sampling: a tap 1.2 texels out with weight 5/16 blends texels 1 and 2 at
// 0.8 / 0.2, giving their 4/16 and 1/16; the centre keeps 6/16. Separable weights (5, 6, 5) / 16 at
// offsets -1.2, 0, 1.2 texels of img. Round, sums to 1, clamp-to-edge (as sample_tent). A point
// keeps 14 % in its texel and puts 22 % at ~2 texels: the off-axis +-2 px shoulder (decision log
// 2026-11-16).
static const float BINOMIAL5_OFFSETS[3] = { -1.2, 0.0, 1.2 };
static const float BINOMIAL5_WEIGHTS[3] = { 0.3125, 0.375, 0.3125 };

[Differentiable]
public float3 sample_binomial5<I : IImage>(I img, no_diff float2 uv)
{
    float2 d = 1.0 / float2(img.size()); // one texel of img, in uv
    float3 s = float3(0.0);
    [ForceUnroll]
    for (int j = 0; j < 3; ++j)
    {
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            s += (BINOMIAL5_WEIGHTS[i] * BINOMIAL5_WEIGHTS[j])
                 * sample_bilinear(img, uv + float2(BINOMIAL5_OFFSETS[i], BINOMIAL5_OFFSETS[j]) * d);
    }
    return s;
}

// A texel, or 0 outside the image (a clamp-to-border sampler with a black border).
[Differentiable]
public float3 load_zero<I : IImage>(I img, no_diff int2 texel)
{
    int2 size = img.size();
    if (any(texel < int2(0)) || any(texel >= size))
        return float3(0.0);
    return img.load(texel);
}

// Bilinear fetch at continuous texel coordinates p (integers are texel centres), 0 outside the
// image. For long gathers (the spike streaks), where clamp-to-edge would repeat the border texel
// once per tap and add light that isn't there.
[Differentiable]
public float3 sample_bilinear_zero<I : IImage>(I img, no_diff float2 p)
{
    return img.bilinear_zero(p);
}

// sample_bilinear_zero from four loads (IImage.bilinear_zero by hand)
[Differentiable]
public float3 bilinear_zero_from_loads<I : IImage>(I img, no_diff float2 p)
{
    float2 p0 = no_diff floor(p); // geometry only, as bilinear_from_loads
    float2 f = p - p0;
    int2 i0 = int2(p0);
    float3 top = lerp(load_zero(img, i0), load_zero(img, i0 + int2(1, 0)), f.x);
    float3 bot = lerp(load_zero(img, i0 + int2(0, 1)), load_zero(img, i0 + int2(1, 1)), f.x);
    return lerp(top, bot, f.y);
}

// Mip 0: one bilinear fetch into the half-resolution target (URP FragPrefilter, no threshold).
public float3 prefilter<I : IImage>(int2 texel, int2 out_size, I src)
{
    return sample_bilinear(src, texel_uv(texel, out_size));
}

// URP FragBlurH: 9 taps 2 source texels apart. With out_size = src size / 2 it also downsamples.
static const float BLUR_H_OFFSETS[9] = { -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0 };
static const float BLUR_H_WEIGHTS[9] = { 0.01621622, 0.05405405, 0.12162162, 0.19459459, 0.22702703,
                                         0.19459459, 0.12162162, 0.05405405, 0.01621622 };

public float3 blur_h<I : IImage>(int2 texel, int2 out_size, I src)
{
    float2 uv = texel_uv(texel, out_size);
    float du = 1.0 / float(src.size().x);
    float3 sum = float3(0.0);
    [ForceUnroll]
    for (int i = 0; i < 9; ++i)
        sum += BLUR_H_WEIGHTS[i] * sample_bilinear(src, uv + float2(BLUR_H_OFFSETS[i] * du, 0.0));
    return sum;
}

// URP FragBlurV: the same 9-tap Gaussian at equal resolution, as 5 bilinear taps.
static const float BLUR_V_OFFSETS[5] = { -3.23076923, -1.38461538, 0.0, 1.38461538, 3.23076923 };
static const float BLUR_V_WEIGHTS[5] = { 0.07027027, 0.31621622, 0.22702703, 0.31621622, 0.07027027 };

public float3 blur_v<I : IImage>(int2 texel, int2 out_size, I src)
{
    float2 uv = texel_uv(texel, out_size);
    float dv = 1.0 / float(src.size().y);
    float3 sum = float3(0.0);
    [ForceUnroll]
    for (int i = 0; i < 5; ++i)
        sum += BLUR_V_WEIGHTS[i] * sample_bilinear(src, uv + float2(0.0, BLUR_V_OFFSETS[i] * dv));
    return sum;
}
