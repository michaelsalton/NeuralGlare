// Fixed downsample chain of the glare pyramid, mirroring URP 17.6 bloom
// without its threshold and clamp: inputs stay HDR

// Code reads images through the IImage interface, so the same source runs on Slangpy
// tensors (training) and on engine textures (wrappers in engines)
// Bilinear filtering is done by hand from four loads, so both sides compute the same arithmetic

module pyramid;

public interface IImage
{
    public int2 size();
    public float3 load(int2 texel);
}

// UV of a texel centre in a target of the given size
public float2 texel_uv(int2 texel, int2 size)
{
    return (float2(texel) + 0.5) / float2(size);
}

// bilinear fetch with clamp-to-edge addressing, like a linear-clamp sampler
public float3 sample_bilinear<I : IImage>(I img, float2 uv)
{
    int2 size = img.size();
    float2 p = uv * float2(size) - 0.5; // continuous texel coords: integers are texel centres
    float2 p0 = floor(p);
    float2 f = p - p0; // blend weight toward the right/lower neighbour
    int2 i0 = clamp(int2(p0), int2(0), size - 1);
    int2 i1 = clamp(int2(p0) + 1, int2(0), size - 1);
    float3 top = lerp(img.load(int2(i0.x, i0.y)), img.load(int2(i1.x, i0.y)), f.x);
    float3 bot = lerp(img.load(int2(i0.x, i1.y)), img.load(int2(i1.x, i1.y)), f.x);
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
