// Upsample passes of the glare pyramid. Each level's MLP maps screen position to filter
// weights, and the pass is a plain weighted sum of its taps, so the output is linear in the
// input light whatever the network outputs.
//
// Four taps: this level's mip at the texel, the mip at +o and -o along the direction from the
// frame centre (field-aligned offsets, for coma; decision log 2026-11-04), and the bilinearly
// upsampled result of the level below, mixed by w = softmax(MLP(uv)). The weights are positive
// and sum to 1, so no output is negative and a flat image stays flat. A point's light is
// conserved only up to the offset field's Jacobian: o grows with distance from the centre, so
// the +o / -o taps read the mip magnified / shrunk by 1 +- 1/R (R = the level's half-diagonal in
// texels) and scale its energy by (1 +- 1/R)^-2, up to +-6 % per unit weight at a 60 x 34 level
// (decision log 2026-11-04). With the offset taps at weight 0 it is the Step 1 lerp; URP bloom
// is the special case t = scatter everywhere.

module glare;

import pyramid;
import mlp;

public typealias LevelMLP = MLP<2, 16, 4>;
public static const int TAPS = LevelMLP.OUT_SIZE;   // mip(x), mip(x + o), mip(x - o), up(low)(x)

// The radial tap offset H * d_hat, in texels of a target of this size: the vector from the
// frame centre, scaled so the corners have length 1. It is zero at the centre, so the taps fade
// out there with no seam where d_hat flips, and the measured coma is zero there too.
public float2 field_offset(float2 uv, int2 size)
{
    float2 p = (uv - 0.5) * float2(size);        // texels from the frame centre
    return p / length(0.5 * float2(size));
}

// This level's tap weights at a screen position: softmax(MLP(uv)). Positive and summing to 1;
// with two taps it is the old lerp, sigmoid(z1 - z0). The radial input MLP(H) was tried and
// rejected (decision log 2026-11-04).
[Differentiable]
public float[TAPS] weights<P : IParams>(no_diff float2 uv, P params, no_diff int offset)
{
    float x[2] = { uv.x, uv.y };
    float z[TAPS] = LevelMLP.eval(params, offset, x);
    return softmax<TAPS>(z);
}

[Differentiable]
public float3 upsample<I : IImage, J : IImage, P : IParams>(no_diff int2 texel, no_diff int2 out_size,
                                                            I mip, J low, P params,
                                                            no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float w[TAPS] = weights(uv, params, offset);
    float2 o = no_diff field_offset(uv, out_size) / float2(out_size);   // texels -> uv
    float3 c = no_diff mip.load(texel);
    float3 a = sample_bilinear(mip, uv + o);
    float3 b = sample_bilinear(mip, uv - o);
    float3 l = sample_bilinear(low, uv);
    return w[0] * c + w[1] * a + w[2] * b + w[3] * l;
}

// ------------------------------------------------------------------------------------- spikes
// The aperture's starburst (decision log 2026-11-04), a term after the full-resolution level.
// 14 rays at fixed angles k * 360 / 14 (the 7-blade diaphragm doesn't rotate), in two families:
// even k and odd k, with different widths and falloffs. The ray opposite ray k is ray k + 7, of
// the other family, so every line through a source is asymmetric.
//
// Each ray is a streak filter (as Kawase's star filter) on mip 0, built in PASSES passes:
//     S_0 = mip0,   S_{p+1}(x) = sum_n c[f][p][n] * S_p(x - n * 4^p * e_k),   n = 0..3
// Pass p moves light by one base-4 digit (0..3) x 4^p texels, so the three passes reach every
// distance 0..63 texels exactly once: a point gets a continuous ray, and the ray's profile is
// the product of one weight per digit. c[f][p] = softmax of this family's logits for the pass:
// positive and summing to 1, so each ray keeps its light. The logits are global (one profile per
// family), not per pixel; SpikeMLP(uv) sets how much light goes to each family per pixel.

public static const int RAYS = 14;
public static const int PASSES = 3;
public static const int STREAK_TAPS = 4;
public static const int PROFILE_COUNT = 2 * PASSES * STREAK_TAPS;  // logits, family-major
public typealias SpikeMLP = MLP<2, 16, 3>;                         // pyramid, even rays, odd rays

// e_k = (cos, sin) of k * 360 / 14 in texel space (x right, y down), as literals so every
// compiler gets the same values.
static const float2 RAY_DIRS[RAYS] = {
    float2(1.000000000, 0.000000000),
    float2(0.900968868, 0.433883739),
    float2(0.623489802, 0.781831482),
    float2(0.222520934, 0.974927912),
    float2(-0.222520934, 0.974927912),
    float2(-0.623489802, 0.781831482),
    float2(-0.900968868, 0.433883739),
    float2(-1.000000000, 0.000000000),
    float2(-0.900968868, -0.433883739),
    float2(-0.623489802, -0.781831482),
    float2(-0.222520934, -0.974927912),
    float2(0.222520934, -0.974927912),
    float2(0.623489802, -0.781831482),
    float2(0.900968868, -0.433883739),
};

// One pass of ray `ray`: S_{stage+1} at a texel from S_stage (src). profile: where the logits
// start in the parameter buffer. The gather reads from x - d * e_k, so a source's light lands
// at +d * e_k: the ray points along e_k. Zero outside the frame (sample_bilinear_zero).
[Differentiable]
public float3 streak<I : IImage, P : IParams>(no_diff int2 texel, I src, P params,
                                              no_diff int profile, no_diff int ray,
                                              no_diff int stage)
{
    int base = profile + ((ray % 2) * PASSES + stage) * STREAK_TAPS;
    float z[STREAK_TAPS];
    [ForceUnroll]
    for (int n = 0; n < STREAK_TAPS; ++n)
        z[n] = params.get(base + n);
    float c[STREAK_TAPS] = softmax<STREAK_TAPS>(z);
    float2 step = RAY_DIRS[ray] * float(1 << (2 * stage));         // 4^stage texels
    float3 sum = float3(0.0);
    [ForceUnroll]
    for (int n = 0; n < STREAK_TAPS; ++n)
        sum += c[n] * sample_bilinear_zero(src, float2(texel) - float(n) * step);
    return sum;
}

// The last pass of a ray, added to its family's running sum (acc) of the rays before it.
[Differentiable]
public float3 streak_accumulate<I : IImage, J : IImage, P : IParams>(no_diff int2 texel, I src,
                                                                     J acc, P params,
                                                                     no_diff int profile,
                                                                     no_diff int ray,
                                                                     no_diff int stage)
{
    return acc.load(texel) + streak(texel, src, params, profile, ray, stage);
}

// How the full-resolution output and the two families mix at a screen position:
// softmax(SpikeMLP(uv)) over (pyramid, even rays, odd rays).
[Differentiable]
public float[3] spike_weights<P : IParams>(no_diff float2 uv, P params, no_diff int offset)
{
    float x[2] = { uv.x, uv.y };
    float z[3] = SpikeMLP.eval(params, offset, x);
    return softmax<3>(z);
}

// The final frame: base is the full-resolution level's output, even / odd the sums of each
// family's 7 rays at mip 0's resolution. Dividing by 7 makes each family's mean ray carry its
// weight's share of the light, so the weights still sum to 1 and light is kept.
[Differentiable]
public float3 spike_composite<I : IImage, J : IImage, K : IImage, P : IParams>(
    no_diff int2 texel, no_diff int2 out_size, I base, J even, K odd, P params, no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float w[3] = spike_weights(uv, params, offset);
    float3 e = sample_bilinear(even, uv);
    float3 o = sample_bilinear(odd, uv);
    return w[0] * base.load(texel) + (w[1] * e + w[2] * o) * (1.0 / float(RAYS / 2));
}
