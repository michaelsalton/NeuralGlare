// Upsample passes of the glare pyramid. Each level's MLP maps screen position to filter
// weights, and the pass is a plain weighted sum of its taps, so the output is linear in the
// input light whatever the network outputs.
//
// Five taps: this level's mip at the texel, the mip at +o and -o along the direction from the
// frame centre (field-aligned offsets, for coma; decision log 2026-11-04), the result of the
// level below upsampled with a 3x3 tent (sample_tent; bilinear before decision log 2026-11-12,
// which made a point in an unblurred mip 0 a square block), and the 3x3 tent of this level's own
// mip (decision log 2026-11-13): at the full-resolution level, whose mip is the input frame, a
// round +-1 px shoulder that, unlike the offset taps, doesn't vanish at the frame centre (o = 0),
// and the 5x5 binomial of this level's own mip (sample_binomial5, decision log 2026-11-16): at the
// full-resolution level the off-axis +-2 px shoulder, which no mix of the other taps reaches
// without spilling past 3 px. Six taps. Mixed by w = softmax(MLP(uv)), one softmax per colour channel (per-channel weights, decision
// log 2026-11-09). Each channel's weights are positive
// and sum to 1, so no output is negative and a flat image stays flat. A point's light is
// conserved only up to the offset field's Jacobian: o grows with distance from the centre, so
// the +o / -o taps read the mip magnified / shrunk by 1 +- 1/R (R = the level's half-diagonal in
// texels) and scale its energy by (1 +- 1/R)^-2, up to +-6 % per unit weight at a 60 x 34 level
// (decision log 2026-11-04). With the offset and own-mip taps at weight 0 it is the Step 1 lerp;
// URP bloom with a tent upsample is the special case t = scatter everywhere.

module glare;

import pyramid;
import mlp;

public static const int CHANNELS = 3;
public typealias LevelMLP = MLP<2, 16, 18>;                   // TAPS x CHANNELS, channel-major
public static const int TAPS = LevelMLP.OUT_SIZE / CHANNELS;  // mip(x), mip(x + o), mip(x - o), up(low)(x),
                                                              // tent(mip)(x), binomial5(mip)(x): the
                                                              // low tap stays index 3

// One softmax per colour channel: z holds N logits per channel, channel-major (R's N, then G's,
// then B's; M = N * CHANNELS). Each channel's weights are positive and sum to 1, so each channel
// keeps its own light, and a white source keeps its total colour.
[Differentiable]
public float3[N] softmax_rgb<let N : int, let M : int>(float z[M])
{
    float3 w[N];
    [ForceUnroll]
    for (int c = 0; c < CHANNELS; ++c)
    {
        float zc[N];
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            zc[i] = z[c * N + i];
        float wc[N] = softmax<N>(zc);
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            w[i][c] = wc[i];
    }
    return w;
}

// The radial tap offset H * d_hat, in texels of a target of this size: the vector from the
// frame centre, scaled so the corners have length 1. It is zero at the centre, so the taps fade
// out there with no seam where d_hat flips, and the measured coma is zero there too.
public float2 field_offset(float2 uv, int2 size)
{
    float2 p = (uv - 0.5) * float2(size);        // texels from the frame centre
    return p / length(0.5 * float2(size));
}

// This level's tap weights at a screen position: softmax(MLP(uv)) per channel. Positive and
// summing to 1 in each channel; with two taps it is the old lerp, sigmoid(z1 - z0). The radial
// input MLP(H) was tried and rejected (decision log 2026-11-04).
[Differentiable]
public float3[TAPS] weights<P : IParams>(no_diff float2 uv, P params, no_diff int offset)
{
    float x[2] = { uv.x, uv.y };
    float z[LevelMLP.OUT_SIZE] = LevelMLP.eval(params, offset, x);
    return softmax_rgb<TAPS, LevelMLP.OUT_SIZE>(z);
}

[Differentiable]
public float3 upsample<I : IImage, J : IImage, P : IParams>(no_diff int2 texel, no_diff int2 out_size,
                                                            I mip, J low, P params,
                                                            no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float3 w[TAPS] = weights(uv, params, offset);                     // per channel
    float2 o = no_diff field_offset(uv, out_size) / float2(out_size);   // texels -> uv
    float3 c = no_diff mip.load(texel);
    float3 a = sample_bilinear(mip, uv + o);
    float3 b = sample_bilinear(mip, uv - o);
    float3 l = sample_tent(low, uv);    // the offset taps a, b stay bilinear: blurring them smears the coma
    float3 s = sample_tent(mip, uv);    // own-mip tent: the +-1 texel shoulder (2026-11-13)
    float3 g = sample_binomial5(mip, uv); // own-mip 5x5 binomial: the +-2 texel shoulder (2026-11-16)
    return w[0] * c + w[1] * a + w[2] * b + w[3] * l + w[4] * s + w[5] * g;
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
// family, shared by the channels), not per pixel; SpikeMLP(uv) sets how much light goes to each
// family per pixel and channel. Grey profiles suffice: the exact halo's tint is constant over
// 6..64 px (decision log 2026-11-09).

public static const int RAYS = 14;
public static const int PASSES = 3;
public static const int STREAK_TAPS = 4;
public static const int PROFILE_COUNT = 2 * PASSES * STREAK_TAPS;  // logits, family-major
public static const int MIX = 3;                                   // pyramid, even rays, odd rays
public typealias SpikeMLP = MLP<2, 16, 9>;                         // MIX x CHANNELS, channel-major

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

// ------------------------------------------------------------------- per-ray cross filter
// After a ray's last streak pass, CROSS_TAPS taps across the ray (decision log 2026-11-19):
//     S'(x) = sum_m a_m(x, k) * S_3(x - (m - 1) * n_k),   n_k = (-e_y, e_x), m = 0, 1, 2
// so tap m moves a source's light by (m - 1) px across ray k: symmetric weights widen the ray,
// lopsided ones shift it sideways, (0, 1, 0) is the plain ray. a = softmax(RayMLP(x)), positive and
// summing to 1. Input x = (e_k . v, n_k . v, +-1 for the even / odd family), v = field_offset =
// H * d_hat at the receiving texel: the field in the ray's own frame, no trig, continuous at the
// frame centre (where it becomes one filter per family). Grey, as the ray profiles. Light is kept up
// to the frame border (read as 0) and up to the weights being taken at the receiver, not the source.

public static const int CROSS_TAPS = 3;                    // -1, 0, +1 px across the ray
public typealias RayMLP = MLP<3, 8, CROSS_TAPS>;          // (e.v, n.v, family) -> tap logits

[Differentiable]
public float[CROSS_TAPS] cross_weights<P : IParams>(no_diff float2 v, P params,
                                                    no_diff int offset, no_diff int ray)
{
    float2 e = RAY_DIRS[ray];
    float2 n = float2(-e.y, e.x);
    float x[3] = { dot(e, v), dot(n, v), (ray % 2 == 0) ? 1.0 : -1.0 };
    float z[CROSS_TAPS] = RayMLP.eval(params, offset, x);
    return softmax<CROSS_TAPS>(z);
}

// A ray's cross filter on its last pass's output src (S_3): S'_k at a texel. The gather reads from
// x - (m - 1) * n_k, so tap m's light lands at +(m - 1) * n_k (as streak).
[Differentiable]
public float3 ray_cross<I : IImage, P : IParams>(no_diff int2 texel, no_diff int2 out_size, I src,
                                                 P params, no_diff int offset, no_diff int ray)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float2 v = no_diff field_offset(uv, out_size);
    float a[CROSS_TAPS] = cross_weights(v, params, offset, ray);
    float2 e = RAY_DIRS[ray];
    float2 n = float2(-e.y, e.x);
    float3 sum = float3(0.0);
    [ForceUnroll]
    for (int m = 0; m < CROSS_TAPS; ++m)
        sum += a[m] * sample_bilinear_zero(src, float2(texel) - float(m - CROSS_TAPS / 2) * n);
    return sum;
}

// The cross-filtered ray added to its family's running sum.
[Differentiable]
public float3 ray_cross_accumulate<I : IImage, J : IImage, P : IParams>(
    no_diff int2 texel, no_diff int2 out_size, I src, J acc, P params, no_diff int offset,
    no_diff int ray)
{
    return acc.load(texel) + ray_cross(texel, out_size, src, params, offset, ray);
}

// ---------------------------------------------------------------------- per-ray brightness
// Ray k's gain (decision log 2026-11-19): g_k(x) = 7 softmax_j(z_j(x)) over its family's 7 rays,
// z = BrightMLP(e_j . v, n_j . v, +-1 for the family), RayMLP's inputs, grey. g sums to 7 over a
// family, so it moves light between a family's rays and SpikeMLP still sets the family's total.
// The softmax's normaliser is pulled out of the family sum:
//     sum_k g_k S'_k = (7 / Z) sum_k e^{z_k} S'_k,   Z = sum_j e^{z_j}
// so each ray's cross pass adds e^{z_k} S'_k (ray_bright_accumulate, one BrightMLP evaluation) and
// the composite divides each family's sum by its Z (spike_composite_bright) where spike_composite
// divides by 7. At v = 0 a family's rays all get the same input, so g = 1: no coma on axis.
// Layout: BrightMLP follows RayMLP. BrightMLP's output bias is common to every ray, so softmax
// cancels it: its gradient is 0.

public typealias BrightMLP = MLP<3, 8, 1>;                 // (e.v, n.v, family) -> brightness logit

[Differentiable]
public float bright_logit<P : IParams>(no_diff float2 v, P params, no_diff int offset,
                                       no_diff int ray)
{
    float2 e = RAY_DIRS[ray];
    float2 n = float2(-e.y, e.x);
    float x[3] = { dot(e, v), dot(n, v), (ray % 2 == 0) ? 1.0 : -1.0 };
    float z[1] = BrightMLP.eval(params, offset, x);
    return z[0];
}

// Z = sum of e^{z_j} over a family's 7 rays (family 0 even, 1 odd) at field vector v.
[Differentiable]
public float bright_norm<P : IParams>(no_diff float2 v, P params, no_diff int offset,
                                      no_diff int family)
{
    float sum = 0.0;
    [ForceUnroll]
    for (int j = 0; j < RAYS / 2; ++j)
        sum += exp(bright_logit(v, params, offset, 2 * j + family));
    return sum;
}

// g_k = 7 e^{z_k} / Z: the gain itself (the pre-fit and the views; the passes never form it).
[Differentiable]
public float bright_gain<P : IParams>(no_diff float2 v, P params, no_diff int offset,
                                      no_diff int ray)
{
    return float(RAYS / 2) * exp(bright_logit(v, params, offset, ray))
           / bright_norm(v, params, offset, ray % 2);
}

// The cross-filtered ray times e^{z_k}, added to its family's running sum. offset: RayMLP's start
// (BrightMLP's is after it).
[Differentiable]
public float3 ray_bright_accumulate<I : IImage, J : IImage, P : IParams>(
    no_diff int2 texel, no_diff int2 out_size, I src, J acc, P params, no_diff int offset,
    no_diff int ray)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float2 v = no_diff field_offset(uv, out_size);
    float z = bright_logit(v, params, offset + RayMLP.PARAM_COUNT, ray);
    return acc.load(texel) + exp(z) * ray_cross(texel, out_size, src, params, offset, ray);
}

// How the full-resolution output and the two families mix at a screen position:
// softmax(SpikeMLP(uv)) over (pyramid, even rays, odd rays), one per channel.
[Differentiable]
public float3[MIX] spike_weights<P : IParams>(no_diff float2 uv, P params, no_diff int offset)
{
    float x[2] = { uv.x, uv.y };
    float z[SpikeMLP.OUT_SIZE] = SpikeMLP.eval(params, offset, x);
    return softmax_rgb<MIX, SpikeMLP.OUT_SIZE>(z);
}

// The final frame: base is the full-resolution level's output, even / odd the sums of each
// family's 7 rays at mip 0's resolution. Dividing by 7 makes each family's mean ray carry its
// weight's share of the light, so the weights still sum to 1 in each channel and light is kept.
[Differentiable]
public float3 spike_composite<I : IImage, J : IImage, K : IImage, P : IParams>(
    no_diff int2 texel, no_diff int2 out_size, I base, J even, K odd, P params, no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float3 w[MIX] = spike_weights(uv, params, offset);
    float3 e = sample_bilinear(even, uv);
    float3 o = sample_bilinear(odd, uv);
    return w[0] * base.load(texel) + (w[1] * e + w[2] * o) * (1.0 / float(RAYS / 2));
}

// spike_composite with per-ray brightness: even / odd hold sum_k e^{z_k} S'_k (ray_bright_accumulate),
// so each family is divided by its normaliser Z at this texel instead of by 7. offset: SpikeMLP's
// start; BrightMLP's follows the profile logits and RayMLP.
[Differentiable]
public float3 spike_composite_bright<I : IImage, J : IImage, K : IImage, P : IParams>(
    no_diff int2 texel, no_diff int2 out_size, I base, J even, K odd, P params, no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float3 w[MIX] = spike_weights(uv, params, offset);
    float2 v = no_diff field_offset(uv, out_size);
    int bright = offset + SpikeMLP.PARAM_COUNT + PROFILE_COUNT + RayMLP.PARAM_COUNT;
    float3 e = sample_bilinear(even, uv);
    float3 o = sample_bilinear(odd, uv);
    return w[0] * base.load(texel) + w[1] * e / bright_norm(v, params, bright, 0)
           + w[2] * o / bright_norm(v, params, bright, 1);
}
