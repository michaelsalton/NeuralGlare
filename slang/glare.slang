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
    float m = max(max(z[0], z[1]), max(z[2], z[3]));   // softmax ignores a shift: no exp overflow
    float w[TAPS];
    float sum = 0.0;
    [ForceUnroll]
    for (int i = 0; i < TAPS; ++i)
    {
        w[i] = exp(z[i] - m);
        sum += w[i];
    }
    [ForceUnroll]
    for (int i = 0; i < TAPS; ++i)
        w[i] /= sum;
    return w;
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
