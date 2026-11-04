// Upsample passes of the glare pyramid. Each level's MLP maps screen position to filter
// weights, and the pass is a plain weighted sum of its taps, so the output is linear in the
// input light whatever the network outputs.
//
// Two taps, this level's mip and the bilinearly upsampled result of the level below, blended
// as a lerp: t = sigmoid(MLP(uv)), w = (1 - t, t). Both weights are positive and sum to 1, so
// no output is negative and no level gains or loses light. URP bloom is the special case
// t = scatter everywhere.

module glare;

import pyramid;
import mlp;

public typealias LevelMLP = MLP<2, 16, 1>;

// This level's blend t at a screen position: t = sigmoid(MLP(uv)). The radial input MLP(H)
// was tried and rejected: it fits the frames worse by 3x the seed noise (decision log
// 2026-11-04). size is unused by MLP(uv); it stays so the callers don't change when an
// input needs the aspect again (field-aligned offsets).
[Differentiable]
public float blend<P : IParams>(no_diff float2 uv, no_diff int2 size, P params, no_diff int offset)
{
    float x[2] = { uv.x, uv.y };
    float z[1] = LevelMLP.eval(params, offset, x);
    return sigmoid(z[0]);
}

[Differentiable]
public float3 upsample<I : IImage, J : IImage, P : IParams>(no_diff int2 texel, no_diff int2 out_size,
                                                            I mip, J low, P params,
                                                            no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float t = blend(uv, out_size, params, offset);
    float3 a = no_diff mip.load(texel);
    float3 b = sample_bilinear(low, uv);
    return (1.0 - t) * a + t * b;
}
