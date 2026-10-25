// Upsample passes of the glare pyramid. Each level's MLP maps screen position to filter
// weights, and the pass is a plain weighted sum of its taps, so the output is linear in the
// input light whatever the network outputs.
//
// Phase 1: two taps, this level's mip and the bilinearly upsampled result of the level below.
// URP bloom is the special case w = (1 - scatter, scatter) everywhere.

module glare;

import pyramid;
import mlp;

public typealias LevelMLP = MLP<2, 16, 2>;

[Differentiable]
public float3 upsample<I : IImage, J : IImage, P : IParams>(no_diff int2 texel, no_diff int2 out_size,
                                                            I mip, J low, P params,
                                                            no_diff int offset)
{
    float2 uv = no_diff texel_uv(texel, out_size);
    float x[2] = { uv.x, uv.y };
    float w[2] = LevelMLP.eval(params, offset, x);
    float3 a = no_diff mip.load(texel);
    float3 b = no_diff sample_bilinear(low, uv);
    return w[0] * a + w[1] * b;
}
