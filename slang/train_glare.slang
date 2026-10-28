// slangpy entry points for glare.slang. Training side only.

import slangpy;
import pyramid;
import mlp;
import glare;

// An (H, W, 3) float tensor, as torch stores an RGB image.
struct TensorImage : IImage
{
    Tensor<float, 3> tensor;

    int2 size() { return int2(tensor.shape[1], tensor.shape[0]); }
    float3 load(int2 texel)
    {
        return float3(tensor.load(texel.y, texel.x, 0), tensor.load(texel.y, texel.x, 1),
                      tensor.load(texel.y, texel.x, 2));
    }
}

struct TensorParams : IParams
{
    DiffTensor<float, 1> tensor;

    [Differentiable]
    float get(int index) { return tensor.load(index); }
}

[Differentiable]
float3 upsample_pass(int2 texel, int2 out_size, Tensor<float, 3> mip, Tensor<float, 3> low,
                     DiffTensor<float, 1> params)
{
    TensorImage m = { mip };
    TensorImage l = { low };
    TensorParams p = { params };
    return upsample(texel, out_size, m, l, p, 0);
}

[Differentiable]
float2 filter_weights(float2 uv, DiffTensor<float, 1> params)
{
    TensorParams p = { params };
    float x[2] = { uv.x, uv.y };
    float w[2] = LevelMLP.eval(p, 0, x);
    return float2(w[0], w[1]);
}
