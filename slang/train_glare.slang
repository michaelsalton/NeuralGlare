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
    [Differentiable] float3 load(int2 texel)
    {
        return float3(no_diff tensor.load(texel.y, texel.x, 0), no_diff tensor.load(texel.y, texel.x, 1),
                      no_diff tensor.load(texel.y, texel.x, 2));
    }
}

struct DiffTensorImage : IImage
{
    DiffTensor<float, 3> tensor;
    int2 size() { return int2(tensor.shape[1], tensor.shape[0]); }
    [Differentiable]
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

// The downsample chain on torch-layout (H, W, 3) tensors, so the whole pyramid runs in Slang
// (train_pyramid.slang's Tensor<float3, 2> doesn't accept torch tensors). Logic: pyramid.slang.
float3 prefilter_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return prefilter(texel, out_size, img);
}

float3 blur_h_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return blur_h(texel, out_size, img);
}

float3 blur_v_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return blur_v(texel, out_size, img);
}

// Teacher forcing (step 3): low is a fixed input, so no gradient flows into it.
[Differentiable]
float3 upsample_pass(int2 texel, int2 out_size, Tensor<float, 3> mip, Tensor<float, 3> low,
                     DiffTensor<float, 1> params)
{
    TensorImage m = { mip };
    TensorImage l = { low };
    TensorParams p = { params };
    return upsample(texel, out_size, m, l, p, 0);
}

// End to end (step 4): low is the output of the level below, so the loss on the final image
// flows back through it. offset selects this level's network in the shared parameter buffer.
[Differentiable]
float3 upsample_chain_pass(int2 texel, int2 out_size, Tensor<float, 3> mip,
                           DiffTensor<float, 3> low, DiffTensor<float, 1> params, int offset)
{
    TensorImage m = { mip };
    DiffTensorImage l = { low };
    TensorParams p = { params };
    return upsample(texel, out_size, m, l, p, offset);
}

[Differentiable]
float2 filter_weights(float2 uv, DiffTensor<float, 1> params)
{
    TensorParams p = { params };
    float x[2] = { uv.x, uv.y };
    float w[2] = LevelMLP.eval(p, 0, x);
    return float2(w[0], w[1]);
}
