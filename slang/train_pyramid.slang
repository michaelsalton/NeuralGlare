// slangpy entry points for pyramid.slang: wraps tensors as IImage. Training side only.

import slangpy;
import pyramid;

struct TensorImage : IImage
{
    Tensor<float3, 2> tensor;

    int2 size() { return int2(tensor.shape[1], tensor.shape[0]); }
    float3 load(int2 texel) { return tensor.load(texel); }
}

float3 prefilter_pass(int2 texel, int2 out_size, Tensor<float3, 2> src)
{
    TensorImage img = { src };
    return prefilter(texel, out_size, img);
}

float3 blur_h_pass(int2 texel, int2 out_size, Tensor<float3, 2> src)
{
    TensorImage img = { src };
    return blur_h(texel, out_size, img);
}

float3 blur_v_pass(int2 texel, int2 out_size, Tensor<float3, 2> src)
{
    TensorImage img = { src };
    return blur_v(texel, out_size, img);
}