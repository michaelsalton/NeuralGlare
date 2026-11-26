#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif

#ifndef __DXC_VERSION_MAJOR
// warning X3557: loop doesn't seem to do anything, forcing loop to unroll
#pragma warning(disable : 3557)
#endif

RWTexture2D<float4 > _Target : register(u0);

Texture2D<float4 > _Source : register(t0);

Texture2DArray<float4 > _RayStack : register(t5);

StructuredBuffer<float > _GlareParams : register(t6);

SamplerState sampler_LinearClamp : register(s0);

struct ImageSlice_0
{
    int slice_0;
};

int2 TextureArray_size_0(Texture2DArray<float4 > this_tex_0)
{
    uint w_0;
    uint h_0;
    uint n_0;
    this_tex_0.GetDimensions(w_0, h_0, n_0);
    return int2(int(w_0), int(h_0));
}

int2 ImageSlice_size_0(ImageSlice_0 this_0, Texture2DArray<float4 > this_images_tex_0)
{
    return TextureArray_size_0(this_images_tex_0);
}

float3 TextureArray_load_0(Texture2DArray<float4 > this_tex_1, int slice_1, int2 texel_0)
{
    return this_tex_1.Load(int4(texel_0, slice_1, int(0))).xyz;
}

float3 ImageSlice_load_0(ImageSlice_0 this_1, Texture2DArray<float4 > this_images_tex_1, int2 texel_1)
{
    return TextureArray_load_0(this_images_tex_1, this_1.slice_0, texel_1);
}

float3 load_zero_0(ImageSlice_0 img_0, Texture2DArray<float4 > img_images_tex_0, int2 texel_2)
{
    int2 _S1 = ImageSlice_size_0(img_0, img_images_tex_0);
    bool _S2;
    if(any(texel_2 < ((int2)int(0))))
    {
        _S2 = true;
    }
    else
    {
        _S2 = any(texel_2 >= _S1);
    }
    if(_S2)
    {
        return (float3)0.0f;
    }
    return ImageSlice_load_0(img_0, img_images_tex_0, texel_2);
}

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void dense_0(StructuredBuffer<float > params_buf_0, int offset_0, float  x_0[int(2)], out float  _S3[int(16)])
{
    int _S4 = offset_0 + int(32);
    float  y_0[int(16)];
    y_0[int(0)] = BufferParams_get_0(params_buf_0, _S4) + BufferParams_get_0(params_buf_0, offset_0) * x_0[int(0)] + BufferParams_get_0(params_buf_0, offset_0 + int(1)) * x_0[int(1)];
    int _S5 = offset_0 + int(2);
    y_0[int(1)] = BufferParams_get_0(params_buf_0, _S4 + int(1)) + BufferParams_get_0(params_buf_0, _S5) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S5 + int(1)) * x_0[int(1)];
    int _S6 = offset_0 + int(4);
    y_0[int(2)] = BufferParams_get_0(params_buf_0, _S4 + int(2)) + BufferParams_get_0(params_buf_0, _S6) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S6 + int(1)) * x_0[int(1)];
    int _S7 = offset_0 + int(6);
    y_0[int(3)] = BufferParams_get_0(params_buf_0, _S4 + int(3)) + BufferParams_get_0(params_buf_0, _S7) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S7 + int(1)) * x_0[int(1)];
    int _S8 = offset_0 + int(8);
    y_0[int(4)] = BufferParams_get_0(params_buf_0, _S4 + int(4)) + BufferParams_get_0(params_buf_0, _S8) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S8 + int(1)) * x_0[int(1)];
    int _S9 = offset_0 + int(10);
    y_0[int(5)] = BufferParams_get_0(params_buf_0, _S4 + int(5)) + BufferParams_get_0(params_buf_0, _S9) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S9 + int(1)) * x_0[int(1)];
    int _S10 = offset_0 + int(12);
    y_0[int(6)] = BufferParams_get_0(params_buf_0, _S4 + int(6)) + BufferParams_get_0(params_buf_0, _S10) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S10 + int(1)) * x_0[int(1)];
    int _S11 = offset_0 + int(14);
    y_0[int(7)] = BufferParams_get_0(params_buf_0, _S4 + int(7)) + BufferParams_get_0(params_buf_0, _S11) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S11 + int(1)) * x_0[int(1)];
    int _S12 = offset_0 + int(16);
    y_0[int(8)] = BufferParams_get_0(params_buf_0, _S4 + int(8)) + BufferParams_get_0(params_buf_0, _S12) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S12 + int(1)) * x_0[int(1)];
    int _S13 = offset_0 + int(18);
    y_0[int(9)] = BufferParams_get_0(params_buf_0, _S4 + int(9)) + BufferParams_get_0(params_buf_0, _S13) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S13 + int(1)) * x_0[int(1)];
    int _S14 = offset_0 + int(20);
    y_0[int(10)] = BufferParams_get_0(params_buf_0, _S4 + int(10)) + BufferParams_get_0(params_buf_0, _S14) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S14 + int(1)) * x_0[int(1)];
    int _S15 = offset_0 + int(22);
    y_0[int(11)] = BufferParams_get_0(params_buf_0, _S4 + int(11)) + BufferParams_get_0(params_buf_0, _S15) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S15 + int(1)) * x_0[int(1)];
    int _S16 = offset_0 + int(24);
    y_0[int(12)] = BufferParams_get_0(params_buf_0, _S4 + int(12)) + BufferParams_get_0(params_buf_0, _S16) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S16 + int(1)) * x_0[int(1)];
    int _S17 = offset_0 + int(26);
    y_0[int(13)] = BufferParams_get_0(params_buf_0, _S4 + int(13)) + BufferParams_get_0(params_buf_0, _S17) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S17 + int(1)) * x_0[int(1)];
    int _S18 = offset_0 + int(28);
    y_0[int(14)] = BufferParams_get_0(params_buf_0, _S4 + int(14)) + BufferParams_get_0(params_buf_0, _S18) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S18 + int(1)) * x_0[int(1)];
    int _S19 = offset_0 + int(30);
    y_0[int(15)] = BufferParams_get_0(params_buf_0, _S4 + int(15)) + BufferParams_get_0(params_buf_0, _S19) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S19 + int(1)) * x_0[int(1)];
    _S3 = y_0;
    return;
}

void dense_1(StructuredBuffer<float > params_buf_1, int offset_1, float  x_1[int(16)], out float  _S20[int(9)])
{
    int _S21 = offset_1 + int(144);
    float  y_1[int(9)];
    y_1[int(0)] = BufferParams_get_0(params_buf_1, _S21) + BufferParams_get_0(params_buf_1, offset_1) * x_1[int(0)] + BufferParams_get_0(params_buf_1, offset_1 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, offset_1 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, offset_1 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, offset_1 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, offset_1 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, offset_1 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, offset_1 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, offset_1 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, offset_1 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, offset_1 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, offset_1 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, offset_1 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, offset_1 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, offset_1 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, offset_1 + int(15)) * x_1[int(15)];
    int _S22 = offset_1 + int(16);
    y_1[int(1)] = BufferParams_get_0(params_buf_1, _S21 + int(1)) + BufferParams_get_0(params_buf_1, _S22) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S22 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S22 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S22 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S22 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S22 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S22 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S22 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S22 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S22 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S22 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S22 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S22 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S22 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S22 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S22 + int(15)) * x_1[int(15)];
    int _S23 = offset_1 + int(32);
    y_1[int(2)] = BufferParams_get_0(params_buf_1, _S21 + int(2)) + BufferParams_get_0(params_buf_1, _S23) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S23 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S23 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S23 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S23 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S23 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S23 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S23 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S23 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S23 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S23 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S23 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S23 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S23 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S23 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S23 + int(15)) * x_1[int(15)];
    int _S24 = offset_1 + int(48);
    y_1[int(3)] = BufferParams_get_0(params_buf_1, _S21 + int(3)) + BufferParams_get_0(params_buf_1, _S24) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S24 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S24 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S24 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S24 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S24 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S24 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S24 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S24 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S24 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S24 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S24 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S24 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S24 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S24 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S24 + int(15)) * x_1[int(15)];
    int _S25 = offset_1 + int(64);
    y_1[int(4)] = BufferParams_get_0(params_buf_1, _S21 + int(4)) + BufferParams_get_0(params_buf_1, _S25) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S25 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S25 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S25 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S25 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S25 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S25 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S25 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S25 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S25 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S25 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S25 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S25 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S25 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S25 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S25 + int(15)) * x_1[int(15)];
    int _S26 = offset_1 + int(80);
    y_1[int(5)] = BufferParams_get_0(params_buf_1, _S21 + int(5)) + BufferParams_get_0(params_buf_1, _S26) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S26 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S26 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S26 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S26 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S26 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S26 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S26 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S26 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S26 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S26 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S26 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S26 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S26 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S26 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S26 + int(15)) * x_1[int(15)];
    int _S27 = offset_1 + int(96);
    y_1[int(6)] = BufferParams_get_0(params_buf_1, _S21 + int(6)) + BufferParams_get_0(params_buf_1, _S27) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S27 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S27 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S27 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S27 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S27 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S27 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S27 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S27 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S27 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S27 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S27 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S27 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S27 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S27 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S27 + int(15)) * x_1[int(15)];
    int _S28 = offset_1 + int(112);
    y_1[int(7)] = BufferParams_get_0(params_buf_1, _S21 + int(7)) + BufferParams_get_0(params_buf_1, _S28) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S28 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S28 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S28 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S28 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S28 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S28 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S28 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S28 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S28 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S28 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S28 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S28 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S28 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S28 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S28 + int(15)) * x_1[int(15)];
    int _S29 = offset_1 + int(128);
    y_1[int(8)] = BufferParams_get_0(params_buf_1, _S21 + int(8)) + BufferParams_get_0(params_buf_1, _S29) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S29 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S29 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S29 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S29 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S29 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S29 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S29 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S29 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S29 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S29 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S29 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S29 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S29 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S29 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S29 + int(15)) * x_1[int(15)];
    _S20 = y_1;
    return;
}

float leaky_relu_0(float x_2)
{
    float _S30;
    if(x_2 > 0.0f)
    {
        _S30 = x_2;
    }
    else
    {
        _S30 = 0.00999999977648258f * x_2;
    }
    return _S30;
}

void MLP_eval_0(StructuredBuffer<float > params_buf_2, int offset_2, float  x_3[int(2)], out float  _S31[int(9)])
{
    float  _S32[int(16)];
    dense_0(params_buf_2, offset_2, x_3, _S32);
    float  h_1[int(16)] = _S32;
    h_1[int(0)] = leaky_relu_0(h_1[int(0)]);
    h_1[int(1)] = leaky_relu_0(h_1[int(1)]);
    h_1[int(2)] = leaky_relu_0(h_1[int(2)]);
    h_1[int(3)] = leaky_relu_0(h_1[int(3)]);
    h_1[int(4)] = leaky_relu_0(h_1[int(4)]);
    h_1[int(5)] = leaky_relu_0(h_1[int(5)]);
    h_1[int(6)] = leaky_relu_0(h_1[int(6)]);
    h_1[int(7)] = leaky_relu_0(h_1[int(7)]);
    h_1[int(8)] = leaky_relu_0(h_1[int(8)]);
    h_1[int(9)] = leaky_relu_0(h_1[int(9)]);
    h_1[int(10)] = leaky_relu_0(h_1[int(10)]);
    h_1[int(11)] = leaky_relu_0(h_1[int(11)]);
    h_1[int(12)] = leaky_relu_0(h_1[int(12)]);
    h_1[int(13)] = leaky_relu_0(h_1[int(13)]);
    h_1[int(14)] = leaky_relu_0(h_1[int(14)]);
    h_1[int(15)] = leaky_relu_0(h_1[int(15)]);
    float  _S33[int(9)];
    dense_1(params_buf_2, offset_2 + int(32) + int(16), h_1, _S33);
    _S31 = _S33;
    return;
}

void dense_2(StructuredBuffer<float > params_buf_3, int offset_3, float  x_4[int(8)], out float  _S34[int(3)])
{
    int _S35 = offset_3 + int(24);
    float  y_2[int(3)];
    y_2[int(0)] = BufferParams_get_0(params_buf_3, _S35) + BufferParams_get_0(params_buf_3, offset_3) * x_4[int(0)] + BufferParams_get_0(params_buf_3, offset_3 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, offset_3 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, offset_3 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, offset_3 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, offset_3 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, offset_3 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, offset_3 + int(7)) * x_4[int(7)];
    int _S36 = offset_3 + int(8);
    y_2[int(1)] = BufferParams_get_0(params_buf_3, _S35 + int(1)) + BufferParams_get_0(params_buf_3, _S36) * x_4[int(0)] + BufferParams_get_0(params_buf_3, _S36 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, _S36 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, _S36 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, _S36 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, _S36 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, _S36 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, _S36 + int(7)) * x_4[int(7)];
    int _S37 = offset_3 + int(16);
    y_2[int(2)] = BufferParams_get_0(params_buf_3, _S35 + int(2)) + BufferParams_get_0(params_buf_3, _S37) * x_4[int(0)] + BufferParams_get_0(params_buf_3, _S37 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, _S37 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, _S37 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, _S37 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, _S37 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, _S37 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, _S37 + int(7)) * x_4[int(7)];
    _S34 = y_2;
    return;
}

void dense_3(StructuredBuffer<float > params_buf_4, int offset_4, float  x_5[int(3)], out float  _S38[int(8)])
{
    int _S39 = offset_4 + int(24);
    float  y_3[int(8)];
    y_3[int(0)] = BufferParams_get_0(params_buf_4, _S39) + BufferParams_get_0(params_buf_4, offset_4) * x_5[int(0)] + BufferParams_get_0(params_buf_4, offset_4 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, offset_4 + int(2)) * x_5[int(2)];
    int _S40 = offset_4 + int(3);
    y_3[int(1)] = BufferParams_get_0(params_buf_4, _S39 + int(1)) + BufferParams_get_0(params_buf_4, _S40) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S40 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S40 + int(2)) * x_5[int(2)];
    int _S41 = offset_4 + int(6);
    y_3[int(2)] = BufferParams_get_0(params_buf_4, _S39 + int(2)) + BufferParams_get_0(params_buf_4, _S41) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S41 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S41 + int(2)) * x_5[int(2)];
    int _S42 = offset_4 + int(9);
    y_3[int(3)] = BufferParams_get_0(params_buf_4, _S39 + int(3)) + BufferParams_get_0(params_buf_4, _S42) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S42 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S42 + int(2)) * x_5[int(2)];
    int _S43 = offset_4 + int(12);
    y_3[int(4)] = BufferParams_get_0(params_buf_4, _S39 + int(4)) + BufferParams_get_0(params_buf_4, _S43) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S43 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S43 + int(2)) * x_5[int(2)];
    int _S44 = offset_4 + int(15);
    y_3[int(5)] = BufferParams_get_0(params_buf_4, _S39 + int(5)) + BufferParams_get_0(params_buf_4, _S44) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S44 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S44 + int(2)) * x_5[int(2)];
    int _S45 = offset_4 + int(18);
    y_3[int(6)] = BufferParams_get_0(params_buf_4, _S39 + int(6)) + BufferParams_get_0(params_buf_4, _S45) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S45 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S45 + int(2)) * x_5[int(2)];
    int _S46 = offset_4 + int(21);
    y_3[int(7)] = BufferParams_get_0(params_buf_4, _S39 + int(7)) + BufferParams_get_0(params_buf_4, _S46) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S46 + int(1)) * x_5[int(1)] + BufferParams_get_0(params_buf_4, _S46 + int(2)) * x_5[int(2)];
    _S38 = y_3;
    return;
}

void MLP_eval_1(StructuredBuffer<float > params_buf_5, int offset_5, float  x_6[int(3)], out float  _S47[int(3)])
{
    float  _S48[int(8)];
    dense_3(params_buf_5, offset_5, x_6, _S48);
    float  h_2[int(8)] = _S48;
    h_2[int(0)] = leaky_relu_0(h_2[int(0)]);
    h_2[int(1)] = leaky_relu_0(h_2[int(1)]);
    h_2[int(2)] = leaky_relu_0(h_2[int(2)]);
    h_2[int(3)] = leaky_relu_0(h_2[int(3)]);
    h_2[int(4)] = leaky_relu_0(h_2[int(4)]);
    h_2[int(5)] = leaky_relu_0(h_2[int(5)]);
    h_2[int(6)] = leaky_relu_0(h_2[int(6)]);
    h_2[int(7)] = leaky_relu_0(h_2[int(7)]);
    float  _S49[int(3)];
    dense_2(params_buf_5, offset_5 + int(24) + int(8), h_2, _S49);
    _S47 = _S49;
    return;
}

float3 bilinear_zero_from_loads_0(ImageSlice_0 img_1, Texture2DArray<float4 > img_images_tex_1, float2 p_0)
{
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S50 = int2(p0_0);
    float3 _S51 = (float3)f_0.x;
    return lerp(lerp(load_zero_0(img_1, img_images_tex_1, _S50), load_zero_0(img_1, img_images_tex_1, _S50 + int2(int(1), int(0))), _S51), lerp(load_zero_0(img_1, img_images_tex_1, _S50 + int2(int(0), int(1))), load_zero_0(img_1, img_images_tex_1, _S50 + int2(int(1), int(1))), _S51), (float3)f_0.y);
}

float3 TextureArray_bilinear_zero_0(Texture2DArray<float4 > this_tex_2, int slice_2, float2 p_1)
{
    int2 _S52 = TextureArray_size_0(this_tex_2);
    float2 p0_1 = floor(p_1);
    bool _S53;
    if(all(p0_1 >= 0.0f))
    {
        _S53 = all(p0_1 < float2(_S52 - int(1)));
    }
    else
    {
        _S53 = false;
    }
    if(_S53)
    {
        return this_tex_2.SampleLevel(sampler_LinearClamp, float3((p_1 + 0.5f) / float2(_S52), float(slice_2)), 0.0f).xyz;
    }
    if(any(p0_1 < -1.0f))
    {
        _S53 = true;
    }
    else
    {
        _S53 = any(p0_1 > float2(_S52 - int(1)));
    }
    if(_S53)
    {
        return (float3)0.0f;
    }
    ImageSlice_0 _S54;
    _S54.slice_0 = slice_2;
    return bilinear_zero_from_loads_0(_S54, this_tex_2, p_1);
}

float3 ImageSlice_bilinear_zero_0(ImageSlice_0 this_2, Texture2DArray<float4 > this_images_tex_2, float2 p_2)
{
    return TextureArray_bilinear_zero_0(this_images_tex_2, this_2.slice_0, p_2);
}

float3 sample_bilinear_zero_0(ImageSlice_0 img_2, Texture2DArray<float4 > img_images_tex_2, float2 p_3)
{
    return ImageSlice_bilinear_zero_0(img_2, img_images_tex_2, p_3);
}

static const float2  RAY_DIRS_0[int(14)] = { float2(1.0f, 0.0f), float2(0.90096884965896606f, 0.43388372659683228f), float2(0.62348979711532593f, 0.78183150291442871f), float2(0.22252093255519867f, 0.97492790222167969f), float2(-0.22252093255519867f, 0.97492790222167969f), float2(-0.62348979711532593f, 0.78183150291442871f), float2(-0.90096884965896606f, 0.43388372659683228f), float2(-1.0f, 0.0f), float2(-0.90096884965896606f, -0.43388372659683228f), float2(-0.62348979711532593f, -0.78183150291442871f), float2(-0.22252093255519867f, -0.97492790222167969f), float2(0.22252093255519867f, -0.97492790222167969f), float2(0.62348979711532593f, -0.78183150291442871f), float2(0.90096884965896606f, -0.43388372659683228f) };
void softmax_0(float  z_0[int(3)], out float  _S55[int(3)])
{
    float _S56 = max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]);
    float  w_1[int(3)];
    float _S57 = exp(z_0[int(0)] - _S56);
    float _S58 = exp(z_0[int(1)] - _S56);
    float _S59 = exp(z_0[int(2)] - _S56);
    float sum_0 = _S57 + _S58 + _S59;
    w_1[int(0)] = _S57 / sum_0;
    w_1[int(1)] = _S58 / sum_0;
    w_1[int(2)] = _S59 / sum_0;
    _S55 = w_1;
    return;
}

void cross_weights_0(float2 v_0, StructuredBuffer<float > params_buf_6, int offset_6, int ray_0, out float  _S60[int(3)])
{
    float _S61 = dot(RAY_DIRS_0[ray_0], v_0);
    float _S62 = dot(float2(- RAY_DIRS_0[ray_0].y, RAY_DIRS_0[ray_0].x), v_0);
    float _S63;
    if((ray_0 % int(2)) == int(0))
    {
        _S63 = 1.0f;
    }
    else
    {
        _S63 = -1.0f;
    }
    float  x_7[int(3)] = { _S61, _S62, _S63 };
    float  _S64[int(3)];
    MLP_eval_1(params_buf_6, offset_6, x_7, _S64);
    float  _S65[int(3)];
    softmax_0(_S64, _S65);
    _S60 = _S65;
    return;
}

float3 ray_cross_weighted_0(int2 texel_3, ImageSlice_0 src_0, Texture2DArray<float4 > src_images_tex_0, int ray_1, float  a_0[int(3)])
{
    float2 _S66 = float2(- RAY_DIRS_0[ray_1].y, RAY_DIRS_0[ray_1].x);
    float2 _S67 = float2(texel_3);
    return a_0[int(0)] * sample_bilinear_zero_0(src_0, src_images_tex_0, _S67 - -1.0f * _S66) + a_0[int(1)] * sample_bilinear_zero_0(src_0, src_images_tex_0, _S67 - 0.0f * _S66) + a_0[int(2)] * sample_bilinear_zero_0(src_0, src_images_tex_0, _S67 - _S66);
}

void dense_4(StructuredBuffer<float > params_buf_7, int offset_7, float  x_8[int(8)], out float  _S68[int(1)])
{
    float  y_4[int(1)];
    y_4[int(0)] = BufferParams_get_0(params_buf_7, offset_7 + int(8)) + BufferParams_get_0(params_buf_7, offset_7) * x_8[int(0)] + BufferParams_get_0(params_buf_7, offset_7 + int(1)) * x_8[int(1)] + BufferParams_get_0(params_buf_7, offset_7 + int(2)) * x_8[int(2)] + BufferParams_get_0(params_buf_7, offset_7 + int(3)) * x_8[int(3)] + BufferParams_get_0(params_buf_7, offset_7 + int(4)) * x_8[int(4)] + BufferParams_get_0(params_buf_7, offset_7 + int(5)) * x_8[int(5)] + BufferParams_get_0(params_buf_7, offset_7 + int(6)) * x_8[int(6)] + BufferParams_get_0(params_buf_7, offset_7 + int(7)) * x_8[int(7)];
    _S68 = y_4;
    return;
}

void MLP_eval_2(StructuredBuffer<float > params_buf_8, int offset_8, float  x_9[int(3)], out float  _S69[int(1)])
{
    float  _S70[int(8)];
    dense_3(params_buf_8, offset_8, x_9, _S70);
    float  h_3[int(8)] = _S70;
    h_3[int(0)] = leaky_relu_0(h_3[int(0)]);
    h_3[int(1)] = leaky_relu_0(h_3[int(1)]);
    h_3[int(2)] = leaky_relu_0(h_3[int(2)]);
    h_3[int(3)] = leaky_relu_0(h_3[int(3)]);
    h_3[int(4)] = leaky_relu_0(h_3[int(4)]);
    h_3[int(5)] = leaky_relu_0(h_3[int(5)]);
    h_3[int(6)] = leaky_relu_0(h_3[int(6)]);
    h_3[int(7)] = leaky_relu_0(h_3[int(7)]);
    float  _S71[int(1)];
    dense_4(params_buf_8, offset_8 + int(24) + int(8), h_3, _S71);
    _S69 = _S71;
    return;
}

float2 texel_uv_0(int2 texel_4, int2 size_0)
{
    return (float2(texel_4) + 0.5f) / float2(size_0);
}

float2 field_offset_0(float2 uv_0, int2 size_1)
{
    float2 _S72 = float2(size_1);
    return (uv_0 - 0.5f) * _S72 / length(0.5f * _S72);
}

float3 ray_cross_0(int2 texel_5, int2 out_size_0, ImageSlice_0 src_1, Texture2DArray<float4 > src_images_tex_1, StructuredBuffer<float > params_buf_9, int offset_9, int ray_2)
{
    float  _S73[int(3)];
    cross_weights_0(field_offset_0(texel_uv_0(texel_5, out_size_0), out_size_0), params_buf_9, offset_9, ray_2, _S73);
    return ray_cross_weighted_0(texel_5, src_1, src_images_tex_1, ray_2, _S73);
}

float bright_logit_0(float2 v_1, StructuredBuffer<float > params_buf_10, int offset_10, int ray_3)
{
    float _S74 = dot(RAY_DIRS_0[ray_3], v_1);
    float _S75 = dot(float2(- RAY_DIRS_0[ray_3].y, RAY_DIRS_0[ray_3].x), v_1);
    float _S76;
    if((ray_3 % int(2)) == int(0))
    {
        _S76 = 1.0f;
    }
    else
    {
        _S76 = -1.0f;
    }
    float  x_10[int(3)] = { _S74, _S75, _S76 };
    float  _S77[int(1)];
    MLP_eval_2(params_buf_10, offset_10, x_10, _S77);
    return _S77[int(0)];
}

void softmax_rgb_0(float  z_1[int(9)], out float3  _S78[int(3)])
{
    float  zc_0[int(3)];
    zc_0[int(0)] = z_1[int(0)];
    zc_0[int(1)] = z_1[int(1)];
    zc_0[int(2)] = z_1[int(2)];
    float  _S79[int(3)];
    softmax_0(zc_0, _S79);
    float3  w_2[int(3)];
    w_2[int(0)][int(0)] = _S79[int(0)];
    w_2[int(1)][int(0)] = _S79[int(1)];
    w_2[int(2)][int(0)] = _S79[int(2)];
    float  zc_1[int(3)];
    zc_1[int(0)] = z_1[int(3)];
    zc_1[int(1)] = z_1[int(4)];
    zc_1[int(2)] = z_1[int(5)];
    float  _S80[int(3)];
    softmax_0(zc_1, _S80);
    w_2[int(0)][int(1)] = _S80[int(0)];
    w_2[int(1)][int(1)] = _S80[int(1)];
    w_2[int(2)][int(1)] = _S80[int(2)];
    float  zc_2[int(3)];
    zc_2[int(0)] = z_1[int(6)];
    zc_2[int(1)] = z_1[int(7)];
    zc_2[int(2)] = z_1[int(8)];
    float  _S81[int(3)];
    softmax_0(zc_2, _S81);
    w_2[int(0)][int(2)] = _S81[int(0)];
    w_2[int(1)][int(2)] = _S81[int(1)];
    w_2[int(2)][int(2)] = _S81[int(2)];
    _S78 = w_2;
    return;
}

void spike_weights_0(float2 uv_1, StructuredBuffer<float > params_buf_11, int offset_11, out float3  _S82[int(3)])
{
    float  x_11[int(2)] = { uv_1.x, uv_1.y };
    float  _S83[int(9)];
    MLP_eval_0(params_buf_11, offset_11, x_11, _S83);
    float3  _S84[int(3)];
    softmax_rgb_0(_S83, _S84);
    _S82 = _S84;
    return;
}

float4 stacked_ray_0(int2 texel_6, int2 out_size_1, float2 v_2, Texture2DArray<float4 > rays_tex_0, StructuredBuffer<float > params_buf_12, int cross_0, int ray_4)
{
    ImageSlice_0 _S85;
    _S85.slice_0 = ray_4;
    float e_0 = exp(bright_logit_0(v_2, params_buf_12, cross_0 + int(59), ray_4));
    return float4(e_0 * ray_cross_0(texel_6, out_size_1, _S85, rays_tex_0, params_buf_12, cross_0, ray_4), e_0);
}

float3 TextureImage_load_0(Texture2D<float4 > this_tex_3, int2 texel_7)
{
    return this_tex_3.Load(int3(texel_7, int(0))).xyz;
}

float3 spike_composite_stack_0(int2 texel_8, int2 out_size_2, Texture2D<float4 > base_tex_0, Texture2DArray<float4 > rays_tex_1, StructuredBuffer<float > params_buf_13, int offset_12, float scale_0)
{
    float2 uv_2 = texel_uv_0(texel_8, out_size_2);
    float3  _S86[int(3)];
    spike_weights_0(uv_2, params_buf_13, offset_12, _S86);
    float3  _S87[int(3)] = _S86;
    float2 _S88 = field_offset_0(uv_2, out_size_2);
    int _S89 = offset_12 + int(201) + int(24);
    float4 _S90 = (float4)0.0f;
    int j_0 = int(0);
    float4 even_0 = _S90;
    float4 odd_0 = _S90;
    for(;;)
    {
        if(j_0 < int(7))
        {
        }
        else
        {
            break;
        }
        int _S91 = int(2) * j_0;
        float4 even_1 = even_0 + stacked_ray_0(texel_8, out_size_2, _S88, rays_tex_1, params_buf_13, _S89, _S91);
        float4 odd_1 = odd_0 + stacked_ray_0(texel_8, out_size_2, _S88, rays_tex_1, params_buf_13, _S89, _S91 + int(1));
        j_0 = j_0 + int(1);
        even_0 = even_1;
        odd_0 = odd_1;
    }
    return _S87[int(0)] * TextureImage_load_0(base_tex_0, texel_8) + _S87[int(1)] * (scale_0 * even_0.xyz) / even_0.w + _S87[int(2)] * (scale_0 * odd_0.xyz) / odd_0.w;
}

int2 target_size_0()
{
    uint w_3;
    uint h_4;
    _Target.GetDimensions(w_3, h_4);
    return int2(int(w_3), int(h_4));
}

[numthreads(8, 8, 1)]
void SpikeCompositeStack(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_2 = target_size_0();
    int2 _S92 = int2(id_0.xy);
    if(any(_S92 >= size_2))
    {
        return;
    }
    uint2 _S93 = uint2(_S92);
    int _S94 = ((_ParamOffset));
    float _S95 = ((_RayScale));
    _Target[_S93] = float4(spike_composite_stack_0(_S92, size_2, _Source, _RayStack, _GlareParams, _S94, _S95), 1.0f);
    return;
}

