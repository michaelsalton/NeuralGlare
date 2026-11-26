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

Texture2D<float4 > _Accum : register(t2);

StructuredBuffer<float > _GlareParams : register(t6);

SamplerState sampler_LinearClamp : register(s0);

int2 TextureImage_size_0(Texture2D<float4 > this_tex_0)
{
    uint w_0;
    uint h_0;
    this_tex_0.GetDimensions(w_0, h_0);
    return int2(int(w_0), int(h_0));
}

float3 TextureImage_load_0(Texture2D<float4 > this_tex_1, int2 texel_0)
{
    return this_tex_1.Load(int3(texel_0, int(0))).xyz;
}

float3 load_zero_0(Texture2D<float4 > img_tex_0, int2 texel_1)
{
    int2 _S1 = TextureImage_size_0(img_tex_0);
    bool _S2;
    if(any(texel_1 < ((int2)int(0))))
    {
        _S2 = true;
    }
    else
    {
        _S2 = any(texel_1 >= _S1);
    }
    if(_S2)
    {
        return (float3)0.0f;
    }
    return TextureImage_load_0(img_tex_0, texel_1);
}

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void dense_0(StructuredBuffer<float > params_buf_0, int offset_0, float  x_0[int(8)], out float  _S3[int(1)])
{
    float  y_0[int(1)];
    y_0[int(0)] = BufferParams_get_0(params_buf_0, offset_0 + int(8)) + BufferParams_get_0(params_buf_0, offset_0) * x_0[int(0)] + BufferParams_get_0(params_buf_0, offset_0 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, offset_0 + int(2)) * x_0[int(2)] + BufferParams_get_0(params_buf_0, offset_0 + int(3)) * x_0[int(3)] + BufferParams_get_0(params_buf_0, offset_0 + int(4)) * x_0[int(4)] + BufferParams_get_0(params_buf_0, offset_0 + int(5)) * x_0[int(5)] + BufferParams_get_0(params_buf_0, offset_0 + int(6)) * x_0[int(6)] + BufferParams_get_0(params_buf_0, offset_0 + int(7)) * x_0[int(7)];
    _S3 = y_0;
    return;
}

void dense_1(StructuredBuffer<float > params_buf_1, int offset_1, float  x_1[int(3)], out float  _S4[int(8)])
{
    int _S5 = offset_1 + int(24);
    float  y_1[int(8)];
    y_1[int(0)] = BufferParams_get_0(params_buf_1, _S5) + BufferParams_get_0(params_buf_1, offset_1) * x_1[int(0)] + BufferParams_get_0(params_buf_1, offset_1 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, offset_1 + int(2)) * x_1[int(2)];
    int _S6 = offset_1 + int(3);
    y_1[int(1)] = BufferParams_get_0(params_buf_1, _S5 + int(1)) + BufferParams_get_0(params_buf_1, _S6) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S6 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S6 + int(2)) * x_1[int(2)];
    int _S7 = offset_1 + int(6);
    y_1[int(2)] = BufferParams_get_0(params_buf_1, _S5 + int(2)) + BufferParams_get_0(params_buf_1, _S7) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S7 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S7 + int(2)) * x_1[int(2)];
    int _S8 = offset_1 + int(9);
    y_1[int(3)] = BufferParams_get_0(params_buf_1, _S5 + int(3)) + BufferParams_get_0(params_buf_1, _S8) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S8 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S8 + int(2)) * x_1[int(2)];
    int _S9 = offset_1 + int(12);
    y_1[int(4)] = BufferParams_get_0(params_buf_1, _S5 + int(4)) + BufferParams_get_0(params_buf_1, _S9) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S9 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S9 + int(2)) * x_1[int(2)];
    int _S10 = offset_1 + int(15);
    y_1[int(5)] = BufferParams_get_0(params_buf_1, _S5 + int(5)) + BufferParams_get_0(params_buf_1, _S10) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S10 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S10 + int(2)) * x_1[int(2)];
    int _S11 = offset_1 + int(18);
    y_1[int(6)] = BufferParams_get_0(params_buf_1, _S5 + int(6)) + BufferParams_get_0(params_buf_1, _S11) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S11 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S11 + int(2)) * x_1[int(2)];
    int _S12 = offset_1 + int(21);
    y_1[int(7)] = BufferParams_get_0(params_buf_1, _S5 + int(7)) + BufferParams_get_0(params_buf_1, _S12) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S12 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S12 + int(2)) * x_1[int(2)];
    _S4 = y_1;
    return;
}

float leaky_relu_0(float x_2)
{
    float _S13;
    if(x_2 > 0.0f)
    {
        _S13 = x_2;
    }
    else
    {
        _S13 = 0.00999999977648258f * x_2;
    }
    return _S13;
}

void MLP_eval_0(StructuredBuffer<float > params_buf_2, int offset_2, float  x_3[int(3)], out float  _S14[int(1)])
{
    float  _S15[int(8)];
    dense_1(params_buf_2, offset_2, x_3, _S15);
    float  h_1[int(8)] = _S15;
    h_1[int(0)] = leaky_relu_0(h_1[int(0)]);
    h_1[int(1)] = leaky_relu_0(h_1[int(1)]);
    h_1[int(2)] = leaky_relu_0(h_1[int(2)]);
    h_1[int(3)] = leaky_relu_0(h_1[int(3)]);
    h_1[int(4)] = leaky_relu_0(h_1[int(4)]);
    h_1[int(5)] = leaky_relu_0(h_1[int(5)]);
    h_1[int(6)] = leaky_relu_0(h_1[int(6)]);
    h_1[int(7)] = leaky_relu_0(h_1[int(7)]);
    float  _S16[int(1)];
    dense_0(params_buf_2, offset_2 + int(24) + int(8), h_1, _S16);
    _S14 = _S16;
    return;
}

void dense_2(StructuredBuffer<float > params_buf_3, int offset_3, float  x_4[int(8)], out float  _S17[int(3)])
{
    int _S18 = offset_3 + int(24);
    float  y_2[int(3)];
    y_2[int(0)] = BufferParams_get_0(params_buf_3, _S18) + BufferParams_get_0(params_buf_3, offset_3) * x_4[int(0)] + BufferParams_get_0(params_buf_3, offset_3 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, offset_3 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, offset_3 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, offset_3 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, offset_3 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, offset_3 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, offset_3 + int(7)) * x_4[int(7)];
    int _S19 = offset_3 + int(8);
    y_2[int(1)] = BufferParams_get_0(params_buf_3, _S18 + int(1)) + BufferParams_get_0(params_buf_3, _S19) * x_4[int(0)] + BufferParams_get_0(params_buf_3, _S19 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, _S19 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, _S19 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, _S19 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, _S19 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, _S19 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, _S19 + int(7)) * x_4[int(7)];
    int _S20 = offset_3 + int(16);
    y_2[int(2)] = BufferParams_get_0(params_buf_3, _S18 + int(2)) + BufferParams_get_0(params_buf_3, _S20) * x_4[int(0)] + BufferParams_get_0(params_buf_3, _S20 + int(1)) * x_4[int(1)] + BufferParams_get_0(params_buf_3, _S20 + int(2)) * x_4[int(2)] + BufferParams_get_0(params_buf_3, _S20 + int(3)) * x_4[int(3)] + BufferParams_get_0(params_buf_3, _S20 + int(4)) * x_4[int(4)] + BufferParams_get_0(params_buf_3, _S20 + int(5)) * x_4[int(5)] + BufferParams_get_0(params_buf_3, _S20 + int(6)) * x_4[int(6)] + BufferParams_get_0(params_buf_3, _S20 + int(7)) * x_4[int(7)];
    _S17 = y_2;
    return;
}

void MLP_eval_1(StructuredBuffer<float > params_buf_4, int offset_4, float  x_5[int(3)], out float  _S21[int(3)])
{
    float  _S22[int(8)];
    dense_1(params_buf_4, offset_4, x_5, _S22);
    float  h_2[int(8)] = _S22;
    h_2[int(0)] = leaky_relu_0(h_2[int(0)]);
    h_2[int(1)] = leaky_relu_0(h_2[int(1)]);
    h_2[int(2)] = leaky_relu_0(h_2[int(2)]);
    h_2[int(3)] = leaky_relu_0(h_2[int(3)]);
    h_2[int(4)] = leaky_relu_0(h_2[int(4)]);
    h_2[int(5)] = leaky_relu_0(h_2[int(5)]);
    h_2[int(6)] = leaky_relu_0(h_2[int(6)]);
    h_2[int(7)] = leaky_relu_0(h_2[int(7)]);
    float  _S23[int(3)];
    dense_2(params_buf_4, offset_4 + int(24) + int(8), h_2, _S23);
    _S21 = _S23;
    return;
}

float3 bilinear_zero_from_loads_0(Texture2D<float4 > img_tex_1, float2 p_0)
{
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S24 = int2(p0_0);
    float3 _S25 = (float3)f_0.x;
    return lerp(lerp(load_zero_0(img_tex_1, _S24), load_zero_0(img_tex_1, _S24 + int2(int(1), int(0))), _S25), lerp(load_zero_0(img_tex_1, _S24 + int2(int(0), int(1))), load_zero_0(img_tex_1, _S24 + int2(int(1), int(1))), _S25), (float3)f_0.y);
}

float3 TextureImage_bilinear_zero_0(Texture2D<float4 > this_tex_2, float2 p_1)
{
    int2 _S26 = TextureImage_size_0(this_tex_2);
    float2 p0_1 = floor(p_1);
    bool _S27;
    if(all(p0_1 >= 0.0f))
    {
        _S27 = all(p0_1 < float2(_S26 - int(1)));
    }
    else
    {
        _S27 = false;
    }
    if(_S27)
    {
        return this_tex_2.SampleLevel(sampler_LinearClamp, (p_1 + 0.5f) / float2(_S26), 0.0f).xyz;
    }
    if(any(p0_1 < -1.0f))
    {
        _S27 = true;
    }
    else
    {
        _S27 = any(p0_1 > float2(_S26 - int(1)));
    }
    if(_S27)
    {
        return (float3)0.0f;
    }
    return bilinear_zero_from_loads_0(this_tex_2, p_1);
}

float3 sample_bilinear_zero_0(Texture2D<float4 > img_tex_2, float2 p_2)
{
    return TextureImage_bilinear_zero_0(img_tex_2, p_2);
}

static const float2  RAY_DIRS_0[int(14)] = { float2(1.0f, 0.0f), float2(0.90096884965896606f, 0.43388372659683228f), float2(0.62348979711532593f, 0.78183150291442871f), float2(0.22252093255519867f, 0.97492790222167969f), float2(-0.22252093255519867f, 0.97492790222167969f), float2(-0.62348979711532593f, 0.78183150291442871f), float2(-0.90096884965896606f, 0.43388372659683228f), float2(-1.0f, 0.0f), float2(-0.90096884965896606f, -0.43388372659683228f), float2(-0.62348979711532593f, -0.78183150291442871f), float2(-0.22252093255519867f, -0.97492790222167969f), float2(0.22252093255519867f, -0.97492790222167969f), float2(0.62348979711532593f, -0.78183150291442871f), float2(0.90096884965896606f, -0.43388372659683228f) };
void softmax_0(float  z_0[int(3)], out float  _S28[int(3)])
{
    float _S29 = max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]);
    float  w_1[int(3)];
    float _S30 = exp(z_0[int(0)] - _S29);
    float _S31 = exp(z_0[int(1)] - _S29);
    float _S32 = exp(z_0[int(2)] - _S29);
    float sum_0 = _S30 + _S31 + _S32;
    w_1[int(0)] = _S30 / sum_0;
    w_1[int(1)] = _S31 / sum_0;
    w_1[int(2)] = _S32 / sum_0;
    _S28 = w_1;
    return;
}

void cross_weights_0(float2 v_0, StructuredBuffer<float > params_buf_5, int offset_5, int ray_0, out float  _S33[int(3)])
{
    float _S34 = dot(RAY_DIRS_0[ray_0], v_0);
    float _S35 = dot(float2(- RAY_DIRS_0[ray_0].y, RAY_DIRS_0[ray_0].x), v_0);
    float _S36;
    if((ray_0 % int(2)) == int(0))
    {
        _S36 = 1.0f;
    }
    else
    {
        _S36 = -1.0f;
    }
    float  x_6[int(3)] = { _S34, _S35, _S36 };
    float  _S37[int(3)];
    MLP_eval_1(params_buf_5, offset_5, x_6, _S37);
    float  _S38[int(3)];
    softmax_0(_S37, _S38);
    _S33 = _S38;
    return;
}

float3 ray_cross_weighted_0(int2 texel_2, Texture2D<float4 > src_tex_0, int ray_1, float  a_0[int(3)])
{
    float2 _S39 = float2(- RAY_DIRS_0[ray_1].y, RAY_DIRS_0[ray_1].x);
    float2 _S40 = float2(texel_2);
    return a_0[int(0)] * sample_bilinear_zero_0(src_tex_0, _S40 - -1.0f * _S39) + a_0[int(1)] * sample_bilinear_zero_0(src_tex_0, _S40 - 0.0f * _S39) + a_0[int(2)] * sample_bilinear_zero_0(src_tex_0, _S40 - _S39);
}

float bright_logit_0(float2 v_1, StructuredBuffer<float > params_buf_6, int offset_6, int ray_2)
{
    float _S41 = dot(RAY_DIRS_0[ray_2], v_1);
    float _S42 = dot(float2(- RAY_DIRS_0[ray_2].y, RAY_DIRS_0[ray_2].x), v_1);
    float _S43;
    if((ray_2 % int(2)) == int(0))
    {
        _S43 = 1.0f;
    }
    else
    {
        _S43 = -1.0f;
    }
    float  x_7[int(3)] = { _S41, _S42, _S43 };
    float  _S44[int(1)];
    MLP_eval_0(params_buf_6, offset_6, x_7, _S44);
    return _S44[int(0)];
}

float2 texel_uv_0(int2 texel_3, int2 size_0)
{
    return (float2(texel_3) + 0.5f) / float2(size_0);
}

float2 field_offset_0(float2 uv_0, int2 size_1)
{
    float2 _S45 = float2(size_1);
    return (uv_0 - 0.5f) * _S45 / length(0.5f * _S45);
}

float3 ray_cross_0(int2 texel_4, int2 out_size_0, Texture2D<float4 > src_tex_1, StructuredBuffer<float > params_buf_7, int offset_7, int ray_3)
{
    float  _S46[int(3)];
    cross_weights_0(field_offset_0(texel_uv_0(texel_4, out_size_0), out_size_0), params_buf_7, offset_7, ray_3, _S46);
    return ray_cross_weighted_0(texel_4, src_tex_1, ray_3, _S46);
}

float3 ray_bright_accumulate_0(int2 texel_5, int2 out_size_1, Texture2D<float4 > src_tex_2, Texture2D<float4 > acc_tex_0, StructuredBuffer<float > params_buf_8, int offset_8, int ray_4, float scale_0)
{
    return TextureImage_load_0(acc_tex_0, texel_5) + scale_0 * exp(bright_logit_0(field_offset_0(texel_uv_0(texel_5, out_size_1), out_size_1), params_buf_8, offset_8 + int(59), ray_4)) * ray_cross_0(texel_5, out_size_1, src_tex_2, params_buf_8, offset_8, ray_4);
}

int2 target_size_0()
{
    uint w_2;
    uint h_3;
    _Target.GetDimensions(w_2, h_3);
    return int2(int(w_2), int(h_3));
}

[numthreads(8, 8, 1)]
void RayBrightAccumulate(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_2 = target_size_0();
    int2 _S47 = int2(id_0.xy);
    if(any(_S47 >= size_2))
    {
        return;
    }
    uint2 _S48 = uint2(_S47);
    int _S49 = ((_ParamOffset));
    int _S50 = ((_Ray));
    float _S51 = ((_RayScale));
    _Target[_S48] = float4(ray_bright_accumulate_0(_S47, size_2, _Source, _Accum, _GlareParams, _S49, _S50, _S51), 1.0f);
    return;
}

