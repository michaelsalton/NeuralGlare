#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif

#ifndef __DXC_VERSION_MAJOR
// warning X3557: loop doesn't seem to do anything, forcing loop to unroll
#pragma warning(disable : 3557)
#endif

RWTexture2DArray<float4 > _RayStackOut : register(u1);

Texture2D<float4 > _Source : register(t0);

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

float3 bilinear_zero_from_loads_0(Texture2D<float4 > img_tex_1, float2 p_0)
{
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S3 = int2(p0_0);
    float3 _S4 = (float3)f_0.x;
    return lerp(lerp(load_zero_0(img_tex_1, _S3), load_zero_0(img_tex_1, _S3 + int2(int(1), int(0))), _S4), lerp(load_zero_0(img_tex_1, _S3 + int2(int(0), int(1))), load_zero_0(img_tex_1, _S3 + int2(int(1), int(1))), _S4), (float3)f_0.y);
}

float3 TextureImage_bilinear_zero_0(Texture2D<float4 > this_tex_2, float2 p_1)
{
    int2 _S5 = TextureImage_size_0(this_tex_2);
    float2 p0_1 = floor(p_1);
    bool _S6;
    if(all(p0_1 >= 0.0f))
    {
        _S6 = all(p0_1 < float2(_S5 - int(1)));
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        return this_tex_2.SampleLevel(sampler_LinearClamp, (p_1 + 0.5f) / float2(_S5), 0.0f).xyz;
    }
    if(any(p0_1 < -1.0f))
    {
        _S6 = true;
    }
    else
    {
        _S6 = any(p0_1 > float2(_S5 - int(1)));
    }
    if(_S6)
    {
        return (float3)0.0f;
    }
    return bilinear_zero_from_loads_0(this_tex_2, p_1);
}

float3 sample_bilinear_zero_0(Texture2D<float4 > img_tex_2, float2 p_2)
{
    return TextureImage_bilinear_zero_0(img_tex_2, p_2);
}

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void softmax_0(float  z_0[int(4)], out float  _S7[int(4)])
{
    float _S8 = max(max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]), z_0[int(3)]);
    float  w_1[int(4)];
    float _S9 = exp(z_0[int(0)] - _S8);
    float _S10 = exp(z_0[int(1)] - _S8);
    float _S11 = exp(z_0[int(2)] - _S8);
    float _S12 = exp(z_0[int(3)] - _S8);
    float sum_0 = _S9 + _S10 + _S11 + _S12;
    w_1[int(0)] = _S9 / sum_0;
    w_1[int(1)] = _S10 / sum_0;
    w_1[int(2)] = _S11 / sum_0;
    w_1[int(3)] = _S12 / sum_0;
    _S7 = w_1;
    return;
}

static const float2  RAY_DIRS_0[int(14)] = { float2(1.0f, 0.0f), float2(0.90096884965896606f, 0.43388372659683228f), float2(0.62348979711532593f, 0.78183150291442871f), float2(0.22252093255519867f, 0.97492790222167969f), float2(-0.22252093255519867f, 0.97492790222167969f), float2(-0.62348979711532593f, 0.78183150291442871f), float2(-0.90096884965896606f, 0.43388372659683228f), float2(-1.0f, 0.0f), float2(-0.90096884965896606f, -0.43388372659683228f), float2(-0.62348979711532593f, -0.78183150291442871f), float2(-0.22252093255519867f, -0.97492790222167969f), float2(0.22252093255519867f, -0.97492790222167969f), float2(0.62348979711532593f, -0.78183150291442871f), float2(0.90096884965896606f, -0.43388372659683228f) };
float3 streak_at_scale_0(int2 texel_2, Texture2D<float4 > src_tex_0, StructuredBuffer<float > params_buf_0, int profile_0, int ray_0, int stage_0, float scale_0, float texel_step_0)
{
    int _S13 = profile_0 + (ray_0 % int(2) * int(3) + stage_0) * int(4);
    float  z_1[int(4)];
    z_1[int(0)] = BufferParams_get_0(params_buf_0, _S13);
    z_1[int(1)] = BufferParams_get_0(params_buf_0, _S13 + int(1));
    z_1[int(2)] = BufferParams_get_0(params_buf_0, _S13 + int(2));
    z_1[int(3)] = BufferParams_get_0(params_buf_0, _S13 + int(3));
    float  _S14[int(4)];
    softmax_0(z_1, _S14);
    float2 _S15 = RAY_DIRS_0[ray_0] * (float(int(1) << (int(2) * stage_0)) * texel_step_0);
    float2 _S16 = float2(texel_2);
    return scale_0 * (_S14[int(0)] * sample_bilinear_zero_0(src_tex_0, _S16 - 0.0f * _S15) + _S14[int(1)] * sample_bilinear_zero_0(src_tex_0, _S16 - _S15) + _S14[int(2)] * sample_bilinear_zero_0(src_tex_0, _S16 - 2.0f * _S15) + _S14[int(3)] * sample_bilinear_zero_0(src_tex_0, _S16 - 3.0f * _S15));
}

float3 streak_0(int2 texel_3, Texture2D<float4 > src_tex_1, StructuredBuffer<float > params_buf_1, int profile_1, int ray_1, int stage_1, float scale_1)
{
    return streak_at_scale_0(texel_3, src_tex_1, params_buf_1, profile_1, ray_1, stage_1, scale_1, 1.0f);
}

[numthreads(8, 8, 1)]
void StreakToStack(uint3 id_0 : SV_DispatchThreadID)
{
    uint w_2;
    uint h_1;
    uint n_0;
    _RayStackOut.GetDimensions(w_2, h_1, n_0);
    int2 _S17 = int2(id_0.xy);
    if(any(_S17 >= int2(int(w_2), int(h_1))))
    {
        return;
    }
    uint2 _S18 = id_0.xy;
    int _S19 = ((_Ray));
    uint3 _S20 = uint3(_S18, uint(_S19));
    int _S21 = ((_ParamOffset));
    int _S22 = ((_Ray));
    int _S23 = ((_Stage));
    float _S24 = ((_RayScale));
    _RayStackOut[_S20] = float4(streak_0(_S17, _Source, _GlareParams, _S21, _S22, _S23, _S24), 1.0f);
    return;
}

