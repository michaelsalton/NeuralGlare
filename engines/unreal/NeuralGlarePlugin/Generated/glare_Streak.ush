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

StructuredBuffer<float > _GlareParams : register(t5);

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

float3 sample_bilinear_zero_0(Texture2D<float4 > img_tex_1, float2 p_0)
{
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S3 = int2(p0_0);
    float3 _S4 = (float3)f_0.x;
    return lerp(lerp(load_zero_0(img_tex_1, _S3), load_zero_0(img_tex_1, _S3 + int2(int(1), int(0))), _S4), lerp(load_zero_0(img_tex_1, _S3 + int2(int(0), int(1))), load_zero_0(img_tex_1, _S3 + int2(int(1), int(1))), _S4), (float3)f_0.y);
}

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void softmax_0(float  z_0[int(4)], out float  _S5[int(4)])
{
    float _S6 = max(max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]), z_0[int(3)]);
    float  w_1[int(4)];
    float _S7 = exp(z_0[int(0)] - _S6);
    float _S8 = exp(z_0[int(1)] - _S6);
    float _S9 = exp(z_0[int(2)] - _S6);
    float _S10 = exp(z_0[int(3)] - _S6);
    float sum_0 = _S7 + _S8 + _S9 + _S10;
    w_1[int(0)] = _S7 / sum_0;
    w_1[int(1)] = _S8 / sum_0;
    w_1[int(2)] = _S9 / sum_0;
    w_1[int(3)] = _S10 / sum_0;
    _S5 = w_1;
    return;
}

static const float2  RAY_DIRS_0[int(14)] = { float2(1.0f, 0.0f), float2(0.90096884965896606f, 0.43388372659683228f), float2(0.62348979711532593f, 0.78183150291442871f), float2(0.22252093255519867f, 0.97492790222167969f), float2(-0.22252093255519867f, 0.97492790222167969f), float2(-0.62348979711532593f, 0.78183150291442871f), float2(-0.90096884965896606f, 0.43388372659683228f), float2(-1.0f, 0.0f), float2(-0.90096884965896606f, -0.43388372659683228f), float2(-0.62348979711532593f, -0.78183150291442871f), float2(-0.22252093255519867f, -0.97492790222167969f), float2(0.22252093255519867f, -0.97492790222167969f), float2(0.62348979711532593f, -0.78183150291442871f), float2(0.90096884965896606f, -0.43388372659683228f) };
float3 streak_0(int2 texel_2, Texture2D<float4 > src_tex_0, StructuredBuffer<float > params_buf_0, int profile_0, int ray_0, int stage_0)
{
    int _S11 = profile_0 + (ray_0 % int(2) * int(3) + stage_0) * int(4);
    float  z_1[int(4)];
    z_1[int(0)] = BufferParams_get_0(params_buf_0, _S11);
    z_1[int(1)] = BufferParams_get_0(params_buf_0, _S11 + int(1));
    z_1[int(2)] = BufferParams_get_0(params_buf_0, _S11 + int(2));
    z_1[int(3)] = BufferParams_get_0(params_buf_0, _S11 + int(3));
    float  _S12[int(4)];
    softmax_0(z_1, _S12);
    float2 _S13 = RAY_DIRS_0[ray_0] * float(int(1) << (int(2) * stage_0));
    float2 _S14 = float2(texel_2);
    return _S12[int(0)] * sample_bilinear_zero_0(src_tex_0, _S14 - 0.0f * _S13) + _S12[int(1)] * sample_bilinear_zero_0(src_tex_0, _S14 - _S13) + _S12[int(2)] * sample_bilinear_zero_0(src_tex_0, _S14 - 2.0f * _S13) + _S12[int(3)] * sample_bilinear_zero_0(src_tex_0, _S14 - 3.0f * _S13);
}

int2 target_size_0()
{
    uint w_2;
    uint h_1;
    _Target.GetDimensions(w_2, h_1);
    return int2(int(w_2), int(h_1));
}

[numthreads(8, 8, 1)]
void Streak(uint3 id_0 : SV_DispatchThreadID)
{
    int2 _S15 = int2(id_0.xy);
    if(any(_S15 >= (target_size_0())))
    {
        return;
    }
    uint2 _S16 = uint2(_S15);
    int _S17 = ((_ParamOffset));
    int _S18 = ((_Ray));
    int _S19 = ((_Stage));
    _Target[_S16] = float4(streak_0(_S15, _Source, _GlareParams, _S17, _S18, _S19), 1.0f);
    return;
}

