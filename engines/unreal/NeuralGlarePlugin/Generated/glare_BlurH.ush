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

SamplerState sampler_LinearClamp : register(s0);

int2 TextureImage_size_0(Texture2D<float4 > this_tex_0)
{
    uint w_0;
    uint h_0;
    this_tex_0.GetDimensions(w_0, h_0);
    return int2(int(w_0), int(h_0));
}

float3 TextureImage_bilinear_0(Texture2D<float4 > this_tex_1, float2 p_0)
{
    return this_tex_1.SampleLevel(sampler_LinearClamp, (p_0 + 0.5f) / float2(TextureImage_size_0(this_tex_1)), 0.0f).xyz;
}

float3 sample_bilinear_0(Texture2D<float4 > img_tex_0, float2 uv_0)
{
    return TextureImage_bilinear_0(img_tex_0, uv_0 * float2(TextureImage_size_0(img_tex_0)) - 0.5f);
}

float2 texel_uv_0(int2 texel_0, int2 size_0)
{
    return (float2(texel_0) + 0.5f) / float2(size_0);
}

float3 blur_h_0(int2 texel_1, int2 out_size_0, Texture2D<float4 > src_tex_0)
{
    float2 _S1 = texel_uv_0(texel_1, out_size_0);
    float _S2 = 1.0f / float(TextureImage_size_0(src_tex_0).x);
    return 0.01621622033417225f * sample_bilinear_0(src_tex_0, _S1 + float2(-8.0f * _S2, 0.0f)) + 0.05405405163764954f * sample_bilinear_0(src_tex_0, _S1 + float2(-6.0f * _S2, 0.0f)) + 0.12162162363529205f * sample_bilinear_0(src_tex_0, _S1 + float2(-4.0f * _S2, 0.0f)) + 0.19459459185600281f * sample_bilinear_0(src_tex_0, _S1 + float2(-2.0f * _S2, 0.0f)) + 0.22702702879905701f * sample_bilinear_0(src_tex_0, _S1 + float2(0.0f * _S2, 0.0f)) + 0.19459459185600281f * sample_bilinear_0(src_tex_0, _S1 + float2(2.0f * _S2, 0.0f)) + 0.12162162363529205f * sample_bilinear_0(src_tex_0, _S1 + float2(4.0f * _S2, 0.0f)) + 0.05405405163764954f * sample_bilinear_0(src_tex_0, _S1 + float2(6.0f * _S2, 0.0f)) + 0.01621622033417225f * sample_bilinear_0(src_tex_0, _S1 + float2(8.0f * _S2, 0.0f));
}

int2 target_size_0()
{
    uint w_1;
    uint h_1;
    _Target.GetDimensions(w_1, h_1);
    return int2(int(w_1), int(h_1));
}

[numthreads(8, 8, 1)]
void BlurH(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_1 = target_size_0();
    int2 _S3 = int2(id_0.xy);
    if(any(_S3 >= size_1))
    {
        return;
    }
    _Target[uint2(_S3)] = float4(blur_h_0(_S3, size_1, _Source), 1.0f);
    return;
}

