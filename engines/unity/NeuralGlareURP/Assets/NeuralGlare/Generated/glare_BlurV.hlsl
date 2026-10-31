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

float3 sample_bilinear_0(Texture2D<float4 > img_tex_0, float2 uv_0)
{
    int2 _S1 = TextureImage_size_0(img_tex_0);
    float2 p_0 = uv_0 * float2(_S1) - 0.5f;
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S2 = int2(p0_0);
    int2 _S3 = (int2)int(0);
    int2 _S4 = _S1 - int(1);
    int2 i0_0 = clamp(_S2, _S3, _S4);
    int2 i1_0 = clamp(_S2 + int(1), _S3, _S4);
    int _S5 = i0_0.x;
    int _S6 = i0_0.y;
    int _S7 = i1_0.x;
    float3 _S8 = (float3)f_0.x;
    int _S9 = i1_0.y;
    return lerp(lerp(TextureImage_load_0(img_tex_0, int2(_S5, _S6)), TextureImage_load_0(img_tex_0, int2(_S7, _S6)), _S8), lerp(TextureImage_load_0(img_tex_0, int2(_S5, _S9)), TextureImage_load_0(img_tex_0, int2(_S7, _S9)), _S8), (float3)f_0.y);
}

float2 texel_uv_0(int2 texel_1, int2 size_0)
{
    return (float2(texel_1) + 0.5f) / float2(size_0);
}

float3 blur_v_0(int2 texel_2, int2 out_size_0, Texture2D<float4 > src_tex_0)
{
    float2 _S10 = texel_uv_0(texel_2, out_size_0);
    float _S11 = 1.0f / float(TextureImage_size_0(src_tex_0).y);
    return 0.07027027010917664f * sample_bilinear_0(src_tex_0, _S10 + float2(0.0f, -3.23076915740966797f * _S11)) + 0.31621623039245605f * sample_bilinear_0(src_tex_0, _S10 + float2(0.0f, -1.38461542129516602f * _S11)) + 0.22702702879905701f * sample_bilinear_0(src_tex_0, _S10 + float2(0.0f, 0.0f * _S11)) + 0.31621623039245605f * sample_bilinear_0(src_tex_0, _S10 + float2(0.0f, 1.38461542129516602f * _S11)) + 0.07027027010917664f * sample_bilinear_0(src_tex_0, _S10 + float2(0.0f, 3.23076915740966797f * _S11));
}

int2 target_size_0()
{
    uint w_1;
    uint h_1;
    _Target.GetDimensions(w_1, h_1);
    return int2(int(w_1), int(h_1));
}

[numthreads(8, 8, 1)]
void BlurV(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_1 = target_size_0();
    int2 _S12 = int2(id_0.xy);
    if(any(_S12 >= size_1))
    {
        return;
    }
    _Target[uint2(_S12)] = float4(blur_v_0(_S12, size_1, _Source), 1.0f);
    return;
}

