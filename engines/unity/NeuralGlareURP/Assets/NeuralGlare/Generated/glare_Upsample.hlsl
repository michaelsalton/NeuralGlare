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

Texture2D<float4 > _SourceLow : register(t1);

StructuredBuffer<float > _GlareParams : register(t2);

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void dense_0(StructuredBuffer<float > params_buf_0, int offset_0, float  x_0[int(2)], out float  _S1[int(16)])
{
    int _S2 = offset_0 + int(32);
    float  y_0[int(16)];
    y_0[int(0)] = BufferParams_get_0(params_buf_0, _S2) + BufferParams_get_0(params_buf_0, offset_0) * x_0[int(0)] + BufferParams_get_0(params_buf_0, offset_0 + int(1)) * x_0[int(1)];
    int _S3 = offset_0 + int(2);
    y_0[int(1)] = BufferParams_get_0(params_buf_0, _S2 + int(1)) + BufferParams_get_0(params_buf_0, _S3) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S3 + int(1)) * x_0[int(1)];
    int _S4 = offset_0 + int(4);
    y_0[int(2)] = BufferParams_get_0(params_buf_0, _S2 + int(2)) + BufferParams_get_0(params_buf_0, _S4) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S4 + int(1)) * x_0[int(1)];
    int _S5 = offset_0 + int(6);
    y_0[int(3)] = BufferParams_get_0(params_buf_0, _S2 + int(3)) + BufferParams_get_0(params_buf_0, _S5) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S5 + int(1)) * x_0[int(1)];
    int _S6 = offset_0 + int(8);
    y_0[int(4)] = BufferParams_get_0(params_buf_0, _S2 + int(4)) + BufferParams_get_0(params_buf_0, _S6) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S6 + int(1)) * x_0[int(1)];
    int _S7 = offset_0 + int(10);
    y_0[int(5)] = BufferParams_get_0(params_buf_0, _S2 + int(5)) + BufferParams_get_0(params_buf_0, _S7) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S7 + int(1)) * x_0[int(1)];
    int _S8 = offset_0 + int(12);
    y_0[int(6)] = BufferParams_get_0(params_buf_0, _S2 + int(6)) + BufferParams_get_0(params_buf_0, _S8) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S8 + int(1)) * x_0[int(1)];
    int _S9 = offset_0 + int(14);
    y_0[int(7)] = BufferParams_get_0(params_buf_0, _S2 + int(7)) + BufferParams_get_0(params_buf_0, _S9) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S9 + int(1)) * x_0[int(1)];
    int _S10 = offset_0 + int(16);
    y_0[int(8)] = BufferParams_get_0(params_buf_0, _S2 + int(8)) + BufferParams_get_0(params_buf_0, _S10) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S10 + int(1)) * x_0[int(1)];
    int _S11 = offset_0 + int(18);
    y_0[int(9)] = BufferParams_get_0(params_buf_0, _S2 + int(9)) + BufferParams_get_0(params_buf_0, _S11) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S11 + int(1)) * x_0[int(1)];
    int _S12 = offset_0 + int(20);
    y_0[int(10)] = BufferParams_get_0(params_buf_0, _S2 + int(10)) + BufferParams_get_0(params_buf_0, _S12) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S12 + int(1)) * x_0[int(1)];
    int _S13 = offset_0 + int(22);
    y_0[int(11)] = BufferParams_get_0(params_buf_0, _S2 + int(11)) + BufferParams_get_0(params_buf_0, _S13) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S13 + int(1)) * x_0[int(1)];
    int _S14 = offset_0 + int(24);
    y_0[int(12)] = BufferParams_get_0(params_buf_0, _S2 + int(12)) + BufferParams_get_0(params_buf_0, _S14) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S14 + int(1)) * x_0[int(1)];
    int _S15 = offset_0 + int(26);
    y_0[int(13)] = BufferParams_get_0(params_buf_0, _S2 + int(13)) + BufferParams_get_0(params_buf_0, _S15) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S15 + int(1)) * x_0[int(1)];
    int _S16 = offset_0 + int(28);
    y_0[int(14)] = BufferParams_get_0(params_buf_0, _S2 + int(14)) + BufferParams_get_0(params_buf_0, _S16) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S16 + int(1)) * x_0[int(1)];
    int _S17 = offset_0 + int(30);
    y_0[int(15)] = BufferParams_get_0(params_buf_0, _S2 + int(15)) + BufferParams_get_0(params_buf_0, _S17) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S17 + int(1)) * x_0[int(1)];
    _S1 = y_0;
    return;
}

void dense_1(StructuredBuffer<float > params_buf_1, int offset_1, float  x_1[int(16)], out float  _S18[int(2)])
{
    int _S19 = offset_1 + int(32);
    float  y_1[int(2)];
    y_1[int(0)] = BufferParams_get_0(params_buf_1, _S19) + BufferParams_get_0(params_buf_1, offset_1) * x_1[int(0)] + BufferParams_get_0(params_buf_1, offset_1 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, offset_1 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, offset_1 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, offset_1 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, offset_1 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, offset_1 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, offset_1 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, offset_1 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, offset_1 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, offset_1 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, offset_1 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, offset_1 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, offset_1 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, offset_1 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, offset_1 + int(15)) * x_1[int(15)];
    int _S20 = offset_1 + int(16);
    y_1[int(1)] = BufferParams_get_0(params_buf_1, _S19 + int(1)) + BufferParams_get_0(params_buf_1, _S20) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S20 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S20 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S20 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S20 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S20 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S20 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S20 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S20 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S20 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S20 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S20 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S20 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S20 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S20 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S20 + int(15)) * x_1[int(15)];
    _S18 = y_1;
    return;
}

float leaky_relu_0(float x_2)
{
    float _S21;
    if(x_2 > 0.0f)
    {
        _S21 = x_2;
    }
    else
    {
        _S21 = 0.00999999977648258f * x_2;
    }
    return _S21;
}

void MLP_eval_0(StructuredBuffer<float > params_buf_2, int offset_2, float  x_3[int(2)], out float  _S22[int(2)])
{
    float  _S23[int(16)];
    dense_0(params_buf_2, offset_2, x_3, _S23);
    float  h_0[int(16)] = _S23;
    h_0[int(0)] = leaky_relu_0(h_0[int(0)]);
    h_0[int(1)] = leaky_relu_0(h_0[int(1)]);
    h_0[int(2)] = leaky_relu_0(h_0[int(2)]);
    h_0[int(3)] = leaky_relu_0(h_0[int(3)]);
    h_0[int(4)] = leaky_relu_0(h_0[int(4)]);
    h_0[int(5)] = leaky_relu_0(h_0[int(5)]);
    h_0[int(6)] = leaky_relu_0(h_0[int(6)]);
    h_0[int(7)] = leaky_relu_0(h_0[int(7)]);
    h_0[int(8)] = leaky_relu_0(h_0[int(8)]);
    h_0[int(9)] = leaky_relu_0(h_0[int(9)]);
    h_0[int(10)] = leaky_relu_0(h_0[int(10)]);
    h_0[int(11)] = leaky_relu_0(h_0[int(11)]);
    h_0[int(12)] = leaky_relu_0(h_0[int(12)]);
    h_0[int(13)] = leaky_relu_0(h_0[int(13)]);
    h_0[int(14)] = leaky_relu_0(h_0[int(14)]);
    h_0[int(15)] = leaky_relu_0(h_0[int(15)]);
    float  _S24[int(2)];
    dense_1(params_buf_2, offset_2 + int(32) + int(16), h_0, _S24);
    _S22 = _S24;
    return;
}

int2 TextureImage_size_0(Texture2D<float4 > this_tex_0)
{
    uint w_0;
    uint h_1;
    this_tex_0.GetDimensions(w_0, h_1);
    return int2(int(w_0), int(h_1));
}

float3 TextureImage_load_0(Texture2D<float4 > this_tex_1, int2 texel_0)
{
    return this_tex_1.Load(int3(texel_0, int(0))).xyz;
}

float3 sample_bilinear_0(Texture2D<float4 > img_tex_0, float2 uv_0)
{
    int2 _S25 = TextureImage_size_0(img_tex_0);
    float2 p_0 = uv_0 * float2(_S25) - 0.5f;
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S26 = int2(p0_0);
    int2 _S27 = (int2)int(0);
    int2 _S28 = _S25 - int(1);
    int2 i0_0 = clamp(_S26, _S27, _S28);
    int2 i1_0 = clamp(_S26 + int(1), _S27, _S28);
    int _S29 = i0_0.x;
    int _S30 = i0_0.y;
    int _S31 = i1_0.x;
    float3 _S32 = (float3)f_0.x;
    int _S33 = i1_0.y;
    return lerp(lerp(TextureImage_load_0(img_tex_0, int2(_S29, _S30)), TextureImage_load_0(img_tex_0, int2(_S31, _S30)), _S32), lerp(TextureImage_load_0(img_tex_0, int2(_S29, _S33)), TextureImage_load_0(img_tex_0, int2(_S31, _S33)), _S32), (float3)f_0.y);
}

float2 texel_uv_0(int2 texel_1, int2 size_0)
{
    return (float2(texel_1) + 0.5f) / float2(size_0);
}

float3 upsample_0(int2 texel_2, int2 out_size_0, Texture2D<float4 > mip_tex_0, Texture2D<float4 > low_tex_0, StructuredBuffer<float > params_buf_3, int offset_3)
{
    float2 uv_1 = texel_uv_0(texel_2, out_size_0);
    float  x_4[int(2)] = { uv_1.x, uv_1.y };
    float  _S34[int(2)];
    MLP_eval_0(params_buf_3, offset_3, x_4, _S34);
    return _S34[int(0)] * TextureImage_load_0(mip_tex_0, texel_2) + _S34[int(1)] * sample_bilinear_0(low_tex_0, uv_1);
}

int2 target_size_0()
{
    uint w_1;
    uint h_2;
    _Target.GetDimensions(w_1, h_2);
    return int2(int(w_1), int(h_2));
}

[numthreads(8, 8, 1)]
void Upsample(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_1 = target_size_0();
    int2 _S35 = int2(id_0.xy);
    if(any(_S35 >= size_1))
    {
        return;
    }
    uint2 _S36 = uint2(_S35);
    int _S37 = ((_ParamOffset));
    _Target[_S36] = float4(upsample_0(_S35, size_1, _Source, _SourceLow, _GlareParams, _S37), 1.0f);
    return;
}

