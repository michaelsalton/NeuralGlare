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

Texture2D<float4 > _SpikeEven : register(t3);

Texture2D<float4 > _SpikeOdd : register(t4);

StructuredBuffer<float > _GlareParams : register(t5);

float BufferParams_get_0(StructuredBuffer<float > this_buf_0, int index_0)
{
    return this_buf_0.Load(index_0);
}

void dense_0(StructuredBuffer<float > params_buf_0, int offset_0, float  x_0[int(3)], out float  _S1[int(8)])
{
    int _S2 = offset_0 + int(24);
    float  y_0[int(8)];
    y_0[int(0)] = BufferParams_get_0(params_buf_0, _S2) + BufferParams_get_0(params_buf_0, offset_0) * x_0[int(0)] + BufferParams_get_0(params_buf_0, offset_0 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, offset_0 + int(2)) * x_0[int(2)];
    int _S3 = offset_0 + int(3);
    y_0[int(1)] = BufferParams_get_0(params_buf_0, _S2 + int(1)) + BufferParams_get_0(params_buf_0, _S3) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S3 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S3 + int(2)) * x_0[int(2)];
    int _S4 = offset_0 + int(6);
    y_0[int(2)] = BufferParams_get_0(params_buf_0, _S2 + int(2)) + BufferParams_get_0(params_buf_0, _S4) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S4 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S4 + int(2)) * x_0[int(2)];
    int _S5 = offset_0 + int(9);
    y_0[int(3)] = BufferParams_get_0(params_buf_0, _S2 + int(3)) + BufferParams_get_0(params_buf_0, _S5) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S5 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S5 + int(2)) * x_0[int(2)];
    int _S6 = offset_0 + int(12);
    y_0[int(4)] = BufferParams_get_0(params_buf_0, _S2 + int(4)) + BufferParams_get_0(params_buf_0, _S6) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S6 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S6 + int(2)) * x_0[int(2)];
    int _S7 = offset_0 + int(15);
    y_0[int(5)] = BufferParams_get_0(params_buf_0, _S2 + int(5)) + BufferParams_get_0(params_buf_0, _S7) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S7 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S7 + int(2)) * x_0[int(2)];
    int _S8 = offset_0 + int(18);
    y_0[int(6)] = BufferParams_get_0(params_buf_0, _S2 + int(6)) + BufferParams_get_0(params_buf_0, _S8) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S8 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S8 + int(2)) * x_0[int(2)];
    int _S9 = offset_0 + int(21);
    y_0[int(7)] = BufferParams_get_0(params_buf_0, _S2 + int(7)) + BufferParams_get_0(params_buf_0, _S9) * x_0[int(0)] + BufferParams_get_0(params_buf_0, _S9 + int(1)) * x_0[int(1)] + BufferParams_get_0(params_buf_0, _S9 + int(2)) * x_0[int(2)];
    _S1 = y_0;
    return;
}

void dense_1(StructuredBuffer<float > params_buf_1, int offset_1, float  x_1[int(8)], out float  _S10[int(1)])
{
    float  y_1[int(1)];
    y_1[int(0)] = BufferParams_get_0(params_buf_1, offset_1 + int(8)) + BufferParams_get_0(params_buf_1, offset_1) * x_1[int(0)] + BufferParams_get_0(params_buf_1, offset_1 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, offset_1 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, offset_1 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, offset_1 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, offset_1 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, offset_1 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, offset_1 + int(7)) * x_1[int(7)];
    _S10 = y_1;
    return;
}

float leaky_relu_0(float x_2)
{
    float _S11;
    if(x_2 > 0.0f)
    {
        _S11 = x_2;
    }
    else
    {
        _S11 = 0.00999999977648258f * x_2;
    }
    return _S11;
}

void MLP_eval_0(StructuredBuffer<float > params_buf_2, int offset_2, float  x_3[int(3)], out float  _S12[int(1)])
{
    float  _S13[int(8)];
    dense_0(params_buf_2, offset_2, x_3, _S13);
    float  h_0[int(8)] = _S13;
    h_0[int(0)] = leaky_relu_0(h_0[int(0)]);
    h_0[int(1)] = leaky_relu_0(h_0[int(1)]);
    h_0[int(2)] = leaky_relu_0(h_0[int(2)]);
    h_0[int(3)] = leaky_relu_0(h_0[int(3)]);
    h_0[int(4)] = leaky_relu_0(h_0[int(4)]);
    h_0[int(5)] = leaky_relu_0(h_0[int(5)]);
    h_0[int(6)] = leaky_relu_0(h_0[int(6)]);
    h_0[int(7)] = leaky_relu_0(h_0[int(7)]);
    float  _S14[int(1)];
    dense_1(params_buf_2, offset_2 + int(24) + int(8), h_0, _S14);
    _S12 = _S14;
    return;
}

static const float2  RAY_DIRS_0[int(14)] = { float2(1.0f, 0.0f), float2(0.90096884965896606f, 0.43388372659683228f), float2(0.62348979711532593f, 0.78183150291442871f), float2(0.22252093255519867f, 0.97492790222167969f), float2(-0.22252093255519867f, 0.97492790222167969f), float2(-0.62348979711532593f, 0.78183150291442871f), float2(-0.90096884965896606f, 0.43388372659683228f), float2(-1.0f, 0.0f), float2(-0.90096884965896606f, -0.43388372659683228f), float2(-0.62348979711532593f, -0.78183150291442871f), float2(-0.22252093255519867f, -0.97492790222167969f), float2(0.22252093255519867f, -0.97492790222167969f), float2(0.62348979711532593f, -0.78183150291442871f), float2(0.90096884965896606f, -0.43388372659683228f) };
float bright_logit_0(float2 v_0, StructuredBuffer<float > params_buf_3, int offset_3, int ray_0)
{
    float _S15 = dot(RAY_DIRS_0[ray_0], v_0);
    float _S16 = dot(float2(- RAY_DIRS_0[ray_0].y, RAY_DIRS_0[ray_0].x), v_0);
    float _S17;
    if((ray_0 % int(2)) == int(0))
    {
        _S17 = 1.0f;
    }
    else
    {
        _S17 = -1.0f;
    }
    float  x_4[int(3)] = { _S15, _S16, _S17 };
    float  _S18[int(1)];
    MLP_eval_0(params_buf_3, offset_3, x_4, _S18);
    return _S18[int(0)];
}

void dense_2(StructuredBuffer<float > params_buf_4, int offset_4, float  x_5[int(2)], out float  _S19[int(16)])
{
    int _S20 = offset_4 + int(32);
    float  y_2[int(16)];
    y_2[int(0)] = BufferParams_get_0(params_buf_4, _S20) + BufferParams_get_0(params_buf_4, offset_4) * x_5[int(0)] + BufferParams_get_0(params_buf_4, offset_4 + int(1)) * x_5[int(1)];
    int _S21 = offset_4 + int(2);
    y_2[int(1)] = BufferParams_get_0(params_buf_4, _S20 + int(1)) + BufferParams_get_0(params_buf_4, _S21) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S21 + int(1)) * x_5[int(1)];
    int _S22 = offset_4 + int(4);
    y_2[int(2)] = BufferParams_get_0(params_buf_4, _S20 + int(2)) + BufferParams_get_0(params_buf_4, _S22) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S22 + int(1)) * x_5[int(1)];
    int _S23 = offset_4 + int(6);
    y_2[int(3)] = BufferParams_get_0(params_buf_4, _S20 + int(3)) + BufferParams_get_0(params_buf_4, _S23) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S23 + int(1)) * x_5[int(1)];
    int _S24 = offset_4 + int(8);
    y_2[int(4)] = BufferParams_get_0(params_buf_4, _S20 + int(4)) + BufferParams_get_0(params_buf_4, _S24) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S24 + int(1)) * x_5[int(1)];
    int _S25 = offset_4 + int(10);
    y_2[int(5)] = BufferParams_get_0(params_buf_4, _S20 + int(5)) + BufferParams_get_0(params_buf_4, _S25) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S25 + int(1)) * x_5[int(1)];
    int _S26 = offset_4 + int(12);
    y_2[int(6)] = BufferParams_get_0(params_buf_4, _S20 + int(6)) + BufferParams_get_0(params_buf_4, _S26) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S26 + int(1)) * x_5[int(1)];
    int _S27 = offset_4 + int(14);
    y_2[int(7)] = BufferParams_get_0(params_buf_4, _S20 + int(7)) + BufferParams_get_0(params_buf_4, _S27) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S27 + int(1)) * x_5[int(1)];
    int _S28 = offset_4 + int(16);
    y_2[int(8)] = BufferParams_get_0(params_buf_4, _S20 + int(8)) + BufferParams_get_0(params_buf_4, _S28) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S28 + int(1)) * x_5[int(1)];
    int _S29 = offset_4 + int(18);
    y_2[int(9)] = BufferParams_get_0(params_buf_4, _S20 + int(9)) + BufferParams_get_0(params_buf_4, _S29) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S29 + int(1)) * x_5[int(1)];
    int _S30 = offset_4 + int(20);
    y_2[int(10)] = BufferParams_get_0(params_buf_4, _S20 + int(10)) + BufferParams_get_0(params_buf_4, _S30) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S30 + int(1)) * x_5[int(1)];
    int _S31 = offset_4 + int(22);
    y_2[int(11)] = BufferParams_get_0(params_buf_4, _S20 + int(11)) + BufferParams_get_0(params_buf_4, _S31) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S31 + int(1)) * x_5[int(1)];
    int _S32 = offset_4 + int(24);
    y_2[int(12)] = BufferParams_get_0(params_buf_4, _S20 + int(12)) + BufferParams_get_0(params_buf_4, _S32) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S32 + int(1)) * x_5[int(1)];
    int _S33 = offset_4 + int(26);
    y_2[int(13)] = BufferParams_get_0(params_buf_4, _S20 + int(13)) + BufferParams_get_0(params_buf_4, _S33) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S33 + int(1)) * x_5[int(1)];
    int _S34 = offset_4 + int(28);
    y_2[int(14)] = BufferParams_get_0(params_buf_4, _S20 + int(14)) + BufferParams_get_0(params_buf_4, _S34) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S34 + int(1)) * x_5[int(1)];
    int _S35 = offset_4 + int(30);
    y_2[int(15)] = BufferParams_get_0(params_buf_4, _S20 + int(15)) + BufferParams_get_0(params_buf_4, _S35) * x_5[int(0)] + BufferParams_get_0(params_buf_4, _S35 + int(1)) * x_5[int(1)];
    _S19 = y_2;
    return;
}

void dense_3(StructuredBuffer<float > params_buf_5, int offset_5, float  x_6[int(16)], out float  _S36[int(9)])
{
    int _S37 = offset_5 + int(144);
    float  y_3[int(9)];
    y_3[int(0)] = BufferParams_get_0(params_buf_5, _S37) + BufferParams_get_0(params_buf_5, offset_5) * x_6[int(0)] + BufferParams_get_0(params_buf_5, offset_5 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, offset_5 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, offset_5 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, offset_5 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, offset_5 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, offset_5 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, offset_5 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, offset_5 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, offset_5 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, offset_5 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, offset_5 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, offset_5 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, offset_5 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, offset_5 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, offset_5 + int(15)) * x_6[int(15)];
    int _S38 = offset_5 + int(16);
    y_3[int(1)] = BufferParams_get_0(params_buf_5, _S37 + int(1)) + BufferParams_get_0(params_buf_5, _S38) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S38 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S38 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S38 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S38 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S38 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S38 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S38 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S38 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S38 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S38 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S38 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S38 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S38 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S38 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S38 + int(15)) * x_6[int(15)];
    int _S39 = offset_5 + int(32);
    y_3[int(2)] = BufferParams_get_0(params_buf_5, _S37 + int(2)) + BufferParams_get_0(params_buf_5, _S39) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S39 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S39 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S39 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S39 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S39 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S39 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S39 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S39 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S39 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S39 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S39 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S39 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S39 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S39 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S39 + int(15)) * x_6[int(15)];
    int _S40 = offset_5 + int(48);
    y_3[int(3)] = BufferParams_get_0(params_buf_5, _S37 + int(3)) + BufferParams_get_0(params_buf_5, _S40) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S40 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S40 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S40 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S40 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S40 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S40 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S40 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S40 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S40 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S40 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S40 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S40 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S40 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S40 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S40 + int(15)) * x_6[int(15)];
    int _S41 = offset_5 + int(64);
    y_3[int(4)] = BufferParams_get_0(params_buf_5, _S37 + int(4)) + BufferParams_get_0(params_buf_5, _S41) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S41 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S41 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S41 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S41 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S41 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S41 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S41 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S41 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S41 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S41 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S41 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S41 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S41 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S41 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S41 + int(15)) * x_6[int(15)];
    int _S42 = offset_5 + int(80);
    y_3[int(5)] = BufferParams_get_0(params_buf_5, _S37 + int(5)) + BufferParams_get_0(params_buf_5, _S42) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S42 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S42 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S42 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S42 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S42 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S42 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S42 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S42 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S42 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S42 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S42 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S42 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S42 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S42 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S42 + int(15)) * x_6[int(15)];
    int _S43 = offset_5 + int(96);
    y_3[int(6)] = BufferParams_get_0(params_buf_5, _S37 + int(6)) + BufferParams_get_0(params_buf_5, _S43) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S43 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S43 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S43 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S43 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S43 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S43 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S43 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S43 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S43 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S43 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S43 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S43 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S43 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S43 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S43 + int(15)) * x_6[int(15)];
    int _S44 = offset_5 + int(112);
    y_3[int(7)] = BufferParams_get_0(params_buf_5, _S37 + int(7)) + BufferParams_get_0(params_buf_5, _S44) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S44 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S44 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S44 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S44 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S44 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S44 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S44 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S44 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S44 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S44 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S44 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S44 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S44 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S44 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S44 + int(15)) * x_6[int(15)];
    int _S45 = offset_5 + int(128);
    y_3[int(8)] = BufferParams_get_0(params_buf_5, _S37 + int(8)) + BufferParams_get_0(params_buf_5, _S45) * x_6[int(0)] + BufferParams_get_0(params_buf_5, _S45 + int(1)) * x_6[int(1)] + BufferParams_get_0(params_buf_5, _S45 + int(2)) * x_6[int(2)] + BufferParams_get_0(params_buf_5, _S45 + int(3)) * x_6[int(3)] + BufferParams_get_0(params_buf_5, _S45 + int(4)) * x_6[int(4)] + BufferParams_get_0(params_buf_5, _S45 + int(5)) * x_6[int(5)] + BufferParams_get_0(params_buf_5, _S45 + int(6)) * x_6[int(6)] + BufferParams_get_0(params_buf_5, _S45 + int(7)) * x_6[int(7)] + BufferParams_get_0(params_buf_5, _S45 + int(8)) * x_6[int(8)] + BufferParams_get_0(params_buf_5, _S45 + int(9)) * x_6[int(9)] + BufferParams_get_0(params_buf_5, _S45 + int(10)) * x_6[int(10)] + BufferParams_get_0(params_buf_5, _S45 + int(11)) * x_6[int(11)] + BufferParams_get_0(params_buf_5, _S45 + int(12)) * x_6[int(12)] + BufferParams_get_0(params_buf_5, _S45 + int(13)) * x_6[int(13)] + BufferParams_get_0(params_buf_5, _S45 + int(14)) * x_6[int(14)] + BufferParams_get_0(params_buf_5, _S45 + int(15)) * x_6[int(15)];
    _S36 = y_3;
    return;
}

void MLP_eval_1(StructuredBuffer<float > params_buf_6, int offset_6, float  x_7[int(2)], out float  _S46[int(9)])
{
    float  _S47[int(16)];
    dense_2(params_buf_6, offset_6, x_7, _S47);
    float  h_1[int(16)] = _S47;
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
    float  _S48[int(9)];
    dense_3(params_buf_6, offset_6 + int(32) + int(16), h_1, _S48);
    _S46 = _S48;
    return;
}

float bright_norm_0(float2 v_1, StructuredBuffer<float > params_buf_7, int offset_7, int family_0)
{
    return exp(bright_logit_0(v_1, params_buf_7, offset_7, family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(2) + family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(4) + family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(6) + family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(8) + family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(10) + family_0)) + exp(bright_logit_0(v_1, params_buf_7, offset_7, int(12) + family_0));
}

void softmax_0(float  z_0[int(3)], out float  _S49[int(3)])
{
    float _S50 = max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]);
    float  w_0[int(3)];
    float _S51 = exp(z_0[int(0)] - _S50);
    float _S52 = exp(z_0[int(1)] - _S50);
    float _S53 = exp(z_0[int(2)] - _S50);
    float sum_0 = _S51 + _S52 + _S53;
    w_0[int(0)] = _S51 / sum_0;
    w_0[int(1)] = _S52 / sum_0;
    w_0[int(2)] = _S53 / sum_0;
    _S49 = w_0;
    return;
}

void softmax_rgb_0(float  z_1[int(9)], out float3  _S54[int(3)])
{
    float  zc_0[int(3)];
    zc_0[int(0)] = z_1[int(0)];
    zc_0[int(1)] = z_1[int(1)];
    zc_0[int(2)] = z_1[int(2)];
    float  _S55[int(3)];
    softmax_0(zc_0, _S55);
    float3  w_1[int(3)];
    w_1[int(0)][int(0)] = _S55[int(0)];
    w_1[int(1)][int(0)] = _S55[int(1)];
    w_1[int(2)][int(0)] = _S55[int(2)];
    float  zc_1[int(3)];
    zc_1[int(0)] = z_1[int(3)];
    zc_1[int(1)] = z_1[int(4)];
    zc_1[int(2)] = z_1[int(5)];
    float  _S56[int(3)];
    softmax_0(zc_1, _S56);
    w_1[int(0)][int(1)] = _S56[int(0)];
    w_1[int(1)][int(1)] = _S56[int(1)];
    w_1[int(2)][int(1)] = _S56[int(2)];
    float  zc_2[int(3)];
    zc_2[int(0)] = z_1[int(6)];
    zc_2[int(1)] = z_1[int(7)];
    zc_2[int(2)] = z_1[int(8)];
    float  _S57[int(3)];
    softmax_0(zc_2, _S57);
    w_1[int(0)][int(2)] = _S57[int(0)];
    w_1[int(1)][int(2)] = _S57[int(1)];
    w_1[int(2)][int(2)] = _S57[int(2)];
    _S54 = w_1;
    return;
}

void spike_weights_0(float2 uv_0, StructuredBuffer<float > params_buf_8, int offset_8, out float3  _S58[int(3)])
{
    float  x_8[int(2)] = { uv_0.x, uv_0.y };
    float  _S59[int(9)];
    MLP_eval_1(params_buf_8, offset_8, x_8, _S59);
    float3  _S60[int(3)];
    softmax_rgb_0(_S59, _S60);
    _S58 = _S60;
    return;
}

int2 TextureImage_size_0(Texture2D<float4 > this_tex_0)
{
    uint w_2;
    uint h_2;
    this_tex_0.GetDimensions(w_2, h_2);
    return int2(int(w_2), int(h_2));
}

float3 TextureImage_load_0(Texture2D<float4 > this_tex_1, int2 texel_0)
{
    return this_tex_1.Load(int3(texel_0, int(0))).xyz;
}

float3 sample_bilinear_0(Texture2D<float4 > img_tex_0, float2 uv_1)
{
    int2 _S61 = TextureImage_size_0(img_tex_0);
    float2 p_0 = uv_1 * float2(_S61) - 0.5f;
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S62 = int2(p0_0);
    int2 _S63 = (int2)int(0);
    int2 _S64 = _S61 - int(1);
    int2 i0_0 = clamp(_S62, _S63, _S64);
    int2 i1_0 = clamp(_S62 + int(1), _S63, _S64);
    int _S65 = i0_0.x;
    int _S66 = i0_0.y;
    int _S67 = i1_0.x;
    float3 _S68 = (float3)f_0.x;
    int _S69 = i1_0.y;
    return lerp(lerp(TextureImage_load_0(img_tex_0, int2(_S65, _S66)), TextureImage_load_0(img_tex_0, int2(_S67, _S66)), _S68), lerp(TextureImage_load_0(img_tex_0, int2(_S65, _S69)), TextureImage_load_0(img_tex_0, int2(_S67, _S69)), _S68), (float3)f_0.y);
}

float2 texel_uv_0(int2 texel_1, int2 size_0)
{
    return (float2(texel_1) + 0.5f) / float2(size_0);
}

float2 field_offset_0(float2 uv_2, int2 size_1)
{
    float2 _S70 = float2(size_1);
    return (uv_2 - 0.5f) * _S70 / length(0.5f * _S70);
}

float3 spike_composite_bright_0(int2 texel_2, int2 out_size_0, Texture2D<float4 > base_tex_0, Texture2D<float4 > even_tex_0, Texture2D<float4 > odd_tex_0, StructuredBuffer<float > params_buf_9, int offset_9)
{
    float2 uv_3 = texel_uv_0(texel_2, out_size_0);
    float3  _S71[int(3)];
    spike_weights_0(uv_3, params_buf_9, offset_9, _S71);
    float2 v_2 = field_offset_0(uv_3, out_size_0);
    int bright_0 = offset_9 + int(201) + int(24) + int(59);
    return _S71[int(0)] * TextureImage_load_0(base_tex_0, texel_2) + _S71[int(1)] * sample_bilinear_0(even_tex_0, uv_3) / bright_norm_0(v_2, params_buf_9, bright_0, int(0)) + _S71[int(2)] * sample_bilinear_0(odd_tex_0, uv_3) / bright_norm_0(v_2, params_buf_9, bright_0, int(1));
}

int2 target_size_0()
{
    uint w_3;
    uint h_3;
    _Target.GetDimensions(w_3, h_3);
    return int2(int(w_3), int(h_3));
}

[numthreads(8, 8, 1)]
void SpikeCompositeBright(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_2 = target_size_0();
    int2 _S72 = int2(id_0.xy);
    if(any(_S72 >= size_2))
    {
        return;
    }
    uint2 _S73 = uint2(_S72);
    int _S74 = ((_ParamOffset));
    _Target[_S73] = float4(spike_composite_bright_0(_S72, size_2, _Source, _SpikeEven, _SpikeOdd, _GlareParams, _S74), 1.0f);
    return;
}

