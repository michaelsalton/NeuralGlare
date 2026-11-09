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

StructuredBuffer<float > _GlareParams : register(t5);

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

void dense_1(StructuredBuffer<float > params_buf_1, int offset_1, float  x_1[int(16)], out float  _S18[int(12)])
{
    int _S19 = offset_1 + int(192);
    float  y_1[int(12)];
    y_1[int(0)] = BufferParams_get_0(params_buf_1, _S19) + BufferParams_get_0(params_buf_1, offset_1) * x_1[int(0)] + BufferParams_get_0(params_buf_1, offset_1 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, offset_1 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, offset_1 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, offset_1 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, offset_1 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, offset_1 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, offset_1 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, offset_1 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, offset_1 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, offset_1 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, offset_1 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, offset_1 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, offset_1 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, offset_1 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, offset_1 + int(15)) * x_1[int(15)];
    int _S20 = offset_1 + int(16);
    y_1[int(1)] = BufferParams_get_0(params_buf_1, _S19 + int(1)) + BufferParams_get_0(params_buf_1, _S20) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S20 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S20 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S20 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S20 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S20 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S20 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S20 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S20 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S20 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S20 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S20 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S20 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S20 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S20 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S20 + int(15)) * x_1[int(15)];
    int _S21 = offset_1 + int(32);
    y_1[int(2)] = BufferParams_get_0(params_buf_1, _S19 + int(2)) + BufferParams_get_0(params_buf_1, _S21) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S21 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S21 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S21 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S21 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S21 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S21 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S21 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S21 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S21 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S21 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S21 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S21 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S21 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S21 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S21 + int(15)) * x_1[int(15)];
    int _S22 = offset_1 + int(48);
    y_1[int(3)] = BufferParams_get_0(params_buf_1, _S19 + int(3)) + BufferParams_get_0(params_buf_1, _S22) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S22 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S22 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S22 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S22 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S22 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S22 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S22 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S22 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S22 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S22 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S22 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S22 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S22 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S22 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S22 + int(15)) * x_1[int(15)];
    int _S23 = offset_1 + int(64);
    y_1[int(4)] = BufferParams_get_0(params_buf_1, _S19 + int(4)) + BufferParams_get_0(params_buf_1, _S23) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S23 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S23 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S23 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S23 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S23 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S23 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S23 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S23 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S23 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S23 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S23 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S23 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S23 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S23 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S23 + int(15)) * x_1[int(15)];
    int _S24 = offset_1 + int(80);
    y_1[int(5)] = BufferParams_get_0(params_buf_1, _S19 + int(5)) + BufferParams_get_0(params_buf_1, _S24) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S24 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S24 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S24 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S24 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S24 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S24 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S24 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S24 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S24 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S24 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S24 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S24 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S24 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S24 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S24 + int(15)) * x_1[int(15)];
    int _S25 = offset_1 + int(96);
    y_1[int(6)] = BufferParams_get_0(params_buf_1, _S19 + int(6)) + BufferParams_get_0(params_buf_1, _S25) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S25 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S25 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S25 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S25 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S25 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S25 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S25 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S25 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S25 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S25 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S25 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S25 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S25 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S25 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S25 + int(15)) * x_1[int(15)];
    int _S26 = offset_1 + int(112);
    y_1[int(7)] = BufferParams_get_0(params_buf_1, _S19 + int(7)) + BufferParams_get_0(params_buf_1, _S26) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S26 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S26 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S26 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S26 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S26 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S26 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S26 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S26 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S26 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S26 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S26 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S26 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S26 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S26 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S26 + int(15)) * x_1[int(15)];
    int _S27 = offset_1 + int(128);
    y_1[int(8)] = BufferParams_get_0(params_buf_1, _S19 + int(8)) + BufferParams_get_0(params_buf_1, _S27) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S27 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S27 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S27 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S27 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S27 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S27 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S27 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S27 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S27 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S27 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S27 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S27 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S27 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S27 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S27 + int(15)) * x_1[int(15)];
    int _S28 = offset_1 + int(144);
    y_1[int(9)] = BufferParams_get_0(params_buf_1, _S19 + int(9)) + BufferParams_get_0(params_buf_1, _S28) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S28 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S28 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S28 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S28 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S28 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S28 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S28 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S28 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S28 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S28 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S28 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S28 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S28 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S28 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S28 + int(15)) * x_1[int(15)];
    int _S29 = offset_1 + int(160);
    y_1[int(10)] = BufferParams_get_0(params_buf_1, _S19 + int(10)) + BufferParams_get_0(params_buf_1, _S29) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S29 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S29 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S29 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S29 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S29 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S29 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S29 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S29 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S29 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S29 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S29 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S29 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S29 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S29 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S29 + int(15)) * x_1[int(15)];
    int _S30 = offset_1 + int(176);
    y_1[int(11)] = BufferParams_get_0(params_buf_1, _S19 + int(11)) + BufferParams_get_0(params_buf_1, _S30) * x_1[int(0)] + BufferParams_get_0(params_buf_1, _S30 + int(1)) * x_1[int(1)] + BufferParams_get_0(params_buf_1, _S30 + int(2)) * x_1[int(2)] + BufferParams_get_0(params_buf_1, _S30 + int(3)) * x_1[int(3)] + BufferParams_get_0(params_buf_1, _S30 + int(4)) * x_1[int(4)] + BufferParams_get_0(params_buf_1, _S30 + int(5)) * x_1[int(5)] + BufferParams_get_0(params_buf_1, _S30 + int(6)) * x_1[int(6)] + BufferParams_get_0(params_buf_1, _S30 + int(7)) * x_1[int(7)] + BufferParams_get_0(params_buf_1, _S30 + int(8)) * x_1[int(8)] + BufferParams_get_0(params_buf_1, _S30 + int(9)) * x_1[int(9)] + BufferParams_get_0(params_buf_1, _S30 + int(10)) * x_1[int(10)] + BufferParams_get_0(params_buf_1, _S30 + int(11)) * x_1[int(11)] + BufferParams_get_0(params_buf_1, _S30 + int(12)) * x_1[int(12)] + BufferParams_get_0(params_buf_1, _S30 + int(13)) * x_1[int(13)] + BufferParams_get_0(params_buf_1, _S30 + int(14)) * x_1[int(14)] + BufferParams_get_0(params_buf_1, _S30 + int(15)) * x_1[int(15)];
    _S18 = y_1;
    return;
}

float leaky_relu_0(float x_2)
{
    float _S31;
    if(x_2 > 0.0f)
    {
        _S31 = x_2;
    }
    else
    {
        _S31 = 0.00999999977648258f * x_2;
    }
    return _S31;
}

void MLP_eval_0(StructuredBuffer<float > params_buf_2, int offset_2, float  x_3[int(2)], out float  _S32[int(12)])
{
    float  _S33[int(16)];
    dense_0(params_buf_2, offset_2, x_3, _S33);
    float  h_0[int(16)] = _S33;
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
    float  _S34[int(12)];
    dense_1(params_buf_2, offset_2 + int(32) + int(16), h_0, _S34);
    _S32 = _S34;
    return;
}

void softmax_0(float  z_0[int(4)], out float  _S35[int(4)])
{
    float _S36 = max(max(max(z_0[int(0)], z_0[int(1)]), z_0[int(2)]), z_0[int(3)]);
    float  w_0[int(4)];
    float _S37 = exp(z_0[int(0)] - _S36);
    float _S38 = exp(z_0[int(1)] - _S36);
    float _S39 = exp(z_0[int(2)] - _S36);
    float _S40 = exp(z_0[int(3)] - _S36);
    float sum_0 = _S37 + _S38 + _S39 + _S40;
    w_0[int(0)] = _S37 / sum_0;
    w_0[int(1)] = _S38 / sum_0;
    w_0[int(2)] = _S39 / sum_0;
    w_0[int(3)] = _S40 / sum_0;
    _S35 = w_0;
    return;
}

void softmax_rgb_0(float  z_1[int(12)], out float3  _S41[int(4)])
{
    float  zc_0[int(4)];
    zc_0[int(0)] = z_1[int(0)];
    zc_0[int(1)] = z_1[int(1)];
    zc_0[int(2)] = z_1[int(2)];
    zc_0[int(3)] = z_1[int(3)];
    float  _S42[int(4)];
    softmax_0(zc_0, _S42);
    float3  w_1[int(4)];
    w_1[int(0)][int(0)] = _S42[int(0)];
    w_1[int(1)][int(0)] = _S42[int(1)];
    w_1[int(2)][int(0)] = _S42[int(2)];
    w_1[int(3)][int(0)] = _S42[int(3)];
    float  zc_1[int(4)];
    zc_1[int(0)] = z_1[int(4)];
    zc_1[int(1)] = z_1[int(5)];
    zc_1[int(2)] = z_1[int(6)];
    zc_1[int(3)] = z_1[int(7)];
    float  _S43[int(4)];
    softmax_0(zc_1, _S43);
    w_1[int(0)][int(1)] = _S43[int(0)];
    w_1[int(1)][int(1)] = _S43[int(1)];
    w_1[int(2)][int(1)] = _S43[int(2)];
    w_1[int(3)][int(1)] = _S43[int(3)];
    float  zc_2[int(4)];
    zc_2[int(0)] = z_1[int(8)];
    zc_2[int(1)] = z_1[int(9)];
    zc_2[int(2)] = z_1[int(10)];
    zc_2[int(3)] = z_1[int(11)];
    float  _S44[int(4)];
    softmax_0(zc_2, _S44);
    w_1[int(0)][int(2)] = _S44[int(0)];
    w_1[int(1)][int(2)] = _S44[int(1)];
    w_1[int(2)][int(2)] = _S44[int(2)];
    w_1[int(3)][int(2)] = _S44[int(3)];
    _S41 = w_1;
    return;
}

void weights_0(float2 uv_0, StructuredBuffer<float > params_buf_3, int offset_3, out float3  _S45[int(4)])
{
    float  x_4[int(2)] = { uv_0.x, uv_0.y };
    float  _S46[int(12)];
    MLP_eval_0(params_buf_3, offset_3, x_4, _S46);
    float3  _S47[int(4)];
    softmax_rgb_0(_S46, _S47);
    _S45 = _S47;
    return;
}

int2 TextureImage_size_0(Texture2D<float4 > this_tex_0)
{
    uint w_2;
    uint h_1;
    this_tex_0.GetDimensions(w_2, h_1);
    return int2(int(w_2), int(h_1));
}

float3 TextureImage_load_0(Texture2D<float4 > this_tex_1, int2 texel_0)
{
    return this_tex_1.Load(int3(texel_0, int(0))).xyz;
}

float3 sample_bilinear_0(Texture2D<float4 > img_tex_0, float2 uv_1)
{
    int2 _S48 = TextureImage_size_0(img_tex_0);
    float2 p_0 = uv_1 * float2(_S48) - 0.5f;
    float2 p0_0 = floor(p_0);
    float2 f_0 = p_0 - p0_0;
    int2 _S49 = int2(p0_0);
    int2 _S50 = (int2)int(0);
    int2 _S51 = _S48 - int(1);
    int2 i0_0 = clamp(_S49, _S50, _S51);
    int2 i1_0 = clamp(_S49 + int(1), _S50, _S51);
    int _S52 = i0_0.x;
    int _S53 = i0_0.y;
    int _S54 = i1_0.x;
    float3 _S55 = (float3)f_0.x;
    int _S56 = i1_0.y;
    return lerp(lerp(TextureImage_load_0(img_tex_0, int2(_S52, _S53)), TextureImage_load_0(img_tex_0, int2(_S54, _S53)), _S55), lerp(TextureImage_load_0(img_tex_0, int2(_S52, _S56)), TextureImage_load_0(img_tex_0, int2(_S54, _S56)), _S55), (float3)f_0.y);
}

float2 texel_uv_0(int2 texel_1, int2 size_0)
{
    return (float2(texel_1) + 0.5f) / float2(size_0);
}

float2 field_offset_0(float2 uv_2, int2 size_1)
{
    float2 _S57 = float2(size_1);
    return (uv_2 - 0.5f) * _S57 / length(0.5f * _S57);
}

float3 upsample_0(int2 texel_2, int2 out_size_0, Texture2D<float4 > mip_tex_0, Texture2D<float4 > low_tex_0, StructuredBuffer<float > params_buf_4, int offset_4)
{
    float2 uv_3 = texel_uv_0(texel_2, out_size_0);
    float3  _S58[int(4)];
    weights_0(uv_3, params_buf_4, offset_4, _S58);
    float2 o_0 = field_offset_0(uv_3, out_size_0) / float2(out_size_0);
    return _S58[int(0)] * TextureImage_load_0(mip_tex_0, texel_2) + _S58[int(1)] * sample_bilinear_0(mip_tex_0, uv_3 + o_0) + _S58[int(2)] * sample_bilinear_0(mip_tex_0, uv_3 - o_0) + _S58[int(3)] * sample_bilinear_0(low_tex_0, uv_3);
}

int2 target_size_0()
{
    uint w_3;
    uint h_2;
    _Target.GetDimensions(w_3, h_2);
    return int2(int(w_3), int(h_2));
}

[numthreads(8, 8, 1)]
void Upsample(uint3 id_0 : SV_DispatchThreadID)
{
    int2 size_2 = target_size_0();
    int2 _S59 = int2(id_0.xy);
    if(any(_S59 >= size_2))
    {
        return;
    }
    uint2 _S60 = uint2(_S59);
    int _S61 = ((_ParamOffset));
    _Target[_S60] = float4(upsample_0(_S59, size_2, _Source, _SourceLow, _GlareParams, _S61), 1.0f);
    return;
}

