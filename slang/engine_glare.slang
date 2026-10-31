// Engine entry points for the glare pyramid: compute kernels over Texture2D and StructuredBuffer
// Compiled to plain HLSL by export/build_hlsl.sh
// Engine side only, training counterpart is train_glare.slang
// Logic: pyramid.slang, mlp.slang, glare.slang

import pyramid;
import mlp;
import glare;

// An engine texture texture. Only .rgb is used; alpha channel is ignored
struct TextureImage : IImage
{
    Texture2D<float4> tex;

    int2 size()
    {
        uint w, h;
        tex.GetDimensions(w, h);
        return int2(w, h);
    }

    [Differentiable]
    float3 load(int2 texel)
    {
        float4 v = no_diff tex.Load(int3(texel, 0));
        return v.rgb;
    }
}

struct BufferParams : IParams
{
    StructuredBuffer<float> buf;

    [Differentiable] float get(int index) { return no_diff buf[index]; }
}

Texture2D<float4> _Source;          // this pass's input (upsample: this level's mip)
Texture2D<float4> _SourceLow;       // upsample only: the output of the level below
StructuredBuffer<float> _GlareParams;
RWTexture2D<float4> _Target;

// Upsample only: this level's network in _GlareParams, k * PARAM_COUNT.
// Declared by the engine wrapper (e.g. `int _ParamOffset;` before the #include), not here:
// slangc wraps every uniform in a struct inside a cbuffer, which Unity's reflection rejects
// ("Unknown parameter type (0)"). The parentheses make Slang emit an expression, not a call.
int param_offset()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "(_ParamOffset)";
    }
}

// Sizes come from the bound textures, not from uniforms, so the engine can't pass a wrong one.
int2 target_size()
{
    uint w, h;
    _Target.GetDimensions(w, h);
    return int2(w, h);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void Prefilter(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    _Target[texel] = float4(prefilter(texel, size, src), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void BlurH(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    _Target[texel] = float4(blur_h(texel, size, src), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void BlurV(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    _Target[texel] = float4(blur_v(texel, size, src), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void Upsample(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage mip = { _Source };
    TextureImage low = { _SourceLow };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(upsample(texel, size, mip, low, params, param_offset()), 1.0);
}