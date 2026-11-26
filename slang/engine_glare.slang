// Engine entry points for the glare pyramid: compute kernels over Texture2D and StructuredBuffer
// Compiled to plain HLSL by export/build_hlsl.sh
// Engine side only, training counterpart is train_glare.slang
// Logic: pyramid.slang, mlp.slang, glare.slang

import pyramid;
import mlp;
import glare;

// Unity: an inline sampler, the name sets the state (linear filter, clamp addressing). The Unreal
// wrapper binds its own bilinear clamp sampler under this name. Efficiency step 1e.
SamplerState sampler_LinearClamp;

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

    // The hardware filter: a linear-clamp sampler clamps to the edge texels as bilinear_from_loads
    // does (step 1e, part 2: the levels, pyramid and composite).
    [Differentiable]
    float3 bilinear(no_diff float2 p)
    {
        float4 v = no_diff tex.SampleLevel(sampler_LinearClamp, (p + 0.5) / float2(size()), 0);
        return v.rgb;
    }

    // Hardware filter where the 2x2 footprint is inside the image; 0 where it is wholly outside;
    // four loads only where it straddles the edge (Unity's samplers have no clamp-to-border).
    [Differentiable]
    float3 bilinear_zero(no_diff float2 p)
    {
        int2 s = size();
        float2 p0 = floor(p);
        if (all(p0 >= 0.0) && all(p0 < float2(s - 1)))
        {
            float4 v = no_diff tex.SampleLevel(sampler_LinearClamp, (p + 0.5) / float2(s), 0);
            return v.rgb;
        }
        if (any(p0 < -1.0) || any(p0 > float2(s - 1)))
            return float3(0.0);
        return bilinear_zero_from_loads(this, p);
    }
}

// An image array (pyramid.slang IImageArray): the ray stack (efficiency step 1c; slice k = ray k's
// last streak pass). Taps as TextureImage: the hardware filter inside the image, 0 wholly
// outside, four loads on the edge.
struct TextureArray : IImageArray
{
    Texture2DArray<float4> tex;

    int2 size()
    {
        uint w, h, n;
        tex.GetDimensions(w, h, n);
        return int2(w, h);
    }

    [Differentiable]
    float3 load(int slice, int2 texel)
    {
        float4 v = no_diff tex.Load(int4(texel, slice, 0));
        return v.rgb;
    }

    [Differentiable]
    float3 bilinear(int slice, no_diff float2 p)
    {
        float3 uvw = float3((p + 0.5) / float2(size()), float(slice));
        float4 v = no_diff tex.SampleLevel(sampler_LinearClamp, uvw, 0);
        return v.rgb;
    }

    [Differentiable]
    float3 bilinear_zero(int slice, no_diff float2 p)
    {
        int2 s = size();
        float2 p0 = floor(p);
        if (all(p0 >= 0.0) && all(p0 < float2(s - 1)))
        {
            float3 uvw = float3((p + 0.5) / float2(s), float(slice));
            float4 v = no_diff tex.SampleLevel(sampler_LinearClamp, uvw, 0);
            return v.rgb;
        }
        if (any(p0 < -1.0) || any(p0 > float2(s - 1)))
            return float3(0.0);
        ImageSlice<TextureArray> one = { this, slice };
        return bilinear_zero_from_loads(one, p);
    }
}

struct BufferParams : IParams
{
    StructuredBuffer<float> buf;

    [Differentiable] float get(int index) { return no_diff buf[index]; }
}

Texture2D<float4> _Source;          // this pass's input (upsample: this level's mip)
Texture2D<float4> _SourceLow;       // upsample only: the output of the level below
Texture2D<float4> _Accum;           // StreakAccumulate / RayCrossAccumulate only: the ray family's sum so far
Texture2D<float4> _SpikeEven;       // SpikeComposite only: the even family's rays
Texture2D<float4> _SpikeOdd;        // SpikeComposite only: the odd family's rays
Texture2DArray<float4> _RayStack;  // SpikeCompositeStack only: slice k = ray k's last streak pass
StructuredBuffer<float> _GlareParams;
RWTexture2D<float4> _Target;
RWTexture2DArray<float4> _RayStackOut;   // StreakToStack only: written at slice _Ray

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

// Streak passes only: which ray (0..RAYS-1) and which pass (0..passes-1). Declared by the
// wrapper, as _ParamOffset (which holds the profile logits' start for the streak kernels and
// SpikeMLP's for SpikeComposite).
int ray_index()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "(_Ray)";
    }
}

int stage_index()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "(_Stage)";
    }
}

// Ray passes only: this pass's pre-exposure factor (glare.slang streak: k for the first streak
// pass, 1/k for the pass that adds to the family sum, 1 otherwise). Declared by the wrapper
// (`float _RayScale;`), as _ParamOffset.
float ray_scale()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "(_RayScale)";
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

// One pass of one spike ray at mip 0's resolution: _Source = S_stage, _Target = S_stage+1.
[shader("compute")]
[numthreads(8, 8, 1)]
void Streak(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(streak(texel, src, params, param_offset(), ray_index(), stage_index(),
                                   ray_scale()), 1.0);
}

// A ray's last pass added to its family's sum: _Target = _Accum + streak(_Source).
[shader("compute")]
[numthreads(8, 8, 1)]
void StreakAccumulate(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    TextureImage acc = { _Accum };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(streak_accumulate(texel, src, acc, params, param_offset(), ray_index(),
                                              stage_index(), ray_scale()), 1.0);
}

// A ray's cross filter (glare.slang ray_cross_accumulate, decision log 2026-11-19): _Target =
// _Accum + 3 taps across ray _Ray of _Source (the ray's last streak pass). _ParamOffset holds
// RayMLP's start (after the profile logits).
[shader("compute")]
[numthreads(8, 8, 1)]
void RayCrossAccumulate(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    TextureImage acc = { _Accum };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(ray_cross_accumulate(texel, size, src, acc, params, param_offset(),
                                                 ray_index(), ray_scale()), 1.0);
}

// RayCrossAccumulate with per-ray brightness (glare.slang ray_bright_accumulate, decision log
// 2026-11-19): _Target = _Accum + e^{z_k} * the cross-filtered ray. _ParamOffset holds RayMLP's
// start (BrightMLP follows it).
[shader("compute")]
[numthreads(8, 8, 1)]
void RayBrightAccumulate(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage src = { _Source };
    TextureImage acc = { _Accum };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(ray_bright_accumulate(texel, size, src, acc, params, param_offset(),
                                                  ray_index(), ray_scale()), 1.0);
}

// The final frame: _Source = the full-resolution level's output, mixed with the two families.
[shader("compute")]
[numthreads(8, 8, 1)]
void SpikeComposite(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage base = { _Source };
    TextureImage even = { _SpikeEven };
    TextureImage odd = { _SpikeOdd };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(spike_composite(texel, size, base, even, odd, params, param_offset()), 1.0);
}

// SpikeComposite with per-ray brightness: each family divided by its normaliser Z, not by 7.
// _ParamOffset holds SpikeMLP's start, as SpikeComposite.
[shader("compute")]
[numthreads(8, 8, 1)]
void SpikeCompositeBright(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage base = { _Source };
    TextureImage even = { _SpikeEven };
    TextureImage odd = { _SpikeOdd };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(spike_composite_bright(texel, size, base, even, odd, params,
                                                   param_offset()), 1.0);
}

// A ray's last streak pass into its slice of the ray stack (efficiency step 1c): _Source = S_stage,
// _RayStackOut[slice _Ray] = S_stage+1. _ParamOffset, _Ray, _Stage, _RayScale as Streak.
[shader("compute")]
[numthreads(8, 8, 1)]
void StreakToStack(uint3 id : SV_DispatchThreadID)
{
    uint w, h, n;
    _RayStackOut.GetDimensions(w, h, n);
    int2 texel = int2(id.xy);
    if (any(texel >= int2(w, h)))
        return;
    TextureImage src = { _Source };
    BufferParams params = { _GlareParams };
    _RayStackOut[uint3(id.xy, uint(ray_index()))] =
        float4(streak(texel, src, params, param_offset(), ray_index(), stage_index(), ray_scale()), 1.0);
}

// The final frame from the ray stack (glare.slang spike_composite_stack): _Source = the
// full-resolution level's output, _RayStack = every ray's last streak pass. _ParamOffset holds
// SpikeMLP's start, as SpikeCompositeBright; _RayScale = 1 / pre-exposure.
[shader("compute")]
[numthreads(8, 8, 1)]
void SpikeCompositeStack(uint3 id : SV_DispatchThreadID)
{
    int2 size = target_size();
    int2 texel = int2(id.xy);
    if (any(texel >= size))
        return;
    TextureImage base = { _Source };
    TextureArray rays = { _RayStack };
    BufferParams params = { _GlareParams };
    _Target[texel] = float4(spike_composite_stack(texel, size, base, rays, params, param_offset(),
                                                  ray_scale()), 1.0);
}
