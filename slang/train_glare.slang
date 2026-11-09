// slangpy entry points for glare.slang. Training side only.

import slangpy;
import pyramid;
import mlp;
import glare;

// An (H, W, 3) float tensor, as torch stores an RGB image.
struct TensorImage : IImage
{
    Tensor<float, 3> tensor;

    int2 size() { return int2(tensor.shape[1], tensor.shape[0]); }
    [Differentiable] float3 load(int2 texel)
    {
        return float3(no_diff tensor.load(texel.y, texel.x, 0), no_diff tensor.load(texel.y, texel.x, 1),
                      no_diff tensor.load(texel.y, texel.x, 2));
    }
}

struct DiffTensorImage : IImage
{
    DiffTensor<float, 3> tensor;
    int2 size() { return int2(tensor.shape[1], tensor.shape[0]); }
    [Differentiable]
    float3 load(int2 texel)
    {
        return float3(tensor.load(texel.y, texel.x, 0), tensor.load(texel.y, texel.x, 1),
                      tensor.load(texel.y, texel.x, 2));
    }
}

struct TensorParams : IParams
{
    DiffTensor<float, 1> tensor;

    [Differentiable]
    float get(int index) { return tensor.load(index); }
}

// The downsample chain on torch-layout (H, W, 3) tensors, so the whole pyramid runs in Slang
// (train_pyramid.slang's Tensor<float3, 2> doesn't accept torch tensors). Logic: pyramid.slang.
float3 prefilter_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return prefilter(texel, out_size, img);
}

float3 blur_h_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return blur_h(texel, out_size, img);
}

float3 blur_v_pass(int2 texel, int2 out_size, Tensor<float, 3> src)
{
    TensorImage img = { src };
    return blur_v(texel, out_size, img);
}

// Teacher forcing (step 3): low is a fixed input, so no gradient flows into it.
[Differentiable]
float3 upsample_pass(int2 texel, int2 out_size, Tensor<float, 3> mip, Tensor<float, 3> low,
                     DiffTensor<float, 1> params)
{
    TensorImage m = { mip };
    TensorImage l = { low };
    TensorParams p = { params };
    return upsample(texel, out_size, m, l, p, 0);
}

// End to end (step 4): low is the output of the level below, so the loss on the final image
// flows back through it. offset selects this level's network in the shared parameter buffer.
[Differentiable]
float3 upsample_chain_pass(int2 texel, int2 out_size, Tensor<float, 3> mip,
                           DiffTensor<float, 3> low, DiffTensor<float, 1> params, int offset)
{
    TensorImage m = { mip };
    DiffTensorImage l = { low };
    TensorParams p = { params };
    return upsample(texel, out_size, m, l, p, offset);
}

// LevelMLP's (inputs, hidden, outputs, PARAM_COUNT): how Python sizes and offsets the shared
// parameter buffer, so glare.slang stays the one source of truth for the network shape.
int4 level_mlp_sizes()
{
    return int4(LevelMLP.IN_SIZE, LevelMLP.HIDDEN_SIZE, LevelMLP.OUT_SIZE, LevelMLP.PARAM_COUNT);
}

// One channel's (0 R, 1 G, 2 B) tap weights at screen positions uv, in upsample()'s order:
// mip(x), mip(x + o), mip(x - o), up(low)(x). size is unused (the weights see uv only); it stays
// so the Python callers don't change.
[Differentiable]
float4 filter_weights(float2 uv, int2 size, DiffTensor<float, 1> params, int channel)
{
    TensorParams p = { params };
    float3 w[TAPS] = weights(uv, p, 0);
    return float4(w[0][channel], w[1][channel], w[2][channel], w[3][channel]);
}


// Spikes (glare.slang). One pass of one ray: src is S_stage (mip 0 for stage 0), a DiffTensor
// so the loss reaches the earlier passes' profile logits through it.
[Differentiable]
float3 streak_pass(int2 texel, DiffTensor<float, 3> src, DiffTensor<float, 1> params, int profile,
                   int ray, int stage)
{
    DiffTensorImage s = { src };
    TensorParams p = { params };
    return streak(texel, s, p, profile, ray, stage);
}

// A ray's last pass, added to its family's sum so far.
[Differentiable]
float3 streak_accumulate_pass(int2 texel, DiffTensor<float, 3> src, DiffTensor<float, 3> acc,
                              DiffTensor<float, 1> params, int profile, int ray, int stage)
{
    DiffTensorImage s = { src };
    DiffTensorImage a = { acc };
    TensorParams p = { params };
    return streak_accumulate(texel, s, a, p, profile, ray, stage);
}

// The final frame: the full-resolution output mixed with the two families' rays.
[Differentiable]
float3 spike_pass(int2 texel, int2 out_size, DiffTensor<float, 3> base, DiffTensor<float, 3> even,
                  DiffTensor<float, 3> odd, DiffTensor<float, 1> params, int offset)
{
    DiffTensorImage b = { base };
    DiffTensorImage e = { even };
    DiffTensorImage o = { odd };
    TensorParams p = { params };
    return spike_composite(texel, out_size, b, e, o, p, offset);
}

// SpikeMLP's (inputs, hidden, outputs, PARAM_COUNT) and the streaks' (RAYS, PASSES, STREAK_TAPS,
// PROFILE_COUNT): how Python sizes the spike section of the parameter buffer, which follows the
// levels: [levels x LevelMLP][SpikeMLP][PROFILE_COUNT logits].
int4 spike_mlp_sizes()
{
    return int4(SpikeMLP.IN_SIZE, SpikeMLP.HIDDEN_SIZE, SpikeMLP.OUT_SIZE, SpikeMLP.PARAM_COUNT);
}

int4 spike_sizes()
{
    return int4(RAYS, PASSES, STREAK_TAPS, PROFILE_COUNT);
}

// One channel's spike mix weights at screen positions uv: (pyramid, even rays, odd rays).
[Differentiable]
float3 spike_mix(float2 uv, DiffTensor<float, 1> params, int offset, int channel)
{
    TensorParams p = { params };
    float3 w[MIX] = spike_weights(uv, p, offset);
    return float3(w[0][channel], w[1][channel], w[2][channel]);
}
