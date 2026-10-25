// Tiny fully connected network, scalar path only (plain loops, no cooperative vectors), so the
// generated HLSL runs in any engine.
//
// All parameters of one network live in one flat float array, read through IParams. Layout,
// layer by layer: weights row-major [out][in], then biases [out]. The same layout is the
// weight file format and the engines' StructuredBuffer<float>.

module mlp;

public interface IParams
{
    [Differentiable]
    public float get(int index);
}

[Differentiable]
public float leaky_relu(float x)
{
    return x > 0.0 ? x : 0.01 * x;
}

// One dense layer: y = W x + b, reading parameters starting at `offset`.
[Differentiable]
public float[Out] dense<let In : int, let Out : int, P : IParams>(P params, no_diff int offset,
                                                                  float x[In])
{
    float y[Out];
    [ForceUnroll]
    for (int o = 0; o < Out; ++o)
    {
        float sum = params.get(offset + In * Out + o);              // bias
        [ForceUnroll]
        for (int i = 0; i < In; ++i)
            sum += params.get(offset + o * In + i) * x[i];          // weight [o][i]
        y[o] = sum;
    }
    return y;
}

// In -> Hidden -> (leaky ReLU) -> Out.
public struct MLP<let In : int, let Hidden : int, let Out : int>
{
    public static const int PARAM_COUNT = (In * Hidden + Hidden) + (Hidden * Out + Out);

    [Differentiable]
    public static float[Out] eval<P : IParams>(P params, no_diff int offset, float x[In])
    {
        float h[Hidden] = dense<In, Hidden>(params, offset, x);
        [ForceUnroll]
        for (int i = 0; i < Hidden; ++i)
            h[i] = leaky_relu(h[i]);
        return dense<Hidden, Out>(params, offset + In * Hidden + Hidden, h);
    }
}
