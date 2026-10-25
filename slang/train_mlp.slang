// slangpy entry points for mlp.slang. Training side only.
import slangpy;
import mlp;

// Parameters as a torch tensor: loads are differentiable and gradients land in tensor.grad.
struct TensorParams : IParams
{
    DiffTensor<float, 1> tensor;

    [Differentiable]
    float get(int index) { return tensor.load(index); }
}

typealias TestMLP = MLP<2, 16, 2>;

[Differentiable]
float2 mlp_test_eval(float2 uv, DiffTensor<float, 1> params)
{
    TensorParams p = { params };
    float x[2] = { uv.x, uv.y };
    float y[2] = TestMLP.eval(p, 0, x);
    return float2(y[0], y[1]);
}
