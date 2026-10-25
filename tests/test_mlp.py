"""slang/mlp.slang: forward pass and Slang autodiff gradients match an independent PyTorch MLP.

The PyTorch network here is a test oracle only; the pipeline never uses it.
"""

from pathlib import Path

import pytest
import slangpy as spy
import torch
import torch.nn.functional as F

SLANG_DIR = Path(__file__).parents[1] / "slang"
IN, HIDDEN, OUT = 2, 16, 2           # must match TestMLP in train_mlp.slang
PARAM_COUNT = (IN * HIDDEN + HIDDEN) + (HIDDEN * OUT + OUT)


@pytest.fixture(scope="module")
def module():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[SLANG_DIR])
    return spy.Module.load_from_file(device, "train_mlp.slang")


@pytest.fixture
def params():
    torch.manual_seed(0)
    return (torch.randn(PARAM_COUNT, device="cuda") * 0.5).requires_grad_(True)


@pytest.fixture
def uv():
    torch.manual_seed(1)
    return torch.rand(1000, 2, device="cuda")


def torch_mlp(params, x):
    """Same network and flat layout: per layer, weights [out][in] row-major, then biases."""
    w1, b1, w2, b2 = torch.split(params, [IN * HIDDEN, HIDDEN, HIDDEN * OUT, OUT])
    h = F.leaky_relu(x @ w1.view(HIDDEN, IN).T + b1, negative_slope=0.01)
    return h @ w2.view(OUT, HIDDEN).T + b2


def test_forward_matches_torch(module, params, uv):
    got = module.mlp_test_eval(uv=uv, params=params)
    assert torch.allclose(got, torch_mlp(params, uv), atol=1e-5)


def test_gradients_match_torch(module, params, uv):
    # An arbitrary scalar loss: the gradient check doesn't care what the loss means
    got = torch.autograd.grad((module.mlp_test_eval(uv=uv, params=params) ** 2).sum(), params)[0]
    expected = torch.autograd.grad((torch_mlp(params, uv) ** 2).sum(), params)[0]
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4 * expected.abs().max().item())
