"""Step 0: Slang autodiff and PyTorch autograd work together.

If these pass, a [Differentiable] Slang function can be called on torch tensors,
torch's loss.backward() reaches the Slang parameters, and torch.optim can train them.
"""

from pathlib import Path

import pytest
import slangpy as spy
import torch

SHADER_DIR = Path(__file__).parent / "slang"


@pytest.fixture(scope="module")
def module():
    # The native torch bridge (slangpy-torch) isn't built; the Python fallback is slower but exact.
    spy.set_allow_torch_fallback(True)
    # CUDA device creation fails on this machine; Vulkan works through CUDA interop.
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[SHADER_DIR])
    return spy.Module.load_from_file(device, "weighted_sum.slang")


@pytest.fixture
def data():
    torch.manual_seed(0)
    a = torch.rand(1000, device="cuda")
    b = torch.rand(1000, device="cuda")
    return a, b, 0.3 * a + 0.7 * b      # target: known weights (0.3, 0.7)


def test_gradients_match_hand_derivation(module, data):
    a, b, target = data
    w0 = torch.zeros(1, device="cuda", requires_grad=True)
    w1 = torch.zeros(1, device="cuda", requires_grad=True)

    out = module.weighted_sum(w0=w0, w1=w1, a=a, b=b)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # d/dw0 mean((w0·a + w1·b − t)²) = mean(2·(out − t)·a), and the same with b for w1
    residual = (out - target).detach()
    assert torch.allclose(w0.grad, (2 * residual * a).mean(), rtol=1e-4)
    assert torch.allclose(w1.grad, (2 * residual * b).mean(), rtol=1e-4)


def test_adam_recovers_weights(module, data):
    a, b, target = data
    w0 = torch.zeros(1, device="cuda", requires_grad=True)
    w1 = torch.zeros(1, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([w0, w1], lr=0.05)

    for _ in range(500):
        optimizer.zero_grad()
        out = module.weighted_sum(w0=w0, w1=w1, a=a, b=b)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    assert abs(w0.item() - 0.3) < 1e-3
    assert abs(w1.item() - 0.7) < 1e-3
