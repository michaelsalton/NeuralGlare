import torch

from neuralglare.storage import half_toward_zero, store_as


def px(*rgb):
    return torch.tensor([[list(rgb)]], dtype=torch.float32)


def test_half_rounds_toward_zero_and_saturates():
    assert torch.equal(half_toward_zero(px(1 + 2**-11, 1 - 2**-12, 1e6)), px(1, 1 - 2**-11, 65504))


def test_store_as_has_a_straight_through_gradient():
    x = px(1.3, 2.7, 0.01).requires_grad_()
    store_as(x, "fp16").sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
