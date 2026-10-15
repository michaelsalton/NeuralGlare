import torch

import neuralglare


def test_package_imports():
    assert neuralglare.__doc__


def test_cuda_available():
    assert torch.cuda.is_available()
