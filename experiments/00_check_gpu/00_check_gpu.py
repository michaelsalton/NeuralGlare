"""Sanity check: can PyTorch and slangpy both see the GPU?

Run with:  uv run python experiments/00_check_gpu.py
"""

import slangpy as spy
import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: torch was installed as a CPU-only build.")
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    # A tensor is PyTorch's GPU array, roughly a texture with any number of dimensions.
    x = torch.rand(1024, 1024, device="cuda")
    print(f"  1024x1024 matmul on GPU ok: {(x @ x).isfinite().all().item()}")

    device = spy.create_device()
    print(f"slangpy device: {device.info.adapter_name} ({device.info.type})")


if __name__ == "__main__":
    main()
