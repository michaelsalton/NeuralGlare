"""How an engine stores the ray intermediates between passes (efficiency step 1a, decision log
2026-11-24), modelled in float32, so Python's FLIP and the parity reference see what ships.
Rounding is toward zero, as the RTX 2070 does on D3D12 UAV stores (measured for fp16 by parity).
R11G11B10 was tried and rejected (1a', decision log 2026-11-24)."""
import torch

FORMATS = ("fp32", "fp16")


def half_toward_zero(x):
    """fp16, toward zero: above 65504 saturates at 65504."""
    h = x.half()
    toward_zero = torch.nextafter(h, torch.zeros_like(h))
    return torch.where(h.float().abs() > x.abs(), toward_zero, h).float()


def store_as(x, fmt):
    """x as stored in format `fmt` (FORMATS) and read back as float32. Straight-through gradient."""
    if fmt == "fp32":
        return x
    q = {"fp16": half_toward_zero}[fmt](x.detach())
    return x + (q - x).detach()
