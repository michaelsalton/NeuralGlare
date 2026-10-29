"""Phase 1, step 3: train each upsample level on its own against URP bloom (teacher forcing).

Every level k gets the *correct* input from the level below (URP's up[k+1]) and learns
    out_k = w0(uv) * mip_k + w1(uv) * bilinear(up[k+1])
against URP's up[k]. The exact answer is w = (1 - s, s) at every uv, s = 0.05 + 0.9 * scatter,
so the learned weights show directly whether the Slang pass, the MLP and training work.

Inputs and targets come from the NumPy URP baseline (threshold 0, no clamp); only the upsample
pass runs in Slang. Step 4 replaces this with the Slang pyramid trained end to end.

    uv run python experiments/02_slang_plumbing/train_levels.py
"""

import argparse
import json
import platform
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import slangpy as spy
import torch

from neuralglare.baselines.pyramid_bloom import (
    URPBloomParams,
    downsample_chain,
    sample_bilinear,
    shader_constants,
    texel_uv,
)
from neuralglare.io import read_hdr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "02_slang_plumbing"
IN, HIDDEN, OUT = 2, 16, 2          # must match LevelMLP in glare.slang
VIEWS = ("center", "mid", "corner")
EPS = 1e-3                          # same ε as the Phase 0 log loss


def upsample_targets(mips, scatter):
    """URP's upsample chain, keeping every level: up[k] = lerp(mip[k], bilinear(up[k+1]), s)."""
    ups = [None] * len(mips)
    ups[-1] = mips[-1]
    for k in range(len(mips) - 2, -1, -1):
        h, w = mips[k].shape[:2]
        low = sample_bilinear(ups[k + 1], *texel_uv(h, w))
        ups[k] = (mips[k] + scatter * (low - mips[k])).astype(np.float32)
    return ups


def load_split(split, p, scatter):
    """[(mips, ups)] for every view of every scene in the split."""
    items = []
    for scene in split:
        for view in VIEWS:
            img = read_hdr(str(ROOT / "data" / "inputs" / f"{scene}_{view}.exr"))[..., :3]
            mips = downsample_chain(img.astype(np.float32), p)
            items.append((mips, upsample_targets(mips, scatter)))
    return items


def init_params(gen):
    """Hidden layer: He init, so the ReLU features start well spread. Output layer: small
    weights and bias 0.5, so every level starts near w = (0.5, 0.5) instead of at random."""
    w1 = torch.randn(HIDDEN, IN, generator=gen) * np.sqrt(2.0 / IN)
    b1 = torch.zeros(HIDDEN)
    w2 = torch.randn(OUT, HIDDEN, generator=gen) * np.sqrt(1.0 / HIDDEN) * 0.1
    b2 = torch.full((OUT,), 0.5)
    return torch.cat([w1.flatten(), b1, w2.flatten(), b2])


def relative_loss(pred, target):
    """Mean squared relative error. Not log(x + ε): an untrained level can output negatives."""
    return (((pred - target) / (target + EPS)) ** 2).mean()


def level_tensors(item, k):
    """Level k's (mip_k, up[k+1], up[k]) of one frame, on the GPU."""
    mips, ups = item
    return [torch.from_numpy(a).cuda() for a in (mips[k], ups[k + 1], ups[k])]


def run(module, mip, low, params):
    """One upsample pass at mip's resolution."""
    h, w = mip.shape[:2]
    return module.upsample_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                mip=mip, low=low, params=params)


def command_output(*cmd):
    """First line a command prints, or None if it isn't installed or fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return ((out.stdout or out.stderr).strip().splitlines() or [None])[0]


def environment(device):
    """Versions and hardware for the experiment log (CLAUDE.md: record versions)."""
    return {"slangpy": spy.__version__, "slangc": command_output("slangc", "-v"),
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": device.info.adapter_name, "api": device.info.api_name,
            "driver": command_output("nvidia-smi", "--query-gpu=driver_version",
                                     "--format=csv,noheader")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    split = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase1"]
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)

    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    module = spy.Module.load_from_file(device, "train_glare.slang")

    p = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)
    _, _, scatter = shader_constants(p)
    data = {name: load_split(split[name], p, scatter) for name in ("train", "test")}
    n_levels = len(data["train"][0][0]) - 1          # the smallest mip has no upsample pass

    # Where to read the learned weights back: a 16 x 9 grid over the screen, corners included
    gu, gv = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 9))
    grid_uv = torch.tensor(np.stack([gu.ravel(), gv.ravel()], -1), dtype=torch.float32,
                           device="cuda")
    exact = torch.tensor([1 - scatter, scatter], device="cuda")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"environment": environment(device), "scatter": scatter, "steps": args.steps,
               "lr": args.lr, "levels": []}
    for k in range(n_levels):
        train = [level_tensors(item, k) for item in data["train"]]
        test = [level_tensors(item, k) for item in data["test"]]
        h, w = train[0][0].shape[:2]

        params = init_params(gen).cuda().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=args.lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            mip, low, target = train[rng.integers(len(train))]   # one random training frame
            optimizer.zero_grad()
            loss = relative_loss(run(module, mip, low, params), target)
            loss.backward()
            optimizer.step()
            schedule.step()

        with torch.no_grad():
            train_loss = float(np.mean([relative_loss(run(module, m, l, params), t).item()
                                        for m, l, t in train]))
            test_loss = float(np.mean([relative_loss(run(module, m, l, params), t).item()
                                       for m, l, t in test]))
            weights = module.filter_weights(uv=grid_uv, params=params)
            max_dev = (weights - exact).abs().max().item()
        np.save(OUT_DIR / f"level{k}_params.npy", params.detach().cpu().numpy())
        results["levels"].append({"level": k, "size": [w, h], "train_loss": train_loss,
                                  "test_loss": test_loss, "max_weight_error": max_dev,
                                  "seconds": time.perf_counter() - t0})
        print(f"level {k} ({w}x{h}): train {train_loss:.2e}  test {test_loss:.2e}  "
              f"max |w - (1-s, s)| {max_dev:.2e}")

    (OUT_DIR / "train_levels.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
