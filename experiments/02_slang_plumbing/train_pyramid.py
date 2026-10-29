"""Phase 1, step 4: train the whole glare pyramid end to end against URP bloom.

The full pyramid runs in Slang: the downsample chain (pyramid.slang, fixed) and the upsample
chain (glare.slang), where every level k reads the *trained* output of the level below,
    out_k = w0_k(uv) * mip_k + w1_k(uv) * bilinear(out_{k+1}),     out_n = mip_n (smallest),
and only the final out_0 is supervised, against URP's up[0] of the same mips. The gradient
of that one loss reaches every level through the chain (tests/test_glare_chain.py). The exact
answer is still w = (1 - s, s) at every level, so per-level weight errors show what end-to-end
supervision pins down, compared with step 3's teacher forcing.

All levels share one flat parameter buffer, level k at offset k * PARAM_COUNT: the weight file
format the engines load.

Pass criteria, fixed before the first run (2026-10-29): test loss on up[0] <= PASS_TEST_LOSS,
and every level's max |w - (1 - s, s)| <= PASS_WEIGHT_ERROR.

Post hoc (added after the first run, never part of the verdict): mip k reaches out_0 through the
product c_k = w0_k * w1_0 * ... * w1_{k-1} (c_n = w1_0 * ... * w1_{n-1}), and supervising only
out_0 pins down these n + 1 coefficients, not the 2n weights. The first run matched URP's
c_k = (1 - s) s^k to ~1 % with individual weights off by up to 0.40, so the script also reports
the coefficient error per mip.

slangpy 0.43.1 decides from the first call of an entry point whether it is differentiable, so
the first upsample_chain_pass call must pass params with requires_grad (training comes first;
chain() asserts the autograd history is there).

    uv run python experiments/02_slang_plumbing/train_pyramid.py
"""

import argparse
import json
import time
import tomllib

import numpy as np
import slangpy as spy
import torch
from train_levels import (
    HIDDEN,
    IN,
    OUT,
    OUT_DIR,
    ROOT,
    VIEWS,
    environment,
    init_params,
    relative_loss,
    upsample_targets,
)

from neuralglare.baselines.pyramid_bloom import URPBloomParams, mip_sizes, shader_constants
from neuralglare.io import read_hdr

PARAM_COUNT = (IN * HIDDEN + HIDDEN) + (HIDDEN * OUT + OUT)     # LevelMLP.PARAM_COUNT
PASS_TEST_LOSS = 1e-6           # step 3's per-level test losses: 4.5e-8 .. 6.1e-7
PASS_WEIGHT_ERROR = 0.02        # step 3: 0.004 .. 0.009
LOG_EVERY = 250


def slang_mips(module, img, p):
    """The downsample chain in Slang: mip 0 = prefilter, mip i = BlurV(BlurH(mip i-1))."""
    h0, w0 = img.shape[:2]
    sizes = mip_sizes(h0, w0, p)
    src = torch.from_numpy(np.ascontiguousarray(img[..., :3], np.float32)).cuda()
    h, w = sizes[0]
    mips = [module.prefilter_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h), src=src)]
    for h, w in sizes[1:]:
        blurred = module.blur_h_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                     src=mips[-1])
        mips.append(module.blur_v_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                       src=blurred))
    return mips


def load_split(module, split, p, scatter):
    """[(mips, up0 target)] for every view of every scene in the split. The target is URP's
    upsample chain on the same Slang mips, so an exact answer exists."""
    items = []
    for scene in split:
        for view in VIEWS:
            img = read_hdr(str(ROOT / "data" / "inputs" / f"{scene}_{view}.exr"))
            mips = slang_mips(module, img, p)
            up0 = upsample_targets([m.cpu().numpy() for m in mips], scatter)[0]
            items.append((mips, torch.from_numpy(up0).cuda()))
    return items


def chain(module, mips, params):
    """The upsample chain from the smallest mip to out_0."""
    out = mips[-1]
    for k in reversed(range(len(mips) - 1)):
        h, w = mips[k].shape[:2]
        out = module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                         mip=mips[k], low=out, params=params,
                                         offset=k * PARAM_COUNT)
        if torch.is_grad_enabled() and params.requires_grad:
            assert out.grad_fn is not None, "slangpy dropped autograd (see module docstring)"
    return out


def weight_errors(module, params, n_levels, grid_uv, exact):
    """Per level, max |w(uv) - (1 - s, s)| over the screen grid."""
    with torch.no_grad():
        return [(module.filter_weights(uv=grid_uv,
                                       params=params[k * PARAM_COUNT:(k + 1) * PARAM_COUNT]
                                       .contiguous()) - exact).abs().max().item()
                for k in range(n_levels)]


def coefficient_errors(module, params, n_levels, grid_uv, scatter):
    """Post hoc: per mip, max over the screen grid of |c_k(uv) - c_k(URP)|, where c_k is the
    product of weights along mip k's path to out_0. Weights vary slowly over the screen, so
    evaluating every level at the same uv is a close approximation of the chained bilinear path."""
    with torch.no_grad():
        w = [module.filter_weights(uv=grid_uv,
                                   params=params[k * PARAM_COUNT:(k + 1) * PARAM_COUNT]
                                   .contiguous()) for k in range(n_levels)]
        errors, low_path = [], torch.ones_like(w[0][:, 0])
        for k in range(n_levels + 1):
            c = low_path * w[k][:, 0] if k < n_levels else low_path
            urp = (1 - scatter) * scatter ** k if k < n_levels else scatter ** n_levels
            errors.append((c - urp).abs().max().item())
            if k < n_levels:
                low_path = low_path * w[k][:, 1]
        return errors


def mean_loss(module, items, params):
    with torch.no_grad():
        return float(np.mean([relative_loss(chain(module, mips, params), up0).item()
                              for mips, up0 in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
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
    data = {name: load_split(module, split[name], p, scatter) for name in ("train", "test")}
    n_levels = len(data["train"][0][0]) - 1          # the smallest mip has no upsample pass

    gu, gv = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 9))
    grid_uv = torch.tensor(np.stack([gu.ravel(), gv.ravel()], -1), dtype=torch.float32,
                           device="cuda")
    exact = torch.tensor([1 - scatter, scatter], device="cuda")

    params = torch.cat([init_params(gen) for _ in range(n_levels)]).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    curve = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        mips, up0 = data["train"][rng.integers(len(data["train"]))]   # one random frame
        optimizer.zero_grad()
        loss = relative_loss(chain(module, mips, params), up0)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % LOG_EVERY == 0 or step == args.steps:
            errors = weight_errors(module, params, n_levels, grid_uv, exact)
            curve.append({"step": step, "loss": loss.item(), "max_weight_error": errors})
            print(f"step {step:5d}: loss {loss.item():.2e}  max |w - (1-s, s)| per level "
                  + " ".join(f"{e:.3f}" for e in errors))
    seconds = time.perf_counter() - t0

    train_loss = mean_loss(module, data["train"], params)
    test_loss = mean_loss(module, data["test"], params)
    errors = weight_errors(module, params, n_levels, grid_uv, exact)
    passed = test_loss <= PASS_TEST_LOSS and max(errors) <= PASS_WEIGHT_ERROR
    coefficients = coefficient_errors(module, params, n_levels, grid_uv, scatter)
    sizes = [list(m.shape[1::-1]) for m in data["train"][0][0][:n_levels]]
    print(f"\ntrain {train_loss:.2e}  test {test_loss:.2e} (pass <= {PASS_TEST_LOSS:.0e})  "
          f"max weight error {max(errors):.3f} (pass <= {PASS_WEIGHT_ERROR})  "
          f"=> {'PASS' if passed else 'FAIL'}  ({seconds:.0f} s)")
    print("post hoc: max |c_k - c_k(URP)| per mip " + " ".join(f"{e:.4f}" for e in coefficients))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "train_pyramid_params.npy", params.detach().cpu().numpy())
    (OUT_DIR / "train_pyramid.json").write_text(json.dumps({
        "environment": environment(device), "scatter": scatter, "steps": args.steps, "lr": args.lr, "param_count": PARAM_COUNT,
        "pass_criteria": {"test_loss": PASS_TEST_LOSS, "max_weight_error": PASS_WEIGHT_ERROR},
        "train_loss": train_loss, "test_loss": test_loss, "pass": passed, "seconds": seconds,
        "levels": [{"level": k, "size": sizes[k], "max_weight_error": errors[k]}
                   for k in range(n_levels)],
        "post_hoc_max_coefficient_error": coefficients, "curve": curve}, indent=2))


if __name__ == "__main__":
    main()
