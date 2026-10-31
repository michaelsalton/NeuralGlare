"""Phase 1, step 5: Unity parity. The same frames and weights through the same Slang source on
both sides: slangpy (Vulkan) here, slangc -> HLSL -> FXC (D3D11) in Unity.

    uv run python experiments/02_slang_plumbing/unity_parity.py export     # inputs for Unity
    Unity: menu NeuralGlare > Run Parity                                   # writes Unity's levels
    uv run python experiments/02_slang_plumbing/unity_parity.py compare    # verdict

Images cross as raw little-endian float32 RGBA, row 0 = texel y = 0, no header: no EXR codec,
no texture importer, no y flip. (Unity's textures are bottom-up in UV space, so an imported EXR
would reach the shader upside down, and the MLP, whose input is uv, would see a mirrored screen.)

Pass criterion, fixed before the first Unity run (2026-10-31): on every frame's up[0], every
channel of every texel satisfies |U - P| <= RTOL * |P| + ATOL_REL * max(P). Every mip and
upsample level is reported as a diagnostic; only up[0] decides.
"""

import argparse
import json
import tomllib

import numpy as np
import slangpy as spy
import torch
from train_levels import OUT_DIR, ROOT, VIEWS, environment
from train_pyramid import PARAM_COUNT, slang_mips

from neuralglare.baselines.pyramid_bloom import URPBloomParams
from neuralglare.io import read_hdr

PARITY_DIR = OUT_DIR / "unity_parity"
WEIGHTS = OUT_DIR / "glare_params.bytes"
P = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)  # as train_pyramid.py
RTOL = 1e-4             # test_glare_chain.py's Slang-vs-NumPy tolerance (float32 vs float64)
ATOL_REL = 1e-7         # ~float32 ulp of the brightest texel, inherited by dark neighbours


def frames():
    test = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase1"]["test"]
    return [f"{scene}_{view}" for scene in test for view in VIEWS]


def write_rgba(path, rgb):
    rgba = np.concatenate([rgb, np.ones_like(rgb[..., :1])], -1)
    rgba.astype("<f4").tofile(path)


def read_rgba(path, h, w):
    data = np.fromfile(path, "<f4")
    assert data.size == h * w * 4, f"{path.name}: {data.size} floats, expected {h}x{w}x4"
    return data.reshape(h, w, 4)[..., :3]


def python_pyramid(module, img, params):
    """Every mip and every upsample level (ups[n] = mips[n]), as numpy arrays."""
    mips = slang_mips(module, img, P)
    ups = [None] * len(mips)
    ups[-1] = mips[-1]
    for k in reversed(range(len(mips) - 1)):
        h, w = mips[k].shape[:2]
        ups[k] = module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                            mip=mips[k], low=ups[k + 1], params=params,
                                            offset=k * PARAM_COUNT)
    return [m.cpu().numpy() for m in mips], [u.cpu().numpy() for u in ups]


def stats(unity, python):
    err = np.abs(unity - python)
    bound = RTOL * np.abs(python) + ATOL_REL * python.max()
    bright = python > ATOL_REL / RTOL * python.max()          # where RTOL is the binding term
    return {"max_bound_ratio": float((err / bound).max()),   # <= 1 passes
            "max_rel_error_bright": float((err[bright] / python[bright]).max()),
            "energy_ratio_minus_1": (unity.sum((0, 1)) / python.sum((0, 1)) - 1).tolist()}


def export():
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in frames():
        img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
        write_rgba(PARITY_DIR / f"{name}_input.f32", img)
        manifest.append({"name": name, "width": img.shape[1], "height": img.shape[0]})
    (PARITY_DIR / "frames.json").write_text(json.dumps({"frames": manifest}, indent=2))
    print(f"wrote {len(manifest)} frames to {PARITY_DIR}")


def compare():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    module = spy.Module.load_from_file(device, "train_glare.slang")
    # Read the engine weight file, not the .npy, so the shared format is tested too
    params = torch.from_numpy(np.fromfile(WEIGHTS, "<f4")).cuda()

    results, passed = {}, True
    with torch.no_grad():
        for name in frames():
            img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
            mips, ups = python_pyramid(module, img, params)
            levels = {}
            for kind, arrays in (("mip", mips), ("up", ups[:-1])):
                for k, ref in enumerate(arrays):
                    got = read_rgba(PARITY_DIR / f"{name}_unity_{kind}{k}.f32", *ref.shape[:2])
                    levels[f"{kind}{k}"] = stats(got, ref)
            frame_pass = levels["up0"]["max_bound_ratio"] <= 1.0
            passed &= frame_pass
            results[name] = {"pass": frame_pass, "levels": levels}
            print(f"{name}: up0 bound ratio {levels['up0']['max_bound_ratio']:.3g}  "
                  f"=> {'PASS' if frame_pass else 'FAIL'}")
            for key, s in levels.items():
                print(f"    {key:5s} ratio {s['max_bound_ratio']:.3g}  "
                      f"rel(bright) {s['max_rel_error_bright']:.2e}")

    unity = json.loads((PARITY_DIR / "unity_run.json").read_text())
    (PARITY_DIR / "unity_parity.json").write_text(json.dumps({
        "environment": environment(device), "unity": unity,
        "criteria": {"rtol": RTOL, "atol_rel": ATOL_REL}, "pass": passed,
        "frames": results}, indent=2))
    print(f"\nverdict: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "compare"))
    {"export": export, "compare": compare}[ap.parse_args().mode]()
