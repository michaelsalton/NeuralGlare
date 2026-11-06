"""Phase 1, step 5: Unity parity. The same frames and weights through the same Slang source on
both sides: slangpy (Vulkan) here, slangc -> HLSL -> FXC (D3D11) in Unity.

    uv run python experiments/02_slang_plumbing/unity_parity.py export     # inputs for Unity
    Unity: menu NeuralGlare > Run Parity                                   # writes Unity's levels
    uv run python experiments/02_slang_plumbing/unity_parity.py compare    # verdict

export --weights FILE copies an engine weight file into the Unity project (UNITY_WEIGHTS, the
file ParityHarness loads) and records its source in frames.json; compare reads that same copy,
so both sides run the same bytes. Default: the selected Phase 2 step 1 model. Weights with one
level more than the pyramid (Phase 2) add the full-resolution level, whose "mip" is the input
frame (train_points.py glare()). Next to the copy, export writes UNITY_LAYOUT: LevelMLP's shape as
glare.slang declares it (level_mlp), so ParityHarness offsets the buffer by the network the HLSL
was compiled from instead of a number typed into C#. Weights with the spike section (decision log
2026-11-04) add its sizes to the layout (spike_layout), and Unity runs the spike term after the
full-resolution level; its output (the frame the engine shows) then decides, and the two ray
family sums are reported as diagnostics. The layout is the one train_points.py wrote next to the
weight file (glare_params<tag>.json, with the image the rays are streaked from, spikeSource),
checked against glare.slang; older weights without one get it built here (rays from mip 0).

Images cross as raw little-endian float32 RGBA, row 0 = texel y = 0, no header: no EXR codec,
no texture importer, no y flip. (Unity's textures are bottom-up in UV space, so an imported EXR
would reach the shader upside down, and the MLP, whose input is uv, would see a mirrored screen.)

Pass criterion, fixed before the first Unity run (2026-10-31): on every frame's up[0], every
channel of every texel satisfies |U - P| <= RTOL * |P| + ATOL_REL * max(P). Every mip and
upsample level is reported as a diagnostic; only up[0] decides. With a full-resolution level
(decision log 2026-11-03), the same test on its output decides instead: it is the frame the
engine shows.
"""

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import slangpy as spy
import torch
from train_levels import OUT_DIR, ROOT, VIEWS, engine_layout, environment, level_mlp, spike_layout
from train_pyramid import slang_mips

from neuralglare.baselines.pyramid_bloom import URPBloomParams
from neuralglare.io import read_hdr

sys.path.insert(0, str(ROOT / "experiments" / "03_train_network"))
from train_points import spikes  # the spike term's pass order, as training runs it

PARITY_DIR = OUT_DIR / "unity_parity"
WEIGHTS = (ROOT / "outputs" / "03_train_network" / "default" / "spikes_full" / "export" /
           "glare_params_spikes_full_abs_fitted_lr0.0001.bytes")   # current best (2026-11-05); the
                                                              # 2-tap baseline no longer runs on glare.slang
UNITY_WEIGHTS = (ROOT / "engines" / "unity" / "NeuralGlareURP" / "Assets" / "NeuralGlare" /
                 "Data" / "glare_params.bytes")              # ParityHarness.WeightsPath
UNITY_LAYOUT = UNITY_WEIGHTS.with_suffix(".json")             # ParityHarness.LayoutPath
P = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)  # as train_pyramid.py
RTOL = 1e-4             # test_glare_chain.py's Slang-vs-NumPy tolerance (float32 vs float64)
ATOL_REL = 1e-7         # ~float32 ulp of the brightest texel, inherited by dark neighbours


def load_module():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    return device, spy.Module.load_from_file(device, "train_glare.slang")


def levels_of(module, n_weights):
    """(LevelMLP networks, has the spike section) of a weight file of n_weights floats."""
    count, extra = level_mlp(module).param_count, spike_layout(module).param_count
    fits = [spikes for spikes in (False, True)
            if n_weights > spikes * extra and (n_weights - spikes * extra) % count == 0]
    assert len(fits) == 1, f"{n_weights} weights: not whole levels of {count} (+ {extra} spikes)"
    return (n_weights - fits[0] * extra) // count, fits[0]


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


def python_pyramid(module, img, params, source="mip0"):
    """Every mip and every upsample level (ups[n] = mips[n]), as numpy arrays, the
    full-resolution level's output if the weights have one (None otherwise), and the spike term's
    {"even", "odd", "spikes"} if they have it (empty otherwise)."""
    mips = slang_mips(module, img, P)
    n = len(mips) - 1
    levels, has_spikes = levels_of(module, params.numel())
    assert levels in (n, n + 1), f"{levels} levels of weights, the pyramid has {n} (+ 1)"

    def level(mip, low, k):
        h, w = mip.shape[:2]
        return module.upsample_chain_pass(texel=spy.grid((h, w)), out_size=spy.int2(w, h),
                                          mip=mip, low=low, params=params,
                                          offset=k * level_mlp(module).param_count)

    ups = [None] * len(mips)
    ups[-1] = mips[-1]
    for k in reversed(range(n)):
        ups[k] = level(mips[k], ups[k + 1], k)
    full, spike_levels = None, {}
    if levels == n + 1:
        src = torch.from_numpy(np.ascontiguousarray(img, np.float32)).cuda()
        full = level(src, ups[0], n)
        if has_spikes:
            rays_from = {"mip0": mips[0], "input": src}[source]
            out, (even, odd) = spikes(module, rays_from, full, params,
                                      levels * level_mlp(module).param_count, families=True)
            spike_levels = {"even": even.cpu().numpy(), "odd": odd.cpu().numpy(),
                            "spikes": out.cpu().numpy()}
        full = full.cpu().numpy()
    assert not has_spikes or spike_levels, "spike weights without the full-resolution level"
    return [m.cpu().numpy() for m in mips], [u.cpu().numpy() for u in ups], full, spike_levels


def stats(unity, python):
    err = np.abs(unity - python)
    bound = RTOL * np.abs(python) + ATOL_REL * python.max()
    bright = python > ATOL_REL / RTOL * python.max()          # where RTOL is the binding term
    return {"max_bound_ratio": float((err / bound).max()),   # <= 1 passes
            "max_rel_error_bright": float((err[bright] / python[bright]).max()),
            "energy_ratio_minus_1": (unity.sum((0, 1)) / python.sum((0, 1)) - 1).tolist()}


def export(weights):
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    _, module = load_module()
    net, lay = level_mlp(module), spike_layout(module)
    levels, has_spikes = levels_of(module, Path(weights).stat().st_size // 4)
    sidecar = Path(weights).with_suffix(".json")
    if sidecar.exists():                    # written by train_points.py; must match glare.slang
        layout = json.loads(sidecar.read_text())
        assert layout == engine_layout(module, levels, has_spikes, layout["spikeSource"] or None), (
            f"{sidecar} doesn't match glare.slang and the weight file")
    else:                                   # weights from before the layout file: rays from mip 0
        layout = engine_layout(module, levels, has_spikes, "mip0" if has_spikes else None)
    shutil.copyfile(weights, UNITY_WEIGHTS)
    UNITY_LAYOUT.write_text(json.dumps(layout, indent=2))
    manifest = []
    for name in frames():
        img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
        write_rgba(PARITY_DIR / f"{name}_input.f32", img)
        manifest.append({"name": name, "width": img.shape[1], "height": img.shape[0]})
    (PARITY_DIR / "frames.json").write_text(json.dumps(
        {"frames": manifest, "weights": Path(weights).resolve().relative_to(ROOT).as_posix()},
        indent=2))
    print(f"wrote {len(manifest)} frames to {PARITY_DIR}; copied {weights} to {UNITY_WEIGHTS} "
          f"({levels} levels of {net.param_count}"
          f"{f' + {lay.param_count} spike, rays from ' + layout['spikeSource'] if has_spikes else ''}"
          f") and its layout to {UNITY_LAYOUT.name}")


def compare(_weights=None):
    device, module = load_module()
    layout = json.loads(UNITY_LAYOUT.read_text())
    assert layout["paramCount"] == level_mlp(module).param_count, (
        f"Unity ran {layout['paramCount']}-float levels, glare.slang now has "
        f"{level_mlp(module).param_count}: rerun export, build_hlsl.sh and Unity")
    # Read the engine weight file Unity loaded, not the .npy, so the shared format is tested too
    params = torch.from_numpy(np.fromfile(UNITY_WEIGHTS, "<f4")).cuda()
    source = json.loads((PARITY_DIR / "frames.json").read_text())["weights"]

    results, passed = {}, True
    with torch.no_grad():
        for name in frames():
            img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
            mips, ups, full, spike_levels = python_pyramid(module, img, params,
                                                           layout.get("spikeSource") or "mip0")
            levels = {}
            for kind, arrays in (("mip", mips), ("up", ups[:-1])):
                for k, ref in enumerate(arrays):
                    got = read_rgba(PARITY_DIR / f"{name}_unity_{kind}{k}.f32", *ref.shape[:2])
                    levels[f"{kind}{k}"] = stats(got, ref)
            if full is not None:
                got = read_rgba(PARITY_DIR / f"{name}_unity_full.f32", *full.shape[:2])
                levels["full"] = stats(got, full)
            for key, ref in spike_levels.items():
                got = read_rgba(PARITY_DIR / f"{name}_unity_{key}.f32", *ref.shape[:2])
                levels[key] = stats(got, ref)
            decides = "spikes" if spike_levels else "full" if full is not None else "up0"
            frame_pass = levels[decides]["max_bound_ratio"] <= 1.0
            passed &= frame_pass
            results[name] = {"pass": frame_pass, "decides": decides, "levels": levels}
            print(f"{name}: {decides} bound ratio {levels[decides]['max_bound_ratio']:.3g}  "
                  f"=> {'PASS' if frame_pass else 'FAIL'}")
            for key, s in levels.items():
                print(f"    {key:5s} ratio {s['max_bound_ratio']:.3g}  "
                      f"rel(bright) {s['max_rel_error_bright']:.2e}")

    unity = json.loads((PARITY_DIR / "unity_run.json").read_text())
    (PARITY_DIR / "unity_parity.json").write_text(json.dumps({
        "environment": environment(device), "unity": unity, "weights": source,
        "criteria": {"rtol": RTOL, "atol_rel": ATOL_REL}, "pass": passed,
        "frames": results}, indent=2))
    print(f"\nverdict: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "compare"))
    ap.add_argument("--weights", default=WEIGHTS, help="export: engine weight file to run")
    args = ap.parse_args()
    {"export": export, "compare": compare}[args.mode](args.weights)
