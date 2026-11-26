"""Phase 1, step 5: Unity parity. The same frames and weights through the same Slang source on
both sides: slangpy (Vulkan) here, slangc -> HLSL -> FXC (D3D11) in Unity.

    uv run python experiments/02_slang_plumbing/unity_parity.py export     # inputs for Unity
    Unity: menu NeuralGlare > Run Parity                                   # writes Unity's levels
    uv run python experiments/02_slang_plumbing/unity_parity.py compare    # verdict
    uv run python experiments/02_slang_plumbing/unity_parity.py score      # FLIP of what ships

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
Weights with the per-ray cross filter (decision log 2026-11-19) carry it after the spike section
(crossParamCount in the layout), and both sides run it on every ray. export --mips (default
train_points.MIPS, 1) is the pyramid's mip count the weights were trained with (the run's
_mips<n>); it goes into UNITY_LAYOUT as `mips`, and compare and ParityHarness both read it there
(layouts without it are LEGACY_MIPS, Phase 1's 6).

Images cross as raw little-endian float32 RGBA, row 0 = texel y = 0, no header: no EXR codec,
no texture importer, no y flip. (Unity's textures are bottom-up in UV space, so an imported EXR
would reach the shader upside down, and the MLP, whose input is uv, would see a mirrored screen.)

Pass criterion, fixed before the first Unity run (2026-10-31): on every frame's up[0], every
channel of every texel satisfies |U - P| <= RTOL * |P| + ATOL_REL * max(P). Every mip and
upsample level is reported as a diagnostic; only up[0] decides. With a full-resolution level
(decision log 2026-11-03), the same test on its output decides instead: it is the frame the
engine shows.

Since efficiency step 1e (decision log 2026-11-24) the engine filters in hardware (one
SampleLevel per bilinear tap) and Python by hand, so they can't agree to float32. compare is then
the bug check, at criteria set from the 2026-11-24 measurement and fixed before they decided:
every level of every frame within RTOL_HW / ATOL_REL_HW, and every channel's energy within
ENERGY_TOL of Python's. The float32 ratio is still printed, for reference. Accuracy is measured on
the engine's own output: export --points adds the validation frame and the test point grid
(train_points.eval_frames) to the manifest, and score scores Unity's output and Python's on them
with train_points.window_records, paired per source. Gate: Unity's validation FLIP within
DELTA_COMA of the trained run's.
"""

import argparse
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import slangpy as spy
import torch
from train_levels import (
    OUT_DIR,
    ROOT,
    VIEWS,
    bright_mlp,
    engine_layout,
    environment,
    level_mlp,
    ray_mlp,
    spike_layout,
)
from train_pyramid import slang_mips

from neuralglare.io import read_hdr

sys.path.insert(0, str(ROOT / "experiments" / "03_train_network"))
# the spike term's pass order and the pyramid, as training runs them
from train_points import (
    MIPS,
    eval_frames,
    load_bank,
    pyramid,
    spikes,
    strongest_baselines,
    summarize,
    window_records,
)

PARITY_DIR = OUT_DIR / "unity_parity"
WEIGHTS = (ROOT / "outputs" / "03_train_network" / "default" / "ray_cross" / "export" /
           "glare_params_cross_abs_fitted_lr0.001_ee0.02_mips1.bytes")   # current best (2026-11-19)
UNITY_WEIGHTS = (ROOT / "engines" / "unity" / "NeuralGlareURP" / "Assets" / "NeuralGlare" /
                 "Data" / "glare_params.bytes")              # ParityHarness.WeightsPath
UNITY_LAYOUT = UNITY_WEIGHTS.with_suffix(".json")             # ParityHarness.LayoutPath
LEGACY_MIPS = 6         # layouts written before the mip count was recorded (Phase 1, runs before 2026-11-10)
RTOL = 1e-4             # test_glare_chain.py's Slang-vs-NumPy tolerance (float32 vs float64)
ATOL_REL = 1e-7         # ~float32 ulp of the brightest texel, inherited by dark neighbours
# Hardware filtering (efficiency step 1e, decision log 2026-11-24): 50x the float32 criteria, ~2x
# the measured worst relative error (2.6e-3, the ray families); energy per channel: measured
# <= 1.4e-5, and fp16 truncation's 1e-3 bias would have failed it
RTOL_HW = 5e-3
ATOL_REL_HW = 5e-6
ENERGY_TOL = 1e-4
DELTA_COMA = 2.18e-4    # the smallest validation-FLIP change counted as real (decision log, delta_coma)
POINT_FRAMES = ("val", "test")   # train_points.eval_frames; parity frames points_val, points_test


def load_module():
    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    return device, spy.Module.load_from_file(device, "train_glare.slang")


def levels_of(module, n_weights):
    """(LevelMLP networks, has the spike section, has the cross filter, has per-ray brightness) of
    a weight file of n_weights floats."""
    count = level_mlp(module).param_count
    spike, cross = spike_layout(module).param_count, ray_mlp(module).param_count
    bright = bright_mlp(module).param_count
    fits = []
    for s, c, b in ((False, False, False), (True, False, False), (True, True, False),
                    (True, True, True)):
        extra = s * spike + c * cross + b * bright
        if n_weights > extra and (n_weights - extra) % count == 0:
            fits.append(((n_weights - extra) // count, s, c, b))
    assert len(fits) == 1, (f"{n_weights} weights: not whole levels of {count} "
                            f"(+ {spike} spikes, + {cross} cross, + {bright} brightness)")
    return fits[0]


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


def python_pyramid(module, img, params, source="mip0", mips_count=MIPS, store="fp32",
                   pre_exposure=1.0, stack=False):
    """Every mip and every upsample level (ups[n] = mips[n]) of a mips_count-mip pyramid, as numpy
    arrays, the full-resolution level's output if the weights have one (None otherwise), and the
    spike term's {"even", "odd", "spikes"} if they have it (empty otherwise). store: the format
    Unity keeps the ray intermediates in (the layout's rayStorage); pre_exposure: the rays' k Unity
    ran with (unity_run.json); stack: Unity ran the ray stack (unity_run.json rayStack, efficiency
    step 1c), so there are no family sums to compare, only {"spikes"}."""
    mips = slang_mips(module, img, pyramid(mips_count))
    n = len(mips) - 1
    levels, has_spikes, has_cross, has_bright = levels_of(module, params.numel())
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
            out, fams = spikes(module, rays_from, full, params,
                               levels * level_mlp(module).param_count, families=True,
                               cross=has_cross, bright=has_bright, store=store,
                               pre_exposure=pre_exposure, stack=stack)
            spike_levels = {"spikes": out.cpu().numpy()} if stack else {
                "even": fams[0].cpu().numpy(), "odd": fams[1].cpu().numpy(),
                "spikes": out.cpu().numpy()}
        full = full.cpu().numpy()
    assert not has_spikes or spike_levels, "spike weights without the full-resolution level"
    return [m.cpu().numpy() for m in mips], [u.cpu().numpy() for u in ups], full, spike_levels


def stats(unity, python):
    err = np.abs(unity - python)
    bound = RTOL_HW * np.abs(python) + ATOL_REL_HW * python.max()
    bound32 = RTOL * np.abs(python) + ATOL_REL * python.max()
    bright = python > ATOL_REL / RTOL * python.max()          # where RTOL is the binding term
    energy = unity.sum((0, 1)) / python.sum((0, 1)) - 1
    return {"max_bound_ratio": float((err / bound).max()),   # <= 1 passes (hardware criteria)
            "max_bound_ratio_fp32": float((err / bound32).max()),   # the float32 criteria, reported
            "max_rel_error_bright": float((err[bright] / python[bright]).max()),
            "energy_ratio_minus_1": energy.tolist(),
            "pass": bool((err <= bound).all() and (np.abs(energy) <= ENERGY_TOL).all())}


def export(weights, mips, points=False, run="default"):
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    _, module = load_module()
    sidecar = Path(weights).with_suffix(".json")
    net, lay = level_mlp(module), spike_layout(module)
    levels, has_spikes, has_cross, has_bright = levels_of(module, Path(weights).stat().st_size // 4)
    if sidecar.exists():                    # written by train_points.py; must match glare.slang
        layout = json.loads(sidecar.read_text())
        assert layout == engine_layout(module, levels, has_spikes, layout["spikeSource"] or None,
                                       has_cross, has_bright, layout.get("rayStorage", "fp32")), (
            f"{sidecar} doesn't match glare.slang and the weight file")
    else:                                   # weights from before the layout file: rays from mip 0
        layout = engine_layout(module, levels, has_spikes, "mip0" if has_spikes else None,
                               has_cross, has_bright)
    layout["mips"] = mips                   # Unity's pyramid (ParityHarness MipSizes); the copy only
    shutil.copyfile(weights, UNITY_WEIGHTS)
    UNITY_LAYOUT.write_text(json.dumps(layout, indent=2))
    manifest = []
    for name in frames():
        img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
        write_rgba(PARITY_DIR / f"{name}_input.f32", img)
        manifest.append({"name": name, "width": img.shape[1], "height": img.shape[0]})
    if points:                              # the frames runs are scored on (score)
        split, bank, shape = load_bank(run)
        for key, (img, _, _) in eval_frames(run, split, bank, shape).items():
            write_rgba(PARITY_DIR / f"points_{key}_input.f32", img)
            manifest.append({"name": f"points_{key}", "width": img.shape[1], "height": img.shape[0]})
    (PARITY_DIR / "frames.json").write_text(json.dumps(
        {"frames": manifest, "weights": Path(weights).resolve().relative_to(ROOT).as_posix(),
         "run": run},
        indent=2))
    print(f"wrote {len(manifest)} frames to {PARITY_DIR}; copied {weights} to {UNITY_WEIGHTS} "
          f"({levels} levels of {net.param_count}"
          f"{f' + {lay.param_count} spike, rays from ' + layout['spikeSource'] if has_spikes else ''}"
          f"{f' + {ray_mlp(module).param_count} cross' if has_cross else ''}"
          f"{f' + {bright_mlp(module).param_count} brightness' if has_bright else ''}"
          f", {mips} mips) and its layout to {UNITY_LAYOUT.name}")


def unity_inputs(module):
    """What Unity ran, as compare and score need it: (layout, params, weights source, ray
    storage, pre-exposure, unity_run.json)."""
    layout = json.loads(UNITY_LAYOUT.read_text())
    assert layout["paramCount"] == level_mlp(module).param_count, (
        f"Unity ran {layout['paramCount']}-float levels, glare.slang now has "
        f"{level_mlp(module).param_count}: rerun export, build_hlsl.sh and Unity")
    # Read the engine weight file Unity loaded, not the .npy, so the shared format is tested too
    params = torch.from_numpy(np.fromfile(UNITY_WEIGHTS, "<f4")).cuda()
    source = json.loads((PARITY_DIR / "frames.json").read_text())["weights"]
    unity = json.loads((PARITY_DIR / "unity_run.json").read_text())
    return (layout, params, source, layout.get("rayStorage", "fp32"),
            unity.get("preExposure", 1.0), unity)    # runs before 2026-11-24 had no k: 1


def compare(*_):
    device, module = load_module()
    layout, params, source, store, pre_exposure, unity = unity_inputs(module)
    mips_count = layout.get("mips", LEGACY_MIPS)

    results, passed = {}, True
    with torch.no_grad():
        for name in frames():
            img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))[..., :3]
            mips, ups, full, spike_levels = python_pyramid(module, img, params,
                                                           layout.get("spikeSource") or "mip0",
                                                           mips_count, store, pre_exposure,
                                                           unity.get("rayStack", False))
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
            frame_pass = all(s["pass"] for s in levels.values())      # every level (1e criteria)
            passed &= frame_pass
            results[name] = {"pass": frame_pass, "decides": decides, "levels": levels}
            print(f"{name}: {decides} bound ratio {levels[decides]['max_bound_ratio']:.3g}  "
                  f"=> {'PASS' if frame_pass else 'FAIL'}")
            for key, s in levels.items():
                print(f"    {key:6s} ratio {s['max_bound_ratio']:.3g} (fp32 "
                      f"{s['max_bound_ratio_fp32']:.3g})  rel(bright) "
                      f"{s['max_rel_error_bright']:.2e}  energy "
                      f"{max(map(abs, s['energy_ratio_minus_1'])):.1e}"
                      f"{'' if s['pass'] else '  FAIL'}")

    (PARITY_DIR / "unity_parity.json").write_text(json.dumps({
        "environment": environment(device), "unity": unity, "weights": source,
        "criteria": {"rtol": RTOL_HW, "atol_rel": ATOL_REL_HW, "energy": ENERGY_TOL,
                     "fp32_reported": {"rtol": RTOL, "atol_rel": ATOL_REL}},
        "pass": passed, "frames": results}, indent=2))
    print(f"\nverdict: {'PASS' if passed else 'FAIL'}")


def trained_eval(source):
    """The eval json train_points.py wrote for the float32 fit of these engine weights (the
    weights' tag without its _ray<storage> suffix), or None."""
    weights = ROOT / source
    tag = re.sub(r"_ray[a-z0-9]+$", "", weights.stem[len("glare_params"):])
    path = weights.parent.parent / "eval" / f"train_points{tag}_eval.json"
    return json.loads(path.read_text()) if path.exists() else None


def score(*_):
    """Window FLIP of Unity's output and Python's on the point-grid frames (export --points),
    scored as train_points.py scores every run, paired per source (efficiency step 1e)."""
    device, module = load_module()
    layout, params, source, store, pre_exposure, unity = unity_inputs(module)
    run = json.loads((PARITY_DIR / "frames.json").read_text()).get("run", "default")
    split, bank, shape = load_bank(run)
    _, _, gap = strongest_baselines(run)
    size, ppd = gap["window_px"], gap["flip_ppd"]
    scored = {}
    with torch.no_grad():
        for key, (img, ref, positions) in eval_frames(run, split, bank, shape).items():
            path = PARITY_DIR / f"points_{key}_unity_spikes.f32"
            assert path.exists(), f"{path.name} missing: export --points, then Run Parity in Unity"
            _, _, _, sl = python_pyramid(module, img, params, layout.get("spikeSource") or "mip0",
                                         layout.get("mips", LEGACY_MIPS), store, pre_exposure,
                                         unity.get("rayStack", False))
            outputs = {"unity": read_rgba(path, *img.shape[:2]), "python": sl["spikes"]}
            scored[key] = {side: window_records(out, ref, positions, size, ppd)
                           for side, out in outputs.items()}
    val = {side: float(np.mean([r["flip"] for r in recs])) for side, recs in scored["val"].items()}
    test = {side: summarize(recs) for side, recs in scored["test"].items()}
    diffs = np.array([u["flip"] - p["flip"] for key in POINT_FRAMES
                      for u, p in zip(scored[key]["unity"], scored[key]["python"])])
    trained = trained_eval(source)
    reference = trained["results"]["trained"]["val"]["flip"] if trained else None
    gate = None if reference is None else val["unity"] - reference <= DELTA_COMA
    print(f"validation FLIP  Unity {val['unity']:.6f}  Python {val['python']:.6f}  "
          f"(Unity - Python {val['unity'] - val['python']:+.2e})")
    if reference is not None:
        print(f"                 trained run (Python, float32 rays) {reference:.6f}: Unity "
              f"{val['unity'] - reference:+.2e}, delta_coma {DELTA_COMA:g} => "
              f"{'PASS' if gate else 'FAIL'}")
    for side in ("unity", "python"):
        print(f"test FLIP {side:6s} " + " / ".join(f"{test[side][b]['flip']:.5f}"
                                                  for b in ("centre", "mid", "edge")))
    print(f"per source (Unity - Python, {diffs.size} sources): mean {diffs.mean():+.2e}, "
          f"max |.| {np.abs(diffs).max():.2e}")
    (PARITY_DIR / "unity_points_eval.json").write_text(json.dumps({
        "environment": environment(device), "unity": unity, "weights": source, "run": run,
        "val": val, "test": test, "trained_val_flip": reference, "delta_coma": DELTA_COMA,
        "pass": gate, "per_source_diff": {"mean": float(diffs.mean()),
                                          "max_abs": float(np.abs(diffs).max())},
        "records": scored}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "compare", "score"))
    ap.add_argument("--weights", default=WEIGHTS, help="export: engine weight file to run")
    ap.add_argument("--mips", type=int, default=MIPS,
                    help="export: the pyramid's mip count the weights were trained with (the "
                         "run's _mips<n>; 6 for runs before 2026-11-10)")
    ap.add_argument("--points", action="store_true",
                    help="export: also the validation frame and the test point grid, for score")
    ap.add_argument("--run", default="default", help="export --points: the run's PSF bank")
    args = ap.parse_args()
    if args.mode == "export":
        export(args.weights, args.mips, args.points, args.run)
    else:
        {"compare": compare, "score": score}[args.mode]()
