"""Phase 1, step 4 report: the bloom the end-to-end trained pyramid makes, against URP.

Reads outputs/02_slang_plumbing/train_pyramid_params.npy and train_pyramid.json (run
train_pyramid.py first) and writes outputs/02_slang_plumbing/report/:
    <frame>_input.png, _urp.png, _mlp.png    tonemapped frame + bloom (Reinhard, display only)
    <frame>_bloom_urp.png, _bloom_mlp.png     the bloom alone, same exposure
    <frame>_error.png                         |MLP - URP| / URP on the bloom, log scale
    report.json                               every number the figures and tables show
    report.md                                 the tables

Frames: the three test views (goegap_road_4k, the unseen sun) and one training view.

    uv run python experiments/02_slang_plumbing/report_pyramid.py
"""

import itertools
import json

import cv2
import numpy as np
import slangpy as spy
import torch
from train_levels import EPS, OUT_DIR, ROOT, relative_loss, upsample_targets
from train_pyramid import PARAM_COUNT, chain, slang_mips

from neuralglare.baselines.pyramid_bloom import (
    URPBloomParams,
    sample_bilinear,
    shader_constants,
    texel_uv,
)
from neuralglare.io import display, luminance, read_hdr

REPORT = OUT_DIR / "report"
FRAMES = [("goegap_road_4k_center", "test"), ("goegap_road_4k_mid", "test"),
          ("goegap_road_4k_corner", "test"), ("cobblestone_street_night_4k_center", "train")]
WEIGHT_GRID = (27, 48)              # (rows, cols) of the screen grid for the weight maps
ERROR_RANGE = (1e-6, 1e-1)          # log colour range of the error maps
# sequential blue ramp (dataviz reference palette, steps 100 -> 700): low error light
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def save_png(path, rgb01):
    cv2.imwrite(str(path), (np.clip(rgb01, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def error_image(rel):
    """Map a relative-error image onto the blue ramp, log scale over ERROR_RANGE."""
    lo, hi = np.log10(ERROR_RANGE)
    t = np.clip((np.log10(np.maximum(rel, ERROR_RANGE[0])) - lo) / (hi - lo), 0, 1)
    stops = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in RAMP])
    x = t * (len(RAMP) - 1)
    i = np.minimum(x.astype(int), len(RAMP) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def radial_profile(img, cy, cx, radii):
    """Mean luminance in rings around (cy, cx), one value per radius bin edge pair."""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    lum = luminance(img)
    return [float(lum[(r >= a) & (r < b)].mean()) if np.any((r >= a) & (r < b)) else None
            for a, b in itertools.pairwise(radii)]


def weight_maps(module, params, n_levels):
    gv, gu = np.meshgrid(np.linspace(0, 1, WEIGHT_GRID[0]), np.linspace(0, 1, WEIGHT_GRID[1]),
                         indexing="ij")
    uv = torch.tensor(np.stack([gu.ravel(), gv.ravel()], -1), dtype=torch.float32,
                      device="cuda")
    with torch.no_grad():
        return [module.filter_weights(uv=uv, params=params[k * PARAM_COUNT:(k + 1) *
                                                            PARAM_COUNT].contiguous())
                .cpu().numpy().reshape(*WEIGHT_GRID, 2) for k in range(n_levels)]


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    trained = json.loads((OUT_DIR / "train_pyramid.json").read_text())
    with open(ROOT / "data" / "inputs" / "manifest.json") as fh:
        manifest = json.load(fh)["images"]

    spy.set_allow_torch_fallback(True)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=[ROOT / "slang"])
    module = spy.Module.load_from_file(device, "train_glare.slang")
    p = URPBloomParams(threshold=0.0, clamp_max=np.inf, max_iterations=6)
    _, _, scatter = shader_constants(p)
    params = torch.from_numpy(np.load(OUT_DIR / "train_pyramid_params.npy")).cuda()
    # slangpy fixes differentiability on an entry point's first call: make it with gradients
    params.requires_grad_(True)

    frames = []
    for name, split in FRAMES:
        img = read_hdr(str(ROOT / "data" / "inputs" / f"{name}.exr"))
        mips = slang_mips(module, img, p)
        urp = upsample_targets([m.cpu().numpy() for m in mips], scatter)[0]
        mlp = chain(module, mips, params).detach().cpu().numpy()
        h, w = img.shape[:2]
        full_uv = texel_uv(h, w)                      # bloom upsampled to the input size
        bloom_urp, bloom_mlp = (sample_bilinear(b, *full_uv).astype(np.float32)
                                for b in (urp, mlp))
        rel = np.abs(mlp - urp) / (np.abs(urp) + EPS)

        save_png(REPORT / f"{name}_input.png", display(img))
        save_png(REPORT / f"{name}_urp.png", display(img + bloom_urp))
        save_png(REPORT / f"{name}_mlp.png", display(img + bloom_mlp))
        save_png(REPORT / f"{name}_bloom_urp.png", display(bloom_urp))
        save_png(REPORT / f"{name}_bloom_mlp.png", display(bloom_mlp))
        save_png(REPORT / f"{name}_error.png", error_image(rel.max(-1)))

        lx, ly = manifest[name]["light_px"]           # brightest pixel, input resolution
        sy, sx = urp.shape[0] / h, urp.shape[1] / w
        radii = np.unique(np.round(np.geomspace(1, 400, 28)).astype(int))
        frames.append({
            "name": name, "split": split, "bloom_size": list(urp.shape[1::-1]),
            "relative_loss": float(relative_loss(torch.from_numpy(mlp),
                                                 torch.from_numpy(urp)).item()),
            "energy_ratio_rgb": (mlp.astype(np.float64).sum((0, 1))
                                 / urp.astype(np.float64).sum((0, 1))).tolist(),
            "rel_error_median": float(np.median(rel)),
            "rel_error_p99": float(np.percentile(rel, 99)),
            "rel_error_max": float(rel.max()),
            "peak_luminance": float(luminance(img).max()),
            "profile_radii": radii.tolist(),
            "profile_urp": radial_profile(urp, ly * sy, lx * sx, radii),
            "profile_mlp": radial_profile(mlp, ly * sy, lx * sx, radii)})
        print(f"{name} ({split}): relative loss {frames[-1]['relative_loss']:.2e}, "
              f"energy MLP/URP {np.round(frames[-1]['energy_ratio_rgb'], 5)}, "
              f"p99 rel error {frames[-1]['rel_error_p99']:.2e}")

    n_levels = len(trained["levels"])
    maps = weight_maps(module, params.detach(), n_levels)
    low_path = np.ones(WEIGHT_GRID)
    coeff_maps = []
    for k in range(n_levels + 1):
        coeff_maps.append(low_path * maps[k][..., 0] if k < n_levels else low_path)
        if k < n_levels:
            low_path = low_path * maps[k][..., 1]
    urp_coeff = [(1 - scatter) * scatter ** k for k in range(n_levels)] + [scatter ** n_levels]
    levels = [{"level": k, "size": trained["levels"][k]["size"],
               "w0_mean": float(maps[k][..., 0].mean()), "w1_mean": float(maps[k][..., 1].mean()),
               "w0_min": float(maps[k][..., 0].min()), "w0_max": float(maps[k][..., 0].max()),
               "w1_min": float(maps[k][..., 1].min()), "w1_max": float(maps[k][..., 1].max()),
               "max_weight_error": trained["levels"][k]["max_weight_error"]}
              for k in range(n_levels)]
    coefficients = [{"mip": k, "learned_mean": float(coeff_maps[k].mean()),
                     "learned_min": float(coeff_maps[k].min()),
                     "learned_max": float(coeff_maps[k].max()), "urp": urp_coeff[k]}
                    for k in range(n_levels + 1)]

    report = {"scatter": scatter, "exact_w": [1 - scatter, scatter],
              "train_loss": trained["train_loss"], "test_loss": trained["test_loss"],
              "pass": trained["pass"], "pass_criteria": trained["pass_criteria"],
              "steps": trained["steps"], "seconds": trained["seconds"],
              "error_range": list(ERROR_RANGE), "frames": frames, "levels": levels,
              "coefficients": coefficients, "curve": trained["curve"],
              "weight_grid": list(WEIGHT_GRID),
              "weight_maps": [m.round(4).tolist() for m in maps],
              "coefficient_maps": [c.round(4).tolist() for c in coeff_maps]}
    (REPORT / "report.json").write_text(json.dumps(report))

    summary = (f"Test loss {report['test_loss']:.2e}, train loss {report['train_loss']:.2e}, "
               f"verdict {'PASS' if report['pass'] else 'FAIL'} "
               f"(criteria {report['pass_criteria']}).")
    md = ["# End-to-end glare pyramid vs URP", "", summary, "", "| Frame | Split | Relative loss | Energy MLP/URP (R, G, B) | p99 rel. error |",
          "|---|---|---|---|---|"]
    md += [f"| `{f['name']}` | {f['split']} | {f['relative_loss']:.2e} | "
           + " / ".join(f"{e:.5f}" for e in f["energy_ratio_rgb"])
           + f" | {f['rel_error_p99']:.2e} |" for f in frames]
    md += ["", "| Level | Size | w0 mean (min–max) | w1 mean (min–max) | max weight error |",
           "|---|---|---|---|---|"]
    md += [f"| {lv['level']} | {lv['size'][0]}×{lv['size'][1]} | {lv['w0_mean']:.3f} "
           f"({lv['w0_min']:.3f}–{lv['w0_max']:.3f}) | {lv['w1_mean']:.3f} "
           f"({lv['w1_min']:.3f}–{lv['w1_max']:.3f}) | {lv['max_weight_error']:.3f} |"
           for lv in levels]
    md += ["", "| Mip | Learned c_k mean (min–max) | URP |", "|---|---|---|"]
    md += [f"| {c['mip']} | {c['learned_mean']:.4f} ({c['learned_min']:.4f}–"
           f"{c['learned_max']:.4f}) | {c['urp']:.4f} |" for c in coefficients]
    (REPORT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}/ (images, report.json, report.md)")


if __name__ == "__main__":
    main()
