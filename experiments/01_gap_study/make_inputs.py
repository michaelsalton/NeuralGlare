""" HDR test inputs for the gap study.

Outputs (float32 linear RGB EXR, plus tonemapped PNG previews and a JSON manifest):
    data/inputs/pointgrid.exr 7x5 grid of single bright pixel on black
    data/inputs/<hdri>_<view>.exr perspective crops of Poly Haven HDRIs
    
Usage:
  uv run python experiments/01_gap_study/make_inputs.py --hdri-dir data/hdri --out data/inputs
  uv run python experiments/01_gap_study/make_inputs.py
 
Put downloaded Poly Haven HDRIs (.exr or .hdr, equirectangular, CC0) in --hdri-dir.
"""

import argparse
import json
import os

import cv2 
import numpy as np

from neuralglare.io import luminance, read_hdr, write_exr, write_preview

W, H = 1920, 1080
HFOV_DEG = 60.0 # horizontal field of view for the virtual camera

# where the brightest light lands in each view: (name, yaw_deg, pitch_deg) offsets
# from the image centre, +yaw = right, +pitch = up
DEFAULT_VIEWS = [("center", 0.0, 0.0), ("mid", 15.0, 6.0), ("corner", 25.0, 14.0)]
    

def stats(rgb):
    lum = luminance(rgb)
    return {
        "max_lum": float(lum.max()),
        "median_lum": float(np.median(lum)),
        "dynamic_range": float(lum.max() / max(np.percentile(lum, 1), 1e-6)),
        "total_energy": float(rgb.sum()),
        "frac_pixels_above_1": float((lum > 1.0).mean()),
        "finite": bool(np.isfinite(rgb).all()),
    }
 
 
# ---------------------------------------------------------- synthetic inputs
def point_grid(cols=7, rows=5, intensity=1e4, margin=0.06):
    """Single-pixel deltas, evenly spaced, inset so edge PSFs stay in frame."""
    img = np.zeros((H, W, 3), np.float32)
    xs = np.round(np.linspace(margin, 1 - margin, cols) * (W - 1)).astype(int)
    ys = np.round(np.linspace(margin, 1 - margin, rows) * (H - 1)).astype(int)
    for y in ys:
        for x in xs:
            img[y, x] = intensity
    return img, {"cols": cols, "rows": rows, "intensity": intensity, "margin": margin,
                 "xs": xs.tolist(), "ys": ys.tolist()}
 
 
# ---------------------------------------------- equirect -> perspective crop
def equirect_to_perspective(equi, yaw_deg, pitch_deg, hfov_deg=HFOV_DEG):
    """Pinhole view of an equirectangular panorama (bilinear, wraps horizontally)."""
    eh, ew = equi.shape[:2]
    f = 0.5 * W / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
    d = np.stack([u, -v, np.full_like(u, f)], -1)          # camera space, +z forward
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
 
    p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    d = d @ (ry @ rx).T
 
    lon = np.arctan2(d[..., 0], d[..., 2])                  # [-pi, pi]
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))              # [-pi/2, pi/2]
    mx = ((lon / (2 * np.pi) + 0.5) * ew - 0.5).astype(np.float32)
    my = ((0.5 - lat / np.pi) * eh - 0.5).astype(np.float32)
    return cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
 
 
def normalize_exposure(rgb, target_median=0.18):
    """Scale so the median luminance is mid-grey. Preserves all ratios (no clamping)."""
    med = np.median(luminance(rgb))
    scale = target_median / max(med, 1e-8)
    return rgb * scale, float(scale)
 
 
# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdri-dir", default="data/hdri")
    ap.add_argument("--out", default="data/inputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {"resolution": [W, H], "hfov_deg": HFOV_DEG, "images": {}}
 
    img, meta = point_grid()
    write_exr(f"{args.out}/pointgrid.exr", img)
    write_preview(f"{args.out}/pointgrid_preview.png", img)
    manifest["images"]["pointgrid"] = {"type": "synthetic", **meta, **stats(img)}
 
    if os.path.isdir(args.hdri_dir):
        for fn in sorted(os.listdir(args.hdri_dir)):
            if not fn.lower().endswith((".exr", ".hdr")):
                continue
            name = os.path.splitext(fn)[0]
            equi = read_hdr(os.path.join(args.hdri_dir, fn))
            src_max = float(luminance(equi).max())
            for view, yaw, pitch in DEFAULT_VIEWS:
                crop, scale = normalize_exposure(equirect_to_perspective(equi, yaw, pitch))
                key = f"{name}_{view}"
                write_exr(f"{args.out}/{key}.exr", crop)
                write_preview(f"{args.out}/{key}_preview.png", crop)
                manifest["images"][key] = {
                    "type": "hdri_crop", "source": fn, "yaw_deg": yaw, "pitch_deg": pitch,
                    "exposure_scale": scale, "source_max_lum": src_max, **stats(crop)}
 
    with open(f"{args.out}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
 
    print(f"{'image':<32}{'max lum':>12}{'DR':>12}{'>1 frac':>10}")
    for k, s in manifest["images"].items():
        print(f"{k:<32}{s['max_lum']:>12.1f}{s['dynamic_range']:>12.0f}{s['frac_pixels_above_1']:>10.4f}")
        if s["max_lum"] < 50 and s["type"] == "hdri_crop":
            print(f"  ! {k}: weak highlights; the bright source may be clipped or out of frame")
 
 
if __name__ == "__main__":
    main()
 