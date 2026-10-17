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
from neuralglare.reference import lens_params as lp

W, H = lp.DEFAULT.camera.width_px, lp.DEFAULT.camera.height_px
# horizontal field of view of the virtual camera: must match the reference lens
# (36 mm on a 36 mm wide sensor -> 53.13 deg), or PSF field heights land in the wrong place
HFOV_DEG = lp.DEFAULT.hfov_deg()

# Field height H along the frame diagonal where each HDRI's brightest region is placed
# (0 = centre, 1 = corner pixel, towards the top right). H is the variable the gap study
# varies, and unlike angle offsets it does not depend on the FOV. 0.85 keeps the source
# and most of its glare inside the frame.
DEFAULT_VIEWS = [("center", 0.0), ("mid", 0.5), ("corner", 0.85)]
BLOCK = 8   # box size for locating the brightest region (decision log 2026-10-15)
    

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
def focal_px(hfov_deg=HFOV_DEG):
    return 0.5 * W / np.tan(np.radians(hfov_deg) / 2)


def rot(yaw, pitch):
    """Camera-to-world rotation (radians). yaw = pitch = 0 looks at lon 0, lat 0;
    +yaw turns right (+lon), +pitch looks up (+lat)."""
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)],
                   [0, -np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    return ry @ rx


def equirect_to_perspective(equi, yaw_deg, pitch_deg, hfov_deg=HFOV_DEG):
    """Pinhole view of an equirectangular panorama (bilinear, wraps horizontally)."""
    eh, ew = equi.shape[:2]
    f = focal_px(hfov_deg)
    u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
    d = np.stack([u, -v, np.full_like(u, f)], -1)          # camera space, +z forward
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d = d @ rot(np.radians(yaw_deg), np.radians(pitch_deg)).T
 
    lon = np.arctan2(d[..., 0], d[..., 2])                  # [-pi, pi]
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))              # [-pi/2, pi/2]
    mx = ((lon / (2 * np.pi) + 0.5) * ew - 0.5).astype(np.float32)
    my = ((0.5 - lat / np.pi) * eh - 0.5).astype(np.float32)
    return cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ------------------------------------------------------- aiming at the source
def brightest_direction(equi, block=BLOCK):
    """(lon, lat) in radians of the brightest block x block box-averaged region."""
    lum = luminance(equi)
    eh, ew = lum.shape
    small = cv2.resize(lum, (ew // block, eh // block), interpolation=cv2.INTER_AREA)
    r, c = np.unravel_index(np.argmax(small), small.shape)
    col, row = (c + 0.5) * block, (r + 0.5) * block            # full-res position
    lon = (col / ew - 0.5) * 2 * np.pi                          # inverse of the remap
    lat = (0.5 - row / eh) * np.pi
    return float(lon), float(lat)


def aim(lon, lat, u, v, f):
    """Yaw, pitch (radians) that put direction (lon, lat) at pixel offset (u right, v up)
    from the image centre. Fixed-point iteration, since yaw and pitch do not separate."""
    d = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
    yaw, pitch = lon - np.arctan2(u, f), lat - np.arctan2(v, f)
    for _ in range(50):
        c = rot(yaw, pitch).T @ d                               # source in camera space
        pu, pv = f * c[0] / c[2], f * c[1] / c[2]
        # turning the camera right (+yaw) moves the source left in the image, so subtract
        yaw -= np.arctan2(u, f) - np.arctan2(pu, f)
        pitch -= np.arctan2(v, f) - np.arctan2(pv, f)
    return float(yaw), float(pitch)


def view_target(h_field):
    """Pixel offset (u right, v up) from the centre for field height h_field."""
    return h_field * (W - 1) / 2, h_field * (H - 1) / 2


def locate_light(rgb, block=BLOCK, win=64):
    """Pixel (x, y) of the brightest source and its field height (0 at the centre pixel,
    1 at the corner pixels). Finds the brightest block x block box-averaged region, then
    takes the luminance-weighted centroid of the pixels above half its peak within a
    win x win window: argmax alone lands on the edge of a flat-topped (saturated) source."""
    lum = luminance(rgb)
    y0, x0 = np.unravel_index(np.argmax(cv2.blur(lum, (block, block))), lum.shape)
    ys, xs = slice(max(y0 - win, 0), y0 + win + 1), slice(max(x0 - win, 0), x0 + win + 1)
    patch = lum[ys, xs]
    w = np.where(patch >= 0.5 * patch.max(), patch, 0.0)
    gy, gx = np.mgrid[ys, xs]
    x, y = (w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()
    cx, cy = (W - 1) / 2, (H - 1) / 2
    return [round(float(x), 1), round(float(y), 1)], float(np.hypot(x - cx, y - cy) / np.hypot(cx, cy))

 
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
            lon, lat = brightest_direction(equi)
            for view, h_field in DEFAULT_VIEWS:
                yaw, pitch = aim(lon, lat, *view_target(h_field), focal_px())
                yaw_deg, pitch_deg = np.degrees(yaw), np.degrees(pitch)
                crop, scale = normalize_exposure(equirect_to_perspective(equi, yaw_deg, pitch_deg))
                light_px, h_measured = locate_light(crop)
                key = f"{name}_{view}"
                write_exr(f"{args.out}/{key}.exr", crop)
                write_preview(f"{args.out}/{key}_preview.png", crop)
                manifest["images"][key] = {
                    "type": "hdri_crop", "source": fn,
                    "source_lon_deg": float(np.degrees(lon)), "source_lat_deg": float(np.degrees(lat)),
                    "H_target": h_field, "H_measured": h_measured, "light_px": light_px,
                    "yaw_deg": float(yaw_deg), "pitch_deg": float(pitch_deg),
                    "exposure_scale": scale, "source_max_lum": src_max, **stats(crop)}
 
    with open(f"{args.out}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
 
    print(f"{'image':<40}{'max lum':>12}{'DR':>12}{'>1 frac':>10}{'H tgt':>7}{'H meas':>8}")
    for k, s in manifest["images"].items():
        h = (f"{s['H_target']:>7.2f}{s['H_measured']:>8.2f}" if s["type"] == "hdri_crop" else "")
        print(f"{k:<40}{s['max_lum']:>12.1f}{s['dynamic_range']:>12.0f}"
              f"{s['frac_pixels_above_1']:>10.4f}{h}")
        if s["type"] != "hdri_crop":
            continue
        if s["max_lum"] < 50:
            print(f"  ! {k}: weak highlights; the HDRI may have no compact bright source")
        if abs(s["H_measured"] - s["H_target"]) > 0.05:
            print(f"  ! {k}: brightest region landed at H={s['H_measured']:.2f}, not "
                  f"{s['H_target']:.2f}; the source is not compact (use H_measured)")
 
 
if __name__ == "__main__":
    main()
 