"""Real-ray-traced pupil tables for the reference lens (offline, uses rayoptics).

Why not Seidel coefficients: a fast lens balances large third-order aberrations against
higher orders. For the Nikkor 35/2 at f/2 the third-order terms are 15-57 waves, while
the real wavefront error is ~2-3 waves RMS, so a Seidel polynomial would be 5-20x too
aberrated. We trace real rays instead and hand psf.py the wavefront itself.

For each field height H and wavelength λ, a grid of rays is traced (unclipped) from the
entrance pupil, then resampled onto a regular grid of the pupil coordinate the Fraunhofer
FFT needs, the image-space ray direction:
    u = (α - α_chief) · 2·F#        α = transverse direction cosines in image space
so |u| = 1 is the f-number cone on axis. The chief ray is the ray through the centre of
the stop. Axis 0 of every table is meridional (from the image centre towards the field
point, = image +field_dir in psf.py), axis 1 sagittal.

Tables, float32 [n_H, n_λ, N, N], over u in [-U, U]:
  opd     wavefront error in waves at λ, w.r.t. a sphere centred on the chief ray's image
          point at the reference wavelength. Lateral colour stays in it as a tilt and
          axial colour as defocus; distortion is dropped (PSFs sit on the chief ray).
          A ray lands at +dOPD/dα from that point (tests/test_lens_trace.py checks this).
  stop_x, stop_y   where the ray crosses the aperture stop, in stop radii (stop_y
          meridional). psf.py cuts the diaphragm blades there, where they really are.
  vig     signed distance in u to the nearest lens rim other than the stop; > 0 means the
          ray gets through. This is the mechanical (cat's-eye) vignetting.
The stop radius is set so the on-axis ray at the edge of the f-number cone just passes.

Usage (tables are cached in data/lens/; psf.py builds them on demand):
    uv run python -m neuralglare.reference.lens_trace --config quick
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import time

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.ndimage import distance_transform_edt

from neuralglare.reference import lens_params as lp

CACHE_DIR = "data/lens"
STOP_KEEP = 1.3      # rays crossing the stop beyond this many stop radii are discarded


def wavelengths_nm(cfg):
    """Spectral samples, then the reference wavelength last."""
    return [1000 * w for w in cfg.spectral.wavelengths_um] + [1000 * cfg.trace.ref_wavelength_um]


def build_model(cfg, heights):
    """rayoptics OpticalModel of cfg.lens at cfg.camera.f_number, one field per height."""
    from rayoptics.environment import FieldSpec, OpticalModel, PupilSpec, WvlSpec

    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    opm.system_spec.dimensions = "mm"
    opm.radius_mode = True
    osp["pupil"] = PupilSpec(osp, key=["image", "f/#"], value=cfg.camera.f_number)
    # Fields as object angles of an ideal pinhole with the lens's focal length, which is
    # how make_inputs.py projects the scene: H maps to the pixel the PSF node sits on.
    # (rayoptics' relative image-height fields scale the angle linearly, not its tangent.)
    angles = [math.degrees(math.atan(h * cfg.camera.half_diag_mm() / cfg.lens.efl_mm))
              for h in heights]
    osp["fov"] = FieldSpec(osp, key=["object", "angle"], value=max(max(angles), 1e-6),
                           is_relative=False, flds=angles)
    wl = wavelengths_nm(cfg)
    osp["wvls"] = WvlSpec([(w, 1.0) for w in wl], ref_wl=len(wl) - 1)
    sm.gaps[0].thi = 1e10                               # object at infinity
    for i, (r, t, nd, vd, _) in enumerate(cfg.lens.surfaces):
        sm.add_surface([r, t, nd, vd] if nd != 1.0 else [r, t])
        if i == cfg.lens.stop_row:
            sm.set_stop()
    sm.do_apertures = False                             # rims are handled by us, below
    opm.update_model()
    return opm


def _ray(opm, fld, px, py, wvl):
    """One unclipped real ray from relative entrance-pupil point (px, py); None on failure
    (missed surface, total internal reflection)."""
    from rayoptics.raytr import trace
    pkg, _ = trace.trace_safe(opm, [px, py], fld, wvl, None, None,
                              check_apertures=False, apply_vignetting=False)
    return pkg


def stop_radius(opm, cfg):
    """Where the on-axis ray at the edge of the paraxial pupil crosses the stop, so the
    real f-number cone just fills the stop. Field 0 must be on axis."""
    axial = _ray(opm, opm["optical_spec"]["fov"].fields[0], 0.0, 1.0, wavelengths_nm(cfg)[-1])
    return abs(axial[0][opm["seq_model"].stop_surface][0][1])


def _scan_bundle(opm, fld, wvl, stop_ifc, r_stop):
    """Find the chief ray and the entrance-pupil box whose rays cross the stop within
    STOP_KEEP stop radii. Off-axis this box is shifted and stretched (pupil aberration)."""
    def stop_xy(px, py):
        pkg = _ray(opm, fld, px, py, wvl)
        return (np.nan, np.nan) if pkg is None else pkg[0][stop_ifc][0][:2] / r_stop

    pys = np.linspace(-8, 8, 641)
    sy = np.array([stop_xy(0.0, p)[1] for p in pys])
    s = np.sign(sy)
    i = np.flatnonzero(np.isfinite(sy[:-1]) & np.isfinite(sy[1:]) & (s[:-1] != s[1:]))[0]
    lo, hi = pys[i], pys[i + 1]
    for _ in range(40):                                 # bisect stop_y = 0: the chief ray
        mid = 0.5 * (lo + hi)
        if np.sign(stop_xy(0.0, mid)[1]) == s[i]:
            lo = mid
        else:
            hi = mid
    py_c = 0.5 * (lo + hi)

    step = pys[1] - pys[0]
    keep = np.isfinite(sy) & (np.abs(sy) <= STOP_KEEP)
    py_lo, py_hi = pys[keep].min() - step, pys[keep].max() + step
    pxs = np.linspace(0, 8, 321)
    sx = np.array([stop_xy(p, py_c)[0] for p in pxs])
    keep = np.isfinite(sx) & (np.abs(sx) <= STOP_KEEP)
    px_hi = pxs[keep].max() + step
    pad = 0.1 * (py_hi - py_lo)                         # the bundle is curved, not a box
    return py_c, py_lo - pad, py_hi + pad, px_hi + pad


def trace_field(opm, cfg, fi, r_stop):
    """Trace the ray grid for field `fi` at every wavelength.

    Returns a dict of arrays on the ray grid [n_λ, ny, nx] (sagittal half traced, then
    mirrored): u (…, 2: mer, sag), opd (waves), stop (…, 2: mer, sag, in stop radii),
    margins (…, n_rims: rim radius - ray height, mm), land (…, 2: mer, sag image offset
    from the chief ray at the reference wavelength, mm), ok. Wavelengths as wavelengths_nm.
    """
    from rayoptics.raytr import trace, waveabr

    sm = opm["seq_model"]
    fod = opm["analysis_results"]["parax_data"].fod
    fld = opm["optical_spec"]["fov"].fields[fi]
    stop_ifc = sm.stop_surface
    rims = [(i + 1, 0.5 * s[4]) for i, s in enumerate(cfg.lens.surfaces)
            if i != cfg.lens.stop_row and s[4] > 0]
    wl = wavelengths_nm(cfg)
    wref = wl[-1]
    two_f = 2.0 * cfg.camera.f_number

    trace.get_chief_ray_pkg(opm, fld, wref, 0.0)      # fixes rayoptics' aim before scanning
    py_c, py_lo, py_hi, px_hi = _scan_bundle(opm, fld, wref, stop_ifc, r_stop)
    chief = _ray(opm, fld, 0.0, py_c, wref)[0]
    img_ref = np.array(chief[-1][0])
    alpha_c = np.array(chief[-2][1][:2])

    ny, nx = cfg.trace.rays, cfg.trace.rays // 2 + 1
    pys, pxs = np.linspace(py_lo, py_hi, ny), np.linspace(0.0, px_hi, nx)
    shape = (len(wl), ny, nx)
    out = {k: np.full(shape + (d,), np.nan) for k, d in
           [("u", 2), ("stop", 2), ("margins", len(rims)), ("land", 2)]}
    out["opd"] = np.full(shape, np.nan)
    for li, w in enumerate(wl):
        ref_sphere, cr_pkg = trace.setup_pupil_coords(opm, fld, w, 0.0, image_pt=img_ref)
        lam_mm = w * 1e-6
        for iy, py in enumerate(pys):
            for ix, px in enumerate(pxs):
                pkg = _ray(opm, fld, px, py, w)
                if pkg is None:
                    continue
                seg = pkg[0]
                opd = waveabr.wave_abr_full_calc(fod, fld, w, 0.0, pkg, cr_pkg, ref_sphere)
                a = (np.array(seg[-2][1][:2]) - alpha_c) * two_f
                st = np.array(seg[stop_ifc][0][:2]) / r_stop
                ld = np.array(seg[-1][0][:2]) - img_ref[:2]
                out["u"][li, iy, ix] = a[1], a[0]
                out["stop"][li, iy, ix] = st[1], st[0]
                out["land"][li, iy, ix] = ld[1], ld[0]
                out["margins"][li, iy, ix] = [r - np.hypot(*seg[k][0][:2]) for k, r in rims]
                out["opd"][li, iy, ix] = opd / lam_mm

    # mirror the sagittal half (px -> -px flips the sagittal components)
    def mirror(a, flip_sag):
        m = a[:, :, :0:-1].copy()
        if flip_sag:
            m[..., 1] *= -1
        return np.concatenate([m, a], axis=2)
    for k in ("u", "stop", "land"):
        out[k] = mirror(out[k], True)
    out["margins"] = mirror(out["margins"], False)
    out["opd"] = mirror(out["opd"], False)
    out["ok"] = np.isfinite(out["opd"])
    out["chief_img_mm"] = float(img_ref[1])
    return out


def transmitted(rays):
    """Rays inside the stop circle and every lens rim."""
    with np.errstate(invalid="ignore"):
        return (rays["ok"] & (np.hypot(*np.moveaxis(rays["stop"], -1, 0)) <= 1.0)
                & (rays["margins"].min(-1) >= 0))


def _resample(rays, li, U, N):
    """Scattered rays of one wavelength -> regular (u_mer, u_sag) grid, NaN-free."""
    with np.errstate(invalid="ignore"):
        use = rays["ok"][li] & (np.hypot(*np.moveaxis(rays["stop"][li], -1, 0)) <= STOP_KEEP)
    pts = rays["u"][li][use]
    vals = np.concatenate([rays["opd"][li][use][:, None], rays["stop"][li][use],
                           rays["margins"][li][use]], axis=1)
    g = np.linspace(-U, U, N)
    gm, gs = np.meshgrid(g, g, indexing="ij")
    res = CloughTocher2DInterpolator(pts, vals)(gm, gs)        # (N, N, C), NaN off hull
    hole = np.isnan(res[..., 0])
    if hole.any():                                             # nearest fill
        idx = distance_transform_edt(hole, return_distances=False, return_indices=True)
        res = res[tuple(idx)]

    du = g[1] - g[0]
    vig = np.full((N, N), np.inf)
    for k in range(3, res.shape[-1]):                          # rim margin -> distance in u
        gy, gx = np.gradient(res[..., k], du)
        vig = np.minimum(vig, res[..., k] / np.maximum(np.hypot(gy, gx), 1e-9))
    vig[hole] = -U                                             # no ray got there: blocked
    return res[..., 0], res[..., 1], res[..., 2], vig


def build_table(cfg, heights, verbose=True):
    """Trace every field height and wavelength and resample to pupil tables (see top)."""
    heights = sorted({round(float(h), 9) for h in heights} | {0.0})
    opm = build_model(cfg, heights)
    fod = opm["analysis_results"]["parax_data"].fod
    r_stop = stop_radius(opm, cfg)

    t0 = time.time()
    fields = []
    for fi, h in enumerate(heights):
        fields.append(trace_field(opm, cfg, fi, r_stop))
        if verbose:
            print(f"  traced H={h:.3f}  ({time.time() - t0:.0f}s)")

    n_l = len(cfg.spectral.wavelengths_um)
    u_max = max(np.hypot(*np.moveaxis(f["u"][:n_l][transmitted(f)[:n_l]], -1, 0)).max()
                for f in fields)
    U, N = 1.1 * u_max, cfg.trace.table_px
    tabs = {k: np.zeros((len(heights), n_l, N, N), np.float32)
            for k in ("opd", "stop_y", "stop_x", "vig")}
    for hi, f in enumerate(fields):
        for li in range(n_l):
            for k, a in zip(("opd", "stop_y", "stop_x", "vig"), _resample(f, li, U, N)):
                tabs[k][hi, li] = a
    return dict(
        tabs,
        heights=np.array(heights),
        wavelengths_um=np.array(cfg.spectral.wavelengths_um),
        U=U, u_max=u_max, r_stop_mm=r_stop,
        chief_img_mm=np.array([f["chief_img_mm"] for f in fields]),
        efl_mm=fod.efl, f_number=fod.fno,
        rayoptics_version=importlib.metadata.version("rayoptics"),
    )


def _key(cfg, heights):
    blob = json.dumps([dataclasses.asdict(cfg.lens), dataclasses.asdict(cfg.camera),
                       dataclasses.asdict(cfg.trace), list(cfg.spectral.wavelengths_um),
                       sorted({round(float(h), 9) for h in heights})])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_or_build(cfg, heights, cache_dir=CACHE_DIR, verbose=True):
    """Cached build_table(): the file name hashes everything the table depends on."""
    path = os.path.join(cache_dir, f"{cfg.lens.name}_f{cfg.camera.f_number:g}_"
                                   f"{_key(cfg, heights)}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if verbose:
        print(f"tracing {cfg.lens.name} at f/{cfg.camera.f_number:g} -> {path}")
    tab = build_table(cfg, heights, verbose)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **tab)
    return tab


def height_index(tab, H):
    i = int(np.argmin(np.abs(tab["heights"] - H)))
    assert abs(tab["heights"][i] - H) < 1e-6, f"field height {H} not in the pupil table"
    return i


def main():
    from neuralglare.reference.psf import field_nodes

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["quick", "default"], default="quick")
    args = ap.parse_args()
    cfg = lp.QUICK if args.config == "quick" else lp.DEFAULT
    tab = load_or_build(cfg, [n[4] for n in field_nodes(cfg)])
    print(f"EFL {float(tab['efl_mm']):.3f} mm, f/{float(tab['f_number']):.3f}, "
          f"stop radius {float(tab['r_stop_mm']):.3f} mm, u_max {float(tab['u_max']):.3f}")
    for h, y in zip(tab["heights"], tab["chief_img_mm"]):
        ideal = h * cfg.camera.half_diag_mm()
        dist = 100 * (y / ideal - 1) if h > 0 else 0.0
        print(f"  H={h:.3f}  chief ray image height {y:7.3f} mm  (distortion {dist:+.2f}%)")


if __name__ == "__main__":
    main()
