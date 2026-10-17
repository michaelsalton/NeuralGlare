from dataclasses import replace

import numpy as np

from neuralglare.reference import lens_params as lp
from neuralglare.reference import lens_trace
from neuralglare.reference.psf import psf_mono

# f/2 (wide open, most aberrated), near-circular stop, one wavelength = the reference one
CFG = replace(lp.QUICK,
              camera=replace(lp.QUICK.camera, f_number=2.0),
              aperture=replace(lp.QUICK.aperture, blades=64),
              trace=replace(lp.QUICK.trace, rays=41, table_px=129),
              spectral=replace(lp.QUICK.spectral, wavelengths_um=(0.5876,)))


def test_paraxial_matches_prescription():
    opm = lens_trace.build_model(CFG, [0.0])
    fod = opm["analysis_results"]["parax_data"].fod
    assert abs(fod.efl - CFG.lens.efl_mm) < 0.01
    assert abs(fod.fno - 2.0) < 0.01


def _psf_centroid_x(tab, H):
    p = psf_mono(CFG, 0.5876, 0, lens_trace.height_index(tab, H), (1.0, 0.0), tab)
    xx = np.arange(p.shape[1]) - p.shape[1] // 2
    return float((p.sum(0) * xx).sum())


def test_psf_centroid_matches_ray_trace():
    # The diffraction PSF's centroid equals the mean ray landing point. This checks the
    # whole chain end to end: OPD sign, u scaling, and the table's orientation on screen.
    tab = lens_trace.build_table(CFG, [1.0], verbose=False)
    opm = lens_trace.build_model(CFG, [0.0, 1.0])
    rays = lens_trace.trace_field(opm, CFG, 1, lens_trace.stop_radius(opm, CFG))

    # Rays sample the entrance pupil uniformly; light is uniform in u, so weight each ray
    # by the area it covers in u (Jacobian of the pupil -> u map).
    um, us = rays["u"][0, ..., 0], rays["u"][0, ..., 1]
    dm_dy, dm_dx = np.gradient(um)
    ds_dy, ds_dx = np.gradient(us)
    w = np.abs(dm_dy * ds_dx - dm_dx * ds_dy)
    w = np.where(lens_trace.transmitted(rays)[0] & np.isfinite(w), w, 0.0)
    land_mer_mm = np.nan_to_num(rays["land"][0, ..., 0])
    geo_px = (w * land_mer_mm).sum() / w.sum() * 1000 / CFG.camera.pixel_pitch_um

    psf_px = _psf_centroid_x(tab, 1.0)
    assert abs(geo_px) > 0.1, "test needs a visibly off-centre spot"
    assert abs(psf_px - geo_px) < 0.1 * abs(geo_px) + 0.05, (psf_px, geo_px)


def test_on_axis_psf_is_centred():
    tab = lens_trace.build_table(CFG, [0.0], verbose=False)
    assert abs(_psf_centroid_x(tab, 0.0)) < 0.02
