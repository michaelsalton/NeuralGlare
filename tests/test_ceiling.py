"""Coma split masks of the ceiling analysis (decision log 2026-11-17)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent
_PATH = _ROOT / "experiments" / "03_train_network" / "ceiling.py"
_spec = importlib.util.spec_from_file_location("ceiling", _PATH)
ceiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ceiling)

K = 127
BANK = _ROOT / ceiling.psf_bank_path("default")


@pytest.mark.parametrize("width", [ceiling.RAY_HALF_WIDTH_PX, ceiling.ROBUST_HALF_WIDTH_PX])
def test_masks_partition_the_psf(width):
    m = ceiling.coma_masks(K, width)
    assert (sum(m[p].astype(int) for p in ceiling.PIECES) == 1).all()
    assert m["core"].sum() > 0 and m["rays"].sum() > 0 and m["glow"].sum() > 0


def test_core_is_aperture_fines_round_core():
    m = ceiling.coma_masks(K)
    assert np.array_equal(m["core"], ceiling.rings(K) <= 2 * ceiling.CORE_PX)


@pytest.mark.parametrize("piece", ceiling.PIECES)
def test_masks_are_mirror_symmetric(piece):
    """The PSF's symmetry includes the mirror about the horizontal axis, exact on the grid."""
    m = ceiling.coma_masks(K)[piece]
    assert np.array_equal(m, m[::-1])


def test_ray_band_lies_along_the_ray_directions():
    """Ray 0 points along +x: the pixels on that axis outside the core are rays; the pixel
    halfway between rays 0 and 1, far out, is glow."""
    m = ceiling.coma_masks(K)
    c = K // 2
    assert m["rays"][c, c + 6:].all()
    half = np.pi / ceiling.SPIKES                     # halfway to ray 1
    r = 50
    assert m["glow"][c + round(r * np.sin(half)), c + round(r * np.cos(half))]


def test_wider_band_takes_light_from_glow():
    m2, m3 = ceiling.coma_masks(K, 2.0), ceiling.coma_masks(K, 3.0)
    assert (m2["rays"] <= m3["rays"]).all() and m3["rays"].sum() > m2["rays"].sum()


def test_masked_psf_selects():
    rng = np.random.default_rng(0)
    sym, exact = rng.random((K, K, 3)), rng.random((K, K, 3))
    m = ceiling.coma_masks(K)
    every = m["core"] | m["rays"] | m["glow"]
    assert np.array_equal(ceiling.masked_psf(sym, exact, every), sym)
    assert np.array_equal(ceiling.masked_psf(sym, exact, np.zeros((K, K), bool)), exact)
    out = ceiling.masked_psf(sym, exact, m["rays"])
    assert np.array_equal(out[m["rays"]], sym[m["rays"]])
    assert np.array_equal(out[~m["rays"]], exact[~m["rays"]])


def test_far_masks_partition_and_mirror():
    m = ceiling.far_masks(K)
    assert (sum(m[p].astype(int) for p in ceiling.FAR_PIECES) == 1).all()
    for p in ceiling.FAR_PIECES:
        assert m[p].sum() > 0
        assert np.array_equal(m[p], m[p][::-1])


def test_far_masks_split_the_coma_pieces_at_the_annulus():
    coma, far = ceiling.coma_masks(K), ceiling.far_masks(K)
    c = K // 2
    y, x = np.mgrid[:K, :K]
    near = np.hypot(x - c, y - c) < ceiling.ANNULUS_PX
    assert np.array_equal(far["core"], coma["core"])
    assert np.array_equal(far["rays_far"] | (coma["rays"] & near), coma["rays"])
    assert not (far["annulus"] & ~near).any() and not (far["rays_far"] & near).any()


def test_ray_index_is_the_nearest_ray():
    """Ray 0 on +x, ray k at k * 360/14 (y down); every band pixel is within 90 deg / 7 of
    its ray's direction, i.e. closer to it than to any other ray."""
    m = ceiling.coma_masks(K)
    c = K // 2
    assert (m["ray"][c, c + 6:] == 0).all()
    assert (m["ray"][c, :c - 5] == ceiling.SPIKES // 2).all()          # ray 7 points along -x
    y, x = np.mgrid[:K, :K]
    ang = np.degrees(np.arctan2(y - c, x - c))
    diff = np.abs((ang - m["ray"] * 360 / ceiling.SPIKES + 180) % 360 - 180)
    off_centre = (x != c) | (y != c)
    assert (diff[off_centre] <= 180 / ceiling.SPIKES + 1e-9).all()


def test_restore_ray_light_gives_each_ray_its_exact_sum():
    rng = np.random.default_rng(1)
    kernel, exact = rng.random((K, K, 3)), rng.random((K, K, 3))
    m = ceiling.far_masks(K)
    out = ceiling.restore_ray_light(kernel, exact, m["rays_far"], m["ray"])
    for k in range(ceiling.SPIKES):
        sel = m["rays_far"] & (m["ray"] == k)
        np.testing.assert_allclose(out[sel].sum(axis=0), exact[sel].sum(axis=0), rtol=1e-12)
        # the shape inside the ray is the kernel's: a per-channel scale of it
        ratio = out[sel] / kernel[sel]
        np.testing.assert_allclose(ratio, np.broadcast_to(ratio[:1], ratio.shape), rtol=1e-12)
    assert np.array_equal(out[~m["rays_far"]], kernel[~m["rays_far"]])


def test_radial_segment_edges():
    seg = ceiling.radial_segment(K)
    c = K // 2
    row = seg[c, c:]                                  # r = 0, 1, 2, ... along +x
    assert (row[:9] == -1).all()
    assert row[9] == 0 and row[15] == 0 and row[16] == 1 and row[31] == 1 and row[32] == 2
    assert seg.max() == len(ceiling.RAY_PROFILE_BINS_PX) - 1


def test_restore_per_ray_and_segment():
    rng = np.random.default_rng(2)
    kernel, exact = rng.random((K, K, 3)), rng.random((K, K, 3))
    m = ceiling.far_masks(K)
    groups = m["ray"] * len(ceiling.RAY_PROFILE_BINS_PX) + ceiling.radial_segment(K)
    out = ceiling.restore_ray_light(kernel, exact, m["rays_far"], groups)
    for g in np.unique(groups[m["rays_far"]]):
        sel = m["rays_far"] & (groups == g)
        np.testing.assert_allclose(out[sel].sum(axis=0), exact[sel].sum(axis=0), rtol=1e-12)
        ratio = out[sel] / kernel[sel]
        np.testing.assert_allclose(ratio, np.broadcast_to(ratio[:1], ratio.shape), rtol=1e-12)
    assert np.array_equal(out[~m["rays_far"]], kernel[~m["rays_far"]])
    assert len(np.unique(groups[m["rays_far"]])) == ceiling.SPIKES * len(ceiling.RAY_PROFILE_BINS_PX)


@pytest.mark.skipif(not BANK.exists(), reason="needs the PSF bank (psf_bank.py)")
def test_ray_band_sits_on_the_banks_rays():
    """Checks the angle convention against the data: beyond 10 px the on-axis PSF is far
    brighter inside the ray band than between the rays (exact peak : ring median 300-600x)."""
    with np.load(BANK) as z:
        i = int(np.argmin(z["field_height"]))
        psf = z["psf"][i].astype(np.float64)
    lum = ceiling.luminance(psf)
    m = ceiling.coma_masks(psf.shape[0])
    c = psf.shape[0] // 2
    y, x = np.mgrid[:psf.shape[0], :psf.shape[0]]
    far = np.hypot(x - c, y - c) > 10
    assert lum[m["rays"] & far].mean() > 10 * lum[m["glow"] & far].mean()
