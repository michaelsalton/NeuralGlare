"""Optical configuration for the reference glare model.

The wavefront and vignetting come from a real-ray trace of a published lens
prescription (see `Prescription` and lens_trace.py).
Every physical value below is cited.
Known limitations:
  - lens rim (clear) diameters other than the front element are PhotonsToPhotos
    estimates; no measured vignetting of this lens was found to validate them;
  - veiling-glare scatter is switched off (no citable data for this lens);
  - glass dispersion is modelled from (nd, vd) only.
psf.py prints a warning while PLACEHOLDER is True.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from neuralglare.reference import camspec

PLACEHOLDER = False


@dataclass(frozen=True)
class Camera:
    width_px: int = 1920
    height_px: int = 1080
    # 36 mm full-frame sensor width / 1920 px. With the 36 mm reference lens this gives a
    # 53.13 deg horizontal FOV (hfov_deg below), which make_inputs.py uses.
    pixel_pitch_um: float = 18.75
    f_number: float = 2.8           # operating aperture (the reference lens opens to f/2)

    def half_diag_mm(self):
        """Image height at field H = 1: centre to corner pixel centre, as psf.field_nodes."""
        return self.pixel_pitch_um * 1e-3 * math.hypot((self.width_px - 1) / 2,
                                                      (self.height_px - 1) / 2)


@dataclass(frozen=True)
class Aperture:
    # Nikkor 35mm f/2 AI-S: 7 straight blades (-> 14-point starburst). Sources: Ken
    # Rockwell, kenrockwell.com/nikon/35f20ais.htm; Kamerastore listing t126053. No Nikon
    # spec sheet found.
    blades: int = 7
    rotation_deg: float = 0.0       # convention: real orientation varies per lens/aperture
    edge_softness: float = 1.0      # anti-aliasing width of the pupil edge, in pupil samples
    # Mechanical vignetting is no longer a parameter: it comes from the lens rims in the
    # ray trace (lens_trace.py).


@dataclass(frozen=True)
class Prescription:
    """A lens design, one row per surface, object side first:
        (radius_mm, thickness_mm, nd, vd, clear_diameter_mm)
    radius 0 = flat; air gaps have nd = 1, vd = 0; the last thickness is the back focal
    distance to the image plane. Row `stop_row` is the aperture stop (its diameter is set
    by Camera.f_number, so its clear diameter is ignored).
    Glass dispersion is modelled from (nd, vd) only (rayoptics ModelGlass): approximate
    at the blue and red ends of the spectrum.
    """
    name: str
    source: str
    efl_mm: float
    stop_row: int
    surfaces: tuple


# Nikon AI Nikkor 35mm f/2S (8 elements / 6 groups, as the production lens).
# Source: US Patent 3,507,558, Example 1 (Y. Shimizu, Nippon Kogaku, 1970; f = 1.0,
# 2w = 62 deg, F:2, B.f = 1.1337), scaled x36. Radii, thicknesses, nd, vd and back focus
# checked against the patent scan. Starting transcription: PhotonsToPhotos Optical Bench,
# Data/US003507558_Example01_Tale84.txt.
# Stop position: not printed in the patent (it gives only the 5.2999 mm gap, d7 = 0.14722f).
#   Placed 1.9332 mm after r7 so rayoptics reproduces the patent's own Seidel sums:
#   coma II = -0.0572022 (fitted) and, independently, distortion V = 0.22371 (patent
#   0.2237184); spherical I = 0.48956 (patent 0.4895778) does not depend on the stop.
# Clear diameters: PhotonsToPhotos estimates. Front element 46.58 mm agrees with the
#   patent's "about 1.3f" (46.8 mm); the others are unvalidated.
# Paraxial check (rayoptics 0.9.10): EFL 36.000 mm, BFL 40.812 mm, f/2.0.
NIKKOR_35_F2 = Prescription(
    name="nikkor35f2",
    source="US 3,507,558 Ex. 1 via PhotonsToPhotos Optical Bench",
    efl_mm=36.0,
    stop_row=7,
    surfaces=(
        (115.621, 3.8999, 1.71300, 53.9, 46.58),
        (0.0,     0.1001, 1.0,     0.0,  46.58),
        (50.000,  2.2000, 1.51680, 64.2, 39.20),
        (17.201, 11.2000, 1.0,     0.0,  29.08),
        (32.202,  8.2001, 1.66755, 41.9, 26.08),
        (-55.001, 10.5001, 1.51454, 54.6, 19.72),
        (33.998,  1.9332, 1.0,     0.0,  18.23),
        (0.0,     3.3667, 1.0,     0.0,  17.946),   # aperture stop
        (-21.571, 1.0001, 1.78470, 26.1, 18.57),
        (38.790,  6.9998, 1.74443, 49.4, 23.56),
        (-25.819, 0.1001, 1.0,     0.0,  23.56),
        (-110.002, 2.9999, 1.76684, 46.2, 23.52),
        (-38.488, 0.1001, 1.0,     0.0,  23.52),
        (130.000, 2.4998, 1.74400, 44.9, 26.52),
        (-79.049, 40.8132, 1.0,    0.0,  26.52),
    ),
)


@dataclass(frozen=True)
class Trace:
    """Sampling of the real-ray trace (lens_trace.py)."""
    rays: int = 101                 # rays per side of the entrance-pupil grid (half traced)
    table_px: int = 257             # pupil table resolution (direction-cosine grid)
    ref_wavelength_um: float = 0.5876   # d line: the patent's design wavelength (nd)


@dataclass(frozen=True)
class Scatter:
    """Wide, low-energy veiling-glare halo from scattering (glass, coatings, dust).
    Kernel  S(r) ∝ (r^2 + r0^2)^(-exponent/2), normalised to sum 1.
    Final PSF = (1 - energy_fraction) * diffraction PSF + energy_fraction * S.
    Switched off (energy_fraction = 0): no citable glare spread function exists for the
    reference lens, the term is spatially invariant (not what the gap study tests), and
    an invented halo would bias the comparison with pyramid bloom. The shape parameters
    are kept only so the code path stays exercised."""
    energy_fraction: float = 0.0
    core_radius_px: float = 2.0     # unused while energy_fraction = 0
    exponent: float = 2.5           # unused while energy_fraction = 0


@dataclass(frozen=True)
class Spectral:
    """Wavelength samples and the sensor RGB response: measured Nikon D700 spectral
    sensitivities (Jiang et al. 2013, see camspec.py), a full-frame body that takes the
    AI-S reference lens. Linearly interpolated from the 10 nm data.
    48 samples: the corner PSF changes 4.5% from 12 to 24 samples and 1.3% from 24 to 48
    (convergence check, decision log 2026-10-17)."""
    wavelengths_um: tuple = tuple(np.round(np.linspace(0.42, 0.68, 48), 4))
    camera: str = "Nikon D700"

    def rgb_weights(self):
        """(n_wavelengths, 3) weights; each column sums to 1 (white-balanced)."""
        s = camspec.sensitivities(self.camera)
        lam_nm = 1000 * np.asarray(self.wavelengths_um)
        w = np.stack([np.interp(lam_nm, camspec.WAVELENGTHS_NM, s[:, c]) for c in range(3)], 1)
        return w / w.sum(0, keepdims=True)


@dataclass(frozen=True)
class LensConfig:
    camera: Camera = field(default_factory=Camera)
    aperture: Aperture = field(default_factory=Aperture)
    lens: Prescription = NIKKOR_35_F2
    trace: Trace = field(default_factory=Trace)
    scatter: Scatter = field(default_factory=Scatter)
    spectral: Spectral = field(default_factory=Spectral)
    kernel_px: int = 127            # diffraction PSF size (odd); scatter halo is separate
    grid: tuple = (6, 8)            # PSF nodes (rows, cols), including frame edges/corners

    def hfov_deg(self):
        """Horizontal field of view of the camera + lens (pinhole, distortion ignored)."""
        half_w_mm = 0.5 * self.camera.width_px * self.camera.pixel_pitch_um * 1e-3
        return math.degrees(2 * math.atan(half_w_mm / self.lens.efl_mm))


DEFAULT = LensConfig()

# Fast settings for iterating on the pipeline (not for reported numbers).
QUICK = LensConfig(
    kernel_px=63,
    grid=(3, 4),
    trace=Trace(rays=61, table_px=129),
    spectral=Spectral(wavelengths_um=tuple(np.round(np.linspace(0.42, 0.68, 6), 4))),
)
