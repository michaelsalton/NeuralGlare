"""Optical configuration for the reference glare model.

!!! PLACEHOLDER VALUES !!!
The Seidel coefficients, vignetting and scatter numbers below are rough, plausible
values so the pipeline can run end to end. Before ANY result is reported:
  - replace the Seidel coefficients with third-order coefficients computed from a
    published lens prescription (e.g. a double-Gauss or Cooke triplet from a textbook
    or patent), and cite the source next to each value;
  - replace the scatter (veiling glare) parameters with values from a cited
    measurement or glare-spread-function model;
  - set PLACEHOLDER = False.
psf.py prints a warning while PLACEHOLDER is True.
"""

from dataclasses import dataclass, field, replace

import numpy as np

PLACEHOLDER = True


@dataclass(frozen=True)
class Camera:
    width_px: int = 1920
    height_px: int = 1080
    # 36 mm full-frame sensor width / 1920 px. Together with the 60 deg HFOV used in
    # make_inputs.py this implies a ~31 mm focal length.
    pixel_pitch_um: float = 18.75
    f_number: float = 2.8           # operating aperture


@dataclass(frozen=True)
class Aperture:
    blades: int = 6                 # 6 blades -> 6-point starburst
    rotation_deg: float = 0.0
    edge_softness: float = 1.0      # anti-aliasing width of the pupil edge, in pupil samples
    # Mechanical vignetting: the aperture is intersected with a circle of radius
    # `vignette_radius` (relative to the aperture circumradius), shifted by
    # `vignette_shift` * H along the field direction (H = 1 at the image corner).
    vignette_radius: float = 1.3    # PLACEHOLDER
    vignette_shift: float = 0.6     # PLACEHOLDER


@dataclass(frozen=True)
class Seidel:
    """Wavefront aberration coefficients, in waves at `lambda_ref_um`, defined at
    `ref_f_number` and full field (H = 1).

    W(rho, theta, H) = W040 rho^4 + W131 H rho^3 cos(theta)
                     + W222 H^2 rho^2 cos^2(theta) + W220 H^2 rho^2
    W311 (distortion) is deliberately omitted: it is a geometric image warp, not glare.
    Chromatic aberration (wavelength-dependent coefficients) is not modelled yet.
    """
    ref_f_number: float = 2.8
    lambda_ref_um: float = 0.55
    W040: float = 0.3   # spherical          PLACEHOLDER
    W131: float = 0.8   # coma               PLACEHOLDER
    W222: float = 0.6   # astigmatism        PLACEHOLDER
    W220: float = 0.4   # field curvature    PLACEHOLDER

    def at_f_number(self, f_number):
        """Rescale to another aperture. Each term scales with pupil radius^k, where k
        is the power of rho in that term, and pupil radius is proportional to 1/F#."""
        s = self.ref_f_number / f_number
        return replace(self, ref_f_number=f_number,
                       W040=self.W040 * s**4, W131=self.W131 * s**3,
                       W222=self.W222 * s**2, W220=self.W220 * s**2)


@dataclass(frozen=True)
class Scatter:
    """Wide, low-energy veiling-glare halo from scattering (glass, coatings, dust).
    Kernel  S(r) ∝ (r^2 + r0^2)^(-exponent/2), normalised to sum 1.
    Final PSF = (1 - energy_fraction) * diffraction PSF + energy_fraction * S."""
    energy_fraction: float = 0.02   # PLACEHOLDER
    core_radius_px: float = 2.0     # PLACEHOLDER
    exponent: float = 2.5           # PLACEHOLDER


@dataclass(frozen=True)
class Spectral:
    """Wavelength samples and an approximate sensor RGB response (Gaussians).
    PLACEHOLDER: replace with measured camera or CIE-derived responses."""
    wavelengths_um: tuple = tuple(np.round(np.linspace(0.42, 0.68, 12), 4))
    peaks_um: tuple = (0.600, 0.540, 0.460)    # R, G, B
    sigma_um: float = 0.040

    def rgb_weights(self):
        """(n_wavelengths, 3) weights; each column sums to 1 (white-balanced)."""
        lam = np.asarray(self.wavelengths_um)[:, None]
        w = np.exp(-0.5 * ((lam - np.asarray(self.peaks_um)[None, :]) / self.sigma_um) ** 2)
        return w / w.sum(0, keepdims=True)


@dataclass(frozen=True)
class LensConfig:
    camera: Camera = field(default_factory=Camera)
    aperture: Aperture = field(default_factory=Aperture)
    seidel: Seidel = field(default_factory=Seidel)
    scatter: Scatter = field(default_factory=Scatter)
    spectral: Spectral = field(default_factory=Spectral)
    kernel_px: int = 127            # diffraction PSF size (odd); scatter halo is separate
    grid: tuple = (6, 8)            # PSF nodes (rows, cols), including frame edges/corners


DEFAULT = LensConfig()

# Fast settings for iterating on the pipeline (not for reported numbers).
QUICK = LensConfig(
    kernel_px=63,
    grid=(3, 4),
    spectral=Spectral(wavelengths_um=tuple(np.round(np.linspace(0.42, 0.68, 6), 4))),
)