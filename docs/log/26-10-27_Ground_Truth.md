- MLPs will learn astigmatism, coma, and similar abberations - this is good
- Those abberations are the spatial variation

- Glare is the tail of the PSF, physically the only effect is the PSF, which is the image a single point of light makes through the lens, every lens effect that affects the shape of that image is part of the PSF:
    - Diffraction from aperture blades: starburst pattern
    - Coma, astigmatism, field curvature: stretch, skew, and flatten the starburst toward the edges of the frame
    - Axial and lateral color: per-wavelength defocus and shift, which show up as color fringes
    - Cat's-eye vignetting: off-axis, lens rims clip the pupil into a lens shapoe, so the starburst changes shape

- "glare" is the faint outer part of the PSF - its only seen around bright HDR sources because everywhere else it is many orders of magnitude below the pixel value
- if these abberations were taken out, the PSF would be the same everywhere, a plain FFT bloom would already be correct

- The rule is keep an effect if it is a convolution (it changes the shape of the light spread, drop it if it doesnt)

    - All abberation orders (coma, astigmatism, spherical, fild curvature): Yes - Real-ray wavefront W from the lens patent prescription
    - Axial and lateral color: Yes - Wavefront traced at 48 wavelengths, weighted by lens/camera sensor curves
    - Cat's-eye vignetting as pupil shape: Yes - Rim clipping the aperture mnask A (psf.py)

    - Vignetting as brightness fallof: Removed - Each PSF is normalized to sum to 1, so edge pixels lose no energy
    - Distortion: Removed - Each PSF is centred on its chief ray
    - Veiling-glare scatter: Off - energy_fraction = 0 (no citable data, it does not vary across frame)
    - Ghosts (internal reflections): Not modelled - Not a convolution (doesnt change the shape of light spread)

- Abberations also widen the bright core, so a correct reference softens the whole image toward the corners, so the network will learn this softening too

- For a research reference this is correct physics, in a game environment it might be desirable to have glare that adds the halo wiuthout blurring textures at the edges of the screen, one option is to split the PSF into a sharp core plus a trail and train only the tail (roughly PSF − δ, where δ is a single-pixel spike holding the core's energy). Or accept the softening as part of the lens look - this is a decision to make.