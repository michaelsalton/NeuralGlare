Pixel pitch
    - Physicaly distance between the centres of neighbouring sensor pixels in µm. 
    - I'ts a spatial size, it's what turns a PSF measured in µm into a PSF measured in pixels.

PSF (Point Spread Function)
    - the image an optical system makes of a single, infintiely small point of light.
    - A perfect lens would turn a point of light int oa perfect point on the sensor, real lenses don't do this. The light always spreads out into a small pattern, that pattern is the PSF
    - The bloom kernel, derived from physics
    - In pyramid bloom you blur bright pixels with a Gaussian blur. The PSF is the kernel nature actually applies - every pixels light gets spread by it.
    - It conserves energy - A PSF sums to 1 - it moves liught around without adding or removing any
    - Diffraction causes the bright core and the spikes from the straight edges of the aperture blades (6 aperture blades give 6 spikes)
    - Abberations (lens defects) smear and skew the core, like the coma fans and astigmatism
    - Wavelength - red spreads wider than blue, which gives color fringes
    - Scatter (participating media) causes a very wide, faint haze
    - A real lens has a different PSF at every position of the sensor - Abberations grow toward the edges of the frame - real time methods including Unreal's FTT bloom use one PSF for the whole image

