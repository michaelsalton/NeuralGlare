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

FFT (Fast Fourier Transform)
    - Fast algorithm for computing the Fourier transform of a signal, such as a 2D image.
    - any image can we written exactley as a sum of 2D sine/cosine waves - each wave has:
        - a frequency: how many stripes per pixel and what direction
        - an amplitude: how strong that wave is
        - a phase: where the stripes are shifted to
    - the Fourier transform takes the image (pixel values at positions) and gives back a sescond oimage of the same size (the frequency domain). Each pixel of the second image is one waves amplitude and phase, stored as a complex number
    - low frequencies hold the smooth, blurry structure, high frequencies hold the edges and fine detail - thats why mipmapping is a low-pass filter: it throws away high frequencies that would alias
    - bloom/glare is convolution: every pixel gets splatted out in the shape of the PSF - like a giant blue kernel

Airy pattern
    - The diffraction pattern formed when light from a point source passes through a circular aperture such as a camera lens, microscope, or telescope
    - Central bright spot (Airy disk) holds about 84% of the total light energy
    - Concentric rings - a series of alternating bright and dark rings (Airy rings) that fade in intensity as they move outward from the center
    - First minimum: the dark ring immedtiaetley surrounding the central desk, marking the edge of the airy disk
    - Larger apertures create smaller airy disks, allowing for sharper higher resolution images
    - Longer wavelengths (like red light) produce larger airy disks than shorter wavelengths (like blue light)
    - Higher f numbers (smaller lens openings) increase the size of the Airy disk, which can eventually limit image sharpness in cameras

Convolution
    - The operation the blur loop performs: at every pixel take a weighted sum of the neighbouring pixels, using the same weights (the kernel) everywhere.
    - Gather (pull) each output pixel reads its neighbours and sums them with the kernel weights - this is how a blur fragment shader is done
    - Scatter (push) each input pixel stamps a copy of the kernel, scaled by its own brightness onto the output, and all the stamps add up
    - A real linear lens is nearly linear (blurring A+B gives the same result as blurring A and B then adding) but its not shift-invariant (the kernel is the same everywhere on screen). Its PSF changes across the field - both the Unreal bloom methods assume shift-invariance

Kernel (in the bloom context)
    - this is a convolutional kernel as in a neural network (not a GPU kernel like CUDA)
    - A grid of weights that slide over an image in a convolution

Unreal's standard bloom
    - Default bloom type, the classic mip-pyramid bloom
    - The HDR scene is repeatedly halved in resolution giving 6 levels, each level is blurred with a Gaussian bllue, then the mips are upsampled and combined and the result is added onto the scene
    - ~24 parameters

Unreal's FFT Bloom
    - PSF is just a texture, the convolution kernel, that gets assigned in the post process volume
    - Take the HDR scene color and the kernel texture, and convolve the scene with the kernel using an FFT, then add that result back to the scene
    - an arbitrary uniform kernel

Unity's bloom
    - Standard mip-pyramid bloom
    - Less expensive then Unreal's bloom because of less properties, like not per band chromatic spread or independent widths and tints, just a single width and a single tint
    - ~5 parameters

Neural Bloom Lighting paper
    - Trains a small CNN to imitate Unity's URP bloom, doesnt model a lens or physical PSF

HDR (high dynamic range)
    - scene-referred linear radiance
    - pixel values proportional to the amount of light with no upper limit, stored as floating point - sun might be 50,000 and a shadow might be 0.05
    - A real lens PSF falls off by around 4-6 orders of magnitude from the core to the outer halo and spikes
    - the tail is only visible around sources bright enough to lift it above the rest of the image
    - clamping the brightness would destroy any visible glare, a sun pixel of 50,000 adds about 5 to its neighbours, which is a bright visible halo, so a shadowed wall at 0.05 would go up to 5.05. If the sun was clamped to 1.0 the same tail would get 0.0001 added to it, which is invisible
    - EXR is the standard HDR image file format, float16 reaches up to about 65,504 with roughly 3 significant digits, which is usually enough for scene radiance - PNG or 8-bit formats would clamp and quantise
