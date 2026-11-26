A real lens's 3D behaviour is captured

- For a distant scene, every pixel of an HDR frame is light arriving from one direction. Inside the lens, everything that happens to that light depends on only two things:
    - its direction: which is the screen space position or field height H
    - its wavelength
- This covers which element rims clip the ray bundle (cats-eye vignetting), how the diaphram cuts it, and the wavefront error from every surface.
- lens_trace.py traces real rays through the real prescription
- Field-varying PSF model models how a real lens captures light

What a flat image misses - the pinhole assumption

- The HDR frames are pinhole images: the scene seen from a single point. A real lens is not a single point. At f/2.8 with f = 36mm, the entrance pupil is about 36 / 2.8 ~= 12.9 mm wide, so the sensor is really averaging what the scene looks like from every point across that 12.9 mm disc
- The PSF formula assumes each scene point lights the whole pupil. That holds unless something in the scene is close enough that different parts of the pupil see different things, in that case:
    - A bright source that's just behind an occluder's edge lights only part of the pupil
    - The effective aperture A(u) for that source is no longer the 7-blade polygon. It's the polygon with a chunk cut off
    - A different aperture gives a different PSF, the starburst changes shape, the cut edge adds its own diffraction streak, and the energy drops
How big is this effect?

Other assumptions

- No depth dependence: real PSFs change with object distance (defocus). NG treats the whole scene as in focus - this matches how engines work, since DOF and bloom are seperate passes
- No off-screen sources: a sun just outside the frame still causes real veiling glare and flare - a 2D frame can't know its there, in practice this is probably a bigger gap than pupil occlusion
- Ghost flares arent modelled: Internal reflections between lens elements (Hullin 2011) is out of scope for NG