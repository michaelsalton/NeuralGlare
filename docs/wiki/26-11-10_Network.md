- Large Halo around center - what is causing this?
- Look into L1/L2 loss
- Does the pyramid structure architecture cause any type of "pyramid artifacts" - like the large halo around center, this is also present in URPs pyramid bloom - is this carrying over to my model because of the pyramid format ?

- Halo is caused by the pyramid starting point - fitted URP pyramid, its the coarsest blur level kept from there
- Training has been shrinking it but hasnt been able to get rid of it
- The pyramid can only make round shapes, and the optimization is held near its starting point

Things to try to mitigate this
- Confirm its mip1 - in a copy of trained params, set level0s low tap logits very negative, render views, check the halo disapears - this rules out any other cause
- Give the optimiser more room - re run per channel with --lr 1e-3 and more steps, if full rest climbs and level 0s t falls, the core will fill in and the disc will shrink
- start from the reference, not from URP - set full res t to the references share of light outside the centre pixel, which can compute from the PFS bank - set level 0s t small, so that light starts in mip0, the shoulder, rather than mip1, the disc, 

Rays
- Right now rays only vary slightly in brightness spatially but the shape and angles dont change
- To fix this:
    - One weight per ray instead of per family
    - Fall-off that depends on uv
    - Something that puts lopsides light in the gaps (the coma smear)
