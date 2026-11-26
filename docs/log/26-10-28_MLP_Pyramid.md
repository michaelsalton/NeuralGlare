The Pyramid: a cheap way to get a very wide blur

- A real glare PSF can cover hundreds of pixels - gathering that many taps per pixel at full resolution is far too slow, the pyramid gets around this the way bloom always has: blue at progressivley lower resolutions
- Downsample chain in pyramid.slang is fixed, it has no learned parameters and copies URP 17.6 Bloom, without the treshold and clamp
    - Prefilter: one bilinear fetch into a half-resolution target, giving mip 0
    - Then for each level, blur_h (9 taps which also halves the resolution) followed by blur_V (the samne Gaussian done as 5 bilinear taps)
- mip_k is the input frame convolved with a Gaussian whose width rougly doubles at each level



The upsample: blending the gaussians back together

- The upsample pass goes back up the chain, in URP each level does this:

out_k = lerp(mip_k, up(out_{k+1}), s)  =  (1−s)·mip_k + s·up(out_{k+1})

- s comes from the scatter slider, if you unroll the recursion, the final image is:

bloom = (1−s)·G₀ + s(1−s)·G₁ + s²(1−s)·G₂ + … + sᴺ·G_N      (G_k = input blurred by Gaussian k)

- Pyramid bloom is a sum of Gaussians at widths 1x, 2x, 4x, 8x, of each other and one scaler sets how much each one contributes
- The effective PSF is the same at every pixel, it is radially symmetric and identical in R, G, B



Where the MLP plugs in

- glare.slang keeps exactley that upsample structure but replaces the constant pair (1-s, s) with the output of a tiny network:

float w[2] = LevelMLP.eval(params, offset, uv);   // MLP_k(screen position)
return w[0] * mip.load(texel) + w[1] * sample_bilinear(low, uv);

- Unrolled the same way the coefficients are now functions of screen position:

bloom(uv) ≈ c₀(uv)·G₀ + c₁(uv)·G₁ + … + c_N(uv)·G_N

- each pixel now gets its own mixture of the same Gaussians, near the centre of the frame the network can favour the narrow levels, and towards the corners it can push weight into the wide ones, gives a spatially varying PSF field
- uv is the MLPs only input, the network is basically a compressed procedural texture of blend weights


Predicting weights

- MLP never sees the HDR pixel values, whatever the network outputs the result is linear in scene light: double the brightness of the sun and the glare doubles, the way real optics behave
- if the network output colors directly it would have to learn that linearity across about 8 orders of magnitude of HDR, and would probably fail


Training

- During training, slangpy differentiates upsample with respect to params, the gradient goes loss --> weighted sum --> W --> MLP weights, the mips carry no gradient (no_diff) which is correct because the downsample chain is fixed