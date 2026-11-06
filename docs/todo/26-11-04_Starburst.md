What the spikes are

- Spikes carry most of the halo, they hold 65-70% of the light between 5 and 63 px
- Beyond 10 px the glare is mostly spikes
- A pyramid of round blurs can only spread that light into a disc, which is why the ceiling analysis gave spikes a gain bound of 0.053 at the centre

- The angles are fixed across the whole frame, they sit at k*360/4 to within 0.25 degrees, because the diaphram doesnt rotate
- Unlike coma nothing here is field aligned
- Only the strength changes with H: peak contrast drops about 20x at the edge, where coma smears the spikes, but their share of the energy stays at 54-70%

- The rays are thin, 1-2.5x px wide, coarse pyramid levels cant make them, a level-3 texel is already 16 px

- Two families and every line through the source has one ray of each
- The ray oppsite k is ray k + 7, and since 7 is odd, the two always belong to different families, so each of the 7 lines needs an asymmetric 1-D kernel, the even profile on one side, odd on the other
- the physics: a heptagon isnt point-symmetric, the PSF would still be point symmetric if the wavefront were perfect but the lens abberations break that

- Color: the spikes are reddish (R/G 1.17 and B/G 0.87, 0.65 at the edge)
- Diffraction spreads lkonger wavelengths further

- The reference truncates the spikes at 63px, because the PSF kernel is 127 px across, thats a property of the reference, not real optics

Design

- Classic star / streak filter (Kawase), a long directional blur is built from a few short passes with growing tap spacing
- For each of the 14 rays:

S_0     = mip0
S_{p+1} = Σ_{n=0..3} c_{f,p,n} · S_p(x − n · 4^p · e_k)       p = 0, 1, 2   (taps 0..63 texels)

- The passes act like digits, pass p moves light 0-3 * 4^p texels, so 3 passes reach every distance from 0 to 63 texels exactley once, as three base-4 digits, a point source therefore gets a continuous ray, not dots
- The profile is a product of per-digit weights, it can approximate the falloff
- Each pass's weights are positive and sum to 1 so a ray keeps its energy
- Each family gets one learned profile, shared by its 7 rays: 2 * 3 * 4 = 24 global paramenters, that is the even/odd assymetry

New Network

- SpikeMLP<2, 16, 3>

w   = softmax(SpikeMLP(uv))                         (pyramid, even family, odd family)
out = w0 · pyramid_out + w1 · up(Σ_even S_3)/7 + w2 · up(Σ_odd S_3)/7

- MLP predicts weights only, its linear with postive weights summing to 1, like the pyramid MLP

Cost

- Fetches: 14 rays x 3 passes x 4 taps = 168 bilinear fetches per half-res texel