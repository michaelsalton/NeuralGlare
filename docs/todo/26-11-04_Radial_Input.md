Radial Input (H, or uv − 0.5, instead of raw uv) - The non-physical diagonal ramp; the lens is radially symmetric

- Use H as input instead of uv
- H is the distance from the center (of the PSF)

Blend Maps

- A blended map is a picture of what each levels small MLP outputs at every point on the screen
- With the radial input, each map is a single 1D curve, spun around the screen centre

t (URP scatter)

- The upsample pass is the usual URP bloom upsample, a lerp between two taps (texture reads)
- in URP, t is the scatter constant, one number for the whole screen
- Here each pyramid level has its own tiny MLP (1 --> 16 --> 1)
- Its output goes through sigmoid, which squashes any number into the range (0, 1)
- So t is always a valid lerp weight, it can never go negative and never adds or removes light
- A small t keeps this levels sharp mip, which gives a tigther glow
- A large t takes more of the blurrier result from below, which gives a wider halo

What the radial input changes

- The MLPs input is the only thing that lets t vary across the screen
- MLP(uv): the network sees the 2D screen position, so t can be any function of x and y
- MLP(H): the network sees only the field height H, H is the distance from the frame centre, measured in pixels and divided by the half-diagonal, 0 at center, 1 in the corners
- With one input, each levels MLP is a 1D curve, like a 1D LUT (lookup texture) indexed by radius
- Any 2D picture of it must be rings around the centre, because two pixels at the same distance get the same t
- This builds in the rotational symmetry of a real lens, which is why the change was made

Why radial input

- Radial input was meant to make the networks spatial variation physically correct, by building in a symmetry the real lens has
- In the MLP(uv) model the network took raw uv so the blend t could be any 2D function of screen position, the trained model used that freedom to learn diagonal ramps, ex. the full resolution glow got stronger toward the top left and weaker toward the bottom right
- A real lens does not do this, its elements are rotationally symmetric around the optical axis, so how a point light spreads depends only on its distance from the frame centre, not on its direction
- A light at the top left corner and one at the bottom right corner should get the same blur, mirrored
- The PSF bank (the set of reference point spread functions) is built the same way; its indexed by field height H, so the ramp was an artefact, the network was fitting something about these particular frames, not something about the optics

What radial input was meant to acheive

- The radial input was meant to remnove the artefact by construction
- If the MLP sees only H, a diagonal ramp is impossible, every pixel as the same radius gets the same t, its the difference between a vignette sotred as a 2D texture, which can hold any smudge, and one sotred as a 1D falloff curve indexed by radius, which can only hold rings

- Give the network a useful built in assumption
- In ML, an assumption built into the architecture is called an inductive bias, the model trains on only about 144 point lights, with that little data a network free to fit any 2D function can latch onto accidents of the layout, such as where the source happens to sit
- Restricting it to radial functions keeps it to the kinds of variation a lens actually produces
- The hope was that it would generalize as well or better to new frames and scenes

- It was meant to make the network cheaper also, one input instead of two means 49 paramenters per level instead of 65, and half the first layer multiply-adds
That matters for the per pixel cost that will be timed in RT engines (Unity, Unreal)

- Match how the reference is described, the reference lens model is paramaterized by H, soa model indexed by H lines up with it
- Later changes like aligning taps with the direction from the centre for coma, are natually described in these radial terms

The goal

- The goal was correctness, the ablation rule has an exception written for this, keep it if it removes a non-physical artefact or lowers cost, as long as it isnt worse than the seed noise
- Success was:
    - Blend maps becomes rings: Achieved
    - Validation FLIP no worse than seed noie (0.00015): Failed, it was worse by 0.00045, about 3 x seed noise

The result

- The restriction costs real fit, training loss went up, not just validation
- So the diagonal ramp was not noise, it was lowering the error on these frames, probably by exploiting the uneven layout of the point sources
- The validation frame comes from the same point-grid setup, so it rewards the same trick
- Current benchmark cant yet tell physicallyu correct structure from layout-fitting, an argument for testing on mirrored or rotates frames, or on real scenes, before deciding whether physical constraints like this one help