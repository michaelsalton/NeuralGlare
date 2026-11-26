The current model

- Learned blend weights on pyramid bloom can not acheive the abberations needed to represent the spatially varying nature of lens glare
- The log error improves by about 30% compared to regulare pyramid bloom
- but FLIP metric does not beat the tuned URP bloom
- The current architecture can only blend round, grey, centered blurs
- Whats missing:
    - Diffraction spikes: thing, long and directional
    - Comma towards the corners: lopsided, smeared away from the centre
    - Color fringing: Red, green blue, spread differently

The next model

- Per channel weights (seperate t for R, G, B): Color fringing
- More taps per level, at offsets: Lopsided / stretched shapes
- Offsets aligned to the direction from the screen centre: coma, which always points away from the centre
- Radial input (H instead of raw uv): the non-physical diagonal tilt it learned
- Spikes: taps, or a seperate cheap spike term - the starburst

- Each architecutre change will be:
    - Trained with the absolute log loss
    - Selected on the validation frame
    - Scored against the Step 1 baseline: edge FLIP 0.110, against a target of 0.066
    - Checked for parity in Unity
    uv run python experiments/02_slang_plumbing/unity_parity.py compare

- These will all be ablations in the paper: what it fixes, by how much, and at what cost per pixel