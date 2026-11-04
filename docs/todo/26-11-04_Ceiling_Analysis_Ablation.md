In 3_error_ratio.png and 4_radial_profile.png, the Step 1 model's errors look almost the same at H = 0 (the centre) as at H = 0.88 (near a corner):

Missing spikes: orange rays, meaning too dark, in every row.
Wrong halo shape: a square blue blob, too bright, about 5–20 px from the source. The reference falls off smoothly there.
A cliff at about 40 px: the model drops to zero, but the reference keeps a long faint tail. Levels 3–4 are frozen at t = 1e-4, so the widest blurs contribute almost nothing.
Core shoulder: at 1–5 px the model is about 1000× too dark.

- The changes that will improve the most are more taps per level, which will give the halo shape, and spieks which will give the starburst

Measure an error budget

- Build the reference with one feature removed, and score it aginst the full reference, this tells how much error that feature can be responsible for, before building anything.

3 oracles

- Round oracle
- replace each bank PSF with its average over an angle around the source: average luminance in thin rings, spun back into 2D
- This removes the spikes and coma but keeps the exact radial profile. FLIP (reference, round oracle) is the best any round kernel can do
- If that floor is already near or above 0.066, no round model reaches the target, and spikes are required
- If its far below, the gap is mostly the halo shape, which is what more taps would fix

- Grey oracle
- Replace each PSFs color with its luminenance times the PSFs average color
- FLIP(refernce, grey oracle) shows how much color matters, which tells whether per channel t is worth the cost

- Center-kernel oracle
- This is done, FFT with the centre PSF: 0.012 at center, 0.066 at the edge, measures how much spatial variation matters at all

- Score these 4:
    - Identity: the control
    - round: removes spikes, coma, any lopsidedness. Keeps color
    - grey: removes color. keeps shape
    - roung_grey: removes both