# Phase 2: train the network against the real PSF field

Phase 1 proved the plumbing on a target the pyramid can represent exactly (URP bloom). Phase 2 asks the research question: **how close can a pyramid of tiny MLPs get to a real lens's spatially varying PSF field, and at what cost?**

Order: point lights first (exact references exist), then the architecture, then real scenes (once their references are fixed).

## Step 1: baseline on point lights

The open reference problem (HDRI crops use the 6×8 interpolated PSF grid, 10–41 % error between nodes) only affects dense images. Sparse point lights already have an exact reference: `experiments/01_gap_study/exact_pointgrid.py` splats each source with the exact PSF at its own pixel. Randomised point-light frames are part of the planned training data (plan §5).

A single bright pixel pushed through the pyramid gives the pyramid's **effective kernel** (impulse response) at that pixel. Training on point sources therefore fits the model's PSF field to the lens's PSF field directly, with no scene content mixed in: the cleanest test of what the architecture can represent.

- [x] Read how `exact_pointgrid.py` computes its targets (PSF per source, cost per source). One exact PSF: **62 s** on one core (48 wavelength FFTs), plus ~2 min of ray tracing per new field height
- [x] Design the point-light data. Random frames are out (1000 sources ≈ 17 h), and unnecessary: model and optics are linear, so a frame is a sum of impulse responses and random intensities or colours teach nothing. The data is **a bank of exact PSFs at fixed positions**; split in `configs/split.toml` (`[phase2_points]`): train = jittered 16 × 9 grid (144, seed 0), test = the Phase 0 7 × 5 point grid (35)
- [x] Generate the bank: `experiments/03_train_network/psf_bank.py` → `data/psf/psf_bank_default.npz` (179 PSFs, 127 px, 48 wavelengths, 8 workers, resumable). Checks: per-position vs shared pupil table 2–3e-6 relative L1; the test PSFs rebuild the Phase 0 exact point-grid reference to 2.4e-8
- [x] Close the resolution gap: a **full-resolution level** after the pyramid, `out = w0 · input + w1 · bilinear(up[0])` (same Upsample kernel, 6th network). Phase 1's `up[0]` is half resolution; the PSF's core is a few pixels wide
- [x] Training frames: the 144 positions packed into 9 fixed frames by (row mod 3, col mod 3), sources ≥ 241 px apart; references splatted from the bank
- [x] Fix the comparison criterion before the first run (decision log 2026-10-31): gap-study metrics on the 35 test sources; (1) beats the fitted pyramid in every H bin, (2) edge FLIP < 0.204 and edge/centre ≤ 2×; expectation: centre stays above 0.048
- [ ] Train (`experiments/03_train_network/train_points.py`): levels 0–4 from Phase 1, log loss ε = 1e-3, Adam 1e-2, 2000 steps
- [ ] Report: how much of the edge FLIP gap (Phase 0: best uniform kernel is 4.25× worse at the edge than at the centre) a mix of Gaussians closes

Expected limit: every level only mixes Gaussians, so every effective kernel is **round, grey and symmetric**. The lens's PSF has:
- 14 diffraction spikes from the 7 aperture blades
- coma: an asymmetric comet tail that grows toward the corners
- lateral colour: R, G and B landing at slightly different radii

This baseline is what every architecture change in step 2 is measured against, and a row in the paper's ablation table.

## Step 2: change the architecture to close the gap

Each change targets a specific part of the PSF:

| Change | What it lets the model represent | Graphics analogy |
|---|---|---|
| **Per-channel weights** (MLP outputs 3 × taps) | Lateral colour, chromatic halos | A separate filter kernel per colour channel |
| **More taps per level**, at offsets around the texel | Anisotropy: coma's skew, elongated cores | From a 1-tap fetch to a small filter footprint, weights chosen per pixel |
| **Offsets aligned to the field direction** (radial / tangential, from uv) | Coma and astigmatism, which point along the line from the image centre | Anisotropic filtering along a direction |
| **Weight constraint** (e.g. weights sum to 1 per level) | Energy conservation; per-level weights become identifiable again (step 4 finding) | A normalised filter kernel |

- [ ] Fix the ablation criterion before the first run
- [ ] One change at a time, each trained on the same point-light data and compared with the step 1 baseline
- [ ] Spikes: find out whether taps can produce them (thin and long, against a pyramid of wide soft blurs) or whether a separate cheap spike term is needed
- [ ] Rerun the Unity parity harness whenever `glare.slang` / `mlp.slang` / the weight layout change
- [ ] Decision-log rows for taps per level, per-channel weights and the weight constraint

## Step 3: real scenes with a fixed reference

Once the architecture is settled, the model has to generalise to dense real frames, which need the interpolation error fixed.

- [ ] Grid-density convergence study: exact PSF halfway between nodes vs interpolated, increase node count until the error is below a threshold fixed in advance (e.g. 5 %); same pattern as `convergence.py`
- [ ] Measure the cost first: the reference does one float64 FFT convolution per node, so 4× the nodes is 4× the time
- [ ] Regenerate the HDRI-crop references; close the open issue in the decision log (2026-10-17)
- [ ] Train on point lights + crops, evaluate on held-out scenes

Real training scenes (Sun Temple, Emerald Square, URP samples) need the Unity capture feature (Phase 3 work). Independent of the above; can be built any time.