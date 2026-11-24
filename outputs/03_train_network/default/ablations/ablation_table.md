# Phase 2, step 2: ablation table (default)

Noise floor δ_baseline = 0.000148: the range of validation FLIP over `baseline` seeds 0, 1, 2. Noise floor δ_coma = 0.000218: the range of validation FLIP over `coma` seeds 0, 1, 2. Noise floor δ_spikes_full = 0.003083: the range of validation FLIP over `spikes_full` seeds 0, 1, 2. Edge gap closed = (0.1100 − edge) / (0.1100 − 0.0657): from the Step 1 baseline's edge FLIP to the best uniform kernel's. Test FLIP: medians over the 35 point-grid windows per H bin.

| Change | Study | Val FLIP | Δ vs compared | Test FLIP c / m / e | edge/centre | Edge gap closed | Params | MACs / texel / level | Criterion | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| Step 1 baseline: MLP(uv) | `baseline` | 0.08181 | - | 0.0789 / 0.0806 / 0.1100 | 1.39x | +0% | 390 | 48 | - | current best 2026-11-03, superseded 2026-11-04 |
| Radial input: MLP(H) | `radial_input` | 0.08226 | +0.00045 (+3.1δ_baseline) | 0.0801 / 0.0806 / 0.1109 | 1.38x | -2% | 294 | 32 + length() | rejected | rejected (decision log 2026-11-04) |
| Field-aligned offsets: 4 taps, softmax(MLP(uv)) | `coma` | 0.08110 | -0.00071 (-4.8δ_baseline) | 0.0783 / 0.0797 / 0.1096 | 1.40x | +1% | 696 | 96 + 2 bilinear taps, softmax, length() | kept | kept, current best 2026-11-04, superseded 2026-11-05 |
| Spikes: 14 streaked rays on mip 0, softmax(SpikeMLP(uv)) | `spikes` | 0.06895 | -0.01216 (-55.7δ_coma) | 0.0661 / 0.0676 / 0.0926 | 1.40x | +39% | 819 | 96 + 2 bilinear taps, softmax, length(); spikes: 168 fetches per mip-0 texel, 80 MACs + 2 fetches per full-res texel | kept | kept, current best 2026-11-05, superseded 2026-11-05 |
| Full-resolution rays: streaked from the input, not mip 0 | `spikes_full` | 0.05398 | -0.01497 (-68.6δ_coma) | 0.0473 / 0.0520 / 0.0786 | 1.66x | +71% | 819 | 96 + 2 bilinear taps, softmax, length(); spikes: 168 fetches per full-res texel, 80 MACs + 2 fetches per full-res texel | kept | kept, current best 2026-11-05, superseded 2026-11-10 |
| Per-channel weights: one softmax per colour channel | `per_channel` | 0.05179 | -0.00219 (-0.7δ_spikes_full) | 0.0447 / 0.0497 / 0.0756 | 1.69x | +78% | 1737 | 224 + 2 bilinear taps, 3 softmaxes, length(); spikes: 168 fetches per full-res texel, 176 MACs + 2 fetches per full-res texel | inconclusive: run seeds 1, 2 | no result row of its own; the base of mip_levels, which reports it at -10.0δ_coma vs spikes_full (decision log 2026-11-10) |
| 2 mips instead of 6 | `mip_levels` | 0.05108 | -0.00071 (-3.3δ_coma) | 0.0430 / 0.0495 / 0.0754 | 1.75x | +78% | 729 | 224 + 2 bilinear taps, 3 softmaxes, length(); spikes: 168 fetches per full-res texel, 176 MACs + 2 fetches per full-res texel | kept | kept over 6 mips, superseded by 1 mip (decision log 2026-11-10) |
| 1 mip: the full-resolution level only | `mip_levels` | 0.04603 | -0.00576 (-26.4δ_coma) | 0.0358 / 0.0435 / 0.0745 | 2.08x | +80% | 477 | 224 + 2 bilinear taps, 3 softmaxes, length(); spikes: 168 fetches per full-res texel, 176 MACs + 2 fetches per full-res texel | kept | kept, current best 2026-11-10, superseded 2026-11-13 |
| Tent upsample: the low tap is a 3x3 tent, not bilinear | `tent_upsample` | 0.04533 | -0.00070 (-3.2δ_coma) | 0.0351 / 0.0430 / 0.0734 | 2.09x | +83% | 477 | 224 + 2 bilinear taps, 9-tap tent low fetch, 3 softmaxes, length(); spikes: 168 fetches per full-res texel, 176 MACs + 2 fetches per full-res texel | kept | kept 2026-11-13, superseded the same day by lr 1e-3 |
| Learning rate 1e-3 (was 1e-4) | `tent_upsample` | 0.04476 | -0.00056 (-2.6δ_coma) | 0.0338 / 0.0430 / 0.0724 | 2.14x | +85% | 477 | 224 + 2 bilinear taps, 9-tap tent low fetch, 3 softmaxes, length(); spikes: 168 fetches per full-res texel, 176 MACs + 2 fetches per full-res texel | kept | kept, current best 2026-11-13; lr 1e-3 becomes the default |
| Fifth tap: the own-mip 3x3 tent | `shoulder_tap` | 0.04469 | -0.00008 (-0.4δ_coma) | 0.0336 / 0.0430 / 0.0722 | 2.15x | +85% | 528 | 272 + 2 bilinear taps, tent low fetch, +9 fetches (own tent), 3 softmaxes, length(); spikes as above | inconclusive: run seeds 1, 2 | no result row in the decision log; core_loss and wide_core build on it |
| Core loss: + 0.02 x encircled energy by ring (training only) | `core_loss` | 0.04483 | +0.00014 (+0.7δ_coma) | 0.0337 / 0.0431 / 0.0723 | 2.15x | +85% | 528 | 272 (no runtime cost: a loss term) | inconclusive: run seeds 1, 2 | no result row in the decision log; λ = 0.02 used from wide_core on |
| Sixth tap: the own-mip 5x5 binomial | `wide_core` | 0.04462 | -0.00006 (-0.3δ_coma) | 0.0341 / 0.0435 / 0.0719 | 2.11x | +86% | 579 | 320 + 2 bilinear taps, tent low fetch, +18 fetches (own tent, binomial), 3 softmaxes, length(); spikes as above | inconclusive: run seeds 1, 2 | kept by its own rule (core loss < 0.6, FLIP within δ/√3), current best 2026-11-17, superseded 2026-11-19 |
| Per-ray cross filter: 3 taps across each ray, softmax(RayMLP) | `ray_cross` | 0.04121 | -0.00342 (-15.6δ_coma) | 0.0316 / 0.0376 / 0.0678 | 2.15x | +95% | 638 | 320 as wide_core; rays: +14 dispatches, +42 fetches and 14 RayMLP<3, 8, 3> per full-res texel | kept | kept, current best 2026-11-19, superseded 2026-11-23 |
| Per-ray brightness: 7 softmax(BrightMLP) per family | `ray_bright` | 0.04093 | -0.00028 (-1.3δ_coma) | 0.0317 / 0.0384 / 0.0659 | 2.08x | +99% | 679 | 320 as ray_cross; rays: + 28 BrightMLP<3, 8, 1> per full-res texel, no new fetches | kept | kept, current best (decision log 2026-11-23) |

Ceiling analysis (`ceiling`, ceiling.py): the exact reference with one feature removed, scored like the models. The ring-median oracles are the best round kernels in the absolute log loss; the ring-mean ones aren't ceilings.

| Oracle | Test FLIP c / m / e | energy edge |
|---|---|---|
| Oracle: identity (control) | 0.0000 / 0.0000 / 0.0000 | -0.0% |
| Oracle: round, ring mean (no spikes / coma) | 0.1052 / 0.1058 / 0.1213 | +0.0% |
| Oracle: grey (no colour) | 0.0087 / 0.0098 / 0.0152 | -0.0% |
| Oracle: round + grey, ring mean | 0.1048 / 0.1053 / 0.1211 | -0.0% |
| Oracle: round, ring median (best round, abs log loss) | 0.0742 / 0.0763 / 0.1055 | -9.9% |
| Oracle: round + grey, ring median (best of Step 1's class, abs log loss) | 0.0746 / 0.0765 / 0.1057 | -15.2% |
| Oracle: aperture-symmetric, 14-fold median (INVALID: control failed, decision log 2026-11-04) | 0.0305 / 0.0384 / 0.0706 | -8.1% |
| Oracle: polygon-symmetric, 7 rotations + mirror, round core (INVALID: control failed, interpolation floor, decision log 2026-11-04) | 0.0212 / 0.0326 / 0.0671 | -9.9% |
| Oracle: identity from the 5x bank, summed to pixels (sampling control) | 0.0000 / 0.0000 / 0.0000 | -0.0% |
| Oracle: polygon-symmetric at 5x, round core (spikes kept, no coma) | 0.0207 / 0.0386 / 0.0723 | -9.9% |
| Coma split: core symmetrised (round core), rays and glow exact | 0.0006 / 0.0009 / 0.0013 | -9.8% |
| Coma split: rays symmetrised (+-2 px band), core and glow exact | 0.0165 / 0.0339 / 0.0510 | -0.1% |
| Coma split: glow symmetrised, core and rays (+-2 px) exact | 0.0048 / 0.0065 / 0.0317 | -0.0% |
| Coma split: rays and glow symmetrised, core exact | 0.0204 / 0.0382 / 0.0718 | -0.1% |
| Coma split: core and glow symmetrised, rays exact | 0.0053 / 0.0073 / 0.0327 | -9.8% |
| Coma split: core and rays symmetrised, glow exact | 0.0169 / 0.0343 / 0.0516 | -9.9% |
| Coma split: rays symmetrised (+-3 px band, robustness) | 0.0171 / 0.0349 / 0.0546 | -0.1% |
| Coma split: glow symmetrised (+-3 px ray band, robustness) | 0.0038 / 0.0042 / 0.0218 | -0.0% |
| Ray follow-up: 5.25-9 px annulus symmetrised, the rest exact | 0.0005 / 0.0012 / 0.0023 | -0.1% |
| Ray follow-up: rays beyond 9 px symmetrised, the rest exact | 0.0163 / 0.0333 / 0.0498 | -0.1% |
| Ray follow-up: glow beyond 9 px symmetrised, the rest exact | 0.0048 / 0.0065 / 0.0317 | -0.0% |
| Ray follow-up: rays beyond 9 px symmetric in shape, exact in light per ray | 0.0156 / 0.0305 / 0.0396 | -0.0% |
| Ray profile: rays beyond 9 px symmetric across, exact in light per ray and segment | 0.0139 / 0.0279 / 0.0370 | -0.0% |
