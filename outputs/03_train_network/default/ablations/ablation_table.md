# Phase 2, step 2: ablation table (default)

Noise floor δ_baseline = 0.000148: the range of validation FLIP over `baseline` seeds 0, 1, 2. Noise floor δ_coma = 0.000218: the range of validation FLIP over `coma` seeds 0, 1, 2. Edge gap closed = (0.1100 − edge) / (0.1100 − 0.0657): from the Step 1 baseline's edge FLIP to the best uniform kernel's. Test FLIP: medians over the 35 point-grid windows per H bin.

| Change | Study | Val FLIP | Δ vs compared | Test FLIP c / m / e | edge/centre | Edge gap closed | Params | MACs / texel / level | Criterion | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| Step 1 baseline: MLP(uv) | `baseline` | 0.08181 | - | 0.0789 / 0.0806 / 0.1100 | 1.39x | +0% | 390 | 48 | - | current best 2026-11-03, superseded 2026-11-04 |
| Radial input: MLP(H) | `radial_input` | 0.08226 | +0.00045 (+3.1δ_baseline) | 0.0801 / 0.0806 / 0.1109 | 1.38x | -2% | 294 | 32 + length() | rejected | rejected (decision log 2026-11-04) |
| Field-aligned offsets: 4 taps, softmax(MLP(uv)) | `coma` | 0.08110 | -0.00071 (-4.8δ_baseline) | 0.0783 / 0.0797 / 0.1096 | 1.40x | +1% | 696 | 96 + 2 bilinear taps, softmax, length() | kept | kept, current best 2026-11-04, superseded 2026-11-05 |
| Spikes: 14 streaked rays on mip 0, softmax(SpikeMLP(uv)) | `spikes` | 0.06895 | -0.01216 (-55.7δ_coma) | 0.0661 / 0.0676 / 0.0926 | 1.40x | +39% | 819 | 96 + 2 bilinear taps, softmax, length(); spikes: 168 fetches per mip-0 texel, 80 MACs + 2 fetches per full-res texel | kept | kept, current best 2026-11-05, superseded 2026-11-05 |
| Full-resolution rays: streaked from the input, not mip 0 | `spikes_full` | 0.05398 | -0.01497 (-68.6δ_coma) | 0.0473 / 0.0520 / 0.0786 | 1.66x | +71% | 819 | 96 + 2 bilinear taps, softmax, length(); spikes: 168 fetches per full-res texel, 80 MACs + 2 fetches per full-res texel | kept | kept, current best (decision log 2026-11-05) |

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
