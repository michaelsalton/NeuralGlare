# Phase 2, ceiling: ceiling analysis on the point grid (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF (sq fit) | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) (sq fit) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Step 1 baseline: MLP(uv) | 0.0789 | 0.0806 | 0.1100 | 1.39x | 1.41e-01 | 1.37e-01 | -0.00% |
| Oracle: identity (control) | 0.0000 | 0.0000 | 0.0000 | 1.54x | 9.32e-14 | 3.11e-13 | -0.00% |
| Oracle: round, ring mean (no spikes / coma) | 0.1052 | 0.1058 | 0.1213 | 1.15x | 8.76e-02 | 7.91e-02 | +0.00% |
| Oracle: grey (no colour) | 0.0087 | 0.0098 | 0.0152 | 1.74x | 4.97e-04 | 6.17e-03 | -0.00% |
| Oracle: round + grey, ring mean | 0.1048 | 0.1053 | 0.1211 | 1.16x | 8.89e-02 | 7.84e-02 | -0.00% |
| Oracle: round, ring median (best round, abs log loss) | 0.0742 | 0.0763 | 0.1055 | 1.42x | 8.09e-02 | 7.03e-02 | -9.92% |
| Oracle: round + grey, ring median (best of Step 1's class, abs log loss) | 0.0746 | 0.0765 | 0.1057 | 1.42x | 8.11e-02 | 6.89e-02 | -15.21% |
| Oracle: aperture-symmetric, 14-fold median (INVALID: control failed, decision log 2026-11-04) | 0.0298 | 0.0375 | 0.0703 | 2.36x | 2.00e-02 | 4.71e-02 | -7.66% |

Each oracle is the exact reference with one feature removed from every PSF. The ring-mean oracles aren't ceilings (see the module docstring); the ring-median ones are the best round kernels in the absolute log loss. Medians over the 35 point-grid windows (254 px), as train_points.py.

Median FLIP differences, from the ring-median oracles (a ranking: FLIP isn't additive):

| Bin | profile: Step 1 - round+grey | roundness: round+grey - grey | colour: round+grey - round |
|---|---|---|---|
| centre | +0.0043 | +0.0659 | +0.0004 |
| mid | +0.0042 | +0.0667 | +0.0002 |
| edge | +0.0044 | +0.0905 | +0.0001 |

Pre-registered questions (decision log 2026-11-04, amended):
- control: identity FLIP < 0.001 in every bin: **True**
- validity: ring-median round+grey below Step 1 in FLIP in every bin: **True**
- a round, grey kernel can reach the target (round+grey edge FLIP < 0.066): **False**
- largest difference at the edge, which picks the next ablation: **roundness**

Spikes vs coma, from the aperture-symmetric oracle (rotation interpolation order 3):

| Bin | spike gain: round median - aperture | coma remainder: aperture |
|---|---|---|
| centre | +0.0443 | 0.0298 |
| mid | +0.0388 | 0.0375 |
| edge | +0.0352 | 0.0703 |

Pre-registered questions (decision log 2026-11-04, aperture oracle):
- control: aperture FLIP at the source nearest H = 0 (H = 0.001) < 0.01: 0.0286, **False**
- validity: aperture below round median in every bin: **True**
- a centred spike term could reach the target (aperture edge FLIP < 0.066): **False**
- built first: **field-aligned offsets**
