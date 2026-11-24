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
| Oracle: aperture-symmetric, 14-fold median (INVALID: control failed, decision log 2026-11-04) | 0.0305 | 0.0384 | 0.0706 | 2.32x | 1.16e-02 | 4.12e-02 | -8.10% |
| Oracle: polygon-symmetric, 7 rotations + mirror, round core (INVALID: control failed, interpolation floor, decision log 2026-11-04) | 0.0212 | 0.0326 | 0.0671 | 3.17x | 5.81e-03 | 4.01e-02 | -9.87% |
| Oracle: identity from the 5x bank, summed to pixels (sampling control) | 0.0000 | 0.0000 | 0.0000 | 1.10x | 5.42e-11 | 5.36e-11 | -0.00% |
| Oracle: polygon-symmetric at 5x, round core (spikes kept, no coma) | 0.0207 | 0.0386 | 0.0723 | 3.49x | 3.53e-03 | 4.11e-02 | -9.95% |
| Coma split: core symmetrised (round core), rays and glow exact | 0.0006 | 0.0009 | 0.0013 | 2.21x | 7.04e-04 | 2.92e-03 | -9.78% |
| Coma split: rays symmetrised (+-2 px band), core and glow exact | 0.0165 | 0.0339 | 0.0510 | 3.09x | 2.63e-03 | 2.79e-02 | -0.12% |
| Coma split: glow symmetrised, core and rays (+-2 px) exact | 0.0048 | 0.0065 | 0.0317 | 6.62x | 1.97e-04 | 1.10e-02 | -0.02% |
| Coma split: rays and glow symmetrised, core exact | 0.0204 | 0.0382 | 0.0718 | 3.52x | 2.83e-03 | 3.81e-02 | -0.14% |
| Coma split: core and glow symmetrised, rays exact | 0.0053 | 0.0073 | 0.0327 | 6.13x | 9.01e-04 | 1.38e-02 | -9.80% |
| Coma split: core and rays symmetrised, glow exact | 0.0169 | 0.0343 | 0.0516 | 3.05x | 3.34e-03 | 3.08e-02 | -9.93% |
| Coma split: rays symmetrised (+-3 px band, robustness) | 0.0171 | 0.0349 | 0.0546 | 3.19x | 2.68e-03 | 3.33e-02 | -0.13% |
| Coma split: glow symmetrised (+-3 px ray band, robustness) | 0.0038 | 0.0042 | 0.0218 | 5.73x | 1.49e-04 | 5.19e-03 | -0.01% |
| Ray follow-up: 5.25-9 px annulus symmetrised, the rest exact | 0.0005 | 0.0012 | 0.0023 | 4.45x | 2.57e-04 | 3.19e-03 | -0.06% |
| Ray follow-up: rays beyond 9 px symmetrised, the rest exact | 0.0163 | 0.0333 | 0.0498 | 3.07x | 2.38e-03 | 2.47e-02 | -0.06% |
| Ray follow-up: glow beyond 9 px symmetrised, the rest exact | 0.0048 | 0.0065 | 0.0317 | 6.62x | 1.97e-04 | 1.10e-02 | -0.02% |
| Ray follow-up: rays beyond 9 px symmetric in shape, exact in light per ray | 0.0156 | 0.0305 | 0.0396 | 2.54x | 2.26e-03 | 2.17e-02 | -0.00% |
| Ray profile: rays beyond 9 px symmetric across, exact in light per ray and segment | 0.0139 | 0.0279 | 0.0370 | 2.66x | 2.20e-03 | 2.12e-02 | -0.00% |

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

Spikes vs coma, from the oversampled polygon-symmetric oracle aperture_fine (the invalid 14-fold oracle used interpolation order 1):

| Bin | spike gain: round median - aperture_fine | coma remainder: aperture_fine |
|---|---|---|
| centre | +0.0534 | 0.0207 |
| mid | +0.0377 | 0.0386 |
| edge | +0.0333 | 0.0723 |

Pre-registered questions (decision log 2026-11-04, oversampled polygon-symmetric oracle):
- sampling control: identity_fine FLIP < 0.001 in every bin: **True**
- control: aperture_fine FLIP at the source nearest H = 0 (H = 0.001) < 0.01: 0.0091, **True**
- validity: aperture_fine below round median in every bin: **True**
- a centred spike term could reach the target (aperture_fine edge FLIP < 0.066): **False**
- built first: **field-aligned offsets**

Coma split (decision log 2026-11-17): aperture_fine (0.0207 / 0.0386 / 0.0723) applied only to the named pieces, the exact PSF elsewhere. Pieces: core = r <= 5.25 px (aperture_fine's round core), rays = within the band of a ray line, glow = the rest.

| Oracle | Symmetrised | Ray band | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 |
|---|---|---|---|---|---|
| `sym_core` | core | 2 px | 0.0006 / 0.0009 / 0.0013 | -9.78 % | 0.0004 |
| `sym_rays` | rays | 2 px | 0.0165 / 0.0339 / 0.0510 | -0.12 % | 0.0047 |
| `sym_glow` | glow | 2 px | 0.0048 / 0.0065 / 0.0317 | -0.02 % | 0.0044 |
| `sym_not_core` | rays + glow | 2 px | 0.0204 / 0.0382 / 0.0718 | -0.14 % | 0.0087 |
| `sym_not_rays` | core + glow | 2 px | 0.0053 / 0.0073 / 0.0327 | -9.80 % | 0.0048 |
| `sym_not_glow` | core + rays | 2 px | 0.0169 / 0.0343 / 0.0516 | -9.93 % | 0.0051 |
| `sym_rays_w3` | rays | 3 px | 0.0171 / 0.0349 / 0.0546 | -0.13 % | 0.0053 |
| `sym_glow_w3` | glow | 3 px | 0.0038 / 0.0042 / 0.0218 | -0.01 % | 0.0037 |

Pre-registered questions (decision log 2026-11-17):
- centre control: every masked oracle < 0.01 at the source nearest H = 0: **True**
- validity: each one-piece oracle <= aperture_fine + 0.001 in every bin: **True**
- largest one-piece edge FLIP, built first: **rays** (per-ray weights by angle to the field direction)
- the picked piece alone could reach the target (sym_not_rays edge FLIP < 0.0657): **True**
- rays vs glow order the same with the +-3 px band: **True**

Ray follow-ups (decision log 2026-11-17, an amendment after the coma split): the ray band split at 9 px, and the far rays' brightness restored.

| Oracle | Symmetrised | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 | Bounded by | Valid |
|---|---|---|---|---|---|---|
| `sym_annulus` | annulus (5.25 <= r < 9 px) | 0.0005 / 0.0012 / 0.0023 | -0.06 % | 0.0001 | sym_rays | **True** |
| `sym_rays_far` | rays (+-2 px), r >= 9 px | 0.0163 / 0.0333 / 0.0498 | -0.06 % | 0.0047 | sym_rays | **True** |
| `sym_glow_far` | glow, r >= 9 px | 0.0048 / 0.0065 / 0.0317 | -0.02 % | 0.0044 | sym_glow | **True** |
| `rays_far_bright` | rays r >= 9 px, shape only (light restored per ray) | 0.0156 / 0.0305 / 0.0396 | -0.00 % | 0.0049 | sym_rays_far | **True** |

Pre-registered questions (decision log 2026-11-17, ray follow-ups):
- centre control: every new oracle < 0.01 at the source nearest H = 0: **True**
- validity: each oracle <= its bound + 0.001 in every bin: **True**
- where (sym_rays_far vs sym_annulus at the edge): **rays_far**
- what: brightness gain (sym_rays_far - rays_far_bright, edge) +0.0102 vs shape remainder (rays_far_bright edge - floor 0.0047) +0.0349: **shape** (the rays' shape: an angle-dependent per-ray blur)

Ray profile oracle (decision log 2026-11-17, a second amendment): the far rays' light restored per ray and radial segment (9 / 16 / 32 px).

| Oracle | FLIP c / m / e | Energy, edge | FLIP at H ~ 0 |
|---|---|---|---|
| `rays_far_profile` | 0.0139 / 0.0279 / 0.0370 | -0.00 % | 0.0057 |

Pre-registered questions (decision log 2026-11-17, ray profile oracle):
- centre control: < 0.01 at the source nearest H = 0: **True**
- validity: rays_far_profile <= rays_far_bright + 0.001 in every bin: **True**
- along-ray gain (rays_far_bright - rays_far_profile, edge) +0.0026 vs across-ray remainder (rays_far_profile edge - floor 0.0047) +0.0323: **across** (across the ray: an angle-dependent per-ray blur); within 0.005: **False**
