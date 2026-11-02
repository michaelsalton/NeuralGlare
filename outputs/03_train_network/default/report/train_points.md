# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF | 0.0481 | 0.0994 | 0.2044 | 4.25x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) | 0.2661 | 0.2661 | 0.3470 | 1.30x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (Phase 1 weights, before training) | 0.9713 | 0.9713 | 0.9712 | 1.00x | 1.17e+01 | 1.19e+01 | +10.75% |
| **Trained (this run)** | 0.6769 | 0.7070 | 0.7901 | 1.17x | 7.71e-01 | 7.10e-01 | +47.82% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 27.68%.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **False**
- (2) closes the edge gap (edge FLIP < 0.204 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.048): **True**
