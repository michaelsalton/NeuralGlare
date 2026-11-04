# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF | 0.0481 | 0.0994 | 0.2044 | 4.25x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) | 0.2661 | 0.2661 | 0.3470 | 1.30x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (fitted pyramid, before training) | 0.2661 | 0.2661 | 0.3470 | 1.30x | 2.04e-01 | 2.09e-01 | +1.70% |
| **Trained (this run)** | 0.8400 | 0.6349 | 0.6100 | 0.73x | 2.58e-01 | 1.97e-01 | -65.88% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 33.01%. Learning rate 0.001; training loss (all 9 frames) 0.0932 at step 0, 0.1083 at step 2000.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **False**
- (2) closes the edge gap (edge FLIP < 0.204 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.048): **True**
