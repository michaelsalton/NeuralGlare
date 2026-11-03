# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (t = 0.5, before training) | 0.9675 | 0.9677 | 0.9682 | 1.00x | 5.04e+00 | 5.74e+00 | -0.31% |
| **Trained (this run)** | 0.7909 | 0.7700 | 0.7872 | 1.00x | 1.53e+00 | 1.53e+00 | -0.02% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 0.00%. Learning rate 0.0001; training loss (all 9 frames) 2.3731 at step 0, 0.7240 at step 2000.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **False**
- (2) closes the edge gap (edge FLIP < 0.066 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.012): **True**
