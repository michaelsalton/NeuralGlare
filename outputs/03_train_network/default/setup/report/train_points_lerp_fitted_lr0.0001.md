# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (fitted pyramid, energy-normalised, t_pass 0.0001, before training) | 0.0811 | 0.0825 | 0.1137 | 1.40x | 2.03e-01 | 2.07e-01 | -0.00% |
| **Trained (this run)** | 0.0789 | 0.0807 | 0.1101 | 1.39x | 1.42e-01 | 1.39e-01 | -0.00% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 0.00%. Learning rate 0.0001; training loss (all 9 frames) 0.0924 at step 0, 0.0639 at step 2000.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **True**
- (2) closes the edge gap (edge FLIP < 0.066 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.012): **True**
