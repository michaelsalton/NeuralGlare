# Phase 2 step 1: Phase 1 architecture on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF (sq fit) | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) (sq fit) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (fitted pyramid, energy-normalised, t_pass 0.0001, before training) | 0.0811 | 0.0825 | 0.1137 | 1.40x | 2.03e-01 | 2.07e-01 | -0.00% |
| **Trained (this run)** | 0.0792 | 0.0805 | 0.1099 | 1.39x | 1.43e-01 | 1.34e-01 | -0.00% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 0.00%. Learning rate 0.0001; abs log loss, training (8 frames) 0.0405 at step 0, 0.0343 at step 2000.

Validation frame (15 sources; selects between runs, decision log 2026-11-03): mean FLIP 0.0840 at init, **0.0819** trained; abs log loss 0.0390 at init, 0.0331 trained.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **True**
- (2) closes the edge gap (edge FLIP < 0.066 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.012): **True**
