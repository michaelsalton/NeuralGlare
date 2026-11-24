# Phase 2, wide_core: `wide` on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF (sq fit) | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) (sq fit) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (fitted pyramid, energy-normalised, t_pass 0.0001, tent share 0.7, binomial share 0.15, before training) | 0.0403 | 0.0465 | 0.0760 | 1.89x | 2.73e-02 | 7.92e-02 | -0.00% |
| **Trained (this run)** | 0.0341 | 0.0435 | 0.0719 | 2.11x | 1.85e-02 | 6.90e-02 | +0.00% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 0.00%. Learning rate 0.001; abs log loss + 0.02 x core loss, training (8 frames) 0.0354 at step 0, 0.0248 at step 2000.

Validation frame (15 sources; selects between runs, decision log 2026-11-03): mean FLIP 0.0491 at init, **0.0446** trained; abs log loss + 0.02 x core loss 0.0332 at init, 0.0227 trained; core loss 0.7595 at init, 0.3508 trained.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **True**
- (2) closes the edge gap (edge FLIP < 0.066 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.012): **True**
