# Phase 2, tent_upsample: `per_channel` on exact point-light PSFs (default)

| Model | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE edge | energy edge |
|---|---|---|---|---|---|---|---|
| Phase 0: FFT, exact centre PSF (sq fit) | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 5.74e-02 | -0.00% |
| Phase 0: pyramid bloom (fitted) (sq fit) | 0.0813 | 0.0826 | 0.1138 | 1.40x | 2.04e-01 | 2.09e-01 | +1.70% |
| Init (fitted pyramid, energy-normalised, t_pass 0.0001, before training) | 0.0578 | 0.0614 | 0.0919 | 1.59x | 2.10e-01 | 2.36e-01 | -0.00% |
| **Trained (this run)** | 0.0457 | 0.0522 | 0.0767 | 1.68x | 7.20e-02 | 8.54e-02 | +0.00% |

Medians over the 35 point-grid windows (254 px). Bins: centre H < 1/3 (n = 5), mid (n = 16), edge >= 2/3 (n = 14). Negative output texels: 0.00%. Learning rate 0.0001; abs log loss, training (8 frames) 0.0418 at step 0, 0.0251 at step 2000.

Validation frame (15 sources; selects between runs, decision log 2026-11-03): mean FLIP 0.0639 at init, **0.0534** trained; abs log loss 0.0399 at init, 0.0238 trained.

Pre-registered questions (decision log 2026-10-31):
- (1) beats the fitted pyramid in every bin: **True**
- (2) closes the edge gap (edge FLIP < 0.066 and edge/centre <= 2x): **False**
- expected: centre FLIP above the FFT centre kernel's (0.012): **True**
