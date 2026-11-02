# Gap study (psf_quick.npz, window 126 px)

### Point grid (same source, only H changes)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0471 | 0.0912 | 0.2000 | 4.25x | 6.76e-03 | 4.39e-02 | 1.89e-01 | 27.92x | -0.00% |
| FFT, best-fit PSF | 0.1015 | 0.1298 | 0.2143 | 2.11x | 4.27e-02 | 7.84e-02 | 2.20e-01 | 5.16x | -0.00% |
| Pyramid bloom (fitted) | 0.2817 | 0.2812 | 0.2804 | 1.00x | 7.59e-01 | 7.10e-01 | 4.76e-01 | 0.63x | +1.78% |
| No bloom | 0.2928 | 0.2925 | 0.2927 | 1.00x | 8.79e-01 | 9.27e-01 | 1.13e+00 | 1.29x | -0.00% |

Medians over windows. Windows per bin: {'centre': 5, 'mid': 16, 'edge': 14}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.

### HDRI crops (at measured H)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0121 | 0.0621 | 0.0949 | 7.84x | 4.96e-05 | 4.54e-03 | 1.16e-02 | 233.88x | +0.00% |
| FFT, best-fit PSF | 0.0338 | 0.0451 | 0.0790 | 2.34x | 1.30e-03 | 1.98e-03 | 7.20e-03 | 5.55x | +0.00% |
| Pyramid bloom (fitted) | 0.0909 | 0.1024 | 0.1246 | 1.37x | 1.35e-02 | 1.73e-02 | 2.35e-02 | 1.74x | +1.77% |
| No bloom | 0.0921 | 0.1180 | 0.1380 | 1.50x | 1.74e-02 | 2.82e-02 | 4.07e-02 | 2.34x | +0.00% |

Medians over windows. Windows per bin: {'centre': 4, 'mid': 4, 'edge': 4}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.
