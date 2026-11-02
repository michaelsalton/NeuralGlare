# Gap study (psf_default.npz, window 254 px)

**Decision: PROCEED**. Best uniform kernel: FFT, centre PSF; point-grid edge/centre FLIP 4.25x (required >= 2.0x); HDRI crops 4.62x (required > 1).

### Point grid (same source, only H changes)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0481 | 0.0994 | 0.2044 | 4.25x | 2.78e-03 | 2.06e-02 | 5.74e-02 | 20.65x | -0.00% |
| FFT, best-fit PSF | 0.0866 | 0.1220 | 0.2244 | 2.59x | 9.53e-03 | 2.60e-02 | 6.37e-02 | 6.68x | -0.00% |
| Pyramid bloom (fitted) | 0.2661 | 0.2661 | 0.3470 | 1.30x | 2.04e-01 | 2.00e-01 | 2.09e-01 | 1.03x | +1.70% |
| No bloom | 0.2690 | 0.2691 | 0.3511 | 1.31x | 2.34e-01 | 2.50e-01 | 3.26e-01 | 1.40x | -0.00% |

Medians over windows. Windows per bin: {'centre': 5, 'mid': 16, 'edge': 14}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.

### HDRI crops (at measured H)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0129 | 0.0455 | 0.0597 | 4.62x | 7.21e-05 | 2.87e-03 | 4.06e-03 | 56.35x | -0.00% |
| FFT, best-fit PSF | 0.0270 | 0.0314 | 0.0465 | 1.72x | 9.25e-04 | 8.91e-04 | 2.08e-03 | 2.24x | -0.00% |
| Pyramid bloom (fitted) | 0.0552 | 0.0730 | 0.0844 | 1.53x | 3.97e-03 | 8.83e-03 | 9.33e-03 | 2.35x | +1.66% |
| No bloom | 0.0591 | 0.0816 | 0.0965 | 1.63x | 5.52e-03 | 1.24e-02 | 1.45e-02 | 2.63x | -0.00% |

Medians over windows. Windows per bin: {'centre': 4, 'mid': 4, 'edge': 4}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.
