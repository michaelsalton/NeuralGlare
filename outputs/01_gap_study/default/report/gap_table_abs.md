# Gap study (psf_default.npz, window 254 px, baselines fitted with the abs log loss)

**Decision: PROCEED**. Best uniform kernel: FFT, centre PSF; point-grid edge/centre FLIP 5.35x (required >= 2.0x); HDRI crops 4.62x (required > 1).

### Point grid (same source, only H changes)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0123 | 0.0281 | 0.0657 | 5.35x | 2.78e-03 | 2.06e-02 | 5.74e-02 | 20.65x | -0.00% |
| FFT, best-fit PSF | 0.0255 | 0.0346 | 0.0701 | 2.75x | 9.65e-03 | 2.51e-02 | 6.37e-02 | 6.60x | +0.00% |
| Pyramid bloom (fitted) | 0.0817 | 0.0833 | 0.1162 | 1.42x | 2.11e-01 | 2.26e-01 | 2.94e-01 | 1.39x | +0.01% |
| No bloom | 0.0817 | 0.0833 | 0.1162 | 1.42x | 2.34e-01 | 2.50e-01 | 3.26e-01 | 1.40x | -0.00% |

Medians over windows. Windows per bin: {'centre': 5, 'mid': 16, 'edge': 14}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.

### HDRI crops (at measured H)

| Baseline | FLIP centre | FLIP mid | FLIP edge | edge/centre | log-MSE centre | log-MSE mid | log-MSE edge | edge/centre | energy edge |
|---|---|---|---|---|---|---|---|---|---|
| FFT, centre PSF | 0.0129 | 0.0455 | 0.0597 | 4.62x | 7.21e-05 | 2.87e-03 | 4.06e-03 | 56.35x | -0.00% |
| FFT, best-fit PSF | 0.0266 | 0.0316 | 0.0474 | 1.78x | 9.05e-04 | 9.32e-04 | 2.16e-03 | 2.38x | -0.01% |
| Pyramid bloom (fitted) | 0.0591 | 0.0816 | 0.0965 | 1.63x | 5.52e-03 | 1.24e-02 | 1.45e-02 | 2.63x | +0.01% |
| No bloom | 0.0591 | 0.0816 | 0.0965 | 1.63x | 5.52e-03 | 1.24e-02 | 1.45e-02 | 2.63x | -0.00% |

Medians over windows. Windows per bin: {'centre': 4, 'mid': 4, 'edge': 4}. Bins: centre H < 1/3, mid 1/3-2/3, edge >= 2/3.
