import numpy as np

from neuralglare.reference.convolve import convolve_uniform


def test_fft_precision_next_to_bright_source():
    # A 4e5 "sun" must not bury a 1e-3 background in FFT round-off.
    img = np.full((1080, 1920, 3), 1e-3, np.float32)
    img[50, 60] = 4e5
    k = np.zeros((63, 63, 3), np.float32)
    k[31, 31] = 1.0                                   # identity kernel
    out = convolve_uniform(img, k)
    np.testing.assert_allclose(out, img, rtol=1e-5)
