import numpy as np
import pytest

from neuralglare.reference import camspec
from neuralglare.reference import lens_params as lp


@pytest.fixture(scope="module")
def weights():
    try:
        return lp.DEFAULT.spectral.rgb_weights()
    except OSError as e:                                     # offline, first run
        pytest.skip(f"camera spectral database unavailable: {e}")


def test_weights_are_white_balanced_and_positive(weights):
    assert weights.shape == (len(lp.DEFAULT.spectral.wavelengths_um), 3)
    np.testing.assert_allclose(weights.sum(0), 1.0, rtol=1e-12)
    assert (weights >= 0).all()


def test_d700_channel_peaks(weights):
    # Jiang et al. D700 data peak at 590 / 530 / 460 nm (R / G / B)
    s = camspec.sensitivities("Nikon D700")
    assert list(camspec.WAVELENGTHS_NM[s.argmax(0)]) == [590, 530, 460]
    lam = 1000 * np.asarray(lp.DEFAULT.spectral.wavelengths_um)
    centroid = (weights * lam[:, None]).sum(0)
    assert centroid[0] > centroid[1] > centroid[2]           # R redder than G redder than B
