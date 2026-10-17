import numpy as np
import OpenEXR

from neuralglare.io import read_exr


def test_read_exr_drops_alpha(tmp_path):
    # Poly Haven HDRIs carry alpha; the bindings return it grouped as one "RGBA" channel.
    rgba = np.random.default_rng(0).uniform(0, 1e4, (4, 6, 4)).astype(np.float32)
    path = str(tmp_path / "rgba.exr")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    OpenEXR.File(header, {"RGBA": rgba}).write(path)

    rgb = read_exr(path)
    assert rgb.shape == (4, 6, 3) and rgb.dtype == np.float32
    np.testing.assert_array_equal(rgb, rgba[..., :3])        # HDR values, unclamped
