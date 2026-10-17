import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parent.parent / "experiments" / "01_gap_study" / "make_inputs.py"
_spec = importlib.util.spec_from_file_location("make_inputs", _PATH)
mi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mi)


def panorama_with_source(lon_deg, lat_deg, eh=1024, block=mi.BLOCK):
    """Dim equirect panorama with one bright block, aligned to the search grid so its
    centre is exactly the direction brightest_direction() returns."""
    ew = 2 * eh
    equi = np.full((eh, ew, 3), 0.1, np.float32)
    col = (lon_deg / 360 + 0.5) * ew
    row = (0.5 - lat_deg / 180) * eh
    c0, r0 = int(col // block) * block, int(row // block) * block
    equi[r0:r0 + block, c0:c0 + block] = 1e4
    return equi, np.radians((c0 + block / 2) / ew * 360 - 180), np.radians(90 - (r0 + block / 2) / eh * 180)


@pytest.mark.parametrize("lon, lat", [(-120.0, 35.0), (40.0, 5.0), (170.0, -20.0)])
@pytest.mark.parametrize("h_field", [0.0, 0.5, 0.85])
def test_source_lands_at_target_pixel(lon, lat, h_field):
    equi, src_lon, src_lat = panorama_with_source(lon, lat)
    found = mi.brightest_direction(equi)
    np.testing.assert_allclose(found, (src_lon, src_lat), atol=1e-9)

    u, v = mi.view_target(h_field)
    yaw, pitch = mi.aim(*found, u, v, mi.focal_px())
    crop = mi.equirect_to_perspective(equi, np.degrees(yaw), np.degrees(pitch))
    (x, y), h_meas = mi.locate_light(crop)

    cx, cy = (mi.W - 1) / 2, (mi.H - 1) / 2
    assert abs(x - (cx + u)) <= 2 and abs(y - (cy - v)) <= 2, ((x, y), (cx + u, cy - v))
    assert abs(h_meas - h_field) < 0.005
