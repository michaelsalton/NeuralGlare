import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parent.parent / "experiments" / "01_gap_study" / "exact_pointgrid.py"
_spec = importlib.util.spec_from_file_location("exact_pointgrid", _PATH)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def test_splat_places_and_crops():
    k = np.zeros((5, 5, 3))
    k[2, 2] = 0.5
    k[2, 4] = 0.5                                   # half the energy two pixels to the right
    points = [(3, 4, np.array([2.0, 2.0, 2.0])), (9, 1, np.array([4.0, 4.0, 4.0]))]
    out, lost = ep.splat((8, 10, 3), points, [k, k])
    assert out[4, 3, 0] == 1.0 and out[4, 5, 0] == 1.0      # centred on (x=3, y=4)
    assert out[1, 9, 0] == 2.0                              # the right half falls off frame
    assert abs(lost - 3 * 2.0) < 1e-12                      # 2.0 per channel lost
