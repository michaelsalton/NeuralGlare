"""Unreal check, step 1: write the centre PSF as an EXR for Unreal's Convolution Kernel.

Reads outputs/01_gap_study/default/fits/fft_kernels.npz, writes
outputs/01_gap_study/default/export/unreal_centre_kernel.exr.

    uv run python experiments/01_gap_study/export_unreal_kernel.py
"""

import os

import numpy as np

from neuralglare.io import read_exr, write_exr
from neuralglare.paths import run_dir

RUN = "default"                     # the run the Phase 0 decision rests on


def main():
    fits = run_dir("01_gap_study", RUN, "fits")
    kernel = np.load(f"{fits}/fft_kernels.npz")["centre"]         # (127, 127, 3), sums to 1
    path = f"{run_dir('01_gap_study', RUN, 'export')}/unreal_centre_kernel.exr"
    write_exr(path, kernel)

    back = read_exr(path)                                         # the file must be bit-exact
    assert np.array_equal(back, kernel.astype(np.float32))
    peak = np.unravel_index(back.sum(-1).argmax(), back.shape[:2])
    print(f"{os.path.basename(path)}: {back.shape}, peak at {peak}, "
          f"sum per channel {back.sum((0, 1))}")


if __name__ == "__main__":
    main()
