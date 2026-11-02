"""Phase 2, step 1: a bank of exact PSFs at fixed pixel positions.

The point-light experiments build every frame and every reference from these PSFs. Both the
network and the optics are linear in light, so a frame of N sources is exactly the sum of N
single-source frames, and a source's intensity or colour scales its glare without changing its
shape. What the network has to learn is how the PSF changes with position, so the data budget
goes into positions, not frames.

Positions (configs/split.toml, [phase2_points]):
    train  one per cell of a jittered rows x cols grid over the frame (seeded), so H covers
           0..1 in every direction and the sub-pixel phase against the mips varies
    test   the Phase 0 7x5 point grid, so results compare directly with the gap study

Each PSF is exact at its own pixel, as in exact_pointgrid.py: the lens is ray-traced at that
pixel's field height and psf_rgb integrates 48 wavelengths. One position costs ~1 min of
tracing plus ~1 min of PSF on one core, so positions run in parallel, and every finished PSF
is saved to a parts folder so an interrupted run resumes where it stopped.

Each position gets its own pupil table (build_table([H]), which always adds H = 0). A table's
extent U depends on the heights traced together; across the cached tables it varies by 0.3 %,
a change of the pupil resampling grid whose effect the table convergence check bounds
(doubling table_px changes the corner PSF by 0.006 %).

    uv run python experiments/03_train_network/psf_bank.py --run default --workers 8
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import tomllib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from neuralglare.paths import psf_bank_path
from neuralglare.reference import lens_params as lp
from neuralglare.reference import lens_trace
from neuralglare.reference.psf import psf_rgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "01_gap_study"))
from exact_pointgrid import field_at  # one definition of a pixel's H and direction

CONFIGS = {"quick": lp.QUICK, "default": lp.DEFAULT}


def train_positions(cfg, rows, cols, seed):
    """One integer pixel per cell of a rows x cols grid, uniformly jittered inside the cell."""
    rng = np.random.default_rng(seed)
    w, h = cfg.camera.width_px, cfg.camera.height_px
    positions = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c * w // cols, (c + 1) * w // cols
            y0, y1 = r * h // rows, (r + 1) * h // rows
            positions.append((int(rng.integers(x0, x1)), int(rng.integers(y0, y1))))
    return positions


def test_positions():
    with open(ROOT / "data" / "inputs" / "manifest.json") as fh:
        meta = json.load(fh)["images"]["pointgrid"]
    return [(x, y) for y in meta["ys"] for x in meta["xs"]]


def exact_psf(job):
    """K x K x 3 PSF at pixel (x, y), each channel summing to 1. Runs in a worker process."""
    run, x, y, part = job
    if os.path.exists(part):
        with np.load(part) as z:
            return z["psf"]
    cfg = CONFIGS[run]
    H, d = field_at(cfg, x, y)
    tab = lens_trace.build_table(cfg, [H], verbose=False)
    psf = psf_rgb(cfg, tab, H, d, cfg.spectral.rgb_weights()).astype(np.float32)
    np.savez(part, psf=psf, x=x, y=y, rayoptics_version=tab["rayoptics_version"])
    return psf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(CONFIGS), default="default")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = CONFIGS[args.run]
    if lp.PLACEHOLDER:
        print("WARNING: lens_params.py contains PLACEHOLDER values. "
              "Do not report results computed with them.\n")

    split = tomllib.loads((ROOT / "configs" / "split.toml").read_text())["phase2_points"]
    assert split["test"] == "pointgrid", split["test"]
    train = train_positions(cfg, *split["train_grid"], split["train_seed"])
    test = test_positions()
    assert not set(train) & set(test), "a training position coincides with a test position"
    positions = train + test
    names = ["train"] * len(train) + ["test"] * len(test)

    out = ROOT / psf_bank_path(args.run)
    parts = out.with_suffix("")
    parts = parts.parent / f"{parts.name}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(args.run, x, y, str(parts / f"{x}_{y}.npz")) for x, y in positions]
    print(f"{len(train)} train + {len(test)} test positions, {args.workers} workers, "
          f"{sum(os.path.exists(j[3]) for j in jobs)} already done")

    # One FFT / BLAS thread per process: the parallelism is over positions
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    psfs = []
    with Pool(args.workers) as pool:
        for i, psf in enumerate(pool.imap(exact_psf, jobs), 1):
            psfs.append(psf)
            print(f"  {i:>3}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    fields = [field_at(cfg, x, y) for x, y in positions]
    np.savez_compressed(
        out,
        psf=np.stack(psfs),                                         # (N, K, K, 3) float32
        pos_px=np.array(positions, np.int32),                       # (N, 2) x, y
        field_height=np.array([f[0] for f in fields]),              # (N,)
        field_dir=np.array([f[1] for f in fields]),                 # (N, 2) unit, from the centre
        split=np.array(names),                                      # (N,) "train" / "test"
        split_config=json.dumps(split),
        config=json.dumps(dataclasses.asdict(cfg)),
        placeholder=lp.PLACEHOLDER,
    )
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
