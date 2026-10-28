"""Where each run's files live. Scripts take a run name (the lens_params config: quick or
default) and derive every path from it, so one run's inputs and outputs can't be mixed
with another's.

    outputs/<experiment>/<run>/<stage>/
        report/   for reading: tables and figures. Look here first
        fits/     fitted baselines that later scripts load
        eval/     raw per-window records, for re-analysis
        export/   files for the engines
        checks/   convergence and sanity checks of the run's settings

Paths are relative to the repo root, so run scripts from there.
"""

import os

RUNS = ("quick", "default")
# convolve.py --out of each config's references; the quick ones predate run names
REFS = {"quick": "data/reference", "default": "data/reference_default"}


def psf_path(run):
    return f"data/psf/psf_{run}.npz"


def run_dir(experiment, run, stage):
    """outputs/<experiment>/<run>/<stage>, created if missing."""
    d = f"outputs/{experiment}/{run}/{stage}"
    os.makedirs(d, exist_ok=True)
    return d
