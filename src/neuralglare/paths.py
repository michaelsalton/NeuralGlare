"""Where each run's files live. Scripts take a run name (the lens_params config: quick or
default) and derive every path from it, so one run's inputs and outputs can't be mixed
with another's.

    outputs/<experiment>/<run>/<stage>/
        report/   for reading: tables and figures. Look here first
        fits/     fitted baselines that later scripts load
        eval/     raw per-window records, for re-analysis
        export/   files for the engines
        checks/   convergence and sanity checks of the run's settings

Experiments made of several studies (03_train_network: Step 1 setup, the baseline, one folder
per Step 2 ablation) add a level, so each study's files stay together:

    outputs/<experiment>/<run>/<study>/<stage>/

Paths are relative to the repo root, so run scripts from there.
"""

import os

RUNS = ("quick", "default")
# convolve.py --out of each config's references; the quick ones predate run names
REFS = {"quick": "data/reference", "default": "data/reference_default"}


def psf_path(run):
    return f"data/psf/psf_{run}.npz"


def psf_bank_path(run, oversample=1):
    """Exact PSFs at fixed pixel positions (experiments/03_train_network/psf_bank.py);
    oversample > 1: the same PSFs on a finer grid (psf_bank.py --oversample)."""
    suffix = f"_x{oversample}" if oversample > 1 else ""
    return f"data/psf/psf_bank_{run}{suffix}.npz"


def run_dir(experiment, run, stage, study=None):
    """outputs/<experiment>/<run>/[<study>/]<stage>, created if missing."""
    d = f"outputs/{experiment}/{run}/{study}/{stage}" if study else f"outputs/{experiment}/{run}/{stage}"
    os.makedirs(d, exist_ok=True)
    return d
