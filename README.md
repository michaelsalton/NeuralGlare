# NeuralGlare

Research into neural rendering for real-time simulation of glare/bloom.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU.

```powershell
uv sync
uv run python experiments/00_check_gpu.py
```

PyTorch is pulled from PyTorch's CUDA 12.8 index (see `[tool.uv.sources]` in `pyproject.toml`).

## Layout

| Path | Contents |
|---|---|
| `src/neuralglare/` | Reusable code: networks, losses, image I/O |
| `experiments/` | One script per experiment, numbered in order |
| `notebooks/` | Interactive exploration and visualization |
| `tests/` | `uv run pytest` |
| `data/` | Datasets (git-ignored) |
| `outputs/` | Experiment results, checkpoints, images (git-ignored) |

## Common commands

```powershell
uv run python experiments/<script>.py   # run an experiment
uv run jupyter lab                      # open notebooks
uv run pytest                           # run tests
uv run ruff check .                     # lint
uv add <package>                        # add a dependency
```
