# Run all tests
uv run pytest

# Make Inputs
uv run python experiments/01_gap_study/make_inputs.py

uv run python -m neuralglare.reference.convolve --psf data/psf/psf_quick.npz --inputs data/inputs/pointgrid.exr --check

uv run python -m neuralglare.reference.convolve --psf data/psf/psf_quick.npz --inputs "data/inputs/*.exr"

# Run baeslines
uv run python -m neuralglare.baselines.fft_bloom --psf data/psf/psf_quick.npz --inputs "data/inputs/*4k_*.exr"

uv run python -m neuralglare.baselines.pyramid_bloom --inputs "data/inputs/*4k_*.exr"

