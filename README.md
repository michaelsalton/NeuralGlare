# NeuralGlare

**Neural spatially varying glare at pyramid cost.**

NeuralGlare: the standard bloom mip pyramid, with tiny MLPs in the upsample passes. It is trained to reproduce **spatially varying, physically based glare** (lens diffraction, real-ray-traced aberrations, mechanical vignetting, chromatic spread) at roughly the cost of ordinary pyramid bloom.

The network is written once in **Slang**:
- trained from Python with **SlangPy**;
- compiled with `slangc -target hlsl` into engine post-processes.

Training and deployment run the same source, so they cannot drift apart.

**Engines**
- **Unity URP** is a benchmark target.
- **Unreal Engine** has two roles:
  1. A **baseline harness**: its built-in FFT convolution bloom, fed our PSF as a custom kernel.
  2. A **secondary deployment** (stretch goal), to show that one Slang source is portable across engines.

**Docs**
- [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md): motivation, contributions, background, method, data, evaluation, timeline and references.
- [`CLAUDE.md`](CLAUDE.md): working rules for Claude Code in this repo.

---

## 1. Repository layout

```
NeuralGlare/
├── slang/
│   ├── pyramid.slang        # downsample chain (fixed, differentiable)
│   ├── mlp.slang            # tiny MLP, [Differentiable], scalar path
│   └── glare.slang          # upsample passes calling the MLPs
├── src/neuralglare/         # Python package (import as `neuralglare.*`)
│   ├── reference/
│   │   ├── psf.py           # diaphragm + ray-traced wavefront → PSF(λ, field position)
│   │   ├── lens_trace.py    # real-ray trace of the lens → pupil tables (cached in data/lens/)
│   │   ├── lens_params.py   # camera, lens prescription, scatter, spectrum (sourced, cited)
│   │   └── convolve.py      # spatially varying convolution → ground truth
│   ├── baselines/
│   │   ├── pyramid_bloom.py # NumPy reimplementation of URP bloom (+ parameter fitting)
│   │   ├── fft_bloom.py     # uniform-kernel FFT convolution (center PSF and best-fit PSF)
│   │   └── neural_bloom_2025.py # Karp et al. reimplementation at full resolution
│   ├── train/
│   │   ├── dataset.py       # EXR inputs + reference targets
│   │   └── train.py         # SlangPy + PyTorch loop, exports weights
│   └── eval/
│       ├── metrics.py       # HDR-FLIP, log-PSNR, energy error, center/edge breakdown
│       ├── flicker.py       # temporal stability metric
│       └── figures.py       # paper figures
├── export/
│   └── build_hlsl.sh        # slangc -target hlsl → engines/*/Generated/ (Unity + Unreal)
├── engines/
│   ├── unity/               # URP renderer features + thin HLSL wrappers
│   │   ├── Capture/         # pre-post-process HDR EXR dump
│   │   ├── NeuralGlare/        # neural glare renderer feature
│   │   └── Generated/       # slangc output, never hand-edited
│   └── unreal/
│       ├── NeuralGlareUE/   # UE 5.8 C++ project hosting both the convolution-bloom baseline
│       │                    #   and the plugin: empty level, injection material, PP volume
│       ├── Baseline/        # PSF → convolution kernel texture, capture + timing scripts
│       └── NeuralGlarePlugin/  # (stretch) C++ plugin: global shader (.usf), RDG pass,
│           │                #   SceneViewExtension hook into the post-process chain
│           └── Generated/   # slangc output, never hand-edited
├── experiments/
│   ├── 00_check_gpu.py      # environment check: torch + slangpy see the GPU
│   └── 01_gap_study/        # Phase 0, see docs/RESEARCH_PLAN.md §7
├── tests/                   # pytest
├── notebooks/               # Jupyter scratch work
├── data/                    # (gitignored) EXR captures, HDRIs, references
├── outputs/                 # (gitignored) experiment outputs
├── docs/
│   └── RESEARCH_PLAN.md     # research plan, evaluation protocol, timeline
├── CLAUDE.md                # rules for Claude Code
└── README.md
```

---

## 2. Setup

**Reference material (clone outside this repo):**
- `shader-slang/neural-shading-s26`. Work through `autodiff/` and `network/` first.

**Not needed:** RTXNS, RTXNTC, falcor2 and neuralappearance. Those target neural materials or tensor-core paths.

**Unity side (URP), primary:**
- Turn on HDR in the URP asset and use R16G16B16A16 color. Verify that light sources in EXR dumps read well above 1.0.
- An EXR capture render feature dumps the camera HDR buffer **before post-processing**.
- A NeuralGlare renderer feature runs the generated HLSL compute passes, with weights in a `StructuredBuffer`.

**Unreal side, secondary:**

*Baseline harness (needs no C++; the project is C++ only so NeuralGlarePlugin can build into it):*
- Post Process Volume → Lens → Bloom → Method = **Convolution**.
- Set **Convolution Kernel** to a texture rendered from our PSF (center PSF and best-fit PSF), exported from `src/neuralglare/reference/psf.py`.
- Match Convolution Scale, Center and Buffer to the reference setup. Record the values per experiment.
- Capture HDR scene color before post-processing for reference generation, and capture the final bloom output for comparison.
- Time it with Unreal Insights / `stat gpu`, and with RenderDoc or Nsight Graphics for parity with Unity measurements.

*Deployment plugin (stretch):*
- A C++ plugin with a global shader (`.usf`) wrapping the generated HLSL.
- A Render Dependency Graph (RDG) compute pass, inserted into the post-process chain with a `SceneViewExtension`.
- Weights uploaded as a structured buffer.