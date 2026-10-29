# Unreal Convolution Bloom Check

Load the centre PSF as Unreal's Convolution Kernel, confirm the output matches what we predict in NumPy, and record its GPU cost. Originally Phase 0 step 4, deferred to Phase 3 on 2026-10-25, now pulled forward.

- [x] Add a decision-log row to `docs/RESEARCH_PLAN.md`: the check moves back out of Phase 3 (reverses the 2026-10-25 row), and the check is redefined as below



## Why Unreal won't match `I ⊛ K_centre`

UE 5.8 (5.8.4, CL 58902052) does not convolve with the kernel texture as given. From `Engine/Shaders/Private/Bloom/*.usf` and `PostProcessFFTBloom.cpp`:

1. **Find the centre:** the brightest texel by luminance (`BloomFindKernelCenter.usf`). The **Center** setting is ignored in 5.8 (`// TODO: remove`)
2. **Clamp level `M`:** the max (over RGB) in a ring 2–4 view pixels from the centre (`BloomSurveyMaxScatterDispersion.usf`)
3. **Clamp the kernel to `M`** (`BloomClampKernel.usf`). Only the centre texel's energy above `M` becomes a pass-through weight on scene colour (`BloomSurveyKernelCenterEnergy.usf`). Convolving with a one-pixel spike just reproduces the image, so multiplying scene colour is cheaper and exact
4. **Renormalise and tint** (`BloomFinalizeApplyConstants.usf`, plus the per-channel kernel normalisation in `GPUFastFourierTransform.usf`):

```
out = Tint · (C·I + I ⊛ K_clamped) / (C + ΣK_clamped)        Tint = total / max(total), per channel
```

- Designed for artist kernels: a sharp one-pixel dot ("the light itself") plus a dim starburst
- Our PSF's core covers 3×3 pixels; in blue the neighbours are about half the centre:

```
B channel, centre 3×3:   0.063 0.121 0.061
                         0.122 0.226 0.121
                         0.063 0.121 0.061
```

- The 8 neighbours are far above `M = 0.0052`, so their excess is clamped away and counted nowhere

Predicted on the centre kernel in `outputs/01_gap_study/default/fits/fft_kernels.npz`:

| | R | G | B |
|---|---|---|---|
| Core energy removed at the 8 neighbours | 0.46 | 0.57 | 0.69 |
| Pass-through fraction (ours: 0.47 / 0.34 / 0.23) | 0.85 | 0.78 | 0.72 |
| **Tint = total output energy** | **1.00** | **0.78** | **0.56** |

Expected: sharper core, brighter halo relative to the core, **~44% less blue energy**, from the tint alone.

Unconfirmed assumptions (the Unreal frame settles them):
- "Scatter dispersion energy" = sum of the clamped kernel (`BloomSumScatterDispersionEnergy.usf`)
- The FFT normalises the kernel per channel

The emulation (1 kernel texel = 1 view pixel):

```python
lum = K @ np.array([0.3, 0.59, 0.11])
cy, cx = np.unravel_index(lum.argmax(), lum.shape)
yy, xx = np.mgrid[:K.shape[0], :K.shape[1]]
d = np.maximum(abs(xx - cx), abs(yy - cy))          # Chebyshev distance from the centre
M = K[(d >= 2) & (d <= 4)].max()                    # ring max, grey
Kc = np.minimum(K, M)
C = np.maximum(K[cy, cx] - M, 0)                    # pass-through energy, per channel
T = C + Kc.sum((0, 1))
tint = np.clip(T / T.max(), 0, 1)
K_eff = (Kc + np.where((d == 0)[..., None], C, 0)) / T * tint
```

The check, redefined: compare Unreal against two NumPy predictions, both **pure convolution** (`convolve_uniform`, not `fft_bloom`: Unreal gets only K, no scatter term), both from the input quantised to R11G11B10 (`Q`, see step 2)
- (a) `Q(I) ⊛ K`, the centre kernel as is
- (b) `Q(I) ⊛ K_eff`, the Unreal-processed kernel (`baselines/unreal_fft_bloom.py`)
- Unreal ≈ (b) and ≠ (a) confirms the capture pipeline *and* shows production convolution bloom drifts from the physical PSF even at the frame centre: an extra row for the evaluation
- [x] Fix the pass threshold for "≈" (e.g. HDR-FLIP) **before** looking at the Unreal output. Fixed 2026-10-29, before any Unreal convolution capture, by `experiments/01_gap_study/unreal_threshold.py` (numbers in `outputs/01_gap_study/default/checks/unreal_threshold.json`). FLIP is the mean HDR-FLIP over 127×127 windows on the 35 point-grid sources, and over the full frame for Ferndale; energy is per channel, relative to (b):

| Scene | FLIP(a, b): signal | FLIP(b_raw, b): noise from `Q` | Energy a/b − 1 (R, G, B) | Energy b_raw/b − 1 |
|---|---|---|---|---|
| `pointgrid` | 0.462 | 0.006 | 0 / +27.7 % / +77.5 % | 0.16 % |
| `ferndale_studio_12_4k_center` | 0.389 | 0.007 | 0 / +27.7 % / +77.5 % | ≤ 0.01 % |

**Rule, per scene:** Unreal ≈ (b) if FLIP(U, b) ≤ τ = ½·FLIP(a, b) (0.23 point grid, 0.19 Ferndale) **and** every channel's energy is within ε = 2 % of (b). Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b). The input-format noise is ~1/60 of the signal in FLIP and < 0.2 % in energy, so the test can separate (a) from (b) on both scenes



## Steps

### 1. Kernel texture
- [x] Write the `centre` kernel from `fft_kernels.npz` to a 127×127 float32 RGB EXR with `io.write_exr` (`export_unreal_kernel.py` → `export/unreal_centre_kernel.exr`, bit-exact). Peak is at (63, 63), the exact centre
- [x] Import settings:
    - Compression: **HDR High Precision (RGBA32F)** (listed in 5.8). Never BC6H
    - sRGB off, `NoMipmaps`, Never Stream, Filter Nearest, Texture Group UI (no LOD bias)
- [x] Check how much kernel energy is below ~6e-5: 0.46 / 0.42 / 0.37 % (R / G / B). The texture keeps it exactly, but Unreal's `Bloom.FFT.ClampedKernel` is `PF_FloatRGBA` (`PostProcessFFTBloom.cpp:577`), so up to that much tail energy can lose precision inside the engine. Don't read a < 0.5 % tail difference as a mismatch

### 2. Scene
- [x] Empty level, 1920×1080, screen percentage 100, `r.AntiAliasingMethod 0` (no TAA/TSR jitter). C++ project `engines/unreal/NeuralGlareUE`, harness console variables in `Config/DefaultEngine.ini` (`[SystemSettings]` and `[/Script/Engine.RendererSettings]`); launched standalone with `-game -windowed -ResX=1920 -ResY=1080`. The editor viewport is not 1:1 and doubles the energy of some points
- [x] Inject the input with a post-process material, **Blendable Location = Scene Color Before Bloom**, loading the input EXR texel-exactly (same import settings as step 1): `M_InjectSceneColor` (ScreenPosition.ViewportUV → TextureSampleParameter2D `Input` → Emissive), instances `MI_Inject_PointGrid`, `MI_Inject_Ferndale`
- [x] Start with the **point grid** (isolated points show the kernel directly), then one HDRI crop: `ferndale_studio_12_4k_center`
- [x] Exposure: Manual, compensation 0, physical camera exposure off
- [x] Off: vignette, grain, chromatic aberration, lens flares (separate from bloom), local exposure, dirt mask, auto exposure
- [x] **Post-process scene colour is `PF_FloatR11G11B10`**, not fp16: `r.SceneColorFormat` sets only the main scene colour, and post-processing switches to R11G11B10 unless `r.PostProcessing.PropagateAlpha` (`PostProcessing.cpp:459`). Unsigned, 6/6/5 mantissa bits (1.6 / 1.6 / 3.1 % steps), max 65024 / 65024 / 64512. Kept, since it is what shipping projects run; modelled by `quantize_r11g11b10`. The HDRI crop must stay below that max, so goegap and cobblestone are out
- [x] Verified (`data/captures/unreal/pointgrid.rdc`, `unreal_beforebloom_pointgrid.exr`): pass `PostProcessMaterial 1920x1080`, 35 nonzero texels at the input positions, not flipped, neighbours 0, value **9984** (= R11G11B10 of 10000). PreExposure is not readable (stripped constant-buffer names), but 9984 bounds it to [0.992, 1.005)
- [ ] Capture Ferndale before bloom too: `quantize_r11g11b10(ferndale)` must match it bit for bit, which settles round-to-nearest vs toward zero

### 3. Bloom settings (log all with the result)
- [ ] Method Convolution, Intensity 1, Convolution Intensity 1, **Scatter Dispersion 1** (1 = use exactly the kernel's own energy). The two intensities multiply (`PostProcessFFTBloom.cpp:874`)
- [ ] **Threshold −1** for the record. In 5.8 it is Gaussian-only (Bloom → Gaussian; read only by the Gaussian setup pass, `PostProcessing.cpp:1527`); the FFT path never uses it
- [ ] **Scale = 127/1920 = 0.06615**. Tooltip says minor axis, but `BloomResizeKernel.usf` uses `max(width, height)`. This puts one kernel texel on one screen pixel
- [ ] **Boost: Min 1, Max 0, Mult 1**. Max ≤ Min turns it off (`GPUFastFourierTransform.usf:1322`). When on, `FilterPixel` also clamps luminance to Max (line 105), so Min 0 / Max 1 would cap every point at 1
- [ ] Buffer ≈ 0.1 so the FFT doesn't wrap light across screen edges
- [x] `r.Bloom.ScreenPercentage 100`, `r.Bloom.CacheKernel 1`, and **`r.Bloom.ApplyLocalExposure 0`**: the FFT applies local exposure to its input by default (`PostProcessing.cpp:175`, pass `Bloom.FFT.Input`, `PostProcessFFTBloom.cpp:826`)
- [ ] Center: ignored in 5.8, log it anyway

### 4. Capture (RenderDoc)
Scene colour and bloom are combined *inside the tonemapper* (`SceneColor · SceneColorApply + bloom`), so no linear buffer holds the final image. From one frame:
- [ ] Save the FFT bloom output texture as EXR
- [ ] Read the buffers `Bloom.FFT.SceneColorApplyParameters` and `Bloom.FFT.FFTMulitplyParameters` (Epic's spelling)
- [ ] Rebuild the result in Python. `FFTMulitply` goes into the convolution (`PostProcessFFTBloom.cpp:911`), so the bloom texture probably includes it already; check against the numbers
- [ ] The apply constants test the table above: they should be ≈ tint × pass-through fraction

### 5. Comparison script
- [ ] `experiments/01_gap_study/unreal_check.py`: predictions (a) and (b) vs the Unreal frame, with the same HDR-FLIP and relative-energy metrics, windows and FLIP reference as `unreal_threshold.py`, judged by the pre-registered rule above
- [x] Put the `K_eff` logic in `src/neuralglare/baselines/` with comments citing the `.usf` files, so it's a documented baseline: `unreal_fft_bloom.py` (`unreal_kernel`, `quantize_r11g11b10`, `unreal_fft_bloom`), tested in `tests/test_unreal_fft_bloom.py`. Reproduces the table above (pass-through 0.85 / 0.78 / 0.72, tint 1.00 / 0.78 / 0.56)
- Load RenderDoc EXR saves with `io.read_exr`: they carry an A channel, which it drops

### 6. Cost
- [ ] `ProfileGPU` or Nsight, FFTBloom passes at 1080p, steady state (`r.Bloom.CacheKernel 1` transforms the kernel only on the first frame)
- [ ] Record UE 5.8.4 (CL 58902052), RHI (DX12/Vulkan), RTX 2070, driver 621.01
