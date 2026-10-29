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

Assumptions, settled by the 2026-10-29 captures:
- "Scatter dispersion energy" = sum of the clamped kernel (`BloomSumScatterDispersionEnergy.usf`): **confirmed**, the apply constants match to 6–7 digits
- The FFT normalises the kernel per channel: **yes, but by the wrong channels' sums**, and blue gets red's multiply constant (see Result)

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
| `ferndale_studio_12_4k_center` | 0.389 | 0.026 | 0 / +27.7 % / +77.5 % | 0.6 / 0.6 / 1.1 % |

**Rule, per scene:** Unreal ≈ (b) if FLIP(U, b) ≤ τ = ½·FLIP(a, b) (0.23 point grid, 0.19 Ferndale) **and** every channel's energy is within ε = 2 % of (b). Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b). The input-format noise is ≤ 1/15 of the signal in FLIP and ≤ 1.1 % in energy, so the test can separate (a) from (b) on both scenes. Noise numbers recomputed 2026-10-29 after the Ferndale before-bloom capture showed truncation, not round-to-nearest (no bloom output used); the rule is unchanged

## Result (2026-10-29)

`uv run python experiments/01_gap_study/unreal_check.py` → `outputs/01_gap_study/default/report/unreal_check.md`, `eval/unreal_check.json`. U is rebuilt as the tonemapper does, `SceneColor · SceneColorApply + bloom`, from the captured before-bloom colour, apply constants and bloom texture.

| Scene | FLIP U–a | FLIP U–b | τ | Energy U/b − 1 (R, G, B) | Pre-registered verdict | FLIP U–c (post hoc) | Energy U/c − 1 |
|---|---|---|---|---|---|---|---|
| `pointgrid` | 0.935 | 0.930 | 0.231 | −0.06 / +1.99 / **−3.87 %** | **fail** (FLIP and energy) | 0.358 | ≤ 0.02 % |
| `ferndale_studio_12_4k_center` | 0.402 | 0.080 | 0.194 | −0.08 / **+2.03** / **−3.79 %** | **fail** (energy) | 0.005 | ≤ 0.001 % |

**By the pre-registered rule, the check fails on both scenes.** Unreal ≠ (a) holds on both (on the point grid only barely), but Unreal ≈ (b) does not. The reasons, all found after seeing the output:

1. **Two channel mix-ups in the 5.8 FFT** (`GPUFastFourierTransform.usf`), the whole energy failure:
    - Normalisation: after the two-for-one transform, spectral texels hold (R, B) in the first half of the columns and (G, A) in the second (lines 569–578). `GetKernelSum` (line 146) therefore returns (R, B) and (G, A), but the convolution (line 640) uses them as (R, G) and (B, A). **Green is divided by blue's kernel sum and blue by green's.** Confirmed from the captured spectral kernel: its DC texels are R 0.0810, G 0.0928, B 0.0844 (`unreal_fft_constants_*.json`)
    - Multiply: the inverse two-for-one scales `b + i·a` by `Scale.xw` (line 415), so **blue gets red's `FFTMulitply` constant** (0.1495 instead of 0.1558)
    - Net bloom-texture gain vs design: R 1, G ×1.099, B ×0.873. Bloom is 15 / 22 / 28 % of the R / G / B output energy, hence the +2 % / −3.8 %
2. **The pre-registered point-grid FLIP is dominated by sub-representable tails, a flaw in the threshold, not in Unreal.** Beyond ~8 px from a source, 83 % of Unreal's pixels are exactly 0 where (b) is 1e-4 to 1e-13: the clamped kernel is stored as fp16 (`Bloom.FFT.ClampedKernel`, `PostProcessFFTBloom.cpp:577`) and the bloom texture as R11G11B10. HDR-FLIP takes its exposure range from (b), so 0 against 1e-10 scores as a large error at negligible energy. `unreal_threshold.py` modelled neither, so the pre-registered noise floor was too low

**Prediction (c), post hoc** (`unreal_fft_bloom_as_run`): (b) + both channel mix-ups + fp16 clamped kernel + R11G11B10 bloom texture. It reproduces the bloom texture **bit for bit on 94–96 % of point-grid texels and 88–98 % of Ferndale's**, energy within 0.08 %, L1 ≤ 0.18 %. Ferndale FLIP(U, c) = 0.005. The point-grid FLIP(U, c) of 0.36 is the remaining sub-1e-6 tail texels

For the evaluation: production convolution bloom at the frame centre differs from the physical PSF by design (clamp, pass-through, tint: −22 / −44 % energy in G / B) **and** by the 5.8 channel mix-ups (bloom G +10 %, B −13 %). Both belong in the Unreal row



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
- [x] Capture Ferndale before bloom too (`unreal_beforebloom_ferndale.exr`): `quantize_r11g11b10` matches it and the point grid **bit for bit with truncation** (round toward zero); round-to-nearest missed half the values by one step

### 3. Bloom settings (log all with the result)
- [x] Method Convolution (**tick its override**: the first convolution captures, 2026-10-29, ran Gaussian bloom because `bOverride_BloomMethod` was missing; check for `Bloom.FFT.*` passes in every capture), Intensity 1, Convolution Intensity 1, **Scatter Dispersion 1** (1 = use exactly the kernel's own energy). The two intensities multiply (`PostProcessFFTBloom.cpp:874`). The capture shows `FinalizeApplyConstants(ScatterDispersion=1.000000)`
- [x] **Threshold −1** for the record. In 5.8 it is Gaussian-only (Bloom → Gaussian; read only by the Gaussian setup pass, `PostProcessing.cpp:1527`); the FFT path never uses it
- [x] **Scale = 127/1920 = 0.06615**. Tooltip says minor axis, but `BloomResizeKernel.usf` uses `max(width, height)`. This puts one kernel texel on one screen pixel. Confirmed: the captured point response matches the kernel ring by ring to ~1e-3, centred at (0, 0)
- [x] **Boost: Min 1, Max 0, Mult 1**. Max ≤ Min turns it off (`GPUFastFourierTransform.usf:1322`). When on, `FilterPixel` also clamps luminance to Max (line 105), so Min 0 / Max 1 would cap every point at 1
- [x] Buffer ≈ 0.1 so the FFT doesn't wrap light across screen edges: FFT is 2048 × 2048 for the 1920 × 1080 frame
- [x] `r.Bloom.ScreenPercentage 100`, `r.Bloom.CacheKernel 1`, and **`r.Bloom.ApplyLocalExposure 0`**: the FFT applies local exposure to its input by default (`PostProcessing.cpp:175`, pass `Bloom.FFT.Input`, `PostProcessFFTBloom.cpp:826`)
- [x] Center: ignored in 5.8, log it anyway (0.5, 0.5)

### 4. Capture (RenderDoc)
Scene colour and bloom are combined *inside the tonemapper* (`SceneColor · SceneColorApply + bloom`), so no linear buffer holds the final image. From one frame (`data/captures/unreal/{pointgrid,ferndale}_convolution.rdc`, extracted by `engines/unreal/Baseline/extract_capture.py`, run with `qrenderdoc --python`):
- [x] Save the FFT bloom output texture as EXR: `unreal_bloom_{pointgrid,ferndale}.exr` (`Bloom.FFT.ScatterDispersion`, R11G11B10)
- [x] Read the buffers `Bloom.FFT.SceneColorApplyParameters` and `Bloom.FFT.FFTMulitplyParameters` (Epic's spelling): `unreal_fft_constants_{pointgrid,ferndale}.json`, with the spectral kernel's DC texels
- [x] Rebuild the result in Python. `FFTMulitply` goes into the convolution (`PostProcessFFTBloom.cpp:911`), so the bloom texture probably includes it already; check against the numbers. It does (applied in the inverse two-for-one, `GPUFastFourierTransform.usf:409-416`)
- [x] The apply constants test the table above: they should be ≈ tint × pass-through fraction. Apply 0.8505343 / 0.6121117 / 0.4075874 and multiply 0.1494656 / 0.1712394 / 0.1557920, equal to the emulation to 6–7 digits, same on both scenes

### 5. Comparison script
- [x] `experiments/01_gap_study/unreal_check.py`: predictions (a) and (b) vs the Unreal frame, with the same HDR-FLIP and relative-energy metrics, windows and FLIP reference as `unreal_threshold.py`, judged by the pre-registered rule above. Also reports the post-hoc (c), which never changes the verdict. See Result
- [x] Put the `K_eff` logic in `src/neuralglare/baselines/` with comments citing the `.usf` files, so it's a documented baseline: `unreal_fft_bloom.py` (`unreal_kernel`, `quantize_r11g11b10`, `unreal_fft_bloom`; post hoc `unreal_bloom_as_run`, `unreal_fft_bloom_as_run`), tested in `tests/test_unreal_fft_bloom.py`. Reproduces the table above (pass-through 0.85 / 0.78 / 0.72, tint 1.00 / 0.78 / 0.56)
- Load RenderDoc EXR saves with `io.read_exr`: they carry an A channel, which it drops

### 6. Cost
- [x] `ProfileGPU` or Nsight, FFTBloom passes at 1080p, steady state (`r.Bloom.CacheKernel 1` transforms the kernel only on the first frame). 2026-10-29, `ProfileGPU` × 5 in the game window, ~10 s warm-up, point grid; log kept as `data/captures/unreal/profilegpu_pointgrid_1080p_convolution.log`. The FFT bloom has no GPU stat (only the RDG scope `Bloom WxH`, `PostProcessFFTBloom.cpp:800`), so `stat gpu` can't separate it:

| Pass | Median of 5, ms | Range, ms |
|---|---|---|
| **`Bloom 1920x1080` (total)** | **1.001** | 1.000–1.005 |
| `FFTBloom FinalizeApplyConstants` | 0.019 | 0.019–0.020 |
| `FFT: Two-For-One Forward Horizontal` (size 2048) | 0.209 | 0.208–0.210 |
| `FFT: Apply Forward Vertical Transform, Multiply Texture, and InverseTransform` (2048) | 0.557 | 0.557–0.559 |
| `FFT: Two-For-One Inverse Horizontal` (2048) | 0.215 | 0.213–0.215 |
| *context:* `PostProcessMaterial` (injection) / `Tonemap` | 0.042 / 0.094 | |
| *context:* whole frame (empty level) | 2.46 | 2.44–2.46 |

- [x] Record UE 5.8.4 (CL 58902052), RHI (DX12/Vulkan), RTX 2070, driver 621.01. **DX12**, RTX 2070, driver 621.01, graphics clock locked to 1410 MHz (`nvidia-smi -lgc 1410,1410`, reference base clock; memory 7000 MHz unlocked), Editor binary `-game -windowed -ResX=1920 -ResY=1080` launched from PowerShell with the editor closed, `[SystemSettings]` incl. `r.VSync=0`, `t.MaxFPS=0`; volume as in step 3
- [x] Confirm `clocks.sm` stayed at 1410 MHz during the 1080p run. Checked 2026-10-29 in a separate 1080p convolution run, `nvidia-smi --query-gpu=... -l 1` for 96 busy seconds: graphics clock 1410 MHz throughout (min = max), memory 7000 MHz, 53–60 °C, 53–102 W of the 215 W limit, only throttle flag `0x1` (idle), never power or thermal. The profiled runs weren't logged directly, but their 0.5 % spread and this margin make a clock change there unlikely
- Pitfalls: `ProfileGPU` typed in the editor profiles the editor viewport (367×683 + Slate UI), and "Play in Standalone Game" keeps the editor rendering; `-trace=default,gpu` alone writes no file ("No trace connection"): start Unreal Insights first or add `-tracefile`. In PowerShell, quote `"-trace=default,gpu"` (the comma is an operator)
- [x] Context runs at 1080p, same procedure: Standard (Gaussian) bloom (`profilegpu_pointgrid_1080p_standard.log`, 6 profiles: `GaussianBlur` passes, no FFT) and bloom off via Bloom Intensity 0 (`profilegpu_pointgrid_1080p_bloomoff.log`, 5 profiles: no bloom or downsample scopes; `PostProcessing.cpp:862` skips the chain). Medians, ms:

| Run | Bloom scope | + scene downsample chain | PostProcessing | Whole frame | Frame − bloom off |
|---|---|---|---|---|---|
| Bloom off | — | — | 0.230 | 1.479 | 0 |
| Standard (Gaussian, threshold −1) | 0.138 | 0.030 (half-res 0.012 + chain 0.018; outside the `Bloom` scope) | 0.397 | 1.655 | **+0.176** |
| Convolution | 1.001 | — (reads full-res scene colour) | 1.204 | 2.456 | **+0.977** |

  Report Gaussian as 0.168 ms (scope + downsample chain), not 0.138: the convolution path doesn't need the chain. Tonemap is ~0.02 ms cheaper with convolution (0.094 vs 0.115–0.118), so the frame delta is slightly below the `Bloom` scope. Convolution costs ~5.8× Gaussian at 1080p
- [x] Convolution at 2560×1440 (`profilegpu_pointgrid_1440p_convolution.log`, 5 profiles, `-windowed -ResX=2560 -ResY=1440` gave the full size). Kernel Scale kept at 0.06615 (a screen fraction: same lens and FOV, more pixels); the 1080p input texture is stretched by ViewportUV, which doesn't change FFT cost. Medians, ms:

| Pass | 1440p | Range |
|---|---|---|
| **`Bloom 2560x1440` (total)** | **2.529** | 2.513–2.536 |
| `FFTBloom FinalizeApplyConstants` | 0.018 | 0.017–0.021 |
| `FFT: Two-For-One Forward Vertical` (size 2048) | 0.356 | 0.354–0.358 |
| `FFT: Apply Forward Horizontal Transform, Multiply Texture, and InverseTransform` (size **4096**) | 1.688 | 1.677–1.705 |
| `FFT: Two-For-One Inverse Vertical` (2048) | 0.460 | 0.458–0.463 |
| *context:* whole frame | 4.873 | 4.850–4.875 |

  **2.53× the 1080p cost for 1.78× the pixels.** With Buffer 0.1 the padded width is 2560 × 1.1 = 2816 > 2048, so the long axis jumps to a 4096 FFT, and Unreal swaps the axis order (two-for-one now vertical, the fused pass horizontal). The cost follows the padded power-of-two FFT size (2048² → 4096 × 2048: 2× the area, plus longer transforms), not the pixel count
- [x] Convolution at 3840×2160 (`profilegpu_pointgrid_4k_convolution.log`, 5 profiles). The monitor is 1440p, so: 1080p window, `r.ScreenPercentage 200` typed in the game console (log: `r.ScreenPercentage = "200"`; on the command line, PowerShell can split `-ExecCmds="r.ScreenPercentage 200"` at the space), ~10 s warm-up. With AA off the upscale runs after the tonemapper, so bloom runs at 3840×2160; the extra `Upscale(PrimaryToOutput) 3840x2160 -> 1920x1080` (0.076 ms) is outside the `Bloom` scope. Medians, ms:

| Pass | 4K | Range |
|---|---|---|
| **`Bloom 3840x2160` (total)** | **5.482** | 5.425–5.513 |
| `FFTBloom FinalizeApplyConstants` | 0.017 | 0.017–0.018 |
| `FFT: Two-For-One Forward Horizontal` (size 4096) | 1.148 | 1.144–1.159 |
| `FFT: Apply Forward Vertical Transform, Multiply Texture, and InverseTransform` (4096) | 3.099 | 3.022–3.117 |
| `FFT: Two-For-One Inverse Horizontal` (4096) | 1.221 | 1.219–1.227 |
| *context:* whole frame (incl. the upscale) | 10.839 | 10.746–10.856 |

  Summary, convolution bloom (RTX 2070 at 1410 MHz):

| Resolution | Pixels vs 1080p | FFT size | `Bloom` scope, ms | vs 1080p |
|---|---|---|---|---|
| 1920×1080 | 1.00× | 2048 × 2048 | 1.001 | 1.00× |
| 2560×1440 | 1.78× | 4096 × 2048 | 2.529 | 2.53× |
| 3840×2160 | 4.00× | 4096 × 4096 | 5.482 | 5.48× |

  Cost follows the padded power-of-two FFT area (1 : 2 : 4) plus longer transforms, not the pixel count. Gaussian at 1080p for comparison: 0.168 ms
- [x] Kernel preparation, 1080p only (`profilegpu_pointgrid_1080p_nocache.log`, 5 profiles, `r.Bloom.CacheKernel 0` typed in the game console): the cost if the kernel changed every frame. With caching on (default, all runs above) it happens once. Medians, ms:

| Pass | ms |
|---|---|
| **`InitBloomKernel` (total)** | **1.097** (1.093–1.140) |
| find centre, survey, sum energy, pack constants (127×127) | ~0.09 |
| `FFTBloom PreprocessKernel` (clamp 127², resize to 2048²) | 0.198 |
| `FFT: Two-For-One Forward Horizontal` of the 2048² kernel | 0.455 |
| `FFT: Complex Forward Vertical` of the kernel (2048) | 0.352 |
| `Bloom 1920x1080` total without cache | 2.086 (vs 1.001 cached: ×2.1) |

  Most of it is the kernel's own 2048² FFT, so it would scale with resolution like the main table. Measure 4K only if the paper argues about per-frame (animated or spatially varying) kernels
