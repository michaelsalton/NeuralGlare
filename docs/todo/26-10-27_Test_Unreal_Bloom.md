# Unreal Convolution Bloom Check

Load the centre PSF as Unreal's Convolution Kernel, confirm the output matches what we predict in NumPy, and record its GPU cost. Originally Phase 0 step 4, deferred to Phase 3 on 2026-10-25, now pulled forward.

- [ ] Add a decision-log row to `docs/RESEARCH_PLAN.md`: the check moves back out of Phase 3 (reverses the 2026-10-25 row), and the check is redefined as below



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

The check, redefined: compare Unreal against two NumPy predictions
- (a) `fft_bloom` with the centre kernel as is
- (b) the same with the Unreal-processed kernel `K_eff`
- Unreal ≈ (b) and ≠ (a) confirms the capture pipeline *and* shows production convolution bloom drifts from the physical PSF even at the frame centre: an extra row for the evaluation
- [ ] Fix the pass threshold for "≈" (e.g. HDR-FLIP) **before** looking at the Unreal output



## Steps

### 1. Kernel texture
- [ ] Write the `centre` kernel from `fft_kernels.npz` to a 127×127 float32 RGB EXR with `io.write_exr`. Peak is at (63, 63), the exact centre
- [ ] Import settings:
    - Compression: 32-bit float HDR if listed, otherwise `HDR (RGBA16F)`. Never BC6H
    - sRGB off, `NoMipmaps`, Never Stream
- [ ] Check how much kernel energy is below ~6e-5 (float16 flushes it to subnormals)

### 2. Scene
- [ ] Empty level, 1920×1080, screen percentage 100, `r.AntiAliasingMethod 0` (no TAA/TSR jitter)
- [ ] Inject the input with a post-process material, **Blendable Location = Scene Color Before Bloom**, loading the input EXR texel-exactly (same import settings as step 1)
- [ ] Start with the **point grid** (isolated points show the kernel directly), then one HDRI crop
- [ ] Exposure: Manual, compensation 0, physical camera exposure off
- [ ] Off: vignette, grain, chromatic aberration, lens flares (separate from bloom), local exposure, dirt mask, auto exposure

### 3. Bloom settings (log all with the result)
- [ ] Method Convolution, Intensity 1, Convolution Intensity 1, **Scatter Dispersion 1** (1 = use exactly the kernel's own energy)
- [ ] **Threshold −1** (anything above −1 turns the threshold on)
- [ ] **Scale = 127/1920 = 0.06615**. Tooltip says minor axis, but `BloomResizeKernel.usf` uses `max(width, height)`. This puts one kernel texel on one screen pixel
- [ ] **Boost: Max < Min** to disable it (the defaults boost bright pixels)
- [ ] Buffer ≈ 0.1 so the FFT doesn't wrap light across screen edges
- [ ] `r.Bloom.ScreenPercentage 100`
- [ ] Center: ignored in 5.8, log it anyway

### 4. Capture (RenderDoc)
Scene colour and bloom are combined *inside the tonemapper* (`SceneColor · SceneColorApply + bloom`), so no linear buffer holds the final image. From one frame:
- [ ] Save the FFT bloom output texture as EXR
- [ ] Read the buffers `Bloom.FFT.SceneColorApplyParameters` and `Bloom.FFT.FFTMulitplyParameters` (Epic's spelling)
- [ ] Rebuild the result in Python. `FFTMulitply` goes into the convolution (`PostProcessFFTBloom.cpp:911`), so the bloom texture probably includes it already; check against the numbers
- [ ] The apply constants test the table above: they should be ≈ tint × pass-through fraction

### 5. Comparison script
- [ ] `experiments/01_gap_study/unreal_check.py`: predictions (a) and (b) vs the Unreal frame, with the same HDR-FLIP and relative-energy metrics as `evaluate_gap.py`
- [ ] Put the `K_eff` logic in `src/neuralglare/baselines/` with comments citing the `.usf` files, so it's a documented baseline

### 6. Cost
- [ ] `ProfileGPU` or Nsight, FFTBloom passes at 1080p, steady state (`r.Bloom.CacheKernel 1` transforms the kernel only on the first frame)
- [ ] Record UE 5.8.4 (CL 58902052), RHI (DX12/Vulkan), RTX 2070, driver 621.01
