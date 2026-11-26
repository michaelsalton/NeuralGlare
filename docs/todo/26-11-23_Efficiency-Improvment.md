# Efficiency improvement: the neural glare model in Unity

Started 2026-11-23, after the first GPU timings (decision log 2026-11-23).

## Where we are

`ray_bright` (the current best, validation FLIP 0.040928, test 0.0317 / 0.0384 / 0.0659) in the URP renderer feature, RTX 2070 at 1410 MHz, D3D12, Development Build:

| Pass | 1080p | 1440p | share |
|---|---|---|---|
| Pyramid (prefilter, 1 mip) | 0.04 ms | 0.06 ms | 0.2 % |
| Levels (full-resolution level: LevelMLP 2-16-18, 6 taps incl. tent and binomial) | 1.58 ms | 2.78 ms | 9.7 % |
| **Rays** (14 rays × 3 streak passes + 1 cross / brightness pass) | **14.13 ms** | **24.66 ms** | **86.4 %** |
| Composite (SpikeMLP, 14 BrightMLP, family normalisers) | 0.61 ms | 1.07 ms | 3.7 % |
| **Total** | **16.36 ms** | **28.58 ms** | |
| Unreal convolution (FFT) bloom | 1.00 ms | 2.53 ms | |
| Unreal Gaussian bloom | 0.17 ms | | |

**After step 1a (fp16 streaks, 2026-11-24): 11.96 ms at 1080p (rays 9.74), 20.71 ms at 1440p (rays 16.81).**

**After step 1e (hardware bilinear, 2026-11-24): 8.64 ms at 1080p (rays 7.37, levels 0.63), 14.68 ms at 1440p**, validation FLIP of Unity's own output +1.17e-5 vs `ray_bright`. Tier A (≤ 4 ms) still needs ≈ 2.2×: next is 1c (the cross stage's fp32 family traffic, ≈ 2.9 ms).

**After step 1c (ray stack, 2026-11-24): 7.05 ms at 1080p** (pyramid 0.04, levels 0.63, rays = 42 streak passes 4.60, composite 1.77), **11.99 ms at 1440p**; Unity validation FLIP unchanged from 1e. Tier A still needs ≈ 1.8×. Every pass is fetch-bound: 1d (baking the networks) measured −3 % and was rejected. The 42 streak passes (4.6 ms) and the composite's 42 ray taps (≈ 1.5 ms) are the model itself (→ 2a, 2c).

**Measured after 1a (GPU Trace, 1080p, 2026-11-24; decision log): the whole model is bound by point loads.** Per dispatch: levels 1572 µs; per ray streak ≈ 167 / 172 / 166 µs, cross + brightness ≈ 202 µs; composite 608 µs. `pyramid.slang` does every bilinear tap as four `Load`s, so a streak pass issues 16 loads per pixel at ≈ 198 G loads/s, ≈ 97 % of the 2070's texel rate. The levels pass fits the same rate to 2 %. → Step 1e.

Not timed: the final blit back into the camera colour. Memory: about 274 MB of RGBA32F targets at 1080p.

**The rays are the problem.** Everything else together is 2.2 ms. The rays run 56 full-resolution dispatches per frame, and each reads and writes a 16 B/px image.

Rough traffic model per ray (1080p image = 33 MB in RGBA32F):
- 3 streak passes: 3 writes + 3 reads (the source, S1, S2)
- cross pass: 1 write + 2 reads (S3 and the family's running sum)
- ≈ 9 image-sized transfers per ray × 14 rays ≈ 4.2 GB per frame
- ≥ 9 ms at the 2070's 448 GB/s, before cache misses on the diagonal 16 px steps

The first hypothesis was that **the rays are bandwidth-bound**: limited by bytes moved, not by arithmetic.

**Measured (Nsight Graphics GPU Trace, one frame, `ray_bright` at 1080p, 2026-11-23; decision log 2026-11-24): the rays are texture-pipe-bound, not DRAM-bound.**
- L1TEX texture data pipe **82–85 %** of peak (the busiest unit); DRAM **52 %** (read 26 %, write 26 %); L2 31 % (hit rate 61 %); SM / FMA pipe 31 %; GPU active 99.99 % of the frame.
- The traffic model's byte count holds: 0.52 × 448 GB/s × 16.8 ms ≈ 3.9 GB per frame, against ≈ 4.2 GB predicted. The bytes are moved, but moving them is not the limit; issuing and filtering the wide fetches is.
- Caveat: these counters are averaged over the whole trace, not per pass. Rays are 85 % of the frame, so they mostly describe the rays, but they don't separate a streak pass from the level pass. Per-pass throughput still needs the Nsight UI's range view or a Shader Profiler trace.
- Per ray the trace shows 4 dispatches of ≈ 190 / 286 / 283 / 261 µs (streak 1, streak 2, streak 3, cross / brightness), 14 rays ≈ 14.26 ms.

## Targets (to fix in the decision log before step 1)

Suggested; to be decided:
- **Tier A: ≤ 4 ms at 1080p.** An expensive but shippable post effect on a mid-range GPU.
- **Tier B: ≤ 1 ms at 1080p**, the FFT's cost. At FFT quality on the edge, this is the paper's strongest claim: spatially varying glare for the price of a uniform kernel.

Accuracy guard for every change: validation FLIP no worse than `ray_bright`'s by more than δ_coma (0.000218), test FLIP reported per bin. A change that trades accuracy for cost is reported as a point on the cost-quality curve, not silently kept.

## Ground rules

- **One change at a time.** Each one gets timing (protocol of 2026-11-23), FLIP, Unity parity and a decision-log row.
- **One source of truth still holds.** A change to how the model computes (fused passes, a new factorisation, lower resolution) goes into `slang/glare.slang` and is used by Python and Unity alike. Only storage formats and dispatch plumbing live in `GlareChain.cs`.
- **Storage precision is modelled in Python too.** If Unity stores an intermediate in fp16, the Python evaluation rounds the same intermediate to fp16 the way the GPU does (`train_points.stored()`: toward zero on the RTX 2070, decision log 2026-11-24), so parity stays meaningful and FLIP describes what ships.
- **Views:** `views.py --study efficiency` draws the unoptimised model, Python's model of the shipping path and Unity's own output (the parity run's `points_test` frame) next to the lens, plus `3b_change_vs_unoptimised.png`. Rerun after Run Parity whenever a step is kept; add a run to `STUDY_RUNS["efficiency"]` when a step changes the Python path.
- **Same machine, same protocol:** clock locked at 1410 MHz, D3D12, Development Build, 10 s warm-up + 5 × 2 s windows, the same scene.

## Step 0: measure before changing anything

- [ ] **GPU trace of one frame** in Nsight Graphics (GPU Trace, D3D12) or PIX. For a streak pass, a cross pass, the level pass and the composite, read throughput as a share of the GPU's peak for memory (DRAM / L2), the texture units and the shader cores. This tells which resource limits each pass.
  - Bandwidth-bound rays → steps 1a–1c pay off.
  - Texture-unit- or latency-bound rays (diagonal 16 px steps missing the cache) → reordering or lower resolution (step 2) matters more.
  - Find out why the level pass costs 1.58 ms. 22 fetches and ≈ 320 MACs per pixel suggest well under 1 ms, so low occupancy (too many registers from 18 logits and 3 softmaxes) is a suspect.
- [ ] **Time the earlier models that run on the current kernels:** `wide_core` (no cross filter, no brightness), `ray_cross`, `ray_bright`. The differences are the measured cost of the cross filter and of the brightness. They become a measured "ms at 1080p" column in the ablation table, next to the MACs column, which missed the rays' real cost.
- [ ] **Time the blit** back into the camera colour once, with a temporary scope, and add it to the total.
- [ ] **4K:** enable DSR 2.25× in the NVIDIA Control Panel and rerun. Check how Unreal's 4K number (5.48 ms) was measured on this monitor.
- [ ] **Camera colour format (HDR rule):** `PC_RPAsset` has HDR precision 32 bits, so the camera target is B10G11R11 (6 / 6 / 5 mantissa bits, ≈ 1.5–3 % relative precision, no alpha). The model's input in Unity is quantised far more coarsely than in Python. Set HDR precision to 64 bits (RGBA16F) (**done 2026-11-24**: the asset reads 64 bits, and the 1a timings' player was built after the change; still to do: confirm the format in the player and make the parity test feed Python the same format), re-time streak 1 (likely slower; the cost belongs in the total), and make the parity test feed Python the same format.
- [ ] **Confirm the Unreal side:** the API and build configuration of the 2026-10-29 timings, so the comparison is like for like.

## Step 1: engine-side changes, no retraining

Ordered by expected gain per effort. The gains are estimates from the traffic model, to be replaced by measurements.

### 1a. fp16 intermediates (RGBA16F) — done, kept (decision log 2026-11-24)
- **Result:** rays 14.09 → **9.74 ms** at 1080p (1.45×), 24.59 → **16.81 ms** at 1440p (1.46×); totals **11.96 / 20.71 ms**. Validation FLIP +1.1e-5 (≈ 5 % of δ_coma), test per source ≤ 2.4e-5. Unity parity PASS (`spikes` 0.41 / 0.38 / 0.81, families ≤ 0.0096).
- **Found on the way:** the RTX 2070 rounds float32 → half UAV stores **toward zero**, not to nearest (D3D allows both). Modelling nearest left Unity's ray families 0.1 % darker than Python's on every channel (parity ≈ 20× outside tolerance). `train_points.stored()` now rounds toward zero. Recheck on any other GPU or engine.
- **What:** store the streak intermediates (S1–S3, `rayA` / `rayB`) as half floats; later possibly the level output too. **Not the family sums** (measured 2026-11-24, `ray_bright` on the 1e4 point grid in fp32): streak outputs peak at 3 540, but with brightness a family sum holds e^{z_k} × each ray before the composite divides by the normaliser, and peaks at **9.1e4**, past fp16's 65 504. Stored in fp16 they become inf and the composite inf / inf = NaN (caught by `save_views`' linearity check). The streak inputs carry 9 of the 13 taps per ray, so most of the gain remains. (`train_points.py --ray-storage fp16`, layout key `rayStorage`.)
- **Range fix: pre-exposure** (decision log 2026-11-24). The parity HDRI (`goegap_road_4k`, sun up to 3.8e5, 188–407 texels > 65 504 per frame) overflowed fp16 streaks to inf. The rays are linear in light, so `glare.slang`'s ray functions take a scale: k into the first streak pass, 1/k into the pass that writes the float32 family sum (Unreal's PreExposure). k = 2⁻⁴ by default (`GlareChain.DefaultPreExposure`, `train_points.RAY_PRE_EXPOSURE`, Inspector field on the feature; a game would set it per frame from exposure). Measured, Python, fp16 + k vs fp32: point grid validation FLIP −3.5e-7 (δ_coma 2.18e-4), test per source ≤ 1.4e-6; HDRI spikes bound ratio 0.32 / 0.23 / 0.15 (parity tolerance 1), all finite (k = 1: inf on every frame).
- **Expected:** the rays turned out texture-pipe-bound (Where we are), so the gain comes mainly from cheaper fetches in the texture unit, not from DRAM. Estimate **≈ 1.3–1.5× on the rays (≈ 14 → 9.5–11 ms)**, not the 2× the bandwidth model predicted.
- **Evidence already in the trace:** streak 1 reads the camera colour (B10G11R11, 4 B/px; `PC_RPAsset` HDR precision 32 bits) and takes ≈ 190 µs; streaks 2 and 3 have the same taps but read RGBA32F intermediates (16 B/px) and take ≈ 285 µs. The narrow input is ≈ 33 % faster. A 4× narrower format buying only 1.5× is why 2× from halving the format is unlikely.
- **Risk:**
  - **Precision:** fp16 has a 10-bit mantissa, ≈ 1e-3 relative error per stored value. The family sums add 7 rays, so their rounding accumulates. That is far above the parity tolerance (1e-4) unless Python rounds identically, but probably far below what FLIP sees. Measure; don't assume.
  - **Range:** fp16 tops out at 65504. The point grid's sources are 1e4, but real HDR scenes (the sun) can exceed it, and fp16 turns those into infinity. Clamping would break the HDR rule (never clamp inputs), so keep the targets that can hold a whole source's light (the first streak pass, the full-resolution level) in fp32, or scale by a known exposure before storing and undo it after.
  - **Underflow** isn't the issue: the faintest tail the loss sees (ε = 1e-7 × a 1e4 source = 1e-3) is well inside fp16's normal range (≥ 6e-5).
- **Checks:** Python rounds the same intermediates; parity tolerance unchanged if both sides round identically; FLIP within δ.

### 1e. Hardware bilinear filtering — done, kept (decision log 2026-11-24)
- **Part 3 result:** Unity's validation FLIP 0.0409398 vs Python's 0.0409395 (+3.0e-7), vs the trained run's 0.040928 **+1.17e-5 (0.05 δ_coma) ⇒ PASS**. Test per bin within 1e-5 of Python's; per source (50) max |Unity − Python| 1.75e-5 (prediction ≤ 3e-5 held). Parity under the hardware criteria PASS on all frames. Still to do: the Unreal wrapper binds its own linear-clamp sampler (and `_RayScale`).
- **Part 1 done (rays):** rays 9.73 → **7.37 ms** at 1080p, 16.81 → **12.50 ms** at 1440p (1.32×); totals 9.59 / 16.40 ms. Streaks 1.5× faster (≈ 110 µs), cross pass unchanged (≈ 209 µs): both now bound by DRAM, the cross pass by its fp32 family sum (32 of 40 B/px). Parity: output PASS, families ≤ 2.6e-3 with no bias.
- **Part 2 done (levels, pyramid, composite):** levels 1.57 → **0.63 ms**; totals **8.64 ms** at 1080p, **14.68 ms** at 1440p. Parity fails at the fp32 tolerance (≈ 11×, relative ≤ 1.5e-3, no bias), and no fraction-rounding model in Python fits both the levels and the rays (best: 8 bits nearest for rays, 9 for levels).
- **Part 3 (design):** measure what ships instead of modelling the hardware. Unity runs the validation and test point-grid frames through the parity harness, and `unity_parity.py score` scores its output with `train_points.window_records`, the same FLIP as every run. Gate: Unity's validation FLIP within δ_coma of Python's. Parity stays as the bug check, at a tolerance for hardware filtering. Python training keeps exact bilinear.
- **What:** the engine's image reads a bilinear tap with one `SampleLevel` through a linear-clamp sampler instead of four `Load`s. Training keeps the four-load version. The filter formulas in `glare.slang` / `pyramid.slang` stay shared: the bilinear tap moves behind the `IImage` interface (`bilinear(p)`, `bilinear_zero(p)`), which each side implements.
- **Expected:** up to 4× fewer texture-unit operations wherever taps dominate. Streak ≈ 45–60 µs (from 168), rays ≈ 3–4 ms, levels ≈ 0.5 ms, frame ≈ 4.5–5.5 ms at 1080p: near tier A.
- **Precision:** hardware bilinear quantises the blend fractions (8 bits on NVIDIA, to be confirmed) and may filter fp16 texels at its own precision. Python models the fraction rounding behind a compile-time switch (0 = exact, so `test_glare_chain`'s NumPy check is unchanged); parity measures the rest.
- **Zero border:** Unity's inline samplers have no clamp-to-border, so `bilinear_zero` uses the hardware fetch when the 2×2 footprint is inside the image, returns 0 when it is wholly outside, and keeps the four loads only where it straddles the edge.
- **Order:** (1) timing-only test: hardware path for `bilinear_zero` (the rays), Python untouched; (2) if it pays, `bilinear` too (levels, pyramid, composite); (3) the precision model, rescore, parity; (4) the Unreal wrapper binds its own sampler.

### 1a′. R11G11B10 streak targets — tried, rejected (decision log 2026-11-24)
- **Result:** rays 9.73 → 9.44 ms at 1080p, 16.81 → 16.29 ms at 1440p (**1.03×**), for validation FLIP +1.66e-4 (0.76 δ_coma; truncation to 6 / 5-bit mantissas darkens the rays). Code removed; `neuralglare.storage` keeps the fp16 model.
- **Reading:** halving the bytes per fetch again did almost nothing, so after 1a the limit isn't bytes. Likely the texture filter rate: 128-bit texels filter at half rate, 64-bit and smaller at full rate (not measured). Storage formats are exhausted on this GPU.

### 1b. Fuse the cross filter into the last streak pass — tried, rejected (decision log 2026-11-24)
- **Result:** rays 9.73 → **12.39 ms** at 1080p (+27 %), 16.81 → **21.63 ms** at 1440p (+29 %), timing only, fp16 weights. Taps per ray 15 → 20 (1.33×): the cost follows the bilinear fetch count, and the saved write barely counts. Code removed; the decision-log row is the record.
- **Lesson for the remaining steps:** on this GPU, rank changes by fetches removed, not by passes or bytes written.
- **What:** S′(x) = Σ_m a_m(x) Σ_n c_n S2(x − (m − 1) n_k − n · step), i.e. 12 taps on S2 in one pass, instead of writing S3 and reading it back with 3 taps.
- **Expected:** removes 14 full-resolution write-and-read round trips: about 7/9 of the ray traffic remains.
- **Risk:** not bit-identical. It samples S2 at off-grid points instead of S3, so the interpolation differs slightly. A new `glare.slang` function used by both sides; FLIP within δ, likely without retraining.
- The same fusion applies to brightness, which already rides on the cross pass.
- **Caution from the trace:** fusing removes a write and a read, but raises streak 3's taps per pixel from 4 to 12, which loads the texture pipe, the unit that is already busiest. The saving may be well below the traffic model's 2/9. Measure it after 1a.

### 1c. Cheaper family accumulation — done, kept (decision log 2026-11-24)
- **Accuracy result:** Unity validation FLIP 0.04093978653351466, identical to 1e's (+1.17e-5 vs 0.040928, 0.05 δ_coma ⇒ PASS); parity PASS (`spikes` 0.224 / 0.212 / 0.247); per source max |Unity − Python| 1.76e-5.
- **Step 1 result:** 1080p 8.64 → **7.05 ms** (rays 7.39 → 4.60 as predicted, composite 0.58 → 1.77, above the 0.8–1.2 predicted); 1440p 14.68 → **11.99 ms**. The composite runs at ≈ 170 GB/s, 38 % of DRAM peak: its cost was read as its ≈ 1.3 k MACs/px of uv-only MLPs, which pointed to 1d; 1d then showed the MLPs were hidden behind fetches (−0.27 ms when removed): the cost is the 42 ray taps.
- **Step 2 (accuracy) authored:**
  - Python: `train_glare.slang` `DiffTensorStack` / `spike_stack_pass`; `train_points.spikes(stack=)`, default `RAY_STACK = True` wherever it applies, training included, as GlareChain.
  - Tests (`test_spikes.py test_stack_*`):
    - Forward matches the per-ray chain within 2e-5 of the peak. The per-ray composite's `sample_bilinear` at texel centres misses the integer by up to an ulp; the stack reads at integer offsets, so the stack is the exact one.
    - Pre-exposure is undone exactly.
    - Gradient matches at rtol 1e-4 outside BrightMLP, and within 1e-3 of the block maximum inside it. There the per-ray chain's split e^{z} / Z cancellation is noise, and the stack's bias gradient is the smaller.
  - Rescore of `ray_bright` fp16: validation FLIP 0.0409392 (−2.8e-7 vs the per-ray path).
  - Harness: loads the stack shaders and records `rayStack` in `unity_run.json`. `compare` / `score` then run Python's stack path and skip the family dumps.
- **Step 1 status:** `glare.slang` (`IRayStack`, `RaySlice`, `spike_composite_stack`), `engine_glare.slang` (`TextureStack`, `StreakToStack`, `SpikeCompositeStack`), the two `.compute` wrappers, `GlareChain.RayStack` (Inspector `rayStack`, command line `-ngRayStack 0|1`, recorded as `rayStack` in the timing JSON). The parity harness doesn't load the stack shaders, so it still runs the per-ray passes until step 2.
**The idea: fold the cross filter, the brightness and the family sums into the composite.** Each ray's last streak pass writes S3 into its own slice of a 14-slice fp16 texture array (the *ray stack*). The composite then reads all 14 slices and does the rest in registers: each ray's 3 cross taps, its e^{z_k}, both family sums and their normalisers Z. The fp32 family sums are never stored.

- **Why it pays (1080p, measured 2026-11-24):**
  - Today the cross stage is 14 × ≈ 209 µs ≈ 2.93 ms plus the composite's 0.58 ms, ≈ 3.5 ms.
  - Each cross pass moves 40 B/px: 8 B reads S3, 16 B reads the family sum, 16 B rewrites it. Across 14 rays that is 560 B/px.
  - The fused composite moves ≈ 144 B/px: 14 × 8 B of slices, 16 B base, 16 B output.
- **Free extras:**
  - The composite already evaluates all 14 BrightMLPs for Z (`bright_norm`). Today the cross passes evaluate them a second time; the fused pass evaluates each once and uses e^{z_k} for both the numerator and Z.
  - The fp32 range problem of the family sums (9.1e4, the reason they couldn't be fp16) disappears, because the sums only ever exist in registers.
- **Same model, no retraining.** It's the same arithmetic in the same order: each family is summed ray 0, 2, 4, … in fp32, times 1/k, divided by Z. The only difference from today is that a fp32 sum is no longer stored and reloaded. A fp32 store is exact, so Unity's output should change by fp32 rounding at most.
- **Shared code (`slang/glare.slang`):**
  - A new interface `IRayStack { int2 size(); float3 load(int ray, int2 texel); float3 bilinear_zero(int ray, float2 p); }`.
  - An adapter `RaySlice<R : IRayStack> : IImage` (stack + ray index), so `ray_cross` is reused unchanged.
  - `spike_composite_stack<I, R, P>(texel, out_size, base, rays, params, offset, scale)`. It loops over the 14 rays, accumulating `e^{z_k} · ray_cross(RaySlice{rays, k})` and `e^{z_k}` per family, then returns `w0·base + w1·E/Z_even + w2·O/Z_odd`.
  - The ray loop stays a loop (not `[ForceUnroll]`), to keep 14 unrolled RayMLP + BrightMLP from blowing up registers.
- **Engine (`engine_glare.slang`):**
  - `TextureStack : IRayStack` over `Texture2DArray<float4> _RayStack`, with the same hardware path as `bilinear_zero`: `SampleLevel(sampler_LinearClamp, float3(uv, ray))` inside the image, 0 outside, four `Load`s on the edge.
  - Two new kernels: `StreakToStack` (the last streak pass, writing `RWTexture2DArray _RayStackOut[uint3(texel, _Ray)]`) and `SpikeCompositeStack`. Both go into the KERNELS list in `build_hlsl.sh`.
- **Unity (`GlareChain.cs`):**
  - One `RenderTexture` with dimension `Tex2DArray`, `volumeDepth` 14, ARGBHalf, random write. The four fp32 family targets and their ping-pong go away.
  - Used when the layout has brightness and `spikeSource == "input"` (the composite reads the families at its own texels). Other layouts keep today's passes.
  - **Memory at 1080p:** stack 232 MB, minus the 133 MB of family targets, net +99 MB. At 1440p +176 MB; at 4K +400 MB.
- **Python (`train_glare.slang`, `train_points.spikes()`):**
  - `TensorRayStack : IRayStack` over the 14 stored S3 tensors.
  - `spikes(..., bright=True)` calls `spike_composite_stack`, so eval and `score` use the same function as Unity.
  - A test checks it against today's chain (`ray_bright_accumulate` + `spike_composite_bright`) to fp32 rounding (rtol 1e-6).
  - Training uses it too if slangpy's backward handles a stack of differentiable tensors. If not, training keeps today's mathematically identical chain, and the test is what ties the two together.
- **Parity:** the even / odd family dumps disappear for this path. `compare` checks mip0, full and spikes; `score` reruns the gate.
- **Expected:**
  - The fused composite is bound by DRAM: ≈ 0.67 ms at 448 GB/s for 144 B/px.
  - ALU: about 1.1 k MACs/px (14 RayMLP + 14 BrightMLP + SpikeMLP) ≈ 0.7 ms at 1410 MHz, partly overlapping with memory.
  - Estimate **≈ 0.8–1.2 ms, replacing ≈ 3.5 ms**: the frame goes from 8.64 to **≈ 6.0–6.4 ms at 1080p**.
  - What remains is mostly the 42 streak passes (≈ 4.6 ms), which is the model itself (→ 2c).
- **Risks:**
  - Register pressure from 14 rays × two MLPs in one thread. If occupancy collapses, try a two-dispatch version (one per family, each writing a fp32 partial that the second dispatch adds): 2 extra transfers instead of 28.
  - Slang must legalise a texture inside the generic `RaySlice`.
  - Unity may insert a UAV barrier between the 14 dispatches that write different slices of one array (small cost; visible in a trace).
- **Order (as 1e):**
  1. Timing-only test: the slang functions, the kernels and GlareChain; Python untouched.
  2. If it pays: Python `TensorRayStack`, `spikes()` and the equivalence test.
  3. Parity, then `score` against the gate (validation FLIP within δ_coma of 0.040928; predicted change vs 1e: |Δ| ≤ 1e-6).
- **Rejected alternative:** a separate family pass (7 slices in, one fp32 family sum out, ≈ 72 B/px per family). It still writes and re-reads two fp32 sums, and evaluates BrightMLP twice. Folding it into the composite saves both.

Original note, kept for the record:
- **What:** today every ray reads and rewrites its family's full-resolution running sum: 14 extra reads and 14 extra writes.
- **Options:** two rays per dispatch (one of each family, writing both sums); or accumulate in the composite from per-ray outputs (more memory: 14 targets).
- **Expected:** a modest saving; take it only if step 0 shows the accumulate reads matter.
- **Update (2026-11-24, after 1e part 1):** they matter now. The cross pass is bandwidth-bound, and the fp32 family read and rewrite is 32 of its 40 B/px: 14 × ≈ 209 µs ≈ 2.9 ms per 1080p frame. Revised option: each family's 7 rays stored (fp16, 7 targets per family) and read by one pass that applies each ray's cross filter and brightness and writes the family sum once: ≈ 72 instead of 280 B/px per family, cross stage ≈ 0.8 ms. Memory ≈ 116 MB per family at 1080p (4K: 4×); batches of rays trade memory for traffic.

### 1d. Bake the uv-only networks into low-resolution textures — tried, rejected (decision log 2026-11-24)
- **Result (timing only, 1080p, RTX 2070 at 1410 MHz, stride 8):** frame 7.05 → 6.82 ms (−3 %).
  - Composite 1.77 → 1.51 ms, against a prediction of 0.7–0.9.
  - Levels 0.63 → **0.68** ms (slower), against a prediction of 0.35–0.45.
  - Unrolling the baked composite's ray loop: 1.59 ms (slower).
  - Baking the composite only: 6.87 ms.
  - 1440p: 11.99 → 11.70 ms.
  - Code removed; the decision-log row is the record. Kept: the `IImageArray` / `ImageSlice` rename and the weights/apply split (`upsample_weighted`, `ray_cross_weighted`).
- **Reading:**
  - The networks' arithmetic was already hidden behind fetches in both passes. Removing all 29 MLPs per pixel saved only 0.27 ms of the composite. The levels pass, fetch-bound at ≈ 30 taps/px, only gained 6 more fetches.
  - The "≈ 0.9 ms per 1 k MACs/px" rule that motivated 1d came from comparing two composites that also differed by 42 ray taps: a fetch cost misread as an arithmetic cost.
  - What is left in the composite (≈ 1.5 ms) is its 42 bilinear taps from 14 slices. Not unroll-latency-bound; possibly texture-cache contention between slices (not confirmed: two GPU traces of the baked build disagreed with the timing by 46–100 %, inflating exactly the passes that read the baked arrays, so neither was read).
- **Consequence:** every remaining pass is fetch-bound. Engine-side changes are largely exhausted on this GPU; the next gains must cut taps, which means changing the model (2a, 2c).

Design, kept for the record:
- **Step 1b (after the first timing):** `spike_composite_baked`'s ray loop unrolled (`[ForceUnroll]`), and `GlareChain` can bake the composite only (`-ngBakeLevels 0`, recorded as `bakeLevels`).
- **Status:**
  - *Shared slang:* `pyramid.slang` `IImageArray` / `ImageSlice` (the 1c stack interface, renamed and moved). In `glare.slang`: `upsample_weighted`, `ray_cross_weighted`, `bake_grid` / `bake_uv` / `baked_coord`, `bake_level_weights`, `bake_spike_weights`, `upsample_baked`, `spike_composite_baked`.
  - *Engine:* `engine_glare.slang` kernels `BakeLevelWeights`, `BakeSpikeWeights`, `UpsampleBaked`, `SpikeCompositeBaked`, with their `.compute` wrappers.
  - *GlareChain:* `BakeStride` from the layout's `bakeStride`, or `-ngBakeStride N`; recorded as `bakeStride` in the timing JSON. The bake runs on the first Record after a resize, in its own `NeuralGlare.Bake` scope outside `NeuralGlare`.
  - *Python and the parity export:* untouched until step 2 (layouts have no `bakeStride`, so parity runs unbaked).
**The idea.** No network in the model ever reads the image:
- LevelMLP sees uv.
- SpikeMLP sees uv.
- RayMLP and BrightMLP see the field vector v, which is a linear function of uv.

So every weight is a fixed function of screen position (a fixed screen-space map for a given resolution and set of weights). Evaluate the networks once on a coarse grid when the resolution or the weights change, store the *weights* (the softmax outputs, not the logits) in small texture arrays, and fetch them with bilinear filtering per pixel. This is the same move as baking a BRDF's split-sum LUT: replace per-pixel ALU by a texture lookup of a smooth function.

- **Why it pays (1080p, after 1c).** The composite takes 1.77 ms at ≈ 38 % of DRAM peak. Its cost follows its ≈ 1.3 k MACs/px of MLPs, and every weight is a buffer load. As a rule of thumb from today's composites, that is ≈ 0.9 ms per 1 k MACs/px.
  - The levels pass carries LevelMLP 2-16-18 (≈ 350 MACs/px at full resolution, plus the half-resolution level), so roughly 0.3–0.4 of its 0.63 ms.
  - Baked, each pixel does fetches only:
    - composite: 3 spike-mix + 14 per-ray fetches;
    - levels: 6 fetches per level.
  - The grid is small and every 8×8 thread group shares 1–4 of its texels, so the fetches hit L1/L2, not DRAM.
- **What is baked (per grid texel).**
  - *Levels:* one array per level, 6 slices, slice t = tap t's weights (R, G, B), the per-channel softmax.
  - *Composite:* one array, 17 slices.
    - Slices 0–2 hold SpikeMLP's mix weights w0, w1, w2 (R, G, B).
    - Slice 3 + k holds ray k's three cross taps **with the brightness folded in**: c_km = a_km(x) · e^{z_k(x)} / Z_f(x), where f is k's family.
    - Then a family's sum is Σ_k Σ_m c_km S_k(x − (m−1) n_k), with no exp and no division per pixel. The c_km of a family sum to 1 at every texel.
- **Weights, not logits, so the properties survive interpolation.** Bilinear interpolation is a convex combination. The softmax weights stay positive and still sum to 1 per channel (per family for c), so light is conserved by construction. Interpolating logits would need the softmax back per pixel.
- **Accuracy: an approximation, measured.**
  - The networks are piecewise linear (leaky ReLU), so a bilinear bake is near-exact away from the kinks, with an error of O(h) at them.
  - *Grid:* corner-aligned, texel i at uv = i / (N − 1), so the frame's edges are exact. N = ceil(size / stride) + 1, with the stride in output pixels.
  - *Pre-registered rule:* sweep strides 4 / 8 / 16 / 32 in Python. Take the coarsest whose validation FLIP is within **2e-5 (≈ 0.1 δ_coma)** of the unbaked stack path. The efficiency steps so far have used 1.17e-5 of δ_coma in total.
  - *Storage:* fp16 (RGBA16F), rounded toward zero as the 2070 stores it, modelled in Python (`neuralglare.storage`). The sweep repeats the chosen stride in fp32. Toward-zero fp16 biases the weights by ≈ −2.4e-4 and with them the energy; if fp32 is measurably better, fp32 it is.
- **Shared code (one source of truth).**
  - *Split each pass into "weights" and "apply":*
    - `upsample_weighted(texel, mip, low, w[TAPS])`;
    - `spike_composite_stack_weighted(texel, base, rays, mix, c)`;
    - the existing functions become "evaluate the MLPs, then apply", unchanged for training.
  - *Bake functions:* `bake_level_weights(cell, grid, …)` and `bake_spike_weights(cell, grid, …)` evaluate the MLPs at `bake_uv(cell, grid)` and return what a texel stores. `baked_coord(uv, grid)` gives the lookup position.
  - *Arrays:* the baked arrays are read through the 1c stack interface. Rename `IRayStack` → `IImageArray` and `RaySlice` → `ImageSlice`, since it now holds more than rays.
  - *Training:* keeps the MLPs. Baking is an evaluation-time approximation, applied to the trained weights.
- **Engine.**
  - Kernels `BakeLevelWeights` / `BakeSpikeWeights` run in `GlareChain.Resize` (and when weights load), outside the timed scopes; their cost is reported once.
  - `UpsampleBaked` / `SpikeCompositeBaked` replace `Upsample` / `SpikeCompositeStack` when the layout says so: a new layout key `bakeStride` (0 = off), written by `unity_parity.py export`, so Python and Unity bake alike.
  - Memory: tiny (stride 8 at 1080p: 241 × 136 × 17 slices × 8 B ≈ 4.5 MB).
- **Python.**
  - `train_points --bake-stride s` (rescore / eval only): bakes the grid with the same slang functions into tensors, stores them as the engine does, and evaluates through the `_weighted` functions.
  - `unity_parity` reads `bakeStride` from the layout.
  - *Tests:*
    - stride 1 with fp32 storage reproduces the unbaked output to fp32 rounding (the grid points are the pixels);
    - the weights still sum to 1 after interpolation;
    - the folded c_km reproduce `spike_composite_stack`.
- **Expected:**
  - Composite 1.77 → ≈ 0.7–0.9 ms. It becomes DRAM- and fetch-bound: 144 B/px and ≈ 60 fetches/px.
  - Levels 0.63 → ≈ 0.35–0.45 ms.
  - **Frame 7.05 → ≈ 5.6–6.0 ms at 1080p.**
  - Timing barely depends on the stride (the grid is cache-resident), so the timing test can run at stride 8 before the sweep picks one.
- **Risks:**
  - Kinks in a coarse grid: the error is O(h) there, which is what the sweep measures.
  - The levels pass may stay fetch-bound (≈ 30 fetches/px for tent, tent and binomial), capping its gain.
  - The 2070's ≈ 8-bit filter fractions on the weight grid: fine for smooth fields; `score` measures what ships.
  - The rays' 42 taps and 4.6 ms of streaks are untouched: tier A still needs the streak cost cut (2a / 2c).
- **Order (as 1c):**
  1. Timing-only at stride 8: the slang split and bake functions, the kernels, GlareChain; Python untouched.
  2. If it pays:
     - Python: bake evaluation, tests, stride sweep and the fp16 / fp32 check, choosing the stride by the rule.
     - Then parity, `score` and the views (`views.py --study efficiency`, with the baked rescore added as a run).
- **Out of scope, noted:** the baked grid could *be* the trained parameters (a small learned texture per weight, no MLP), the neural-texture idea. That is a model change (Step 2 territory) and gets its own decision-log row if pursued.

## Step 2: changes that need retraining

### 2a. Two streak passes of 8 taps instead of three of 4 — rejected on timing, not trained (decision log 2026-11-25)
- **Result (2026-11-25, idle machine, compile-time variant build):** 1080p 4 × 3 7.40 ms (rays 4.99), 8 × 2 init 7.31 ms (rays 4.92); 1440p 12.77 / 12.74 ms. Rays 1.4 % faster at best, against a predicted frame of 5.6–6.2 ms: the passes are fetch-bound, so 28 passes × 16 taps cost what 42 × 12 do. The 2026-11-24 7.45 ms was not machine load; the 4 × 3 rays are still +0.4 ms over pre-2a (4.60).
- **Done (2026-11-25):** plumbing removed, back to the constants 4 / 3 / 24; rescore bit-identical. Re-timed: **6.98 ms at 1080p** (rays 4.57), **11.95 ms at 1440p**, so the +0.4 ms was the 2a code. These are the new shipping figures.
- *What follows is the record of the attempt.*
- **Status:**
  - *Slang:* the streak geometry is a run-time `taps` (4 or 8), through `streak` / `streak_accumulate` / `spike_composite_bright` / `spike_composite_stack`. The body is `streak_n<N>`, compiled for both; `streak_passes`, `profile_count`; the constants `PASSES` / `STREAK_TAPS` / `PROFILE_COUNT` are gone.
  - *Engine:* `_StreakTaps` in 5 kernels; `GlareChain` passes the layout's `streakTaps` (missing = 4) and checks passes and profile count against it.
  - *Python:* `train_levels.set_streak_taps` / `spike_layout(module, taps)`; `engine_layout` writes `streakTaps` when not 4; `train_points --streak-taps` (tag `_taps8`), study `streak_taps`; `--steps 0` exports the untrained init (tag `_init`); `unity_parity` takes the geometry from the layout.
  - *Checks:* 196 tests, with the geometry tests parametrised over 4 and 8, plus `test_exponential_init_is_exact`. The `ray_bright` fp16 rescore is bit-identical (0.040939203649759295).
  - *Timing model:* untrained 8×2 init exported as `outputs/04_timing/models/ray_bright_taps8_init_fp16.{bytes,json}`; its init validation FLIP is 0.0477, against 0.0479 for 4×3.
- **First timing (same build, 2026-11-24): the run-time tap count slowed the 4×3 model.** 1080p rays 4.596 → 5.013 ms (+9 %), composite 1.774 → 1.877; levels unchanged (same clock).
  - The 8×2 runs recorded nothing: the copied timing layout lacked `mips` (train_points writes no `mips`; `unity_parity export` adds it). Added to the copy.
  - *Cause, by FXC:* with `taps` at run time a streak kernel holds both bodies: 12 DXBC temps and ≈ 2.6× the code, against 10 temps for a 4-only build. A GPU allocates for the worst path, so the uniform branch wasn't free.
  - *Fix:* the tap count is a compile-time variant in the streak wrappers (keyword `NG_STREAK_TAPS_8` defines `_StreakTaps` as 8, else 4; `GlareChain.SetStreakTaps`), so each variant compiles one body. The composites keep an int uniform: it only shifts an offset, and FXC gives 60 temps either way, so their +0.1 ms is unexplained so far.
**The idea.** A ray today is three passes of 4 taps with steps 1, 4 and 16 px. Each pass moves light by one base-4 digit of the distance, so together they reach every distance 0–63 exactly once (like a separable blur done as a cascade of small kernels). Base 8 does the same in two passes: 8 taps with steps 1 and 8 px reach d = e0 + 8 e1, with e0 and e1 in 0..7. One full-resolution pass per ray fewer (42 → 28 per frame), at 8 instead of 4 taps per pass (12 → 16 taps per ray).

- **Why it might pay, and why it might not.** Two cost models give opposite answers. Neither can be settled from the measurements so far, so the timing comes first.
  - *DRAM-bound passes:* a streak pass reads and writes 16 B/px whatever its tap count, because neighbouring pixels share taps along the ray, so they hit cache. 1e measured streaks at ≈ 110 µs against a ≈ 74 µs bandwidth floor. Then an 8-tap pass costs ≈ 110–130 µs, and a ray 2 × 120 = 240 instead of 330 µs: **rays 4.6 → ≈ 3.2–3.8 ms, frame 7.05 → ≈ 5.6–6.2 ms**.
  - *Fetch-bound passes:* if the cost follows taps, as before 1e (1b: +33 % taps gave +27 % time), an 8-tap pass is ≈ 2× today's, and the rays get slower (≈ 6 ms).
  - After 1e the streak passes dropped 1.5× when their fetch instructions dropped 4×, which points to the first model, but doesn't prove it.
- **Accuracy: a different family of profiles.**
  - Today a ray's profile over distance is p0(d0) · p1(d1) · p2(d2), three 4-way softmaxes per family (12 logits). 2a makes it q0(e0) · q1(e1), two 8-way softmaxes (16 logits).
  - Neither family contains the other. Today's p1 spans e0 and e1, so it doesn't factor into base-8 digits.
  - The init is exact in both: z[p][n] = −n · taps^p / decay gives exp(−d / decay) for any base. So 2a trains with `ray_bright`'s recipe unchanged (`--arch bright --ee-weight 0.02`, lr 1e-3, 2000 steps, seed 0, the cross and brightness pre-fits), a like-for-like retrain.
  - A possible gain: off-axis rays take a bilinear tap per pass, and each pass blurs the ray across its width a little. Two passes blur less than three, and the remaining error lives in the far rays' 1.2–2.4 px cross-section (ceiling, 2026-11-19).
- **Shared code: geometry at run time, not compile time.**
  - The streak geometry becomes a run-time parameter (taps ∈ {4, 8}, passes = log_taps(64)) instead of the constants `STREAK_TAPS` / `PASSES`.
  - That way the shipping `ray_bright` (4 × 3) and the new model (8 × 2) run on the same kernels: one build times both, and nothing that ships stops running while 2a is tried.
  - *`glare.slang`:*
    - `streak(…, taps, stage)` takes the tap count. Arrays are sized `MAX_STREAK_TAPS = 8`, the loop is bounded by `taps`, and the softmax runs over the first `taps` logits. The bound is uniform, so the branch costs nothing.
    - Step = taps^stage.
    - The offsets after the profile logits (RayMLP, BrightMLP, in `spike_composite_*` / `ray_bright_accumulate`) take the profile count as a parameter instead of `PROFILE_COUNT`.
  - *Layout key `streakTaps` (missing = 4):* `passes` and `profileCount` already exist and follow from it. GlareChain passes `_StreakTaps` and dispatches `passes` passes, as it already loops over `layout.passes`.
  - *Python:* `train_points --streak-taps 8` (tag `_taps8`); `spike_sizes()` / `spike_layout()` report taps and passes from the run's setting, not from constants.
- **Pre-registered:**
  - *Timing-only first:* the untrained 8 × 2 init, exported, since cost doesn't depend on the weights. If the rays aren't faster, stop there; log it and don't train.
  - *If faster:* train `ray_bright_taps8`. **Kept if** validation FLIP ≤ 0.040928 + δ_coma (0.041146), with test bins reported. Unity parity and `score` as for 1c (gate on Unity's own output), and the `efficiency` views with the new model added.
  - If it is faster but over the gate, it is a point on the cost-quality curve, not the shipping model.
- **Order:**
  1. Slang and engine generalised (taps at run time), with tests: 4 × 3 must reproduce today's outputs bit for bit, and 8 × 2 with the exponential init must reach exp(−d / decay) exactly, as the 4 × 3 init test does.
  2. Timing-only with the 8 × 2 init.
  3. Train, evaluate, parity, score, views.
- **Doesn't reach tier A on its own:** at best ≈ 5.6 ms. The composite's 42 taps (≈ 1.5 ms) and two passes of 14 rays remain → 2c.

### 2b. The far passes at half resolution — not pursued (decision log 2026-11-24, 1a′ row)
- **Why not:** the streak passes are in series, so a pass at half resolution widens the whole ray, not just its far part (half-resolution rays were 3.4 px wide against the exact 1.2–2.4 px; full resolution was worth −68.6δ, 2026-11-05). The hybrid that keeps the near part at full resolution and runs only the 16–63 px taps at half resolution saves ≤ 13 % of the fetches (≈ 15 → 13 per ray) and still widens a third of each ray's light.
- **What:** the pass that reaches 16–63 px works on a half-resolution copy and is upsampled at the end; the near passes stay at full resolution.
- **Expected:** the far pass costs about 1/4.
- **Fit to the bottleneck:** of all the options, lower resolution is the one that cuts texture-pipe work directly (4× fewer fetches per pass), so it targets the measured limit best. The accuracy risk below is unchanged.
- **Risk:** high. Full-resolution rays were the largest single gain (−68.6δ, 2026-11-05), and the remaining error lives in the far rays' cross-section (1.2–2.4 px wide, ceiling 2026-11-19), which a 2 px texel can't hold. Try it only if 1a–2a don't reach tier A, and expect a FLIP cost.

### 2c. Rays only from bright pixels (sparse splatting) — stage A designed (decision log 2026-11-25)
- **Stage A design (2026-11-25):**
  - *Exact stamp:* a ray is shift-invariant on the integer grid (the tap fractions depend on n, pass and ray only), so stamping its impulse response K_k at a source texel equals the gather. K_k comes from glare.slang `streak` on an impulse, once per weight load. Differences: ≈ 2 px at the frame edge, fp16 blend rounding.
  - *Split:* b = saturate((L − T) / (knee · T)); b·src splatted, (1 − b)·src the residual (dropped, or a low-resolution gather: measured first). Soft knee against popping.
  - *Unity:* ordered compaction (prefix sum; deterministic) → `DrawProceduralIndirect`, 14 oriented quads per bright texel into the fp16 ray stack (additive, `SV_RenderTargetArrayIndex`) → composite unchanged.
  - *Python:* by linearity, `spikes()` on b·src is the splat. No new ray code.
  - *Measured (12 HDRI crops):* the brightest 1k px hold 0.83–0.95 of the light in sun / lamp scenes, 0.02–0.10 in studio / cloister (0.46–0.97 in the brightest 50k).
  - *Cost model:* ≈ 2.8k fragments per bright texel, ≈ 0.1 ms per 1k (± 2×); break-even ≈ 30k–60k; frame ≈ 3.5 ms at 10k with the residual dropped.
  - *Order:* (1) Python study on the HDRI crops (T sweep, residual dropped; FLIP vs the full model and vs the reference; gate: ≤ δ_coma rise against the reference on every crop); (2) Unity timing-only N sweep; then the slang split and kernel bake, the splat pass, parity on goegap (fp16 overlap precision).
  - *Stage B (later):* cross filter and brightness at the source, splat into the family sums, cheaper composite; retraining.
- *Original note:*
- **What:** find pixels above a threshold and draw each one's 14 rays as sprites (scatter), instead of gathering at every pixel. This is the usual game approach for lens-flare streaks and bokeh.
- **Expected:** cost scales with the number of bright pixels, not the resolution. It is the only option here that can plausibly reach tier B on real scenes.
- **Risk:** the biggest design change.
  - The output is no longer linear in the light (the threshold), which the model has kept by construction since 2026-11-02.
  - Cost becomes content-dependent, so it must be timed on real scenes (Step 3 of Phase 2).
  - Rays from dim pixels are dropped: an approximation to measure on HDRI crops, not point lights.
  - A separate research question; plan it in its own decision-log row before building anything.

## Step 3: report

- [ ] Every change: timing at 1080p / 1440p (/ 4K once DSR works), FLIP, parity, decision-log row, ablation-table row with the measured ms column.
- [ ] A cost-quality plot: edge FLIP against ms at 1080p for every timed model, with Unreal's FFT and Gaussian bloom and the fitted URP pyramid as reference points.
- [ ] Repeat the final model's timings on a second GPU (the plan's "at least 2 GPUs, one low-end").

## Framing for the paper

The FFT costs 1 ms but applies one kernel everywhere. A spatially varying FFT reference would need several kernels, i.e. several FFTs blended across the frame (e.g. a 3 × 3 grid ≈ 9 FFTs). The honest comparison is the model's cost against what the FFT route would need for the same quality at the edge, as well as against the single-kernel FFT. Measure that baseline's cost before claiming either way.
