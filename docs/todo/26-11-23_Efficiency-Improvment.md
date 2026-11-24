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

Not timed: the final blit back into the camera colour. Memory: about 274 MB of RGBA32F targets at 1080p.

**The rays are the problem.** Everything else together is 2.2 ms. The rays run 56 full-resolution dispatches per frame, and each reads and writes a 16 B/px image.

Rough traffic model per ray (1080p image = 33 MB in RGBA32F):
- 3 streak passes: 3 writes + 3 reads (the source, S1, S2)
- cross pass: 1 write + 2 reads (S3 and the family's running sum)
- ≈ 9 image-sized transfers per ray × 14 rays ≈ 4.2 GB per frame
- ≥ 9 ms at the 2070's 448 GB/s, before cache misses on the diagonal 16 px steps

So the working hypothesis is that **the rays are bandwidth-bound**: limited by bytes moved, not by arithmetic. Step 0 tests that before anything is built on it.

## Targets (to fix in the decision log before step 1)

Suggested; to be decided:
- **Tier A: ≤ 4 ms at 1080p.** An expensive but shippable post effect on a mid-range GPU.
- **Tier B: ≤ 1 ms at 1080p**, the FFT's cost. At FFT quality on the edge, this is the paper's strongest claim: spatially varying glare for the price of a uniform kernel.

Accuracy guard for every change: validation FLIP no worse than `ray_bright`'s by more than δ_coma (0.000218), test FLIP reported per bin. A change that trades accuracy for cost is reported as a point on the cost-quality curve, not silently kept.

## Ground rules

- **One change at a time.** Each one gets timing (protocol of 2026-11-23), FLIP, Unity parity and a decision-log row.
- **One source of truth still holds.** A change to how the model computes (fused passes, a new factorisation, lower resolution) goes into `slang/glare.slang` and is used by Python and Unity alike. Only storage formats and dispatch plumbing live in `GlareChain.cs`.
- **Storage precision is modelled in Python too.** If Unity stores an intermediate in fp16, the Python evaluation rounds the same intermediate to fp16 (`.half().float()`), so parity stays meaningful and FLIP describes what ships.
- **Same machine, same protocol:** clock locked at 1410 MHz, D3D12, Development Build, 10 s warm-up + 5 × 2 s windows, the same scene.

## Step 0: measure before changing anything

- [ ] **GPU trace of one frame** in Nsight Graphics (GPU Trace, D3D12) or PIX. For a streak pass, a cross pass, the level pass and the composite, read throughput as a share of the GPU's peak for memory (DRAM / L2), the texture units and the shader cores. This tells which resource limits each pass.
  - Bandwidth-bound rays → steps 1a–1c pay off.
  - Texture-unit- or latency-bound rays (diagonal 16 px steps missing the cache) → reordering or lower resolution (step 2) matters more.
  - Find out why the level pass costs 1.58 ms. 22 fetches and ≈ 320 MACs per pixel suggest well under 1 ms, so low occupancy (too many registers from 18 logits and 3 softmaxes) is a suspect.
- [ ] **Time the earlier models that run on the current kernels:** `wide_core` (no cross filter, no brightness), `ray_cross`, `ray_bright`. The differences are the measured cost of the cross filter and of the brightness. They become a measured "ms at 1080p" column in the ablation table, next to the MACs column, which missed the rays' real cost.
- [ ] **Time the blit** back into the camera colour once, with a temporary scope, and add it to the total.
- [ ] **4K:** enable DSR 2.25× in the NVIDIA Control Panel and rerun. Check how Unreal's 4K number (5.48 ms) was measured on this monitor.
- [ ] **Confirm the Unreal side:** the API and build configuration of the 2026-10-29 timings, so the comparison is like for like.

## Step 1: engine-side changes, no retraining

Ordered by expected gain per effort. The gains are estimates from the traffic model, to be replaced by measurements.

### 1a. fp16 intermediates (RGBA16F)
- **What:** store the ray intermediates (S1–S3, the family sums) as half floats; later possibly the level output too.
- **Expected:** if bandwidth-bound, about 2× on the rays (≈ 14 → 7 ms).
- **Risk:**
  - **Precision:** fp16 has a 10-bit mantissa, ≈ 1e-3 relative error per stored value. The family sums add 7 rays, so their rounding accumulates. That is far above the parity tolerance (1e-4) unless Python rounds identically, but probably far below what FLIP sees. Measure; don't assume.
  - **Range:** fp16 tops out at 65504. The point grid's sources are 1e4, but real HDR scenes (the sun) can exceed it, and fp16 turns those into infinity. Clamping would break the HDR rule (never clamp inputs), so keep the targets that can hold a whole source's light (the first streak pass, the full-resolution level) in fp32, or scale by a known exposure before storing and undo it after.
  - **Underflow** isn't the issue: the faintest tail the loss sees (ε = 1e-7 × a 1e4 source = 1e-3) is well inside fp16's normal range (≥ 6e-5).
- **Checks:** Python rounds the same intermediates; parity tolerance unchanged if both sides round identically; FLIP within δ.

### 1b. Fuse the cross filter into the last streak pass
- **What:** S′(x) = Σ_m a_m(x) Σ_n c_n S2(x − (m − 1) n_k − n · step), i.e. 12 taps on S2 in one pass, instead of writing S3 and reading it back with 3 taps.
- **Expected:** removes 14 full-resolution write-and-read round trips: about 7/9 of the ray traffic remains.
- **Risk:** not bit-identical. It samples S2 at off-grid points instead of S3, so the interpolation differs slightly. A new `glare.slang` function used by both sides; FLIP within δ, likely without retraining.
- The same fusion applies to brightness, which already rides on the cross pass.

### 1c. Cheaper family accumulation
- **What:** today every ray reads and rewrites its family's full-resolution running sum: 14 extra reads and 14 extra writes.
- **Options:** two rays per dispatch (one of each family, writing both sums); or accumulate in the composite from per-ray outputs (more memory: 14 targets).
- **Expected:** a modest saving; take it only if step 0 shows the accumulate reads matter.

### 1d. Cache the uv-only networks
- **What:** LevelMLP, SpikeMLP, RayMLP and BrightMLP depend only on screen position (and the ray), never on the image. Their outputs could be baked once per resolution into small textures and fetched with bilinear filtering.
- **Expected:** small. Step 0 decides it: shader cores are cheap, the MLPs total well under 0.5 ms of arithmetic, and a baked full-resolution weight texture costs more bandwidth than it saves. Only a low-resolution bake (e.g. 1/8 resolution, interpolated) can win, and it is an approximation. The networks are piecewise linear (leaky ReLU), so a bake is near-exact away from the kinks.
- **Checks:** the Python evaluation uses the same baked texture; FLIP within δ.

## Step 2: changes that need retraining

### 2a. Two streak passes of 8 taps instead of three of 4
- **What:** base 8: steps of 1 and 8 px, still reaching every distance 0–63 exactly once, with 16 taps per pixel per ray instead of 12, but one intermediate fewer.
- **Expected:** with 1b, ≈ 5/9 of today's ray traffic.
- **Risk:** the profile becomes the product of two 8-way softmaxes instead of three 4-way ones, a different family of shapes. Retrain from `ray_bright`'s settings; FLIP within δ.

### 2b. The far passes at half resolution
- **What:** the pass that reaches 16–63 px works on a half-resolution copy and is upsampled at the end; the near passes stay at full resolution.
- **Expected:** the far pass costs about 1/4.
- **Risk:** high. Full-resolution rays were the largest single gain (−68.6δ, 2026-11-05), and the remaining error lives in the far rays' cross-section (1.2–2.4 px wide, ceiling 2026-11-19), which a 2 px texel can't hold. Try it only if 1a–2a don't reach tier A, and expect a FLIP cost.

### 2c. Rays only from bright pixels (sparse splatting)
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
