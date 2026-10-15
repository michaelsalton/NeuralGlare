# NeuralGlare: Research Plan

The living plan for the project. Update it when decisions change (see the decision log at the end).

---

## 1. Motivation

| Method | Cost | Physically correct? | Spatially varying? |
|---|---|---|---|
| Pyramid bloom (URP / Unreal standard) | Very cheap | No: threshold plus Gaussian blur, not energy conserving | No |
| FFT convolution bloom (Unreal "Convolution" bloom) | Expensive, ~1–3 ms on low-end GPUs | Yes, for one kernel | **No**: one PSF applied everywhere |
| Offline spatially varying convolution | Seconds per frame | Yes | Yes |
| **NeuralGlare (goal)** | **≈ pyramid bloom** | **≈ reference** | **Yes** |

Real lens and eye point spread functions (PSFs) change across the image field. Off-axis, coma, astigmatism, field curvature and cat's-eye vignetting stretch and skew the starburst. No real-time method models this.

**Prior neural bloom work:** Karp et al., *Neural Bloom* (arXiv 2509.05963, 2025; preprint, not peer reviewed).
- Shallow CNN at 128×128, trained per scene, LDR output.
- Its receptive field is far too small for wide glare.
- We use it as a weak baseline only.

---

## 2. Contributions (planned)

1. **Neural glare pyramid.** A mip pyramid plus per-level tiny MLPs that reproduce spatially varying, chromatic, physically based glare in real time.
2. **Differentiable training pipeline.**
   - A wave-optics reference generator: aperture plus Seidel wavefront, giving per-wavelength PSFs on a field grid and a spatially varying convolution.
   - A differentiable Slang pyramid. One Slang source is used for training and deployment.
3. **Engine evaluation.**
   - Unity URP deployment, plus Unreal deployment as a stretch goal.
   - Quality and cost at 1080p, 1440p and 4K against tuned classical baselines, Unreal's production convolution bloom, and the prior neural method.
4. **(Optional) dataset and benchmark release.** HDR frames plus reference glare. No glare benchmark exists today.

**Target venues:**
- HPG 2027 (performance framing).
- EGSR 2027 (optics and rendering framing).
- Fallback: an HPG poster or JCGT.
- Deadlines are likely early to mid April 2027.

---

## 3. Background: what a PSF is

A point spread function is the image an optical system forms of a single point of light. Glare is the HDR frame convolved with the PSF:

```
output = HDR ⊛ PSF                         # spatially invariant (FFT bloom)
output = Σ_k PSF_k ⊛ (w_k · HDR)           # spatially varying (our reference)
```

PSF from Fourier optics (Fraunhofer), per wavelength λ:

```
PSF_λ = | FFT( A(x,y) · exp(i · 2π · W(x,y; H) / λ) ) |²
```

- **A** is the aperture mask: blades and dust. Off-axis it is intersected with a shifted circle to model vignetting.
- **W** is the Seidel wavefront, where H is normalized field height and ρ, θ are pupil coordinates:
  `W = W040 ρ⁴ + W131 H ρ³ cosθ + W222 H² ρ² cos²θ + W220 H² ρ² + W311 H³ ρ cosθ`
- Integrate over about 8–16 wavelengths to get RGB.

**Aberration magnitudes must come from a real lens design or published lens data, never values tuned for drama.**

---

## 4. Method (planned)

```
HDR frame ──► downsample chain (fixed, as URP bloom) ──► mip_0 … mip_N
                                                            │
         upsample pass k:  out_k = MLP_k( mip_k taps, up(out_{k+1}),
                                          screen position, luminance, … )
                                                            │
                                         glare image ──► composite with HDR
```

- The downsample pyramid gives a frame-sized receptive field almost for free.
- **MLPs are tiny,** a few hundred multiply-adds per pixel per level. They use the **scalar path in plain HLSL**, so no cooperative vectors or tensor cores are needed in Unity.
- **Screen-position input** is what lets the model represent spatial variation.
- Weights live in a `StructuredBuffer`.

---

## 5. Data

Split train and test **by scene**. Never train per scene.

| | Train | Test (held out) |
|---|---|---|
| Real scenes | Sun Temple, Emerald Square (ORCA), URP 3D Sample Garden + Oasis, Poly Haven HDRI crops | **Bistro Exterior (night)**, Bistro Interior, URP Cockpit |
| Synthetic | Randomized point-light grids, randomized bright sources | Fixed 7×5 light grid, sun / extreme dynamic range test, moving sub-pixel highlights (flicker) |

**Licenses:**
- Bistro: CC BY 4.0.
- Sun Temple and Emerald Square: CC BY-NC-SA. Keep any derived dataset non-commercial and share-alike.
- Poly Haven: CC0.

**Bistro texture packing:** occlusion in R, roughness in G, metalness in B. Repack for URP.

---

## 6. Evaluation

**Metrics**

| Axis | Metric |
|---|---|
| Quality vs reference | HDR-FLIP (primary), log-space PSNR; reported before and after tonemapping |
| Spatial breakdown | All quality metrics for **center third vs outer ring** |
| Physical plausibility | Energy error: total output light minus total input light |
| Cost | GPU ms at 1080p / 1440p / 4K, plus extra VRAM, on at least 2 GPUs (one low-end) |
| Temporal stability | Flicker: output frame-to-frame variation relative to the reference's |
| Generalization | All of the above on held-out scenes |

**Baselines.** Re-measure every one on our own hardware. Published numbers are not comparable.

1. **URP pyramid bloom.** Its parameters are **fitted** to the reference, not left at defaults. For scale, one public measurement gives about 0.16 ms on a GeForce MX450.
2. **Uniform FFT convolution bloom**, measured three ways:
   - (a) the NumPy reference implementation, for quality;
   - (b) **Unreal Engine's built-in Convolution bloom** with our PSF as the kernel, for production quality and cost;
   - (c) optionally, StellarWarp's Unity implementation, which reports about 1.2 ms at 1024² and about 2.9 ms at 2048×1024 (4-channel) on an MX450, with the kernel FFT excluded.

   Use both the center PSF and the best-fit PSF.
3. **Neural Bloom (Karp et al. 2025)**, reimplemented at full resolution.
4. **Offline spatially varying reference.** This is ground truth.

**Targets** (self-set goals, not published thresholds):
- Cost within about 2× of URP bloom, and clearly below FFT convolution, with the margin growing at 4K.
- HDR-FLIP clearly better than both baselines, especially in the outer ring.
- Energy error close to FFT convolution's.
- Flicker no worse than pyramid bloom.

---

## 7. Plan

### Phase 0: Gap study (do this first, ~1 week)
Before any training code, prove the problem exists.

1. **Test inputs.** `experiments/01_gap_study/make_inputs.py` writes them to `data/inputs/` with a `manifest.json`.
   - A synthetic 7×5 single-pixel point-light grid (intensity 1e4, 6% margin).
   - 4 Poly Haven HDRIs × 3 views = 12 perspective crops at 1920×1080 with a 60° horizontal FOV. Each view is aimed so the HDRI's brightest region lands at a fixed field offset: `center` 0°, `mid` (15°, 6°) ≈ 16°, `corner` (25°, 14°) ≈ 28°, along the frame diagonal.
   - Each crop is exposure-normalized to a median luminance of 0.18, with no clamping. The manifest records where the brightest pixel actually landed (`light_px`). Use it, not the nominal offsets, as the field position.
   - `historic_cloister_passage` is an overcast sky with no point source, so it gives only broad, low-contrast glare. Consider replacing it with a sun HDRI.
2. **PSF set.** Implement `psf.py` from Seidel coefficients and the aperture with vignetting, on an 8×6 field grid.
3. **Reference.** Implement the spatially varying convolution in `convolve.py`. Sanity check: each grid point should match its local PSF.
4. **Baselines.** Uniform FFT (center and best-fit PSF), plus pyramid bloom **with fitted parameters**.
   - **Unreal check:** load the center PSF as Unreal's Convolution Kernel on one test scene. Confirm its output matches our NumPy uniform-FFT baseline, and record its GPU cost.
5. **Outputs.** A motivation figure (side-by-side panels plus corner zooms) and a gap table (FLIP / PSNR / energy, center vs edge).

**Decision:**
- **Gap large and growing toward the edges:** proceed.
- **Gap small with realistic aberrations:** pivot. Either make aperture shape and chromatic effects (uniform-kernel quality at pyramid cost) the framing, or switch to camera-matched bloom learned from photographs.

### Phase 1: Slang plumbing (~2 weeks)
- Train a Slang MLP pyramid to reproduce plain Gaussian bloom. This is the sanity check.
- Compile to HLSL and confirm Unity output matches Python output on the same frame.

### Phase 2: Neural glare model (Dec–Jan)
- Train against the spatially varying reference.
- Ablations: MLP size, which levels get MLPs, input features, and the screen-position encoding.

### Phase 3: Engine deployment and timing (Feb)
- **Unity (required):** run in the URP renderer feature.
- **Unreal (required):** time the convolution-bloom baseline on the same scenes and GPUs.
- **Unreal deployment (stretch):** the NeuralGlare plugin from the same generated HLSL. Do this only if Unity is finished early; otherwise list it as future work.
- Timings with RenderDoc / Nsight at 1080p, 1440p and 4K on multiple GPUs, using the same GPUs and resolutions in both engines.

### Phase 4: Evaluation and writing (Mar → April deadline)
- Full metric table, held-out scenes, flicker study, figures, paper.

---

## 8. Key references

- Karp, Gruszka, Trzciński. *Neural Bloom: A Deep Learning Approach to Real-Time Lighting.* arXiv 2509.05963, 2025.
- Nalbach et al. *Deep Shading: Convolutional Neural Networks for Screen Space Shading.* CGF 2017.
- Ritschel et al. *Temporal Glare: Real-Time Dynamic Simulation of the Scattering in the Human Eye.* CGF 2009.
- Hullin et al. *Physically-Based Real-Time Lens Flare Rendering.* SIGGRAPH 2011.
- Quad-Based Fourier Transform for Efficient Diffraction Synthesis. EGSR 2018.
- Csoba & Kunkli. *Efficient Rendering of Ocular Wavefront Aberrations using Tiled PSF Splatting.* CGF 2021.
- Zeltner et al. *Real-Time Neural Appearance Models.* TOG 2024. This is the distillation pattern we borrow.
- Bangaru et al. *SLANG.D: Fast, Modular and Differentiable Shader Programming.* TOG 2023.
- SIGGRAPH 2025 / 2026 courses, *An Introduction to Neural Shading* (`shader-slang/neural-shading-s25`, `-s26`).
- Unreal Engine docs: Bloom / Convolution Bloom.
- StellarWarp, *High-Performance Convolution Bloom on Unity* (GitHub).

---

## 9. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-15 | Project named NeuralGlare | |
| 2026-10-15 | Unity URP primary engine; Unreal as baseline harness + stretch deployment | HDRP in maintenance mode; Unreal ships FFT convolution bloom |
| 2026-10-15 | Python code (reference, baselines, train, eval) lives under `src/neuralglare/` | Importable from experiments, tests and notebooks without path hacks |
| 2026-10-15 | Gap-study HDRI crops are aimed at each HDRI's brightest region (argmax of an 8×8 box-downsampled luminance), with views at fixed field offsets 0° / (15°, 6°) / (25°, 14°) | Fixed-angle views missed the light in 5 of 12 crops. Fixed field offsets give a controlled sweep of field angle, which is the variable the gap study measures |
| 2026-10-15 | Gap-study crops are exposure-normalized to a median luminance of 0.18, scaling only (no clamp or tonemap) | Puts the scenes on a comparable exposure while preserving every HDR ratio. The scale factor is logged per crop |