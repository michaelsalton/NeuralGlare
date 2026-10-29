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
   - A wave-optics reference generator: aperture plus a real-ray-traced wavefront from a published lens prescription, giving per-wavelength PSFs on a field grid and a spatially varying convolution.
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
PSF_λ = | FFT( A(u; H, λ) · exp(i · 2π · W(u; H, λ) / λ) ) |²
```

- **u** is the pupil coordinate: the image-space ray direction, scaled so |u| = 1 is the f-number cone. H is normalized field height.
- **A** is the aperture mask: the diaphragm blades, cut where each ray crosses the stop, times mechanical vignetting from the lens rims. Both come from the ray trace.
- **W** is the wavefront error from a real-ray trace (rayoptics) of the lens prescription, tabulated per field height and wavelength (`reference/lens_trace.py`). It contains all aberration orders plus axial and lateral colour. Distortion is removed: each PSF sits on its chief ray.
- A Seidel polynomial (`W = W040 ρ⁴ + W131 H ρ³ cosθ + W222 H² ρ² cos²θ + W220 H² ρ²`) is *not* used. A fast lens balances large third-order terms against higher orders, so third order alone overstates the aberrations several-fold (see decision log).
- Integrate over about 8–16 wavelengths to get RGB.

**Reference lens:** Nikon AI Nikkor 35mm f/2S, US 3,507,558 Example 1, scaled to f = 36 mm (PhotonsToPhotos transcription). All-spherical, 53° horizontal FOV on a 36 mm-wide sensor, in the 28–40 mm range of cinematic game cameras. Ultra-wide gameplay FOVs (~90° horizontal) are out of scope.

**Aberration magnitudes must come from a real lens design or published lens data, never values tuned for drama.**

---

## 4. Method (planned)

```
HDR frame ──► downsample chain (fixed, as URP bloom) ──► mip_0 … mip_N
                                                            │
         upsample pass k:  w     = MLP_k( screen position )
                           out_k = Σ_i w_i · tap_i      taps: mip_k (± offsets), up(out_{k+1})
                                                            │
                                         glare image ──► composite with HDR
```

- The downsample pyramid gives a frame-sized receptive field almost for free.
- **The MLPs predict filter weights, not colours.** HDR values pass only through the weighted sum, so the output is linear in scene light by construction, as real glare is. The weights form an interpretable PSF field (see decision log, 2026-10-25).
- **Screen position is the only MLP input,** and it's what lets the model represent spatial variation. Content-dependent inputs (e.g. luminance) are excluded because they would break linearity.
- **MLPs are tiny,** a few hundred multiply-adds per pixel per level. They use the **scalar path in plain HLSL**, so no cooperative vectors or tensor cores are needed in Unity.
- Network weights live in a `StructuredBuffer`.

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

### Phase 0: Gap study
Before any training code, prove the problem exists.

The claim: Real lenses blue light differently at the edge of the frame than at the center. A single bloom kernel applied everywhere, which is what every real-time method does, is therefore visibly wrong. The error grows toward the edges

1. **Test inputs.** `experiments/01_gap_study/make_inputs.py` writes them to `data/inputs/` with a `manifest.json`.
   - A synthetic 7×5 single-pixel point-light grid (intensity 1e4, 6% margin).
   - 4 Poly Haven HDRIs × 3 views = 12 perspective crops at 1920×1080 with a 53.13° horizontal FOV (the reference lens on a 36 mm-wide sensor; taken from `lens_params`). Each view is aimed so the HDRI's brightest region lands at a fixed field offset: `center` 0°, `mid` (15°, 6°) ≈ 16°, `corner` (25°, 14°) ≈ 28°, along the frame diagonal.
   - Each crop is exposure-normalized to a median luminance of 0.18, with no clamping. The manifest records where the brightest pixel actually landed (`light_px`). Use it, not the nominal offsets, as the field position.
   - `historic_cloister_passage` is an overcast sky with no point source, so it gives only broad, low-contrast glare. Consider replacing it with a sun HDRI.
2. **PSF set.** Implement `psf.py` from the ray-traced pupil tables (`lens_trace.py`) and the diaphragm, on an 8×6 field grid.
3. **Reference.** Implement the spatially varying convolution in `convolve.py`. Sanity check: each grid point should match its local PSF.
4. **Baselines.** Uniform FFT (center and best-fit PSF), plus pyramid bloom **with fitted parameters**.
   - **Unreal check:** load the center PSF as Unreal's Convolution Kernel on one test scene. Confirm its output matches our NumPy uniform-FFT baseline, and record its GPU cost. **Done 2026-10-29** (redefined 2026-10-27; plan and results in `docs/todo/26-10-27_Test_Unreal_Bloom.md`):
     - **Output:** the pre-registered check fails. Unreal 5.8 doesn't convolve with the kernel as given (clamp, pass-through, tint), and its FFT mixes up channels: green and blue are normalised by each other's kernel sums, and blue gets red's multiply constant. The post-hoc prediction `unreal_fft_bloom_as_run` reproduces Unreal's bloom bit for bit on 88–98 % of texels.
     - **Cost** (RTX 2070 at 1410 MHz): 1.00 / 2.53 / 5.48 ms at 1080p / 1440p / 4K, Gaussian bloom 0.17 ms at 1080p.
5. **Outputs.** A motivation figure (side-by-side panels plus corner zooms) and a gap table (FLIP / PSNR / energy, center vs edge).

**Decision** (criterion fixed 2026-10-17, before the default-config run; computed by `evaluate_gap.py`):
- The *best uniform kernel* is the FFT baseline (centre or best-fit PSF) with the lowest median HDR-FLIP over all point-grid windows.
- **Proceed** if, on the point grid with the default config, its median edge FLIP (H ≥ 2/3) is **at least 2×** its median centre FLIP (H < 1/3), and the HDRI crops agree in direction (edge/centre > 1). Numerical settings must pass the convergence check (`convergence.py`, every change < 1%) for the result to count.
- **Gap large and growing toward the edges:** proceed.
- **Gap small with realistic aberrations:** pivot. Either make aperture shape and chromatic effects (uniform-kernel quality at pyramid cost) the framing, or switch to camera-matched bloom learned from photographs.

### Phase 1: Slang plumbing
- Train a Slang MLP pyramid to reproduce plain Gaussian bloom. This is the sanity check.
- Compile to HLSL and confirm Unity output matches Python output on the same frame.

Goals
- Slang MLP Pyramid can learn on plain Gaussian bloom.
- The HLSL compiled from the Slang course gives the same result in Unity as it does in Python.

- Mainly a pipeline test
- MLP network outputs filter weights, no colors.
- Output is a weighted sum of the input light, doubling the input doubles the output

pyramid.slang
- Fixed URP downsample chain: prefilter plus the 9-tap H and 5-tap V Gaussian - no ML.
- Matches the mips from pyramid_bloom.py

mlp.slang
- a tiny scaler MLP that reads all its weights from one flat buffer with offsets

glare.slang
- Upsample pass that predicts the weights.
- Each level converges to URP's scatter weights under teacher forcing; trained end to end, only the per-mip coefficients c_k do (the per-level weights are not identifiable)

### Phase 2: Neural glare model
- Train against the spatially varying reference.
- Ablations: MLP size, which levels get MLPs, input features, and the screen-position encoding.

### Phase 3: Engine deployment and timings
- **Unity (required):** run in the URP renderer feature.
- **Unreal (required):** time the convolution-bloom baseline on the same scenes and GPUs. The Phase 0 check and the RTX 2070 timings are done (see Phase 0, step 4); the harness and timing protocol are ready for the other GPUs. Report Unreal's quality from its actual output, not from the designed (b) prediction.
- **Unreal deployment (stretch):** the NeuralGlare plugin from the same generated HLSL. Do this only if Unity is finished early; otherwise list it as future work.
- Timings with RenderDoc / Nsight at 1080p, 1440p and 4K on multiple GPUs, using the same GPUs and resolutions in both engines.

### Phase 4: Evaluation
- Full metric table, held-out scenes, flicker study.

---

## 8. Key references

- Karp, Gruszka, Trzciński. *Neural Bloom: A Deep Learning Approach to Real-Time Lighting.* arXiv 2509.05963, 2025.
- Nalbach et al. *Deep Shading: Convolutional Neural Networks for Screen Space Shading.* CGF 2017.
- Spencer, Shirley, Zimmerman, Greenberg. *Physically-Based Glare Effects for Digital Images.* SIGGRAPH 1995. (Citation to verify.)
- Kakimoto, Matsuoka, Nishita, Naemura, Harashima. *Glare Generation Based on Wave Optics.* CGF 24(2), 2005. doi:10.1111/j.1467-8659.2005.00842.x
- Ritschel et al. *Temporal Glare: Real-Time Dynamic Simulation of the Scattering in the Human Eye.* CGF 2009.
- Liu, Wu, Zheng. GPU real-time glare via FFT Fraunhofer / Fresnel diffraction. *Journal of Computer-Aided Design & Computer Graphics*, 2013. (Title to verify; check whether it is spatially varying.)
- Hullin et al. *Physically-Based Real-Time Lens Flare Rendering.* SIGGRAPH 2011.
- Lee, Eisemann. *Practical Real-Time Lens-Flare Rendering.* CGF 32(4), 2013.
- Jimenez. *Next Generation Post Processing in Call of Duty: Advanced Warfare.* SIGGRAPH 2014 Advances in Real-Time Rendering course. Energy-conserving pyramid bloom. (Citation to verify.)
- Flicker, Rigaut. *Anisoplanatic Deconvolution of Adaptive Optics Images.* JOSA A, 2005. Low-rank decomposition of a spatially varying PSF; candidate non-neural baseline. (Citation to verify.)
- Wu, He, Xue, Garg, Chen, Veeraraghavan, Barron. *How to Train Neural Networks for Flare Removal.* ICCV 2021. Wave-optics flare synthesis; template for the reference generator.
- Tseng et al. *Neural Nano-Optics for High-Quality Thin Lens Imaging.* Nature Communications 2021. Patch-wise spatially varying PSFs. (Citation to verify.)
- *Successive Optimization of Optics and Post-Processing with Differentiable Coherent PSF Operator and Field Information.* arXiv 2412.14603, 2024.
- DeepLens: differentiable lens modelling library (PyTorch). `PSFNetLens` is a neural, field- and depth-varying PSF surrogate. https://deeplens.readthedocs.io
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
| 2026-10-16 | Pyramid-bloom baseline (`baselines/pyramid_bloom.py`) reimplements URP 17.6.0 (Unity 6.6) Gaussian bloom in NumPy: non-HQ path, tint white, no dirt. Shader constants still to be checked against the 17.6 `Bloom.shader` | Matches the Unity deployment target, so fitted parameters can be typed into a Volume and checked against a Unity capture. HQ filtering is added once the 17.6 source has been read |
| 2026-10-16 | Pyramid bloom is fitted as one joint parameter set across the inputs, not per image | A single setting is what a developer would ship. A per-image "oracle" fit remains an optional upper bound |
| 2026-10-16 | Pyramid fit uses the 12 HDRI crops only; the point grid is reported separately as a diagnostic | With an unweighted mean over images, the point grid's loss (≈0.014 vs ≤0.0022 for every crop) made up about half the objective, so the fit was mostly tuned to a synthetic pattern |
| 2026-10-16 | Fit loss: mean squared difference of log(x + ε), ε = 1e-3, comparing the composite (image + bloom) to the reference | Losses in log space weight relative error equally from 1e-3 halo to 1e4 sources. ε sits about two orders below the crops' 0.18 median |
| 2026-10-16 | Reference lens: Nikon AI Nikkor 35mm f/2S (US 3,507,558 Ex. 1, via PhotonsToPhotos), 36 mm, operated at f/2.8; camera HFOV becomes 53.13° | All-spherical, documented, citable. Matches the 28–40 mm cinematic game camera and Unreal's CineCamera default. Ultra-wide (~90°) gameplay FOVs excluded: third-order theory and simple prescriptions are unreliable there |
| 2026-10-16 | Wavefront and vignetting from a real-ray trace (rayoptics 0.9.10) instead of Seidel coefficients | For this lens at f/2, the third-order terms are 15–57 waves but the real error is ~2–3 waves RMS (fifth-order balancing), and a per-field Seidel fit leaves up to 1.3 waves residual. Real rays also give mechanical vignetting from the element rims, and lateral/axial colour |
| 2026-10-16 | Prescription checked against the patent scan; aperture stop placed 1.9332 mm after r7 (PhotonsToPhotos had 2.0132) | The patent prints every radius, thickness, nd and vd (all match ×36) but not the stop position. The stop is placed so rayoptics reproduces the patent's own coma Seidel sum (II = −0.0572022). Distortion V then matches independently (0.22371 vs 0.2237184), and spherical I (stop-independent) matches to 5 digits |
| 2026-10-16 | Lens rim diameters stay PhotonsToPhotos estimates | The patent gives only the front element (≈1.3f = 46.8 mm vs 46.58 used). No vignetting measurement of this lens was found (LensTip's AF 35/2D is a different design). Limitation: mechanical vignetting is unvalidated |
| 2026-10-16 | 7 diaphragm blades, rotation 0° by convention | 7 blades per Ken Rockwell and Kamerastore (no Nikon spec sheet found). Real blade orientation depends on the lens and aperture setting, so it isn't a physical parameter |
| 2026-10-16 | Veiling-glare scatter switched off (energy_fraction = 0) | No citable glare spread function for this lens. Scatter is spatially invariant, so it isn't what the gap study tests, and an invented wide halo would bias the comparison with pyramid bloom, which fits wide halos well. Revisit with a measurement if the halo is needed |
| 2026-10-16 | Sensor RGB response: measured Nikon D700 spectral sensitivities (Jiang et al. 2013, Zenodo 3245883), downloaded to data/ with an MD5 check | Real, non-negative curves from a full-frame body that takes this lens. CC BY-NC-SA 4.0, so the data is not committed. CIE→Rec.709 was rejected for now because its negative lobes give negative PSF weights |
| 2026-10-17 | **Phase 0 result: PROCEED.** Default config, 48 wavelengths, exact point-grid reference: the best uniform kernel (FFT, exact centre PSF) has point-grid median HDR-FLIP 0.048 at the centre and 0.204 at the edge (4.25×, required ≥ 2×). The best-fit uniform kernel gives 2.59×; the HDRI crops agree in direction (4.62×). Fitted pyramid bloom stays near no-bloom everywhere | Convergence: rays 0.000%, table 0.006%, kernel 0.046%, wavelengths 48→96 0.41%, all < 1% (the first three were checked at 12 wavelengths). Details: `outputs/01_gap_study/default/` (report, eval), `default/checks/convergence*.json`. Caveats: centre bin has 5 point-grid windows; the pyramid fit ends at its parameter limits; the HDRI references carry grid-interpolation error (open issue above) |
| 2026-10-25 | Phase 0 Unreal check (centre PSF as Convolution Kernel: confirm the output matches the NumPy uniform-FFT baseline, record its GPU cost) deferred to Phase 3 | It doesn't affect the proceed/pivot decision, which uses only the NumPy baselines. It needs the same Unreal scene, capture and timing setup as the Phase 3 convolution-bloom timings, so building that setup once avoids doing the work twice. Phase 1 doesn't depend on it |
| 2026-10-17 | Spectral sampling: 48 wavelengths (420–680 nm) in the default config | Convergence check at the corner PSF: 12 → 24 samples changed it 4.5%, 24 → 48 changed it 1.3%. 48 → 96 is verified in `outputs/01_gap_study/default/checks/convergence_wavelengths.json` |
| 2026-10-17 | The point-grid reference is computed **exactly** (`exact_pointgrid.py`: each source × the PSF at its own pixel), and the centre-PSF baseline uses the exact on-axis PSF (`centre_psf` in the PSF npz) | The 6×8 node blend is not the PSF between nodes: 10% (centre and mid cells) to 41% (corner cell) relative L1 against the exact PSF. Every point-grid source sits between nodes, so the interpolated reference would inflate the measured gap. With sparse sources an exact reference is cheap. The centre pixel isn't a node either: the old "centre PSF" was a blend of 4 nodes at H = 0.16 |
| 2026-10-17 | Open issue: the HDRI-crop references still use the 6×8 interpolated grid (10–41% PSF error between nodes) | Fixing it needs a denser grid (cost ∝ nodes) and/or interpolation in rotated, field-aligned PSFs (the lens is rotationally symmetric; only the diaphragm is not). Until fixed, the crops serve only as the direction check in the decision criterion, and their magnitudes are not reported as results |
| 2026-10-17 | Proceed/pivot criterion: best uniform kernel's point-grid edge/centre median FLIP ≥ 2×, with the HDRI crops agreeing in direction; numerical settings must change the corner PSF by < 1% when doubled (`convergence.py`) | Fixed before the default-config run so the result can't steer the threshold. Set after the quick-config rehearsal (4.3× centre PSF, 2.1× best-fit), so it is informed by that run, and the best-fit kernel was borderline there. Chosen on the user's delegation; the user may revise it before reading the default results |
| 2026-10-17 | Gap evaluation (`experiments/01_gap_study/evaluate_gap.py`): metrics in a 2K × 2K window around each light (HDR-FLIP at 67 ppd from a full-frame map, log-MSE, log-PSNR, window energy), medians per H bin (centre < 1/3, mid, edge ≥ 2/3), edge/centre ratios. The point grid is the primary measurement and HDRI crops (at H_measured) are the check | Whole-frame losses mixed scene content with field position and diluted the local effect. The point grid holds the source fixed so only H changes. One FLIP map per frame gives every window the same exposure range |
| 2026-10-17 | Gap-study views are defined by field height along the diagonal, H ∈ {0, 0.5, 0.85}, replacing the 2026-10-15 angle offsets. Each HDRI's brightest 8×8 region is aimed there exactly (yaw/pitch solved by iteration), and the manifest records the measured `light_px` / `H_measured` (luminance-weighted centroid) | The aiming in the 2026-10-15 decision was never implemented (fixed absolute angles: goegap's sun was out of frame in its center view). With the 53° FOV, the old corner offset (25°, 14°) landed ~10 px from the top edge. H is the variable the study measures and does not depend on the FOV |
| 2026-10-16 | Pupil tables are sampled over the image-space ray direction u, not the entrance pupil; field heights are pinhole object angles atan(H·half_diag/f) | Off-axis, the lens has strong pupil aberration (the paraxial pupil centre misses the stop centre by 32% of its radius at the corner), and the Fraunhofer FFT needs direction cosines. Pinhole angles match make_inputs' projection. rayoptics' relative image-height fields scale the angle linearly, which put mid-field nodes up to 9% off |
| 2026-10-16 | Fit search: grid over threshold {0, 0.25, 0.5, 0.9}, downscale {half, quarter} and max_iterations 2–8; Nelder–Mead over scatter and log10(intensity) | Threshold and downscale change the mip images and are costly to vary; scatter, intensity and level count are cheap. log10 intensity gives scale-free steps |
| 2026-10-16 | Fit speed-up: precomputed per-mip basis (bloom = Σ level weight × basis) evaluated on a seeded (seed 0) 1/16 pixel subsample; the best parameters are re-scored on full frames | One 1080p NumPy bloom takes 1.3 s. The basis identity is exact (≤1e-6, covered by tests), and the reported losses come from the full-frame re-score, not the subsample |
| 2026-10-16 | Fit grid narrowed to threshold {0, 0.9} and downscale half only (supersedes the grid above) | In the first fit, Quarter was about 16× worse in every cell (best loss 0.028 vs 0.0016), and the loss was nearly flat in threshold (0.00164–0.00179 at 8 levels). Keeping 0 (physically motivated) and 0.9 (URP default) cuts the run from about 15 to about 4 minutes |
| 2026-10-16 | Uniform FFT baseline (`baselines/fft_bloom.py`) = (1 − η)·I ⊛ K + η·I ⊛ S, with the reference's exact scatter term; only the diffraction kernel K is uniform. Two variants: K = centre PSF (interpolated at the frame centre) and K = best-fit PSF | The only difference from the reference is then the spatial variation of the diffraction PSF, which is what the gap study isolates. Leaving out scatter would also remove the long tail and confound the comparison |
| 2026-10-16 | Best-fit uniform PSF is free-form: every K×K×3 kernel texel is a parameter, softmax over texels per channel (non-negative, sums to 1), initialised from the centre PSF, fitted with Adam in PyTorch on the GPU, using the same log loss (ε = 1e-3) and the same 12 HDRI crops as the pyramid fit. Baseline outputs are still computed with the NumPy `convolve_uniform` path | A free-form kernel is the strongest uniform baseline (an upper bound for any single kernel, including any texture given to Unreal). A blend of node PSFs or a parametric kernel could be beaten by a better kernel |
| 2026-10-16 | All FFT convolutions (reference in `convolve.py`, baselines, and the PyTorch fit) run in float64; results are still stored as float32 EXR. References regenerated | FFT round-off spreads over every pixel in proportion to the brightest content in the frame. In float32, a 4e5 sun put ≈0.04 of noise on every pixel, which was half or more of the measured centre-FFT loss on sun crops (noise floor up to 7.8e-6 vs 1.2e-5 measured) and produced negative values. Covered by `tests/test_convolve.py` |
| 2026-10-25 | Phase 1 sanity target: the NumPy URP pyramid bloom (`baselines/pyramid_bloom.py`) with threshold 0 | With threshold 0, URP bloom is linear and exactly representable by the pyramid: each upsample is `lerp(mip_k, up(out_{k+1}), s)`, with s = 0.05 + 0.9 · scatter. The correct answer is known, so a residual error indicates a pipeline bug, not a capacity limit |
| 2026-10-25 | The per-level MLPs predict **filter weights, not colours**: `w = MLP_k(uv)`, `out_k = Σ_i w_i · tap_i`. The MLP input is screen position only; HDR values go only through the weighted sum. Phase 1 uses 2 taps (`mip_k`, `up(out_{k+1})`), whose exact answer is w = (1 − s, s). Supersedes the "mip taps, luminance" MLP inputs in the §4 sketch | Glare is linear in scene light, and a weighted sum is linear by construction for any MLP output, so the network never has to learn linearity across 8 orders of magnitude of HDR, and never sees HDR values as inputs. The weights are an interpretable, plottable PSF field, the same structure as the spatially varying reference (cf. kernel-predicting networks, Bako et al. 2017). Content-dependent weights would break linearity, so they are excluded. Open for Phase 2: taps per level and per-channel weights (they set which kernel shapes are reachable: coma, spikes, colour fringing); a weight sign/normalisation constraint (energy conservation); the gather (receiver-side) vs scatter (source-side) error; and, because the weights depend only on uv, whether to bake them to a per-resolution weight texture for runtime (this affects how the timings are framed) |
| 2026-10-25 | Training toolchain: slangpy 0.43.1 on the **Vulkan** backend, called on torch tensors through slangpy's **Python torch fallback** (`spy.set_allow_torch_fallback(True)`). Torch owns the loss, the chaining between kernels and the optimiser (Adam); Slang owns the per-pixel kernels and their derivatives. Standalone `slangc` is **2026.12**, the exact Slang version slangpy 0.43.1 is built against (`SGL_SLANG_VERSION` in slangpy's `external/CMakeLists.txt`), installed at `D:\Development\Libraries\Slang`. Verified by `tests/test_slang_torch.py` (gradients match a hand derivation; Adam recovers known weights) | The CUDA backend fails at device creation on this machine (RTX 2070, driver 621.01; cause not found). The native bridge (`slangpy-torch` 0.7.0) is a source build against the installed torch and needs `--no-build-isolation` plus a C++ toolchain; the fallback is exact, only slower, which is acceptable at Phase 1 data sizes. Revisit both if training speed becomes a bottleneck. Matching `slangc` to slangpy's compiler keeps the engine HLSL and the trained kernels from the same compiler version |
| 2026-10-25 | `slang/pyramid.slang`: URP's downsample chain (bilinear prefilter, 9-tap BlurH, 5-tap BlurV) **without threshold or clamp**. Passes are generic over an `IImage` interface (`size`, `load`): slangpy tensors in training (`slang/train_pyramid.slang`), `Texture2D` in the engine wrappers. Bilinear filtering is done by hand from 4 loads, not with a hardware sampler. Parity with the NumPy baseline (`tests/test_pyramid.py`) is measured against the brightest source texel in each tap footprint, with a tolerance of 4 float32 ulps of the image size | One source for training and engines; generics specialise away, so the HLSL has no interfaces (checked with `slangc` 2026.12). Manual bilinear gives identical arithmetic on both sides (hardware samplers resolve blend weights to ~1/256 and differ by vendor) and is differentiable for the upsample in training. HDR rule: no clamp or threshold. NumPy computes coordinates in float64 and the shader in float32, so a per-pixel relative tolerance fails next to bright texels (3.4e-3) for reasons that are not bugs; the footprint-relative error is 1.9e-5 against a 6.1e-5 bound |
| 2026-10-25 | `slang/mlp.slang`: scalar MLP, generic over an `IParams` interface (`[Differentiable] float get(int)`). All of a network's parameters sit in one flat float array: per layer, weights `[out][in]` row-major, then biases `[out]`, with an offset per network so several share one buffer. This layout is the weight file format and the engines' `StructuredBuffer<float>`. Training reads it through a slangpy `DiffTensor` (gradients land in the torch tensor's `.grad`). Phase 1 network: uv (2) → 16 → leaky ReLU (0.01) → 2, no input encoding. Verified against an independent PyTorch MLP (`tests/test_mlp.py`: forward pass and gradients agree to ~1e-7) | Same one-source pattern as `IImage`; `slangc` emits plain HLSL with no cooperative-vector or LinAlg intrinsics. A flat layout maps 1:1 to a structured buffer and a binary file in both engines. `DiffTensor` does the gradient atomics that `neural_image` did by hand; forward + backward over 518k points takes ~66 ms through the torch fallback. No frequency encoding: the PSF field varies smoothly, and the Phase 1 answer is a constant. Encoding and width belong to the Phase 2 ablations |
| 2026-10-28 | Output layout: `outputs/<experiment>/<run>/<stage>/`, where the run is the lens config (`quick`, `default`) and the stage is `report/` (tables and figures, for reading), `fits/` (fitted baselines later scripts load), `eval/` (raw per-window records), `export/` (engine files) or `checks/` (convergence). Scripts take `--run` and derive the PSF, reference folder and output folder from it (`src/neuralglare/paths.py`), replacing separate `--psf`, `--refs`, `--fft`, `--pyramid` and `--out` arguments. Each experiment folder gets a hand-written `README.md` naming its canonical run | The quick smoke-test run sat at the top of `outputs/01_gap_study/` and the decision run in a subfolder, with fits, raw records, figures and engine exports mixed together. One run name per invocation makes it impossible to pair one config's PSF with another's references or output folder (`evaluate_gap.py` caught that only at runtime). `outputs/` is gitignored, so results that matter still go into this log and `docs/findings.md` |
| 2026-10-27 | Unreal check pulled forward out of Phase 3 (reverses the 2026-10-25 row) and redefined: Unreal's output with the centre PSF is compared with two NumPy predictions, (a) the kernel as is and (b) the kernel as Unreal 5.8 processes it (centre search, clamp to a ring maximum, pass-through split, renormalisation, tint; `baselines/unreal_fft_bloom.py`). Plan in `docs/todo/26-10-27_Test_Unreal_Bloom.md` | Reading the 5.8 bloom shaders showed Unreal does not convolve with the kernel as given, so "matches the NumPy uniform-FFT baseline" would fail for a known reason. Predicted on our kernel: tint 1.00 / 0.78 / 0.56, i.e. ~44 % less blue energy. Unreal ≈ (b) ≠ (a) both validates the capture pipeline and shows production convolution bloom drifts from the physical PSF even at the frame centre, an extra row for the evaluation |
| 2026-10-29 | Unreal harness: one C++ project, `engines/unreal/NeuralGlareUE` (UE 5.8, Blank, Maximum, no starter content), hosts both the convolution-bloom baseline and the NeuralGlare plugin; the mode will be switched by console variable. Harness settings in `Config/DefaultEngine.ini`: ray tracing and ray-tracing proxies off, Lumen GI and reflections off, local exposure neutral (1.0), `r.AntiAliasingMethod 0`, screen percentage 100, dynamic resolution off, `r.Bloom.ApplyLocalExposure 0`. Fab plugin disabled in the `.uproject`. Captures from a standalone launch (`-game -windowed -ResX=1920 -ResY=1080`) through RenderDoc; the input is injected by a Scene Color Before Bloom post-process material from RGBA32F, Nearest-filtered textures | One project keeps every engine setting identical between the baseline and the plugin (cross-engine fairness). Each disabled feature is a pass that would otherwise run, cost time or touch the bloom input. The editor viewport is not 1:1 and doubled the energy of some point sources; the standalone 1:1 capture reproduced all 35 points exactly. Fab: its 2026-10-28 auto-update added a module the installed engine's precompiled rules (`UE5Rules.dll`) don't know, which broke every C++ build |
| 2026-10-29 | Unreal's post-process scene colour is `PF_FloatR11G11B10` (`PostProcessing.cpp:459`; `r.SceneColorFormat` only sets the main scene colour). Kept as is; the NumPy predictions quantise their input with `quantize_r11g11b10` (6/6/5 mantissa bits, truncated, unsigned, max 65024). HDRI crop for the check: `ferndale_studio_12_4k_center` | It is what shipping projects run, and the claim is about production bloom. Forcing fp16 (`r.PostProcessing.PropagateAlpha`) would measure a configuration nobody ships. Confirmed bit for bit on the point-grid and Ferndale before-bloom captures; truncation, not round-to-nearest, which misses half of Ferndale's values. Goegap and cobblestone crops exceed the format's max |
| 2026-10-29 | Unreal check pass rule, fixed before any Unreal convolution output was seen (`experiments/01_gap_study/unreal_threshold.py`): both predictions are pure convolution of the quantised input (no scatter term: Unreal gets only K). Unreal ≈ (b) if mean HDR-FLIP(U, b) ≤ ½·FLIP(a, b) (0.23 point grid over 35 source windows, 0.19 Ferndale full frame) and every channel's energy is within 2 % of (b); Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b) | Pre-registration: the rule can't be tuned to the result. Signal FLIP(a, b) = 0.46 / 0.39 against an input-quantisation noise floor of 0.006 / 0.026, and energy differences of 0 / +28 / +78 % against ≤ 1.1 %, so the two predictions are clearly separable |
| 2026-10-29 | Unreal check result, recorded as pre-registered: **fail** on both scenes (`experiments/01_gap_study/unreal_check.py`, `outputs/01_gap_study/default/report/unreal_check.md`). Unreal ≠ (a) holds (point grid only barely), Unreal ≈ (b) does not: energy U/b − 1 = −0.1 / +2.0 / −3.8 % (R, G, B) on both scenes against the 2 % tolerance, and point-grid FLIP(U, b) = 0.93 against τ = 0.23 (Ferndale 0.08 against 0.19 passes) | The rule was fixed before the data, so the verdict stands as is; everything below was found afterwards and is labelled post hoc. The capture pipeline itself is validated independently: before-bloom colour bit-exact, apply constants equal to the emulation to 6–7 digits |
| 2026-10-29 | UE 5.8 FFT bloom has two channel mix-ups in `GPUFastFourierTransform.usf`, modelled in the post-hoc prediction (c), `unreal_fft_bloom_as_run`: (1) `GetKernelSum` (line 146) returns the spectral DC texels as (R, B), (G, A), the two-for-one layout (lines 569–578), but the convolution (line 640) uses them as (R, G), (B, A), so green is normalised by blue's kernel sum and blue by green's; (2) the inverse two-for-one scales `b + i·a` by `Scale.xw` (line 415), so blue gets red's `FFTMulitply` constant. (c) also stores the clamped kernel as fp16 and the bloom texture as R11G11B10. (c) reproduces the bloom texture bit for bit on 94–96 % (point grid) and 88–98 % (Ferndale) of texels, energy within 0.08 % | Both mix-ups are read off the shader source and confirmed by the capture (spectral DC texels R 0.0810, G 0.0928, B 0.0844); together they explain the whole energy failure (bloom gain G ×1.099, B ×0.873). For the evaluation, Unreal's row differs from the physical PSF by design (clamp, pass-through, tint) and by these mix-ups; report both. Baselines are compared as shipped, so the Unreal timings and quality use Unreal's actual output, not (b) |
| 2026-10-29 | Flaw in the pre-registered threshold, kept for the record and not repaired retroactively: on the point grid, HDR-FLIP over 127×127 source windows is dominated by tail pixels below what Unreal can store (fp16 clamped kernel, R11G11B10 bloom texture: 83 % of Unreal's window pixels are exactly 0 where (b) is 1e-4 to 1e-13). `unreal_threshold.py` modelled only the input quantisation, so its noise floor was too low; even (c) scores FLIP 0.36 there at negligible energy (L1 0.18 %). Future pre-registered engine checks must model output storage formats in the noise floor, or measure FLIP only where the reference is representable | HDR-FLIP picks its exposure range from the reference, so 0 against 1e-10 counts as a large error. Ferndale, a real image, is unaffected (FLIP(U, c) = 0.005) |
| 2026-10-29 | GPU timing protocol (both engines): graphics clock locked to 1410 MHz with `nvidia-smi -lgc 1410,1410` (RTX 2070 reference base clock, memory unlocked at 7000 MHz), vsync and frame cap off, editor closed, standalone game launched from the command line, ~10 s warm-up, median of 5 per-pass GPU profiles (Unreal: `ProfileGPU`, which reads RDG scopes). First result: Unreal 5.8 convolution bloom at 1920×1080 costs **1.001 ms** (range 1.000–1.005; FFT size 2048 × 2048; the fused vertical forward + multiply + inverse pass is 0.557 ms), steady state with `r.Bloom.CacheKernel 1`; 2.529 ms at 2560×1440 (FFT 4096 × 2048) and 5.482 ms at 3840×2160 (4096 × 4096), so cost follows the padded power-of-two FFT area, not the pixel count. Gaussian (Standard) bloom at 1080p: 0.168 ms including the scene downsample chain it needs. Re-preparing the kernel every frame (`r.Bloom.CacheKernel 0`) adds 1.097 ms at 1080p | Boost clocks follow temperature and power, so the same work takes different times between runs and between engines; a clock every run can hold keeps Unity and Unreal comparable, at the price of absolute numbers below boost. The FFT bloom has no GPU stat, only an RDG scope, so `stat gpu` can't isolate it. Five samples spread 0.5 %, so a longer Insights trace isn't needed for the headline number |
| 2026-10-29 | Phase 1 step 4 (end-to-end pyramid): `IImage.load` and `sample_bilinear` are `[Differentiable]` (bilinear weights `no_diff`: gradients flow through the loads, not the geometry); constant images return `no_diff` loads, and `DiffTensorImage` (`train_glare.slang`) backs the chained `low` input, so the loss on the final image reaches every level. New entry point `upsample_chain_pass(…, offset)` uses one flat parameter buffer, level k at `k · PARAM_COUNT`; `upsample_pass` (teacher forcing) is unchanged. Verified by `tests/test_glare_chain.py`: with URP's exact weights the Slang chain reproduces the NumPy upsample chain (rtol 1e-4), and the gradient of every level's low-tap bias matches a central difference (rel 1e-3) | Teacher forcing treated `low` as a constant (`no_diff`), which can't train a chain. One differentiable interface keeps a single `sample_bilinear` for training and engines (generics specialise away; the engine wrappers only mark `load` `[Differentiable]` and return the fetch as `no_diff`). Pitfall: slangpy 0.43.1 decides from the first call of an entry point whether it is differentiable, so a first call with a params tensor without `requires_grad` silently disables autograd for later calls; training scripts must make the first call with gradients |
| 2026-10-29 | Phase 1 step 4 result (`experiments/02_slang_plumbing/train_pyramid.py`, record in `docs/todo/26-10-29_Train_Pyramid.md`): **FAIL** by the pre-registered criteria. Test loss on `up[0]` 7.1e-7 (≤ 1e-6 passes), but per-level weight errors 0.11–0.40 (≤ 0.02 fails). Post hoc: each mip's effective coefficient in the output, c_k = w0_k · ∏_{j<k} w1_j, matches URP's (1 − s) s^k to 0.001–0.018 over the screen | End-to-end supervision of the final image determines only the n + 1 path coefficients, not the 2n weights, so the weights are not identifiable (4 free directions for 5 levels); the criterion assumed they were. Consequences for Phase 2: report and plot the learned glare as effective coefficients or effective kernels, not per-level weights; if per-level weights must carry meaning (energy per level), add a constraint that removes the free directions, e.g. w0 + w1 = 1 per level (URP's lerp form, which also builds in ∑ c_k = 1), to be decided together with the open weight sign/normalisation constraint |
