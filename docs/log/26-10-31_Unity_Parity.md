# Phase 1, step 5: Unity parity

Run the trained pyramid (step 4's weights) in Unity from the generated HLSL, and confirm it gives the same result as Python on the same frames. The two sides share the Slang source and the weight file; only the compiler path and the API differ: `slangc` → HLSL → FXC → D3D11 in Unity, slangpy → SPIR-V → Vulkan in Python.

## Steps
- [x] **Engine entry points:** `slang/engine_glare.slang` wraps `Texture2D` as `IImage` and `StructuredBuffer<float>` as `IParams`, with kernels Prefilter / BlurH / BlurV / Upsample. Target sizes come from `GetDimensions`, not uniforms
    - Fixed on review: `tex.load` → `tex.Load` (HLSL methods are capitalised), and `no_diff` only decorates a call, so the `.rgb` swizzle comes after it
- [x] **`export/build_hlsl.sh`:** `-target hlsl -profile cs_5_0 -no-mangle -line-directive-mode none`, one file per kernel (slangc 2026.12 emits one entry point per HLSL output), writes `slangc_version.txt`, fails if the output contains cooperative-vector or LinAlg intrinsics
    - `-no-mangle`: without it `_Source` became `U_Source_0`, and Unity binds by name
- [x] **Unity project** `engines/unity/NeuralGlareURP` (6000.6.7f1, URP 17.6.0, D3D11 first, Linear). Generated HLSL in `Assets/NeuralGlare/Generated/` so Unity recompiles the wrappers when it changes; one two-line `.compute` per kernel in `Assets/NeuralGlare/Shaders/`
- [x] **`_ParamOffset` binding.** First attempt failed in Unity: `Unknown parameter type (0) for globalParams_0`. slangc wraps every uniform in a struct inside a cbuffer and has no option to flatten it; Unity's reflection rejects structs. Fix: `param_offset()` emits the bare identifier through `__intrinsic_asm "(_ParamOffset)"` (the parentheses make it an expression, not a call), and `GlareUpsample.compute` declares `int _ParamOffset;` before the `#include`. Verified in Unity's compiled code: one `CB0[1]` on D3D11, a plain `int _ParamOffset` on OpenGL Core
- [x] **Weight file** `glare_params.bytes`: raw little-endian float32, flat `mlp.slang` layout, no header (410 floats = 5 levels × 82), written by `train_pyramid.py`
- [x] **Harness.** Python `experiments/02_slang_plumbing/unity_parity.py` (`export`, `compare`) and Unity `Assets/NeuralGlare/Editor/ParityHarness.cs` (menu NeuralGlare → Run Parity). Images cross as raw float32 RGBA, row 0 = texel y = 0, no header: no EXR codec, no texture importer, no y flip. Python reads the weights from the `.bytes` file, not the `.npy`, so the shared format is tested too
- [x] **Pass criterion fixed before the first run:** on `up[0]` of each `goegap_road_4k` view (center, mid, corner), every channel of every texel satisfies |U − P| ≤ 1e-4 · |P| + 1e-7 · max(P). Every mip and upsample level is reported as a diagnostic; only `up[0]` decides

Pitfall: Unity textures are bottom-up in UV space. An imported EXR reaches the shader upside down relative to torch / NumPy, and since the MLP's only input is uv, every texel would get its mirror position's weights. Raw files avoid the question; the renderer feature (Phase 3) has to handle it explicitly.

## Result (2026-10-31)

`unity_parity.py export` → Unity: NeuralGlare → Run Parity → `unity_parity.py compare`. Record: `outputs/02_slang_plumbing/unity_parity/unity_parity.json`.

| Frame | Max error, every mip and level | `up[0]` bound ratio | Criterion | |
|---|---|---|---|---|
| `goegap_road_4k_center` | 0 | 0 | ≤ 1 | ✅ |
| `goegap_road_4k_mid` | 0 | 0 | ≤ 1 | ✅ |
| `goegap_road_4k_corner` | 0 | 0 | ≤ 1 | ✅ |

**Pre-registered verdict: PASS, bit-exact.** Energy differences are 0 in every channel at every level.

Environment: Unity 6000.6.7f1, Direct3D 11.0 (feature level 11.1); slangpy 0.43.1 on Vulkan; slangc 2026.12; NVIDIA GeForce RTX 2070, driver 621.01.

**Sensitivity check (post hoc, corner view):** the comparison catches the bugs it was built for.

| Deliberate bug, applied to the Python side | `up[0]` bound ratio |
|---|---|
| none | 0 |
| every level uses level 0's network (`_ParamOffset` stuck at 0) | 6968 |
| input flipped vertically (the Unity texture-convention trap) | 238 |

Unity's `up[0]` is real data, not a copy: values from 0.036 to 1.76e5, and it differs from `mip0` by up to 2.1e5.

**Why exact, not ~1e-6 as expected:** both paths end in the same NVIDIA driver compiler (DXBC and SPIR-V are both intermediate languages), and neither front end contracted or approximated an operation the other didn't. That is a property of this GPU and driver, not a guarantee; other vendors may differ by a few ulps, which the tolerance allows, so the rule carries over unchanged to the Phase 3 GPUs.

## What it means
- **Phase 1 is complete:** one Slang source trains in Python and runs in Unity with identical results, and the weight file format works in both
- **The engine path adds no error of its own** on this hardware, so any Phase 2/3 difference between Unity and the reference is the model's, not the deployment's
- The parity harness is reusable: rerun it whenever `glare.slang`, `mlp.slang` or the weights change shape (e.g. Phase 2's taps per level)

## Next
- [ ] Phase 2: fix the HDRI-crop references (6×8 interpolated PSF grid, 10–41 % error between nodes) before training against them
- [ ] Phase 2: decide taps per level / per-channel weights, and the weight constraint (e.g. w0 + w1 = 1 per level)
- [ ] Phase 3: URP renderer feature on the live camera buffer (handle the y flip there)
