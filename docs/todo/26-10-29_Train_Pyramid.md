# Phase 1, step 4: train the glare pyramid end to end

Train every upsample level at once, each reading the *trained* output of the level below, with one loss on the final image, against URP bloom (threshold 0, no clamp). The exact answer is w = (1 − s, s) at every level (s = 0.68), as in step 3's teacher forcing.

## Steps
- [x] **Slang: gradients through `low`.** `IImage.load` and `sample_bilinear` are `[Differentiable]` (`pyramid.slang`; the bilinear weights are `no_diff floor(p)`: gradients flow through the loads, not the geometry). Constant images return `no_diff` loads; `DiffTensorImage` (`train_glare.slang`) backs the chained input. New entry point `upsample_chain_pass(…, low, params, offset)`; `upsample_pass` (step 3) unchanged and still reproduces step 3's results
    - Fixed on review: `upsample_pass`'s signature had declared `low` twice (as `Tensor` and `DiffTensor`, no comma) and didn't compile; and once `sample_bilinear` was differentiable, Slang rejected `floor(p)` (no derivative; Slang can't tell `p` depends only on the `no_diff` uv), hence `no_diff floor(p)`
- [x] **Downsample in Slang on torch tensors:** `prefilter_pass`, `blur_h_pass`, `blur_v_pass` in `train_glare.slang` (`train_pyramid.slang`'s `Tensor<float3, 2>` doesn't accept torch `(H, W, 3)` tensors), so the whole pyramid runs in Slang. Logic stays in `pyramid.slang`
- [x] **Tests before training** (`tests/test_glare_chain.py`):
    - with URP's exact weights (all weights and hidden biases 0, output bias (1 − s, s)), the Slang chain reproduces the NumPy upsample chain (rtol 1e-4)
    - one backward pass reaches every level, and each level's low-tap bias gradient matches a central difference (rel 1e-3)
    - the torch-layout Slang downsample matches the NumPy chain (rtol 2e-4)
- [x] **Training script** `experiments/02_slang_plumbing/train_pyramid.py`: one flat parameter buffer (level k at `k · PARAM_COUNT` = 82 floats), `init_params` per level, Adam lr 1e-2 with cosine decay, 2000 steps, one random training frame per step, relative loss on `out_0` only. Writes `outputs/02_slang_plumbing/train_pyramid.json` and `train_pyramid_params.npy`
- [x] **Pass criteria fixed before the first run:** test loss on `up[0]` ≤ 1e-6 (step 3's level losses 4.5e-8 to 6.1e-7) **and** every level's max |w − (1 − s, s)| ≤ 0.02 (step 3: 0.004 to 0.009)

Pitfall: slangpy 0.43.1 decides from the first call of an entry point whether it is differentiable. If that call's params tensor has no `requires_grad`, later calls silently get no autograd history. Train first; `chain()` asserts the history is there.

## Result (2026-10-29)

`uv run python experiments/02_slang_plumbing/train_pyramid.py` (155 s, reproducible run to run):

| | Value | Criterion | |
|---|---|---|---|
| Train loss on `up[0]` | 9.9e-7 | | |
| Test loss on `up[0]` (`goegap_road_4k`, unseen sun) | **7.1e-7** | ≤ 1e-6 | ✅ |
| Max weight error, levels 0–4 | 0.113 / 0.249 / 0.187 / 0.344 / **0.402** | ≤ 0.02 | ❌ |

**Pre-registered verdict: FAIL** (weights). The output is right; the individual weights are not.

**Why (found after the run):** mip k reaches `out_0` through a product of weights along its path,

```
c_k = w0_k · w1_0 · w1_1 · … · w1_{k-1}        c_n = w1_0 · … · w1_{n-1}   (smallest mip)
```

and the final image fixes only these n + 1 = 6 coefficients, while the network has 2n = 10 weights. That leaves 4 directions in which the weights trade off with no change to the output. For example, level 0 learned w1 = 0.596 instead of 0.68, and the levels below compensate. Only level 0's w0 (= c_0) is unique: it learned 0.3201 against 0.32. Teacher forcing (step 3) avoided this by supervising every level's output.

| Mip | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Learned c_k (screen mean) | 0.3201 | 0.2180 | 0.1485 | 0.1018 | 0.0689 | 0.1467 |
| URP (1 − s) s^k, s^5 | 0.3200 | 0.2176 | 0.1480 | 0.1006 | 0.0684 | 0.1454 |
| Max \|c_k − URP\| over the screen (post hoc) | 0.012 | 0.018 | 0.011 | 0.003 | 0.002 | 0.001 |

Learned weights per level, screen mean, and their spread over the 16 × 9 grid (URP: w0 = 0.32, w1 = 0.68 everywhere):

| Level | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| w0 (mip tap) | 0.320 | 0.366 | 0.470 | 0.530 | 0.446 |
| w1 (low tap) | 0.596 | 0.530 | 0.608 | 0.804 | 0.949 |
| Spread of w0 / w1 over the screen | 0.02 / 0.07 | 0.05 / 0.21 | 0.12 / 0.09 | 0.25 / 0.15 | 0.10 / 0.27 |

The weights vary over the screen although URP's are constant: the trade-off directions are free at every uv, so each screen position settles on its own combination.

Learning curve (logged every 250 steps): the per-level weight errors were already at their final values by step 500 (0.12 / 0.25 / 0.19 / 0.32 / 0.41), while the loss kept falling from 2.1e-5 (step 250) to ~1e-6 (step 1500). The network picked one of the equivalent solutions early and then refined it; it isn't drifting slowly towards URP's, so more steps won't fix the weights.

The weight criterion asked for something end-to-end supervision can't determine. It stays as the verdict; the coefficient error is the post-hoc diagnostic, reported in `train_pyramid.json` (`post_hoc_max_coefficient_error`).

## What it means for Phase 2
- **End to end works as plumbing:** gradients reach every level, the trained pyramid reproduces URP on an unseen scene, and the parameter buffer is in the engine format
- **Per-level weights are not interpretable on their own.** To plot or compare the learned "PSF field", use the effective coefficients c_k (or the effective kernel at a pixel), not w per level
- If per-level weights must be meaningful (e.g. to constrain energy per level), the network needs a constraint that removes the free directions: e.g. w0 + w1 = 1 per level (URP's lerp form, one free number per level: n weights for n + 1 coefficients, with ∑ c_k = 1 built in), or a regulariser. That's a Phase 2 design choice, alongside the open "weight sign / normalisation constraint" in the decision log

## Next
- [x] Phase 1, step 5: export HLSL (`export/build_hlsl.sh`), run the trained pyramid in Unity, and confirm Unity matches Python on the same frame. Done 2026-10-31 in an editor harness rather than a renderer feature (the renderer feature moves to Phase 3): **PASS, bit-exact**. Record in `26-10-31_Unity_Parity.md`
