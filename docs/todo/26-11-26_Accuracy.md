# Accuracy improvement: the rays' cross-section

Started 2026-11-26, after the efficiency work (decision log 2026-11-26).

## Where we are

`ray_bright` (the current best): validation FLIP 0.040928, test FLIP 0.0317 / 0.0384 / 0.0659 (centre / mid / edge), edge/centre 2.08×. Criterion (2) fails on the **centre** bin: the FFT's centre kernel scores 0.0123 there, and the centre has no coma, so the gap isn't about spatial variation. The model can't draw the lens's ray shape.

**Error split** (scratch; the exact reference pasted into one region of every window, re-scored; regions as `ceiling.far_masks`):

| Region made exact | centre | mid | edge |
|---|---|---|---|
| model as is | 0.0317 | 0.0384 | 0.0659 |
| core (< 5.25 px) | −0.0007 | −0.0006 | −0.0007 |
| annulus (5.25–9 px) | −0.0007 | −0.0009 | −0.0010 |
| **far rays (±2 px, ≥ 9 px)** | **−0.0234** | **−0.0286** | **−0.0325** |
| far glow | −0.0040 | −0.0048 | −0.0201 |
| (glow at 32–127 px) | −0.0039 | −0.0044 | −0.0145 |
| model's rays, light per ray exact | −0.0005 | −0.0030 | −0.0032 |
| model's rays, light per ray × 3 segments exact | −0.0001 | −0.0029 | −0.0051 |

- The core is done (the core loss and taps worked).
- The rays' **brightness and falloff along the ray are right**. The cost is the shape **across** the ray.
- Across the ray (16–48 px): odd rays rms width 0.42 px (lens) vs 0.63 px (model), light within 0.7 px 93 % vs 75 %; even rays 0.67 vs 0.72 px. Peak : nearby glow 489 (lens) vs 3715 (model): the model's line is too wide and has nothing around it.
- Between the rays the model is often exactly 0 where the lens has 1e-10–1e-7 of the source. Since 1 mip there is no wide term.

## Step 0: ray cross-section oracle (`ceiling.py`)

What class of profile across the ray is enough, and is the centre bin a *line* problem or a *skirt* problem? Design and pre-registered questions in the decision log (2026-11-26).

- [x] Prototype on 3 sources (scratch). Found that the floor needs colour per channel and per 1 px of radius, and that widths must vary with r (power law) and carry an offset
- [x] Code in `ceiling.py`: `xs_*` oracles, `separable_rays` run in a process pool; smoke test passes (energy to 1e-14)
- [ ] Full run: `uv run python experiments/03_train_network/ceiling.py --run default` (≈ 2 h of CPU for the six variants, split over the pool, plus the existing oracles)
- [ ] Result row in the decision log: answers, predictions true / false, the fitted widths at 16 px
- [ ] Pick the first engine-side change from the answer

Prototype (3 sources, H = 0.00 / 0.55 / 0.88; indicative, not the 35-source medians):

| Oracle | H = 0.00 | H = 0.55 | H = 0.88 |
|---|---|---|---|
| `xs_exact` (floor) | 0.0023 | 0.0022 | 0.0025 |
| `xs_line` | 0.0145 | 0.0151 | 0.0469 |
| `xs_line_bg` | 0.0129 | 0.0139 | 0.0341 |
| `xs_line_skirt` | 0.0123 | 0.0135 | 0.0234 |
| `xs_line_cskirt` | 0.0117 | 0.0130 | 0.0234 |
| `xs_line_cskirt_bg` | 0.0123 | 0.0135 | 0.0201 |

## Step 1: candidate changes (after the oracle)

Ordered by what the prototype suggests; the oracle decides.

- **A sharper, per-channel ray line** (centre). The streak takes bilinear taps along diagonal directions, which smears the line sideways by an amount that depends on the ray's angle; the 3-tap cross filter only redistributes what's left. Options: taps aligned to texel rows or columns (interpolate along one axis only), or a cross filter that can sharpen. Per-channel cross weights for the colour across the ray.
- **A skirt** (edge). A second, wider copy of each ray (a Gaussian skirt is a blur of the ray; a Lorentzian is roughly a sum of blurs of increasing width), weighted by field position.
- **A faint wide glow** (edge, 32–127 px), field-aligned. Not a round mip-1 disc: the 2026-11-10 result showed the lens has none.

Accuracy rule as in Step 2 of the plan (δ_coma = 0.000218); each change also gets a timing against the efficiency budget (`26-11-23_Efficiency-Improvment.md`), since the rays are already 65 % of the frame.
