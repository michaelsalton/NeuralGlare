# Unreal check (UE 5.8.4, centre kernel)

Pre-registered rule (`checks/unreal_threshold.json`): Unreal ≈ (b) if FLIP(U, b) ≤ τ = ½·FLIP(a, b) and every channel's energy is within 2 % of (b); Unreal ≠ (a) if FLIP(U, a) > FLIP(U, b). (c) is post hoc and does not change the verdict.

| Scene | FLIP U–a | FLIP U–b | τ | FLIP U–c (post hoc) | Energy U/b − 1, % | Energy U/c − 1, % | Verdict |
|---|---|---|---|---|---|---|---|
| `pointgrid` | 0.935 | 0.930 | 0.231 | 0.358 | -0.06 / +1.99 / -3.87 | -0.00 / -0.01 / -0.02 | fail |
| `ferndale_studio_12_4k_center` | 0.402 | 0.080 | 0.194 | 0.005 | -0.08 / +2.03 / -3.79 | +0.00 / +0.00 / +0.00 | fail |
