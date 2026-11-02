# 01 Gap study outputs

**Canonical run: `default/`.** Decision: **PROCEED**. Best uniform kernel (FFT, centre PSF) has 4.25× edge/centre FLIP on the point grid (≥ 2× required); HDRI crops agree in direction (4.62×). Recorded in `docs/RESEARCH_PLAN.md` (decision log, 2026-10-17).

Start with `default/report/gap_table.md`.

| Run | What it is |
|---|---|
| `default/` | Default lens config (psf_default), exact point-grid reference. The decision rests on this one |
| `quick/` | Quick config (psf_quick), interpolated point-grid reference. Pipeline smoke test only, do not report |

Each run folder:

| Folder | For | Contents |
|---|---|---|
| `report/` | reading | gap table, error-vs-H plot, motivation figures |
| `fits/` | later scripts | FFT kernels and pyramid-bloom fit that `evaluate_gap.py` loads |
| `eval/` | re-analysis | every window record, summary, versions |
| `export/` | engines | centre PSF as an EXR for Unreal's Convolution Kernel |
| `checks/` | trust | convergence of the default config's numerical settings |
