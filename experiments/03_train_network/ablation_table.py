"""Phase 2, step 2: the ablation table. Collects every Step 2 change's results from its study's
eval JSON (train_points.py), and the ceiling analysis (ceiling.py), into one table, so the
paper's numbers are read from outputs/ and never typed by hand.

Every change reports what the ablation criterion requires (decision log 2026-11-03): validation
FLIP and its difference from the run it was compared with, in units of the noise floor δ (the
range of validation FLIP over the Step 1 baseline's seeds 0, 1, 2, recomputed here), test FLIP
per H bin, edge/centre, the share of the edge gap closed, parameters, and multiply-adds per
texel per level. The criterion column is recomputed from those numbers; the decision column is
the decision log's (the criterion's exception, for an artefact or a cost, can't be computed).

    uv run python experiments/03_train_network/ablation_table.py --run default

Writes outputs/03_train_network/<run>/ablations/ablation_table.md and .json. Add a row to
ABLATIONS for every new change.
"""

import argparse
import json
from pathlib import Path

from ceiling import LABELS as ORACLE_LABELS
from ceiling import STUDY as CEILING_STUDY
from train_points import EXPERIMENT

from neuralglare.paths import RUNS, run_dir

BASELINE = ("baseline", "_lerp_abs_fitted_lr0.0001")
SEEDS = ("", "_seed1", "_seed2")    # the baseline's seed runs that set δ
BINS = ("centre", "mid", "edge")

# One row per change, in the order they were run. against: the study it was compared with (the
# current best at the time). mlp: LevelMLP's (inputs, hidden, outputs), checked against the run's
# stored param_count. extra: per-texel cost the MLP's multiply-adds don't show.
ABLATIONS = [
    {"label": "Step 1 baseline: MLP(uv)", "study": "baseline",
     "tag": "_lerp_abs_fitted_lr0.0001", "against": None, "mlp": (2, 16, 1), "extra": "",
     "decision": "current best (decision log 2026-11-03)"},
    {"label": "Radial input: MLP(H)", "study": "radial_input",
     "tag": "_lerp_h_abs_fitted_lr0.0001", "against": "baseline", "mlp": (1, 16, 1),
     "extra": " + length()", "decision": "rejected (decision log 2026-11-04)"},
]


def load_eval(run, study, tag):
    return json.loads((Path(run_dir(EXPERIMENT, run, "eval", study)) /
                       f"train_points{tag}_eval.json").read_text())


def mlp_cost(shape, param_count):
    """(parameters, multiply-adds) of one level's MLP<In, Hidden, Out>, as mlp.slang."""
    i, h, o = shape
    assert i * h + h + h * o + o == param_count, f"MLP{shape} doesn't match the run's {param_count}"
    return param_count, i * h + h * o


def criterion(diff, delta):
    """The ablation criterion's reading of a seed-0 validation FLIP difference."""
    if diff < -delta:
        return "kept"
    if diff > delta:
        return "rejected"
    return "inconclusive: run seeds 1, 2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="default")
    args = ap.parse_args()

    seeds = [load_eval(args.run, BASELINE[0], BASELINE[1] + s)["results"]["trained"]["val"]["flip"]
             for s in SEEDS]
    delta = max(seeds) - min(seeds)
    base = load_eval(args.run, *BASELINE)
    start = base["results"]["trained"]["summary"]["edge"]["flip"]      # the edge gap's start
    target = base["phase0"]["fft_centre"]["edge"]["flip"]               # and its end

    evals = {a["study"]: load_eval(args.run, a["study"], a["tag"]) for a in ABLATIONS}
    rows = []
    for a in ABLATIONS:
        d = evals[a["study"]]
        trained = d["results"]["trained"]
        val, summary = trained["val"]["flip"], trained["summary"]
        params, macs = mlp_cost(a["mlp"], d["param_count"])
        row = {"label": a["label"], "study": a["study"], "tag": a["tag"], "against": a["against"],
               "val_flip": val, "test_flip": {b: summary[b]["flip"] for b in BINS},
               "edge_over_centre": summary["edge_over_centre_flip"],
               "gap_closed": (start - summary["edge"]["flip"]) / (start - target),
               "params": params * d["levels"], "macs_per_texel_level": macs,
               "extra_cost": a["extra"], "decision": a["decision"]}
        if a["against"]:
            diff = val - evals[a["against"]]["results"]["trained"]["val"]["flip"]
            row |= {"val_diff": diff, "val_diff_delta": diff / delta,
                    "criterion": criterion(diff, delta)}
        rows.append(row)

    lines = [("| Change | Study | Val FLIP | Δ vs compared | Test FLIP c / m / e | edge/centre | "
             "Edge gap closed | Params | MACs / texel / level | Criterion | Decision |"),
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        diff = (f"{r['val_diff']:+.5f} ({r['val_diff_delta']:+.1f}δ)" if r["against"] else "-")
        flips = " / ".join(f"{r['test_flip'][b]:.4f}" for b in BINS)
        lines.append(f"| {r['label']} | `{r['study']}` | {r['val_flip']:.5f} | {diff} | {flips} | "
                     f"{r['edge_over_centre']:.2f}x | {r['gap_closed']:+.0%} | {r['params']} | "
                     f"{r['macs_per_texel_level']}{r['extra_cost']} | {r.get('criterion', '-')} | "
                     f"{r['decision']} |")

    ceiling_path = Path(run_dir(EXPERIMENT, args.run, "eval", CEILING_STUDY)) / "ceiling_eval.json"
    ceiling = json.loads(ceiling_path.read_text()) if ceiling_path.exists() else None
    if ceiling:
        lines += ["", (f"Ceiling analysis (`{CEILING_STUDY}`, ceiling.py): the exact reference with one "
                  "feature removed, scored like the models. The ring-median oracles are the best "
                  "round kernels in the absolute log loss; the ring-mean ones aren't ceilings."), "",
                  "| Oracle | Test FLIP c / m / e | energy edge |", "|---|---|---|"]
        for name, res in ceiling["results"].items():
            s = res["summary"]
            lines.append(f"| {ORACLE_LABELS[name]} | "
                         + " / ".join(f"{s[b]['flip']:.4f}" for b in BINS)
                         + f" | {s['edge']['energy']:+.1%} |")

    out = Path(run_dir(EXPERIMENT, args.run, "ablations"))
    report = (f"# Phase 2, step 2: ablation table ({args.run})\n\n"
              f"Noise floor δ = {delta:.6f}: the range of validation FLIP over `{BASELINE[0]}` "
              f"seeds 0, 1, 2. Edge gap closed = ({start:.4f} − edge) / ({start:.4f} − "
              f"{target:.4f}): from the Step 1 baseline's edge FLIP to the best uniform kernel's. "
              f"Test FLIP: medians over the 35 point-grid windows per H bin.\n\n"
              + "\n".join(lines) + "\n")
    (out / "ablation_table.md").write_text(report, encoding="utf-8")
    (out / "ablation_table.json").write_text(json.dumps(
        {"delta": delta, "baseline_seed_val_flip": seeds, "gap": {"start": start, "target": target},
         "rows": rows, "ceiling": ceiling and {k: v["summary"] for k, v in ceiling["results"].items()}},
        indent=2))
    print(f"wrote {out / 'ablation_table.md'} and .json")


if __name__ == "__main__":
    main()
