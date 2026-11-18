"""`python -m exportspredictor.loop <command>`: one command per experiment, the confirmation and the timing."""

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from exportspredictor import experiments, search
from exportspredictor.features import PRODUCT_DETAIL_FEATURES
from exportspredictor.loop.arms import (
    CONSTRAINTS_EXPERIMENT,
    NAIVE_EXPERIMENT,
    NOISE_FLOOR,
    SCOPE_EXPERIMENTS,
    SELECT_EXPERIMENT,
    TWO_PART_EXPERIMENT,
    Arm,
    Settings,
    blend_arm,
    constraint_arms,
    product_detail_experiment,
    select_arms,
    starting_point_arm,
    two_part_arm,
)
from exportspredictor.loop.confirmation import V2, WINNER, confirm
from exportspredictor.loop.data import RESULTS_FILE, load, loop_dir, real_data
from exportspredictor.loop.runner import Loop, run_product_detail
from exportspredictor.loop.selection import SELECTION_MEASURE
from exportspredictor.loop.served import v2_plan


def main(argv: Sequence[str]) -> None:
    command = argv[0] if argv else ""
    if command == "noise":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        floor = lp.noise_floor()
        summary = {k: floor[k] for k in ("seeds", "selection_measures", "sd", "win_threshold")}
        print(json.dumps({**summary, "seconds": [row["seconds"] for row in floor["arms"]]}), flush=True)
    elif command == "naive":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(NAIVE_EXPERIMENT, [blend_arm(), starting_point_arm()])["experiments"][NAIVE_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "select":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(SELECT_EXPERIMENT, select_arms(search.load()))["experiments"][SELECT_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "two-part":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(TWO_PART_EXPERIMENT, [two_part_arm()])["experiments"][TWO_PART_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "scope":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        for name, arms in SCOPE_EXPERIMENTS:
            section = lp.run_experiment(name, arms)["experiments"][name]
            print(json.dumps({name: section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "product-detail":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        results = run_product_detail(lp)
        for feature in PRODUCT_DETAIL_FEATURES:
            section = results["experiments"][product_detail_experiment(feature)]
            print(json.dumps({feature: section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "constraints":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(CONSTRAINTS_EXPERIMENT, constraint_arms())["experiments"][CONSTRAINTS_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "confirm":
        section = confirm_best()
        summary = {n: section[n][SELECTION_MEASURE] for n in (WINNER, V2, "naive")}
        print(json.dumps({**summary, **{k: section[k] for k in ("difference", "win_threshold", "beats_v2")}}), flush=True)
    elif command == "time":
        print(json.dumps(time_one_arm()), flush=True)
    else:
        raise SystemExit(
            "usage: python -m exportspredictor.loop noise|naive|select|two-part|scope|product-detail|constraints|confirm|time"
        )


def confirm_best(root: Path | None = None) -> dict:
    """#76: the best arm so far beside v2 (the noise floor's first seed, the model alone) on all five folds."""
    results = load(root)
    v2 = Settings.from_dict(results[NOISE_FLOOR]["arms"][0]["settings"])
    winner = Settings.from_dict(results["best"]["settings"])
    plan = v2_plan(root)
    return confirm(
        real_data(root), replace(plan, fold=0), winner, v2,
        threshold=results[NOISE_FLOOR]["win_threshold"], folder=loop_dir(root),
    )


def time_one_arm(root: Path | None = None) -> dict:
    """Train the start configuration once more, in memory (the cache is not read), and save its timing.

    The timing is saved in the loop's results under `timing`; nothing else there changes.
    """
    started = time.perf_counter()
    lp = Loop(real_data(root), v2_plan(root))
    row = lp.run_arm(Arm("timing"), lp.plan.start)
    timing = {
        "setup_seconds": lp.results["setup_seconds"],
        **{k: row[k] for k in ("seconds", "matrix_seconds", "train_seconds", "rounds", "trained_rounds", "training_rows", SELECTION_MEASURE)},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    path = loop_dir(root) / RESULTS_FILE
    results = json.loads(path.read_text(encoding="utf-8"))
    results["timing"] = timing
    path.write_text(json.dumps(results, indent=2, default=experiments.json_default), encoding="utf-8")
    return timing
