"""`python -m exportspredictor.loop <command>`: one command per experiment, the confirmation and the timing."""

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from exportspredictor import experiments, search
from exportspredictor.features import FEATURE_SET_ENV, PRODUCT_DETAIL_FEATURES
from exportspredictor.loop.arms import (
    CONSTRAINTS_EXPERIMENT,
    EXCLUDED_PRODUCTS_BASE_SET,
    EXCLUDED_PRODUCTS_EXPERIMENT,
    EXCLUDED_PRODUCTS_SET,
    EXPORTER_COUNT_EXPERIMENT,
    EXTERNAL_MEMORY_EXPERIMENT,
    HYPERPARAMETER_STAGES,
    MARKET_REACH_CAP_EXPERIMENT,
    MODEL_FAMILY_ARMS,
    NAIVE_EXPERIMENT,
    NOISE_FLOOR,
    SCOPE_EXPERIMENTS,
    SELECT_EXPERIMENT,
    SIBLING_DEMAND_BASE_SET,
    SIBLING_DEMAND_EXPERIMENT,
    SIBLING_DEMAND_SET,
    TWO_PART_EXPERIMENT,
    WIDER_SCOPE_BASE_SET,
    WIDER_SCOPE_EXPERIMENT,
    WIDER_SCOPE_SET,
    Arm,
    Settings,
    blend_arm,
    constraint_arms,
    decided_by_stopping_rule,
    hyperparameter_experiment,
    model_family_experiment,
    product_detail_experiment,
    select_arms,
    starting_point_arm,
    two_part_arm,
)
from exportspredictor.loop.confirmation import V2, WINNER, confirm
from exportspredictor.loop.data import RESULTS_FILE, load, loop_dir, real_data
from exportspredictor.loop.recheck import current_win_threshold
from exportspredictor.loop.runner import (
    Loop,
    run_excluded_products,
    run_exporter_count,
    run_external_memory,
    run_hyperparameter_stage,
    run_market_reach_cap,
    run_model_family,
    run_product_detail,
    run_sibling_demand,
    run_wider_scope,
)
from exportspredictor.loop.selection import SELECTION_MEASURE
from exportspredictor.loop.served import kept_feature_set, v2_plan

EXPORTER_COUNT_SET = "exporter_count"  # the rebuilt feature files the Exporter count is tried on (#126)


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
    elif command == "exporter-count":
        data = real_data()
        if data.feature_set != EXPORTER_COUNT_SET:
            raise SystemExit(f"Run on the rebuilt files: set {FEATURE_SET_ENV}={EXPORTER_COUNT_SET}")
        section = run_exporter_count(Loop(data, v2_plan(), loop_dir()))["experiments"][EXPORTER_COUNT_EXPERIMENT]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in rows}}), flush=True)
    elif command == "market-reach-cap":
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != served:
            raise SystemExit(f"Run on the served feature files: set {FEATURE_SET_ENV}={served}")
        section = run_market_reach_cap(Loop(data, v2_plan(), loop_dir()))["experiments"][MARKET_REACH_CAP_EXPERIMENT]
        (arm,) = section["arms"]
        print(json.dumps({
            "kept": section["kept"], "cap": arm["settings"]["market_reach_cap"], "win_threshold": section["win_threshold"],
            **{r["arm"]: r[SELECTION_MEASURE] for r in (section["base_arm"], arm)},
        }), flush=True)
    elif command == "sibling-demand":
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != SIBLING_DEMAND_SET:
            raise SystemExit(f"Run on the Sibling Demand files: set {FEATURE_SET_ENV}={SIBLING_DEMAND_SET}")
        if served not in (SIBLING_DEMAND_BASE_SET, SIBLING_DEMAND_SET):
            raise SystemExit(f"The served files are {served!r}, not the {SIBLING_DEMAND_BASE_SET!r} files this set extends")
        section = run_sibling_demand(Loop(data, v2_plan(), loop_dir()))["experiments"][SIBLING_DEMAND_EXPERIMENT]
        (arm,) = section["arms"]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"], "naive_weight": arm["settings"]["naive_weight"],
            **{r["arm"]: r[SELECTION_MEASURE] for r in (section["base_arm"], arm)},
        }), flush=True)
    elif command == "hyperparameters":
        stage = " ".join(argv[1:]).replace("-", " ")
        if stage not in HYPERPARAMETER_STAGES:
            stages = "|".join(s.replace(" ", "-") for s in HYPERPARAMETER_STAGES)
            raise SystemExit(f"usage: python -m exportspredictor.loop hyperparameters {stages}")
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != served:
            raise SystemExit(f"Run on the served feature files: set {FEATURE_SET_ENV}={served}")
        results = run_hyperparameter_stage(Loop(data, v2_plan(), loop_dir()), stage)
        section = results["experiments"][hyperparameter_experiment(stage)]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"],
            "decided_by_stopping_rule": decided_by_stopping_rule(rows),
            **{r["arm"]: {k: r[k] for k in (SELECTION_MEASURE, "rounds", "seconds")} for r in rows},
        }), flush=True)
    elif command == "model-family":
        family = " ".join(argv[1:])
        if family not in MODEL_FAMILY_ARMS:
            raise SystemExit(f"usage: python -m exportspredictor.loop model-family {'|'.join(MODEL_FAMILY_ARMS)}")
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != served:
            raise SystemExit(f"Run on the served feature files: set {FEATURE_SET_ENV}={served}")
        section = run_model_family(Loop(data, v2_plan(), loop_dir()), family)["experiments"][model_family_experiment(family)]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"],
            "decided_by_stopping_rule": decided_by_stopping_rule(rows),
            **{r["arm"]: {k: r[k] for k in (SELECTION_MEASURE, "rounds", "seconds")} for r in rows},
        }), flush=True)
    elif command == "wider-scope":
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != WIDER_SCOPE_SET:
            raise SystemExit(f"Run on the wider scope's files: set {FEATURE_SET_ENV}={WIDER_SCOPE_SET}")
        if served != WIDER_SCOPE_BASE_SET:
            raise SystemExit(f"The served files are {served!r}, not the {WIDER_SCOPE_BASE_SET!r} files this set widens")
        section = run_wider_scope(Loop(data, v2_plan(), loop_dir()))["experiments"][WIDER_SCOPE_EXPERIMENT]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"],
            "decided_by_stopping_rule": decided_by_stopping_rule(rows),
            **{r["arm"]: {k: r[k] for k in (SELECTION_MEASURE, "rounds", "training_rows", "seconds")} for r in rows},
        }), flush=True)
    elif command == "external-memory":
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != served:
            raise SystemExit(f"Run on the served feature files: set {FEATURE_SET_ENV}={served}")
        section = run_external_memory(Loop(data, v2_plan(), loop_dir()))["experiments"][EXTERNAL_MEMORY_EXPERIMENT]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"],
            "decided_by_stopping_rule": decided_by_stopping_rule(rows),
            **{r["arm"]: {k: r[k] for k in (SELECTION_MEASURE, "rounds", "training_rows", "seconds")} for r in rows},
        }), flush=True)
    elif command == "excluded-products":
        data, served = real_data(), kept_feature_set(load())
        if data.feature_set != EXCLUDED_PRODUCTS_SET:
            raise SystemExit(f"Run on the files that hold them: set {FEATURE_SET_ENV}={EXCLUDED_PRODUCTS_SET}")
        if served != EXCLUDED_PRODUCTS_BASE_SET:
            raise SystemExit(f"The served files are {served!r}, not the {EXCLUDED_PRODUCTS_BASE_SET!r} files this set adds to")
        section = run_excluded_products(Loop(data, v2_plan(), loop_dir()))["experiments"][EXCLUDED_PRODUCTS_EXPERIMENT]
        rows = [section["base_arm"], *section["arms"]]
        print(json.dumps({
            "kept": section["kept"], "win_threshold": section["win_threshold"],
            "decided_by_stopping_rule": decided_by_stopping_rule(rows),
            **{r["arm"]: {k: r[k] for k in (SELECTION_MEASURE, "rounds", "training_rows", "non_uk_rate", "seconds")}
               for r in ({**row, "non_uk_rate": row["settings"]["non_uk_rate"]} for row in rows)},
        }), flush=True)
    elif command == "confirm":
        section = confirm_best()
        summary = {n: section[n][SELECTION_MEASURE] for n in (WINNER, V2, "naive")}
        print(json.dumps({**summary, **{k: section[k] for k in ("difference", "win_threshold", "beats_v2")}}), flush=True)
    elif command == "time":
        print(json.dumps(time_one_arm()), flush=True)
    else:
        raise SystemExit(
            "usage: python -m exportspredictor.loop "
            "noise|naive|select|two-part|scope|product-detail|constraints|exporter-count|market-reach-cap|sibling-demand|"
            "hyperparameters <stage>|model-family <family>|wider-scope|external-memory|excluded-products|confirm|time"
        )


def confirm_best(root: Path | None = None) -> dict:
    """#76: the best arm so far beside v2 (the noise floor's first seed, the model alone) on all five folds."""
    results = load(root)
    v2 = Settings.from_dict(results[NOISE_FLOOR]["arms"][0]["settings"])
    winner = Settings.from_dict(results["best"]["settings"])
    plan = v2_plan(root)
    return confirm(
        real_data(root), replace(plan, fold=0), winner, v2,
        threshold=current_win_threshold(results), folder=loop_dir(root),
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
