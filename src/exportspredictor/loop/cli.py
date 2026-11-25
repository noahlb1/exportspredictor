"""`python -m exportspredictor.loop <command>`: one command per experiment and the confirmation."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from exportspredictor import search
from exportspredictor.features import (
    EXCLUDED_PRODUCTS_SET,
    EXPORTER_COUNT_SET,
    FEATURE_SET_ENV,
    PRODUCT_DETAIL_FEATURES,
    SIBLING_DEMAND_SET,
)
from exportspredictor.loop.arms import (
    CONSTRAINTS_EXPERIMENT,
    EXCLUDED_PRODUCTS_BASE_SET,
    EXCLUDED_PRODUCTS_EXPERIMENT,
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
    TWO_PART_EXPERIMENT,
    WIDER_SCOPE_BASE_SET,
    WIDER_SCOPE_EXPERIMENT,
    WIDER_SCOPE_SET,
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
from exportspredictor.loop.data import load, loop_dir, real_data
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
        lp, _ = _loop_on_files()
        section = run_market_reach_cap(lp)["experiments"][MARKET_REACH_CAP_EXPERIMENT]
        (arm,) = section["arms"]
        print(json.dumps({
            "kept": section["kept"], "cap": arm["settings"]["market_reach_cap"], "win_threshold": section["win_threshold"],
            **{r["arm"]: r[SELECTION_MEASURE] for r in (section["base_arm"], arm)},
        }), flush=True)
    elif command == "sibling-demand":
        lp, served = _loop_on_files(SIBLING_DEMAND_SET, "the Sibling Demand files")
        if served not in (SIBLING_DEMAND_BASE_SET, SIBLING_DEMAND_SET):
            raise SystemExit(f"The served files are {served!r}, not the {SIBLING_DEMAND_BASE_SET!r} files this set extends")
        section = run_sibling_demand(lp)["experiments"][SIBLING_DEMAND_EXPERIMENT]
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
        lp, _ = _loop_on_files()
        results = run_hyperparameter_stage(lp, stage)
        _print_round_section(results["experiments"][hyperparameter_experiment(stage)])
    elif command == "model-family":
        family = " ".join(argv[1:])
        if family not in MODEL_FAMILY_ARMS:
            raise SystemExit(f"usage: python -m exportspredictor.loop model-family {'|'.join(MODEL_FAMILY_ARMS)}")
        lp, _ = _loop_on_files()
        _print_round_section(run_model_family(lp, family)["experiments"][model_family_experiment(family)])
    elif command == "wider-scope":
        lp, served = _loop_on_files(WIDER_SCOPE_SET, "the wider scope's files")
        _require_served(served, WIDER_SCOPE_BASE_SET, "widens")
        section = run_wider_scope(lp)["experiments"][WIDER_SCOPE_EXPERIMENT]
        _print_round_section(section, "training_rows")
    elif command == "external-memory":
        lp, _ = _loop_on_files()
        _print_round_section(run_external_memory(lp)["experiments"][EXTERNAL_MEMORY_EXPERIMENT], "training_rows")
    elif command == "excluded-products":
        lp, served = _loop_on_files(EXCLUDED_PRODUCTS_SET, "the files that hold them")
        _require_served(served, EXCLUDED_PRODUCTS_BASE_SET, "adds to")
        section = run_excluded_products(lp)["experiments"][EXCLUDED_PRODUCTS_EXPERIMENT]
        _print_round_section(section, "training_rows", "non_uk_rate")
    elif command == "confirm":
        section = confirm_best()
        summary = {n: section[n][SELECTION_MEASURE] for n in (WINNER, V2, "naive")}
        print(json.dumps({**summary, **{k: section[k] for k in ("difference", "win_threshold", "beats_v2")}}), flush=True)
    else:
        raise SystemExit(
            "usage: python -m exportspredictor.loop "
            "noise|naive|select|two-part|scope|product-detail|constraints|exporter-count|market-reach-cap|sibling-demand|"
            "hyperparameters <stage>|model-family <family>|wider-scope|external-memory|excluded-products|confirm"
        )


def _loop_on_files(wanted: str | None = None, files: str = "the served feature files") -> tuple[Loop, str]:
    """The real loop and the served feature set, refused unless the loop reads `wanted` (default: the served set)."""
    data, served = real_data(), kept_feature_set(load())
    wanted = served if wanted is None else wanted
    if data.feature_set != wanted:
        raise SystemExit(f"Run on {files}: set {FEATURE_SET_ENV}={wanted}")
    return Loop(data, v2_plan(), loop_dir()), served


def _require_served(served: str, base: str, extends: str) -> None:
    """Refuse a feature set built from `base` unless `base` is the served set."""
    if served != base:
        raise SystemExit(f"The served files are {served!r}, not the {base!r} files this set {extends}")


def _print_round_section(section: dict, *extra: str) -> None:
    """One line for an experiment of the final round (spec #155): the decision and each Arm's measure, best round,
    seconds and `extra` fields (from the row, else from its settings)."""
    rows = [section["base_arm"], *section["arms"]]
    keys = (SELECTION_MEASURE, "rounds", *extra, "seconds")
    print(json.dumps({
        "kept": section["kept"], "win_threshold": section["win_threshold"],
        "decided_by_stopping_rule": decided_by_stopping_rule(rows),
        **{r["arm"]: {k: r[k] if k in r else r["settings"][k] for k in keys} for r in rows},
    }), flush=True)


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

