"""The hyperparameter search (ticket #54): Optuna TPE with a Median pruner, then confirmation on every Product.

v1 tried five hand-picked configurations. v2 searches the space below in two stages, on
the validation year (2022) only, with one fold of UK pairs held out (ADR 0001):

1. **TPE on the Sector subset** (`training.SECTOR_SUBSET`): train on 2012–2021 at the
   final rates (zeros at `features.ZERO_SAMPLING_RATE`, other Exporters' rows at the
   largest rate the VRAM budget allows), validate on 2022. v1's winning configuration is
   enqueued as trial 0. The Median pruner stops a trial whose validation loss, round by
   round, is worse than the median of earlier trials at the same round (Optuna's XGBoost
   pruning callback). The stage stops when its time budget is spent.
2. **Confirmation on every Product**: the best `top` configurations of stage 1 are
   retrained on all Products; the lowest validation deviance wins.

The objective is the 2022 held-out weighted Tweedie deviance at variance power 1.5,
whatever a candidate's own variance power, so that candidates are comparable. The
within-Product Spearman of the held-out UK pairs is logged per trial as a guardrail: the
winner should not rank UK Markets worse than v1's configuration by more than
`SPEARMAN_GUARDRAIL`. Trials that hit the round cap are logged too.

The study is stored in SQLite under the data directory, and every finished step is saved
to `results.json` beside it, so killing the run and starting it again resumes it: the
interrupted trial is marked failed and retried with the same configuration, finished
trials are kept, and only the rest of the time budget is spent.

    uv run python -m exportspredictor.search   # GPU, about 3 h; resumable

**The test year (2023) is never read**: both stages use `experiments.Data`, which refuses it.
"""

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from exportspredictor import baci, experiments
from exportspredictor.experiments import V1_CONFIG, Data
from exportspredictor.features import FEATURES, ZERO_SAMPLING_RATE, features_dir
from exportspredictor.folds import N_FOLDS
from exportspredictor.paths import data_dir
from exportspredictor.training import (
    SECTOR_SUBSET,
    SELECTION_METRIC,
    RowCounts,
    count_rows,
    fit_non_uk_rate,
    read_chunk,
    row_budget,
)

# The searched space (spec #39, decision #36). eta is fixed: the round count absorbs it.
SPACE = {
    "max_depth": (4, 12),
    "min_child_weight": (1.0, 1000.0),  # log scale
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "reg_lambda": (0.1, 100.0),  # log scale
    "tweedie_variance_power": (1.1, 1.9),
}
LOG_SCALE = ("min_child_weight", "reg_lambda")
ETA = 0.1
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 50
# v1's winning configuration as a point of the space; v1 left reg_lambda at XGBoost's default of 1.
V1_TRIAL = {**{name: V1_CONFIG[name] for name in SPACE if name in V1_CONFIG}, "reg_lambda": 1.0}
# The pruner reads the validation loss XGBoost logs each round under this name.
PRUNING_METRIC = f"valid-{SELECTION_METRIC}"
# A winner whose UK Spearman is lower than v1's configuration's by more than this is flagged.
SPEARMAN_GUARDRAIL = 0.01
STUDY_NAME = "v2_search"
STUDY_FILE = "study.sqlite3"
RESULTS_FILE = "results.json"


@dataclass(frozen=True)
class SearchPlan:
    """How the search runs: its budget, the rates, the held-out fold and the device."""

    timeout_seconds: float | None  # stage 1's budget, summed over runs when resumed
    n_trials: int | None  # stage 1's trial count (with the timeout, whichever comes first)
    top: int = 3  # configurations confirmed on every Product
    max_rounds: int = MAX_ROUNDS
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    eta: float = ETA
    sectors: tuple[str, ...] | None = SECTOR_SUBSET  # stage 1's Sectors; None: every Sector
    zero_rate: float | None = ZERO_SAMPLING_RATE  # None: the rate the rows were built at
    non_uk_rate: float | None = None  # None: the largest rate the VRAM budget allows (`final_non_uk_rate`)
    pruner_startup_trials: int = 5  # trials never pruned, so the median has a basis
    pruner_warmup_rounds: int = 100  # rounds before a trial can be pruned
    # The learning-curve point re-checked with the tuned configuration: (non-UK rate, zero
    # rate); a zero rate of None is the rate the rows were built at.
    recheck_rates: tuple[float, float | None] = (0.65, 0.05)
    threads: int | None = None  # XGBoost threads for the search's models; None: every core
    space: Mapping[str, tuple] | None = None  # None: `SPACE`
    fold: int = 0
    n_folds: int = N_FOLDS
    device: str = "cuda"
    seed: int = 0
    uk: str = baci.UK


REAL_SEARCH = SearchPlan(timeout_seconds=2 * 3600, n_trials=None)
# The fixture's tiny search: a handful of trials on a few hundred rows, so shallow trees
# with small leaves (v1's trial 0 lies outside this space, and is still tried first).
FIXTURE_SPACE = {**SPACE, "max_depth": (2, 3), "min_child_weight": (1.0, 10.0)}
FIXTURE_SEARCH = SearchPlan(
    timeout_seconds=None,
    n_trials=4,
    top=2,
    max_rounds=100,
    early_stopping_rounds=10,
    eta=0.2,
    space=FIXTURE_SPACE,
    sectors=None,
    zero_rate=None,
    pruner_startup_trials=2,
    pruner_warmup_rounds=5,
    recheck_rates=(1.0, None),
    threads=1,  # a few hundred rows: more threads only add overhead
    n_folds=2,
    device="cpu",
)


def suggest(trial, space: Mapping[str, tuple] | None = None) -> dict:
    """One configuration of `space` (default `SPACE`), as the Optuna `trial` samples it."""
    params = {}
    for name, (low, high) in (space or SPACE).items():
        if isinstance(low, int):
            params[name] = trial.suggest_int(name, low, high)
        else:
            params[name] = trial.suggest_float(name, low, high, log=name in LOG_SCALE)
    return params


def xgboost_params(params: Mapping, plan: SearchPlan) -> dict:
    """A searched configuration as XGBoost parameters (eta fixed by the plan)."""
    return {**params, "eta": plan.eta} | ({"nthread": plan.threads} if plan.threads else {})


# --- The run ----------------------------------------------------------------------------------


def run(subset: Data, everything: Data, plan: SearchPlan, folder: Path | None = None) -> dict:
    """Both stages and the learning-curve re-check; resumed from `folder` if given (else in memory).

    `subset` holds the rows stage 1 trains on (cut further to `plan.sectors`), `everything`
    the rows of every Product for stage 2. Returns the results `save` writes, including the
    trial table, the confirmation models, the winner and its Spearman against v1's
    configuration.
    """
    for data in (subset, everything):
        if data.validation_year != subset.validation_year:
            raise ValueError("Both stages must validate on the same year")
    results = experiments.read_results(folder) or {}
    non_uk_rate = plan.non_uk_rate if plan.non_uk_rate is not None else final_non_uk_rate(everything)
    results |= {"plan": experiments.plan_dict(plan), "non_uk_rate": non_uk_rate, "validation_year": subset.validation_year}

    started = time.perf_counter()
    study = _study(plan, folder)
    _optimise(study, subset, plan, non_uk_rate)
    results["trials"] = trial_table(study)
    results["capped_trials"] = [t["number"] for t in results["trials"] if t["hit_round_cap"]]
    results["stage1_seconds"] = round(sum(t["seconds"] or 0 for t in results["trials"]), 3)
    _save_results(results, folder)

    results["confirmation"] = _confirm(everything, plan, non_uk_rate, results, folder)
    results |= _winner(results)
    _save_results(results, folder)

    if results.get("learning_curve_recheck", {}).get("trial") != results["winner"]["trial"]:
        results["learning_curve_recheck"] = {
            "trial": results["winner"]["trial"], **_recheck(subset, plan, results["winner"]["params"])
        }
        _save_results(results, folder)
    results["seconds_this_run"] = round(time.perf_counter() - started, 1)
    _save_results(results, folder)
    return json.loads(_dumps(results))  # as saved: lists, not tuples


def scaled_rounds(rounds: int, *, training_rows: float, tuning_rows: float) -> int:
    """The early-stopped round count scaled by the ratio of training rows (a heuristic).

    The search stops early on models trained on the tuning years; the fold models train on
    more years (2012–2023 against 2012–2021, about 1.2 times the rows), so they get
    proportionally more rounds at the same learning rate. Nothing validates the scaled count.
    """
    return max(1, round(rounds * training_rows / tuning_rows))


def seconds(results: Mapping) -> float:
    """Time spent by both stages of the search, over every run that resumed it."""
    return round(results["stage1_seconds"] + sum(row["seconds"] for row in results["confirmation"]), 3)


def final_non_uk_rate(data: Data, n_features: int = len(FEATURES)) -> float:
    """The largest non-UK rate at which the training years' rows and the whole validation year fit the VRAM budget.

    The validation year is never sampled, so it is counted with the rows always kept.
    """
    training = count_rows([data.training[y] for y in sorted(data.training)], uk=baci.UK)
    validation = read_chunk_rows(data.validation)
    return fit_non_uk_rate(RowCounts(uk=training.uk + validation, non_uk=training.non_uk), row_budget(n_features))


def read_chunk_rows(chunk) -> int:
    if isinstance(chunk, (str, Path)):
        return pl.scan_parquet(chunk).select(pl.len()).collect().item()
    return read_chunk(chunk).height


def _study(plan: SearchPlan, folder: Path | None):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(folder / STUDY_FILE).as_posix()}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=plan.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=plan.pruner_startup_trials, n_warmup_steps=plan.pruner_warmup_rounds
        ),
        load_if_exists=True,
    )
    _recover_interrupted(study)
    if not study.trials:
        study.enqueue_trial(V1_TRIAL, user_attrs={"v1_config": True})
    return study


def _recover_interrupted(study) -> None:
    """Mark trials a killed run left running as failed, and enqueue their configurations again."""
    from optuna.trial import TrialState

    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
        study.enqueue_trial(
            trial.params, user_attrs={"v1_config": bool(trial.user_attrs.get("v1_config")), "retry_of": trial.number}
        )


def _optimise(study, data: Data, plan: SearchPlan, non_uk_rate: float) -> None:
    from optuna.trial import TrialState
    from optuna_integration import XGBoostPruningCallback

    done = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))
    n_trials = None if plan.n_trials is None else plan.n_trials - len(done)
    timeout = None
    if plan.timeout_seconds is not None:
        timeout = plan.timeout_seconds - sum(t.duration.total_seconds() for t in study.trials if t.duration)
    if (n_trials is not None and n_trials <= 0) or (timeout is not None and timeout <= 0):
        return
    fitter = experiments._Fitter(data, _fitter_plan(plan, data, plan.sectors))
    zero_rate = plan.zero_rate or data.built_zero_rate
    train, valid = fitter.matrices(FEATURES, non_uk_rate=non_uk_rate, zero_rate=zero_rate)
    experiments.log_progress(f"stage 1: {train.num_row():,} training rows; {len(done)} trials done")

    def objective(trial) -> float:
        trial.set_user_attr("v1_config", bool(trial.user_attrs.get("v1_config")))
        row = fitter.measure(
            train, valid, FEATURES, xgboost_params(suggest(trial, plan.space), plan),
            max_rounds=plan.max_rounds, early_stopping_rounds=plan.early_stopping_rounds,
            callbacks=[XGBoostPruningCallback(trial, PRUNING_METRIC)],
        )
        for key in ("rounds", "hit_round_cap", "training_rows", "validation_rows", "uk_pairs", "uk_spearman"):
            trial.set_user_attr(key, row[key])
        experiments.log_progress(f"trial {trial.number}: {trial.params} -> {row}")
        return row["validation_deviance"]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)


def trial_table(study) -> list[dict]:
    """Every trial of the study: its configuration, state, deviance, rounds and UK Spearman."""
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        intermediate = trial.intermediate_values
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "v1_config": bool(attrs.get("v1_config")),
                "retry_of": attrs.get("retry_of"),
                "params": dict(trial.params),
                "validation_deviance": trial.value,
                "rounds": attrs.get("rounds", (max(intermediate) + 1) if intermediate else None),
                "hit_round_cap": bool(attrs.get("hit_round_cap", False)),
                "uk_spearman": attrs.get("uk_spearman"),
                "training_rows": attrs.get("training_rows"),
                "seconds": round(trial.duration.total_seconds(), 3) if trial.duration else None,
            }
        )
    return rows


def _confirm(everything: Data, plan: SearchPlan, non_uk_rate: float, results: dict, folder: Path | None) -> list[dict]:
    """Retrain the best `plan.top` stage-1 configurations on every Product; keeps rows already saved."""
    complete = [t for t in results["trials"] if t["state"] == "COMPLETE"]
    top = sorted(complete, key=lambda t: t["validation_deviance"])[: plan.top]
    done = {row["trial"]: row for row in results.get("confirmation", [])}
    todo = [t for t in top if t["number"] not in done]
    if todo:
        fitter = experiments._Fitter(everything, _fitter_plan(plan, everything, None))
        zero_rate = plan.zero_rate or everything.built_zero_rate
        train, valid = fitter.matrices(FEATURES, non_uk_rate=non_uk_rate, zero_rate=zero_rate)
        experiments.log_progress(f"stage 2: {train.num_row():,} training rows")
        for trial in todo:
            row = fitter.measure(
                train, valid, FEATURES, xgboost_params(trial["params"], plan),
                max_rounds=plan.max_rounds, early_stopping_rounds=plan.early_stopping_rounds,
            )
            done[trial["number"]] = {"trial": trial["number"], "params": trial["params"], **row}
            experiments.log_progress(f"confirmation of trial {trial['number']}: {row}")
            results["confirmation"] = [done[t["number"]] for t in top if t["number"] in done]
            _save_results(results, folder)
        del train, valid
    return [done[t["number"]] for t in top if t["number"] in done]


def _winner(results: dict) -> dict:
    """The confirmed configuration with the lowest deviance, and its Spearman against v1's configuration (stage 1)."""
    best = min(results["confirmation"], key=lambda row: row["validation_deviance"])
    trials = {t["number"]: t for t in results["trials"]}
    v1 = next(t for t in results["trials"] if t["v1_config"] and t["state"] == "COMPLETE")
    change = experiments._difference(trials[best["trial"]]["uk_spearman"], v1["uk_spearman"])
    plan = results["plan"]
    return {
        "winner": {
            "trial": best["trial"],
            "params": {**best["params"], "eta": plan["eta"]},
            "rounds": best["rounds"],
            "hit_round_cap": best["hit_round_cap"],
            "validation_deviance": best["validation_deviance"],
            "uk_spearman": best["uk_spearman"],
            "training_rows": best["training_rows"],
        },
        "v1": {
            "trial": v1["number"],
            "validation_deviance": v1["validation_deviance"],
            "uk_spearman": v1["uk_spearman"],
            "rounds": v1["rounds"],
        },
        # Both on stage 1 (the Sector subset), where v1's configuration was trained.
        "spearman_vs_v1": change,
        "spearman_guardrail": SPEARMAN_GUARDRAIL,
        "loses_to_v1": change is not None and change < -SPEARMAN_GUARDRAIL,
    }


def _recheck(subset: Data, plan: SearchPlan, params: Mapping) -> dict:
    """One learning-curve point of ticket #53 again, with the tuned configuration instead of v1's."""
    non_uk_rate, zero_rate = plan.recheck_rates
    zero_rate = zero_rate or subset.built_zero_rate
    curve_plan = replace(
        _fitter_plan(plan, subset, plan.sectors),
        params=xgboost_params(params, plan),
        non_uk_rates=(non_uk_rate,),
        zero_rates=(zero_rate,),
    )
    tuned = experiments.run_learning_curve(subset, curve_plan)["learning_curve"][0]
    return {"non_uk_rate": non_uk_rate, "zero_rate": zero_rate, "tuned": tuned}


def _fitter_plan(plan: SearchPlan, data: Data, sectors: Sequence[str] | None) -> experiments.Plan:
    """The experiments' plan for one stage: its Sectors, fold, round cap and device."""
    return experiments.Plan(
        params=xgboost_params(V1_TRIAL, plan),
        max_rounds=plan.max_rounds,
        sectors=tuple(sectors) if sectors else None,
        built_zero_rate=data.built_zero_rate,
        ablation_rates=(1.0, data.built_zero_rate),
        non_uk_rates=(),
        zero_rates=(),
        validation_year=data.validation_year,
        early_stopping_rounds=plan.early_stopping_rounds,
        fold=plan.fold,
        n_folds=plan.n_folds,
        device=plan.device,
        seed=plan.seed,
        uk=plan.uk,
    )


# --- Data and saved results -----------------------------------------------------------------


def feature_data(root: Path | None = None, *, validation_year: int = experiments.VALIDATION_YEAR) -> Data:
    """Every Product's rows from the feature files of `features.build()`, up to the validation year only."""
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    return Data(
        training={y: folder / f"training_{y}.parquet" for y in settings["years"] if y < validation_year},
        validation=folder / f"training_{validation_year}.parquet",
        grid=folder / f"uk_grid_{validation_year}.parquet",
        validation_year=validation_year,
        built_zero_rate=settings["zero_sampling_rate"],
    )


def search_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"search_{baci.BACI_VERSION}"


def results_path(root: Path | None = None) -> Path:
    return search_dir(root) / RESULTS_FILE


def load(root: Path | None = None) -> dict:
    """The saved results of the real search (`run` with `search_dir`)."""
    return json.loads(results_path(root).read_text(encoding="utf-8"))


def _save_results(results: dict, folder: Path | None) -> None:
    if folder is None:
        return
    path = folder / RESULTS_FILE
    partial = path.with_suffix(".partial")
    partial.write_text(_dumps(results), encoding="utf-8")
    partial.replace(path)


def _dumps(results: dict) -> str:
    return json.dumps(results, indent=2, default=experiments.json_default)


def main(argv: Sequence[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m exportspredictor.search")
    results = run(experiments.real_data(), feature_data(), REAL_SEARCH, search_dir())
    print(json.dumps({k: results[k] for k in ("winner", "v1", "spearman_vs_v1", "capped_trials")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
