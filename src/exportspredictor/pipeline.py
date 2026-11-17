"""From feature tables to a servable artefact bundle, for the fixture and for real data.

Both paths run the same steps (`_train_and_score`), so the fixture tests what real data runs:

1. **Tune** on the validation year (Baseline year − 2, i.e. 2022): one model per
   hyperparameter configuration, trained on the years before it, with early stopping on
   it. The test year (Baseline year − 1) is never read here.
2. **Refit** the fold models (ADR 0001) on every year before the Baseline year, with the
   best configuration and its round count; the feature transform is fitted on the same years.
3. **Score** the Baseline year's full UK Product × Market grid out-of-fold and assemble the
   Baseline table, the Scenario inputs and the Actual Exports history.
4. **Blocked Opportunity values**: re-predict every pair without a Trade Agreement with
   one added (`scenarios.blocked_effects`), for the `rank_by=blocked` rankings.
5. **Evaluate** on the test year (Baseline year − 1): fold models refitted on the years
   before it with the tuned configuration score it, against a naive baseline
   (`evaluation`). The report is saved in the artefact as `evaluation.json`.

`build_artefacts(world)` runs this on the fixture world (CPU, seconds).
`build_real_artefacts()` runs it on the per-year feature files of `features.build()`
(GPU); `uv run python -m exportspredictor.pipeline` runs it and saves the artefact folder.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import baci, backtests, constraints, evaluation
from exportspredictor.artefacts import ARTEFACT_FORMAT, Artefacts, new_version
from exportspredictor.exclusions import (
    EXCLUSIONS,
    drop_excluded,
    excluded_products,
    is_excluded,
)
from exportspredictor.features import (
    FEATURES,
    MONOTONE_CONSTRAINTS,
    FeatureTransform,
    build_features,
    features_dir,
    load_sources,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK, World
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models
from exportspredictor.lookups import market_lookup
from exportspredictor.paths import data_dir
from exportspredictor.scenarios import blocked_effects, with_blocked_effects
from exportspredictor.sources import source_versions
from exportspredictor.training import Chunk, model_params, read_chunk, scan_chunks, tune

# The fixture world has only a few dozen relevant zeros, so it keeps half of them. Real
# data uses `features.ZERO_SAMPLING_RATE`, chosen in the notebook.
FIXTURE_ZERO_SAMPLING_RATE = 0.5
FIXTURE_CONFIGS = [{"max_depth": 2, "eta": 0.3}, {"max_depth": 3, "eta": 0.3}]
FIXTURE_MAX_ROUNDS = 50

_COMMON = {"eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}
# A deliberately small search on the 2022 validation year: tree depth around a middle
# value, and the Tweedie variance power either side of 1.5.
REAL_CONFIGS = [
    {**_COMMON, "max_depth": 8},
    {**_COMMON, "max_depth": 6},
    {**_COMMON, "max_depth": 10},
    {**_COMMON, "max_depth": 8, "tweedie_variance_power": 1.3},
    {**_COMMON, "max_depth": 8, "tweedie_variance_power": 1.7},
]
REAL_MAX_ROUNDS = 1500
# Share of non-UK training rows kept (reweighted by 1/rate; UK rows are always kept). All
# ~150 M rows of 2012–2023 fill the RTX 3080's 10 GB once training starts, so the real
# run trains on a Horvitz–Thompson sample of them.
REAL_NON_UK_RATE = 0.3
HISTORY_YEARS = 6  # Actual Exports history: the Baseline year and the five before it

# The fixture's backtests, inside its four years: Lift from a 2021–2022 model on 2022 gaps,
# grown by 2023; the UK–Japan agreement of 2022 as the Event Backtest, with a 2021 model.
# The fixture has no high-income Market without a UK agreement, so controls ignore income.
FIXTURE_BACKTESTS = backtests.Plan(
    lift=backtests.LiftSettings(
        training_years=(2021, 2022), gap_year=2022, outcome_year=2023, min_growth_share=0.05, min_growth_usd=0.0
    ),
    event=backtests.Event(
        name="uk_japan",
        title="The fixture's UK–Japan Trade Agreement",
        in_force="2022",
        event_year=2022,
        training_years=(2021,),
        pre_years=(2021,),
        post_years=(2022, 2023),
        directions=(backtests.Direction("UK exports to Japan", (UK,), ("JPN",)),),
        caveat="The fixture world is synthetic; this only shows the Event Backtest working end to end.",
    ),
    high_income_controls=False,
    resamples=200,
)


@dataclass(frozen=True)
class _Trained:
    models: list[xgb.Booster]
    baseline: pl.DataFrame
    baseline_features: pl.DataFrame
    metadata: dict


def build_artefacts(
    world: World,
    *,
    version: str | None = None,
    baseline_year: int | None = None,
    n_folds: int = N_FOLDS,
    device: str = "cpu",
    source: str = "fixture",
    zero_sampling_rate: float = FIXTURE_ZERO_SAMPLING_RATE,
    configs: Sequence[dict] = FIXTURE_CONFIGS,
    max_rounds: int = FIXTURE_MAX_ROUNDS,
    seed: int = 0,
) -> Artefacts:
    """Tune, train fold models and score the Baseline for every UK pair of the fixture world.

    `zero_sampling_rate` is the share of relevant zero cells kept for training; it is
    recorded in the metadata.
    """
    flows, sources = world.flows, world.sources
    baseline_year = baseline_year or int(flows["year"].max())
    training = fixture_training(world, baseline_year=baseline_year, zero_sampling_rate=zero_sampling_rate, seed=seed)
    grids = {}
    for year in (baseline_year - 1, baseline_year):
        year_flows = flows.filter(pl.col("year") == year)
        grids[year] = build_features(uk_grid(year_flows, UK), flows, sources)  # all years: growth reads t-3
    grid = grids[baseline_year]
    trained = _train_and_score(
        training, grid, baseline_year=baseline_year, configs=configs, max_rounds=max_rounds,
        n_folds=n_folds, device=device, non_uk_rate=1.0, seed=seed,
    )
    recent = (pl.col("year") > baseline_year - HISTORY_YEARS) & (pl.col("year") <= baseline_year)
    metadata = {
        "version": version or new_version(),
        "source": source,
        **trained.metadata,
        "zero_sampling_rate": zero_sampling_rate,
        "zero_sampling_seed": seed,
    }
    report = _evaluate(
        training, grids[baseline_year - 1], trained, metadata, markets=world.markets, device=device, non_uk_rate=1.0,
        seed=seed, backtest_plan=FIXTURE_BACKTESTS, high_income=world.high_income,
        constraint_plan=constraints.FIXTURE_PLAN,
    )
    return Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=_history(flows.filter((pl.col("exporter") == UK) & recent)),
        products=world.products,
        markets=world.markets,
        excluded_products=excluded_products(world.products),
        evaluation=report,
    )


def fixture_training(
    world: World,
    *,
    baseline_year: int | None = None,
    zero_sampling_rate: float = FIXTURE_ZERO_SAMPLING_RATE,
    seed: int = 0,
) -> dict[int, pl.DataFrame]:
    """The fixture world's training rows with features, one table per year before the Baseline year."""
    flows = world.flows
    baseline_year = baseline_year or int(flows["year"].max())
    training = {}
    for year in sorted(flows["year"].unique().to_list()):
        if year < baseline_year:
            year_flows = flows.filter(pl.col("year") == year)
            rows = training_rows(year_flows, rate=zero_sampling_rate, seed=seed)
            training[year] = build_features(rows, flows, world.sources)
    return training


def build_real_artefacts(
    root: Path | None = None,
    *,
    version: str | None = None,
    baseline_year: int = 2024,
    configs: Sequence[dict] = REAL_CONFIGS,
    max_rounds: int = REAL_MAX_ROUNDS,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    non_uk_rate: float = REAL_NON_UK_RATE,
    seed: int = 0,
    backtest_plan: backtests.Plan = backtests.REAL_PLAN,
    constraint_plan: constraints.Plan = constraints.REAL_PLAN,
) -> Artefacts:
    """Tune, train and score on the real per-year feature files written by `features.build()`.

    Years are read one at a time (`training`), so host memory stays far below 32 GB.
    Backtest models are cached under the data root (`backtests.cache_dir`).
    """
    root = Path(root or data_dir())
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    built_without = settings.get("excluded_products")
    if built_without != [exclusion.codes for exclusion in EXCLUSIONS]:
        raise ValueError(
            f"The feature files in {folder} were built excluding {built_without}, not the current Excluded "
            "Products; rebuild them with `python -m exportspredictor.features`"
        )
    if settings.get("features") != FEATURES:
        raise ValueError(
            f"The feature files in {folder} hold {settings.get('features')}, not the current features; "
            "rebuild them with `python -m exportspredictor.features`"
        )
    training = {y: folder / f"training_{y}.parquet" for y in settings["years"] if y < baseline_year}
    grid = pl.read_parquet(folder / f"uk_grid_{baseline_year}.parquet")
    trained = _train_and_score(
        training, grid, baseline_year=baseline_year, configs=configs, max_rounds=max_rounds,
        n_folds=n_folds, device=device, non_uk_rate=non_uk_rate, seed=seed,
    )
    history_years = [y for y in settings["years"] if baseline_year - HISTORY_YEARS < y <= baseline_year]
    history = _history(
        pl.concat(
            [
                pl.scan_parquet(folder / f"uk_grid_{y}.parquet")
                .filter(pl.col("value_usd") > 0)
                .select(pl.col("hs6").cast(pl.String), "importer", "year", "value_usd")
                for y in history_years
            ]
        ).collect()
    )
    country_map = pl.read_parquet(root / "derived" / "country_map.parquet")
    markets = market_lookup(baci.markets(root), country_map)
    metadata = {
        "version": version or new_version(),
        "source": f"BACI {baci.BACI_VERSION}",
        "sources": source_versions(root),
        **trained.metadata,
        "zero_sampling_rate": settings["zero_sampling_rate"],
        "zero_sampling_seed": settings["zero_sampling_seed"],
    }
    test_grid = folder / f"uk_grid_{baseline_year - 1}.parquet"
    report = _evaluate(
        training, test_grid, trained, metadata, markets=evaluation.markets_with_income(markets, root),
        device=device, non_uk_rate=non_uk_rate, seed=seed,
        backtest_plan=backtest_plan,
        backtest_chunks={y: folder / f"training_{y}.parquet" for y in settings["years"]},
        high_income=load_sources(root)[0].high_income,
        backtest_cache=backtests.cache_dir(root),
        constraint_plan=constraint_plan,
    )
    return Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=history,
        products=baci.products(root),
        markets=markets,
        excluded_products=excluded_products(baci.products(root)),
        evaluation=report,
    )


def _train_and_score(
    training: Mapping[int, Chunk],
    grid: pl.DataFrame,
    *,
    baseline_year: int,
    configs: Sequence[dict],
    max_rounds: int,
    n_folds: int,
    device: str,
    non_uk_rate: float,
    seed: int,
) -> _Trained:
    validation_year, test_year = baseline_year - 2, baseline_year - 1
    tuning_years = [y for y in sorted(training) if y < validation_year]
    training_years = [y for y in sorted(training) if y < baseline_year]
    if not tuning_years or validation_year not in training:
        raise ValueError(f"Need training years before {validation_year} and {validation_year} itself")
    # The test year is reserved for the evaluation: tuning must never read it.
    assert test_year not in tuning_years and test_year != validation_year

    # 1. Tune on the validation year (the test year, baseline_year - 1, is not read).
    started = time.perf_counter()
    tuning_chunks = [training[y] for y in tuning_years]
    tuning = tune(
        tuning_chunks,
        [training[validation_year]],
        configs,
        uk=UK,
        transform=FeatureTransform.fit(scan_chunks(tuning_chunks), tuning_years),
        device=device,
        non_uk_rate=non_uk_rate,
        max_rounds=max_rounds,
        seed=seed,
    )
    tuning_seconds = time.perf_counter() - started
    best = tuning.row(0, named=True)
    params, rounds = json.loads(best["config"]), best["rounds"]
    print(f"Tuned in {tuning_seconds:.0f} s: {params}, {rounds} rounds", flush=True)

    # 2. Refit the fold models on every year before the Baseline year.
    started = time.perf_counter()
    chunks = [training[y] for y in training_years]
    transform = FeatureTransform.fit(scan_chunks(chunks), training_years)
    models = train_fold_models(
        chunks, uk=UK, params=params, rounds=rounds, n_folds=n_folds, device=device, seed=seed,
        transform=transform, non_uk_rate=non_uk_rate,
    )
    training_seconds = time.perf_counter() - started
    print(f"Trained {n_folds} fold models in {training_seconds:.0f} s", flush=True)

    # 3. Score the Baseline year's full UK grid out-of-fold.
    scored = (
        score_out_of_fold(transform.apply(grid), models, uk=UK)
        .with_columns(hs6=pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String))
        .sort("hs6", "iso3")
    )
    if scored.select("hs6", "iso3").is_duplicated().any() or scored.height != grid.height:
        raise ValueError("The Baseline must hold exactly one row per UK Product–Market pair of the grid")
    if scored.select(is_excluded().any()).item():
        raise ValueError("The UK grid holds Excluded Products; they are never scored (ADR 0005)")
    baseline = scored.select(
        "hs6",
        "iso3",
        "export_potential_usd",
        actual_exports_usd="value_usd",
    ).with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
    baseline_features = scored.select("hs6", "iso3", "fold", *FEATURES)
    # 4. Blocked Opportunity values: the Scenario Effect of adding a Trade Agreement where none exists.
    baseline = with_blocked_effects(baseline, blocked_effects(baseline_features, baseline, models))
    metadata = {
        "artefact_format": ARTEFACT_FORMAT,
        "baseline_year": baseline_year,
        "validation_year": validation_year,
        "test_year": test_year,
        "tuning_years": tuning_years,
        "training_years": training_years,
        "n_folds": n_folds,
        "features": FEATURES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "non_uk_sampling_rate": non_uk_rate,
        "feature_transform": transform.to_dict(),
        "params": params,
        "model_params": model_params(params, device=device, seed=seed),
        "rounds": rounds,
        "tuning": [{**row, "config": json.loads(row["config"])} for row in tuning.iter_rows(named=True)],
        "device": device,
        "xgboost_version": xgb.__version__,
        "tuning_seconds": round(tuning_seconds, 2),
        "training_seconds": round(training_seconds, 2),
        "created_at": datetime.now(UTC).isoformat(),
    }
    return _Trained(
        models=models,
        baseline=baseline,
        baseline_features=baseline_features,
        metadata=metadata,
    )


def _evaluate(
    training: Mapping[int, Chunk],
    test_grid: Chunk,
    trained: _Trained,
    metadata: dict,
    *,
    markets: pl.DataFrame,
    device: str,
    non_uk_rate: float,
    seed: int,
    backtest_plan: backtests.Plan,
    high_income,
    constraint_plan: constraints.Plan,
    backtest_chunks: Mapping[int, Chunk] | None = None,
    backtest_cache: Path | None = None,
) -> dict:
    """Evaluate on the test year, run Lift and the Event Backtests (`backtests`), then compare
    the monotone-constrained model with an unconstrained one (`constraints`).

    The test-year fold models are trained with the tuned configuration; the backtest
    models use it too. `backtest_chunks` defaults to `training`.
    """
    test_year = metadata["test_year"]
    started = time.perf_counter()
    test_models = evaluation.train_test_models(
        {y: chunk for y, chunk in training.items() if y < test_year},
        test_year=test_year,
        uk=UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        n_folds=metadata["n_folds"],
        device=device,
        non_uk_rate=non_uk_rate,
        seed=seed,
    )
    report = evaluation.evaluate(
        read_chunk(training[test_year]),
        read_chunk(test_grid),
        test_models=test_models,
        baseline=trained.baseline,
        baseline_models=trained.models,
        baseline_transform=FeatureTransform.from_dict(metadata["feature_transform"]),
        markets=markets,
        uk=UK,
        settings=evaluation.report_settings(metadata),
    )
    report["evaluation_seconds"] = round(time.perf_counter() - started, 2)
    print(f"Evaluated on {test_year} in {report['evaluation_seconds']:.0f} s", flush=True)
    report |= backtests.run(
        backtest_chunks or training, backtest_plan, uk=UK, params=metadata["params"], rounds=metadata["rounds"],
        device=device, non_uk_rate=non_uk_rate, seed=seed, high_income=high_income, test_models=test_models,
        cache=backtest_cache,
    )
    print(f"Backtests in {report['event_backtest']['seconds']:.0f} s", flush=True)
    report["monotone_constraints"] = constraints.compare(
        {y: chunk for y, chunk in training.items() if y in metadata["training_years"]},
        trained.baseline_features,
        plan=constraint_plan,
        uk=UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        transform=FeatureTransform.from_dict(metadata["feature_transform"]),
        n_folds=metadata["n_folds"],
        non_uk_rate=non_uk_rate,
        device=device,
        seed=seed,
    )
    print(f"Monotone-constraint comparison in {report['monotone_constraints']['seconds']:.0f} s", flush=True)
    return report


def _history(uk_flows: pl.DataFrame) -> pl.DataFrame:
    """Non-zero UK Actual Exports per pair and year, for every Product but the Excluded ones."""
    return (
        uk_flows.filter(pl.col("value_usd") > 0)
        .pipe(drop_excluded)
        .select(
            pl.col("hs6").cast(pl.String),
            iso3=pl.col("importer").cast(pl.String),
            year=pl.col("year").cast(pl.Int16),
            actual_exports_usd=pl.col("value_usd"),
        )
        .sort("hs6", "iso3", "year")
    )


def refresh(bundle: Artefacts, *, version: str | None = None, data_root: Path | None = None) -> Artefacts:
    """A new version of a saved bundle with its derived metadata recomputed; no retraining.

    The fold models and tables are reused as they are. Use it when a derived value (such as
    the source versions read from the raw downloads under `data_root`) is added after the
    models were trained. A v1 `gbp_rate` is dropped: every output is in USD.
    """
    metadata = {
        **{key: value for key, value in bundle.metadata.items() if key != "gbp_rate"},
        "version": version or new_version(),
        "refreshed_from": bundle.version,
        "sources": source_versions(data_root),
    }
    return replace(bundle, metadata=metadata)


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m exportspredictor.pipeline` trains and saves a real bundle (about 30 min on GPU).

    `python -m exportspredictor.pipeline refresh` instead saves a refreshed copy of the newest
    saved bundle (`refresh`), in seconds.
    """
    import sys

    from exportspredictor import artefacts
    from exportspredictor.paths import artefacts_dir

    argv = sys.argv[1:] if argv is None else argv
    if list(argv) == ["refresh"]:
        source = artefacts.latest_folder(artefacts_dir())
        bundle = refresh(artefacts.load(source))
        print(json.dumps({k: bundle.metadata[k] for k in ("version", "refreshed_from", "sources")}), flush=True)
        # Copied file for file, so files this step does not know about (e.g. the model card) survive.
        print(artefacts.copy_version(source, artefacts_dir(), metadata=bundle.metadata), flush=True)
        return
    bundle = build_real_artefacts()
    summary = {k: bundle.metadata[k] for k in ("version", "params", "rounds", "tuning_seconds", "training_seconds")}
    print(json.dumps(summary), flush=True)
    print(artefacts.save(bundle, artefacts_dir()), flush=True)


if __name__ == "__main__":
    main()
