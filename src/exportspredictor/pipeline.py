"""From feature tables to a servable artefact bundle, for the fixture and for real data.

Both paths run the same steps (`_train_and_score`), so the fixture tests what real data runs:

0. **Fit the row budget** (ADR 0004): count the training years' UK and other rows, then
   derive the largest non-UK sampling rate that fits the VRAM row budget, or refuse a
   given rate that does not, before any training.
1. **The model form** the experiment loop chose (spec #65, `loop.chosen_form`): the model
   alone or a blend with the naive baseline. The real run reads the saved loop results, the
   fixture runs a tiny loop; the artefact records the form, the selection measure, the noise
   floor and the experiment results (`loop.ranking_record`). Every row scored below carries
   its naive baseline (`naive_usd`, from every Exporter's Flows of its year), which a blend
   needs and a Scenario never changes.
   **The hyperparameter search** the loop started from (`search`, validated on Baseline year − 2, 2022) is
   recorded beside it: the real run reads the saved search, the fixture runs a tiny one.
   The test year (Baseline year − 1) is never read here.
2. **Refit** the fold models (ADR 0001) on every year before the Baseline year, with the
   configuration the loop selected and the round count its selection measure chose
   (`loop.served_configuration`: the median of the five-fold confirmation's rounds), scaled
   by the ratio of training rows (`search.scaled_rounds`, a heuristic); the feature
   transform is fitted on the same years. Loop results validated on another year are refused.
3. **Score** every year of the Gap Period (`gap_period`, 2022–2024): each year's UK grid
   (the union of the Gap Period's pairs) out-of-fold, with that year's features. The
   Baseline table holds the Gap Period means and the Baseline year's Export Potential; the
   history holds each year's scores; the Scenario inputs are the Baseline year's features.
4. **Blocked Opportunity values**: re-predict every pair without a Trade Agreement with
   one added (`scenarios.blocked_opportunity_values`), against the Baseline year only.
   **Potential Range** (ADR 0007, `conformal`): score other Exporters' Baseline-year
   training rows (never the UK's) with their hash-assigned fold models, fit the band
   offsets, apply them around the Gap Period mean Export Potential, and measure coverage
   on a held-out half of those rows.
5. **Evaluate** on the test year (Baseline year − 1): fold models refitted on the years
   before it with the tuned configuration score it, against a naive baseline
   (`evaluation`), and the v1 comparison's primary metric is read against the frozen v1
   tables. The report is saved in the artefact as `evaluation.json`.
6. **Opt-in parts** (`Parts`), each run only when asked for: the Event Backtests (Lift
   included), the monotone-constraint comparison and the gain ladder. A skipped part is
   absent from the report, which then shows it as "not run for this version".

`build_artefacts(world)` runs this on the fixture world (CPU, seconds).
`build_real_artefacts()` runs it on the per-year feature files of `features.build()`
(GPU); `uv run python -m exportspredictor.pipeline` runs it and saves the artefact folder.
`... pipeline refresh` copies the newest version with fresh metadata (`refresh`), `... pipeline rerange`
copies it with its Potential Range recomputed (`rerange`), and `--help` lists both.
"""

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import (
    baci,
    backtests,
    comparison,
    conformal,
    constraints,
    evaluation,
    experiments,
    fixture,
    freeze,
    governance,
    loop,
    ppml,
    search,
)
from exportspredictor.artefacts import ARTEFACT_FORMAT, Artefacts, new_version
from exportspredictor.exclusions import (
    EXCLUSIONS,
    drop_excluded,
    excluded_products,
    flag_columns,
    is_excluded,
)
from exportspredictor.features import (
    FEATURES,
    GROWTH_YEARS,
    KEYS,
    MONOTONE_CONSTRAINTS,
    FeatureTransform,
    build_features,
    features_dir,
    grid_pairs,
    load_sources,
    load_year_flows,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK, World
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models
from exportspredictor.gap_period import (
    baseline_year_scores,
    export_gaps,
    gap_period,
    gap_years,
    with_opportunity_columns,
)
from exportspredictor.lookups import market_lookup
from exportspredictor.paths import data_dir
from exportspredictor.prediction import ModelForm
from exportspredictor.rankings import DEFAULT_MIN_POTENTIAL_USD
from exportspredictor.scenarios import (
    DEFAULT_MIN_SHARE,
    DEFAULT_MIN_USD,
    blocked_opportunity_values,
    with_blocked_opportunity_values,
)
from exportspredictor.scope import exporter_scope
from exportspredictor.sources import source_versions
from exportspredictor.tariffs import LAST_TRAINS_YEAR
from exportspredictor.training import (
    GPU_CHECK,
    Chunk,
    check_row_budget,
    count_rows,
    fit_non_uk_rate,
    model_params,
    read_chunk,
    scan_chunks,
)
from exportspredictor.training import row_budget as vram_row_budget

# The fixture world has only a few dozen relevant zeros, so it keeps half of them. Real
# data uses `features.ZERO_SAMPLING_RATE`, chosen in the notebook.
FIXTURE_ZERO_SAMPLING_RATE = 0.5
# Share of non-UK training rows kept (reweighted by 1/rate; UK rows are always kept). None
# derives it: the largest rate whose 2012–2023 rows fit the VRAM row budget (ADR 0004,
# `training.fit_non_uk_rate`). A given rate is checked against the budget before training.
# The learning curve (ticket #53) chose None: on 2022 the deviance still fell by 6% from a
# rate of 0.3 to 0.65, so the model takes every non-UK row the budget allows.
REAL_NON_UK_RATE: float | None = None
HISTORY_YEARS = 6  # Actual Exports history: the Baseline year and the five before it
# The fixture world has about 60 calibration rows, so its bands need far fewer than real data's 1,000.
FIXTURE_MIN_BAND_ROWS = 12

# The fixture's backtests, inside its four years: Lift from a 2021–2022 model on gaps averaged
# over a 2021–2022 Gap Period, grown by 2023; the UK–Japan agreement of 2022 as the Event
# Backtest, with a 2021 model. The fixture has no high-income Market without a UK agreement,
# so controls ignore income.
FIXTURE_BACKTESTS = backtests.Plan(
    lift=backtests.LiftSettings(
        training_years=(2021, 2022), gap_years=(2021, 2022), outcome_year=2023, min_growth_share=0.05,
        min_growth_usd=0.0,
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
class Parts:
    """The opt-in parts of a run (spec #65, story 23). The default trains only the fold models
    and the test-year models; each part switched on adds exactly its own models and report block.
    """

    event_backtests: bool = False  # Lift and the Event Backtests (`backtests.run`)
    constraint_comparison: bool = False  # the monotone-constrained model against an unconstrained one
    gain_ladder: bool = False  # rungs A and B of the v1 comparison's gain ladder
    ppml_benchmark: bool = False  # the PPML benchmark (`ppml.benchmark`): fitted on the CPU, trains no model

    @classmethod
    def every(cls) -> "Parts":
        """Every part: the full run of before the switches, with the PPML benchmark."""
        return cls(event_backtests=True, constraint_comparison=True, gain_ladder=True, ppml_benchmark=True)


# The fixture's stand-in for v1's frozen held-out Lift Exporters: MEX is outside the fixture's
# Exporter scope, so the Lift model holds out FRA alone.
FIXTURE_V1_LIFT_EXPORTERS = ("FRA", "MEX")
# The `source` an artefact built from the fixture world records; real artefacts name the BACI release.
FIXTURE_SOURCE = "fixture"
# The synthetic frozen v1's manifest fields (`freeze.write` adds the tables' checksums).
FIXTURE_V1_MANIFEST = {
    "tag": "fixture",
    "commit": None,
    "source_models": {"artefact": "fixture"},
    "test_year": FIXTURE_BACKTESTS.lift.outcome_year,
    "lift_years": list(FIXTURE_BACKTESTS.lift.gap_years),
    "event": FIXTURE_BACKTESTS.event.name,
    "event_post_years": list(FIXTURE_BACKTESTS.event.post_years),
}


def fixture_v1_frozen(world: World) -> comparison.Frozen:
    """The fixture's synthetic frozen v1 (`fixture.frozen_v1_tables`), shaped for its backtest plan.

    `freeze.write(folder, frozen.tables, frozen.manifest)` writes it as a frozen folder.
    """
    event = FIXTURE_BACKTESTS.event
    tables = fixture.frozen_v1_tables(
        world,
        test_year=FIXTURE_V1_MANIFEST["test_year"],
        lift_years=FIXTURE_BACKTESTS.lift.gap_years,
        lift_held_out=FIXTURE_V1_LIFT_EXPORTERS,
        event_directions=[(d.label, d.exporters, d.importers) for d in event.directions],
        event_years=event.post_years,
    )
    return comparison.Frozen(manifest=dict(FIXTURE_V1_MANIFEST), tables=tables)


@dataclass(frozen=True)
class _Trained:
    models: list[xgb.Booster]
    baseline: pl.DataFrame
    baseline_features: pl.DataFrame
    yearly: pl.DataFrame  # hs6, iso3, year, actual_exports_usd, export_potential_usd, export_gap_usd, tariff_imputed
    potential_range_bands: pl.DataFrame
    potential_range_report: dict  # held-out coverage and the share of Opportunities that are uncertain
    metadata: dict


def build_artefacts(
    world: World,
    *,
    version: str | None = None,
    baseline_year: int | None = None,
    n_folds: int = N_FOLDS,
    device: str = "cpu",
    source: str = FIXTURE_SOURCE,
    zero_sampling_rate: float = FIXTURE_ZERO_SAMPLING_RATE,
    search_plan: search.SearchPlan = search.FIXTURE_SEARCH,
    seed: int = 0,
    row_budget: int | None = None,
    min_band_rows: int = FIXTURE_MIN_BAND_ROWS,
    experiment_results: dict | None = None,
    v1_frozen: comparison.Frozen | None = None,
    search_results: dict | None = None,
    parts: Parts = Parts(),
    ranking_results: dict | None = None,
) -> Artefacts:
    """Tune, train fold models and score the Baseline for every UK pair of the fixture world.

    `zero_sampling_rate` is the share of other Exporters' Zero Cells kept for training; it
    is recorded in the metadata. `row_budget` overrides the VRAM row budget. A tiny
    hyperparameter search (`search_plan`) tunes on the validation year.
    `min_band_rows` is the Potential Range's minimum band size. `experiment_results` (the
    feature ablation and learning curve, `experiments.run`) are recorded in the metadata.
    `v1_frozen` is the frozen v1 the evaluation compares with (default: the fixture's
    synthetic one, `fixture_v1_frozen`). `search_results` skips the search and uses these
    (`metadata["search"]` of an earlier run). `parts` switches on the opt-in parts (`Parts`).
    `ranking_results` are the experiment loop's results whose chosen form is served (default:
    the fixture's own tiny loop, `loop.fixture_ranking`).
    """
    flows, sources = world.flows, world.sources
    baseline_year = baseline_year or int(flows["year"].max())
    scope = exporter_scope(world.country_year, uk=UK)
    training = fixture_training(
        world, baseline_year=baseline_year, zero_sampling_rate=zero_sampling_rate, seed=seed, exporters=scope
    )

    def flows_of_year(year: int) -> pl.DataFrame:
        """Every Exporter's Flows of one year."""
        return flows.filter(pl.col("year") == year)

    # Every Gap Period year is scored over the union of the Gap Period's grids, as `features.build` writes them.
    # `build_features` gets every year's Flows: Market import growth reads year t-3.
    gap_flows = {year: flows_of_year(year) for year in gap_years(baseline_year)}
    pairs = grid_pairs(uk_grid(f, UK) for f in gap_flows.values())
    grids = {year: build_features(uk_grid(f, UK, pairs=pairs), flows, sources) for year, f in gap_flows.items()}
    # The Potential Range's calibration rows: the Baseline year's training rows (ADR 0007).
    calibration = build_features(
        training_rows(gap_flows[baseline_year], rate=zero_sampling_rate, seed=seed, uk=UK, exporters=scope),
        flows,
        sources,
    )
    if row_budget is not None:  # refuse before the search trains anything
        check_row_budget(count_rows(list(training.values()), uk=UK), 1.0, row_budget)
    validation_year = baseline_year - 2
    tuning_data = experiments.Data(
        training={y: rows for y, rows in training.items() if y < validation_year},
        validation=training[validation_year],
        grid=build_features(uk_grid(flows.filter(pl.col("year") == validation_year), UK), flows, sources),
        validation_year=validation_year,
        built_zero_rate=zero_sampling_rate,
    )
    if search_results is None:
        search_results = search.run(tuning_data, tuning_data, replace(search_plan, device=device, seed=seed))
    if ranking_results is None:
        ranking_results = loop.fixture_ranking(world)
    trained = _train_and_score(
        training, grids, baseline_year=baseline_year, search_results=search_results,
        n_folds=n_folds, device=device, non_uk_rate=1.0, seed=seed, row_budget=row_budget,
        calibration=calibration, min_band_rows=min_band_rows, ranking_results=ranking_results,
        year_flows=flows_of_year,
    )
    recent = (pl.col("year") > baseline_year - HISTORY_YEARS) & (pl.col("year") <= baseline_year)
    metadata = {
        "version": version or new_version(),
        "source": source,
        **trained.metadata,
        "exporter_scope": scope,
        "zero_sampling_rate": zero_sampling_rate,
        "uk_zero_sampling_rate": 1.0,
        "zero_sampling_seed": seed,
        "experiments": experiment_results,
    }
    frozen = v1_frozen or fixture_v1_frozen(world)
    v1 = comparison.Inputs(
        frozen=frozen,
        rows=comparison.feature_rows(frozen, scope=scope, uk=UK, flows=flows, sources=sources),
        flows=flows,
        ladder=comparison.FIXTURE_LADDER,
    )
    report = _evaluate(
        training, grids[baseline_year - 1], trained, metadata, markets=world.markets, device=device, non_uk_rate=1.0,
        test_flows=gap_flows[baseline_year - 1],
        seed=seed, backtest_plan=FIXTURE_BACKTESTS, high_income=world.high_income,
        constraint_plan=constraints.FIXTURE_PLAN,
        lift_held_out=backtests.lift_exporters(frozen.lift_held_out_exporters, scope=scope, uk=UK),
        v1=v1,
        parts=parts,
        year_flows=flows_of_year,
    )
    bundle = Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=_history(flows.filter((pl.col("exporter") == UK) & recent), trained.yearly),
        products=world.products,
        markets=world.markets,
        excluded_products=excluded_products(world.products),
        potential_range_bands=trained.potential_range_bands,
        evaluation=report,
    )
    if not parts.ppml_benchmark:
        return bundle
    # The fixture world has no 3-year intervals: its benchmark is fitted on every year it holds.
    panel = ppml.sector_flows(flows, sources.agreements, years=flows["year"].unique().sort().to_list())
    return replace(bundle, ppml_benchmark=_timed_benchmark(panel, bundle))


def _timed_benchmark(panel: pl.DataFrame, bundle: Artefacts) -> dict:
    """The PPML benchmark of `bundle` (`ppml.benchmark`) with the seconds it took."""
    started = time.perf_counter()
    benchmark = ppml.benchmark(panel, bundle)
    print(f"PPML benchmark in {time.perf_counter() - started:.0f} s", flush=True)
    return {**benchmark, "seconds": round(time.perf_counter() - started, 2)}


def fixture_training(
    world: World,
    *,
    baseline_year: int | None = None,
    zero_sampling_rate: float = FIXTURE_ZERO_SAMPLING_RATE,
    seed: int = 0,
    exporters: Sequence[str] | None = None,
) -> dict[int, pl.DataFrame]:
    """The fixture world's training rows with features, one table per year before the Baseline year."""
    flows = world.flows
    baseline_year = baseline_year or int(flows["year"].max())
    training = {}
    for year in sorted(flows["year"].unique().to_list()):
        if year < baseline_year:
            one_year = flows.filter(pl.col("year") == year)
            rows = training_rows(one_year, rate=zero_sampling_rate, seed=seed, uk=UK, exporters=exporters)
            # Sorted: row order decides which rows `subsample` draws, so the fixture models are reproducible.
            training[year] = build_features(rows, flows, world.sources).sort(KEYS)
    return training


def build_real_artefacts(
    root: Path | None = None,
    *,
    version: str | None = None,
    baseline_year: int = 2024,
    search_results: dict | None = None,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    non_uk_rate: float | None = REAL_NON_UK_RATE,
    seed: int = 0,
    row_budget: int | None = None,
    min_band_rows: int = conformal.MIN_BAND_ROWS,
    backtest_plan: backtests.Plan = backtests.REAL_PLAN,
    constraint_plan: constraints.Plan = constraints.REAL_PLAN,
    v1_frozen: Path | None = None,
    ladder_plan: comparison.LadderPlan = comparison.REAL_LADDER,
    parts: Parts = Parts(),
    ranking_results: dict | None = None,
    ppml_years: Sequence[int] = ppml.YEARS,
) -> Artefacts:
    """Tune, train and score on the real per-year feature files written by `features.build()`.

    Years are read one at a time (`training`), so host memory stays far below 32 GB.
    Backtest models are cached under the data root (`backtests.cache_dir`). The Lift model
    holds out v1's frozen held-out Exporters within scope, read from `v1_frozen`
    (default: `freeze.v1_folder()`). The saved feature ablation and learning curve
    (`experiments.save`) are recorded in the metadata. The tuned configuration comes from
    `search_results`, by default the saved hyperparameter search (`search.load`). The evaluation
    runs the pre-registered v1 comparison against the same frozen v1, with the gain ladder's
    rungs trained with `ladder_plan` (cached under `ladder_dir`). The Event Backtests, the
    constraint comparison and the gain ladder run only when switched on in `parts` (`Parts`).
    The model form served is the one the experiment loop chose, read from `ranking_results`,
    by default the saved loop results (`loop.load`). The PPML benchmark (`parts.ppml_benchmark`) is fitted
    on the Flows of `ppml_years` only.
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
    if settings.get("gap_years") != gap_period(baseline_year):
        raise ValueError(
            f"The feature files in {folder} were built for the Gap Period {settings.get('gap_years')}, not "
            f"{gap_period(baseline_year)}; rebuild them with `python -m exportspredictor.features`"
        )
    if "exporter_scope" not in settings:
        raise ValueError(
            f"The feature files in {folder} were built without the training scope (ADR 0004); "
            "rebuild them with `python -m exportspredictor.features`"
        )
    if search_results is None:
        if not search.results_path(root).exists():
            raise ValueError(
                f"No hyperparameter search in {search.search_dir(root)}; run `python -m exportspredictor.search`"
            )
        search_results = search.load(root)
    if ranking_results is None:
        if not (loop.loop_dir(root) / loop.RESULTS_FILE).exists():
            raise ValueError(f"No experiment loop results in {loop.loop_dir(root)}; run `python -m exportspredictor.loop`")
        ranking_results = loop.load(root)
    to_iso3 = load_sources(root)[1]

    def flows_of_year(year: int) -> pl.DataFrame:
        """Every Exporter's Flows of one year, read from its own file."""
        return load_year_flows(year, to_iso3, root)

    training = {y: folder / f"training_{y}.parquet" for y in settings["years"] if y < baseline_year}
    grids = {y: folder / f"uk_grid_{y}.parquet" for y in gap_years(baseline_year)}
    trained = _train_and_score(
        training, grids, baseline_year=baseline_year, search_results=search_results,
        n_folds=n_folds, device=device, non_uk_rate=non_uk_rate, seed=seed, row_budget=row_budget,
        calibration=folder / f"training_{baseline_year}.parquet", min_band_rows=min_band_rows,
        cache=features_dir(root).parent / "fold_models" / _file_key(folder / "features.json"),
        ranking_results=ranking_results, year_flows=flows_of_year,
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
        ).collect(),
        trained.yearly,
    )
    country_map = pl.read_parquet(root / "derived" / "country_map.parquet")
    markets = market_lookup(baci.markets(root), country_map)
    metadata = {
        "version": version or new_version(),
        "source": f"BACI {baci.BACI_VERSION}",
        "sources": source_versions(root),
        **trained.metadata,
        "exporter_scope": settings["exporter_scope"],
        "zero_sampling_rate": settings["zero_sampling_rate"],
        "uk_zero_sampling_rate": settings["uk_zero_sampling_rate"],
        "zero_sampling_seed": settings["zero_sampling_seed"],
        # The feature ablation and learning curve that chose the features and rates (2022 only).
        "experiments": experiments.load(root) if experiments.results_path(root).exists() else None,
    }
    test_grid = folder / f"uk_grid_{baseline_year - 1}.parquet"
    v1_folder = v1_frozen or freeze.v1_folder()
    report = _evaluate(
        training, test_grid, trained, metadata, markets=evaluation.markets_with_income(markets, root),
        test_flows=flows_of_year(baseline_year - 1),
        device=device, non_uk_rate=trained.metadata["non_uk_sampling_rate"], seed=seed,
        backtest_plan=backtest_plan,
        backtest_chunks={y: folder / f"training_{y}.parquet" for y in settings["years"]},
        high_income=load_sources(root)[0].high_income,
        backtest_cache=backtests.cache_dir(root),
        constraint_plan=constraint_plan,
        lift_held_out=backtests.lift_exporters(
            freeze.lift_held_out_exporters(v1_folder), scope=settings["exporter_scope"], uk=UK
        ),
        # Where `evaluation.real_test_models` looks for them, so later steps (the v1 comparison) reuse them.
        test_models_cache=evaluation.test_models_dir(root, metadata),
        v1=real_v1_inputs(
            root, v1_folder, scope=settings["exporter_scope"], test_year=baseline_year - 1,
            lift_outcome_year=backtest_plan.lift.outcome_year, ladder_plan=ladder_plan,
        ),
        ladder_cache=ladder_dir(root),
        parts=parts,
        year_flows=flows_of_year,
        earlier_v2=evaluation.V2_REPORTED,
    )
    bundle = Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=history,
        products=baci.products(root),
        markets=markets,
        excluded_products=excluded_products(baci.products(root)),
        potential_range_bands=trained.potential_range_bands,
        evaluation=report,
    )
    if not parts.ppml_benchmark:
        return bundle
    return replace(bundle, ppml_benchmark=_timed_benchmark(ppml.real_sector_flows(root, years=ppml_years), bundle))


def _file_key(path: Path) -> str:
    """A short hash of a file's bytes: cached models are kept apart per feature build."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _train_and_score(
    training: Mapping[int, Chunk],
    grids: Mapping[int, Chunk],
    *,
    baseline_year: int,
    search_results: dict,
    n_folds: int,
    device: str,
    non_uk_rate: float | None,
    seed: int,
    calibration: Chunk,
    min_band_rows: int,
    ranking_results: Mapping,
    year_flows: Callable[[int], pl.DataFrame],
    row_budget: int | None = None,
    cache: Path | None = None,
) -> _Trained:
    """Steps 0–4 of the module docstring. With `cache` (a folder), the fold models are saved
    there keyed by their training settings (`fold_models_dir`), and a re-run reads them.
    `year_flows(year)` gives every Exporter's Flows of a year, for the naive baseline."""
    selected = loop.served_configuration(ranking_results)
    form = selected.form
    print(f"Model form: {form.to_dict()}", flush=True)
    validation_year, test_year = baseline_year - 2, baseline_year - 1
    tuning_years = [y for y in sorted(training) if y < validation_year]
    training_years = [y for y in sorted(training) if y < baseline_year]
    if not tuning_years or validation_year not in training:
        raise ValueError(f"Need training years before {validation_year} and {validation_year} itself")
    # The test year is reserved for the evaluation: tuning must never read it.
    assert test_year not in tuning_years and test_year != validation_year

    # 0. Fit the row budget (ADR 0004) before any training: every matrix built below holds at
    # most the rows of the training years, so derive or check the non-UK rate on those.
    budget = row_budget if row_budget is not None else vram_row_budget(len(FEATURES))
    counts = count_rows([training[y] for y in training_years], uk=UK)
    derived = non_uk_rate is None
    if derived:
        non_uk_rate = fit_non_uk_rate(counts, budget)
    else:
        check_row_budget(counts, non_uk_rate, budget)
    print(f"Training rows: {counts.uk:,} UK + {counts.non_uk:,} other at rate {non_uk_rate}", flush=True)

    # 1. The configuration the experiment loop selected (the test year, baseline_year - 1, was
    # not read), with the round count its selection measure chose on the tuning years' rows at
    # the loop's non-UK rate. The fold models train on more years, so the count is scaled by
    # the ratio of training rows (a heuristic). Deviance chooses neither.
    for name, results in (("hyperparameter search", search_results), ("experiment loop", ranking_results)):
        if results["validation_year"] != validation_year:
            raise ValueError(f"The {name} validated on {results['validation_year']}, not {validation_year}")
    params, selected_rounds = selected.params, selected.rounds
    tuning_counts = count_rows([training[y] for y in tuning_years], uk=UK)
    training_rows, tuning_rows = counts.expected(non_uk_rate), tuning_counts.expected(selected.non_uk_rate)
    rounds = search.scaled_rounds(selected_rounds, training_rows=training_rows, tuning_rows=tuning_rows)
    tuning_seconds = search.seconds(search_results)
    print(f"Selected configuration: {params}, {selected_rounds} rounds scaled to {rounds}", flush=True)

    # 2. Refit the fold models on every year before the Baseline year (or read them from `cache`).
    cached = None
    if cache is not None:
        cached = fold_models_dir(
            cache,
            baseline_year=baseline_year,
            settings={
                "training_years": training_years, "params": params, "rounds": rounds, "n_folds": n_folds,
                "non_uk_rate": non_uk_rate, "seed": seed, "features": FEATURES,
            },
        )
    if cached is not None and (cached / evaluation.TEST_MODELS_FILE).exists():
        saved = evaluation.TestModels.load(cached)
        models, transform = saved.models, saved.transform
        training_seconds = json.loads((cached / _TRAINING_FILE).read_text(encoding="utf-8"))["training_seconds"]
        print(f"Read {n_folds} fold models from {cached} (trained in {training_seconds:.0f} s)", flush=True)
    else:
        started = time.perf_counter()
        chunks = [training[y] for y in training_years]
        transform = FeatureTransform.fit(scan_chunks(chunks), training_years)
        models = train_fold_models(
            chunks, uk=UK, params=params, rounds=rounds, n_folds=n_folds, device=device, seed=seed,
            transform=transform, non_uk_rate=non_uk_rate,
        )
        training_seconds = time.perf_counter() - started
        print(f"Trained {n_folds} fold models in {training_seconds:.0f} s", flush=True)
        if cached is not None:
            cached.mkdir(parents=True, exist_ok=True)
            (cached / _TRAINING_FILE).write_text(json.dumps({"training_seconds": training_seconds}), encoding="utf-8")
            evaluation.TestModels(baseline_year, training_years, models, transform).save(cached)

    # 3. Score every Gap Period year's UK grid out-of-fold, each with that year's features.
    scored = {
        year: _score_grid(evaluation.naive_potential(read_chunk(grids[year]), year_flows(year)), transform, models, form)
        for year in gap_years(baseline_year)
    }
    pairs = scored[baseline_year].select("hs6", "iso3")
    if any(not table.select("hs6", "iso3").equals(pairs) for table in scored.values()):
        raise ValueError("Every Gap Period year must score the same UK pairs (the union of the years' grids)")
    yearly = pl.concat(
        [
            table.select(
                "hs6",
                "iso3",
                year=pl.lit(year, pl.Int16),
                actual_exports_usd="value_usd",
                export_potential_usd="export_potential_usd",
                tariff_imputed="tariff_imputed",
            )
            for year, table in scored.items()
        ]
    ).with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
    baseline = (
        export_gaps(yearly)
        .join(
            scored[baseline_year].select(
                "hs6", "iso3", "tariff_imputed", baseline_export_potential_usd="export_potential_usd"
            ),
            on=["hs6", "iso3"],
        )
        .with_columns(*flag_columns())
        .pipe(with_opportunity_columns)
    )
    # Scenarios start from the Baseline year's features and Export Potential, never the Gap Period mean.
    # `mfn_tariff` (not a feature) is the rate a Scenario restores when it removes a Trade Agreement (ADR 0006).
    # `naive_usd` (not a feature) is the naive baseline a blend takes; a Scenario leaves it unchanged.
    baseline_features = scored[baseline_year].select("hs6", "iso3", "fold", *FEATURES, "mfn_tariff", "naive_usd")
    # The Potential Range (ADR 0007): other Exporters' Baseline-year rows, never the UK's.
    started = time.perf_counter()
    scored_calibration = conformal.score_calibration(
        evaluation.naive_potential(read_chunk(calibration), year_flows(baseline_year)), models, transform, uk=UK,
        form=form,
    )
    bands = conformal.fit_bands(scored_calibration, min_rows=min_band_rows)
    baseline = conformal.with_potential_range(baseline, bands)
    potential_range_report = {
        **conformal.coverage_report(scored_calibration, min_rows=min_band_rows),
        **conformal.uncertain_share(baseline),
    }
    potential_range_seconds = time.perf_counter() - started
    print(
        f"Potential Range from {scored_calibration.height:,} calibration rows in {potential_range_seconds:.0f} s: "
        f"{bands.height} bands, held-out coverage {potential_range_report['coverage']:.1%}",
        flush=True,
    )
    # 4. Blocked Opportunity values: the Scenario Effect of adding a Trade Agreement where none exists.
    baseline = with_blocked_opportunity_values(
        baseline, blocked_opportunity_values(baseline_features, baseline, models, form=form)
    )
    metadata = {
        "artefact_format": ARTEFACT_FORMAT,
        "baseline_year": baseline_year,
        "gap_years": gap_period(baseline_year),
        "validation_year": validation_year,
        "test_year": test_year,
        "tuning_years": tuning_years,
        "training_years": training_years,
        "n_folds": n_folds,
        "features": FEATURES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        # The served form and the experiments that chose it (spec #65, ADR 0008).
        "model_form": form.to_dict(),
        "ranking_experiments": loop.ranking_record(ranking_results),
        "non_uk_sampling_rate": non_uk_rate,
        "non_uk_sampling_rate_derived": derived,
        "row_budget": budget,
        "training_rows": {"uk": counts.uk, "non_uk": counts.non_uk, "expected": round(counts.expected(non_uk_rate))},
        "gpu_check": GPU_CHECK,
        "feature_transform": transform.to_dict(),
        "params": params,
        "model_params": model_params(params, device=device, seed=seed),
        "rounds": rounds,
        "round_scaling": {
            "selected_rounds": selected_rounds,
            "rule": selected.rounds_rule,
            "selection_non_uk_rate": selected.non_uk_rate,
            "tuning_years": tuning_years,
            "training_years": training_years,
            "tuning_rows": round(tuning_rows),
            "training_rows": round(training_rows),
            "ratio": round(training_rows / tuning_rows, 4),
            "rounds": rounds,
        },
        "search": search_results,
        "device": device,
        "xgboost_version": xgb.__version__,
        "tuning_seconds": round(tuning_seconds, 2),
        "training_seconds": round(training_seconds, 2),
        "potential_range": {
            "coverage": conformal.COVERAGE,
            "scale": conformal.SCALE,
            "n_bands": conformal.N_BANDS,
            "min_band_rows": min_band_rows,
            "bands": bands.height,
            "calibration_year": baseline_year,
            "calibration_rows": scored_calibration.height,
            "calibration_exporters": scored_calibration["exporter"].n_unique(),
            "seconds": round(potential_range_seconds, 2),
        },
        # The defaults the tables were built and are ranked with (spec #39 *Artefacts*).
        "thresholds": {
            "blocked_min_share": DEFAULT_MIN_SHARE,
            "blocked_min_usd": DEFAULT_MIN_USD,
            "relative_gap_min_potential_usd": DEFAULT_MIN_POTENTIAL_USD,
            "potential_range_coverage": conformal.COVERAGE,
            "potential_range_bands": conformal.N_BANDS,
            "potential_range_min_band_rows": min_band_rows,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    return _Trained(
        models=models,
        baseline=baseline,
        baseline_features=baseline_features,
        yearly=yearly.sort("hs6", "iso3", "year"),
        potential_range_bands=bands,
        potential_range_report=potential_range_report,
        metadata=metadata,
    )


_TRAINING_FILE = "training.json"


def fold_models_dir(cache: Path, *, baseline_year: int, settings: dict) -> Path:
    """Where fold models trained with `settings` are cached under `cache`, keyed by a hash of them."""
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cache) / f"{baseline_year}_{key}"


def _score_grid(
    grid: pl.DataFrame, transform: FeatureTransform, models: list[xgb.Booster], form: ModelForm
) -> pl.DataFrame:
    """One year's UK grid (with its `naive_usd`) scored out-of-fold (ADR 0001) in the served form,
    keyed by hs6 and iso3 strings."""
    scored = (
        score_out_of_fold(transform.apply(grid), models, uk=UK, form=form)
        .with_columns(hs6=pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String))
        .sort("hs6", "iso3")
    )
    if scored.select("hs6", "iso3").is_duplicated().any() or scored.height != grid.height:
        raise ValueError("A scored grid must hold exactly one row per UK Product–Market pair")
    if scored.select(is_excluded().any()).item():
        raise ValueError("The UK grid holds Excluded Products; they are never scored (ADR 0005)")
    return scored


def _evaluate(
    training: Mapping[int, Chunk],
    test_grid: Chunk,
    trained: _Trained,
    metadata: dict,
    *,
    markets: pl.DataFrame,
    test_flows: pl.DataFrame,
    device: str,
    non_uk_rate: float,
    seed: int,
    backtest_plan: backtests.Plan,
    high_income,
    constraint_plan: constraints.Plan,
    lift_held_out: Sequence[str],
    backtest_chunks: Mapping[int, Chunk] | None = None,
    backtest_cache: Path | None = None,
    v1: comparison.Inputs | None = None,
    test_models_cache: Path | None = None,
    ladder_cache: Path | None = None,
    parts: Parts = Parts(),
    year_flows: Callable[[int], pl.DataFrame] | None = None,
    earlier_v2: dict | None = None,
) -> dict:
    """Evaluate on the test year; with `parts`, run Lift and the Event Backtests (`backtests`)
    and compare the monotone-constrained model with an unconstrained one (`constraints`).

    The test-year fold models are trained with the tuned configuration; the backtest
    models use it too. `backtest_chunks` defaults to `training`. With `v1`, the report
    gains the pre-registered `v1_comparison` (`comparison.compare`), scored by the same
    test-year fold models. With `test_models_cache` (a folder), the test-year models are
    read from it when saved there, or saved there after training. The gain ladder's rung
    models (`v1.ladder`, trained only with `parts.gain_ladder`) are cached under `ladder_cache`
    the same way. `test_flows` are every Exporter's Flows of the test year, for the naive
    baseline (the evaluation's reference, and the blend's other side).

    The test-year models are scored in the served model form, and so is Lift (a blend takes
    each Gap Period year's naive baseline from `year_flows(year)`, every Exporter's Flows of that
    year). The other opt-in parts measure the model alone (the blend's naive side has no policy
    features: it would scale their effects by 1 − w).
    """
    test_year = metadata["test_year"]
    form = ModelForm.from_dict(metadata["model_form"])
    started = time.perf_counter()
    if test_models_cache is not None and (test_models_cache / evaluation.TEST_MODELS_FILE).exists():
        test_models = evaluation.TestModels.load(test_models_cache)
        print(f"Read the {test_year} test models from {test_models_cache}", flush=True)
    else:
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
        if test_models_cache is not None:
            test_models.save(test_models_cache)
        print(f"Trained the {test_year} test models in {time.perf_counter() - started:.0f} s", flush=True)
    report = evaluation.evaluate(
        read_chunk(training[test_year]),
        read_chunk(test_grid),
        test_models=test_models,
        baseline=baseline_year_scores(trained.yearly, metadata["baseline_year"]),
        baseline_models=trained.models,
        baseline_transform=FeatureTransform.from_dict(metadata["feature_transform"]),
        markets=markets,
        uk=UK,
        settings=evaluation.report_settings(metadata),
        flows=test_flows,
        form=form,
    )
    report["evaluation_seconds"] = round(time.perf_counter() - started, 2)
    print(f"Evaluated on {test_year} in {report['evaluation_seconds']:.0f} s", flush=True)
    if parts.event_backtests:
        report |= backtests.run(
            backtest_chunks or training, backtest_plan, uk=UK, params=metadata["params"], rounds=metadata["rounds"],
            device=device, non_uk_rate=non_uk_rate, seed=seed, high_income=high_income, test_models=test_models,
            cache=backtest_cache, lift_held_out=lift_held_out, form=form, year_flows=year_flows,
        )
        print(f"Backtests in {report['event_backtest']['seconds']:.0f} s", flush=True)
    if parts.constraint_comparison:
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
    report["potential_range"] = {
        **trained.potential_range_report, "text": conformal.coverage_text(trained.potential_range_report)
    }
    if v1 is not None:
        if not parts.gain_ladder:
            v1 = replace(v1, ladder=None)
        report["v1_comparison"] = _v1_comparison(
            v1, test_models, report, metadata, training=training, lift_settings=backtest_plan.lift, device=device,
            non_uk_rate=non_uk_rate, seed=seed, ladder_cache=ladder_cache,
        )
    # The test year beside v2's reported figures (`earlier_v2`, real runs only) and v1's.
    report["earlier_versions"] = evaluation.earlier_versions(report, v2=earlier_v2)
    return report


def _v1_comparison(
    v1: comparison.Inputs,
    test_models: evaluation.TestModels,
    report: dict,
    metadata: dict,
    *,
    training: Mapping[int, Chunk],
    lift_settings: backtests.LiftSettings,
    device: str,
    non_uk_rate: float,
    seed: int,
    ladder_cache: Path | None,
) -> dict:
    """The pre-registered `v1_comparison` block (`comparison.compare`); its Lift and Event Backtest
    secondaries only when the report holds them.

    v2 is scored by its test-year fold models (`test_models`, rung C), in the served model form. With `v1.ladder`, rungs
    A and B are trained like them (or read from `ladder_cache`) with v1's configuration, and
    score the UK common rows out-of-fold.
    """
    test_year = test_models.test_year
    ladder = None
    if v1.ladder is not None:
        ladder = {}
        for name, features in (("A", comparison.V1_FEATURES), ("B", FEATURES)):
            rung = _rung_models(
                training, name=name, features=features, plan=v1.ladder, test_year=test_year,
                n_folds=metadata["n_folds"], device=device, non_uk_rate=non_uk_rate, seed=seed, cache=ladder_cache,
            )
            ladder[name] = evaluation.score_rows(v1.rows["uk"], rung.models, rung.transform, UK, features=rung.features)
    started = time.perf_counter()
    form = ModelForm.from_dict(metadata["model_form"])
    test_flows = v1.flows.filter(pl.col("year") == test_year) if "year" in v1.flows.columns else v1.flows
    block = comparison.compare(
        v1.frozen,
        v2={
            name: evaluation.score_rows(
                evaluation.naive_potential(rows, test_flows), test_models.models, test_models.transform, UK, form=form
            )
            for name, rows in v1.rows.items()
        },
        flows=v1.flows,
        lift_v2=report.get("lift"),
        event_v2=report["event_backtest"]["events"][0] if "event_backtest" in report else None,
        lift_settings=lift_settings,
        scope=metadata["exporter_scope"],
        uk=UK,
        test_year=test_year,
        seed=seed,
        name=comparison.version_name(report),
        ladder=ladder,
        ladder_plan=v1.ladder,
    )
    print(f"v1 comparison in {time.perf_counter() - started:.0f} s: {block['verdict']['text']}", flush=True)
    return block


def _rung_models(
    training: Mapping[int, Chunk],
    *,
    name: str,
    features: Sequence[str],
    plan: comparison.LadderPlan,
    test_year: int,
    n_folds: int,
    device: str,
    non_uk_rate: float,
    seed: int,
    cache: Path | None,
) -> evaluation.TestModels:
    """One gain-ladder rung's test-year fold models: v1's configuration on v2's rows, with `features`."""
    years = [y for y in sorted(training) if y < test_year]
    settings = {
        "rung": name, "features": list(features), "params": plan.params, "rounds": plan.rounds, "n_folds": n_folds,
        "non_uk_rate": non_uk_rate, "seed": seed, "test_year": test_year, "training_years": years,
    }
    folder = None if cache is None else Path(cache) / f"{name}_{_settings_key(settings)}"
    if folder is not None and (folder / evaluation.TEST_MODELS_FILE).exists():
        print(f"Read ladder rung {name} from {folder}", flush=True)
        return evaluation.TestModels.load(folder)
    started = time.perf_counter()
    models = evaluation.train_test_models(
        {y: training[y] for y in years}, test_year=test_year, uk=UK, params=plan.params, rounds=plan.rounds,
        n_folds=n_folds, device=device, non_uk_rate=non_uk_rate, seed=seed, features=features,
    )
    print(f"Trained ladder rung {name} in {time.perf_counter() - started:.0f} s", flush=True)
    if folder is not None:
        models.save(folder)
    return models


def _settings_key(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def ladder_dir(root: Path | None = None) -> Path:
    """Where the gain ladder's rung models are cached, per feature build."""
    folder = features_dir(root)
    return folder.parent / "ladder_models" / _file_key(folder / "features.json")


def real_v1_inputs(
    root: Path,
    v1_frozen: Path,
    *,
    scope: Sequence[str],
    test_year: int,
    lift_outcome_year: int,
    ladder_plan: comparison.LadderPlan | None,
) -> comparison.Inputs:
    """The frozen v1 (checked against its manifest) and v2's features on the common rows, from the real data root.

    The features are built as `features.build` builds them: from the test year's Flows and those
    `GROWTH_YEARS` earlier, with the saved proximity. The comparison's Flows are every Exporter's,
    of the test year and the Lift outcome year.
    """
    frozen = comparison.load_frozen(v1_frozen)
    sources, to_iso3 = load_sources(root)
    sources = replace(sources, proximity=pl.read_parquet(features_dir(root) / "proximity.parquet"))
    years = sorted({test_year, test_year - GROWTH_YEARS, lift_outcome_year})
    flows = pl.concat([load_year_flows(year, to_iso3, root) for year in years])
    rows = comparison.feature_rows(frozen, scope=scope, uk=UK, flows=flows, sources=sources)
    outcome = flows.filter(pl.col("year").is_in([test_year, lift_outcome_year]))
    return comparison.Inputs(frozen=frozen, rows=rows, flows=outcome, ladder=ladder_plan)


def add_v1_comparison(
    root: Path | None = None,
    artefacts_root: Path | None = None,
    *,
    device: str = "cuda",
    v1_frozen: Path | None = None,
    ladder_plan: comparison.LadderPlan = comparison.REAL_LADDER,
    backtest_plan: backtests.Plan = backtests.REAL_PLAN,
    version: str | None = None,
) -> Path:
    """Add the pre-registered `v1_comparison` to the newest artefact, saved as a new version.

    The source version is not changed: its models and tables are copied file for file, with the
    evaluation report gaining the block. v2's test-year models are read from the cache
    (`evaluation.real_test_models`); the ladder's rungs are read or trained (`ladder_dir`).
    """
    from exportspredictor import artefacts as store
    from exportspredictor.paths import artefacts_dir

    root = Path(root or data_dir())
    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    metadata, report = bundle.metadata, dict(bundle.evaluation or {})
    if "lift" not in report or "event_backtest" not in report:
        raise ValueError(f"The artefact {bundle.version} has no Lift or Event Backtest to compare with v1's")
    test_models = evaluation.real_test_models(bundle, root, device=device)
    folder = features_dir(root)
    years = json.loads((folder / "features.json").read_text(encoding="utf-8"))["years"]
    v1 = real_v1_inputs(
        root, v1_frozen or freeze.v1_folder(artefacts_root), scope=metadata["exporter_scope"],
        test_year=test_models.test_year, lift_outcome_year=backtest_plan.lift.outcome_year, ladder_plan=ladder_plan,
    )
    report["v1_comparison"] = _v1_comparison(
        v1, test_models, report, metadata,
        training={y: folder / f"training_{y}.parquet" for y in years if y < test_models.test_year},
        lift_settings=backtest_plan.lift, device=device, non_uk_rate=metadata["non_uk_sampling_rate"],
        seed=metadata["model_params"].get("seed", 0), ladder_cache=ladder_dir(root),
    )
    copied = {**metadata, "version": version or new_version(), "copied_from": bundle.version}
    return store.copy_version(source, artefacts_root, metadata=copied, evaluation=report)


def _history(uk_flows: pl.DataFrame, yearly: pl.DataFrame) -> pl.DataFrame:
    """UK Actual Exports per pair and year, with each Gap Period year's Export Potential and Export Gap.

    Gap Period years come from `yearly` (every scored pair, $0 Actual Exports included), so a
    pair's yearly Export Gaps average to its Export Gap. Earlier years keep non-zero Actual
    Exports only, with no score. Excluded Products are never included. `tariff_imputed` marks
    a year whose Applied Tariff is carried forward from 2021 (ADR 0006).
    """
    gap = yearly.select(
        "hs6", "iso3", pl.col("year").cast(pl.Int16), "actual_exports_usd", "export_potential_usd", "export_gap_usd",
        "tariff_imputed",
    )
    earlier = (
        uk_flows.filter(pl.col("value_usd") > 0, ~pl.col("year").is_in(gap["year"].unique().to_list()))
        .pipe(drop_excluded)
        .select(
            pl.col("hs6").cast(pl.String),
            iso3=pl.col("importer").cast(pl.String),
            year=pl.col("year").cast(pl.Int16),
            actual_exports_usd=pl.col("value_usd"),
            export_potential_usd=pl.lit(None, pl.Float64),
            export_gap_usd=pl.lit(None, pl.Float64),
            tariff_imputed=pl.col("year") > LAST_TRAINS_YEAR,
        )
    )
    return pl.concat([earlier, gap]).sort("hs6", "iso3", "year")


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


_RANGE_COLUMNS = ["potential_range_low_usd", "potential_range_high_usd", "uncertain_opportunity"]


def rerange(
    bundle: Artefacts,
    calibration: Chunk,
    year_flows: pl.DataFrame,
    *,
    version: str | None = None,
    min_band_rows: int | None = None,
) -> Artefacts:
    """A new version of a saved bundle with its Potential Range recomputed as `conformal` now builds it; no retraining.

    Only the bands, each UK pair's Potential Range and Uncertain Opportunity flag, and the range's
    report and metadata change; the fold models and Export Potential are reused. `calibration` is the
    Baseline year's training rows (untransformed, as `features.training_rows` builds them) and
    `year_flows` every Exporter's Flows of that year, for the naive baseline a Blend takes. Bands keep
    the bundle's minimum size unless `min_band_rows` is given.
    """
    metadata = bundle.metadata
    min_rows = metadata["potential_range"]["min_band_rows"] if min_band_rows is None else min_band_rows
    started = time.perf_counter()
    scored = conformal.score_calibration(
        evaluation.naive_potential(read_chunk(calibration), year_flows), bundle.fold_models,
        FeatureTransform.from_dict(metadata["feature_transform"]), uk=UK, form=bundle.model_form,
    )
    bands = conformal.fit_bands(scored, min_rows=min_rows)
    baseline = conformal.with_potential_range(bundle.baseline.drop(_RANGE_COLUMNS), bands).select(
        bundle.baseline.columns
    )
    report = {**conformal.coverage_report(scored, min_rows=min_rows), **conformal.uncertain_share(baseline)}
    seconds = time.perf_counter() - started
    print(
        f"Potential Range from {scored.height:,} calibration rows in {seconds:.0f} s: "
        f"{bands.height} bands, held-out coverage {report['coverage']:.1%}",
        flush=True,
    )
    range_metadata = {
        **metadata["potential_range"],
        "scale": conformal.SCALE,
        "min_band_rows": min_rows,
        "bands": bands.height,
        "calibration_rows": scored.height,
        "calibration_exporters": scored["exporter"].n_unique(),
        "seconds": round(seconds, 2),
    }
    return replace(
        bundle,
        metadata={**metadata, "version": version or new_version(), "refreshed_from": bundle.version,
                  "potential_range": range_metadata},
        baseline=baseline,
        potential_range_bands=bands,
        evaluation=None if bundle.evaluation is None else {
            **bundle.evaluation, "potential_range": {**report, "text": conformal.coverage_text(report)}
        },
    )


def _rerange_latest(root: Path | None = None) -> Path:
    """Save `rerange` of the newest real bundle, from the Baseline year's feature file, with its model card."""
    from exportspredictor import artefacts
    from exportspredictor.paths import artefacts_dir

    root = Path(root or data_dir())
    bundle = artefacts.load(artefacts.latest_folder(artefacts_dir()))
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    if settings.get("features") != bundle.metadata["features"]:
        raise ValueError(f"The feature files in {folder} hold other features than version {bundle.version}")
    year = bundle.baseline_year
    ranged = rerange(bundle, folder / f"training_{year}.parquet", load_year_flows(year, load_sources(root)[1], root))
    saved = artefacts.save(ranged, artefacts_dir())
    governance.write_model_card(ranged, saved)
    return saved


def parse_parts(argv: Sequence[str]) -> Parts:
    """The opt-in parts a command line asks for: `--event-backtests`, `--constraint-comparison`,
    `--gain-ladder`, `--ppml-benchmark`, or `--every-part` for all four. None of them: the default run."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m exportspredictor.pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands that copy the newest saved version instead of training:\n"
            "  python -m exportspredictor.pipeline refresh   fresh metadata, in seconds\n"
            "  python -m exportspredictor.pipeline rerange   Potential Range recomputed, a few minutes on the CPU\n"
            "  python -m exportspredictor.pipeline compare   v1 comparison and gain ladder added"
        ),
    )
    parser.add_argument("--event-backtests", action="store_true", help="also run Lift and the Event Backtests")
    parser.add_argument("--constraint-comparison", action="store_true", help="also run the monotone-constraint comparison")
    parser.add_argument("--gain-ladder", action="store_true", help="also train the v1 comparison's gain ladder")
    parser.add_argument("--ppml-benchmark", action="store_true", help="also fit the PPML benchmark (CPU, no training)")
    parser.add_argument("--every-part", action="store_true", help="all four: the full run")
    args = parser.parse_args(list(argv))
    if args.every_part:
        return Parts.every()
    return Parts(
        event_backtests=args.event_backtests, constraint_comparison=args.constraint_comparison,
        gain_ladder=args.gain_ladder, ppml_benchmark=args.ppml_benchmark,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m exportspredictor.pipeline` trains and saves a real bundle: by default the fold and
    test-year models only (about 2.5 h on GPU, less when either is cached); `parse_parts` names the
    switches that add the Event Backtests, the constraint comparison, the gain ladder and the PPML benchmark.

    `python -m exportspredictor.pipeline refresh` instead saves a refreshed copy of the newest
    saved bundle (`refresh`), in seconds. `python -m exportspredictor.pipeline rerange` saves a
    copy of the newest bundle with its Potential Range recomputed (`rerange`), in a few minutes
    on the CPU. `python -m exportspredictor.pipeline compare` saves a
    copy of the newest bundle with the v1 comparison and gain ladder added (`add_v1_comparison`).
    """
    import sys

    from exportspredictor import artefacts
    from exportspredictor.paths import artefacts_dir

    argv = sys.argv[1:] if argv is None else argv
    if list(argv) == ["compare"]:
        print(add_v1_comparison(), flush=True)
        return
    if list(argv) == ["refresh"]:
        source = artefacts.latest_folder(artefacts_dir())
        bundle = refresh(artefacts.load(source))
        print(json.dumps({k: bundle.metadata[k] for k in ("version", "refreshed_from", "sources")}), flush=True)
        # Copied file for file, so files this step does not know about (e.g. the model card) survive.
        print(artefacts.copy_version(source, artefacts_dir(), metadata=bundle.metadata), flush=True)
        return
    if list(argv) == ["rerange"]:
        print(_rerange_latest(), flush=True)
        return
    bundle = build_real_artefacts(parts=parse_parts(argv))
    summary = {k: bundle.metadata[k] for k in ("version", "params", "rounds", "tuning_seconds", "training_seconds")}
    print(json.dumps(summary), flush=True)
    folder = artefacts.save(bundle, artefacts_dir())
    print(folder, flush=True)
    # The notebook reads the model card and writes none (#88), so the run that saves the version writes it.
    print(governance.write_model_card(bundle, folder), flush=True)


if __name__ == "__main__":
    main()
