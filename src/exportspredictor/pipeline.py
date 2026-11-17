"""From feature tables to a servable artefact bundle, for the fixture and for real data.

Both paths run the same steps (`_train_and_score`), so the fixture tests what real data runs:

1. **Tune** on the validation year (Baseline year − 2, i.e. 2022): one model per
   hyperparameter configuration, trained on the years before it, with early stopping on
   it. The test year (Baseline year − 1) is never read here.
2. **Refit** the fold models (ADR 0001) on every year before the Baseline year, with the
   best configuration and its round count; the feature transform is fitted on the same years.
3. **Score** the Baseline year's full UK Product × Market grid out-of-fold and assemble the
   Baseline table, the Scenario inputs and the Actual Exports history.
4. **Evaluate** on the test year (Baseline year − 1): fold models refitted on the years
   before it with the tuned configuration score it, against a naive baseline
   (`evaluation`). The report is saved in the artefact as `evaluation.json`.

`build_artefacts(world)` runs this on the fixture world (CPU, seconds).
`build_real_artefacts()` runs it on the per-year feature files of `features.build()`
(GPU); `uv run python -m exportspredictor.pipeline` runs it and saves the artefact folder.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import baci, evaluation
from exportspredictor.artefacts import Artefacts, new_version
from exportspredictor.features import (
    FEATURES,
    MONOTONE_CONSTRAINTS,
    FeatureTransform,
    build_features,
    features_dir,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK, World
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models
from exportspredictor.lookups import market_lookup
from exportspredictor.paths import data_dir
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
    training = {}
    for year in sorted(flows["year"].unique().to_list()):
        if year < baseline_year:
            year_flows = flows.filter(pl.col("year") == year)
            rows = training_rows(year_flows, rate=zero_sampling_rate, seed=seed)
            training[year] = build_features(rows, year_flows, sources)
    grids = {}
    for year in (baseline_year - 1, baseline_year):
        year_flows = flows.filter(pl.col("year") == year)
        grids[year] = build_features(uk_grid(year_flows, UK), year_flows, sources)
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
        training, grids[baseline_year - 1], trained, metadata, markets=world.markets, device=device, non_uk_rate=1.0, seed=seed
    )
    return Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=_history(flows.filter((pl.col("exporter") == UK) & recent)),
        products=world.products,
        markets=world.markets,
        evaluation=report,
    )


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
) -> Artefacts:
    """Tune, train and score on the real per-year feature files written by `features.build()`.

    Years are read one at a time (`training`), so host memory stays far below 32 GB.
    """
    root = Path(root or data_dir())
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
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
        **trained.metadata,
        "zero_sampling_rate": settings["zero_sampling_rate"],
        "zero_sampling_seed": settings["zero_sampling_seed"],
    }
    test_grid = folder / f"uk_grid_{baseline_year - 1}.parquet"
    report = _evaluate(training, test_grid, trained, metadata, markets=markets, device=device, non_uk_rate=non_uk_rate, seed=seed)
    return Artefacts(
        metadata=metadata,
        fold_models=trained.models,
        baseline=trained.baseline,
        baseline_features=trained.baseline_features,
        history=history,
        products=baci.products(root),
        markets=markets,
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
    baseline = scored.select(
        "hs6",
        "iso3",
        "export_potential_usd",
        actual_exports_usd="value_usd",
    ).with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
    metadata = {
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
        baseline_features=scored.select("hs6", "iso3", "fold", *FEATURES),
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
) -> dict:
    """Train the test-year fold models with the tuned configuration and evaluate them on the test year."""
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
    return report


def _history(uk_flows: pl.DataFrame) -> pl.DataFrame:
    """Non-zero UK Actual Exports per pair and year."""
    return (
        uk_flows.filter(pl.col("value_usd") > 0)
        .select(
            pl.col("hs6").cast(pl.String),
            iso3=pl.col("importer").cast(pl.String),
            year=pl.col("year").cast(pl.Int16),
            actual_exports_usd=pl.col("value_usd"),
        )
        .sort("hs6", "iso3", "year")
    )


def main() -> None:
    from exportspredictor import artefacts
    from exportspredictor.paths import artefacts_dir

    bundle = build_real_artefacts()
    summary = {k: bundle.metadata[k] for k in ("version", "params", "rounds", "tuning_seconds", "training_seconds")}
    print(json.dumps(summary), flush=True)
    print(artefacts.save(bundle, artefacts_dir()), flush=True)


if __name__ == "__main__":
    main()
