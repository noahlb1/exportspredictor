"""From a world of Flows to a servable artefact bundle.

`build_artefacts` is the one call the notebook makes to train and score: it builds
training rows (non-zero Flows plus sampled relevant zeros) and their features for the
years before the Baseline year (time split), fits the feature transform and the fold
models on those years only, scores the Baseline year's full UK Product × Market grid
out-of-fold, and assembles the Baseline table.
"""

from datetime import UTC, datetime

import polars as pl

from exportspredictor.artefacts import Artefacts, new_version
from exportspredictor.features import (
    FEATURES,
    MONOTONE_CONSTRAINTS,
    FeatureTransform,
    build_features,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK, World
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models

# The fixture world has only a few dozen relevant zeros, so it keeps half of them. Real
# data uses `features.ZERO_SAMPLING_RATE`, chosen in the notebook.
FIXTURE_ZERO_SAMPLING_RATE = 0.5


def build_artefacts(
    world: World,
    *,
    version: str | None = None,
    baseline_year: int | None = None,
    n_folds: int = N_FOLDS,
    device: str = "cpu",
    source: str = "fixture",
    zero_sampling_rate: float = FIXTURE_ZERO_SAMPLING_RATE,
    seed: int = 0,
) -> Artefacts:
    """Train fold models and score the Baseline for every UK Product–Market pair.

    `device` is "cpu" for the fixture (CUDA start-up dwarfs its training time) and
    "cuda" for real data. `zero_sampling_rate` is the share of relevant zero cells kept
    for training; it is recorded in the metadata.
    """
    flows, sources = world.flows, world.sources
    baseline_year = baseline_year or int(flows["year"].max())
    training_flows = flows.filter(pl.col("year") < baseline_year)
    training_years = sorted(training_flows["year"].unique().to_list())
    rows = training_rows(training_flows, rate=zero_sampling_rate, seed=seed)
    raw_training = build_features(rows, training_flows, sources)
    transform = FeatureTransform.fit(raw_training, training_years)
    models = train_fold_models(transform.apply(raw_training), uk=UK, n_folds=n_folds, device=device, seed=seed)

    baseline_flows = flows.filter(pl.col("year") == baseline_year)
    grid = transform.apply(build_features(uk_grid(baseline_flows, UK), baseline_flows, sources))
    scored = score_out_of_fold(grid, models, uk=UK)
    baseline = (
        scored.select(
            "hs6",
            iso3="importer",
            export_potential_usd="export_potential_usd",
            actual_exports_usd="value_usd",
        )
        .with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
        .sort("hs6", "iso3")
    )
    metadata = {
        "version": version or new_version(),
        "source": source,
        "baseline_year": baseline_year,
        "training_years": training_years,
        "n_folds": n_folds,
        "features": FEATURES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "zero_sampling_rate": zero_sampling_rate,
        "zero_sampling_seed": seed,
        "feature_transform": transform.to_dict(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    return Artefacts(
        metadata=metadata,
        fold_models=models,
        baseline=baseline,
        products=world.products,
        markets=world.markets,
    )
