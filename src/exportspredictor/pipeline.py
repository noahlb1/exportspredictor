"""From a world of Flows to a servable artefact bundle.

`build_artefacts` is the one call the notebook makes to train and score: it builds
features, fits fold models on years before the Baseline year (time split), scores every
UK pair in the Baseline year out-of-fold, and assembles the Baseline table.
"""

from datetime import UTC, datetime

import polars as pl

from exportspredictor.artefacts import Artefacts, new_version
from exportspredictor.features import FEATURES, MONOTONE_CONSTRAINTS, build_features
from exportspredictor.fixture import UK, World
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models


def build_artefacts(
    world: World,
    *,
    version: str | None = None,
    baseline_year: int | None = None,
    n_folds: int = N_FOLDS,
    device: str = "cpu",
    source: str = "fixture",
) -> Artefacts:
    """Train fold models and score the Baseline for every UK Product–Market pair.

    `device` is "cpu" for the fixture (CUDA start-up dwarfs its training time) and
    "cuda" for real data.
    """
    features = build_features(world.flows, world.policy)
    baseline_year = baseline_year or int(features["year"].max())
    training = features.filter(pl.col("year") < baseline_year)
    models = train_fold_models(training, uk=UK, n_folds=n_folds, device=device)

    scored = score_out_of_fold(features.filter(pl.col("year") == baseline_year), models, uk=UK)
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
    training_years = sorted(training["year"].unique().to_list())
    metadata = {
        "version": version or new_version(),
        "source": source,
        "baseline_year": baseline_year,
        "training_years": training_years,
        "n_folds": n_folds,
        "features": FEATURES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return Artefacts(
        metadata=metadata,
        fold_models=models,
        baseline=baseline,
        products=world.products,
        markets=world.markets,
    )
