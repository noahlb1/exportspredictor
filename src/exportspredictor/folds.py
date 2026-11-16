"""Out-of-fold training and scoring for UK pairs (ADR 0001).

UK Product–Market pairs are split into folds by a deterministic hash of the pair, so all
years of a pair share a fold. Fold model k trains on every non-UK row plus the UK pairs
outside fold k, and each UK pair is scored only by the model that held it out.
"""

import hashlib

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES, MONOTONE_CONSTRAINTS

N_FOLDS = 5

_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "max_depth": 3,
    "eta": 0.1,
    "monotone_constraints": "(" + ",".join(str(MONOTONE_CONSTRAINTS.get(f, 0)) for f in FEATURES) + ")",
}
_ROUNDS = 100


def assign_folds(pairs: pl.DataFrame, n_folds: int = N_FOLDS) -> pl.DataFrame:
    """Add a `fold` column to rows keyed by `hs6` and `iso3`. Deterministic across runs."""

    def fold_of(hs6: str, iso3: str) -> int:
        digest = hashlib.sha256(f"{hs6}|{iso3}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % n_folds

    folds = [fold_of(hs6, iso3) for hs6, iso3 in pairs.select("hs6", "iso3").iter_rows()]
    return pairs.with_columns(fold=pl.Series(folds, dtype=pl.Int8))


def train_fold_models(
    features: pl.DataFrame,
    uk: str,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    seed: int = 0,
) -> list[xgb.Booster]:
    """Train one model per fold on all non-UK rows plus the UK pairs outside that fold."""
    rows = _with_uk_fold(features, uk, n_folds)
    models = []
    for k in range(n_folds):
        train = rows.filter(pl.col("fold") != k)
        matrix = xgb.DMatrix(
            train.select(FEATURES).to_numpy(), label=train["value_usd"].to_numpy(), feature_names=FEATURES
        )
        params = {**_PARAMS, "device": device, "seed": seed}
        models.append(xgb.train(params, matrix, num_boost_round=_ROUNDS))
    return models


def score_out_of_fold(features: pl.DataFrame, models: list[xgb.Booster], uk: str) -> pl.DataFrame:
    """Return UK rows with `export_potential_usd`, each scored by the fold model that held it out."""
    rows = _with_uk_fold(features.filter(pl.col("exporter") == uk), uk, len(models))
    potential = np.empty(rows.height)
    for k, model in enumerate(models):
        mask = (rows["fold"] == k).to_numpy()
        if mask.any():
            matrix = xgb.DMatrix(rows.filter(pl.col("fold") == k).select(FEATURES).to_numpy(), feature_names=FEATURES)
            potential[mask] = model.predict(matrix)
    return rows.drop("fold").with_columns(export_potential_usd=pl.Series(potential))


def _with_uk_fold(features: pl.DataFrame, uk: str, n_folds: int) -> pl.DataFrame:
    """Give UK rows their pair's fold; non-UK rows get -1, so every fold model trains on them."""
    is_uk = pl.col("exporter") == uk
    uk_rows = assign_folds(features.filter(is_uk).rename({"importer": "iso3"}), n_folds).rename(
        {"iso3": "importer"}
    )
    other = features.filter(~is_uk).with_columns(fold=pl.lit(-1, dtype=pl.Int8))
    return pl.concat([other, uk_rows.select(other.columns)])
