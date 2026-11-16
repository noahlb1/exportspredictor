"""Out-of-fold training and scoring for UK pairs (ADR 0001).

UK Product–Market pairs are split into folds by a deterministic hash of the pair, so all
years of a pair share a fold. Fold model k trains on every non-UK row plus the UK pairs
outside fold k, and each UK pair is scored only by the model that held it out. Each model
records the fold it held out (`held_out_fold` attribute), and scoring returns each pair's
`fold`, so Scenarios and explanations can use the same model as the Baseline.

Training data comes in chunks (one feature table per year, see `training`), so the real
~165 M rows never sit in memory at once.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES, FeatureTransform
from exportspredictor.training import (
    DEFAULT_ROUNDS,
    Chunk,
    model_params,
    preparer,
    training_matrix,
)

N_FOLDS = 5


def assign_folds(pairs: pl.DataFrame, n_folds: int = N_FOLDS) -> pl.DataFrame:
    """Add a `fold` column to rows keyed by `hs6` and `iso3`. Deterministic across runs.

    The fold depends only on the pair's codes (SHA-256), never on the year or on row order.
    """

    def fold_of(hs6: str, iso3: str) -> int:
        digest = hashlib.sha256(f"{hs6}|{iso3}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % n_folds

    unique = pairs.select("hs6", "iso3").unique()
    folds = [fold_of(str(hs6), str(iso3)) for hs6, iso3 in unique.iter_rows()]
    lookup = unique.with_columns(fold=pl.Series(folds, dtype=pl.Int8))
    return pairs.join(lookup, on=["hs6", "iso3"], how="left", maintain_order="left")


def train_fold_models(
    chunks: Sequence[Chunk],
    *,
    uk: str,
    params: dict | None = None,
    rounds: int = DEFAULT_ROUNDS,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    seed: int = 0,
    transform: FeatureTransform | None = None,
    non_uk_rate: float = 1.0,
) -> list[xgb.Booster]:
    """Train one model per fold on all non-UK rows plus the UK pairs outside that fold.

    `chunks` are feature tables (or Parquet paths), read one at a time. A `weight` column
    weights each row (sampled zero cells carry 1/rate). `non_uk_rate` < 1 trains on a
    reweighted hash sample of non-UK rows; UK rows are always kept in full. The returned
    models are set to predict on CPU, as the API does.
    """
    models = []
    for k in range(n_folds):
        keep = _outside_fold(uk, n_folds, k)
        prepare = preparer(uk=uk, transform=transform, non_uk_rate=non_uk_rate, seed=seed, keep=keep)
        matrix = training_matrix(chunks, prepare)
        model = xgb.train(model_params(params, device=device, seed=seed), matrix, num_boost_round=rounds)
        del matrix
        model.set_param({"device": "cpu"})
        model.set_attr(held_out_fold=str(k))
        models.append(model)
    return models


def score_out_of_fold(features: pl.DataFrame, models: list[xgb.Booster], uk: str) -> pl.DataFrame:
    """UK rows with their pair's `fold` and `export_potential_usd` from the model that held it out."""
    rows = _with_fold(features.filter(pl.col("exporter").cast(pl.String) == uk), len(models))
    fold = rows["fold"].to_numpy()
    potential = np.empty(rows.height)
    for k, model in enumerate(models):
        if model.attr("held_out_fold") not in (None, str(k)):
            raise ValueError(f"Model {k} held out fold {model.attr('held_out_fold')}, not {k}")
        mask = fold == k
        if mask.any():
            data = rows.filter(pl.Series(mask)).select(FEATURES).to_numpy().astype(np.float32)
            potential[mask] = model.inplace_predict(data)
    return rows.with_columns(export_potential_usd=pl.Series(potential))


def _with_fold(rows: pl.DataFrame, n_folds: int) -> pl.DataFrame:
    return assign_folds(rows.rename({"importer": "iso3"}), n_folds).rename({"iso3": "importer"})


def _outside_fold(uk: str, n_folds: int, k: int):
    """Row filter for fold model k: every non-UK row, and UK rows whose pair is not in fold k."""

    def keep(table: pl.DataFrame) -> pl.DataFrame:
        is_uk = pl.col("exporter").cast(pl.String) == uk
        uk_rows = _with_fold(table.filter(is_uk), n_folds).filter(pl.col("fold") != k).drop("fold")
        return pl.concat([table.filter(~is_uk), uk_rows])

    return keep
