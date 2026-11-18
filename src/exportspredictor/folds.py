"""Out-of-fold training and scoring for UK pairs (ADR 0001).

UK Product–Market pairs are split into folds by a deterministic hash of the pair, so all
years of a pair share a fold. Fold model k trains on every non-UK row plus the UK pairs
outside fold k, and each UK pair is scored only by the model that held it out. Each model
records the fold it held out (`held_out_fold` attribute), and scoring returns each pair's
`fold`, so Scenarios and explanations can use the same model as the Baseline.

Training data comes in chunks (one feature table per year, see `training`), so the real
rows never sit in host memory at once. UK rows, Zero Cells included, are never sampled
(ADR 0004), so fold model k trains on every UK pair outside fold k, zeros and all.
"""

import hashlib
from collections.abc import Sequence

import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES, FeatureTransform
from exportspredictor.prediction import export_potential
from exportspredictor.training import (
    DEFAULT_ROUNDS,
    Chunk,
    model_params,
    preparer,
    training_matrix,
)

N_FOLDS = 5


UK_PAIR_KEYS = ("hs6", "iso3")
# Other Exporters' rows (the Potential Range's calibration rows, ADR 0007) are hashed with
# their Exporter too, so one Product–Market pair's Exporters spread over the folds.
EXPORTER_PAIR_KEYS = ("exporter", "hs6", "iso3")


def assign_folds(pairs: pl.DataFrame, n_folds: int = N_FOLDS, keys: Sequence[str] = UK_PAIR_KEYS) -> pl.DataFrame:
    """Add a `fold` column to rows keyed by `keys` (UK pairs: `hs6` and `iso3`). Deterministic across runs.

    The fold depends only on the key codes (SHA-256 of "a|b|..."), never on the year or on row order.
    """

    def fold_of(codes: tuple) -> int:
        digest = hashlib.sha256("|".join(str(code) for code in codes).encode()).digest()
        return int.from_bytes(digest[:8], "big") % n_folds

    keys = list(keys)
    unique = pairs.select(keys).unique()
    folds = [fold_of(codes) for codes in unique.iter_rows()]
    lookup = unique.with_columns(fold=pl.Series(folds, dtype=pl.Int8))
    return pairs.join(lookup, on=keys, how="left", maintain_order="left")


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
    features: Sequence[str] = FEATURES,
) -> list[xgb.Booster]:
    """Train one model per fold on all non-UK rows plus the UK pairs outside that fold.

    `chunks` are feature tables (or Parquet paths), read one at a time. A `weight` column
    weights each row (sampled zero cells carry 1/rate). `non_uk_rate` < 1 trains on a
    reweighted hash sample of non-UK rows; UK rows are always kept in full. The returned
    models are set to predict on CPU, as the API does. `features` are the model inputs, in
    order (a rung of the v1/v2 gain ladder trains on v1's).
    """
    models = []
    for k in range(n_folds):
        keep = outside_fold(uk, n_folds, k)
        prepare = preparer(uk=uk, transform=transform, non_uk_rate=non_uk_rate, seed=seed, keep=keep)
        matrix = training_matrix(chunks, prepare, device=device, features=features)
        model = xgb.train(
            model_params(params, device=device, seed=seed, features=features), matrix, num_boost_round=rounds
        )
        del matrix
        model.set_param({"device": "cpu"})
        model.set_attr(held_out_fold=str(k))
        models.append(model)
    return models


def score_out_of_fold(
    features: pl.DataFrame, models: list[xgb.Booster], uk: str, *, inputs: Sequence[str] = FEATURES
) -> pl.DataFrame:
    """UK rows with their pair's `fold` and `export_potential_usd` from the model that held it out.

    `inputs` are the models' features, in order.
    """
    rows = _with_fold(features.filter(pl.col("exporter").cast(pl.String) == uk), len(models))
    return rows.with_columns(export_potential_usd=export_potential(rows, models, features=inputs))


def _with_fold(rows: pl.DataFrame, n_folds: int) -> pl.DataFrame:
    return assign_folds(rows.rename({"importer": "iso3"}), n_folds).rename({"iso3": "importer"})


def outside_fold(uk: str, n_folds: int, k: int):
    """Row filter for fold model k: every non-UK row, and UK rows whose pair is not in fold k."""

    def keep(table: pl.DataFrame) -> pl.DataFrame:
        is_uk = pl.col("exporter").cast(pl.String) == uk
        uk_rows = _with_fold(table.filter(is_uk), n_folds).filter(pl.col("fold") != k).drop("fold")
        return pl.concat([table.filter(~is_uk), uk_rows])

    return keep
