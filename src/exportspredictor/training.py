"""XGBoost training on feature tables too large to hold in memory, and the hyperparameter search.

Training data is given as **chunks**: a sequence of feature tables (DataFrames, LazyFrames
or Parquet paths), typically one per year. A chunk is read only while XGBoost consumes it
through a data iterator into a `QuantileDMatrix`, so peak host memory is about one year of
rows, never the ~165 M training rows at once. The fixture passes one DataFrame; real data
passes the per-year files written by `features.build()`.

The model is a single XGBoost model (ADR 0002): Tweedie objective, histogram trees and a
+1 monotone constraint on every policy feature. Hyperparameters are tuned on the
validation year only (`tune`); the test year is never read here.
"""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import (
    FEATURES,
    KEYS,
    MONOTONE_CONSTRAINTS,
    FeatureTransform,
)

Chunk = pl.DataFrame | pl.LazyFrame | str | Path
Prepare = Callable[[pl.DataFrame], pl.DataFrame]

# Model selection uses one fixed metric, whatever the candidate's own variance power, so
# that candidates are comparable: Tweedie negative log-likelihood at p = 1.5, weighted.
SELECTION_METRIC = "tweedie-nloglik@1.5"

DEFAULT_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.1,
}
DEFAULT_ROUNDS = 100


def model_params(params: dict | None = None, *, device: str = "cpu", seed: int = 0) -> dict:
    """Full XGBoost parameters: defaults, the given overrides and the monotone constraints."""
    constraints = "(" + ",".join(str(MONOTONE_CONSTRAINTS.get(f, 0)) for f in FEATURES) + ")"
    return {**DEFAULT_PARAMS, **(params or {}), "monotone_constraints": constraints, "device": device, "seed": seed}


def read_chunk(chunk: Chunk) -> pl.DataFrame:
    if isinstance(chunk, pl.DataFrame):
        return chunk
    if isinstance(chunk, pl.LazyFrame):
        return chunk.collect()
    return pl.read_parquet(chunk)


def scan_chunks(chunks: Sequence[Chunk]) -> pl.LazyFrame:
    """All chunks as one lazy frame (e.g. to fit a `FeatureTransform` without loading them)."""
    frames = [
        c.lazy() if isinstance(c, pl.DataFrame) else c if isinstance(c, pl.LazyFrame) else pl.scan_parquet(c)
        for c in chunks
    ]
    return pl.concat([f.select(*KEYS, *FEATURES, "value_usd", "weight") for f in frames], how="vertical_relaxed")


def sample_non_uk(table: pl.DataFrame, *, uk: str, rate: float, seed: int = 0) -> pl.DataFrame:
    """Keep every UK row and a hash sample of other rows, reweighted by 1/rate (Horvitz–Thompson).

    `rate` = 1 keeps everything. A row's fate depends only on its keys and `seed`.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    if rate == 1:
        return table
    is_uk = pl.col("exporter").cast(pl.String) == uk
    keys = pl.struct(*(pl.col(k).cast(pl.String) for k in KEYS))
    kept = is_uk | (keys.hash(seed) < pl.lit(int(rate * 2**64), pl.UInt64))
    return table.filter(kept).with_columns(
        weight=pl.when(is_uk).then(pl.col("weight")).otherwise(pl.col("weight") / rate)
    )


class _ChunkIter(xgb.DataIter):
    """Feeds one prepared chunk at a time to XGBoost."""

    def __init__(self, chunks: Sequence[Chunk], prepare: Prepare):
        self._chunks, self._prepare, self._i = list(chunks), prepare, 0
        super().__init__()

    def reset(self) -> None:
        self._i = 0

    def next(self, input_data) -> bool:
        while self._i < len(self._chunks):
            table = self._prepare(read_chunk(self._chunks[self._i]))
            self._i += 1
            if table.height:
                input_data(
                    data=table.select(FEATURES).to_numpy().astype(np.float32),
                    label=table["value_usd"].to_numpy(),
                    weight=table["weight"].to_numpy(),
                    feature_names=FEATURES,
                )
                return True
        return False


def training_matrix(
    chunks: Sequence[Chunk], prepare: Prepare | None = None, *, ref: xgb.QuantileDMatrix | None = None
) -> xgb.QuantileDMatrix:
    """A QuantileDMatrix built chunk by chunk; `prepare` filters or transforms each chunk first."""
    return xgb.QuantileDMatrix(_ChunkIter(chunks, prepare or (lambda t: t)), ref=ref)


def preparer(
    *, uk: str, transform: FeatureTransform | None = None, non_uk_rate: float = 1.0, seed: int = 0,
    keep: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> Prepare:
    """The per-chunk preparation every training matrix uses: sample, select rows, transform."""

    def prepare(table: pl.DataFrame) -> pl.DataFrame:
        table = sample_non_uk(table, uk=uk, rate=non_uk_rate, seed=seed)
        if keep is not None:
            table = keep(table)
        return transform.apply(table) if transform is not None else table

    return prepare


def tune(
    train_chunks: Sequence[Chunk],
    valid_chunks: Sequence[Chunk],
    configs: Sequence[dict],
    *,
    uk: str,
    transform: FeatureTransform | None = None,
    device: str = "cpu",
    non_uk_rate: float = 1.0,
    max_rounds: int = 1000,
    early_stopping_rounds: int = 30,
    seed: int = 0,
) -> pl.DataFrame:
    """Train one model per config on the training chunks and score it on the validation chunks.

    Early stopping on the validation chunks picks each config's number of rounds. Returns one
    row per config: its parameters (as a JSON string), best round count, validation
    `SELECTION_METRIC` and seconds taken, best first.
    """
    prepare = preparer(uk=uk, transform=transform, non_uk_rate=non_uk_rate, seed=seed)
    train = training_matrix(train_chunks, prepare)
    valid = training_matrix(valid_chunks, prepare, ref=train)
    results = []
    for config in configs:
        params = model_params(config, device=device, seed=seed) | {"eval_metric": SELECTION_METRIC}
        started = time.perf_counter()
        model = xgb.train(
            params,
            train,
            num_boost_round=max_rounds,
            evals=[(valid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        results.append(
            {
                "config": json.dumps(dict(config), sort_keys=True),
                "rounds": model.best_iteration + 1,
                "valid_metric": float(model.best_score),
                "seconds": time.perf_counter() - started,
            }
        )
    del train, valid
    return pl.DataFrame(results).sort("valid_metric")
