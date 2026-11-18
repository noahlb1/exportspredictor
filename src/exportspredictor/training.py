"""XGBoost training on feature tables too large to hold in memory, and the hyperparameter search.

Training data is given as **chunks**: a sequence of feature tables (DataFrames, LazyFrames
or Parquet paths), typically one per year. A chunk is read only while XGBoost consumes it
through a data iterator into a `QuantileDMatrix`, so peak host memory is about one year of
rows, never every training row at once. On the GPU each chunk is copied to the device and
the matrix is built there (`training_matrix(..., device="cuda")`). The fixture passes one
DataFrame; real data passes the per-year files written by `features.build()`.

The rows must fit the VRAM row budget (ADR 0004): `count_rows` measures them, and
`fit_non_uk_rate` derives the largest non-UK sampling rate that fits, or
`check_row_budget` refuses a given one, before any training.

The model is a single XGBoost model (ADR 0002): Tweedie objective, histogram trees and a
+1 monotone constraint on every policy feature. Hyperparameters are tuned on the
validation year only (`tune`); the test year is never read here.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


def model_params(
    params: dict | None = None, *, device: str = "cpu", seed: int = 0, features: Sequence[str] = FEATURES
) -> dict:
    """Full XGBoost parameters: defaults, the given overrides and the monotone constraints.

    `features` are the matrix's columns, in order (a feature ablation trains on fewer).
    """
    constraints = "(" + ",".join(str(MONOTONE_CONSTRAINTS.get(f, 0)) for f in features) + ")"
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


# --- The row budget (ADR 0004) -------------------------------------------------------------
#
# Training runs in-core on the 10 GB GPU, so the rows of every training matrix must fit in
# VRAM. Peak VRAM grows linearly with rows (`GPU_CHECK`): about 300 MiB fixed plus 72.9
# bytes a row with 22 features, measured on a device-built matrix of up to 107 M rows.
# The per-feature share is XGBoost's compressed bin index (about 12 bits a feature for
# 256 bins); the rest of a row's cost (label, weight, gradients, predictions, row
# partition) does not depend on the feature count.
VRAM_BUDGET_BYTES = 7680 * 2**20  # peak above idle: the desktop holds about 2 of the 10 GiB
VRAM_FIXED_BYTES = 300 * 2**20  # CUDA context and XGBoost's own buffers, whatever the rows
FEATURE_BYTES = 1.5  # per row and feature: the compressed histogram index
ROW_BYTES = 72.86 - 22 * FEATURE_BYTES  # per row, whatever the features
RATE_STEP = 1e-4  # the derived non-UK rate is a multiple of this
SAMPLE_SDS = 4  # headroom for hash sampling: a sample this many standard deviations large still fits
# Short GPU runs (the row-budget check, and the experiments of ticket #53) train on this
# Sector subset: the three large UK Sectors of `constraints.REAL_PLAN` (beverages,
# pharmaceuticals, vehicles) plus fish, plastics, knitted apparel and electrical machinery,
# so that food, chemicals, textiles and machinery all appear and the subset has tens of
# millions of rows.
SECTOR_SUBSET = ("03", "22", "30", "39", "61", "85", "87")
# The GPU check (ticket #52, `scripts/gpu_check.py`): v1's configuration on a device-built
# matrix of the Sector subset's 2012–2021 rows (top-100 scope, every UK Zero Cell, 5% of
# other Zero Cells) fed four times. Recorded in every artefact's metadata.
GPU_CHECK = {
    "measured": "2026-11-17",
    "gpu": "RTX 3080 10 GB",
    "xgboost_version": "3.4.1",
    "rows": 107_464_440,
    "features": 22,
    "peak_vram_mib": 7670,  # above the idle desktop's 1.9 GiB; peaks while training, not while building
    "build_peak_vram_mib": 4165,
    "build_seconds": 32.1,
    "seconds_per_round": 0.2875,
    "bytes_per_row": 72.86,
    "fixed_bytes": 300 * 2**20,
}


class RowBudgetExceeded(ValueError):
    """The training rows would not fit the VRAM row budget; raised before any training."""


@dataclass(frozen=True)
class RowCounts:
    """Training rows by Exporter: the UK's (never sampled) and every other Exporter's."""

    uk: int
    non_uk: int

    def expected(self, rate: float) -> float:
        """Expected rows once other Exporters' rows are sampled at `rate`."""
        return self.uk + rate * self.non_uk

    def high(self, rate: float):
        """Expected rows plus `SAMPLE_SDS` standard deviations of the hash sample's size."""
        rate = np.asarray(rate, dtype=float)
        return self.expected(rate) + SAMPLE_SDS * np.sqrt(self.non_uk * rate * (1 - rate))


def row_budget(n_features: int = len(FEATURES)) -> int:
    """The most training rows that fit `VRAM_BUDGET_BYTES` with `n_features` features (about 110 M)."""
    return int((VRAM_BUDGET_BYTES - VRAM_FIXED_BYTES) // (ROW_BYTES + FEATURE_BYTES * n_features))


def count_rows(chunks: Sequence[Chunk], *, uk: str) -> RowCounts:
    """Count the UK's and other Exporters' rows in the chunks, reading only the Exporter column."""
    frames = [
        c.lazy() if isinstance(c, pl.DataFrame) else c if isinstance(c, pl.LazyFrame) else pl.scan_parquet(c)
        for c in chunks
    ]
    is_uk = pl.col("exporter").cast(pl.String) == uk
    counts = pl.concat([f.select(is_uk.alias("is_uk")) for f in frames]).select(
        uk=pl.col("is_uk").sum(), non_uk=(~pl.col("is_uk")).sum()
    ).collect()
    return RowCounts(uk=int(counts["uk"][0] or 0), non_uk=int(counts["non_uk"][0] or 0))


def fit_non_uk_rate(counts: RowCounts, budget: int) -> float:
    """The largest non-UK sampling rate (a multiple of `RATE_STEP`, at most 1) whose rows fit `budget`.

    UK rows are always kept, so they take their share of the budget first. Raises
    `RowBudgetExceeded` if not even the smallest rate fits.
    """
    if counts.expected(1.0) <= budget:
        return 1.0
    rates = np.arange(1, round(1 / RATE_STEP)) * RATE_STEP
    fits = np.flatnonzero(counts.high(rates) <= budget)
    if not fits.size:
        raise RowBudgetExceeded(
            f"{counts.uk:,} UK rows leave no room for other Exporters' rows in a budget of {budget:,} rows"
        )
    return round(float(rates[fits[-1]]), 4)


def check_row_budget(counts: RowCounts, rate: float, budget: int) -> None:
    """Raise `RowBudgetExceeded` if sampling other Exporters' rows at `rate` might not fit `budget`."""
    if counts.high(rate) > budget:
        try:
            fits = f"the largest rate that fits is {fit_non_uk_rate(counts, budget)}"
        except RowBudgetExceeded:
            fits = "no rate fits"
        raise RowBudgetExceeded(
            f"{counts.uk:,} UK rows and {counts.non_uk:,} other rows sampled at {rate} make about "
            f"{counts.expected(rate):,.0f} rows, over the budget of {budget:,} rows; {fits}"
        )


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


BASE_MARGIN = "base_margin"  # a prepared chunk's per-row starting point on the log scale, if it has one


class _ChunkIter(xgb.DataIter):
    """Feeds one prepared chunk at a time to XGBoost.

    On a CUDA device each chunk is copied to the GPU as CuPy arrays, so the
    `QuantileDMatrix` is sketched and compressed on the device and no host copy of the
    whole matrix is kept (ADR 0004). The previous chunk's device memory is handed back
    before the next is copied, so only one chunk is ever on the device beside the matrix.
    """

    def __init__(
        self, chunks: Sequence[Chunk], prepare: Prepare, device: str = "cpu", features: Sequence[str] = FEATURES
    ):
        self._chunks, self._prepare, self._i, self._features = list(chunks), prepare, 0, list(features)
        self.device = _Device(device)
        super().__init__()

    def reset(self) -> None:
        self._i = 0

    def next(self, input_data) -> bool:
        self.device.release()  # XGBoost has dropped the previous chunk by now
        while self._i < len(self._chunks):
            table = self._prepare(read_chunk(self._chunks[self._i]))
            self._i += 1
            if table.height:
                # A table with a `base_margin` column trains from that starting point (log scale).
                margin = {} if BASE_MARGIN not in table.columns else {
                    "base_margin": self.device.array(table[BASE_MARGIN].to_numpy())
                }
                input_data(
                    data=self.device.array(table.select(self._features).to_numpy()),
                    label=self.device.array(table["value_usd"].to_numpy()),
                    weight=self.device.array(table["weight"].to_numpy()),
                    feature_names=self._features,
                    **margin,
                )
                return True
        return False


class _Device:
    """Where a matrix's chunks go: float32 host arrays for the CPU, CuPy arrays for CUDA."""

    def __init__(self, device: str):
        self._cupy = None
        if device.startswith("cuda"):
            import cupy  # only where training runs on the GPU; the API never trains

            self._cupy = cupy

    def array(self, array: np.ndarray):
        array = np.ascontiguousarray(array, dtype=np.float32)  # row-major, as XGBoost reads it
        return array if self._cupy is None else self._cupy.asarray(array)

    def release(self) -> None:
        """Hand device memory CuPy holds but no array uses back to the driver, for XGBoost."""
        if self._cupy is not None:
            self._cupy.get_default_memory_pool().free_all_blocks()


def training_matrix(
    chunks: Sequence[Chunk],
    prepare: Prepare | None = None,
    *,
    ref: xgb.QuantileDMatrix | None = None,
    device: str = "cpu",
    features: Sequence[str] = FEATURES,
) -> xgb.QuantileDMatrix:
    """A QuantileDMatrix of `features`, built chunk by chunk on `device`; `prepare` filters or transforms each chunk first."""
    chunk_iter = _ChunkIter(chunks, prepare or (lambda t: t), device, features)
    matrix = xgb.QuantileDMatrix(chunk_iter, ref=ref)
    chunk_iter.device.release()
    return matrix


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
    train = training_matrix(train_chunks, prepare, device=device)
    valid = training_matrix(valid_chunks, prepare, ref=train, device=device)
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
