"""The trainer seam (ticket #165): how the loop trains one model, behind one interface per model family.

The loop builds the rows (`TrainingRows`) and says how to train on them (`TrainingJob`); a
`Trainer` turns them into a model that predicts Export Potential for the held-out UK grid,
stopping on the Selection Measure of that grid. `Settings.family` picks the trainer
(`trainer_for`); `XGBOOST` is the default and the only family so far.
"""

import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import xgboost as xgb
from xgboost.callback import EarlyStopping

from exportspredictor.loop.selection import SELECTION_MEASURE
from exportspredictor.training import Chunk, Prepare, model_params, training_matrix

XGBOOST = "xgboost"

# What a trainer returns. Only XGBoost Boosters so far; a new family widens this where
# `prediction.export_potential` learns to predict with its models.
Model = xgb.Booster


@dataclass(frozen=True)
class TrainingRows:
    """The rows one model learns from, and the held-out UK grid it stops on.

    Each training chunk, once `prepare`d, and the `grid` hold the `features`, the label
    `value_usd`, the row `weight` and, for a model with a starting point, `training.BASE_MARGIN`
    (log scale). `key` is the same whenever the rows are, so a trainer may keep what it built
    from them for the next model (`Trainer.release` drops it).
    """

    chunks: Sequence[Chunk]
    prepare: Prepare
    grid: pl.DataFrame
    features: tuple[str, ...]
    key: Hashable


@dataclass(frozen=True)
class TrainingJob:
    """How to train one model on `rows`.

    `measure` is the Selection Measure of predictions for the grid's rows, in the grid's order;
    training keeps the round where it is highest and stops after `patience` rounds without a
    better one, or at `max_rounds`. With `monotone`, the policy features' monotone constraints
    (`features.MONOTONE_CONSTRAINTS`) must hold. `binary` trains a two-part model's "any Flow"
    part: the label is 0 or 1 and the prediction a probability.
    """

    rows: TrainingRows
    params: Mapping
    seed: int
    monotone: bool
    max_rounds: int
    patience: int
    measure: Callable[[np.ndarray], float]
    device: str
    threads: int | None = None
    binary: bool = False


class Trainer(Protocol):
    """One model family's training, saving and loading. A family that cannot enforce the monotone
    constraints refuses a job with `monotone`, so it is never run with them silently dropped."""

    suffix: str  # a saved model's file extension

    def train(self, job: TrainingJob) -> tuple[Model, dict]:
        """The model at its best round, and its metadata: at least `rounds` (the best round count),
        `hit_round_cap`, `trained_rounds`, `stopping_measure`, `training_rows`, `matrix_seconds` and
        `train_seconds` (matrices included)."""
        ...

    def save(self, model: Model, path: Path) -> None: ...

    def load(self, path: Path) -> Model: ...

    def release(self) -> None:
        """Drop anything kept from the last rows (device memory included)."""
        ...


# The binary part keeps the best so far's tree parameters; only the objective changes.
TWEEDIE_ONLY = ("tweedie_variance_power",)
ANY_FLOW_PARAMS = {"objective": "binary:logistic"}


class XGBoostTrainer:
    """XGBoost on `QuantileDMatrix`es built chunk by chunk on the device; the matrices are reused
    while the rows' key stays the same."""

    suffix = ".ubj"

    def __init__(self) -> None:
        self._key: Hashable = None
        self._matrices: tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix] | None = None

    def train(self, job: TrainingJob) -> tuple[Model, dict]:
        started = time.perf_counter()
        train, grid = self._built(job.rows, job.device)
        matrix_seconds = round(time.perf_counter() - started, 1)  # 0 when the matrices were reused
        params = model_params(
            dict(job.params), device=job.device, seed=job.seed, features=job.rows.features, monotone=job.monotone
        )
        if job.binary:
            params = {k: v for k, v in params.items() if k not in TWEEDIE_ONLY} | ANY_FLOW_PARAMS
        params |= {"disable_default_eval_metric": 1} | ({"nthread": job.threads} if job.threads else {})

        def metric(predictions, _matrix) -> tuple[str, float]:
            return SELECTION_MEASURE, job.measure(predictions)

        model = xgb.train(
            params,
            train,
            num_boost_round=job.max_rounds,
            evals=[(grid, "grid")],
            custom_metric=metric,
            callbacks=[EarlyStopping(rounds=job.patience, metric_name=SELECTION_MEASURE, data_name="grid", maximize=True)],
            verbose_eval=False,
        )
        rounds = model.best_iteration + 1
        return model, {
            "rounds": rounds,
            "hit_round_cap": rounds >= job.max_rounds,
            "trained_rounds": model.num_boosted_rounds(),
            "stopping_measure": float(model.best_score),  # the selection measure at `rounds`, as training saw it
            "training_rows": train.num_row(),
            "matrix_seconds": matrix_seconds,
            "train_seconds": round(time.perf_counter() - started, 1),  # matrices included
        }

    def _built(self, rows: TrainingRows, device: str) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]:
        if rows.key != self._key or self._matrices is None:
            self.release()  # free the device memory before building the next
            train = training_matrix(rows.chunks, rows.prepare, device=device, features=rows.features)
            grid = training_matrix([rows.grid], ref=train, device=device, features=rows.features)
            self._key, self._matrices = rows.key, (train, grid)
        return self._matrices

    def save(self, model: Model, path: Path) -> None:
        model.save_model(str(path))

    def load(self, path: Path) -> Model:
        return xgb.Booster(model_file=str(path))

    def release(self) -> None:
        self._key, self._matrices = None, None


# Each family's trainer. A family named here is a valid `Settings.family`.
TRAINERS: dict[str, Callable[[], Trainer]] = {XGBOOST: XGBoostTrainer}


def trainer_for(family: str) -> Trainer:
    """A new trainer of `family`; refuses a family with none."""
    if family not in TRAINERS:
        raise ValueError(f"The loop has no trainer for family {family!r}; it has {sorted(TRAINERS)}")
    return TRAINERS[family]()
