"""The trainer seam (ticket #165): how the loop trains one model, behind one interface per model family.

The loop builds the rows (`TrainingRows`) and says how to train on them (`TrainingJob`); a
`Trainer` turns them into a model that predicts Export Potential for the held-out UK grid,
stopping on the Selection Measure of that grid. `Settings.family` picks the trainer
(`trainer_for`); `XGBOOST` is the default. `LIGHTGBM` and `CATBOOST` (#167) train on the CPU:
LightGBM's wheel has no GPU learner, and CatBoost enforces monotone constraints only on the CPU and
only for symmetric trees. Both read the rows from a float32 matrix on disk, so the raw rows are
never held in memory beside the binned ones.
"""

import gc
import tempfile
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import xgboost as xgb
from xgboost.callback import EarlyStopping

from exportspredictor.features import MONOTONE_CONSTRAINTS
from exportspredictor.loop.selection import SELECTION_MEASURE
from exportspredictor.prediction import Model
from exportspredictor.training import (
    BASE_MARGIN,
    Chunk,
    Prepare,
    model_params,
    read_chunk,
    training_matrix,
)

XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
CATBOOST = "catboost"


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
    # `Settings.external_memory` (#170): XGBoost only; another family refuses a job with it.
    external_memory: bool = False


def _refuse_external_memory(job: TrainingJob, family: str) -> None:
    """Only the XGBoost trainer trains through external memory."""
    if job.external_memory:
        raise ValueError(f"The {family} trainer has no external memory; only XGBoost's does")


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
    while the rows' key stays the same.

    With `TrainingJob.external_memory` (#170), the training rows are an `ExtMemQuantileDMatrix` instead: one
    page per chunk, cached in host memory (on the CPU, in files in a temporary folder) and streamed to the
    device each round, so they need not fit in VRAM. The held-out grid stays in-core, binned by its cuts.
    """

    suffix = ".ubj"

    def __init__(self) -> None:
        self._key: Hashable = None
        self._matrices: tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix] | None = None
        self._cache: tempfile.TemporaryDirectory | None = None

    def train(self, job: TrainingJob) -> tuple[Model, dict]:
        started = time.perf_counter()
        train, grid = self._built(job.rows, job.device, job.external_memory)
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

    def _built(
        self, rows: TrainingRows, device: str, external_memory: bool
    ) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]:
        key = (rows.key, external_memory)
        if key != self._key or self._matrices is None:
            self.release()  # free the device memory before building the next
            cache = None
            if external_memory:
                self._cache = tempfile.TemporaryDirectory(prefix="extmem-", ignore_cleanup_errors=True)
                cache = Path(self._cache.name)
            train = training_matrix(rows.chunks, rows.prepare, device=device, features=rows.features, cache=cache)
            grid = training_matrix([rows.grid], ref=train, device=device, features=rows.features)
            self._key, self._matrices = key, (train, grid)
        return self._matrices

    def save(self, model: Model, path: Path) -> None:
        model.save_model(str(path))

    def load(self, path: Path) -> Model:
        return xgb.Booster(model_file=str(path))

    def release(self) -> None:
        self._key, self._matrices = None, None
        if self._cache is not None:
            gc.collect()  # the external-memory matrix holds its page files open until it is freed
            self._cache.cleanup()
            self._cache = None


# --- Other families (#167) ---------------------------------------------------------------------

# LightGBM's defaults under a family's `params`: XGBoost's objective, link and histogram width.
# `basic` is LightGBM's strict monotone method (each split's children bounded at their midpoint, as
# XGBoost bounds them); the default metric is off, so only the Selection Measure is evaluated. No
# feature is dropped before training for having too few rows to split (XGBoost drops none).
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.1,
    "max_bin": 255,
    "feature_pre_filter": False,
    "monotone_constraints_method": "basic",
    "metric": "None",
    "verbose": -1,
}
# CatBoost's: `tweedie_variance_power` is written into the loss's name. Symmetric trees only: CatBoost
# enforces monotone constraints for no other grow policy, and on the CPU only.
CATBOOST_DEFAULT_PARAMS = {
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.1,
    "depth": 6,
    "grow_policy": "SymmetricTree",
    "border_count": 254,
}


def _monotone(features: Sequence[str]) -> list[int]:
    """The policy features' monotone constraints (`features.MONOTONE_CONSTRAINTS`), one per feature in order."""
    return [MONOTONE_CONSTRAINTS.get(f, 0) for f in features]


def _refuse_binary(job: TrainingJob, family: str) -> None:
    """Only the XGBoost trainer trains a two-part model's binary part."""
    if job.binary:
        raise ValueError(f"The {family} trainer trains no two-part model")


@dataclass(frozen=True)
class _DiskRows:
    """Prepared training rows: the features as a float32 matrix mapped from disk, label and weight in memory."""

    features: np.ndarray
    label: np.ndarray
    weight: np.ndarray


def _rows_on_disk(rows: TrainingRows, folder: Path, family: str) -> _DiskRows:
    """Prepare each chunk in turn and append its features to one file, so one chunk's raw rows at most are in memory."""
    path = folder / "features.f32"
    labels, weights, n = [], [], 0
    with path.open("wb") as file:
        for chunk in rows.chunks:
            table = rows.prepare(read_chunk(chunk))
            if BASE_MARGIN in table.columns:
                raise ValueError(f"The {family} trainer trains no model from a starting point (base_margin)")
            if not table.height:
                continue
            file.write(np.ascontiguousarray(table.select(rows.features).to_numpy(), dtype=np.float32).tobytes())
            labels.append(table["value_usd"].cast(pl.Float32).to_numpy())
            weights.append(table["weight"].cast(pl.Float32).to_numpy())
            n += table.height
    matrix = np.memmap(path, dtype=np.float32, mode="r", shape=(n, len(rows.features)))
    return _DiskRows(matrix, np.concatenate(labels), np.concatenate(weights))


def _grid(rows: TrainingRows) -> tuple[np.ndarray, np.ndarray]:
    """The held-out UK grid's features and label, in the grid's order."""
    grid = rows.grid
    return (
        np.ascontiguousarray(grid.select(rows.features).to_numpy(), dtype=np.float32),
        grid["value_usd"].cast(pl.Float32).to_numpy(),
    )


class LightGBMTrainer:
    """LightGBM on the CPU: leaf-wise trees with the Tweedie objective and log link."""

    suffix = ".lgbm"

    def train(self, job: TrainingJob) -> tuple[Model, dict]:
        import lightgbm as lgb

        _refuse_binary(job, LIGHTGBM)
        _refuse_external_memory(job, LIGHTGBM)
        started = time.perf_counter()
        params = {**LIGHTGBM_DEFAULT_PARAMS, **job.params, "seed": job.seed}
        params |= {"num_threads": job.threads} if job.threads else {}
        if job.monotone:
            params["monotone_constraints"] = _monotone(job.rows.features)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            disk = _rows_on_disk(job.rows, Path(folder), LIGHTGBM)
            training_rows = len(disk.label)
            train = lgb.Dataset(
                disk.features, label=disk.label, weight=disk.weight, feature_name=list(job.rows.features),
                params=params, free_raw_data=True,
            ).construct()
            del disk  # the binned Dataset holds the rows now; unmap the file before it is deleted
        grid_features, grid_label = _grid(job.rows)
        grid = lgb.Dataset(grid_features, label=grid_label, reference=train, params=params).construct()
        matrix_seconds = round(time.perf_counter() - started, 1)

        def metric(predictions, _data) -> tuple[str, float, bool]:
            return SELECTION_MEASURE, job.measure(predictions), True

        model = lgb.train(
            params, train, num_boost_round=job.max_rounds, valid_sets=[grid], valid_names=["grid"], feval=metric,
            callbacks=[lgb.early_stopping(job.patience, first_metric_only=True, verbose=False)],
            keep_training_booster=True,  # every trained round, as XGBoost keeps them (else only the best are kept)
        )
        rounds = model.best_iteration or model.current_iteration()
        return model, {
            "rounds": rounds,
            "hit_round_cap": rounds >= job.max_rounds,
            "trained_rounds": model.current_iteration(),
            "stopping_measure": float(model.best_score["grid"][SELECTION_MEASURE]),
            "training_rows": training_rows,
            "matrix_seconds": matrix_seconds,
            "train_seconds": round(time.perf_counter() - started, 1),  # rows included
        }

    def save(self, model: Model, path: Path) -> None:
        model.save_model(str(path), num_iteration=-1)  # every round; prediction stops at the chosen one

    def load(self, path: Path) -> Model:
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))

    def release(self) -> None:
        """Nothing is kept between models."""


class _CatBoostMeasure:
    """The Selection Measure as a CatBoost eval metric, on the held-out grid only. CatBoost asks on the
    training rows too, where the measure is not defined; it reads 0 there and is never used."""

    def __init__(self, measure: Callable[[np.ndarray], float], grid_rows: int):
        self.measure, self.grid_rows = measure, grid_rows

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes, target, weight) -> tuple[float, float]:
        if len(approxes[0]) != self.grid_rows:
            return 0.0, 1.0
        return self.measure(np.asarray(approxes[0])), 1.0  # log scale, which ranks as its exponential does

    def get_final_error(self, error: float, weight: float) -> float:
        return error


def _column_major(features: np.ndarray, folder: Path, block: int = 4_000_000) -> np.ndarray:
    """The row-major matrix copied, block by block, to a column-major one on disk. CatBoost reads a column-major
    float32 matrix in place, but copies a row-major one into memory whole before quantizing it."""
    columns = np.lib.format.open_memmap(folder / "columns.npy", mode="w+", dtype=np.float32, shape=features.shape,
                                        fortran_order=True)
    for start in range(0, features.shape[0], block):
        columns[start : start + block] = features[start : start + block]
    columns.flush()
    return np.load(folder / "columns.npy", mmap_mode="r")


class CatBoostTrainer:
    """CatBoost on the CPU: symmetric trees with the Tweedie loss and log link.

    CatBoost starts no Tweedie model from the average (`boost_from_average` refuses the loss), so the trees
    start from a constant baseline, the log of the rows' weighted mean label, as XGBoost's estimated
    intercept does; the baseline is then written into the model as its bias, so it predicts without one.
    """

    suffix = ".cbm"

    def train(self, job: TrainingJob) -> tuple[Model, dict]:
        import catboost as cb

        _refuse_binary(job, CATBOOST)
        _refuse_external_memory(job, CATBOOST)
        params = {**CATBOOST_DEFAULT_PARAMS, **job.params}
        if job.monotone and params["grow_policy"] != "SymmetricTree":
            raise ValueError("CatBoost enforces monotone constraints only for symmetric trees (grow_policy)")
        started = time.perf_counter()
        power = params.pop("tweedie_variance_power")
        metric = _CatBoostMeasure(job.measure, job.rows.grid.height)
        params |= {
            "loss_function": f"Tweedie:variance_power={power}", "eval_metric": metric, "task_type": "CPU",
            "iterations": job.max_rounds, "od_type": "Iter", "od_wait": job.patience, "use_best_model": True,
            "random_seed": job.seed, "thread_count": job.threads or -1, "allow_writing_files": False,
            "verbose": False,
        }
        if job.monotone:
            params["monotone_constraints"] = _monotone(job.rows.features)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            disk = _rows_on_disk(job.rows, Path(folder), CATBOOST)
            training_rows = len(disk.label)
            bias = float(np.log(np.average(disk.label, weights=disk.weight)))
            train = cb.Pool(
                _column_major(disk.features, Path(folder)), label=disk.label, weight=disk.weight,
                baseline=np.full(training_rows, bias), feature_names=list(job.rows.features),
            )
            train.quantize(border_count=params["border_count"])
            del disk
        grid_features, grid_label = _grid(job.rows)
        grid = cb.Pool(
            grid_features, label=grid_label, baseline=np.full(len(grid_label), bias),
            feature_names=list(job.rows.features),
        )
        matrix_seconds = round(time.perf_counter() - started, 1)

        model = cb.CatBoost(params)
        model.fit(train, eval_set=grid)
        model.set_scale_and_bias(1.0, bias)
        measured = model.get_evals_result()["validation"][type(metric).__name__]
        best = model.get_best_iteration()
        rounds = (best if best is not None else len(measured) - 1) + 1
        return model, {
            "rounds": rounds,
            "hit_round_cap": rounds >= job.max_rounds,
            "trained_rounds": len(measured),
            "stopping_measure": float(measured[rounds - 1]),
            "training_rows": training_rows,
            "matrix_seconds": matrix_seconds,
            "train_seconds": round(time.perf_counter() - started, 1),  # rows included
        }

    def save(self, model: Model, path: Path) -> None:
        model.save_model(str(path))

    def load(self, path: Path) -> Model:
        import catboost as cb

        return cb.CatBoost().load_model(str(path))

    def release(self) -> None:
        """Nothing is kept between models."""


# Each family's trainer. A family named here is a valid `Settings.family`.
TRAINERS: dict[str, Callable[[], Trainer]] = {
    XGBOOST: XGBoostTrainer, LIGHTGBM: LightGBMTrainer, CATBOOST: CatBoostTrainer,
}


def trainer_for(family: str) -> Trainer:
    """A new trainer of `family`; refuses a family with none."""
    if family not in TRAINERS:
        raise ValueError(f"The loop has no trainer for family {family!r}; it has {sorted(TRAINERS)}")
    return TRAINERS[family]()
