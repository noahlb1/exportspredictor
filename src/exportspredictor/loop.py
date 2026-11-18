"""The experiment loop (spec #65, ticket #67): every modelling choice is kept or dropped on one selection measure.

ADR 0008 makes Opportunity ranking the one objective. This module fixes, in code, how a
choice is measured and when it counts as a win:

- **The selection measure** (`SELECTION_MEASURE`): the mean over Products of the Spearman
  correlation between Export Potential and Actual Exports across Markets, on the held-out
  fold's UK pairs in the validation year (2022), on every non-Excluded Product. It is the
  headline measure of `evaluation` moved to the validation year. Nothing here selects on
  Tweedie deviance: it is reported with the other diagnostics (`measures`), never used.
- **An arm** (`Arm`) is a named set of settings that differ from the best arm so far.
  Running it trains one model on the training years (2012–2021) with one fold of UK pairs
  held out (ADR 0001), scores the validation year, and returns one result row. The round
  count is chosen by the selection measure itself, evaluated on the held-out UK grid every
  round, with early stopping after `Settings.patience` rounds without improvement.
- **The noise floor** (`Loop.noise_floor`) runs first: the current v2 configuration with
  `LoopPlan.noise_seeds`, each seed changing the non-UK hash sample and the XGBoost seed
  together. The win threshold is twice the standard deviation of the selection measure
  over those seeds, and at least `MIN_WIN_THRESHOLD`.
- **The win rule** (`decide`): in an experiment, the arm with the highest selection
  measure is marked best; it is kept, and becomes the best so far, only if it beats the
  best so far by more than the win threshold. Every other arm is dropped.

Trained models are cached by the settings that decide them (`training_key`), so an arm
that changes only how a saved model is scored (`SCORING_FIELDS`) trains nothing. Results
are saved as each arm finishes, and a re-run skips finished arms, so a killed run resumes.

**The test year (2023) is never read**: the loop's data is `experiments.Data`, which
refuses it, and `real_data` refuses it before opening a file.

    uv run python -m exportspredictor.loop noise   # GPU, three arms of about 11 minutes each

The results are associations measured on one fold and one year: an arm that wins ranks
the held-out UK Markets better in 2022, which says nothing about why.
"""

import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import baci, experiments, search
from exportspredictor.evaluation import (
    MIN_PREDICTION_USD,
    MIN_RANKED_ROWS,
    TWEEDIE_POWER,
    naive_potential,
    tweedie_deviance,
)
from exportspredictor.exclusions import drop_excluded
from exportspredictor.experiments import Data, TestYearRead, spearman_within_product
from exportspredictor.features import (
    FEATURES,
    KEYS,
    ZERO_SAMPLING_RATE,
    FeatureTransform,
    features_dir,
    subsample_zeros,
)
from exportspredictor.folds import N_FOLDS, assign_folds, outside_fold
from exportspredictor.paths import data_dir
from exportspredictor.prediction import export_potential
from exportspredictor.training import (
    model_params,
    preparer,
    read_chunk,
    scan_chunks,
    training_matrix,
)

__all__ = ["TestYearRead"]

SELECTION_MEASURE_TEXT = (
    "Mean over Products of the Spearman correlation between Export Potential and Actual Exports across "
    "Markets, on the held-out fold's UK pairs in the validation year, every non-Excluded Product"
)

# The one measure every choice is made on (ADR 0008), and the diagnostics reported beside it.
SELECTION_MEASURE = "selection_measure"
DIAGNOSTICS = (
    "non_zero_spearman",  # the selection measure on non-zero UK Flows only
    "zero_vs_non_zero_spearman",  # within-Product Spearman against the indicator "the UK exports at all"
    "value_weighted_spearman",  # the per-Product correlations weighted by the Product's UK exports
    "top_ten_share",  # share of UK exports in each Product's ten Markets with the highest Export Potential
    "tweedie_deviance",  # weighted, p = 1.5, on the validation year's training rows
)
TOP_MARKETS = 10
MIN_WIN_THRESHOLD = 0.005
ROUNDING = 1e-9  # float error in a difference of measures, never counted as a win
NOISE_SEEDS = (0, 1, 2)
MAX_ROUNDS = 5000
# Rounds without a better selection measure before training stops. The measure is noisy
# round to round, so this is wider than the search's 50 on deviance.
PATIENCE = 100
NO_MEASURE = -1.0  # what a round scores when no Product can be ranked (a tiny fixture fold)
NOISE_FLOOR = "noise_floor"
RESULTS_FILE = "results.json"
MODELS_DIR = "models"


@dataclass(frozen=True)
class Settings:
    """Everything that decides one arm's model (`TRAINING_FIELDS`) and how it is scored (`SCORING_FIELDS`)."""

    params: Mapping  # XGBoost parameters over `training.DEFAULT_PARAMS` (eta included)
    features: tuple[str, ...] = tuple(FEATURES)
    non_uk_rate: float | None = None  # None: the largest rate the VRAM budget allows
    zero_rate: float | None = None  # other Exporters' Zero Cells kept; None: the rate the rows were built at
    seed: int = 0  # the non-UK hash sample and the XGBoost seed, together
    max_rounds: int = MAX_ROUNDS
    patience: int = PATIENCE
    # Scoring only: Export Potential = exp((1 − w) · log model + w · log naive). 0 is the model alone.
    naive_weight: float = 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "params": dict(self.params), "features": list(self.features)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Settings":
        return cls(**{**data, "params": dict(data["params"]), "features": tuple(data["features"])})


SCORING_FIELDS = ("naive_weight",)
TRAINING_FIELDS = tuple(f.name for f in fields(Settings) if f.name not in SCORING_FIELDS)
# The training fields of the first loop (#67), always part of a model's cache key. A training
# field added later is keyed only when it leaves its default, so adding one keeps every
# cached model's key.
KEYED_FIELDS = ("params", "features", "non_uk_rate", "zero_rate", "seed", "max_rounds", "patience")


@dataclass(frozen=True)
class Arm:
    """A named experiment arm: the settings that differ from the best arm so far.

    `changes` maps `Settings` fields to new values; `params` is merged into the best arm's
    parameters (so a full configuration should name every parameter it sets), every other
    field is replaced.
    """

    name: str
    changes: Mapping = field(default_factory=dict)

    def apply(self, base: Settings) -> Settings:
        unknown = set(self.changes) - {f.name for f in fields(Settings)}
        if unknown:
            raise ValueError(f"Arm {self.name!r} changes unknown settings {sorted(unknown)}")
        changes = dict(self.changes)
        if "params" in changes:
            changes["params"] = {**base.params, **changes["params"]}
        if "features" in changes:
            changes["features"] = tuple(changes["features"])
        return replace(base, **changes)


@dataclass(frozen=True)
class LoopPlan:
    """Where the loop starts (the current v2 configuration) and how it trains."""

    start: Settings
    fold: int = 0  # the held-out fold of UK pairs
    n_folds: int = N_FOLDS
    noise_seeds: tuple[int, ...] = NOISE_SEEDS
    min_win_threshold: float = MIN_WIN_THRESHOLD
    device: str = "cuda"
    threads: int | None = None  # XGBoost threads; None: every core
    uk: str = baci.UK


@dataclass(frozen=True)
class LoopData:
    """The experiment rows of every Product, and the validation year's Flows for the naive baseline."""

    data: Data
    flows: Callable[[], pl.DataFrame]  # every Flow of the validation year (exporter, importer, hs6, value_usd)
    zero_seed: int = 0  # the seed the rows' Zero Cells were sampled with

    @property
    def validation_year(self) -> int:
        return self.data.validation_year


# The fixture world has a few hundred rows and a UK grid of a few dozen pairs, too few to
# rank in a fifth or a half of them: one "fold" holds every UK pair out, so the model learns
# from other Exporters only. Small trees, every row and one thread.
FIXTURE_LOOP = LoopPlan(
    start=Settings(params={"max_depth": 2, "eta": 0.3}, non_uk_rate=1.0, max_rounds=60, patience=10),
    n_folds=1,
    device="cpu",
    threads=1,
)


# --- The win rule -----------------------------------------------------------------------------


def win_threshold(selection_measures: Sequence[float], minimum: float = MIN_WIN_THRESHOLD) -> float:
    """Twice the standard deviation (n − 1) of the noise floor's selection measures, and at least `minimum`."""
    return max(2 * statistics.stdev(selection_measures), minimum)


def decide(rows: Sequence[Mapping], best_so_far: float, threshold: float) -> tuple[list[dict], str | None]:
    """Mark the experiment's rows: the highest selection measure is `best`; it is kept only if it wins.

    It wins when it beats `best_so_far` by more than `threshold` (a margin within rounding
    of the threshold does not). Returns the marked rows and the name of the kept arm (None
    when every arm is dropped). Deviance plays no part.
    """
    measured = [row for row in rows if row[SELECTION_MEASURE] is not None]
    top = max(measured, key=lambda row: row[SELECTION_MEASURE]) if measured else None
    wins = top is not None and top[SELECTION_MEASURE] - best_so_far > threshold + ROUNDING
    kept = top["arm"] if wins else None
    marked = [
        {
            **row,
            "best": top is not None and row["arm"] == top["arm"],
            "change": None if row[SELECTION_MEASURE] is None else row[SELECTION_MEASURE] - best_so_far,
            "status": "kept" if row["arm"] == kept else "dropped",
        }
        for row in rows
    ]
    return marked, kept


# --- The measures -----------------------------------------------------------------------------


def measures(grid: pl.DataFrame, validation: pl.DataFrame, prediction: str) -> dict:
    """The selection measure and every diagnostic of `prediction`.

    `grid` is the held-out UK grid of the validation year (hs6, value_usd); `validation`
    its training rows (value_usd, weight) for the deviance.
    """
    within = _within_product(grid, prediction, pl.col("value_usd"))
    weights = grid.group_by("hs6").agg(_uk=pl.col("value_usd").cast(pl.Float64).sum())
    weighted = within.join(weights, on="hs6", how="left")
    total = float(weighted["_uk"].sum()) if weighted.height else 0.0
    top = grid.with_columns(_rank=pl.col(prediction).rank("ordinal", descending=True).over("hs6"))
    uk_value = float(grid["value_usd"].cast(pl.Float64).sum())
    return {
        SELECTION_MEASURE: spearman_within_product(grid, prediction),
        "non_zero_spearman": _mean(_within_product(grid.filter(pl.col("value_usd") > 0), prediction, pl.col("value_usd"))),
        "zero_vs_non_zero_spearman": _mean(
            _within_product(grid, prediction, (pl.col("value_usd") > 0).cast(pl.Float64))
        ),
        "value_weighted_spearman": float((weighted["_rho"] * weighted["_uk"]).sum() / total) if total > 0 else None,
        "top_ten_share": (
            float(top.filter(pl.col("_rank") <= TOP_MARKETS)["value_usd"].cast(pl.Float64).sum() / uk_value)
            if uk_value > 0 else None
        ),
        "tweedie_deviance": float(validation.select(tweedie_deviance(prediction, power=TWEEDIE_POWER)).item()),
        "uk_pairs": grid.height,
        "ranked_products": within.height,
    }


def _within_product(rows: pl.DataFrame, prediction: str, target: pl.Expr) -> pl.DataFrame:
    """Per Product with at least MIN_RANKED_ROWS rows, the Spearman correlation of `prediction` and `target`."""
    return (
        rows.group_by("hs6")
        .agg(_n=pl.len(), _rho=pl.corr(pl.col(prediction), target, method="spearman"))
        .filter((pl.col("_n") >= MIN_RANKED_ROWS) & pl.col("_rho").is_not_nan() & pl.col("_rho").is_not_null())
    )


def _mean(within: pl.DataFrame) -> float | None:
    return float(within["_rho"].mean()) if within.height else None


# --- The loop ---------------------------------------------------------------------------------


class Loop:
    """The experiment loop on one fold: the noise floor, then experiments, each against the best so far.

    With a `folder`, results are saved there as each arm finishes, models are cached in
    it, and a new `Loop` on the same folder resumes; without one, both live in memory.
    """

    def __init__(self, data: LoopData, plan: LoopPlan, folder: Path | None = None):
        experiments._check_years([data.validation_year])
        started = time.perf_counter()
        self.data, self.plan, self.folder = data, plan, folder
        self._models: dict[str, tuple[xgb.Booster, dict]] = {}
        self._matrix_key, self._matrices = None, None
        self.results = _load(folder) or {}
        rows = data.data
        years = sorted(rows.training)
        self.chunks = [rows.training[y] for y in years]
        self.transform = FeatureTransform.fit(scan_chunks(self.chunks), years)
        flows = data.flows()
        is_uk = pl.col("exporter").cast(pl.String) == plan.uk
        validation = drop_excluded(read_chunk(rows.validation))
        validation = pl.concat([validation.filter(~is_uk), self._held_out(validation.filter(is_uk))])
        grid = self._held_out(drop_excluded(read_chunk(rows.grid)).filter(is_uk))
        self.validation = self.transform.apply(naive_potential(validation, flows))
        self.grid = self.transform.apply(naive_potential(grid, flows))
        del flows
        self._grid_keys = self.grid.select("hs6", "value_usd")
        if "naive" not in self.results:
            self.results |= {
                "selection_measure": SELECTION_MEASURE_TEXT,
                "validation_year": data.validation_year,
                "fold": plan.fold,
                "n_folds": plan.n_folds,
                "naive": {
                    **measures(self.grid, self.validation, "naive_usd"),
                    "validation_year": data.validation_year,
                },
                "experiments": {},
            }
        self.results["setup_seconds"] = round(time.perf_counter() - started, 1)
        self._save()

    def _held_out(self, uk_rows: pl.DataFrame) -> pl.DataFrame:
        folded = assign_folds(uk_rows.rename({"importer": "iso3"}), self.plan.n_folds).rename({"iso3": "importer"})
        return folded.filter(pl.col("fold") == self.plan.fold).drop("fold")

    # --- Running arms --------------------------------------------------------------------------

    def run_experiment(self, name: str, arms: Sequence[Arm]) -> dict:
        """Run each arm against the best so far, decide by the win rule, and return the results."""
        if name == NOISE_FLOOR:
            raise ValueError(f"{NOISE_FLOOR!r} is the noise floor's name")
        self.noise_floor()
        section = self.results["experiments"].setdefault(name, {"arms": []})
        if "decided" not in section:
            section["base"] = dict(self.results["best"])
        base = Settings.from_dict(section["base"]["settings"])
        rows = self._run_arms(section, arms, base)
        if "decided" not in section:
            threshold = self.results[NOISE_FLOOR]["win_threshold"]
            section["arms"], kept = decide(rows, section["base"][SELECTION_MEASURE], threshold)
            section |= {"win_threshold": threshold, "kept": kept, "decided": True}
            if kept is not None:
                row = next(r for r in section["arms"] if r["arm"] == kept)
                self.results["best"] = _best(name, row)
            self._save()
        return self.results

    def noise_floor(self) -> dict:
        """The start configuration with each noise seed; the win threshold; and the best so far."""
        section = self.results.setdefault(NOISE_FLOOR, {"arms": []})
        arms = [Arm(f"seed {seed}", {"seed": seed}) for seed in self.plan.noise_seeds]
        rows = self._run_arms(section, arms, self.plan.start)
        if "win_threshold" not in section:
            values = [row[SELECTION_MEASURE] for row in rows]
            section |= {
                "seeds": list(self.plan.noise_seeds),
                "selection_measures": values,
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values),
                "win_threshold": win_threshold(values, self.plan.min_win_threshold),
            }
            # Arms run on the first seed, so the best so far is that seed's row: a paired comparison.
            self.results["best"] = _best(NOISE_FLOOR, rows[0])
            self._save()
        return section

    def _run_arms(self, section: dict, arms: Sequence[Arm], base: Settings) -> list[dict]:
        done = {row["arm"] for row in section["arms"]}
        for arm in arms:
            if arm.name in done:
                continue
            row = self.run_arm(arm, base)
            section["arms"].append(row)
            self._save()
            _log(f"{arm.name}: {json.dumps({k: row[k] for k in (SELECTION_MEASURE, 'rounds', 'seconds')})}")
        order = {arm.name: i for i, arm in enumerate(arms)}
        return sorted((row for row in section["arms"] if row["arm"] in order), key=lambda row: order[row["arm"]])

    def run_arm(self, arm: Arm, base: Settings) -> dict:
        """One result row: the arm's settings over `base`, trained (or taken from the cache) and measured."""
        started = time.perf_counter()
        settings = self._resolved(arm.apply(base))
        key = training_key(settings, self._fingerprint())
        trained = key not in self._models and not self._model_file(key).exists()
        model, meta = self._model(settings, key)
        scored_grid = self.grid.with_columns(export_potential_usd=self._predict(model, meta, settings, self.grid))
        scored = self.validation.with_columns(
            export_potential_usd=self._predict(model, meta, settings, self.validation)
        )
        return {
            "arm": arm.name,
            "changes": dict(arm.changes),
            "settings": settings.to_dict(),
            "training_key": key,
            "trained": trained,
            **measures(scored_grid, scored, "export_potential_usd"),
            "validation_year": self.data.validation_year,
            **meta,
            "seconds": round(time.perf_counter() - started, 1),
        }

    # --- Models --------------------------------------------------------------------------------

    def _resolved(self, settings: Settings) -> Settings:
        zero_rate = settings.zero_rate if settings.zero_rate is not None else self.data.data.built_zero_rate
        non_uk_rate = settings.non_uk_rate
        if non_uk_rate is None:
            non_uk_rate = search.final_non_uk_rate(self.data.data, len(settings.features))
        return replace(settings, zero_rate=zero_rate, non_uk_rate=non_uk_rate)

    def _fingerprint(self) -> dict:
        rows = self.data.data
        return {
            "training_years": sorted(rows.training),
            "validation_year": rows.validation_year,
            "built_zero_rate": rows.built_zero_rate,
            "zero_seed": self.data.zero_seed,
            "fold": self.plan.fold,
            "n_folds": self.plan.n_folds,
            "device": self.plan.device,
        }

    def _model_file(self, key: str) -> Path:
        return Path(self.folder or ".nowhere") / MODELS_DIR / f"{key}.ubj"

    def _model(self, settings: Settings, key: str) -> tuple[xgb.Booster, dict]:
        if key in self._models:
            return self._models[key]
        path = self._model_file(key)
        if self.folder is not None and path.exists():
            model = xgb.Booster(model_file=str(path))
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        else:
            model, meta = self._train(settings)
            if self.folder is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(path))
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._models[key] = (model, meta)
        return model, meta

    def _train(self, settings: Settings) -> tuple[xgb.Booster, dict]:
        """Train with the selection measure on the held-out UK grid each round, stopping after `patience` flat rounds."""
        started = time.perf_counter()
        train, grid = self._training_matrices(settings)
        matrix_seconds = round(time.perf_counter() - started, 1)  # 0 when the matrices were reused
        params = model_params(dict(settings.params), device=self.plan.device, seed=settings.seed, features=settings.features)
        params |= {"disable_default_eval_metric": 1} | ({"nthread": self.plan.threads} if self.plan.threads else {})
        model = xgb.train(
            params,
            train,
            num_boost_round=settings.max_rounds,
            evals=[(grid, "grid")],
            custom_metric=self._round_measure,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=settings.patience, metric_name=SELECTION_MEASURE, data_name="grid", maximize=True
                )
            ],
            verbose_eval=False,
        )
        rounds = model.best_iteration + 1
        return model, {
            "rounds": rounds,
            "hit_round_cap": rounds >= settings.max_rounds,
            "trained_rounds": model.num_boosted_rounds(),
            "training_rows": train.num_row(),
            "matrix_seconds": matrix_seconds,
            "train_seconds": round(time.perf_counter() - started, 1),  # matrices included
        }

    def _round_measure(self, predictions, _matrix) -> tuple[str, float]:
        value = spearman_within_product(self._grid_keys.with_columns(prediction=pl.Series(predictions)), "prediction")
        return SELECTION_MEASURE, NO_MEASURE if value is None else value

    def _training_matrices(self, settings: Settings):
        """The training matrix and the held-out grid's, reused while the rows' settings stay the same."""
        key = (settings.features, settings.non_uk_rate, settings.zero_rate, settings.seed)
        if key != self._matrix_key:
            self._matrices = None  # free the device memory before building the next
            plan, built = self.plan, self.data.data.built_zero_rate
            held_out = outside_fold(plan.uk, plan.n_folds, plan.fold)
            zero_seed = self.data.zero_seed

            def keep(table: pl.DataFrame) -> pl.DataFrame:
                rows = subsample_zeros(table, built_rate=built, rate=settings.zero_rate, seed=zero_seed, uk=plan.uk)
                return held_out(drop_excluded(rows)).sort(KEYS)  # a fixed order: the same rows, the same model

            prepare = preparer(
                uk=plan.uk, transform=self.transform, non_uk_rate=settings.non_uk_rate, seed=settings.seed, keep=keep
            )
            train = training_matrix(self.chunks, prepare, device=plan.device, features=settings.features)
            grid = training_matrix([self.grid], ref=train, device=plan.device, features=settings.features)
            self._matrix_key, self._matrices = key, (train, grid)
        return self._matrices

    def _predict(self, model: xgb.Booster, meta: dict, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """Export Potential in dollars: the model at its chosen round count, blended with the naive baseline if asked."""
        predicted = export_potential(
            table, model, features=list(settings.features), rounds=meta["rounds"], device=self.plan.device
        )
        w = settings.naive_weight
        if w == 0:
            return predicted
        naive = table["naive_usd"].cast(pl.Float64).clip(lower_bound=MIN_PREDICTION_USD)
        return ((1 - w) * predicted.clip(lower_bound=MIN_PREDICTION_USD).log() + w * naive.log()).exp()

    def _save(self) -> None:
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / RESULTS_FILE
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(self.results, indent=2, default=experiments._json_default), encoding="utf-8")
        partial.replace(path)


def training_key(settings: Settings, fingerprint: Mapping) -> str:
    """A short hash of everything a trained model depends on: its training settings and its rows."""
    defaults = Settings(params={}).to_dict()
    training = {
        name: value
        for name, value in settings.to_dict().items()
        if name in KEYED_FIELDS or (name in TRAINING_FIELDS and value != defaults[name])
    }
    text = json.dumps({"settings": training, "rows": dict(fingerprint)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _best(experiment: str, row: Mapping) -> dict:
    return {
        "experiment": experiment,
        "arm": row["arm"],
        "settings": row["settings"],
        SELECTION_MEASURE: row[SELECTION_MEASURE],
    }


def _load(folder: Path | None) -> dict | None:
    if folder is None or not (folder / RESULTS_FILE).exists():
        return None
    return json.loads((folder / RESULTS_FILE).read_text(encoding="utf-8"))


def _log(message: str) -> None:
    print(message, flush=True)


# --- Data -------------------------------------------------------------------------------------


def fixture_data(world, *, validation_year: int = experiments.VALIDATION_YEAR) -> LoopData:
    """The fixture world's rows for every Product, built only from Flows up to the validation year."""
    rows = experiments.fixture_data(world, validation_year=validation_year)
    flows = world.flows.filter(pl.col("year") == validation_year)
    return LoopData(data=rows, flows=lambda: flows)


def real_data(root: Path | None = None, *, validation_year: int = experiments.VALIDATION_YEAR) -> LoopData:
    """Every Product's rows from `features.build()`'s files, and the validation year's Flows; never a later year."""
    experiments._check_years([validation_year])
    root = Path(root or data_dir())
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    rows = Data(
        training={y: folder / f"training_{y}.parquet" for y in settings["years"] if y < validation_year},
        validation=folder / f"training_{validation_year}.parquet",
        grid=folder / f"uk_grid_{validation_year}.parquet",
        validation_year=validation_year,
        built_zero_rate=settings["zero_sampling_rate"],
    )

    def flows() -> pl.DataFrame:
        country_map = pl.read_parquet(root / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
        return experiments._year_flows(root, validation_year, dict(country_map.iter_rows()))

    return LoopData(data=rows, flows=flows, zero_seed=settings.get("zero_sampling_seed", 0))


def loop_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"loop_{baci.BACI_VERSION}"


def load(root: Path | None = None) -> dict:
    """The saved results of the real loop."""
    return json.loads((loop_dir(root) / RESULTS_FILE).read_text(encoding="utf-8"))


def v2_plan(root: Path | None = None) -> LoopPlan:
    """The real loop's start: v2's configuration (the saved search winner) at the search's rates, on fold 0."""
    saved = search.load(root)
    return LoopPlan(
        start=Settings(
            params=dict(saved["winner"]["params"]),
            non_uk_rate=saved["non_uk_rate"],
            zero_rate=saved["plan"]["zero_rate"] or ZERO_SAMPLING_RATE,
        )
    )


def main(argv: Sequence[str]) -> None:
    command = argv[0] if argv else ""
    if command == "noise":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        floor = lp.noise_floor()
        summary = {k: floor[k] for k in ("seeds", "selection_measures", "sd", "win_threshold")}
        print(json.dumps({**summary, "seconds": [row["seconds"] for row in floor["arms"]]}), flush=True)
    elif command == "time":
        print(json.dumps(time_one_arm()), flush=True)
    else:
        raise SystemExit("usage: python -m exportspredictor.loop noise|time")


def time_one_arm(root: Path | None = None) -> dict:
    """Train the start configuration once more, in memory (the cache is not read), and save its timing.

    The timing is saved in the loop's results under `timing`; nothing else there changes.
    """
    started = time.perf_counter()
    lp = Loop(real_data(root), v2_plan(root))
    row = lp.run_arm(Arm("timing"), lp.plan.start)
    timing = {
        "setup_seconds": lp.results["setup_seconds"],
        **{k: row[k] for k in ("seconds", "matrix_seconds", "train_seconds", "rounds", "trained_rounds", "training_rows", SELECTION_MEASURE)},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    path = loop_dir(root) / RESULTS_FILE
    results = json.loads(path.read_text(encoding="utf-8"))
    results["timing"] = timing
    path.write_text(json.dumps(results, indent=2, default=experiments._json_default), encoding="utf-8")
    return timing


if __name__ == "__main__":
    main(sys.argv[1:])
