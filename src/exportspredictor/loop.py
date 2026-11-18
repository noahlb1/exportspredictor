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
    uv run python -m exportspredictor.loop naive   # #70: the blend (no training) and the starting point (one model)
    uv run python -m exportspredictor.loop select  # #71: v1, three confirmed v2 configurations, depth 6 (four models)
    uv run python -m exportspredictor.loop two-part  # #72: a binary "any Flow" part times a size part (two models)
    uv run python -m exportspredictor.loop scope   # #73: UK Zero Cells sampled; the zero rate at 2% (two models)
    uv run python -m exportspredictor.loop product-detail  # #74: three Product features, one experiment each (three models)
    uv run python -m exportspredictor.loop constraints  # #75: the monotone constraints on (cached) and off (one model)
    uv run python -m exportspredictor.loop confirm  # #76: the best arm beside v2 on all five folds (fold 0 cached)

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
    MIN_RANKED_ROWS,
    TWEEDIE_POWER,
    naive_from_totals,
    naive_potential,
    tweedie_deviance,
)
from exportspredictor.exclusions import drop_excluded
from exportspredictor.experiments import Data, TestYearRead, spearman_within_product
from exportspredictor.features import (
    FEATURES,
    KEYS,
    PRODUCT_DETAIL_FEATURES,
    ZERO_SAMPLING_RATE,
    FeatureTransform,
    features_dir,
    fit_product_detail,
    subsample_zeros,
)
from exportspredictor.folds import N_FOLDS, assign_folds, outside_fold
from exportspredictor.paths import data_dir
from exportspredictor.prediction import (
    MIN_PREDICTION_USD,
    ModelForm,
    blend,
    export_potential,
    flow_probability,
    two_part_potential,
)
from exportspredictor.training import (
    BASE_MARGIN,
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
MAX_TRAINED_MODELS = 6  # the budget of one experiment (spec #65); a cached model is free
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
    # The model's starting point: None (a constant, XGBoost's base score) or "naive" (the log of
    # the naive baseline, so the trees learn a correction to it).
    base_margin: str | None = None
    # True: a two-part model (#72), a binary model of "any Flow" times a size model fitted on
    # non-zero Flows; False: one Tweedie model.
    two_part: bool = False
    # True: UK Zero Cells sampled at `zero_rate` and weighted 1/rate as other Exporters' are
    # (#73, v1's rule); False: every UK Zero Cell kept at weight 1 (ADR 0004).
    uk_zeros_sampled: bool = False
    # True: the monotone constraints on policy features (ADR 0002, kept by ADR 0008 until
    # tested); False: none (#75).
    monotone: bool = True
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

    `grid` names one scoring setting (`SCORING_FIELDS`) and a fixed list of values for it:
    the arm scores its model with each value on the validation year and keeps the value
    with the highest selection measure. Every value's measures are recorded in the row.
    """

    name: str
    changes: Mapping = field(default_factory=dict)
    grid: Mapping[str, Sequence] = field(default_factory=dict)

    def apply(self, base: Settings) -> Settings:
        unknown = set(self.changes) - {f.name for f in fields(Settings)}
        if unknown:
            raise ValueError(f"Arm {self.name!r} changes unknown settings {sorted(unknown)}")
        if len(self.grid) > 1 or set(self.grid) - set(SCORING_FIELDS):
            raise ValueError(f"Arm {self.name!r}: a grid names one scoring setting of {SCORING_FIELDS}")
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
    # World exports of each Product in each training year (hs6, year, world_product_usd), for the
    # naive baseline of the training rows; read only by an arm that starts from it.
    world_products: Callable[[], pl.DataFrame] | None = None
    # Product features finer than Sector, fitted on the training years (`features.fit_product_detail`);
    # read only by an arm whose features include one of them.
    product_detail: Callable[[], pl.DataFrame] | None = None

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


def measures(grid: pl.DataFrame, validation: pl.DataFrame | None, prediction: str) -> dict:
    """The selection measure and every diagnostic of `prediction`.

    `grid` is the held-out UK grid of the validation year (hs6, value_usd); `validation`
    its training rows (value_usd, weight) for the deviance (None: no deviance, as for a rank).
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
        "tweedie_deviance": (
            None if validation is None
            else float(validation.select(tweedie_deviance(prediction, power=TWEEDIE_POWER)).item())
        ),
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



# --- The naive baseline in the model (#70) ----------------------------------------------------

# The weights of the naive baseline the blend chooses from. The ends are left out: 0 is the
# model alone (the best so far), and 1 is the naive baseline alone, which is no model at all
# (no Scenario could move it).
BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BLEND = "blend"
STARTING_POINT = "starting point"
NAIVE_EXPERIMENT = "naive baseline"  # experiments 1 and 2: the blend and the starting point, one decision


def starting_point_arm() -> Arm:
    """Experiment 2: the log of the naive baseline is the model's base margin; the trees learn the correction."""
    return Arm(STARTING_POINT, {"base_margin": "naive"})


def naive_margin(naive_usd: pl.Series) -> pl.Series:
    """The naive baseline as a starting point on the log scale, floored at MIN_PREDICTION_USD (a Zero Cell's is 0)."""
    return naive_usd.cast(pl.Float64).clip(lower_bound=MIN_PREDICTION_USD).log()


def naive_from_features(rows: pl.DataFrame, world_products: pl.DataFrame) -> pl.Series:
    """The naive baseline (`evaluation.naive_from_totals`) of untransformed feature rows, from their own features.

    The rows' leave-one-out supply and demand are already features (`log_supply_loo`,
    `log_demand_loo`); `world_products` (hs6, year, world_product_usd) gives the Product's world
    exports, from which the row's own Flow is left out. So the training rows need no Flows.
    """
    world = world_products.select(
        _p=pl.col("hs6").cast(pl.String),
        _y=pl.col("year").cast(pl.Int32),
        _world_usd=pl.col("world_product_usd").cast(pl.Float64),
    )
    keys = rows.select(
        _p=pl.col("hs6").cast(pl.String),
        _y=pl.col("year").cast(pl.Int32),
        _supply_usd=pl.col("log_supply_loo").cast(pl.Float64).exp() - 1,
        _demand_usd=pl.col("log_demand_loo").cast(pl.Float64).exp() - 1,
        _own_usd=pl.col("value_usd").cast(pl.Float64),
    ).join(world, on=["_p", "_y"], how="left", maintain_order="left")
    other_world_usd = pl.col("_world_usd").fill_null(0.0) - pl.col("_own_usd")
    return keys.select(
        naive_usd=naive_from_totals(pl.col("_supply_usd"), pl.col("_demand_usd"), other_world_usd)
    )["naive_usd"]


def world_products(flows: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """World exports of each Product in each year (hs6, year, world_product_usd), from non-zero Flows."""
    value = pl.col("value_usd")
    return (
        flows.lazy()
        .filter(value > 0)
        .group_by("hs6", "year")
        .agg(world_product_usd=value.cast(pl.Float64).sum())
        .collect()
    )


def blend_arm(weights: Sequence[float] = BLEND_WEIGHTS) -> Arm:
    """Experiment 1: the best so far's model blended with the naive baseline, the weight chosen on 2022."""
    return Arm(BLEND, grid={"naive_weight": tuple(weights)})


def rank_average_measures(grid: pl.DataFrame) -> dict:
    """The rank-average of the model (`model_usd`) and the naive baseline within each Product: a reference only.

    It has no dollar value, so no Export Gap and no deviance; it is never selected.
    """
    ranked = grid.with_columns(
        rank_average=(pl.col("model_usd").rank().over("hs6") + pl.col("naive_usd").rank().over("hs6")) / 2
    )
    return measures(ranked, None, "rank_average")

# --- Select on ranking (#71) ------------------------------------------------------------------

SELECT_EXPERIMENT = "select on ranking"  # experiment 3
SHALLOW_DEPTH = 6  # the one shallower configuration: the best so far's parameters at this depth


def select_arms(saved_search: Mapping, weights: Sequence[float] = BLEND_WEIGHTS) -> list[Arm]:
    """Experiment 3: five fixed configurations, no search, each chosen on the selection measure alone.

    v1's configuration (`search.V1_TRIAL`), the three configurations the search confirmed
    on every Product (`saved_search`, as `search.load()` returns it) and one shallower
    configuration. Their round counts are chosen by the selection measure while training.
    Each is scored as a blend with the naive baseline, its weight re-chosen from `weights`
    for that configuration's model, since the best so far is a blend.
    """
    eta = saved_search["plan"]["eta"]
    grid = {"naive_weight": tuple(weights)}
    confirmed = [
        Arm(f"search trial {row['trial']}", {"params": {**row["params"], "eta": eta}}, grid)
        for row in saved_search["confirmation"]
    ]
    return [
        Arm("v1", {"params": {**search.V1_TRIAL, "eta": eta}}, grid),
        *confirmed,
        Arm(f"depth {SHALLOW_DEPTH}", {"params": {"max_depth": SHALLOW_DEPTH}}, grid),
    ]


# --- The two-part model (#72) -----------------------------------------------------------------

TWO_PART_EXPERIMENT = "two-part model"  # experiment 4
TWO_PART = "two-part"
ANY_FLOW, SIZE = "any_flow", "size"  # the two parts, trained in this order
# The binary part keeps the best so far's tree parameters; only the objective changes.
ANY_FLOW_PARAMS = {"objective": "binary:logistic"}
TWEEDIE_ONLY = ("tweedie_variance_power",)


def two_part_arm(weights: Sequence[float] = BLEND_WEIGHTS) -> Arm:
    """Experiment 4: the best so far's configuration as a two-part model, blended with the naive baseline.

    The binary part learns whether a Flow exists, on every training row with its weight;
    the size part learns how large it is, on non-zero Flows. Export Potential is their
    product. Both keep the monotone constraints. The blend weight is re-chosen for it.
    """
    return Arm(TWO_PART, {"two_part": True}, {"naive_weight": tuple(weights)})


@dataclass(frozen=True)
class TwoPart:
    """A two-part model's Boosters."""

    any_flow: xgb.Booster
    size: xgb.Booster


# --- Split the scope change (#73) ------------------------------------------------------------

# Experiment 5, split in two experiments run in this order, so that each arm is compared with
# the best so far and is kept if it clears the win threshold (the second starts from the
# first's result). Both subsample the built feature rows; nothing is rebuilt.
UK_ZEROS_EXPERIMENT = "scope: UK Zero Cells"
ZERO_RATE_EXPERIMENT = "scope: zero rate"
UK_ZEROS_SAMPLED = "UK Zero Cells sampled"
LOWER_ZERO_RATE = 0.02  # v1's rate for other Exporters' Zero Cells
BLEND_GRID = {"naive_weight": BLEND_WEIGHTS}  # the blend weight re-chosen for each arm's model (#71)
SCOPE_EXPERIMENTS = (
    (UK_ZEROS_EXPERIMENT, [Arm(UK_ZEROS_SAMPLED, {"uk_zeros_sampled": True}, BLEND_GRID)]),
    (ZERO_RATE_EXPERIMENT, [Arm(f"zero rate {LOWER_ZERO_RATE:.0%}", {"zero_rate": LOWER_ZERO_RATE}, BLEND_GRID)]),
)


# --- Product features finer than Sector (#74) ------------------------------------------------

# Experiment 6, one experiment per feature, run in `PRODUCT_DETAIL_FEATURES` order: each adds
# its feature to the best so far's and is kept if it clears the win threshold, so the next
# starts from it. The features are fitted on the training years' Flows of other Exporters
# (`features.fit_product_detail`) and joined to the loop's rows only; nothing is rebuilt.


def product_detail_experiment(feature: str) -> str:
    return f"product detail: {feature}"


def product_detail_arm(feature: str, best: Settings) -> Arm:
    """The best so far's model with one more feature, the blend weight re-chosen for it."""
    return Arm(f"+ {feature}", {"features": (*best.features, feature)}, BLEND_GRID)


def run_product_detail(lp: "Loop") -> dict:
    """Run the three Product detail experiments in order, each from the best so far."""
    for feature in PRODUCT_DETAIL_FEATURES:
        lp.noise_floor()
        best = Settings.from_dict(lp.results["best"]["settings"])
        lp.run_experiment(product_detail_experiment(feature), [product_detail_arm(feature, best)])
    return lp.results


# --- The monotone constraints on and off (#75) ------------------------------------------------

# Experiment 7, which answers the open point of ADR 0008. "On" is the best so far itself (its
# cached model, nothing trained); "off" is the same settings with no monotone constraints.
# Both re-choose the blend weight. If "off" wins beyond the threshold the constraints go, and
# a further ADR says what serves Scenarios; otherwise they stay.
CONSTRAINTS_EXPERIMENT = "monotone constraints"


def constraint_arms(weights: Sequence[float] = BLEND_WEIGHTS) -> list[Arm]:
    grid = {"naive_weight": tuple(weights)}
    return [Arm("constraints on", {"monotone": True}, grid), Arm("constraints off", {"monotone": False}, grid)]


# --- The loop ---------------------------------------------------------------------------------


class Loop:
    """The experiment loop on one fold: the noise floor, then experiments, each against the best so far.

    With a `folder`, results are saved there as each arm finishes, models are cached in
    it, and a new `Loop` on the same folder resumes; without one, both live in memory.
    `models_dir` caches models elsewhere: the confirmation's fold loops share the loop's
    cache but keep no results of their own.
    """

    def __init__(
        self, data: LoopData, plan: LoopPlan, folder: Path | None = None, *, models_dir: Path | None = None
    ):
        experiments._check_years([data.validation_year])
        started = time.perf_counter()
        self.data, self.plan, self.folder = data, plan, folder
        self.models_dir = models_dir if models_dir is not None else (folder / MODELS_DIR if folder else None)
        self._models: dict[str, tuple[xgb.Booster | TwoPart, dict]] = {}
        self._matrix_key, self._matrices = None, None
        self._detail: pl.DataFrame | None = None
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
        if "decided" not in section:
            self._check_budget(name, section, arms, base)
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

    def _check_budget(self, name: str, section: dict, arms: Sequence[Arm], base: Settings) -> None:
        """Refuse, before training anything, an experiment that would train more than MAX_TRAINED_MODELS models.

        Counted: the models its finished arms trained and those its other arms would train
        (a model already cached is free; a two-part model is two).
        """
        models = {row["training_key"]: _models_in(row["settings"]) for row in section["arms"] if row.get("trained")}
        done = {row["arm"] for row in section["arms"]}
        for arm in arms:
            if arm.name in done:
                continue
            settings = self._resolved(arm.apply(base))
            key = training_key(settings, self._fingerprint())
            if key not in self._models and not self._model_file(key).exists():
                models.setdefault(key, _models_in(settings.to_dict()))
        if sum(models.values()) > MAX_TRAINED_MODELS:
            raise ValueError(
                f"Experiment {name!r} would train {sum(models.values())} models; the budget is at most "
                f"{MAX_TRAINED_MODELS} trained models per experiment (spec #65)"
            )

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
        model_grid = self._model_potential(model, meta, settings, self.grid)
        extra = {}
        if arm.grid:
            # Choose the scoring setting on the validation year's held-out UK grid only.
            (name, values), = arm.grid.items()
            by_value = []
            for value in values:
                value_settings = replace(settings, **{name: value})
                scored_grid = self.grid.with_columns(export_potential_usd=self._score(model_grid, value_settings, self.grid))
                by_value.append({name: value, **measures(scored_grid, None, "export_potential_usd")})
            chosen = max(by_value, key=lambda v: -2.0 if v[SELECTION_MEASURE] is None else v[SELECTION_MEASURE])
            settings = replace(settings, **{name: chosen[name]})
            extra["grid"] = by_value
        if arm.grid.keys() & {"naive_weight"}:
            extra["rank_average"] = rank_average_measures(self.grid.with_columns(model_usd=model_grid))
            # The model unblended (w = 0), a reference only: the blend's grid leaves 0 out.
            extra["model_alone"] = measures(self.grid.with_columns(model_usd=model_grid), None, "model_usd")
        if settings.two_part:
            extra["parts"] = meta["parts"]
        scored_grid = self.grid.with_columns(export_potential_usd=self._score(model_grid, settings, self.grid))
        scored = self.validation.with_columns(
            export_potential_usd=self._score(
                self._model_potential(model, meta, settings, self.validation), settings, self.validation
            )
        )
        return {
            "arm": arm.name,
            "changes": dict(arm.changes),
            "settings": settings.to_dict(),
            "training_key": key,
            "trained": trained,
            **measures(scored_grid, scored, "export_potential_usd"),
            **extra,
            "validation_year": self.data.validation_year,
            **meta,
            "seconds": round(time.perf_counter() - started, 1),
        }

    # --- Models --------------------------------------------------------------------------------

    def export_potential(self, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """Export Potential (USD) of transformed rows (with `naive_usd`) as an arm with `settings` scores them.

        The model is taken from the cache (or trained); e.g. to re-predict after a Scenario
        changes policy features.
        """
        settings = self._resolved(settings)
        model, meta = self._model(settings, training_key(settings, self._fingerprint()))
        return self._score(self._model_potential(model, meta, settings, table), settings, table)

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
        return Path(self.models_dir or ".nowhere") / f"{key}.ubj"

    def _model(self, settings: Settings, key: str) -> tuple[xgb.Booster | TwoPart, dict]:
        if key in self._models:
            return self._models[key]
        path = self._model_file(key)
        # A two-part model's binary part sits beside its size part, which is saved last.
        any_flow_path = path.with_name(f"{key}.{ANY_FLOW}.ubj")
        if self.models_dir is not None and path.exists():
            model = xgb.Booster(model_file=str(path))
            if settings.two_part:
                model = TwoPart(any_flow=xgb.Booster(model_file=str(any_flow_path)), size=model)
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        else:
            model, meta = self._train_two_part(settings) if settings.two_part else self._train(settings)
            if self.models_dir is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(model, TwoPart):
                    model.any_flow.save_model(str(any_flow_path))
                (model.size if isinstance(model, TwoPart) else model).save_model(str(path))
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._models[key] = (model, meta)
        return model, meta

    def _train_two_part(self, settings: Settings) -> tuple[TwoPart, dict]:
        """Train the binary part, then the size part with the two-part Export Potential measured each round.

        The binary part stops on the selection measure of its probability alone; the size
        part stops on the selection measure of probability × size, the binary part held at
        its chosen round. Both measures are the model alone, never the blend (#67's rule).
        The matrices are built one part at a time, so only one training matrix is on the device.
        """
        if settings.base_margin:
            raise ValueError("A two-part model has no starting point: base_margin and two_part do not combine")
        started = time.perf_counter()
        any_flow, any_flow_meta = self._train(settings, part=ANY_FLOW)
        probability = flow_probability(
            self.grid, any_flow, features=list(settings.features), rounds=any_flow_meta["rounds"],
            device=self.plan.device,
        ).to_numpy()
        self._matrix_key, self._matrices = None, None  # free the binary part's matrices first
        size, size_meta = self._train(settings, part=SIZE, grid_probability=probability)
        self._matrix_key, self._matrices = None, None
        return TwoPart(any_flow=any_flow, size=size), {
            "rounds": size_meta["rounds"],  # the size part's, chosen on the two-part Export Potential
            "hit_round_cap": any_flow_meta["hit_round_cap"] or size_meta["hit_round_cap"],
            "trained_rounds": size_meta["trained_rounds"],
            "training_rows": any_flow_meta["training_rows"],
            "matrix_seconds": round(any_flow_meta["matrix_seconds"] + size_meta["matrix_seconds"], 1),
            "train_seconds": round(time.perf_counter() - started, 1),
            "parts": {ANY_FLOW: any_flow_meta, SIZE: size_meta},
        }

    def _train(
        self, settings: Settings, *, part: str | None = None, grid_probability=None
    ) -> tuple[xgb.Booster, dict]:
        """Train with the selection measure on the held-out UK grid each round, stopping after `patience` flat rounds.

        `part` trains one part of a two-part model; the size part's measure each round is
        that of `grid_probability` × size.
        """
        started = time.perf_counter()
        train, grid = self._training_matrices(settings, part)
        matrix_seconds = round(time.perf_counter() - started, 1)  # 0 when the matrices were reused
        params = model_params(
            dict(settings.params), device=self.plan.device, seed=settings.seed, features=settings.features,
            monotone=settings.monotone,
        )
        if part == ANY_FLOW:
            params = {k: v for k, v in params.items() if k not in TWEEDIE_ONLY} | ANY_FLOW_PARAMS
        params |= {"disable_default_eval_metric": 1} | ({"nthread": self.plan.threads} if self.plan.threads else {})
        metric = self._round_measure
        if grid_probability is not None:

            def metric(predictions, matrix):
                return self._round_measure(grid_probability * predictions, matrix)

        model = xgb.train(
            params,
            train,
            num_boost_round=settings.max_rounds,
            evals=[(grid, "grid")],
            custom_metric=metric,
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
            "stopping_measure": float(model.best_score),  # the selection measure at `rounds`, as training saw it
            "training_rows": train.num_row(),
            "matrix_seconds": matrix_seconds,
            "train_seconds": round(time.perf_counter() - started, 1),  # matrices included
        }

    def _round_measure(self, predictions, _matrix) -> tuple[str, float]:
        value = spearman_within_product(self._grid_keys.with_columns(prediction=pl.Series(predictions)), "prediction")
        return SELECTION_MEASURE, NO_MEASURE if value is None else value

    def _training_matrices(self, settings: Settings, part: str | None = None):
        """The training matrix and the held-out grid's, reused while the rows' settings stay the same.

        For a two-part model's binary part (`ANY_FLOW`) the label is "any Flow" (0 or 1) and
        every row keeps its weight; its size part (`SIZE`) trains on non-zero Flows only.
        """
        key = (
            settings.features, settings.non_uk_rate, settings.zero_rate, settings.seed, settings.base_margin,
            settings.uk_zeros_sampled, part,
        )
        if key != self._matrix_key:
            self._matrices = None  # free the device memory before building the next
            plan, built = self.plan, self.data.data.built_zero_rate
            held_out = outside_fold(plan.uk, plan.n_folds, plan.fold)
            zero_seed = self.data.zero_seed
            world = self._world_products() if settings.base_margin else None
            grid_rows = self._with_product_detail(self.grid, settings)

            def keep(table: pl.DataFrame) -> pl.DataFrame:
                rows = subsample_zeros(
                    table, built_rate=built, rate=settings.zero_rate, seed=zero_seed, uk=plan.uk,
                    uk_zeros_sampled=settings.uk_zeros_sampled,
                )
                rows = held_out(drop_excluded(rows)).sort(KEYS)  # a fixed order: the same rows, the same model
                rows = self._with_product_detail(rows, settings)
                if part == ANY_FLOW:
                    rows = rows.with_columns(_any_flow(pl.col("value_usd")))
                elif part == SIZE:
                    rows = rows.filter(pl.col("value_usd") > 0)
                if world is None:
                    return rows
                # Before the transform clips the supply and demand features: the naive baseline as built.
                return rows.with_columns(naive_margin(naive_from_features(rows, world)).alias(BASE_MARGIN))

            if settings.base_margin:
                grid_rows = grid_rows.with_columns(naive_margin(self.grid["naive_usd"]).alias(BASE_MARGIN))
            if part == ANY_FLOW:  # a binary label on the grid too; the measure each round reads value_usd itself
                grid_rows = grid_rows.with_columns(_any_flow(pl.col("value_usd")))

            prepare = preparer(
                uk=plan.uk, transform=self.transform, non_uk_rate=settings.non_uk_rate, seed=settings.seed, keep=keep
            )
            train = training_matrix(self.chunks, prepare, device=plan.device, features=settings.features)
            grid = training_matrix([grid_rows], ref=train, device=plan.device, features=settings.features)
            self._matrix_key, self._matrices = key, (train, grid)
        return self._matrices

    def _model_potential(self, model: xgb.Booster | TwoPart, meta: dict, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """The model's Export Potential in dollars at its chosen round count."""
        table = self._with_product_detail(table, settings)
        if isinstance(model, TwoPart):
            parts = meta["parts"]
            return two_part_potential(
                table, model.any_flow, model.size, features=list(settings.features),
                rounds=(parts[ANY_FLOW]["rounds"], parts[SIZE]["rounds"]), device=self.plan.device,
            )
        margin = naive_margin(table["naive_usd"]) if settings.base_margin else None
        return export_potential(
            table, model, features=list(settings.features), rounds=meta["rounds"], device=self.plan.device,
            base_margin=margin,
        )

    def _with_product_detail(self, table: pl.DataFrame, settings: Settings) -> pl.DataFrame:
        """`table` with the Product detail features its settings use, joined on the Product in the same row order."""
        wanted = [f for f in settings.features if f in PRODUCT_DETAIL_FEATURES]
        if not wanted:
            return table
        if self._detail is None:
            if self.data.product_detail is None:
                raise ValueError("This loop's data has no Product detail features, so no arm can use them")
            self._detail = self.data.product_detail().select(pl.col("hs6").cast(pl.String).alias("_hs6"), *PRODUCT_DETAIL_FEATURES)
        return (
            table.drop(wanted, strict=False)
            .with_columns(_hs6=pl.col("hs6").cast(pl.String))
            .join(self._detail.select("_hs6", *wanted), on="_hs6", how="left", maintain_order="left")
            .drop("_hs6")
        )

    def _world_products(self) -> pl.DataFrame:
        if self.data.world_products is None:
            raise ValueError("This loop's data has no world exports by Product, so no arm can start from the naive baseline")
        return self.data.world_products()

    @staticmethod
    def _score(predicted: pl.Series, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """Export Potential in dollars as the arm scores it: the served blend (`prediction.blend`), the model alone at w = 0."""
        return pl.Series(
            "export_potential_usd",
            blend(predicted.to_numpy(), table["naive_usd"].to_numpy(), settings.naive_weight),
            dtype=pl.Float64,
        )

    def _save(self) -> None:
        if self.folder is not None:
            _save_results(self.folder, self.results)


def _any_flow(value: pl.Expr) -> pl.Expr:
    """The binary part's label: 1 where there is a Flow, 0 in a Zero Cell."""
    return (value > 0).cast(pl.Float64).alias("value_usd")


def _models_in(settings: Mapping) -> int:
    """How many models an arm with these settings trains: two for a two-part model, else one."""
    return 2 if settings.get("two_part") else 1


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


# --- The five-fold confirmation (#76) ---------------------------------------------------------

# The best arm after the experiments ("winner") beside the current v2 configuration ("v2") on
# every fold: each fold's model scores only the UK pairs it held out (ADR 0001), and the
# selection measure is taken once over every UK pair pooled, as the headline measure is.
CONFIRMATION = "confirmation"
WINNER, V2 = "winner", "v2"


def confirm(
    data: LoopData, plan: LoopPlan, winner: Settings, v2: Settings, *, threshold: float, folder: Path | None = None
) -> dict:
    """The winner and v2 trained on each of the plan's folds and measured on every held-out UK pair of the validation year.

    Each fold has its own `Loop` (fold `0..n_folds − 1`) sharing the loop's model cache, so
    a fold model the experiments already trained is read, not retrained, and a winner that
    differs from v2 only in scoring (`SCORING_FIELDS`) shares v2's model. Each fold's row
    holds every measure on that fold alone (deviance on its validation rows). The pooled
    rows (`winner`, `v2`, `naive`) hold the measures over every fold's held-out UK pairs
    together; their deviance is the mean of the folds'. `beats_v2` is True when the
    winner beats v2 by more than `threshold`.

    With a `folder`, models are cached in its `models`, each fold's scored grid is saved in
    `confirmation/fold_<k>.parquet` and the section in `results.json` under `confirmation`,
    as each fold finishes; a re-run with the same settings skips finished folds. The
    validation year must not be the test year: `Loop` refuses it.
    """
    experiments._check_years([data.validation_year])
    arms = {V2: v2, WINNER: winner}  # v2 first: a winner that only re-scores reuses its model
    settings = {name: s.to_dict() for name, s in arms.items()}
    results = _load(folder) or {}
    section = results.get(CONFIRMATION, {})
    if section.get("settings") != settings or section.get("n_folds") != plan.n_folds:
        section = {"settings": settings, "n_folds": plan.n_folds, "validation_year": data.validation_year, "folds": []}
    folds_dir = folder / CONFIRMATION if folder is not None else None
    models_dir = folder / MODELS_DIR if folder is not None else None
    grids: dict[int, pl.DataFrame] = {}
    for fold in range(plan.n_folds):
        saved = folds_dir / f"fold_{fold}.parquet" if folds_dir is not None else None
        if saved is not None and saved.exists() and any(row["fold"] == fold for row in section["folds"]):
            grids[fold] = pl.read_parquet(saved)
            continue
        section["folds"] = [row for row in section["folds"] if row["fold"] != fold]
        row, grids[fold] = _confirm_fold(data, replace(plan, fold=fold), arms, models_dir)
        section["folds"] = sorted([*section["folds"], row], key=lambda r: r["fold"])
        if saved is not None:
            saved.parent.mkdir(parents=True, exist_ok=True)
            grids[fold].write_parquet(saved)
        if folder is not None:
            _save_results(folder, results | {CONFIRMATION: section})
        _log(f"fold {fold}: {json.dumps({n: row[n][SELECTION_MEASURE] for n in arms})}")
    pooled = pl.concat([grids[fold] for fold in range(plan.n_folds)])
    for name in arms:
        deviances = [row[name]["tweedie_deviance"] for row in section["folds"]]
        section[name] = {
            **measures(pooled, None, name),
            "tweedie_deviance": None if None in deviances else statistics.fmean(deviances),
            "settings": settings[name],
            "rounds": [row[name]["rounds"] for row in section["folds"]],
        }
    section["naive"] = measures(pooled, None, "naive_usd")
    difference = section[WINNER][SELECTION_MEASURE] - section[V2][SELECTION_MEASURE]
    section |= {
        "difference": difference,
        "win_threshold": threshold,
        "beats_v2": difference > threshold + ROUNDING,
    }
    if folder is not None:
        _save_results(folder, results | {CONFIRMATION: section})
    return section


def _confirm_fold(
    data: LoopData, plan: LoopPlan, arms: Mapping[str, Settings], models_dir: Path | None
) -> tuple[dict, pl.DataFrame]:
    """One fold: each Arm's model (cached or trained) scores the fold's held-out UK grid and validation rows."""
    started = time.perf_counter()
    lp = Loop(data, plan, models_dir=models_dir)
    grid = lp.grid.select("hs6", "importer", "value_usd", "naive_usd")
    row: dict = {"fold": plan.fold, "uk_pairs": lp.grid.height, "setup_seconds": lp.results["setup_seconds"]}
    for name, settings in arms.items():
        resolved = lp._resolved(settings)
        key = training_key(resolved, lp._fingerprint())
        trained = key not in lp._models and not lp._model_file(key).exists()
        _, meta = lp._model(resolved, key)
        grid = grid.with_columns(lp.export_potential(resolved, lp.grid).alias(name))
        validation = lp.validation.select("value_usd", "weight").with_columns(
            lp.export_potential(resolved, lp.validation).alias(name)
        )
        row[name] = {
            **measures(grid, validation, name),
            "training_key": key,
            "trained": trained,
            **{k: meta[k] for k in ("rounds", "trained_rounds", "training_rows", "train_seconds") if k in meta},
        }
        del validation
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row, grid


def _save_results(folder: Path, results: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RESULTS_FILE
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(results, indent=2, default=experiments.json_default), encoding="utf-8")
    partial.replace(path)


# --- Data -------------------------------------------------------------------------------------


def fixture_data(world, *, validation_year: int = experiments.VALIDATION_YEAR) -> LoopData:
    """The fixture world's rows for every Product, built only from Flows up to the validation year."""
    rows = experiments.fixture_data(world, validation_year=validation_year)
    flows = world.flows.filter(pl.col("year") == validation_year)
    world_exports = world_products(world.flows.filter(pl.col("year") < validation_year))
    detail = fit_product_detail(world.flows, sorted(rows.training), uk=baci.UK)
    return LoopData(data=rows, flows=lambda: flows, world_products=lambda: world_exports, product_detail=lambda: detail)


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

    def world_exports() -> pl.DataFrame:
        # Each training year's own file, so no later year is ever opened.
        files = [str(experiments._flows_file(root, y)) for y in sorted(rows.training)]
        return world_products(pl.scan_parquet(files).select("hs6", "year", "value_usd"))

    def product_detail() -> pl.DataFrame:
        # Each training year's own file, so no later year is ever opened.
        years = sorted(rows.training)
        files = [str(experiments._flows_file(root, y)) for y in years]
        flows = pl.scan_parquet(files).select("exporter", "importer", "hs6", "year", "value_usd")
        return fit_product_detail(flows, years, uk=baci.UK)

    return LoopData(
        data=rows, flows=flows, zero_seed=settings.get("zero_sampling_seed", 0), world_products=world_exports,
        product_detail=product_detail,
    )


def loop_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"loop_{baci.BACI_VERSION}"


def load(root: Path | None = None) -> dict:
    """The saved results of the real loop."""
    return json.loads((loop_dir(root) / RESULTS_FILE).read_text(encoding="utf-8"))


# --- What the pipeline serves (#77) -------------------------------------------------------------

RECORDED = ("selection_measure", "validation_year", "fold", "n_folds", "naive", "noise_floor", "experiments", "best",
            CONFIRMATION)  # the parts of the results an artefact records


def chosen_form(results: Mapping) -> ModelForm:
    """The model form the loop chose (its best so far), as the pipeline serves it.

    Only the forms the pipeline can serve: the model alone or a blend, on the package's
    features, with the monotone constraints. Any other winner raises ValueError, so an
    artefact never serves a form other than the one its experiments chose.
    """
    settings = Settings.from_dict(results["best"]["settings"])
    if settings.base_margin is not None or settings.two_part:
        raise ValueError(f"The loop chose a form the pipeline does not serve: {results['best']['arm']!r}")
    if list(settings.features) != list(FEATURES) or not settings.monotone:
        raise ValueError("The loop's best so far changed the features or the constraints; the pipeline serves neither")
    return ModelForm(naive_weight=settings.naive_weight)


@dataclass(frozen=True)
class ServedConfiguration:
    """What the pipeline serves of the loop's choice: the form, the configuration and its round count.

    `rounds` was chosen by the selection measure on the loop's training rows (the years before
    the validation year, at `non_uk_rate`); the pipeline scales it to its own rows. Deviance
    chose none of it.
    """

    form: ModelForm
    params: dict
    rounds: int
    rounds_rule: str  # how `rounds` was taken from the loop's results
    non_uk_rate: float
    validation_year: int


def served_configuration(results: Mapping) -> ServedConfiguration:
    """The loop's best so far as the pipeline serves it, with the round count its selection measure chose.

    With a five-fold confirmation, the rounds are the median (lower) of the winner's per-fold
    rounds, each chosen by the selection measure on that fold's held-out UK pairs; the
    confirmation must be of the best so far's settings. Without one (the fixture), they are the
    best arm's own. Raises ValueError for a form the pipeline cannot serve (`chosen_form`) or a
    best so far that changed how Zero Cells are sampled (the pipeline trains on the rows as built).
    """
    form = chosen_form(results)
    best = Settings.from_dict(results["best"]["settings"])
    start = Settings.from_dict(results[NOISE_FLOOR]["arms"][0]["settings"])
    if best.zero_rate != start.zero_rate or best.uk_zeros_sampled:
        raise ValueError("The loop's best so far changed how Zero Cells are sampled; the pipeline trains on the built rows")
    confirmation = results.get(CONFIRMATION)
    if confirmation is not None:
        confirmed = Settings.from_dict(confirmation["settings"][WINNER])
        if confirmed.to_dict() != best.to_dict():
            raise ValueError("The five-fold confirmation is of other settings than the loop's best so far; re-run it")
        fold_rounds = confirmation[WINNER]["rounds"]
        rounds = statistics.median_low(fold_rounds)
        rule = f"median of the five-fold confirmation's rounds {fold_rounds}"
    else:
        section = results[NOISE_FLOOR] if results["best"]["experiment"] == NOISE_FLOOR else (
            results["experiments"][results["best"]["experiment"]]
        )
        rounds = next(row["rounds"] for row in section["arms"] if row["arm"] == results["best"]["arm"])
        rule = "the best arm's own rounds (no five-fold confirmation)"
    return ServedConfiguration(
        form=form, params=dict(best.params), rounds=int(rounds), rounds_rule=rule, non_uk_rate=best.non_uk_rate,
        validation_year=results["validation_year"],
    )


def ranking_record(results: Mapping) -> dict:
    """What an artefact records of the loop: the selection measure, the noise floor, every experiment's
    rows, the best so far and the five-fold confirmation (where run)."""
    return {key: results[key] for key in RECORDED if key in results}


CONFIRMED_NAIVE_WEIGHT = 0.3  # the real loop's blend weight, confirmed on five folds (#76)


def fixture_ranking(world) -> dict:
    """The fixture world's loop results: the noise floor and the blend experiment (CPU, under a second).

    The blend sweeps only `CONFIRMED_NAIVE_WEIGHT`, the weight the real loop chose and confirmed (#76),
    so the fixture artefact serves the same form as the real one; the fixture's 20 UK pairs cannot
    choose a weight of their own.
    """
    lp = Loop(fixture_data(world), FIXTURE_LOOP)
    lp.noise_floor()
    return lp.run_experiment(NAIVE_EXPERIMENT, [blend_arm((CONFIRMED_NAIVE_WEIGHT,))])


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
    elif command == "naive":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(NAIVE_EXPERIMENT, [blend_arm(), starting_point_arm()])["experiments"][NAIVE_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "select":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(SELECT_EXPERIMENT, select_arms(search.load()))["experiments"][SELECT_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "two-part":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(TWO_PART_EXPERIMENT, [two_part_arm()])["experiments"][TWO_PART_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "scope":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        for name, arms in SCOPE_EXPERIMENTS:
            section = lp.run_experiment(name, arms)["experiments"][name]
            print(json.dumps({name: section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "product-detail":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        results = run_product_detail(lp)
        for feature in PRODUCT_DETAIL_FEATURES:
            section = results["experiments"][product_detail_experiment(feature)]
            print(json.dumps({feature: section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "constraints":
        lp = Loop(real_data(), v2_plan(), loop_dir())
        section = lp.run_experiment(CONSTRAINTS_EXPERIMENT, constraint_arms())["experiments"][CONSTRAINTS_EXPERIMENT]
        print(json.dumps({"kept": section["kept"], **{r["arm"]: r[SELECTION_MEASURE] for r in section["arms"]}}), flush=True)
    elif command == "confirm":
        section = confirm_best()
        summary = {n: section[n][SELECTION_MEASURE] for n in (WINNER, V2, "naive")}
        print(json.dumps({**summary, **{k: section[k] for k in ("difference", "win_threshold", "beats_v2")}}), flush=True)
    elif command == "time":
        print(json.dumps(time_one_arm()), flush=True)
    else:
        raise SystemExit(
            "usage: python -m exportspredictor.loop noise|naive|select|two-part|scope|product-detail|constraints|confirm|time"
        )


def confirm_best(root: Path | None = None) -> dict:
    """#76: the best arm so far beside v2 (the noise floor's first seed, the model alone) on all five folds."""
    results = load(root)
    v2 = Settings.from_dict(results[NOISE_FLOOR]["arms"][0]["settings"])
    winner = Settings.from_dict(results["best"]["settings"])
    plan = v2_plan(root)
    return confirm(
        real_data(root), replace(plan, fold=0), winner, v2,
        threshold=results[NOISE_FLOOR]["win_threshold"], folder=loop_dir(root),
    )


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
    path.write_text(json.dumps(results, indent=2, default=experiments.json_default), encoding="utf-8")
    return timing


if __name__ == "__main__":
    main(sys.argv[1:])
