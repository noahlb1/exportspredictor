"""Arms and their settings, the plan a loop runs, the model cache key and the definitions of the seven experiments (spec #65)."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace

from exportspredictor import baci, market_reach, search
from exportspredictor.features import (
    EXPORTER_COUNT_FEATURES,
    FEATURES,
    SIBLING_DEMAND_FEATURES,
)
from exportspredictor.folds import N_FOLDS
from exportspredictor.loop.selection import MIN_WIN_THRESHOLD
from exportspredictor.loop.trainers import CATBOOST, LIGHTGBM, XGBOOST

MAX_TRAINED_MODELS = 6  # the budget of one experiment (spec #65); a cached model is free
NOISE_SEEDS = (0, 1, 2)
MAX_ROUNDS = 5000
# Rounds without a better selection measure before training stops. The measure is noisy
# round to round, so this is wider than the search's 50 on deviance.
PATIENCE = 100
NOISE_FLOOR = "noise_floor"


@dataclass(frozen=True)
class Settings:
    """Everything that decides one arm's model (`TRAINING_FIELDS`) and how it is scored (`SCORING_FIELDS`)."""

    # The family's parameters: XGBoost's over `training.DEFAULT_PARAMS` (eta included); another family's over
    # its trainer's defaults (`trainers.LIGHTGBM_DEFAULT_PARAMS`, `trainers.CATBOOST_DEFAULT_PARAMS`).
    params: Mapping
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
    # The model family, which picks the trainer (`trainers.TRAINERS`, #165). `params` are that family's.
    family: str = XGBOOST
    # Scoring only: Export Potential = exp((1 − w) · log model + w · log naive). 0 is the model alone.
    naive_weight: float = 0.0
    # Scoring only: Market Reach's beta (`market_reach`, spec #113, ticket #127). 0 scores without it;
    # otherwise each held-out UK pair's Export Potential is multiplied by exp(beta · offset) of its fold
    # and Market, learned from the year before, and rescaled to keep the scored rows' summed Export Potential.
    market_reach_beta: float = 0.0
    # Scoring only (spec #155, ticket #163): the highest Market Reach multiplier exp(beta · offset) any Market
    # takes, before the rescale; None caps nothing.
    market_reach_cap: float | None = None

    def to_dict(self) -> dict:
        data = {**asdict(self), "params": dict(self.params), "features": list(self.features)}
        # Left out at their defaults, so settings saved before they existed compare equal (the confirmation's).
        if not self.market_reach_beta:
            del data["market_reach_beta"]
        if self.market_reach_cap is None:
            del data["market_reach_cap"]
        if self.family == XGBOOST:
            del data["family"]
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "Settings":
        return cls(**{**data, "params": dict(data["params"]), "features": tuple(data["features"])})


SCORING_FIELDS = ("naive_weight", "market_reach_beta", "market_reach_cap")
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
    field is replaced. An Arm naming another `family` replaces the parameters with its own.

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
        if changes.get("family", base.family) != base.family:
            changes["params"] = dict(changes.get("params", {}))  # another family's own, never merged into these
        elif "params" in changes:
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


# The fixture world has a few hundred rows and a UK grid of a few dozen pairs, too few to
# rank in a fifth or a half of them: one "fold" holds every UK pair out, so the model learns
# from other Exporters only. Small trees, every row and one thread.
FIXTURE_LOOP = LoopPlan(
    start=Settings(params={"max_depth": 2, "eta": 0.3}, non_uk_rate=1.0, max_rounds=60, patience=10),
    n_folds=1,
    device="cpu",
    threads=1,
)


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


def blend_arm(weights: Sequence[float] = BLEND_WEIGHTS) -> Arm:
    """Experiment 1: the best so far's model blended with the naive baseline, the weight chosen on 2022."""
    return Arm(BLEND, grid={"naive_weight": tuple(weights)})


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


TWO_PART_EXPERIMENT = "two-part model"  # experiment 4
TWO_PART = "two-part"
ANY_FLOW, SIZE = "any_flow", "size"  # the two parts, trained in this order


def two_part_arm(weights: Sequence[float] = BLEND_WEIGHTS) -> Arm:
    """Experiment 4: the best so far's configuration as a two-part model, blended with the naive baseline.

    The binary part learns whether a Flow exists, on every training row with its weight;
    the size part learns how large it is, on non-zero Flows. Export Potential is their
    product. Both keep the monotone constraints. The blend weight is re-chosen for it.
    """
    return Arm(TWO_PART, {"two_part": True}, {"naive_weight": tuple(weights)})


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


# Experiment 6, one experiment per feature, run in `PRODUCT_DETAIL_FEATURES` order: each adds
# its feature to the best so far's and is kept if it clears the win threshold, so the next
# starts from it. The features are fitted on the training years' Flows of other Exporters
# (`features.fit_product_detail`) and joined to the loop's rows only; nothing is rebuilt.



def product_detail_experiment(feature: str) -> str:
    return f"product detail: {feature}"


def product_detail_arm(feature: str, best: Settings) -> Arm:
    """The best so far's model with one more feature, the blend weight re-chosen for it."""
    return Arm(f"+ {feature}", {"features": (*best.features, feature)}, BLEND_GRID)


# Experiment 7, which answers the open point of ADR 0008. "On" is the best so far itself (its
# cached model, nothing trained); "off" is the same settings with no monotone constraints.
# Both re-choose the blend weight. If "off" wins beyond the threshold the constraints go, and
# a further ADR says what serves Scenarios; otherwise they stay.
CONSTRAINTS_EXPERIMENT = "monotone constraints"


def constraint_arms(weights: Sequence[float] = BLEND_WEIGHTS) -> list[Arm]:
    grid = {"naive_weight": tuple(weights)}
    return [Arm("constraints on", {"monotone": True}, grid), Arm("constraints off", {"monotone": False}, grid)]


# Spec #113, Phase 2 (#127): the Exporter count on the rebuilt `exporter_count` files (which also fill
# missing GDP). Its base is the best so far retrained on those files (`Loop.run_experiment`'s
# `base_arm`), and both are scored with Market Reach at the adopted beta, so the feature is kept only
# if it adds something on top of Reach. The base keeps the best so far's blend weight; the Arm
# re-chooses its own, as every Arm with a new model has (#71).
EXPORTER_COUNT_EXPERIMENT = "exporter count"
REBUILT_BASE = "base on rebuilt files"


def exporter_count_arms(best: Settings, beta: float = market_reach.CHOSEN_BETA) -> tuple[Arm, list[Arm]]:
    """The base Arm (the best so far with Market Reach) and the Arm adding the Exporter count to it."""
    (feature,) = EXPORTER_COUNT_FEATURES
    base = Arm(REBUILT_BASE, {"market_reach_beta": beta})
    return base, [Arm(f"+ {feature}", {"features": (*best.features, feature)}, BLEND_GRID)]


# Spec #155, the final round. Market Reach was adopted outside the loop (#124's five-fold gate), so the
# loop's recorded best so far carries no Market Reach until an Arm of this round is kept. Every experiment of
# the round runs against `served_base_arm` of the best so far (`Loop.run_experiment`'s `base_arm`): the best so
# far as the pipeline serves it, on this loop's files.
SERVED_BASE = "best so far as served"


def served_base_arm(best: Settings, beta: float = market_reach.CHOSEN_BETA) -> Arm:
    """The best so far as served: with Market Reach at `beta` where its settings carry none, else unchanged
    (a kept Arm of this round already carries Reach and any cap)."""
    return Arm(SERVED_BASE, {} if best.market_reach_beta else {"market_reach_beta": beta})


# #163: the cap on Market Reach's upside. The grid of multipliers was fixed before any 2022 result was read
# (docs/research/final-round-2022.md); at beta 4 it caps offsets above log(cap) / 4, from 0.17 (2×) to 0.69 (16×).
MARKET_REACH_CAP_EXPERIMENT = "market reach cap"
MARKET_REACH_CAP = "cap Market Reach"
MARKET_REACH_CAP_GRID = (2.0, 4.0, 8.0, 16.0)


def market_reach_cap_arm(caps: Sequence[float] = MARKET_REACH_CAP_GRID) -> Arm:
    """The best so far as served with the upward Market Reach multiplier capped, the cap chosen on 2022 from `caps`."""
    return Arm(MARKET_REACH_CAP, grid={"market_reach_cap": tuple(caps)})


# #164: Sibling Demand, the UK-free demand proxy (idea 9 of the open review), as a model feature. It runs on the
# `sibling_demand` files, the served `exporter_count` files with the feature stored, against the best so far as
# served retrained on them. A dropped Arm leaves the served files as they were (`Loop.run_experiment`'s
# `files_kept_if_dropped`). The Arm re-chooses its blend weight, as every Arm with a new model has (#71).
SIBLING_DEMAND_EXPERIMENT = "sibling demand"
SIBLING_DEMAND_SET = "sibling_demand"
SIBLING_DEMAND_BASE_SET = "exporter_count"  # the served files `sibling_demand` extends by one column


def sibling_demand_arm(best: Settings) -> Arm:
    """The best so far as served with Sibling Demand added to its features."""
    (feature,) = SIBLING_DEMAND_FEATURES
    return Arm(f"+ {feature}", {"features": (*best.features, feature)}, BLEND_GRID)


# #166: a fresh hyperparameter search, in stages run in this order, each one experiment of at most
# MAX_TRAINED_MODELS models against the best so far as served (its model cached, so each Arm is one model).
# Winners stack: a kept stage becomes the best so far, and the next stage's Arms change its parameters. Each
# Arm names only the parameters it changes (merged into the best so far's) and re-chooses its blend weight.
# The grids were fixed before any 2022 result of the stage was read (docs/research/final-round-2022.md).
STOPPING_RULE_ROUNDS = 50  # a best round below this: the stopping rule, not the configuration, decided the Arm


def _param_arm(**params: float) -> Arm:
    return Arm(" ".join(f"{k} {v:g}" for k, v in params.items()), {"params": params}, BLEND_GRID)


HYPERPARAMETER_STAGES: dict[str, list[Arm]] = {
    "depth": [_param_arm(max_depth=d) for d in (8, 10, 14)],
    "learning rate": [_param_arm(eta=eta) for eta in (0.05, 0.2)],
    "min child weight": [_param_arm(min_child_weight=w) for w in (1.0, 10.0, 30.0)],
    "subsampling": [
        _param_arm(subsample=0.7),
        _param_arm(colsample_bytree=0.6),
        _param_arm(subsample=0.7, colsample_bytree=0.6),
    ],
    "regularisation": [_param_arm(reg_lambda=lam) for lam in (1.0, 20.0, 50.0)],
}


def hyperparameter_experiment(stage: str) -> str:
    return f"hyperparameters: {stage}"


# #167: other model families, one experiment each against the best so far as served, on the same rows,
# weights, features, fold and stopping rule, with the monotone constraints on the policy features. Each Arm's
# parameters are its family's own, translated from the best so far's XGBoost parameters where the family has
# an equivalent, and fixed before any 2022 result was read (docs/research/final-round-2022.md). Each Arm
# re-chooses its blend weight and keeps Market Reach (from the served base). The best so far's parameters:
_BEST = {
    "tweedie_variance_power": 1.2655329893555092, "min_child_weight": 3.0468601316016213,
    "subsample": 0.9158324106709628, "colsample_bytree": 0.8794644314402613, "reg_lambda": 4.953922320798769,
}
MODEL_FAMILY_ARMS: dict[str, Arm] = {
    LIGHTGBM: Arm("LightGBM", {"family": LIGHTGBM, "params": {
        "tweedie_variance_power": _BEST["tweedie_variance_power"], "learning_rate": 0.1,
        "max_depth": 12, "num_leaves": 1023, "min_sum_hessian_in_leaf": _BEST["min_child_weight"],
        "min_data_in_leaf": 1, "bagging_fraction": _BEST["subsample"], "bagging_freq": 1,
        "feature_fraction": _BEST["colsample_bytree"], "lambda_l2": _BEST["reg_lambda"],
    }}, BLEND_GRID),
    CATBOOST: Arm("CatBoost", {"family": CATBOOST, "params": {
        "tweedie_variance_power": _BEST["tweedie_variance_power"], "learning_rate": 0.1,
        "depth": 8, "l2_leaf_reg": _BEST["reg_lambda"], "bootstrap_type": "Bernoulli",
        "subsample": _BEST["subsample"], "rsm": _BEST["colsample_bytree"],
    }}, BLEND_GRID),
}


def model_family_experiment(family: str) -> str:
    return f"model family: {family}"


def decided_by_stopping_rule(rows: Sequence[Mapping]) -> list[str]:
    """The Arms whose best round (or a two-part model's either part's) is below `STOPPING_RULE_ROUNDS`."""

    def best_rounds(row: Mapping) -> list[int]:
        return [part["rounds"] for part in row["parts"].values()] if row.get("parts") else [row["rounds"]]

    return [row["arm"] for row in rows if min(best_rounds(row)) < STOPPING_RULE_ROUNDS]


# The research note numbers the seven experiments (docs/research/opportunity-ranking-experiments.md).
# The loop records nine decisions of the rule: experiments 1 and 2 share one, and experiments 5 and 6
# decide each Arm on its own. These map a recorded decision and its Arms back to the numbers.
EXPERIMENT_NUMBERS = {
    SELECT_EXPERIMENT: 3, TWO_PART_EXPERIMENT: 4, UK_ZEROS_EXPERIMENT: 5, ZERO_RATE_EXPERIMENT: 5,
    CONSTRAINTS_EXPERIMENT: 7,
}
NAIVE_ARM_NUMBERS = {BLEND: 1, STARTING_POINT: 2}
# What a decision is called where it is shown, where its recorded name is not the Glossary's.
EXPERIMENT_DISPLAY_NAMES = {
    NAIVE_EXPERIMENT: "Naive Baseline",
    EXPORTER_COUNT_EXPERIMENT: "Exporter count",
    SIBLING_DEMAND_EXPERIMENT: "Sibling Demand",
}


def experiment_numbers(name: str, arms: Sequence[str]) -> tuple[int, ...]:
    """The research note's numbers for a recorded decision with these Arms; () for a decision it does not number."""
    if name == NAIVE_EXPERIMENT:
        return tuple(sorted({NAIVE_ARM_NUMBERS[arm] for arm in arms if arm in NAIVE_ARM_NUMBERS}))
    if name.startswith(product_detail_experiment("")):
        return (6,)
    return (EXPERIMENT_NUMBERS[name],) if name in EXPERIMENT_NUMBERS else ()


def experiment_label(name: str, arms: Sequence[str]) -> str:
    """A recorded decision as shown: its experiment number(s), then its name in the Glossary's words ("1–2. Naive Baseline")."""
    numbers = experiment_numbers(name, arms)
    shown = EXPERIMENT_DISPLAY_NAMES.get(name, name)
    if not numbers:
        return shown
    span = str(numbers[0]) if len(numbers) == 1 else f"{numbers[0]}–{numbers[-1]}"
    return f"{span}. {shown}"


def count_experiments(decisions: Mapping[str, Sequence[str]]) -> int:
    """How many of the research note's experiments these recorded decisions (name -> Arms) cover.
    A decision the note does not number counts as one experiment."""
    numbered = [experiment_numbers(name, arms) for name, arms in decisions.items()]
    return len({n for numbers in numbered for n in numbers}) + sum(not numbers for numbers in numbered)


def models_trained_by(settings: Mapping) -> int:
    """How many models an arm with these settings trains: two for a two-part model, else one."""
    return 2 if settings.get("two_part") else 1


def training_key(settings: Settings, fingerprint: Mapping) -> str:
    """A short hash of everything a trained model depends on: its training settings and its rows."""
    defaults = Settings(params={}).to_dict()
    training = {
        name: value
        for name, value in settings.to_dict().items()
        # A field `to_dict` leaves out at its default (`family`) is missing from `defaults` and keyed when present.
        if name in KEYED_FIELDS or (name in TRAINING_FIELDS and (name not in defaults or value != defaults[name]))
    }
    text = json.dumps({"settings": training, "rows": dict(fingerprint)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
