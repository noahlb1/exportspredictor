"""Arms and their settings, the plan a loop runs, the model cache key and the definitions of the seven experiments (spec #65)."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace

from exportspredictor import baci, market_reach, search
from exportspredictor.features import EXPORTER_COUNT_FEATURES, FEATURES
from exportspredictor.folds import N_FOLDS
from exportspredictor.loop.selection import MIN_WIN_THRESHOLD

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
    # Scoring only: Market Reach's beta (`market_reach`, spec #113, ticket #127). 0 scores without it;
    # otherwise each held-out UK pair's Export Potential is multiplied by exp(beta · offset) of its fold
    # and Market, learned from the year before, and rescaled to keep the scored rows' summed Export Potential.
    market_reach_beta: float = 0.0

    def to_dict(self) -> dict:
        data = {**asdict(self), "params": dict(self.params), "features": list(self.features)}
        # Left out at its default, so settings saved before it existed compare equal (the confirmation's).
        if not self.market_reach_beta:
            del data["market_reach_beta"]
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "Settings":
        return cls(**{**data, "params": dict(data["params"]), "features": tuple(data["features"])})


SCORING_FIELDS = ("naive_weight", "market_reach_beta")
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


# The research note numbers the seven experiments (docs/research/opportunity-ranking-experiments.md).
# The loop records nine decisions of the rule: experiments 1 and 2 share one, and experiments 5 and 6
# decide each Arm on its own. These map a recorded decision and its Arms back to the numbers.
EXPERIMENT_NUMBERS = {
    SELECT_EXPERIMENT: 3, TWO_PART_EXPERIMENT: 4, UK_ZEROS_EXPERIMENT: 5, ZERO_RATE_EXPERIMENT: 5,
    CONSTRAINTS_EXPERIMENT: 7,
}
NAIVE_ARM_NUMBERS = {BLEND: 1, STARTING_POINT: 2}
# What a decision is called where it is shown, where its recorded name is not the Glossary's.
EXPERIMENT_DISPLAY_NAMES = {NAIVE_EXPERIMENT: "Naive Baseline", EXPORTER_COUNT_EXPERIMENT: "Exporter count"}


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
        if name in KEYED_FIELDS or (name in TRAINING_FIELDS and value != defaults[name])
    }
    text = json.dumps({"settings": training, "rows": dict(fingerprint)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
