"""Export Potential from the model: the one place in the project that predicts from a Booster.

Out-of-fold scoring, Scenario application, the Blocked Opportunity values, the Potential
Range, the explanations, the evaluation, the backtests and the API all get Export Potential
(USD) from `export_potential`, and the log-scale SHAP factors from `log_contributions`. So
every number the app shows comes from the same model in the same way, and a change to the
form of the model (spec #65) is made here only.

The model form is a single XGBoost model with a Tweedie objective and a log link: Export
Potential is the exponential of its margin. A model trained from a starting point (spec
#65, experiment 2) takes that point per row as `base_margin` on the log scale, and its trees
add a correction to it.

The served form is a `ModelForm`: the model alone, or a **blend** of the model with the naive
baseline (spec #65, experiment 1, confirmed on five folds in #76): Export Potential =
exp((1 − w) · log model + w · log naive), each floored at $1, with w the naive weight. The
naive baseline (`evaluation.naive_potential`) is a column of the rows (`naive_usd`), not a
feature, so a Scenario, which changes only policy features, leaves it unchanged.

A two-part model (spec #65, experiment 4) is a binary model of whether a Flow exists and a
size model fitted on non-zero Flows: its Export Potential is the probability of a Flow times
the size (`two_part_potential`).

`by="fold"` scores each row with the fold model named by its `fold` column, the model that
held the row's Product–Market pair out (ADR 0001). `by="mean"` scores every row with the mean
of the models (rows no fold model held out, such as other Exporters'). A lone Booster scores
every row.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES

PREDICT_ROWS = 2_000_000  # rows predicted at a time, so a large table fits on the device

Models = xgb.Booster | Sequence[xgb.Booster]
NAIVE_COLUMN = "naive_usd"  # the naive baseline (USD) of each row, which a blend needs
# The floor of a prediction in dollars, on every log scale: each side of a blend (a Zero Cell's naive baseline
# is $0), a base margin, and the deviance (a zero naive prediction of a non-zero Flow is penalised heavily
# but finitely).
MIN_PREDICTION_USD = 1.0
SUPPLY, DEMAND = "log_supply_loo", "log_demand_loo"  # the features a blend's naive share is attributed to


@dataclass(frozen=True)
class ModelForm:
    """How Export Potential is made from the fold models: the model alone (w = 0) or a blend.

    `naive_weight` is w, the weight of the naive baseline on the log scale; the model's is 1 − w.
    """

    naive_weight: float = 0.0

    def __post_init__(self):
        if not 0 <= self.naive_weight < 1:
            raise ValueError(f"The naive weight must be in [0, 1), not {self.naive_weight}")

    @property
    def name(self) -> str:
        return "blend" if self.naive_weight else "model alone"

    @property
    def text(self) -> str:
        """How this form makes Export Potential, in one sentence: the one wording of the blend."""
        if not self.naive_weight:
            return "Export Potential comes from the model alone."
        return (
            f"Export Potential blends the model (weight {1 - self.naive_weight:g}) with the Naive Baseline "
            f"(weight {self.naive_weight:g}) on the log scale."
        )

    def served_effect(self, model_effect: float) -> float:
        """A Scenario Effect of the model (a relative change) as this form serves it: the Naive Baseline
        has no policy features, so only the model's share, 1 − w on the log scale, moves."""
        return (1 + model_effect) ** (1 - self.naive_weight) - 1

    def to_dict(self) -> dict:
        return {"form": self.name, "naive_weight": self.naive_weight}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ModelForm":
        if data.get("form") not in ("blend", "model alone"):
            raise ValueError(f"No served model form {data.get('form')!r}: only the model alone or a blend")
        return cls(naive_weight=float(data["naive_weight"]))


MODEL_ALONE = ModelForm()


def export_potential(
    rows: pl.DataFrame,
    models: Models,
    *,
    features: Sequence[str] = FEATURES,
    by: Literal["fold", "mean"] = "fold",
    rounds: int | None = None,
    device: str = "cpu",
    base_margin: np.ndarray | pl.Series | None = None,
    form: ModelForm = MODEL_ALONE,
) -> pl.Series:
    """Export Potential (USD) for feature rows (holding `features`, transformed as the models saw them).

    `rounds` limits prediction to the first trees (a search trial); `device` is where the
    models predict ("cuda" only where they were trained there); `base_margin` is each row's
    starting point on the log scale, for models trained from one. A blend `form` blends the
    models' Export Potential with the rows' `naive_usd`. Raises ValueError if a row's `fold`
    has no model, a fold model held out another fold, or a blend's rows have no `naive_usd`.
    """
    naive = _naive(rows, form)
    potential = _model_potential(rows, models, features, by, rounds, device, base_margin)
    if naive is None:
        return potential
    return pl.Series("export_potential_usd", blend(potential.to_numpy(), naive, form.naive_weight))


def blend(model_usd: np.ndarray, naive_usd: np.ndarray, naive_weight: float) -> np.ndarray:
    """exp((1 − w) · log model + w · log naive), each floored at MIN_PREDICTION_USD; the model unchanged when w = 0."""
    model_usd = np.asarray(model_usd, dtype=np.float64)
    if naive_weight == 0:
        return model_usd
    return np.exp((1 - naive_weight) * _floored_log(model_usd) + naive_weight * _floored_log(naive_usd))


def _floored_log(usd) -> np.ndarray:
    return np.log(np.maximum(np.asarray(usd, dtype=np.float64), MIN_PREDICTION_USD))


def _naive(rows: pl.DataFrame, form: ModelForm) -> np.ndarray | None:
    """The rows' naive baseline when `form` is a blend (None for the model alone)."""
    if not form.naive_weight:
        return None
    if NAIVE_COLUMN not in rows.columns or rows[NAIVE_COLUMN].null_count():
        raise ValueError(f"A blend needs the naive baseline ({NAIVE_COLUMN}) on every row")
    return rows[NAIVE_COLUMN].cast(pl.Float64).to_numpy()


def _model_potential(rows, models, features, by, rounds, device, base_margin) -> pl.Series:
    """The models' own Export Potential (the model alone)."""
    margin = None if base_margin is None else np.asarray(base_margin, dtype=np.float32)
    if isinstance(models, xgb.Booster):
        models, by = [models], "mean"
    if by == "mean":
        potential = np.mean([_predict(model, rows, features, rounds, device, margin) for model in models], axis=0)
        return pl.Series("export_potential_usd", potential, dtype=pl.Float64)
    potential = np.zeros(rows.height)
    for k, mask in _fold_masks(rows, models):
        part = None if margin is None else margin[mask]
        potential[mask] = _predict(models[k], rows.filter(pl.Series(mask)), features, rounds, device, part)
    return pl.Series("export_potential_usd", potential)


def flow_probability(
    rows: pl.DataFrame,
    model: xgb.Booster,
    *,
    features: Sequence[str] = FEATURES,
    rounds: int | None = None,
    device: str = "cpu",
) -> pl.Series:
    """The probability of a Flow, from the binary part of a two-part model."""
    return pl.Series("flow_probability", _predict(model, rows, features, rounds, device), dtype=pl.Float64)


def two_part_potential(
    rows: pl.DataFrame,
    any_flow: xgb.Booster,
    size: xgb.Booster,
    *,
    features: Sequence[str] = FEATURES,
    rounds: tuple[int | None, int | None] = (None, None),
    device: str = "cpu",
) -> pl.Series:
    """Export Potential (USD) of a two-part model: the probability of a Flow times its size if there is one.

    `rounds` limits each part (any_flow, size) to its first trees. Both parts are
    monotone-constrained alike, so their product moves with the policy features as each does.
    """
    probability = flow_probability(rows, any_flow, features=features, rounds=rounds[0], device=device).to_numpy()
    size_usd = _predict(size, rows, features, rounds[1], device)
    return pl.Series("export_potential_usd", probability * size_usd, dtype=pl.Float64)


def log_contributions(
    rows: pl.DataFrame,
    models: Sequence[xgb.Booster],
    *,
    features: Sequence[str] = FEATURES,
    form: ModelForm = MODEL_ALONE,
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values (TreeSHAP) on the log scale, each row from its own fold model.

    Returns (contributions, log_export_potential): contributions has one column per feature,
    in `features` order, then the bias; each row's contributions sum to its log Export
    Potential, whose exponential is `export_potential` with the same `form`.

    For the model alone the bias is the model's average log Export Potential. For a blend,
    each SHAP value and the bias are scaled by the model's weight 1 − w; the naive baseline's
    share, w · log naive, is split equally between UK supply (`log_supply_loo`) and Market
    demand (`log_demand_loo`), the two sides of the market-share product it is made of; and
    where the model's own Export Potential is below the $1 floor, the bias takes up the lift
    to $1, so the contributions still add up.
    """
    naive = _naive(rows, form)
    contributions = np.zeros((rows.height, len(features) + 1), dtype=np.float32)
    margin = np.zeros(rows.height, dtype=np.float32)
    for k, mask in _fold_masks(rows, models):
        values = rows.filter(pl.Series(mask)).select(pl.col(features).cast(pl.Float32)).to_numpy()
        matrix = xgb.DMatrix(values, feature_names=list(features), missing=np.nan)
        contributions[mask] = models[k].predict(matrix, pred_contribs=True)
        margin[mask] = models[k].predict(matrix, output_margin=True)
    if naive is None:
        return contributions, margin
    if SUPPLY not in features or DEMAND not in features:
        raise ValueError(f"A blend's naive share is attributed to {SUPPLY} and {DEMAND}, which the models lack")
    w = form.naive_weight
    floored = np.maximum(margin.astype(np.float64), np.log(MIN_PREDICTION_USD))
    naive_share = w * _floored_log(naive)
    blended = (1 - w) * contributions.astype(np.float64)
    blended[:, -1] += (1 - w) * (floored - margin)
    blended[:, list(features).index(SUPPLY)] += naive_share / 2
    blended[:, list(features).index(DEMAND)] += naive_share / 2
    return blended, (1 - w) * floored + naive_share


def _fold_masks(rows: pl.DataFrame, models: Sequence[xgb.Booster]):
    """(k, mask) for each fold model k that holds rows, after checking every row has its fold model."""
    fold = rows["fold"].to_numpy()
    if rows.height and (fold.min() < 0 or fold.max() >= len(models)):
        raise ValueError(f"A row's fold has no fold model: folds run from 0 to {len(models) - 1}")
    for k, model in enumerate(models):
        if model.attr("held_out_fold") not in (None, str(k)):
            raise ValueError(f"Model {k} held out fold {model.attr('held_out_fold')}, not {k}")
        mask = fold == k
        if mask.any():
            yield k, mask


def _predict(
    model: xgb.Booster,
    rows: pl.DataFrame,
    features: Sequence[str],
    rounds: int | None,
    device: str,
    margin: np.ndarray | None = None,
) -> np.ndarray:
    """One model's predictions (float32), `PREDICT_ROWS` rows at a time."""
    if rows.is_empty():
        return np.zeros(0, dtype=np.float32)
    extra = {} if rounds is None else {"iteration_range": (0, rounds)}
    parts = []
    for start in range(0, rows.height, PREDICT_ROWS):
        data = np.ascontiguousarray(rows.slice(start, PREDICT_ROWS).select(features).to_numpy(), dtype=np.float32)
        offset = {} if margin is None else {"base_margin": margin[start : start + PREDICT_ROWS]}
        if device.startswith("cuda"):
            import cupy  # only where the models were trained on the GPU

            offset = {k: cupy.asarray(v) for k, v in offset.items()}
            parts.append(cupy.asnumpy(model.inplace_predict(cupy.asarray(data), **extra, **offset)))
            cupy.get_default_memory_pool().free_all_blocks()
        else:
            parts.append(model.inplace_predict(data, **extra, **offset))
    return np.concatenate(parts)
