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

A two-part model (spec #65, experiment 4) is a binary model of whether a Flow exists and a
size model fitted on non-zero Flows: its Export Potential is the probability of a Flow times
the size (`two_part_potential`).

`by="fold"` scores each row with the fold model named by its `fold` column, the model that
held the row's Product–Market pair out (ADR 0001). `by="mean"` scores every row with the mean
of the models (rows no fold model held out, such as other Exporters'). A lone Booster scores
every row.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES

PREDICT_ROWS = 2_000_000  # rows predicted at a time, so a large table fits on the device

Models = xgb.Booster | Sequence[xgb.Booster]


def export_potential(
    rows: pl.DataFrame,
    models: Models,
    *,
    features: Sequence[str] = FEATURES,
    by: Literal["fold", "mean"] = "fold",
    rounds: int | None = None,
    device: str = "cpu",
    base_margin: np.ndarray | pl.Series | None = None,
) -> pl.Series:
    """Export Potential (USD) for feature rows (holding `features`, transformed as the models saw them).

    `rounds` limits prediction to the first trees (a search trial); `device` is where the
    models predict ("cuda" only where they were trained there); `base_margin` is each row's
    starting point on the log scale, for models trained from one. Raises ValueError if a
    row's `fold` has no model, or a fold model held out another fold.
    """
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
    rows: pl.DataFrame, models: Sequence[xgb.Booster], *, features: Sequence[str] = FEATURES
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values (TreeSHAP) on the log scale, each row from its own fold model.

    Returns (contributions, log_export_potential): contributions has one column per feature,
    in `features` order, then the bias (the model's average log Export Potential); each row's
    contributions sum to its log Export Potential, whose exponential is `export_potential`.
    """
    contributions = np.zeros((rows.height, len(features) + 1), dtype=np.float32)
    margin = np.zeros(rows.height, dtype=np.float32)
    for k, mask in _fold_masks(rows, models):
        values = rows.filter(pl.Series(mask)).select(pl.col(features).cast(pl.Float32)).to_numpy()
        matrix = xgb.DMatrix(values, feature_names=list(features), missing=np.nan)
        contributions[mask] = models[k].predict(matrix, pred_contribs=True)
        margin[mask] = models[k].predict(matrix, output_margin=True)
    return contributions, margin


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
