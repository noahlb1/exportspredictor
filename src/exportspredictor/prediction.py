"""Export Potential from the model: the one place in the project that predicts from a Booster.

Out-of-fold scoring, Scenario application, the Blocked Opportunity values, the Potential
Range, the explanations, the evaluation, the backtests and the API all get Export Potential
(USD) from `export_potential`, and the log-scale SHAP factors from `log_contributions`. So
every number the app shows comes from the same model in the same way, and a change to the
form of the model (spec #65) is made here only.

The only model form today is the single XGBoost model with a Tweedie objective and a log
link: Export Potential is the exponential of its margin.

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
) -> pl.Series:
    """Export Potential (USD) for feature rows (holding `features`, transformed as the models saw them).

    `rounds` limits prediction to the first trees (a search trial); `device` is where the
    models predict ("cuda" only where they were trained there). Raises ValueError if a row's
    `fold` has no model, or a fold model held out another fold.
    """
    if isinstance(models, xgb.Booster):
        models, by = [models], "mean"
    if by == "mean":
        potential = np.mean([_predict(model, rows, features, rounds, device) for model in models], axis=0)
        return pl.Series("export_potential_usd", potential, dtype=pl.Float64)
    potential = np.zeros(rows.height)
    for k, mask in _fold_masks(rows, models):
        potential[mask] = _predict(models[k], rows.filter(pl.Series(mask)), features, rounds, device)
    return pl.Series("export_potential_usd", potential)


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
    model: xgb.Booster, rows: pl.DataFrame, features: Sequence[str], rounds: int | None, device: str
) -> np.ndarray:
    """One model's predictions (float32), `PREDICT_ROWS` rows at a time."""
    if rows.is_empty():
        return np.zeros(0, dtype=np.float32)
    extra = {} if rounds is None else {"iteration_range": (0, rounds)}
    parts = []
    for start in range(0, rows.height, PREDICT_ROWS):
        data = np.ascontiguousarray(rows.slice(start, PREDICT_ROWS).select(features).to_numpy(), dtype=np.float32)
        if device.startswith("cuda"):
            import cupy  # only where the models were trained on the GPU

            parts.append(cupy.asnumpy(model.inplace_predict(cupy.asarray(data), **extra)))
            cupy.get_default_memory_pool().free_all_blocks()
        else:
            parts.append(model.inplace_predict(data, **extra))
    return np.concatenate(parts)
