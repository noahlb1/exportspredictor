"""The selection measure, its diagnostics and the win rule (ADR 0008): how an arm is measured and when it wins."""

import statistics
from collections.abc import Mapping, Sequence

import polars as pl

from exportspredictor.evaluation import MIN_RANKED_ROWS, TWEEDIE_POWER, tweedie_deviance
from exportspredictor.experiments import spearman_within_product

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


def rank_average_measures(grid: pl.DataFrame) -> dict:
    """The rank-average of the model (`model_usd`) and the naive baseline within each Product: a reference only.

    It has no dollar value, so no Export Gap and no deviance; it is never selected.
    """
    ranked = grid.with_columns(
        rank_average=(pl.col("model_usd").rank().over("hs6") + pl.col("naive_usd").rank().over("hs6")) / 2
    )
    return measures(ranked, None, "rank_average")
