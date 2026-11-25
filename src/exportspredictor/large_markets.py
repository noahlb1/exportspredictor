"""The Selection Measure on the largest importing Markets, reported beside the headline and never selected on.

The headline weighs each of the UK's Markets equally, and about 100 of them are microstates
whose zeros dominate narrow Products. Advisors mostly look at the large Markets, so the
evaluation also reports the measure restricted to the 30 and 60 largest importing Markets
(`LARGE_MARKET_COUNTS`): a Market's size is its total imports in the evaluated year, summed
over every Exporter's Flows of every Product. Only Markets in the measured rows compete,
so the UK, which is no Market for UK exports, never takes a place.

Nothing is chosen on these figures (ADR 0008 chooses on the Selection Measure over every
Market). The open review of 2022 (`docs/research/open-review-2022.md`) first measured them;
`scripts/research/open_review_2022/s22_large_markets.py` reproduces its figures.
"""

from collections.abc import Callable, Iterable, Sequence

import polars as pl

from exportspredictor.selection_measure import selection_measure
from exportspredictor.wording import number_text

LARGE_MARKET_COUNTS = (30, 60)

Measure = Callable[[pl.DataFrame, str], float | None]


def largest_importers(flows: pl.DataFrame, markets: Iterable[str], n: int) -> list[str]:
    """The `n` of `markets` with the largest total imports in `flows`, largest first (ties by ISO3).

    `flows` hold every Exporter's Flows of one year (`importer`, `value_usd`); a Market with
    no Flow imports 0.
    """
    candidates = pl.DataFrame({"importer": sorted(set(markets))}, schema={"importer": pl.String})
    totals = flows.group_by(pl.col("importer").cast(pl.String)).agg(imports=pl.col("value_usd").cast(pl.Float64).sum())
    ranked = (
        candidates.join(totals, on="importer", how="left")
        .with_columns(pl.col("imports").fill_null(0.0))
        .sort(["imports", "importer"], descending=[True, False])
    )
    return ranked["importer"].head(n).to_list()


def restricted_measures(
    rows: pl.DataFrame,
    flows: pl.DataFrame,
    prediction: str,
    *,
    counts: Sequence[int] = LARGE_MARKET_COUNTS,
    measure: Measure = selection_measure,
) -> dict[int, float | None]:
    """For each count N, `measure` of `prediction` on the `rows` of the N largest importers among their Markets.

    `rows` hold one row per Product and Market (`hs6`, `importer`, `value_usd`, `prediction`);
    `flows` every Exporter's Flows of the same year. `measure` is the tie-proof Selection
    Measure by default; `legacy_selection_measure` gives the figures measured before the measure was tie-proof.
    """
    markets = rows["importer"].cast(pl.String).unique().to_list()
    ranked = largest_importers(flows, markets, max(counts, default=0))
    return {
        n: measure(rows.filter(pl.col("importer").cast(pl.String).is_in(ranked[:n])), prediction) for n in counts
    }


def _headline_measure(rows: pl.DataFrame, prediction: str) -> float | None:
    # The headline hashes ties by Product, Exporter and Importer (`evaluation._ROW_MARKET`); so does this, so
    # with every Market among the largest it gives the headline's own figure.
    return selection_measure(rows, prediction, market=("exporter", "importer"))


def large_markets_report(
    scored: pl.DataFrame,
    flows: pl.DataFrame,
    test_year: int,
    *,
    predictors: dict[str, str],
    counts: Sequence[int] = LARGE_MARKET_COUNTS,
) -> dict:
    """The evaluation report's `large_markets` block: the headline's measure on the largest importing Markets.

    `scored` is the UK's scored grid of `test_year` without Excluded Products (the headline's
    rows), `flows` every Exporter's Flows of that year, and `predictors` maps a name ("model",
    "naive") to its column. Each row holds the count of Markets and each predictor's figure.
    """
    keyed = scored.with_columns(pl.col("hs6").cast(pl.String), pl.col("importer").cast(pl.String))
    measured = {
        name: restricted_measures(keyed, flows, column, counts=counts, measure=_headline_measure)
        for name, column in predictors.items()
    }
    rows = [{"markets": n, **{name: measured[name][n] for name in predictors}} for n in counts]
    return {"test_year": test_year, "rows": rows, "text": large_markets_text(rows, test_year)}


def large_markets_text(rows: Sequence[dict], test_year: int) -> str:
    """The `large_markets` sentence, worded from its rows (each with `markets`, `model` and `naive`)."""
    figures = "; ".join(
        f"on the {row['markets']} largest, {number_text(row['model'], '.2f')} "
        f"(Naive Baseline {number_text(row['naive'], '.2f')})"
        for row in rows
    )
    return (
        f"On the Markets that import the most (total imports in {test_year}, from every Exporter), the model's "
        f"ranking scores: {figures}. The headline weighs every Market equally, many of them small; these figures "
        "are shown beside it, and no choice is made on them."
    )
