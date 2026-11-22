"""Offset Pairs (ADR 0005, spec #129): Opportunities whose Export Gap the rest of their Heading offsets in that Market.

A Heading is the first four digits of a Product's HS6 code. It groups Products for this rule only: it is not a
feature, and the model is neither trained nor measured at it.

For each Market and Heading, the Heading gap is the summed Export Gap of every scored Product of the Heading in that
Market. A pair is an Offset Pair when all three hold:

- its Export Gap is positive (it is an Opportunity);
- its Heading holds at least two scored Products in that Market;
- the Heading gap is at most (1 - `THRESHOLD`) times the pair's Export Gap.

Equivalently, the other Products of the Heading sell above their Export Potential, net of their own Export Gaps, by at
least `THRESHOLD` of the pair's Export Gap. That amount (the pair's Export Gap minus the Heading gap) is the offset its
caveat shows, with the other Product of the Heading that sells furthest above its Export Potential in that Market.

The rule reads the served Baseline's own figures (Gap Period means), after scoring, so the caveat agrees with the row
and no model or feature reads a pair's Actual Exports (ADR 0001). Excluded Products are not in the Baseline, so they
are not in the Heading sums. It changes no score, rank or metric.

The result (`OffsetPairs`) reaches the Offset Pair Caveat Tier through `caveats.CaveatContext.offset_pair_reasons`,
and the pipeline records its reach in the artefact metadata (`METADATA_KEY`, `OffsetPairs.metadata`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from exportspredictor.currency import format_usd

THRESHOLD = 0.5  # the share of a pair's Export Gap the rest of its Heading must offset
TOP_DEPTHS = (100, 1_000, 10_000)  # the Leaderboard depths the metadata counts Offset Pairs in
METADATA_KEY = "offset_pairs"


@dataclass(frozen=True)
class OffsetPairs:
    """The rule's result: its threshold and each Offset Pair's reason, keyed by (hs6, iso3)."""

    threshold: float
    reasons: Mapping[tuple[str, str], str]

    def metadata(self, baseline: pl.DataFrame) -> dict:
        """The rule's reach in `baseline` (hs6, iso3, export_gap_usd): how many Offset Pairs there are, how many are
        in the top `TOP_DEPTHS` Opportunities (largest Export Gap first, ties by hs6 and iso3), and the share of
        summed positive Export Gap they hold."""
        keys = pl.DataFrame(
            list(self.reasons), schema={"hs6": pl.String, "iso3": pl.String}, orient="row"
        ).with_columns(offset=pl.lit(True))
        ranked = (
            baseline.select(pl.col("hs6").cast(pl.String), pl.col("iso3").cast(pl.String), "export_gap_usd")
            .filter(pl.col("export_gap_usd") > 0)
            .sort(["export_gap_usd", "hs6", "iso3"], descending=[True, False, False])
            .join(keys, on=["hs6", "iso3"], how="left", maintain_order="left")
            .with_columns(pl.col("offset").fill_null(False))
        )
        total = float(ranked["export_gap_usd"].sum())
        offset = float(ranked.filter("offset")["export_gap_usd"].sum())
        return {
            "threshold": self.threshold,
            "offset_pairs": int(ranked["offset"].sum()),
            "opportunities": ranked.height,
            "in_top": {str(depth): int(ranked.head(depth)["offset"].sum()) for depth in TOP_DEPTHS},
            "offset_export_gap_usd": offset,
            "opportunity_export_gap_usd": total,
            "share_of_export_gap": offset / total if total else None,
        }


def heading(hs6: pl.Expr) -> pl.Expr:
    """The Heading of each Product: the first four digits of its HS6 code."""
    return hs6.cast(pl.String).str.slice(0, 4)


def offset_pairs(
    baseline: pl.DataFrame,
    *,
    products: pl.DataFrame,
    markets: pl.DataFrame,
    threshold: float = THRESHOLD,
) -> OffsetPairs:
    """The Offset Pairs of `baseline` (hs6, iso3, export_gap_usd: one row per scored UK pair), with each one's reason.

    `products` (hs6, description) and `markets` (iso3, name) name the Product with the largest surplus and the Market.
    """
    rows = (
        baseline.select(pl.col("hs6").cast(pl.String), pl.col("iso3").cast(pl.String), "export_gap_usd")
        .with_columns(heading=heading(pl.col("hs6")))
        .with_columns(
            heading_gap=pl.col("export_gap_usd").sum().over("iso3", "heading"),
            neighbours=pl.len().over("iso3", "heading") - 1,
            # The Product of the Heading furthest above its Export Potential in the Market (most negative Export Gap,
            # ties by hs6). For an Offset Pair it is another Product: its own Export Gap is positive and the rest of
            # the Heading nets below zero.
            largest_surplus=pl.col("hs6").sort_by("export_gap_usd", "hs6").first().over("iso3", "heading"),
        )
    )
    offset = rows.filter(
        (pl.col("export_gap_usd") > 0)
        & (pl.col("neighbours") >= 1)
        & (pl.col("heading_gap") <= (1 - threshold) * pl.col("export_gap_usd"))
    ).with_columns(offset_usd=pl.col("export_gap_usd") - pl.col("heading_gap"))
    names = dict(markets.select(pl.col("iso3").cast(pl.String), "name").iter_rows())
    descriptions = dict(products.select(pl.col("hs6").cast(pl.String), "description").iter_rows())
    reasons = {
        (hs6, iso3): (
            f"the UK already sells {names.get(iso3, iso3)} {format_usd(amount)} above Export Potential in the other "
            f"Products of Heading {code}, mostly {descriptions.get(surplus, surplus)}"
        )
        for hs6, iso3, code, amount, surplus in offset.select(
            "hs6", "iso3", "heading", "offset_usd", "largest_surplus"
        ).iter_rows()
    }
    return OffsetPairs(threshold=threshold, reasons=reasons)
