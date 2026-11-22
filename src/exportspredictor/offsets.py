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

The result (`offset_pairs`, each Offset Pair's reason) reaches the Offset Pair Caveat Tier through
`caveats.CaveatContext.offset_pair_reasons`, and the pipeline records the tier's reach in the artefact metadata
(`METADATA_KEY`, `reach`, `OffsetPairReach`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from exportspredictor.currency import format_usd
from exportspredictor.rankings import in_top, ranked_opportunities

THRESHOLD = 0.5  # the share of a pair's Export Gap the rest of its Heading must offset
METADATA_KEY = "offset_pairs"


@dataclass(frozen=True)
class OffsetPairReach:
    """The rule's reach in a marked Baseline, as the artefact metadata records it (`METADATA_KEY`)."""

    threshold: float
    offset_pair_count: int
    opportunities: int
    in_top: Mapping[int, int]  # depth of the Leaderboard's default ranking -> Offset Pairs among its top rows
    offset_export_gap_usd: float
    opportunity_export_gap_usd: float
    share_of_export_gap: float | None  # None when no Opportunity has a positive Export Gap

    @property
    def share_of_opportunities(self) -> float:
        return self.offset_pair_count / self.opportunities if self.opportunities else 0.0

    def in_top_text(self) -> str:
        """The counts by depth in words, e.g. "21 of the top 100, 159 of the top 1,000 and 1,709 of the top 10,000"."""
        counts = [f"{count:,} of the top {depth:,}" for depth, count in sorted(self.in_top.items())]
        return ", ".join(counts[:-1]) + f" and {counts[-1]}" if len(counts) > 1 else "".join(counts)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "offset_pair_count": self.offset_pair_count,
            "opportunities": self.opportunities,
            "in_top": {str(depth): count for depth, count in sorted(self.in_top.items())},  # JSON keys are strings
            "offset_export_gap_usd": self.offset_export_gap_usd,
            "opportunity_export_gap_usd": self.opportunity_export_gap_usd,
            "share_of_export_gap": self.share_of_export_gap,
        }

    @classmethod
    def from_dict(cls, saved: Mapping) -> "OffsetPairReach":
        return cls(
            threshold=saved["threshold"], offset_pair_count=saved["offset_pair_count"],
            opportunities=saved["opportunities"],
            in_top={int(depth): count for depth, count in saved["in_top"].items()},
            offset_export_gap_usd=saved["offset_export_gap_usd"],
            opportunity_export_gap_usd=saved["opportunity_export_gap_usd"],
            share_of_export_gap=saved["share_of_export_gap"],
        )


def reach(marked: pl.DataFrame, column: str) -> OffsetPairReach:
    """The rule's reach in a Baseline `marked` with the Offset Pair tier's boolean `column`: the threshold, how many
    Offset Pairs there are, how many are in the top `rankings.REACH_DEPTHS` Opportunities of the Leaderboard's default
    ranking (`rankings.ranked_opportunities`), and the share of summed positive Export Gap they hold."""
    ranked = ranked_opportunities(marked)
    total = float(ranked["export_gap_usd"].sum())
    offset = float(ranked.filter(column)["export_gap_usd"].sum())
    return OffsetPairReach(
        threshold=THRESHOLD,
        offset_pair_count=int(ranked[column].sum()),
        opportunities=ranked.height,
        in_top=in_top(ranked, column),
        offset_export_gap_usd=offset,
        opportunity_export_gap_usd=total,
        share_of_export_gap=offset / total if total else None,
    )


def heading(hs6: pl.Expr) -> pl.Expr:
    """The Heading of each Product: the first four digits of its HS6 code."""
    return hs6.cast(pl.String).str.slice(0, 4)


def offset_pairs(
    baseline: pl.DataFrame, *, products: pl.DataFrame, markets: pl.DataFrame
) -> dict[tuple[str, str], str]:
    """The Offset Pairs of `baseline` (hs6, iso3, export_gap_usd: one row per scored UK pair), each one's reason keyed
    by (hs6, iso3).

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
        & (pl.col("heading_gap") <= (1 - THRESHOLD) * pl.col("export_gap_usd"))
    ).with_columns(offset_usd=pl.col("export_gap_usd") - pl.col("heading_gap"))
    names = dict(markets.select(pl.col("iso3").cast(pl.String), "name").iter_rows())
    descriptions = dict(products.select(pl.col("hs6").cast(pl.String), "description").iter_rows())
    return {
        (hs6, iso3): (
            f"the UK already sells {names.get(iso3, iso3)} {format_usd(amount)} above Export Potential in the other "
            f"Products of Heading {code}, mostly {descriptions.get(surplus, surplus)}"
        )
        for hs6, iso3, code, amount, surplus in offset.select(
            "hs6", "iso3", "heading", "offset_usd", "largest_surplus"
        ).iter_rows()
    }
