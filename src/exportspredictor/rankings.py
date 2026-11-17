"""Opportunity rankings: the Leaderboard of every UK pair, the Markets for one Product, or the Products for one Market.

Every ranking view (the API's `/opportunities/top`, `/opportunities/by-product` and
`/opportunities/by-market`) goes through `rank_opportunities`, which filters, sorts and limits
once; `summarise` sizes the filtered rows for the Leaderboard's summary row, and
`summarise_blocked` for the Blocked Opportunities page's (`/opportunities/blocked`, ordered by
`BLOCKED`). A new ranking order is one entry in `RANK_BY`; a new row field is added in
`rank_opportunities`.
"""

from dataclasses import dataclass

import polars as pl

from exportspredictor.artefacts import Artefacts
from exportspredictor.gap_period import OPPORTUNITY_TYPES
from exportspredictor.scenarios import DEFAULT_MIN_SHARE, DEFAULT_MIN_USD, is_blocked


@dataclass(frozen=True)
class RankOrder:
    """One way to order a ranking: the `rank_by` key, the column it sorts by, and its wording.

    `ties` are the columns that break ties, in order (all largest first). An order with
    `min_potential` ranks only pairs whose Export Potential is at least `min_potential_usd`.
    """

    name: str
    column: str
    title: str
    description: str
    ties: tuple[str, ...] = ()
    min_potential: bool = False


# Relative Export Gap ranks only pairs above a minimum Export Potential, which the Advisor picks.
MIN_POTENTIAL_USD_OPTIONS = (0.0, 1e5, 1e6, 1e7, 1e8)
DEFAULT_MIN_POTENTIAL_USD = 1e6

RANK_BY = {
    order.name: order
    for order in [
        RankOrder("gap", "export_gap_usd", "Export Gap", "Export Potential minus Actual Exports, largest first."),
        RankOrder(
            "relative",
            "relative_export_gap",
            "Relative Export Gap",
            "Export Gap divided by Export Potential: the share of its Export Potential the UK does not yet "
            "realise, largest first. Every pair with no Actual Exports scores 100%, so only pairs above a "
            "minimum Export Potential are ranked, and ties are broken by Export Gap.",
            ties=("export_gap_usd",),
            min_potential=True,
        ),
    ]
}
DEFAULT_RANK_BY = "gap"
# The Blocked Opportunities page's order. It is not in `RANK_BY`, so the Leaderboard and ranking
# pages do not offer it: Blocked Opportunities show no Export Gap and have their own page.
BLOCKED = RankOrder(
    "blocked",
    "blocked_effect_usd",
    "Blocked Opportunity value",
    "The Scenario Effect the model associates with adding a Trade Agreement where none exists "
    f"(0 where one does). Flagged rows clear {DEFAULT_MIN_SHARE:.0%} of Baseline Export Potential "
    f"and ${DEFAULT_MIN_USD / 1e6:g}M.",
)
_ORDERS = {**RANK_BY, BLOCKED.name: BLOCKED}  # every order `rank_opportunities` accepts


def rank_opportunities(
    artefacts: Artefacts,
    *,
    hs6: str | None = None,
    iso3: str | None = None,
    sector: str | None = None,
    region: str | None = None,
    opportunity_type: str | None = None,
    min_gap_usd: float | None = None,
    trade_agreement: bool | None = None,
    hide_flagged: bool = False,
    hide_hubs: bool = False,
    rank_by: str = DEFAULT_RANK_BY,
    min_potential_usd: float | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """The Baseline's UK Product–Market pairs (all, or one Product's or one Market's), ordered by `rank_by`.

    Each row names its Product (hs6, description, sector) and Market (iso3, market, region),
    says whether a UK Trade Agreement with the Market is in force in the Baseline year
    (`trade_agreement`, the policy feature the model saw), gives Export Potential, Actual
    Exports and Export Gap as Gap Period means with the Relative Export Gap and the
    Opportunity Type, the Baseline year's Export Potential and the Blocked Opportunity value
    in nominal USD, flags a Blocked Opportunity at the default thresholds against the Baseline
    year (`blocked_opportunity`), and carries the Baseline's `flagged_product` and
    `reexport_hub` caveat flags (ADR 0005).

    Filters keep the rows of one Product (`hs6`), one Market (`iso3`), one Sector (HS2
    chapter), one Market region, one Opportunity Type, an Export Gap of at least
    `min_gap_usd`, or pairs with (`trade_agreement=True`) or without (`False`) a UK Trade
    Agreement in force; `hide_flagged` drops Flagged Products and `hide_hubs` drops Re-export
    Hubs (both off by default, as they are ranked as normal). An order with a minimum Export
    Potential (Relative Export Gap) keeps only pairs whose Export Potential is at least
    `min_potential_usd` (default `DEFAULT_MIN_POTENTIAL_USD`); other orders ignore it. Rows
    are sorted by the `rank_by` order's column (`RANK_BY`, or `BLOCKED`) and its tie-breaks, largest first (then hs6,
    iso3); `limit` then keeps the top rows.
    """
    order = _ORDERS[rank_by]
    if opportunity_type is not None and opportunity_type not in OPPORTUNITY_TYPES:
        raise ValueError(f"Unknown Opportunity Type: {opportunity_type}")
    rows = artefacts.baseline
    policy = artefacts.baseline_features.select("hs6", "iso3", pl.col("trade_agreement") == 1)
    if hs6 is not None:
        rows, policy = (t.filter(pl.col("hs6") == hs6) for t in (rows, policy))
    if iso3 is not None:
        rows, policy = (t.filter(pl.col("iso3") == iso3) for t in (rows, policy))
    rows = rows.join(policy, on=["hs6", "iso3"], how="left")
    if min_gap_usd is not None:
        rows = rows.filter(pl.col("export_gap_usd") >= min_gap_usd)
    if opportunity_type is not None:
        rows = rows.filter(pl.col("opportunity_type") == opportunity_type)
    if trade_agreement is not None:
        rows = rows.filter(pl.col("trade_agreement") == trade_agreement)
    if hide_flagged:
        rows = rows.filter(~pl.col("flagged_product"))
    if hide_hubs:
        rows = rows.filter(~pl.col("reexport_hub"))
    if order.min_potential:
        floor = DEFAULT_MIN_POTENTIAL_USD if min_potential_usd is None else min_potential_usd
        rows = rows.filter((pl.col("export_potential_usd") >= floor) & pl.col(order.column).is_not_null())
    if sector is not None:
        rows = rows.join(artefacts.products.filter(pl.col("sector") == sector).select("hs6"), on="hs6", how="semi")
    if region is not None:
        rows = rows.join(artefacts.markets.filter(pl.col("region") == region).select("iso3"), on="iso3", how="semi")
    keys = [order.column, *order.ties]
    rows = rows.sort([*keys, "hs6", "iso3"], descending=[True] * len(keys) + [False, False], nulls_last=True)
    if limit is not None:
        rows = rows.head(limit)
    return (
        rows.join(artefacts.products, on="hs6", how="left", maintain_order="left")
        .join(artefacts.markets.rename({"name": "market"}), on="iso3", how="left", maintain_order="left")
        .with_columns(blocked_opportunity=is_blocked("blocked_effect_usd", "baseline_export_potential_usd"))
    )


def summarise(rows: pl.DataFrame) -> dict:
    """The Leaderboard's summary of ranked rows: the pair count and the total positive Export Gap (USD)."""
    return {
        "pairs": rows.height,
        "positive_export_gap_usd": float(rows["export_gap_usd"].clip(lower_bound=0).sum()),
    }


def summarise_blocked(rows: pl.DataFrame) -> dict:
    """The Blocked Opportunities page's summary: `summarise`, plus the Blocked Opportunities and total value (USD)."""
    return {
        **summarise(rows),
        "blocked_opportunities": int(rows["blocked_opportunity"].sum()),
        "blocked_effect_usd": float(rows["blocked_effect_usd"].sum()),
    }
