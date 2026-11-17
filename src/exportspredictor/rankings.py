"""Opportunity rankings: the Markets for one Product, or the Products for one Market.

Both ranking views (the API's `/opportunities/by-product` and `/opportunities/by-market`)
go through `rank_opportunities`, which filters, sorts and limits once. A new ranking order
is one entry in `RANK_BY`; a new row field is added in `rank_opportunities`.
"""

from dataclasses import dataclass

import polars as pl

from exportspredictor.artefacts import Artefacts
from exportspredictor.scenarios import DEFAULT_MIN_SHARE, DEFAULT_MIN_USD, is_blocked


@dataclass(frozen=True)
class RankOrder:
    """One way to order a ranking: the `rank_by` key, the USD column it sorts by, and its wording."""

    name: str
    column: str
    title: str
    description: str


RANK_BY = {
    order.name: order
    for order in [
        RankOrder("gap", "export_gap_usd", "Export Gap", "Export Potential minus Actual Exports, largest first."),
        RankOrder(
            "blocked",
            "blocked_effect_usd",
            "Blocked Opportunity value",
            "The Scenario Effect the model associates with adding a Trade Agreement where none exists "
            f"(0 where one does). Flagged rows clear {DEFAULT_MIN_SHARE:.0%} of Baseline Export Potential "
            f"and ${DEFAULT_MIN_USD / 1e6:g}M.",
        ),
    ]
}
DEFAULT_RANK_BY = "gap"


def rank_opportunities(
    artefacts: Artefacts,
    *,
    hs6: str | None = None,
    iso3: str | None = None,
    sector: str | None = None,
    region: str | None = None,
    min_gap_usd: float | None = None,
    hide_flagged: bool = False,
    hide_hubs: bool = False,
    rank_by: str = DEFAULT_RANK_BY,
    limit: int | None = None,
) -> pl.DataFrame:
    """The Baseline's UK Product–Market pairs for one Product or one Market, ordered by `rank_by`.

    Each row names its Product (hs6, description, sector) and Market (iso3, market, region),
    says whether a UK Trade Agreement with the Market is in force in the Baseline year
    (`trade_agreement`, the policy feature the model saw), gives Export Potential, Actual
    Exports and Export Gap as Gap Period means, the Baseline year's Export Potential and the
    Blocked Opportunity value in nominal USD, flags a Blocked Opportunity at the default
    thresholds against the Baseline year (`blocked_opportunity`), and carries the Baseline's
    `flagged_product` and `reexport_hub` caveat flags (ADR 0005). Filters keep rows of one
    Sector (HS2 chapter), one Market region, or an Export Gap of at least `min_gap_usd`;
    `hide_flagged` drops Flagged Products and `hide_hubs` drops Re-export Hubs (both off by
    default, as they are ranked as normal). Rows are sorted by the
    `RANK_BY[rank_by]` column, largest first (ties by hs6, iso3); `limit` then keeps the top rows.
    """
    order = RANK_BY[rank_by]
    rows = artefacts.baseline
    policy = artefacts.baseline_features.select("hs6", "iso3", pl.col("trade_agreement") == 1)
    if hs6 is not None:
        rows, policy = (t.filter(pl.col("hs6") == hs6) for t in (rows, policy))
    if iso3 is not None:
        rows, policy = (t.filter(pl.col("iso3") == iso3) for t in (rows, policy))
    rows = rows.join(policy, on=["hs6", "iso3"], how="left")
    if min_gap_usd is not None:
        rows = rows.filter(pl.col("export_gap_usd") >= min_gap_usd)
    if hide_flagged:
        rows = rows.filter(~pl.col("flagged_product"))
    if hide_hubs:
        rows = rows.filter(~pl.col("reexport_hub"))
    rows = rows.join(artefacts.products, on="hs6", how="left").join(
        artefacts.markets.rename({"name": "market"}), on="iso3", how="left"
    )
    if sector is not None:
        rows = rows.filter(pl.col("sector") == sector)
    if region is not None:
        rows = rows.filter(pl.col("region") == region)
    rows = (
        rows.with_columns(blocked_opportunity=is_blocked("blocked_effect_usd", "baseline_export_potential_usd"))
        .sort(order.column, "hs6", "iso3", descending=[True, False, False])
    )
    return rows if limit is None else rows.head(limit)
