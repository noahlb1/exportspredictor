"""Opportunity rankings: the Markets for one Product, or the Products for one Market.

Both ranking views (the API's `/opportunities/by-product` and `/opportunities/by-market`)
go through `rank_opportunities`, so a new ranking option or row field is added here once.
"""

import polars as pl

from exportspredictor.artefacts import Artefacts

RANKED_BY = "export_gap_usd"
# Each in _usd and _gbp. blocked_effect: the Scenario Effect of adding a Trade Agreement where none exists.
VALUES = ("export_potential", "actual_exports", "export_gap", "blocked_effect")


def rank_opportunities(
    artefacts: Artefacts,
    *,
    hs6: str | None = None,
    iso3: str | None = None,
    sector: str | None = None,
    region: str | None = None,
    min_gap_usd: float | None = None,
    min_gap_gbp: float | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """The Baseline's UK Product–Market pairs for one Product or one Market, largest Export Gap first.

    Each row names its Product (hs6, description, sector) and Market (iso3, market, region),
    says whether a UK Trade Agreement with the Market is in force in the Baseline year
    (`trade_agreement`, the policy feature the model saw), and gives Export Potential, Actual Exports and Export Gap in USD and in GBP (at
    `artefacts.gbp_rate`). Filters keep rows of one Sector (HS2 chapter), one Market region,
    or an Export Gap of at least `min_gap_usd` / `min_gap_gbp`; `limit` keeps the top rows.
    """
    usd_per_gbp = artefacts.gbp_rate["usd_per_gbp"]
    rows = artefacts.baseline
    policy = artefacts.baseline_features.select("hs6", "iso3", pl.col("trade_agreement") == 1)
    if hs6 is not None:
        rows, policy = (t.filter(pl.col("hs6") == hs6) for t in (rows, policy))
    if iso3 is not None:
        rows, policy = (t.filter(pl.col("iso3") == iso3) for t in (rows, policy))
    rows = rows.join(policy, on=["hs6", "iso3"], how="left")
    if min_gap_usd is not None:
        rows = rows.filter(pl.col("export_gap_usd") >= min_gap_usd)
    if min_gap_gbp is not None:
        rows = rows.filter(pl.col("export_gap_usd") >= min_gap_gbp * usd_per_gbp)
    rows = rows.join(artefacts.products, on="hs6", how="left").join(
        artefacts.markets.rename({"name": "market"}), on="iso3", how="left"
    )
    if sector is not None:
        rows = rows.filter(pl.col("sector") == sector)
    if region is not None:
        rows = rows.filter(pl.col("region") == region)
    rows = rows.with_columns(
        (pl.col(f"{value}_usd") / usd_per_gbp).alias(f"{value}_gbp") for value in VALUES
    ).sort(RANKED_BY, "hs6", "iso3", descending=[True, False, False])
    return rows if limit is None else rows.head(limit)
