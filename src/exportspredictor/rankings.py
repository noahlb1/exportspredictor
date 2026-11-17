"""Opportunity rankings: the Markets for one Product, or the Products for one Market.

Both ranking views (the API's `/opportunities/by-product` and `/opportunities/by-market`)
go through `rank_opportunities`, so a new ranking option or row field is added here once.
"""

import polars as pl

from exportspredictor.artefacts import Artefacts

RANKED_BY = "export_gap_usd"


def rank_opportunities(
    artefacts: Artefacts,
    *,
    hs6: str | None = None,
    iso3: str | None = None,
    sector: str | None = None,
    region: str | None = None,
    min_gap_usd: float | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """The Baseline's UK Product–Market pairs for one Product or one Market, largest Export Gap first.

    Each row names its Product (hs6, description, sector) and Market (iso3, market, region).
    Filters keep rows of one Sector (HS2 chapter), one Market region, or an Export Gap of at
    least `min_gap_usd`; `limit` keeps the top rows.
    """
    rows = artefacts.baseline
    if hs6 is not None:
        rows = rows.filter(pl.col("hs6") == hs6)
    if iso3 is not None:
        rows = rows.filter(pl.col("iso3") == iso3)
    if min_gap_usd is not None:
        rows = rows.filter(pl.col("export_gap_usd") >= min_gap_usd)
    rows = rows.join(artefacts.products, on="hs6", how="left").join(
        artefacts.markets.rename({"name": "market"}), on="iso3", how="left"
    )
    if sector is not None:
        rows = rows.filter(pl.col("sector") == sector)
    if region is not None:
        rows = rows.filter(pl.col("region") == region)
    rows = rows.sort(RANKED_BY, "hs6", "iso3", descending=[True, False, False])
    return rows if limit is None else rows.head(limit)
