"""The training scope (ADR 0004): which Exporters' rows train the model.

The model trains on Flows and Zero Cells whose Exporter is one of the top 100 economies by
mean GDP over 2012–2023, with the UK always included. Every Importer is kept, because the
UK is only ever scored as an Exporter. The scope limits rows, not features: small
Exporters' Flows still feed every Importer-side and demand feature
(`features.training_rows` drops their rows, never their Flows).
"""

import polars as pl

TOP_EXPORTERS = 100
GDP_YEARS = (2012, 2023)  # inclusive: the years whose mean GDP ranks the Exporters


def exporter_scope(
    country_year: pl.DataFrame, *, uk: str, n: int = TOP_EXPORTERS, years: tuple[int, int] = GDP_YEARS
) -> list[str]:
    """The ISO3 codes of the `n` countries with the highest mean GDP over `years`, plus `uk`, sorted.

    The mean is over the years a country has a GDP; a country with none in `years` is never in scope.
    """
    first, last = years
    top = (
        country_year.filter(pl.col("year").is_between(first, last))
        .group_by("iso3")
        .agg(pl.col("gdp_usd").mean())
        .drop_nulls("gdp_usd")
        .sort(["gdp_usd", "iso3"], descending=[True, False])
        .head(n)["iso3"]
        .to_list()
    )
    return sorted({*top, uk})
