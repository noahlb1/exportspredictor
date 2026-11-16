"""Gravity features: time-invariant country-pair ties and country-year size.

Two derived tables feed the feature table:

- **Pair features** from CEPII Gravity (V202211): distance, contiguity, common official
  language and colonial ties. They are treated as time-invariant, taking each pair's
  latest non-null value.
- **Country-year features**: GDP (current USD) and population for every year 2012–2024.
  World Bank WDI is the main source. Missing values follow one documented rule
  (`fill_country_year`), and every value records the source it came from.

BACI identifies countries by numeric code; `map_baci_countries` maps them to the ISO3
codes Gravity and WDI use. `build` is the single entry point the data rebuild script
calls: it downloads what is missing and writes the derived Parquet files.
"""

import polars as pl

# BACI codes whose ISO3 field is not a real ISO3 code, mapped by hand.
BACI_ISO3_OVERRIDES = {
    "S19": "TWN",  # BACI code 490 "Other Asia, nes" is Taiwan (BACI FAQ)
}


def map_baci_countries(
    baci_countries: pl.DataFrame, *, gravity_iso3: set[str], wdi_iso3: set[str]
) -> pl.DataFrame:
    """Map BACI country codes to ISO3 and flag whether Gravity and WDI know each one.

    `baci_countries` has BACI's `country_code`, `country_name` and `country_iso3`.
    """
    return baci_countries.select(
        baci_code=pl.col("country_code"),
        name=pl.col("country_name"),
        baci_iso3=pl.col("country_iso3"),
        iso3=pl.col("country_iso3").replace(BACI_ISO3_OVERRIDES),
    ).with_columns(
        in_gravity=pl.col("iso3").is_in(sorted(gravity_iso3)),
        in_wdi=pl.col("iso3").is_in(sorted(wdi_iso3)),
    )


# Each variable and the column recording where its value came from.
COUNTRY_YEAR_VARIABLES = {"gdp_usd": "gdp_source", "population": "population_source"}


def fill_country_year(
    sources: dict[str, pl.DataFrame], *, iso3: list[str], years: list[int]
) -> pl.DataFrame:
    """GDP and population for every country and year, following the missing-value rule.

    Each variable is filled on its own, for each country:

    1. the first source in `sources` (in order) that has a value for that year;
    2. otherwise linear interpolation between the nearest observed years before and after;
    3. otherwise the nearest observed year's value, carried forward or back;
    4. otherwise missing (a country with no data in any source), left to the model's
       native missing-value handling.

    Each source is a frame of `iso3`, `year`, `gdp_usd` and `population`. The output adds
    `gdp_source` and `population_source`: a source name, "interpolated" or "carried".
    """
    grid = pl.DataFrame({"iso3": iso3}).join(pl.DataFrame({"year": years}), how="cross")
    grid = grid.with_columns(pl.col("year").cast(pl.Int32))
    for variable, source_column in COUNTRY_YEAR_VARIABLES.items():
        value, source = pl.lit(None, dtype=pl.Float64), pl.lit(None, dtype=pl.String)
        for name, frame in sources.items():
            column = f"{variable}__{name}"
            grid = grid.join(
                frame.select(
                    "iso3", pl.col("year").cast(pl.Int32), pl.col(variable).cast(pl.Float64).alias(column)
                ).filter(pl.col(column).is_not_null()),
                on=["iso3", "year"],
                how="left",
            )
            source = pl.when(value.is_null() & pl.col(column).is_not_null()).then(pl.lit(name)).otherwise(source)
            value = pl.coalesce(value, pl.col(column))
        observed = pl.col(variable).is_not_null()
        interpolated = pl.col(variable).interpolate().over("iso3", order_by="year")
        carried = pl.col(variable).forward_fill().backward_fill().over("iso3", order_by="year")
        grid = (
            grid.with_columns(value.alias(variable), source.alias(source_column))
            .with_columns(
                pl.when(observed)
                .then(pl.col(source_column))
                .when(interpolated.is_not_null())
                .then(pl.lit("interpolated"))
                .when(carried.is_not_null())
                .then(pl.lit("carried"))
                .alias(source_column),
                pl.coalesce(pl.col(variable), interpolated, carried).alias(variable),
            )
            .drop(pl.selectors.contains("__"))
        )
    variable_columns = [c for pair in COUNTRY_YEAR_VARIABLES.items() for c in pair]
    return grid.select("iso3", "year", *variable_columns).sort("iso3", "year")


def unresolved(mapping: pl.DataFrame) -> pl.DataFrame:
    """BACI countries with no Gravity pair data, so pair features would be missing."""
    return mapping.filter(~pl.col("in_gravity"))
