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

import json
import shutil
import zipfile
from pathlib import Path

import polars as pl

from exportspredictor import paths
from exportspredictor.downloads import download as download_file
from exportspredictor.downloads import write_parquet, writing

# BACI codes whose ISO3 field is not a real ISO3 code, mapped by hand.
BACI_ISO3_OVERRIDES = {
    "S19": "TWN",  # BACI code 490 "Other Asia, nes" is Taiwan (BACI FAQ)
}
# BACI countries missing from CEPII Gravity -> the nearby territory whose pair features
# stand in for theirs. Both are small French overseas territories with real trade.
GRAVITY_PROXIES = {
    "BLM": "GLP",  # Saint Barthelemy: Guadeloupe, its administrative parent until 2007
    "ATF": "REU",  # French Southern and Antarctic Lands: Reunion, which administers them
}


def map_baci_countries(
    baci_countries: pl.DataFrame, *, gravity_iso3: set[str], wdi_iso3: set[str]
) -> pl.DataFrame:
    """Map BACI country codes to ISO3 and flag whether Gravity and WDI know each one.

    `baci_countries` has BACI's `country_code`, `country_name` and `country_iso3`.
    `gravity_proxy` names the country whose Gravity pair features stand in, if any;
    `in_gravity` is true when the country or its proxy is in Gravity.
    """
    return baci_countries.select(
        baci_code=pl.col("country_code"),
        name=pl.col("country_name"),
        baci_iso3=pl.col("country_iso3"),
        iso3=pl.col("country_iso3").replace(BACI_ISO3_OVERRIDES),
    ).with_columns(
        gravity_proxy=pl.col("iso3").replace_strict(GRAVITY_PROXIES, default=None),
    ).with_columns(
        in_gravity=pl.coalesce("gravity_proxy", "iso3").is_in(sorted(gravity_iso3)),
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


# Our pair feature name -> CEPII Gravity column.
PAIR_FEATURES = {
    "distance_km": "dist",  # between the most populated cities
    "contiguous": "contig",
    "common_language": "comlang_off",  # a common official or primary language
    "colonial_tie": "col_dep_ever",  # ever in a colonial or dependency relationship
    "common_colonizer": "comcol",  # shared a colonizer after 1945
}


def gravity_pairs(gravity: pl.LazyFrame, *, proxies: dict[str, str] = GRAVITY_PROXIES) -> pl.DataFrame:
    """Time-invariant features for every ordered country pair in CEPII Gravity.

    Only rows where both countries exist that year count, so historical entities sharing
    an ISO3 code (such as Sudan before 2011) do not leak in. Each feature takes the
    pair's latest non-null value. Each country in `proxies` gets copies of its proxy's
    pairs, in both directions.
    """
    pairs = (
        gravity.filter((pl.col("country_exists_o") == 1) & (pl.col("country_exists_d") == 1))
        .sort("year")
        .group_by("iso3_o", "iso3_d")
        .agg(pl.col(source).drop_nulls().last().alias(name) for name, source in PAIR_FEATURES.items())
        .collect()
    )
    # Each country's name in the output -> the Gravity country whose pairs it takes.
    known = pl.concat([pairs["iso3_o"], pairs["iso3_d"]]).unique()
    stand_in = pl.concat(
        [
            pl.DataFrame({"iso3": known, "source": known}),
            pl.DataFrame({"iso3": list(proxies), "source": list(proxies.values())}),
        ]
    )
    return (
        pairs.join(stand_in.rename({"iso3": "out_o", "source": "iso3_o"}), on="iso3_o")
        .join(stand_in.rename({"iso3": "out_d", "source": "iso3_d"}), on="iso3_d")
        .filter(pl.col("out_o") != pl.col("out_d"))
        .drop("iso3_o", "iso3_d")
        .rename({"out_o": "iso3_o", "out_d": "iso3_d"})
        .select("iso3_o", "iso3_d", *PAIR_FEATURES)
        .sort("iso3_o", "iso3_d")
    )


# --- Sources: download and read -------------------------------------------------------

YEARS = list(range(2012, 2025))
GRAVITY_ZIP = "Gravity_csv_V202211.zip"
GRAVITY_CSV = "Gravity_V202211.csv"
GRAVITY_URL = f"https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/{GRAVITY_ZIP}"
# World Bank WDI indicator -> our variable: GDP in current USD, total population.
WDI_INDICATORS = {"NY.GDP.MKTP.CD": "gdp_usd", "SP.POP.TOTL": "population"}
WDI_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?date=2012:2024&format=json&per_page=20000"
# IMF World Economic Outlook (DataMapper API) indicator -> (our variable, unit multiplier).
IMF_INDICATORS = {"NGDPD": ("gdp_usd", 1e9), "LP": ("population", 1e6)}
IMF_URL = "https://www.imf.org/external/datamapper/api/v1/{indicator}"
# CEPII Gravity reports GDP in thousand USD and population in thousands.
GRAVITY_UNITS = {"gdp_o": ("gdp_usd", 1e3), "pop_o": ("population", 1e3)}


def _raw_files(data_dir: Path) -> dict[Path, str]:
    """Every raw file this module needs, with the URL it comes from."""
    raw = data_dir / "raw"
    files = {raw / "gravity" / GRAVITY_ZIP: GRAVITY_URL}
    files |= {raw / "wdi" / f"{i}.json": WDI_URL.format(indicator=i) for i in WDI_INDICATORS}
    files |= {raw / "imf_weo" / f"{i}.json": IMF_URL.format(indicator=i) for i in IMF_INDICATORS}
    return files


def download(data_dir: Path | None = None) -> None:
    """Download each raw source not already present.

    Each file is written to `<name>.part` and renamed when complete, so a reader never
    sees a half-written file.
    """
    data_dir = data_dir or paths.data_dir()
    for target, url in _raw_files(data_dir).items():
        download_file(url, target)


def _long_frame(rows: list[tuple], variable: str) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"iso3": pl.String, "year": pl.Int32, variable: pl.Float64}, orient="row")


def read_wdi(raw_dir: Path) -> pl.DataFrame:
    """WDI GDP and population as `iso3`, `year`, `gdp_usd`, `population`."""
    frames = []
    for indicator, variable in WDI_INDICATORS.items():
        meta, records = json.loads((raw_dir / f"{indicator}.json").read_text(encoding="utf-8"))
        if meta.get("pages", 1) != 1:
            raise ValueError(f"WDI {indicator} came back in several pages; raise per_page")
        rows = [(r["countryiso3code"], int(r["date"]), r["value"]) for r in records if r["countryiso3code"]]
        frames.append(_long_frame(rows, variable))
    return frames[0].join(frames[1], on=["iso3", "year"], how="full", coalesce=True)


def read_imf_weo(raw_dir: Path) -> pl.DataFrame:
    """IMF WEO GDP and population in USD and persons, as `iso3`, `year`, `gdp_usd`, `population`."""
    frames = []
    for indicator, (variable, unit) in IMF_INDICATORS.items():
        series = json.loads((raw_dir / f"{indicator}.json").read_text(encoding="utf-8"))["values"][indicator]
        rows = [(iso3, int(year), v * unit) for iso3, by_year in series.items() for year, v in by_year.items()]
        frames.append(_long_frame(rows, variable))
    return frames[0].join(frames[1], on=["iso3", "year"], how="full", coalesce=True)


def gravity_country_year(gravity: pl.LazyFrame) -> pl.DataFrame:
    """CEPII Gravity's own GDP and population, in USD and persons, per existing country-year."""
    return (
        gravity.filter(pl.col("country_exists_o") == 1)
        .group_by(iso3="iso3_o", year=pl.col("year").cast(pl.Int32))
        .agg((pl.col(column).drop_nulls().first() * unit).alias(name) for column, (name, unit) in GRAVITY_UNITS.items())
        .collect()
    )


def read_baci_countries(raw_dir: Path) -> pl.DataFrame:
    """BACI's country code file: a loose `country_codes_*.csv`, or the one inside the BACI zip."""
    loose = sorted(raw_dir.glob("country_codes_*.csv"))
    if loose:
        return pl.read_csv(loose[-1], encoding="utf8-lossy")
    archives = sorted(raw_dir.glob("BACI_*.zip"))
    if not archives:
        raise FileNotFoundError(f"No BACI country code file or BACI zip in {raw_dir}")
    with zipfile.ZipFile(archives[-1]) as archive:
        member = next(n for n in archive.namelist() if Path(n).name.startswith("country_codes_"))
        return pl.read_csv(archive.read(member), encoding="utf8-lossy")


def _gravity_extract(data_dir: Path) -> Path:
    """Convert CEPII Gravity rows from 2012 on (every column) to Parquet, for lazy scans."""
    raw = data_dir / "raw" / "gravity"
    csv = raw / GRAVITY_CSV
    if not csv.exists():
        with (
            writing(csv) as partial,
            zipfile.ZipFile(raw / GRAVITY_ZIP) as archive,
            archive.open(GRAVITY_CSV) as src,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(src, out)
    target = data_dir / "derived" / "gravity_extract.parquet"
    with writing(target) as partial:
        pl.scan_csv(csv, infer_schema_length=None).filter(pl.col("year") >= YEARS[0]).sink_parquet(partial)
    return target


def build(data_dir: Path | None = None, *, download_missing: bool = True) -> dict[str, Path]:
    """Download what is missing, then build every Gravity and WDI table; return their paths.

    Writes, under `<data_dir>/derived/`:

    - `gravity_extract.parquet`: CEPII Gravity, every column, 2012 on (the policy
      cross-check reads it too);
    - `country_map.parquet`: BACI country code -> ISO3, with Gravity and WDI coverage flags;
    - `gravity_pairs.parquet`: time-invariant pair features by ISO3 pair;
    - `country_year.parquet`: GDP and population for every BACI country, 2012–2024.

    The BACI country code file comes from the BACI download (`raw/baci/`).
    """
    data_dir = data_dir or paths.data_dir()
    if download_missing:
        download(data_dir)
    raw, derived = data_dir / "raw", data_dir / "derived"

    extract = _gravity_extract(data_dir)
    gravity = pl.scan_parquet(extract)
    wdi = read_wdi(raw / "wdi")
    mapping = map_baci_countries(
        read_baci_countries(raw / "baci"),
        gravity_iso3=set(gravity.select(pl.col("iso3_o").unique()).collect()["iso3_o"]),
        wdi_iso3=set(wdi["iso3"]),
    )
    sources = {"wdi": wdi, "imf_weo": read_imf_weo(raw / "imf_weo"), "cepii_gravity": gravity_country_year(gravity)}
    country_year = fill_country_year(sources, iso3=sorted(set(mapping["iso3"])), years=YEARS)
    return {
        "gravity_extract": extract,
        "country_map": write_parquet(mapping, derived / "country_map.parquet"),
        "gravity_pairs": write_parquet(gravity_pairs(gravity), derived / "gravity_pairs.parquet"),
        "country_year": write_parquet(country_year, derived / "country_year.parquet"),
    }


if __name__ == "__main__":
    for name, path in build().items():
        print(f"{name}: {path}")
