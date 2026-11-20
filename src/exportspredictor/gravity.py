"""Gravity features: time-invariant country-pair ties and country-year size.

These derived tables feed the feature table:

- **Pair features** from CEPII Gravity (V202211): distance, contiguity, common official
  language and colonial ties. They are treated as time-invariant, taking each pair's
  latest non-null value.
- **Country-year features**: GDP (current USD) and population for every year 2012–2024.
  World Bank WDI is the main source. Missing values follow one documented rule
  (`fill_country_year`), and every value records the source it came from.
- **Gravity gaps** (#126): about 20 small Markets have no GDP or population in any of those
  sources. `country_year_filled.parquet` fills them from secondary sources (UN AMA, Statistics
  Netherlands, the CIA World Factbook; `fill_gaps`), never overriding a primary value, and the
  feature set `exporter_count` reads it. Markets no source covers stay missing (`GDP_GAPS`).

BACI identifies countries by numeric code; `map_baci_countries` maps them to the ISO3
codes Gravity and WDI use. `build` is the single entry point the data rebuild script
calls: it downloads what is missing and writes the derived Parquet files.
"""

import json
import re
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
_COUNTRY_YEAR_SCHEMA = {"iso3": pl.String, "year": pl.Int32, "gdp_usd": pl.Float64, "population": pl.Float64}


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


def fill_gaps(country_year: pl.DataFrame, secondary: dict[str, pl.DataFrame], *, years: list[int]) -> pl.DataFrame:
    """`country_year` with each variable no primary source holds for a country filled from `secondary`.

    A country whose GDP (or population) is missing in every year of `country_year` takes it from
    the secondary sources, in order, by the same rule as `fill_country_year`. Every other value
    is left as it is, so a secondary source never overrides a primary one. An interpolated or
    carried value names the source it came from: "carried (cia_factbook)". A one-off estimate
    dated up to `PIN_YEARS` after the last of `years` (a 2025 population estimate) stands for the
    last year, unless the source has a value for that year itself. Values before the first year
    are never read: a dissolved country's history must not become its GDP.
    """
    secondary = {name: _pinned(frame, years) for name, frame in secondary.items()}
    out = country_year
    for variable, source_column in COUNTRY_YEAR_VARIABLES.items():
        gaps = (
            country_year.group_by("iso3").agg(pl.col(variable).is_null().all().alias("_gap"))
            .filter("_gap")["iso3"].sort().to_list()
        )
        if not gaps or not secondary:
            continue
        filled = fill_country_year(secondary, iso3=gaps, years=years)
        origin = filled.filter(pl.col(source_column).is_in(list(secondary))).group_by("iso3").agg(
            _origin=pl.col(source_column).sort_by("year").first()
        )
        derived = pl.col(source_column).is_in(["interpolated", "carried"])
        filled = filled.join(origin, on="iso3", how="left").select(
            "iso3",
            pl.col("year").cast(out.schema["year"]),
            pl.col(variable).alias("_value"),
            pl.when(derived).then(pl.format("{} ({})", source_column, "_origin"))
            .otherwise(pl.col(source_column)).alias("_source"),
        )
        gap = pl.col("iso3").is_in(gaps)
        out = (
            out.join(filled, on=["iso3", "year"], how="left")
            .with_columns(
                pl.when(gap).then("_value").otherwise(variable).alias(variable),
                pl.when(gap).then("_source").otherwise(source_column).alias(source_column),
            )
            .drop("_value", "_source")
        )
    return out


# How many years after the last year a one-off estimate may be dated and still stand for it.
PIN_YEARS = 2


def _pinned(source: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """`source` within `years`, a value up to PIN_YEARS later standing for the last year (the closest-dated wins)."""
    first, last = min(years), max(years)
    parts = []
    for variable in COUNTRY_YEAR_VARIABLES:
        year = pl.col("year").cast(pl.Int32)
        parts.append(
            source.filter(pl.col(variable).is_not_null(), year.is_between(first, last + PIN_YEARS))
            .select("iso3", year.clip(first, last).alias("year"), (year - year.clip(first, last)).abs().alias("_d"),
                    pl.col(variable).cast(pl.Float64))
            .sort("_d")
            .unique(["iso3", "year"], keep="first")
            .drop("_d")
        )
    return parts[0].join(parts[1], on=["iso3", "year"], how="full", coalesce=True).select(*_COUNTRY_YEAR_SCHEMA)


def fill_record(country_year: pl.DataFrame) -> dict:
    """Which countries' values came from a secondary source, and which have no GDP at all.

    `filled` maps each such country to {variable: its distinct source labels}; `gdp_missing`
    lists the countries whose GDP is missing in every year.
    """
    filled: dict[str, dict[str, list[str]]] = {}
    secondary = "|".join(SECONDARY_SOURCES)
    for variable, source_column in COUNTRY_YEAR_VARIABLES.items():
        if source_column not in country_year.columns:
            continue
        rows = (
            country_year.filter(pl.col(source_column).str.contains(secondary))
            .group_by("iso3").agg(pl.col(source_column).unique().sort())
            .sort("iso3")
        )
        for iso3, labels in rows.iter_rows():
            filled.setdefault(iso3, {})[variable] = list(labels)
    missing = country_year.group_by("iso3").agg(pl.col("gdp_usd").is_null().all().alias("_gap")).filter("_gap")
    return {"filled": dict(sorted(filled.items())), "gdp_missing": sorted(missing["iso3"])}


_FIGURE = re.compile(r"^\$?([\d.,]+)\s*(trillion|billion|million)?\s*\((\d{4})")
_SCALE = {None: 1.0, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def factbook_figure(text: str) -> tuple[float, int] | None:
    """A World Factbook figure such as "$2.044 billion (2014 est.)" as (2.044e9, 2014); None if it has none."""
    match = _FIGURE.match(text.strip())
    if match is None:
        return None
    number, scale, year = match.groups()
    return float(number.replace(",", "")) * _SCALE[scale], int(year)


def read_factbook(raw_dir: Path) -> pl.DataFrame:
    """GDP at the official exchange rate and population from World Factbook entries (`<ISO3>.json`).

    Each figure is one estimate for the year it states. A purchasing-power estimate is a different
    measure from current-dollar GDP, so it is not read.
    """
    rows = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        gdp = entry.get("Economy", {}).get("GDP (official exchange rate)", {}).get("text", "")
        population = entry.get("People and Society", {}).get("Population", {}).get("total", {}).get("text", "")
        for variable, text in (("gdp_usd", gdp), ("population", population)):
            figure = factbook_figure(text)
            if figure is not None:
                value, year = figure
                rows.append((path.stem, year, value if variable == "gdp_usd" else None,
                             value if variable == "population" else None))
    return pl.DataFrame(rows, schema=_COUNTRY_YEAR_SCHEMA, orient="row")


def unsd_ama(gdp_sheet: pl.DataFrame, population_sheet: pl.DataFrame, code_to_iso3: dict[int, str]) -> pl.DataFrame:
    """UN Statistics Division AMA GDP (current USD) and population, from its two download sheets.

    Each sheet, read without a header, has a title row, then a header row: CountryID, Country,
    the measure's column (IndicatorName for GDP, Measure for population) and one column per year. Countries are matched on their UN code; codes not in `code_to_iso3` are
    dropped.
    """

    def long(sheet: pl.DataFrame, measure_column: str, measure: str, variable: str) -> pl.DataFrame:
        header = [str(h) for h in sheet.row(1)]
        table = sheet.slice(2).rename(dict(zip(sheet.columns, header)))
        years = [h for h in header if h.isdigit()]
        return (
            table.filter(pl.col(measure_column) == measure)
            .with_columns(pl.col("CountryID").cast(pl.Int64).replace_strict(code_to_iso3, default=None).alias("iso3"))
            .filter(pl.col("iso3").is_not_null())
            .unpivot(on=years, index="iso3", variable_name="year", value_name=variable)
            .select("iso3", pl.col("year").cast(pl.Int32), pl.col(variable).cast(pl.Float64))
            .filter(pl.col(variable).is_not_null())
        )

    gdp = long(gdp_sheet, "IndicatorName", "Gross Domestic Product (GDP)", "gdp_usd")
    population = long(population_sheet, "Measure", "Population", "population")
    return gdp.join(population, on=["iso3", "year"], how="full", coalesce=True).select(*_COUNTRY_YEAR_SCHEMA)


def read_unsd_ama(raw_dir: Path, code_to_iso3: dict[int, str]) -> pl.DataFrame:
    """`unsd_ama` on the downloaded workbooks."""
    sheets = [
        pl.read_excel(Path(raw_dir) / name, sheet_id=1, has_header=False, infer_schema_length=0)
        for name in UNSD_AMA_FILES
    ]
    return unsd_ama(*sheets, code_to_iso3)


def read_cbs_statline(path: Path) -> pl.DataFrame:
    """The Caribbean Netherlands' GDP (CBS StatLine table 84789ENG, million USD at current prices) as BES.

    BACI's Bonaire (535) is Bonaire, Sint Eustatius and Saba, which CBS reports together as CN01.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))["value"]
    rows = [
        ("BES", int(r["Periods"][:4]), r["GDPValueAtCurrentPrices_1"] * 1e6, None)
        for r in records
        if r["CarribeanNetherlands"].strip() == "CN01" and r["GDPValueAtCurrentPrices_1"] is not None
    ]
    return pl.DataFrame(rows, schema=_COUNTRY_YEAR_SCHEMA, orient="row")


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
GRAVITY_VERSION = "V202211"
GRAVITY_ZIP = f"Gravity_csv_{GRAVITY_VERSION}.zip"
GRAVITY_CSV = f"Gravity_{GRAVITY_VERSION}.csv"
GRAVITY_URL = f"https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/{GRAVITY_ZIP}"
# World Bank WDI indicator -> our variable: GDP in current USD, total population.
WDI_INDICATORS = {"NY.GDP.MKTP.CD": "gdp_usd", "SP.POP.TOTL": "population"}
WDI_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?date=2012:2024&format=json&per_page=20000"
# IMF World Economic Outlook (DataMapper API) indicator -> (our variable, unit multiplier).
IMF_INDICATORS = {"NGDPD": ("gdp_usd", 1e9), "LP": ("population", 1e6)}
IMF_URL = "https://www.imf.org/external/datamapper/api/v1/{indicator}"
# CEPII Gravity reports GDP in thousand USD and population in thousands.
GRAVITY_UNITS = {"gdp_o": ("gdp_usd", 1e3), "pop_o": ("population", 1e3)}

# Secondary sources (#126), read only for a variable that WDI, the IMF and CEPII Gravity all lack
# for a country (`fill_gaps`); in order of preference.
# UN Statistics Division, National Accounts Analysis of Main Aggregates (AMA), from
# https://unstats.un.org/unsd/snaama/Downloads: GDP at current prices in US dollars (file 2), and
# exchange rates and population (file 30), for every UN member and territory, 1970 on.
UNSD_AMA_URL = "https://unstats.un.org/unsd/amaapi/api/file/{number}"
UNSD_AMA_FILES = {"gdp_current_usd.xlsx": 2, "exchange_rates_population.xlsx": 30}
# Statistics Netherlands (CBS) StatLine table 84789ENG: Caribbean Netherlands GDP, million USD.
CBS_STATLINE_URL = "https://opendata.cbs.nl/ODataApi/odata/84789ENG/TypedDataSet?$format=json"
CBS_STATLINE_FILE = "84789ENG.json"
# The CIA World Factbook, as archived weekly in JSON by the factbook.json project
# (https://github.com/factbook/factbook.json): one entry per territory, by its Factbook code.
FACTBOOK_URL = "https://raw.githubusercontent.com/factbook/factbook.json/master/{entry}.json"
FACTBOOK_ENTRIES = {
    "GIB": "europe/gi",
    "FLK": "south-america/fk",
    "SHN": "africa/sh",
    "NIU": "australia-oceania/ne",
    "TKL": "australia-oceania/tl",
    "WLF": "australia-oceania/wf",
    "SPM": "north-america/sb",
    "BLM": "central-america-n-caribbean/tb",
    "PCN": "australia-oceania/pc",
    "NFK": "australia-oceania/nf",
    "CCK": "australia-oceania/ck",
    "CXR": "australia-oceania/kt",
    "IOT": "south-asia/io",
    "ATF": "antarctica/fs",
}
SECONDARY_SOURCES = ("unsd_ama", "cbs_statline", "cia_factbook")
# The Markets no source gives a GDP for, so their GDP feature stays missing (a known limitation):
# Factbook gives none, and CBS, AMA and WDI do not cover them. Checked on the real build.
GDP_GAPS = {
    "ATF": "French Southern and Antarctic Lands",
    "BLM": "Saint Barthélemy",
    "CCK": "Cocos (Keeling) Islands",
    "CXR": "Christmas Island",
    "IOT": "British Indian Ocean Territory",
    "NFK": "Norfolk Island",
    "NIU": "Niue",
    "PCN": "Pitcairn",
    "SHN": "Saint Helena",
    "WLF": "Wallis and Futuna",
}


def _raw_files(data_dir: Path) -> dict[Path, str]:
    """Every raw file this module needs, with the URL it comes from."""
    raw = data_dir / "raw"
    files = {raw / "gravity" / GRAVITY_ZIP: GRAVITY_URL}
    files |= {raw / "wdi" / f"{i}.json": WDI_URL.format(indicator=i) for i in WDI_INDICATORS}
    files |= {raw / "imf_weo" / f"{i}.json": IMF_URL.format(indicator=i) for i in IMF_INDICATORS}
    files |= {raw / "unsd_ama" / name: UNSD_AMA_URL.format(number=n) for name, n in UNSD_AMA_FILES.items()}
    files |= {raw / "cbs_statline" / CBS_STATLINE_FILE: CBS_STATLINE_URL}
    files |= {raw / "factbook" / f"{iso3}.json": FACTBOOK_URL.format(entry=e) for iso3, e in FACTBOOK_ENTRIES.items()}
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
    - `country_year.parquet`: GDP and population for every BACI country, 2012–2024;
    - `country_year_filled.parquet`: the same, with the gaps filled from secondary sources
      (`build_filled`).

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
    outputs = {
        "gravity_extract": extract,
        "country_map": write_parquet(mapping, derived / "country_map.parquet"),
        "gravity_pairs": write_parquet(gravity_pairs(gravity), derived / "gravity_pairs.parquet"),
        "country_year": write_parquet(country_year, derived / "country_year.parquet"),
    }
    return outputs | {"country_year_filled": build_filled(data_dir, download_missing=False)}


COUNTRY_YEAR_FILLED = "country_year_filled.parquet"


def secondary_sources(raw: Path, mapping: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The secondary sources (`SECONDARY_SOURCES`, in order) whose raw files are under `raw`."""
    readers = {
        "unsd_ama": (
            [raw / "unsd_ama" / name for name in UNSD_AMA_FILES],
            lambda: read_unsd_ama(raw / "unsd_ama", dict(zip(mapping["baci_code"], mapping["iso3"]))),
        ),
        "cbs_statline": (
            [raw / "cbs_statline" / CBS_STATLINE_FILE], lambda: read_cbs_statline(raw / "cbs_statline" / CBS_STATLINE_FILE)
        ),
        "cia_factbook": ([raw / "factbook"], lambda: read_factbook(raw / "factbook")),
    }
    return {name: read() for name, (needed, read) in readers.items() if all(path.exists() for path in needed)}


def build_filled(data_dir: Path | None = None, *, download_missing: bool = True) -> Path:
    """Write `country_year_filled.parquet`: `country_year.parquet` with its gaps filled (`fill_gaps`).

    Reads the derived `country_year.parquet` and `country_map.parquet` and the secondary sources
    on disk, downloading those missing first. `country_year.parquet` itself is never rewritten,
    so feature builds made from it stay reproducible.
    """
    data_dir = data_dir or paths.data_dir()
    if download_missing:
        download(data_dir)
    derived = data_dir / "derived"
    country_year = pl.read_parquet(derived / "country_year.parquet")
    mapping = pl.read_parquet(derived / "country_map.parquet")
    filled = fill_gaps(country_year, secondary_sources(data_dir / "raw", mapping), years=YEARS)
    return write_parquet(filled, derived / COUNTRY_YEAR_FILLED)


if __name__ == "__main__":
    for name, path in build().items():
        print(f"{name}: {path}")
