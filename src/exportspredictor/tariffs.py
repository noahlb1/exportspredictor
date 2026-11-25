"""Applied Tariffs: the HS6 rate an Importer applies to an Exporter's goods, 2012–2024 (ADR 0006).

Source: UNCTAD TRAINS through the WITS API (docs/research/hs6-tariff-data.md). TRAINS ends
in 2021. The rate is the simple average of the lower of the preferential and MFN rates,
on BACI's HS12 codes.

The download is cached per reporter-year (`raw/trains/<year>/<reporter>.parquet`): each
reporter-year fetches its MFN schedule (partner `000`) and every preference partner, and
is written only when complete, so an interrupted download resumes where it stopped.

`panel` turns the raw rows into three compact tables (`Tariffs`), never the dense
Importer × Exporter × Product grid:
- **HS revision**: reporter-years in HS17 (and HS07) map to HS12 through the UNSD
  correlation tables. Where several codes map to one HS12 code, the rate is their mean,
  and a preference counts on the codes it covers, with MFN on the rest.
- **EU**: members apply the EU schedule (`EUN`). Preferences granted to EU groups are
  expanded to the EU's members in that year. Other partner groups (GSP beneficiaries and
  the like) have no membership list in the API, so they are dropped and their members pay
  MFN unless they have a preference of their own.
- **Gap fill**: a missing reporter-year uses the same schedule's nearest year within
  `MAX_GAP_YEARS` (the earlier year on a tie); beyond that the tariff is missing.
- **2022–2024** carry 2021 forward and are marked `tariff_imputed`.

`applied_tariffs` attaches the rate to rows keyed by exporter, importer, hs6 and year. It
sets Customs Union pairs (intra-EU included) to 0, and pairs whose Trade Agreement entered
force after 2021 (RCEP, UK–Australia, UK–New Zealand, ...) to 0 in the carried years. It
also returns the Importer's MFN rate, which in 2022–2024 is the 2021 MFN rate that
Scenarios restore when they remove a Trade Agreement.
"""

import json
import re
import time
import urllib.error
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from exportspredictor import downloads
from exportspredictor.paths import data_dir

WITS_API = "https://wits.worldbank.org/API/V1"
AVAILABILITY_URL = f"{WITS_API}/wits/datasource/trn/dataavailability/country/all/year/all"
COUNTRIES_URL = f"{WITS_API}/wits/datasource/trn/country/ALL"

TRAINS_YEARS = list(range(2012, 2022))
LAST_TRAINS_YEAR = 2021  # TRAINS has no later year; 2022–2024 carry it forward
# Shown with a pair whose Baseline-year Applied Tariff is carried forward (ADR 0006).
TARIFF_CARRIED_FORWARD_CAVEAT = (
    f"Applied Tariff carried forward from {LAST_TRAINS_YEAR}: tariff data end in {LAST_TRAINS_YEAR}, so later "
    f"years use the {LAST_TRAINS_YEAR} rate, or 0 where a Trade Agreement entered force after {LAST_TRAINS_YEAR}."
)
YEARS = list(range(2012, 2025))
MAX_GAP_YEARS = 3
MFN_PARTNER = "000"  # TRAINS' "World": the MFN schedule
EU_SCHEDULE = "EUN"

# Countries applying the EU schedule: iso3 -> (first year, last year), by the 1 July rule
# the policy features use. Croatia joined on 1 July 2013; the UK applied the EU tariff
# until the end of the transition period (31 Dec 2020).
_EU27 = ["AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GRC", "HUN",
         "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE"]
EU_MEMBERS = {**{iso3: (1958, 9999) for iso3 in _EU27}, "HRV": (2014, 9999), "GBR": (1958, 2020)}

# WITS partner codes that stand for the EU, with any extra members they name.
EU_GROUPS = {
    "918": (),  # EUN, the EU as a partner
    "N04": (),  # European Union
    "A36": (),  # EU 28 members 2013
    "M02": (),  # EU 27 members
    "N63": (),  # EU25
    "M13": (),  # EU countries (32 territories): Switzerland 2009
    "N55": (),  # EU beneficiaries: Switzerland 2004
    "A37": ("GUF", "SMR"),  # EU 28 + GUF, SMR
    "A11": ("CHE", "ISL", "LIE", "NOR"),  # EU + EFTA
}

# WITS's own codes where they differ from ISO 3166 alpha-3.
WITS_ISO3 = {"MNT": "MNE", "ROM": "ROU", "SER": "SRB", "SUD": "SDN", "TMP": "TLS", "ZAR": "COD"}

RAW_SCHEMA = {
    "reporter": pl.String,  # ISO3 (EUN for the EU)
    "partner": pl.String,  # WITS partner code: 000 (MFN), a numeric country or a group code
    "year": pl.Int16,
    "product": pl.String,  # HS6 in the reporter's own revision
    "nomenclature": pl.String,  # H3 = HS07, H4 = HS12, H5 = HS17, ...
    "tariff_type": pl.String,  # MFN or PREF
    "rate": pl.Float64,  # simple average of ad valorem rates, percent
}


# --- Download ------------------------------------------------------------------------------


def raw_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "raw" / "trains"


def tariff_url(reporter_code: str, partner: str, year: int) -> str:
    """The WITS request for one reporter's full HS6 schedule towards one partner in one year."""
    return (f"{WITS_API}/SDMX/V21/datasource/TRN/reporter/{reporter_code}/partner/{partner}"
            f"/product/all/year/{year}/datatype/reported")


def reporter_year_path(root: Path | None, reporter: str, year: int) -> Path:
    return raw_dir(root) / str(year) / f"{reporter}.parquet"


def manifest_path(root: Path | None = None) -> Path:
    return raw_dir(root) / "manifest.json"


def manifest(root: Path | None = None) -> dict:
    """What the last complete download pulled: reporter_years and the date it finished (`pulled`)."""
    return json.loads(manifest_path(root).read_text(encoding="utf-8"))


def download(
    root: Path | None = None,
    *,
    years: list[int] = TRAINS_YEARS,
    fetch: Callable[..., bytes] = downloads.fetch,
    workers: int = 4,
    retries: int = 3,
) -> Path:
    """Fetch every TRAINS reporter-year in `years` not already cached. Safe to interrupt and re-run.

    Writes the availability and country lists, one Parquet file per reporter-year, and
    finally `manifest.json`, which records the pull date. Returns the raw folder.
    """
    folder = raw_dir(root)
    availability = read_availability(_cached(folder / "availability.xml", AVAILABILITY_URL, fetch))
    _cached(folder / "countries.xml", COUNTRIES_URL, fetch)
    # A few countries are listed under two WITS codes in one year (Sudan as 729 and 736):
    # keep the entry with the most partners, as the cache holds one file per ISO3 and year.
    wanted = (
        availability.filter(pl.col("year").is_in(years))
        .sort(pl.col("partners").list.len(), descending=True)
        .unique(["reporter", "year"], keep="first", maintain_order=True)
    )
    todo = [
        row for row in wanted.iter_rows(named=True)
        if not reporter_year_path(root, row["reporter"], row["year"]).exists()
    ]

    def fetch_one(row: dict) -> None:
        frames = [
            parse_tariff_xml(text, row["reporter"])
            for partner in row["partners"]
            if (text := _fetch_text(tariff_url(row["code"], partner, row["year"]), fetch, retries)) is not None
        ]
        table = pl.concat([pl.DataFrame(schema=RAW_SCHEMA), *frames])
        downloads.write_parquet(table, reporter_year_path(root, row["reporter"], row["year"]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_one, todo):
            pass
    with downloads.writing(manifest_path(root)) as part:
        part.write_text(json.dumps({
            "source": "UNCTAD TRAINS via the WITS API (reported rates)",
            "years": list(years),
            "reporter_years": wanted.height,
            "pulled": datetime.now(UTC).date().isoformat(),
        }, indent=2), encoding="utf-8")
    return folder


def _cached(target: Path, url: str, fetch: Callable[..., bytes]) -> str:
    if not target.exists():
        body = fetch(url)
        with downloads.writing(target) as part:
            part.write_bytes(body)
    return target.read_text(encoding="utf-8-sig")


def _fetch_text(url: str, fetch: Callable[..., bytes], retries: int) -> str | None:
    """The body of `url`, None if WITS has no records for it, retrying transient failures."""
    for attempt in range(retries + 1):
        try:
            return fetch(url).decode("utf-8-sig")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == retries:
                raise
        except OSError:
            if attempt == retries:
                raise
        time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


# --- Reading WITS responses ----------------------------------------------------------------

_REPORTER = re.compile(
    r'<wits:reporter countrycode="(?P<code>[^"]*)" iso3Code="(?P<iso3>[^"]*)".*?'
    r"<wits:year>(?P<year>\d+)</wits:year>.*?reporternernomenclaturecode=\"(?P<nomen>[^\"]*)\".*?"
    r"<wits:partnerlist>(?P<partners>[^<]*)</wits:partnerlist>",
    re.DOTALL,
)
_COUNTRY = re.compile(
    r'<wits:country countrycode="(?P<code>[^"]*)"[^>]*isgroup="(?P<group>[^"]*)"[^>]*>\s*'
    r"<wits:iso3Code>(?P<iso3>[^<]*)</wits:iso3Code>"
)
_SERIES = re.compile(r"<Series (?P<series>[^>]*)>\s*<Obs (?P<obs>[^>]*?)/?>")
_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


def read_availability(text: str) -> pl.DataFrame:
    """One row per TRAINS reporter-year: code, reporter (ISO3), year, nomenclature, partners."""
    rows = [
        {
            "code": m["code"],
            "reporter": WITS_ISO3.get(m["iso3"], m["iso3"]),
            "year": int(m["year"]),
            "nomenclature": m["nomen"],
            "partners": [p for p in m["partners"].split(";") if p],
        }
        for m in _REPORTER.finditer(text)
    ]
    schema = {"code": pl.String, "reporter": pl.String, "year": pl.Int16, "nomenclature": pl.String,
              "partners": pl.List(pl.String)}
    return pl.DataFrame(rows, schema=schema)


def read_countries(text: str) -> dict[str, str]:
    """WITS country code -> ISO3, for single countries (groups are left out)."""
    return {m["code"]: WITS_ISO3.get(m["iso3"], m["iso3"]) for m in _COUNTRY.finditer(text) if m["group"] == "No"}


def parse_tariff_xml(text: str, reporter: str) -> pl.DataFrame:
    """The rows of one WITS SDMX (structure-specific XML) tariff response, as RAW_SCHEMA."""
    rows = []
    for match in _SERIES.finditer(text):
        series, obs = dict(_ATTRIBUTE.findall(match["series"])), dict(_ATTRIBUTE.findall(match["obs"]))
        if obs.get("OBS_VALUE", "") in ("", "NaN"):
            continue  # no ad valorem line in this HS6
        rows.append((reporter, series["PARTNER"], int(obs["TIME_PERIOD"]), series["PRODUCTCODE"],
                     obs.get("NOMENCODE"), obs.get("TARIFFTYPE"), float(obs["OBS_VALUE"])))
    return pl.DataFrame(rows, schema=RAW_SCHEMA, orient="row")


def read_raw(root: Path | None = None) -> pl.DataFrame:
    """Every cached reporter-year, as RAW_SCHEMA."""
    files = sorted(raw_dir(root).glob("*/*.parquet"))
    if not files:
        return pl.DataFrame(schema=RAW_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


# --- The panel -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tariffs:
    """The Applied Tariff panel in three compact tables; `applied_tariffs` combines them per row.

    - schedules: importer, year, schedule (the TRAINS reporter whose rates apply, EUN for
      EU members), source_year (the TRAINS year used after gap fill and carry-forward),
      tariff_imputed (the year is after 2021, so its rates are carried forward).
    - mfn: schedule, source_year, hs6 (HS12), mfn_tariff (percent).
    - preferences: schedule, source_year, exporter, hs6, preferential_tariff (percent),
      only where it is below the MFN rate.
    """

    schedules: pl.DataFrame
    mfn: pl.DataFrame
    preferences: pl.DataFrame


def panel(
    raw: pl.DataFrame, concordance: pl.DataFrame, countries: dict[str, str], years: list[int] = YEARS
) -> Tariffs:
    """Build the panel from raw TRAINS rows (RAW_SCHEMA) for the target `years`.

    `concordance` maps (nomenclature, product) to HS12 `hs6` for every revision but HS12
    itself (`read_concordances`); rows in a revision it does not cover are dropped.
    `countries` maps WITS partner codes to ISO3 (`read_countries`).
    """
    hs12 = _to_hs12(raw, concordance)
    keys = ["reporter", "year", "product", "hs6"]
    mfn_codes = (
        hs12.filter((pl.col("partner") == MFN_PARTNER) & (pl.col("tariff_type") == "MFN"))
        .group_by(keys).agg(mfn=pl.col("rate").mean())
    )
    mfn = mfn_codes.group_by("reporter", "year", "hs6").agg(mfn_tariff=pl.col("mfn").mean(), codes=pl.len())
    # A preference saves (MFN − preference) on the codes it covers; the HS12 rate averages
    # that over every code mapping to it, so codes without a preference pay MFN.
    preferences = (
        hs12.filter((pl.col("partner") != MFN_PARTNER) & (pl.col("tariff_type") == "PREF"))
        .group_by(*keys, "partner").agg(preference=pl.col("rate").min())
        .join(mfn_codes, on=keys)
        .group_by("reporter", "year", "partner", "hs6")
        .agg(saving=(pl.col("mfn") - pl.col("preference")).clip(lower_bound=0).sum())
        .filter(pl.col("saving") > 0)
        .join(mfn, on=["reporter", "year", "hs6"])
        .with_columns(preferential_tariff=pl.col("mfn_tariff") - pl.col("saving") / pl.col("codes"))
        .join(_partner_members(countries, hs12["year"].unique().to_list()), on=["partner", "year"])
        .filter(pl.col("exporter") != pl.col("reporter"))
        .group_by(schedule="reporter", source_year="year", exporter="exporter", hs6="hs6")
        .agg(pl.col("preferential_tariff").min())
    )
    return Tariffs(
        schedules=_schedules(mfn.select("reporter", "year").unique(), years),
        mfn=mfn.select(schedule="reporter", source_year="year", hs6="hs6", mfn_tariff="mfn_tariff")
        .sort("schedule", "source_year", "hs6"),
        preferences=preferences.sort("schedule", "source_year", "exporter", "hs6"),
    )


def _to_hs12(raw: pl.DataFrame, concordance: pl.DataFrame) -> pl.DataFrame:
    """Raw rows with an HS12 `hs6`; a code in another revision becomes every HS12 code it maps to."""
    concordance = concordance.select("nomenclature", "product", "hs6").filter(pl.col("nomenclature") != "H4")
    return pl.concat([
        raw.filter(pl.col("nomenclature") == "H4").with_columns(hs6=pl.col("product")),
        raw.join(concordance, on=["nomenclature", "product"]),
    ])


def eu_members(year: int) -> list[str]:
    """Countries applying the EU schedule in `year` (1 July rule)."""
    return sorted(iso3 for iso3, (first, last) in EU_MEMBERS.items() if first <= year <= last)


def _partner_members(countries: dict[str, str], years: list[int]) -> pl.DataFrame:
    """partner, year, exporter: single countries are themselves; EU groups are the EU's members that year."""
    rows = [(code, year, iso3) for code, iso3 in countries.items() if code not in EU_GROUPS for year in years]
    rows += [
        (code, year, iso3)
        for code, extra in EU_GROUPS.items()
        for year in years
        for iso3 in [*eu_members(year), *extra]
    ]
    return pl.DataFrame(rows, schema={"partner": pl.String, "year": pl.Int16, "exporter": pl.String}, orient="row")


def _schedules(available: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """Which schedule and TRAINS year each Importer's tariffs come from in each target year."""
    have: dict[str, list[int]] = {}
    for reporter, year in available.iter_rows():
        have.setdefault(reporter, []).append(year)
    importers = sorted((set(have) - {EU_SCHEDULE}) | set(EU_MEMBERS))
    rows = []
    for importer in importers:
        for year in years:
            observed = min(year, LAST_TRAINS_YEAR)
            schedule = EU_SCHEDULE if importer in eu_members(observed) else importer
            near = [y for y in have.get(schedule, []) if abs(y - observed) <= MAX_GAP_YEARS]
            if near:
                source = min(near, key=lambda y: (abs(y - observed), y))
                rows.append((importer, year, schedule, source, year > LAST_TRAINS_YEAR))
    schema = {"importer": pl.String, "year": pl.Int16, "schedule": pl.String, "source_year": pl.Int16,
              "tariff_imputed": pl.Boolean}
    return pl.DataFrame(rows, schema=schema, orient="row")


def applied_tariffs(rows: pl.DataFrame, tariffs: Tariffs, agreements: pl.DataFrame) -> pl.DataFrame:
    """Add applied_tariff, mfn_tariff (percent) and tariff_imputed to rows keyed by exporter, importer, hs6, year.

    `agreements` is the policy table (exporter, importer, year, trade_agreement, customs_union).
    The Applied Tariff is the lower of the preferential and MFN rates; 0 for a Customs
    Union pair, and 0 in a carried-forward year for a pair whose Trade Agreement entered
    force after 2021. It is missing where the Importer has no schedule within the gap-fill
    window. `tariff_imputed` is True in every year after 2021.
    """
    year = rows.schema["year"]
    policy = agreements.select("exporter", "importer", pl.col("year").cast(year),
                               _agreement="trade_agreement", _union="customs_union")
    before = agreements.filter(pl.col("year") == LAST_TRAINS_YEAR).select(
        "exporter", "importer", _before="trade_agreement")
    after = pl.col("year") > LAST_TRAINS_YEAR
    zero = (pl.col("_union") == 1) | (after & (pl.col("_agreement") == 1) & (pl.col("_before").fill_null(0) == 0))
    return (
        rows.join(tariffs.schedules.select("importer", pl.col("year").cast(year), _schedule="schedule",
                                           _source="source_year"), on=["importer", "year"], how="left")
        .join(tariffs.mfn.rename({"schedule": "_schedule", "source_year": "_source"}),
              on=["_schedule", "_source", "hs6"], how="left")
        .join(tariffs.preferences.rename({"schedule": "_schedule", "source_year": "_source"}),
              on=["_schedule", "_source", "exporter", "hs6"], how="left")
        .join(policy, on=["exporter", "importer", "year"], how="left")
        .join(before, on=["exporter", "importer"], how="left")
        .with_columns(
            applied_tariff=pl.when(zero).then(0.0)
            .otherwise(pl.min_horizontal("mfn_tariff", "preferential_tariff")),
            tariff_imputed=after,
        )
        .select(*rows.columns, "applied_tariff", "mfn_tariff", "tariff_imputed")
    )


def zero_unrecorded_preferences(rows: pl.DataFrame) -> pl.DataFrame:
    """Feature rows with the Applied Tariff set to 0 where a Trade Agreement's preference is unrecorded.

    An Exporter–Importer pair has an *unrecorded preference* in a year when it has a Trade Agreement, no Customs
    Union, and on none of its rows with a known MFN rate is the Applied Tariff below MFN: TRAINS holds no preference
    schedule for the agreement, so `applied_tariffs` charged MFN everywhere. Those rows'
    `log_applied_tariff` becomes 0 where the MFN rate is known, and `unrecorded_preference` marks the pair's rows.
    Read from `rows` alone (`log_applied_tariff`, `mfn_tariff`, `trade_agreement`, `customs_union`), so a pair counts
    as recorded if any of its rows here shows a preference. A research probe; the model is trained without it.
    """
    pair = ["exporter", "importer", "year"]
    applied = (pl.col("log_applied_tariff").exp() - 1.0) * 100.0
    below_mfn = (applied < pl.col("mfn_tariff") - 1e-3).fill_null(False)
    unrecorded = (
        (pl.col("trade_agreement") == 1) & (pl.col("customs_union") == 0) & ~below_mfn.any().over(pair)
    ).fill_null(False)
    return rows.with_columns(
        log_applied_tariff=pl.when(unrecorded & pl.col("mfn_tariff").is_not_null())
        .then(pl.lit(0.0, rows.schema["log_applied_tariff"]))
        .otherwise(pl.col("log_applied_tariff")),
        unrecorded_preference=unrecorded,
    )


# --- HS concordances (UN Statistics Division) ----------------------------------------------

_UNSD = "https://unstats.un.org/unsd/classifications/Econ/tables"


@dataclass(frozen=True)
class Concordance:
    """One UNSD correlation table: the sheet, and which columns hold the old and the HS12 code."""

    url: str
    file: str
    sheet: str
    old_column: int
    hs12_column: int


CONCORDANCES = {
    "H5": Concordance(f"{_UNSD}/HS2017toHS2012ConversionAndCorrelationTables.xlsx",
                      "HS2017toHS2012ConversionAndCorrelationTables.xlsx", "Correlation HS17-HS12", 0, 2),
    "H3": Concordance(f"{_UNSD}/HS%202012%20to%20HS%202007%20Correlation%20and%20conversion%20tables.xls",
                      "HS2012toHS2007CorrelationAndConversionTables.xls", "Correlation HS2012-HS2007", 2, 0),
}


def concordance_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "raw" / "unsd"


def read_concordances(root: Path | None = None) -> pl.DataFrame:
    """nomenclature, product (old code), hs6 (HS12) for every revision in CONCORDANCES, downloading if missing."""
    frames = []
    for nomenclature, table in CONCORDANCES.items():
        path = concordance_dir(root) / table.file
        downloads.download(table.url, path)
        sheet = pl.read_excel(path, sheet_name=table.sheet, has_header=False, infer_schema_length=0)
        old, new = sheet.columns[table.old_column], sheet.columns[table.hs12_column]
        frames.append(
            sheet.select(nomenclature=pl.lit(nomenclature), product=pl.col(old).str.strip_chars(),
                         hs6=pl.col(new).str.strip_chars())
            .filter(pl.col("product").str.contains(r"^\d{6}$") & pl.col("hs6").str.contains(r"^\d{6}$"))
            .unique()
        )
    return pl.concat(frames)


# --- Build entry point ---------------------------------------------------------------------


def derived_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / "tariffs"


def build(root: Path | None = None) -> dict[str, Path]:
    """Download what is missing, then write the panel's three tables under `derived/tariffs/`."""
    if not manifest_path(root).exists():
        download(root)
    countries = read_countries((raw_dir(root) / "countries.xml").read_text(encoding="utf-8-sig"))
    built = panel(read_raw(root), read_concordances(root), countries)
    return {
        name: downloads.write_parquet(getattr(built, name), derived_dir(root) / f"{name}.parquet")
        for name in ("schedules", "mfn", "preferences")
    }


def load(root: Path | None = None) -> Tariffs:
    """The panel `build` wrote."""
    return Tariffs(**{name: pl.read_parquet(derived_dir(root) / f"{name}.parquet")
                      for name in ("schedules", "mfn", "preferences")})


if __name__ == "__main__":
    for name, path in build().items():
        print(f"{name}: {path}")
