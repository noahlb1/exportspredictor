"""BACI HS12 Flows: download once, convert once to Parquet, then scan lazily.

Source: CEPII BACI, HS 2012 revision, release V202601 (2012–2024; 2024 is provisional).
See docs/research/baci-hs12-feasibility.md and ADR 0003.

`prepare()` is the single entry point (also `uv run python -m exportspredictor.baci`).
It is idempotent: each step checks for its output and skips work already done, and
every file is written as `<name>.part` and renamed when complete.

Layout under the data root (`EXPORTSPREDICTOR_DATA_DIR`):

    raw/baci/BACI_HS12_V202601.zip                   the release (1.27 GB)
    raw/baci/country_codes_V202601.csv               extracted from the zip
    raw/baci/product_codes_HS12_V202601.csv          extracted from the zip
    raw/worldbank/countries.json                     World Bank regions and income levels
    derived/baci_hs12_V202601/flows_<year>.parquet   one file per year
    derived/lookups/products.parquet                 hs6, description, sector
    derived/lookups/markets.parquet                  iso3, name, region, income_level, baci_code

Flows Parquet columns: `year` Int16, `exporter` and `importer` Enum of ISO3 codes,
`hs6` Enum of zero-padded HS6 codes, `value_usd` Float64 (BACI's `v` is thousand USD;
we store USD). BACI's quantity `q` is unused (spec #10) and dropped. Only strictly
positive Flows exist in BACI: a missing row means zero or unreported.
"""

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from exportspredictor.downloads import download, write_parquet, writing
from exportspredictor.paths import data_dir
from exportspredictor.sectors import sector_of

BACI_VERSION = "V202601"
BACI_URL = f"https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_{BACI_VERSION}.zip"
COUNTRIES_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
UK = "GBR"
PROVISIONAL_YEAR = 2024  # BACI's latest year is revised in the next release

_ZIP_NAME = f"BACI_HS12_{BACI_VERSION}.zip"
_COUNTRY_CODES = f"country_codes_{BACI_VERSION}.csv"
_PRODUCT_CODES = f"product_codes_HS12_{BACI_VERSION}.csv"
_YEAR_FILE = re.compile(rf"BACI_HS12_Y(\d{{4}})_{BACI_VERSION}\.csv")
_HISTORICAL = r"\(\.\.\.\d{4}\)"  # e.g. "Fed. Rep. of Germany (...1990)"
_UNCLASSIFIED = "Unclassified"
# BACI codes the World Bank does not list. "Other Asia, nes" is BACI's proxy for Taiwan.
_COUNTRY_OVERRIDES = {
    "S19": ("East Asia & Pacific", "High income"),
    "R20": ("Europe & Central Asia", "High income"),
}


@dataclass
class PrepareReport:
    """What one `prepare()` run actually did."""

    downloaded: list[str] = field(default_factory=list)
    extracted: list[str] = field(default_factory=list)
    converted_years: list[int] = field(default_factory=list)
    built_lookups: bool = False


def raw_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "raw" / "baci"


def flows_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"baci_hs12_{BACI_VERSION}"


def lookups_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / "lookups"


def prepare(
    root: Path | None = None,
    *,
    baci_url: str = BACI_URL,
    countries_url: str = COUNTRIES_URL,
) -> PrepareReport:
    """Download BACI and the World Bank country list, convert Flows to Parquet, build lookups.

    Safe to re-run: finished outputs are skipped. Peak memory stays around 1–2 GB, because
    each year is extracted to disk and streamed CSV -> Parquet.
    """
    root = Path(root or data_dir())
    report = PrepareReport()
    raw = raw_dir(root)
    archive_path = raw / _ZIP_NAME
    countries_path = root / "raw" / "worldbank" / "countries.json"

    for url, target in ((baci_url, archive_path), (countries_url, countries_path)):
        if download(url, target):
            report.downloaded.append(target.name)

    with zipfile.ZipFile(archive_path) as archive:
        for name in (_COUNTRY_CODES, _PRODUCT_CODES):
            if _extract(archive, name, raw / name):
                report.extracted.append(name)
        country_codes = _read_country_codes(raw / _COUNTRY_CODES)
        product_codes = _read_product_codes(raw / _PRODUCT_CODES)
        years = sorted(
            (int(match[1]), name) for name in archive.namelist() if (match := _YEAR_FILE.fullmatch(name))
        )
        out = flows_dir(root)
        for year, name in years:
            target = out / f"flows_{year}.parquet"
            if not target.exists():
                _convert_year(archive, name, target, country_codes, product_codes)
                report.converted_years.append(year)

    lookups = lookups_dir(root)
    if not ((lookups / "products.parquet").exists() and (lookups / "markets.parquet").exists()):
        lookups.mkdir(parents=True, exist_ok=True)
        write_parquet(_products_lookup(product_codes), lookups / "products.parquet")
        write_parquet(_markets_lookup(country_codes, countries_path), lookups / "markets.parquet")
        report.built_lookups = True
    return report


def scan_flows(root: Path | None = None) -> pl.LazyFrame:
    """All converted Flows as one lazy frame (filter years and columns before collecting)."""
    return pl.scan_parquet(flows_dir(root) / "flows_*.parquet")


def products(root: Path | None = None) -> pl.DataFrame:
    """Product lookup: hs6, description (HS 2012 edition), sector (HS2 chapter)."""
    return pl.read_parquet(lookups_dir(root) / "products.parquet")


def markets(root: Path | None = None) -> pl.DataFrame:
    """Market lookup: iso3, name, region, income_level (World Bank), baci_code."""
    return pl.read_parquet(lookups_dir(root) / "markets.parquet")


def summarise_flows(flows: pl.LazyFrame | pl.DataFrame, uk: str = UK) -> pl.DataFrame:
    """Per year: non-zero Flow count, Products, Exporters, Importers, world and UK value, UK share.

    Works on any Flows frame with year, exporter, importer, hs6 and value_usd, so the
    fixture world and real BACI are described the same way.
    """
    return (
        flows.lazy()
        .filter(pl.col("value_usd") > 0)
        .group_by("year")
        .agg(
            flows=pl.len(),
            products=pl.col("hs6").n_unique(),
            exporters=pl.col("exporter").n_unique(),
            importers=pl.col("importer").n_unique(),
            world_value_usd=pl.col("value_usd").sum(),
            uk_value_usd=pl.col("value_usd").filter(pl.col("exporter").cast(pl.String) == uk).sum(),
        )
        .with_columns(uk_share=pl.col("uk_value_usd") / pl.col("world_value_usd"))
        .sort("year")
        .collect()
    )


def _extract(archive: zipfile.ZipFile, name: str, target: Path) -> bool:
    if target.exists():
        return False
    with writing(target) as part, archive.open(name) as source, open(part, "wb") as out:
        shutil.copyfileobj(source, out, length=1 << 20)
    return True


def _read_country_codes(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema=False).with_columns(
        pl.col("country_code").cast(pl.UInt16)
    )


def _read_product_codes(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema=False).with_columns(pl.col("code").str.zfill(6))


def _convert_year(
    archive: zipfile.ZipFile, name: str, target: Path, country_codes: pl.DataFrame, product_codes: pl.DataFrame
) -> None:
    """Stream one year's CSV to Parquet via a temporary extracted copy."""
    countries = pl.Enum(sorted(country_codes["country_iso3"].unique()))
    hs6 = pl.Enum(product_codes["code"].sort())
    tmp = target.parent / ".tmp"
    csv_path = tmp / name
    _extract(archive, name, csv_path)
    to_iso3 = {"old": country_codes["country_code"], "new": country_codes["country_iso3"]}
    with writing(target) as part:
        (
            pl.scan_csv(
                csv_path,
                schema={"t": pl.Int16, "i": pl.UInt16, "j": pl.UInt16, "k": pl.String, "v": pl.Float64, "q": pl.Float64},
            )
            .select(
                year="t",
                exporter=pl.col("i").replace_strict(**to_iso3, return_dtype=pl.String).cast(countries),
                importer=pl.col("j").replace_strict(**to_iso3, return_dtype=pl.String).cast(countries),
                hs6=pl.col("k").str.zfill(6).cast(hs6),
                value_usd=pl.col("v") * 1000.0,
            )
            .sink_parquet(part, compression="zstd")
        )
    csv_path.unlink()
    shutil.rmtree(tmp, ignore_errors=True)


def _products_lookup(product_codes: pl.DataFrame) -> pl.DataFrame:
    return product_codes.select(
        hs6="code", description="description", sector=sector_of("code")
    ).sort("hs6")


def _markets_lookup(country_codes: pl.DataFrame, countries_json: Path) -> pl.DataFrame:
    _, entries = json.loads(countries_json.read_text(encoding="utf-8"))
    world_bank = pl.DataFrame(
        {
            "iso3": [entry["id"] for entry in entries],
            "region": [entry["region"]["value"].strip() for entry in entries],
            "income_level": [entry["incomeLevel"]["value"].strip() for entry in entries],
        }
    ).filter(pl.col("region") != "Aggregates")
    overrides = pl.DataFrame(
        {
            "iso3": list(_COUNTRY_OVERRIDES),
            "region": [region for region, _ in _COUNTRY_OVERRIDES.values()],
            "income_level": [income for _, income in _COUNTRY_OVERRIDES.values()],
        }
    )
    return (
        country_codes.filter(~pl.col("country_name").str.contains(_HISTORICAL))
        .select(
            iso3="country_iso3",
            name=pl.col("country_name").map_elements(_repair_double_encoding, return_dtype=pl.String),
            baci_code="country_code",
        )
        .join(pl.concat([world_bank, overrides]), on="iso3", how="left")
        .with_columns(pl.col("region", "income_level").fill_null(_UNCLASSIFIED))
        .select("iso3", "name", "region", "income_level", "baci_code")
        .sort("iso3")
    )


def _repair_double_encoding(text: str) -> str:
    """BACI's country file stores accented names UTF-8-encoded twice ("CÃ´te" for "Côte")."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


if __name__ == "__main__":
    print(prepare())
