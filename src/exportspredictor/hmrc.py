"""HMRC Overseas Trade Statistics (OTS): a 2023 cross-check of BACI's UK export totals.

BACI's UK Flows are not HMRC's figures. BACI blends the UK's export report with each
Importer's import report, recasts every year to HS 2012 and works in USD. HMRC's OTS is
what the UK itself declared, in £, in that year's Combined Nomenclature (CN8; 2023 uses
HS 2022 at six digits). This module puts OTS 2023 on BACI's footing and measures the
difference (docs/research/baci-hs12-feasibility.md, section 4):

1. `read_exports` parses HMRC's bulk export files (fixed-width, one record per CN8 ×
   country × port × month);
2. `allocate` sums them over ports and months to HS6 × Market, converts HS 2022 codes to
   HS 2012 with the UN Statistics Division's conversion table, and maps HMRC's two-letter
   country codes to ISO3. Records HMRC does not attribute to a Product (chapter 99:
   low-value consignments and HMRC's suppressed "commodity codes", and the chapter-only
   estimates) or to a Market (country codes such as "ZY", "QV" and "YY") keep a null
   `hs6` or `iso3`, so totals still add up;
3. `crosscheck` compares the result with BACI's UK Flows of the same year, in USD at the
   Bank of England's annual average rate, and breaks the difference down.

OTS is used only for this check, never as a model input. Its download lands in
`<data>/raw/hmrc/` and `python -m exportspredictor.hmrc` (a step in scripts/rebuild.py)
writes `derived/hmrc_ots_2023.parquet`.
"""

import re
import urllib.parse
import zipfile
from pathlib import Path

import polars as pl

from exportspredictor import currency
from exportspredictor.downloads import download as download_file
from exportspredictor.downloads import fetch, write_parquet
from exportspredictor.paths import data_dir

YEAR = 2023
ARCHIVE_PAGE = "https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/bulk-data-sets-archive/"
ARCHIVE_NAME = f"bdsexp_{YEAR % 100:02d}archive.zip"
CONCORDANCE_URL = (
    "https://unstats.un.org/unsd/classifications/Econ/tables/HS2022toHS2012ConversionAndCorrelationTables.xlsx"
)
CONCORDANCE_NAME = "HS2022toHS2012ConversionAndCorrelationTables.xlsx"
DERIVED_NAME = f"hmrc_ots_{YEAR}.parquet"

# Positions (1-based, inclusive) in the 85-character BDSexpYYMM record, from HMRC's bulk
# data set technical specification.
_FIELDS = {
    "month": (1, 6),  # PERREF: the month the trade belongs to
    "line_type": (7, 7),  # 1 declared, 2 below-threshold allocation, 3/4 non-response estimates
    "comcode": (14, 21),  # CN8, or a chapter / blank for estimates, or 990.. for suppressed codes
    "country": (30, 31),  # COD-ALPHA, HMRC's "country of ultimate destination": the Market
    "value_gbp": (45, 56),  # STAT-VALUE, whole £, FOB
}


def raw_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "raw" / "hmrc"


def read_exports(archive: Path) -> pl.DataFrame:
    """Every export record in an HMRC bulk archive: month, line_type, comcode, country, value_gbp."""
    frames = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            lines = z.read(name).decode("latin-1").splitlines()
            raw = pl.DataFrame({"line": [line for line in lines if line.strip()]})
            frames.append(
                raw.select(
                    **{
                        field: pl.col("line").str.slice(start - 1, end - start + 1)
                        for field, (start, end) in _FIELDS.items()
                    }
                )
            )
    return pl.concat(frames).with_columns(
        pl.col("month").cast(pl.Int32),
        pl.col("line_type").cast(pl.Int8),
        pl.col("value_gbp").cast(pl.Int64),
    )


# HMRC country codes that the World Bank list lacks or spells differently (Eurostat
# geonomenclature). Codes such as "ZY" (non-EU, not specified: mostly low-value
# consignments), "QV" (EU, estimated), "YY" (country suppressed), "QS" (ships' stores) and
# "QR" name no Market and stay unmapped.
COUNTRY_OVERRIDES = {"TW": "TWN", "XS": "SRB", "XK": "XKX", "NA": "NAM", "XC": "ESP", "XL": "ESP"}
CLOSE_SHARE = 0.25  # a pair "agrees" when OTS is within 25% of BACI


def market_codes(countries_json: Path) -> pl.DataFrame:
    """HMRC two-letter country code -> ISO3: the World Bank country list plus `COUNTRY_OVERRIDES`."""
    import json

    entries = json.loads(Path(countries_json).read_text(encoding="utf-8"))[1]
    codes = {
        entry["iso2Code"]: entry["id"]
        for entry in entries
        if entry.get("region", {}).get("value", "").strip() != "Aggregates"
    }
    codes.update(COUNTRY_OVERRIDES)
    return pl.DataFrame({"country": list(codes), "iso3": list(codes.values())})


def read_concordance(path: Path) -> pl.DataFrame:
    """The UN Statistics Division's HS 2022 -> HS 2012 conversion table: hs22, hs12 (one target each)."""
    table = pl.read_excel(path, sheet_name="HS2022-HS2012 Conversions", has_header=True)
    first, second = table.columns[:2]
    return table.select(hs22=pl.col(first).cast(pl.String).str.zfill(6), hs12=pl.col(second).cast(pl.String).str.zfill(6))


def allocate(records: pl.DataFrame, concordance: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
    """OTS records summed to HS 2012 Product × Market: hs6, iso3, value_gbp.

    `hs6` is null for value HMRC does not attribute to a declared CN8 code (HMRC's "commodity code")
    (estimates, chapter 99), and `iso3` is null for value it does not attribute to a
    Market. Nothing is dropped, so the column sums to HMRC's total.
    """
    declared = pl.col("comcode").str.contains(r"^\d{8}$") & ~pl.col("comcode").str.starts_with("99")
    return (
        records.with_columns(hs22=pl.when(declared).then(pl.col("comcode").str.slice(0, 6)))
        .join(concordance, on="hs22", how="left")
        .join(markets, on="country", how="left")
        .group_by(hs6=pl.col("hs12"), iso3=pl.col("iso3"))
        .agg(value_gbp=pl.col("value_gbp").sum())
        .sort("hs6", "iso3", nulls_last=True)
    )


def baci_uk_exports(flows: pl.LazyFrame, country_map: pl.DataFrame, year: int = YEAR, uk: str = "GBR") -> pl.DataFrame:
    """BACI's UK Flows of one year keyed by standard ISO3 (BACI's `S19` -> `TWN`): hs6, iso3, value_usd."""
    return (
        flows.filter((pl.col("year") == year) & (pl.col("exporter").cast(pl.String) == uk))
        .select(hs6=pl.col("hs6").cast(pl.String), baci_iso3=pl.col("importer").cast(pl.String), value_usd="value_usd")
        .collect()
        .join(country_map.select("baci_iso3", "iso3").unique("baci_iso3"), on="baci_iso3", how="left")
        .select("hs6", pl.col("iso3").fill_null(pl.col("baci_iso3")), "value_usd")
    )


def _compare(ots: pl.DataFrame, baci_uk: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    return (
        ots.group_by(key)
        .agg(ots_usd=pl.col("ots_usd").sum())
        .join(baci_uk.group_by(key).agg(baci_usd=pl.col("value_usd").sum()), on=key, how="full", coalesce=True)
        .with_columns(pl.col("ots_usd", "baci_usd").fill_null(0.0))
        .with_columns(difference_usd=pl.col("ots_usd") - pl.col("baci_usd"))
    )


def _largest_first(table: pl.DataFrame, top: int) -> list[dict]:
    return table.sort(pl.col("difference_usd").abs(), descending=True).head(top).to_dicts()


def crosscheck(
    ots: pl.DataFrame, baci_uk: pl.DataFrame, *, usd_per_gbp: float | None = None, year: int = YEAR, top: int = 10
) -> dict:
    """Compare HMRC OTS (from `allocate`) with BACI's UK Flows (hs6, iso3, value_usd) of the same year.

    Positive differences mean HMRC records more than BACI. Returns totals, the value HMRC
    leaves without a Product or a Market, the largest differences by Sector (HS2) and by
    Market, the agreement of the pairs both sources record, and a plain-English summary.
    HMRC's £ are converted at `year`'s own Bank of England annual average unless
    `usd_per_gbp` is given.
    """
    if usd_per_gbp is None:
        usd_per_gbp = currency.USD_PER_GBP[year]
    ots = ots.with_columns(ots_usd=pl.col("value_gbp") * usd_per_gbp)
    ots_usd = float(ots["ots_usd"].sum())
    baci_usd = float(baci_uk["value_usd"].sum())
    has_product = pl.col("hs6").is_not_null()
    has_market = pl.col("iso3").is_not_null()
    no_product = float(ots.filter(~has_product)["ots_usd"].sum())
    no_market = float(ots.filter(has_product & ~has_market)["ots_usd"].sum())

    sector = pl.col("hs6").str.slice(0, 2).alias("sector")
    by_sector = _compare(
        ots.filter(has_product).with_columns(sector), baci_uk.with_columns(sector), ["sector"]
    )
    by_market = _compare(ots.filter(has_market), baci_uk, ["iso3"])
    pairs = _compare(ots.filter(has_product & has_market), baci_uk, ["hs6", "iso3"])
    both = pairs.filter((pl.col("ots_usd") > 0) & (pl.col("baci_usd") > 0))
    close = both.filter(pl.col("difference_usd").abs() <= CLOSE_SHARE * pl.col("baci_usd"))
    spearman = (
        float(both.select(pl.corr(pl.col("ots_usd"), pl.col("baci_usd"), method="spearman")).item())
        if both.height >= 3
        else None
    )
    report = {
        "year": year,
        "usd_per_gbp": usd_per_gbp,
        "ots_usd": ots_usd,
        "baci_usd": baci_usd,
        "difference_usd": ots_usd - baci_usd,
        "difference_share": (ots_usd - baci_usd) / baci_usd if baci_usd else None,
        "no_product_usd": no_product,
        "no_market_usd": no_market,
        "allocated_ots_usd": ots_usd - no_product - no_market,
        "by_sector": _largest_first(by_sector, top),
        "by_market": _largest_first(by_market, top),
        "pairs": {
            "in_both": both.height,
            "ots_only": pairs.filter(pl.col("baci_usd") == 0).height,
            "baci_only": pairs.filter(pl.col("ots_usd") == 0).height,
            "spearman": spearman,
            "close_share": CLOSE_SHARE,
            "close_value_share": float(close["baci_usd"].sum()) / float(both["baci_usd"].sum()) if both.height else None,
        },
    }
    report["text"] = _text(report)
    return report


def _text(report: dict) -> str:
    share = report["difference_share"] or 0.0
    direction = "more" if share >= 0 else "less"
    unallocated = report["no_product_usd"] + report["no_market_usd"]
    return (
        f"HMRC's Overseas Trade Statistics record ${report['ots_usd'] / 1e9:,.1f}bn of UK goods exports in "
        f"{report['year']} (at ${report['usd_per_gbp']:.4f} per £), {abs(share):.0%} {direction} than BACI's "
        f"${report['baci_usd'] / 1e9:,.1f}bn. ${unallocated / 1e9:,.1f}bn of HMRC's total is not attributed to a "
        f"Product or a Market, so BACI cannot hold it at HS6 level."
    )


def download(root: Path | None = None) -> dict[str, Path]:
    """Download the OTS export archive for `YEAR` and the HS 2022 -> 2012 table, if missing."""
    folder = raw_dir(root)
    archive, concordance = folder / ARCHIVE_NAME, folder / CONCORDANCE_NAME
    if not archive.exists():
        page = fetch(ARCHIVE_PAGE).decode("utf-8", errors="replace")
        # The archive page links each year's zip under a /media/<id>/ path that changes with releases.
        match = re.search(rf'href="([^"]*/{re.escape(ARCHIVE_NAME)})"', page)
        if match is None:
            raise RuntimeError(f"No link to {ARCHIVE_NAME} on {ARCHIVE_PAGE}; download it into {folder} by hand")
        download_file(urllib.parse.urljoin(ARCHIVE_PAGE, match.group(1)), archive)
    download_file(CONCORDANCE_URL, concordance)
    return {"archive": archive, "concordance": concordance}


def build(root: Path | None = None) -> Path:
    """Download if needed, then write `derived/hmrc_ots_<YEAR>.parquet` (hs6, iso3, value_gbp)."""
    root = root or data_dir()
    files = download(root)
    ots = allocate(
        read_exports(files["archive"]),
        read_concordance(files["concordance"]),
        market_codes(root / "raw" / "worldbank" / "countries.json"),
    )
    return write_parquet(ots, root / "derived" / DERIVED_NAME)


def load(root: Path | None = None) -> pl.DataFrame:
    """The allocated OTS table written by `build`."""
    return pl.read_parquet((root or data_dir()) / "derived" / DERIVED_NAME)


if __name__ == "__main__":
    print(build())
